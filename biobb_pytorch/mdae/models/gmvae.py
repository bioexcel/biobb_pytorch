import torch
import torch.nn as nn
import lightning
from mlcolvar.cvs import BaseCV
from biobb_pytorch.mdae.featurization.normalization import Normalization
from mlcolvar.core.transform.utils import Inverse
from biobb_pytorch.mdae.models.nn.feedforward import FeedForward 
from biobb_pytorch.mdae.loss import ELBOGaussiansLoss, ELBOGaussianMixtureLoss


__all__ = ["GaussianMixtureVariationalAutoEncoder"]


class GaussianMixtureVariationalAutoEncoder(BaseCV, lightning.LightningModule):
    """Gaussian Mixture Variational AutoEncoder Collective Variable.
    This class implements a Gaussian Mixture Variational AutoEncoder (GMVAE) for
    collective variable (CV) learning. The GMVAE is a generative model that combines
    the principles of Gaussian Mixture Models (GMM) and Variational Autoencoders (VAE).
    It learns a latent representation of the input data by modeling it as a mixture
    of Gaussians, where each Gaussian corresponds to a different cluster in the data.
    The model consists of an encoder that maps the input data to a latent space,
    and a decoder that reconstructs the input data from the latent representation.
    The GMVAE is trained using a variational inference approach, where the model
    learns to maximize the evidence lower bound (ELBO) on the data likelihood.
    The ELBO consists of two terms: the reconstruction loss and the KL divergence
    between the learned latent distribution and a prior distribution (usually a
    standard normal distribution). The GMVAE can be used for various tasks such as
    clustering, dimensionality reduction, and generative modeling.
    The model is designed to work with PyTorch and PyTorch Lightning, making it easy
    to integrate into existing workflows and leverage GPU acceleration.
    Parameters
    ----------
    k : int
        The number of clusters in the Gaussian Mixture Model.
    n_cvs : int
        The dimension of the CV or, equivalently, the dimension of the latent
        space of the autoencoder.
    n_features : int
        The dimension of the input data.
    r_nent : float
        The weight for the entropy regularization term.
    qy_dims : list
        The dimensions of the layers in the encoder for the cluster assignment.
    qz_dims : list
        The dimensions of the layers in the encoder for the latent variable.
    pz_dims : list
        The dimensions of the layers in the decoder for the latent variable.
    px_dims : list
        The dimensions of the layers in the decoder for the reconstruction.
    options : dict, optional
        Additional options for the model, such as normalization and dropout rates.
    """

    BLOCKS = ["norm_in", "encoder", "decoder", "k"]

    def __init__(self, n_features, n_cvs, encoder_layers, decoder_layers, options=None, **kwargs):
        super().__init__(in_features=n_features, out_features=n_cvs, **kwargs)

        options = self.parse_options(options)

        if "norm_in" in options and options["norm_in"] is not None:
            self.norm_in = Normalization(self.in_features, **options["norm_in"])

        self.k = options["k"]
        self.r_nent = options.get('loss_function', {}).get("r_nent", 0.5)
        n_features = encoder_layers["qy_dims"][0]

        qy_dims = encoder_layers["qy_dims"]
        qz_dims = encoder_layers["qz_dims"]
        pz_dims = decoder_layers["pz_dims"]
        px_dims = decoder_layers["px_dims"]

        self.loss_fn = ELBOGaussianMixtureLoss(r_nent=self.r_nent, k=self.k)

        self.encoder = nn.ModuleDict()
        self.decoder = nn.ModuleDict()

        self.encoder['y_transform'] = nn.Linear(self.k, self.k)

        self.encoder['qy_nn'] = FeedForward(qy_dims + [self.k], **options["encoder"]['qy_nn'])

        self.encoder['qz_nn'] = FeedForward([n_features + self.k] + qz_dims, **options["encoder"]['qz_nn'])
        self.encoder['zm_layer'] = nn.Linear(qz_dims[-1], n_cvs)
        self.encoder['zv_layer'] = nn.Linear(qz_dims[-1], n_cvs)

        self.decoder['pz_nn'] = FeedForward([self.k] + pz_dims, **options["decoder"]['pz_nn'])
        self.decoder['zm_prior_layer'] = nn.Linear(pz_dims[-1], n_cvs)
        self.decoder['zv_prior_layer'] = nn.Linear(pz_dims[-1], n_cvs)

        self.decoder['px_nn'] = FeedForward([n_cvs] + px_dims, **options["decoder"]['px_nn'])
        self.decoder['xm_layer'] = nn.Linear(px_dims[-1], n_features)
        self.decoder['xv_layer'] = nn.Linear(px_dims[-1], n_features)

        self.eval_variables = ["xhat", "z", "qy"]

    @staticmethod
    def log_normal(x, mu, var, eps=1e-10):
        return -0.5 * torch.sum(torch.log(torch.tensor(2.0) * torch.pi) + (x - mu).pow(2) / var + var.log(), dim=-1)  # log probability of a normal (Gaussian) distribution

    def loss_function(self, x, xm, xv, z, zm, zv, zm_prior, zv_prior):
        return (
            -self.log_normal(x, xm, xv)                                                             # Reconstruction Loss
            + self.log_normal(z, zm, zv) - self.log_normal(z, zm_prior, zv_prior)                   # Regularization Loss (KL Divergence)
            - torch.log(torch.tensor(1/self.k, device=x.device))                                    # Entropy Regularization
        )

    def encode_decode(self, x):

        if self.norm_in is not None:
            data = self.norm_in(x)

        qy_logit = self.encoder['qy_nn'](data)
        qy = torch.softmax(qy_logit, dim=1)

        y_ = torch.zeros([data.shape[0], self.k]).to(data.device)

        zm_list, zv_list, z_list = [], [], []
        xm_list, xv_list, x_list = [], [], []
        zm_prior_list, zv_prior_list = [], []

        for i in range(self.k):
            # One-hot y
            y = y_ + torch.eye(self.k).to(data.device)[i]

            # Qz
            h0 = self.encoder['y_transform'](y)
            xy = torch.cat([data, h0], dim=1)
            qz_logit = self.encoder['qz_nn'](xy)
            zm = self.encoder['zm_layer'](qz_logit)
            zv = torch.nn.functional.softplus(self.encoder['zv_layer'](qz_logit))
            noise = torch.randn_like(torch.sqrt(zv))
            z_sample = zm + noise * zv

            zm_list.append(zm)
            zv_list.append(zv)
            z_list.append(z_sample)

            # Pz (prior)
            pz_logit = self.decoder['pz_nn'](y)
            zm_prior = self.decoder['zm_prior_layer'](pz_logit)
            zv_prior = torch.nn.functional.softplus(self.decoder['zv_prior_layer'](pz_logit))
            noise = torch.randn_like(torch.sqrt(zv_prior))
            z_prior_sample = zm_prior + noise * zv_prior

            zm_prior_list.append(zm_prior)
            zv_prior_list.append(zv_prior)

            # Px
            px_logit = self.decoder['px_nn'](z_prior_sample)
            xm = self.decoder['xm_layer'](px_logit)
            xv = torch.nn.functional.softplus(self.decoder['xv_layer'](px_logit))
            noise = torch.randn_like(torch.sqrt(xv))
            x_sample = xm + noise * xv

            xm_list.append(xm)
            xv_list.append(xv)
            x_list.append(x_sample)

        return (
            data, qy_logit, xm_list, xv_list,
            z_list, zm_list, zv_list,
            zm_prior_list, zv_prior_list
        )

    def evaluate_model(self, batch, batch_idx):
        """Evaluate the model on the data, computing average loss."""

        x = batch['data']

        if self.norm_in is not None:
            data = self.norm_in(x)

        qy_logit = self.encoder['qy_nn'](data)
        qy = torch.softmax(qy_logit, dim=1)

        y_ = torch.zeros([data.shape[0], self.k]).to(data.device)

        zm_list, zv_list, z_list = [], [], []
        xm_list, xv_list, x_list = [], [], []
        zm_prior_list, zv_prior_list = [], []

        for i in range(self.k):
            # One-hot y
            y = y_ + torch.eye(self.k).to(data.device)[i]

            # Qz
            h0 = self.encoder['y_transform'](y)
            xy = torch.cat([data, h0], dim=1)
            qz_logit = self.encoder['qz_nn'](xy)
            zm = self.encoder['zm_layer'](qz_logit)
            zv = torch.nn.functional.softplus(self.encoder['zv_layer'](qz_logit))
            noise = torch.randn_like(torch.sqrt(zv))
            z_sample = zm + noise * zv

            zm_list.append(zm)
            zv_list.append(zv)
            z_list.append(z_sample)

            # Pz (prior)
            pz_logit = self.decoder['pz_nn'](y)
            zm_prior = self.decoder['zm_prior_layer'](pz_logit)
            zv_prior = torch.nn.functional.softplus(self.decoder['zv_prior_layer'](pz_logit))
            noise = torch.randn_like(torch.sqrt(zv_prior))
            z_prior_sample = zm_prior + noise * zv_prior

            zm_prior_list.append(zm_prior)
            zv_prior_list.append(zv_prior)

            # Px
            px_logit = self.decoder['px_nn'](z_prior_sample)
            xm = self.decoder['xm_layer'](px_logit)
            xv = torch.nn.functional.softplus(self.decoder['xv_layer'](px_logit))
            noise = torch.randn_like(torch.sqrt(xv))
            x_sample = xm + noise * xv

            xm_list.append(xm)
            xv_list.append(xv)
            x_list.append(x_sample)
        
        xhat = torch.sum(qy.unsqueeze(-1) * torch.stack(x_list, dim=1), dim=1)

        if self.norm_in is not None:
            xhat = self.norm_in.inverse(xhat)

        z = torch.sum(qy.unsqueeze(-1) * torch.stack(z_list, dim=1), dim=1)

        return xhat, z, qy

    def decode(self, z):
        """
        Reconstruct x' from aggregated z 
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)

        px_logit = self.decoder['px_nn'](z)
        xm = self.decoder['xm_layer'](px_logit)
        xv = torch.nn.functional.softplus(self.decoder['xv_layer'](px_logit))
        noise = torch.randn_like(torch.sqrt(xv))
        x = xm + noise * xv

        if self.norm_in is not None:
            x = self.norm_in.inverse(x)
        
        return x

    def forward_cv(self, x):

        if self.norm_in is not None:
            x = self.norm_in(x)
        
        qy_logit = self.encoder['qy_nn'](x)
        qy = torch.softmax(qy_logit, dim=1)

        y_ = torch.zeros([x.shape[0], self.k]).to(x.device)

        zm_list, zv_list, z_list = [], [], []

        for i in range(self.k):
            # One-hot y
            y = y_ + torch.eye(self.k).to(x.device)[i]

            # Qz
            h0 = self.encoder['y_transform'](y)
            xy = torch.cat([x, h0], dim=1)
            qz_logit = self.encoder['qz_nn'](xy)
            zm = self.encoder['zm_layer'](qz_logit)
            zv = torch.nn.functional.softplus(self.encoder['zv_layer'](qz_logit))
            noise = torch.randn_like(torch.sqrt(zv))
            z_sample = zm + noise * zv

            zm_list.append(zm)
            zv_list.append(zv)
            z_list.append(z_sample)

        Z = torch.stack(z_list, dim=1)
        a = torch.sum(qy.unsqueeze(-1) * Z, dim=1)      

        return a
    
    def training_step(self, train_batch, batch_idx):

        x = train_batch["data"]

        if "target" in train_batch:
            x_ref = train_batch["target"]
        else:
            x_ref = x

        data, qy_logit, xm_list, xv_list, z_list, zm_list, zv_list, zm_prior_list, zv_prior_list = self.encode_decode(x_ref)

        batch_loss, nent = self.loss_fn(data, 
                                        qy_logit, 
                                        xm_list, xv_list, 
                                        z_list, zm_list, zv_list, 
                                        zm_prior_list, zv_prior_list)

        loss = batch_loss.mean()
        ce_loss = nent.mean()

        name = "train" if self.training else "valid"
        self.log(f"{name}_loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        self.log(f"{name}_cross_entropy", ce_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def get_decoder(self, return_normalization=False):
        """Return a torch model with the decoder and optionally the normalization inverse"""
        if return_normalization:
            if self.norm_in is not None:
                inv_norm = Inverse(module=self.norm_in)
                decoder_model = torch.nn.Sequential(*[self.decoder, inv_norm])
            else:
                raise ValueError(
                    "return_normalization is set to True but self.norm_in is None"
                )
        else:
            decoder_model = self.decoder
            
        return decoder_model


# # Example of usage:

# # Define dimensions
# n_features = 1551   # Input dimension
# n_clusters = 5      # Output dimension for Qy
# n_cvs = 3           # Latent dimension (CVs)
# r_nent = 0.5        # Weight for the entropy regularization term.

# # Encoder sizes
# qy_dims = [32]   
# qz_dims = [16, 16]  

# # Decoder sizes
# pz_dims = [16, 16]  
# px_dims = [128]    

# options = {
#     "norm_in": {
#         "mode": "mean_std"
#     },
#     "optimizer": {
#         "lr": 1e-4
#     }
# }

# # Instantiate your GMVAECV
# model = GaussianMixtureVariationalAutoEncoder(k=n_clusters,
#                                                 n_cvs=n_cvs,
#                                                 n_features=n_features,
#                                                 r_nent=r_nent,
#                                                 qy_dims=qy_dims,
#                                                 qz_dims=qz_dims,
#                                                 pz_dims=pz_dims,
#                                                 px_dims=px_dims,
#                                                 options=options)


