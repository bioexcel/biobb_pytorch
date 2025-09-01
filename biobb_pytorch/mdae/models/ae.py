# --------------------------------------------------------------------------------------
# autoencoder.py 
# 
# from the mlcolvar repository
#     https://github.com/mlcolvar/mlcolvar
#     Copyright (c) 2023 Luigi Bonati, Enrico Trizio, Andrea Rizzi & Michele Parrinello
#     Licensed under the MIT License (see project LICENSE file for full text)
# --------------------------------------------------------------------------------------

import torch
import lightning
from mlcolvar.cvs import BaseCV
from .nn.feedforward import FeedForward
from biobb_pytorch.mdae.featurization.normalization import Normalization
from mlcolvar.core.transform.utils import Inverse
from biobb_pytorch.mdae.loss import MSELoss

__all__ = ["AutoEncoder"]


class AutoEncoder(BaseCV, lightning.LightningModule):
    """AutoEncoding Collective Variable.
    It is composed by a first neural network (encoder) which projects
    the input data into a latent space (the CVs). Then a second network (decoder) takes
    the CVs and tries to reconstruct the input data based on them. It is an unsupervised learning approach,
    typically used when no labels are available. This CV is inspired by [1]_.

    Furthermore, it can also be used lo learn a representation which can be used not to reconstruct the data but
    to predict, e.g. future configurations.

    **Data**: for training it requires a DictDataset with the key 'data' and optionally 'weights' to reweight the
    data as done in [2]_. If a 'target' key is present this will be used as reference for the output of the decoder,
    otherway this will be compared with the input 'data'. This feature can be used to train a time-lagged autoencoder [3]_
    where the task is not to reconstruct the input but the output at a later step.

    **Loss**: reconstruction loss (MSELoss)

    References
    ----------
    .. [1] W. Chen and A. L. Ferguson, “ Molecular enhanced sampling with autoencoders: On-the-fly collective
        variable discovery and accelerated free energy landscape exploration,” JCC 39, 2079–2102 (2018)
    .. [2] Z. Belkacemi, P. Gkeka, T. Lelièvre, and G. Stoltz, “ Chasing collective variables using autoencoders and biased
        trajectories,” JCTC 18, 59–78 (2022)
    .. [3] C. Wehmeyer and F. Noé, “Time-lagged autoencoders: Deep learning of slow collective variables for molecular
        kinetics,” JCP 148, 241703 (2018).

    See also
    --------
    mlcolvar.core.loss.MSELoss
        (weighted) Mean Squared Error (MSE) loss function.
    """

    BLOCKS = ["norm_in", "encoder", "decoder"]

    def __init__(
        self,
        n_features: int,
        n_cvs: int,
        encoder_layers: list,
        decoder_layers: list = None,
        options: dict = None,
        **kwargs,
    ):
        """
        Define a CV defined as the output layer of the encoder of an autoencoder model (latent space).
        The decoder part is used only during the training for the reconstruction loss.
        By default a module standardizing the inputs is also used.

        Parameters
        ----------
        encoder_layers : list
            Number of neurons per layer of the encoder
        decoder_layers : list, optional
            Number of neurons per layer of the decoder, by default None
            If not set it takes automaically the reversed architecture of the encoder
        options : dict[str,Any], optional
            Options for the building blocks of the model, by default None.
            Available blocks: ['norm_in', 'encoder','decoder'].
            Set 'block_name' = None or False to turn off that block
        """
        super().__init__(
            in_features=n_features, out_features=n_cvs, **kwargs
        )

        # =======   LOSS  =======
        # Reconstruction (MSE) loss
        self.loss_fn = MSELoss()

        # ======= OPTIONS =======
        # parse and sanitize
        options = self.parse_options(options)

        # if decoder is not given reverse the encoder
        if decoder_layers is None:
            decoder_layers = encoder_layers[::-1]

        # ======= BLOCKS =======

        # initialize norm_in
        o = "norm_in"
        if (options[o] is not False) and (options[o] is not None):
            self.norm_in = Normalization(self.in_features, **options[o])

        # initialize encoder
        o = "encoder"
        self.encoder = FeedForward([n_features] + encoder_layers + [n_cvs], **options[o])

        # initialize decoder
        o = "decoder"
        self.decoder = FeedForward([n_cvs] + decoder_layers + [n_features], **options[o])

        self.eval_variables = ["xhat", "z"]

    def forward_cv(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the CV without pre or post/processing modules."""
        if self.norm_in is not None:
            x = self.norm_in(x)
        x = self.encoder(x)
        return x
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode the latent space into the original input space."""
        x = self.decoder(z)
        if self.norm_in is not None:
            x = self.norm_in.inverse(x)
        return x

    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        """Pass the inputs through both the encoder and the decoder networks."""
        x = self.forward_cv(x)
        x = self.decoder(x)
        if self.norm_in is not None:
            x = self.norm_in.inverse(x)
        return x
    
    def evaluate_model(self, batch, batch_idx=None):
        """Evaluate the model on the data, computing the reconstruction loss."""

        x = batch['data']
        z = self.forward_cv(x)
        x_hat = self.decoder(z)

        if self.norm_in is not None:
            x_hat = self.norm_in.inverse(x_hat)
        
        return x_hat, z

    def training_step(self, train_batch, batch_idx):
        """Compute and return the training loss and record metrics."""
        # =================get data===================
        x = train_batch["data"]
        loss_kwargs = {}
        if "weights" in train_batch:
            loss_kwargs["weights"] = train_batch["weights"]
        # =================forward====================
        x_hat = self.encode_decode(x)
        # ===================loss=====================
        if "target" in train_batch:
            x_ref = train_batch["target"]
        else:
            x_ref = x

        # if self.norm_in is not None:
        #     x_ref = self.norm_in(x_ref)

        loss = self.loss_fn(x_hat, x_ref, **loss_kwargs)

        # ====================log=====================
        name = "train" if self.training else "valid"
        self.log(f"{name}_loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
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

