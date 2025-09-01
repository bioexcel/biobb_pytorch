"""
SPIB: A deep learning-based framework to learn RCs 
from MD trajectories. Code maintained by Dedi.

Read and cite the following when using this method:
https://aip.scitation.org/doi/abs/10.1063/5.0038198
"""

# --------------------
# Model
# --------------------   

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from mlcolvar.cvs import BaseCV
from biobb_pytorch.mdae.models.nn.feedforward import FeedForward
from biobb_pytorch.mdae.featurization.normalization import Normalization
from mlcolvar.core.transform.utils import Inverse
from biobb_pytorch.mdae.loss import InformationBottleneckLoss
from typing import Optional, List

__all__ = ["SPIB"]

class SPIB(BaseCV, pl.LightningModule):
    BLOCKS = ["norm_in", "encoder", "decoder", "k", 
              "UpdateLabel", "beta", "threshold", "patience", "refinements", 
              "learning_rate", "lr_step_size", "lr_gamma"]

    def __init__(self, n_features, n_cvs, encoder_layers, decoder_layers, options=None, **kwargs):
        super().__init__(in_features=n_features, out_features=n_cvs, **kwargs)
        
        options = self.parse_options(options)

        self._n_cvs = n_cvs
        self.k = options.get("k", 2)
        self.output_dim = n_features

        self.learning_rate = options.get("optimizer", {}).get("lr", 0.001)
        self.lr_step_size = options.get("optimizer", {}).get("step_size", 10)
        self.lr_gamma = options.get("optimizer", {}).get("gamma", 0.1)

        self.beta = 0.01
        self.threshold = options.get("threshold", 0.01)
        self.patience = options.get("patience", 10)
        self.refinements = options.get("refinements", 5)
        
        self.update_times = 0
        self.unchanged_epochs = 0
        self.state_population0 = None
        self.eps = 1e-10

        # Representative inputs
        self.representative_inputs = torch.eye(
            self.k, self.output_dim, device=self.device, requires_grad=False
        )
        self.idle_input = torch.eye(
            self.k, self.k, device=self.device, requires_grad=False
        )
        self.representative_weights = nn.Sequential(
            nn.Linear(self.k, 1, bias=False),
            nn.Softmax(dim=0)
        )
        nn.init.ones_(self.representative_weights[0].weight)

        # Encoder / Decoder
        o = "encoder"
        self.encoder = FeedForward([n_features] + encoder_layers, **options[o])
        self.encoder_mean = torch.nn.Linear(
            in_features=encoder_layers[-1], out_features=n_cvs
        )
        self.encoder_logvar = torch.nn.Linear(
            in_features=encoder_layers[-1], out_features=n_cvs
        )

        o = "decoder"
        self.decoder = FeedForward([n_cvs] + decoder_layers + [n_features], **options[o])

        # IB loss
        self.loss_fn = InformationBottleneckLoss(beta=self.beta, eps=self.eps)

        self.eval_variables = ["xhat", "z", "mu", "logvar", "labels"]

    def encode(self, inputs: torch.Tensor):
        h = self.encoder(inputs)
        mu = self.encoder_mean(h)
        logvar = -10 * F.sigmoid(self.encoder_logvar(h))
        return mu, logvar
    
    def decode(self, z: torch.Tensor):
        return F.log_softmax(self.decoder(z), dim=1)
    
    def forward_cv(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_in is not None:
            x = self.norm_in(x)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def encode_decode(self, x: torch.Tensor):
        flat = x.view(x.size(0), -1)
        if self.norm_in is not None:
            flat = self.norm_in(flat)
        mu, logvar = self.encode(flat)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        if self.norm_in is not None:
            x_hat = self.norm_in.inverse(x_hat)
        return x_hat, z, mu, logvar
    
    def evaluate_model(self, batch, batch_idx):

        xhat, z, mu, logvar = self.encode_decode(batch['data'])

        pred = xhat.exp()
        labels = pred.argmax(1)

        return xhat, z, mu, logvar, labels

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=self.lr_step_size, gamma=self.lr_gamma)
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': sch, 'interval': 'epoch'}}

    @torch.no_grad()
    def update_labels(self, inputs: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.UpdateLabel:
            return None
        loader = self.trainer.datamodule.train_dataloader()
        bs = loader.batch_size
        labels = []
        for i in range(0, len(inputs), bs):
            batch = inputs[i:i+bs].to(self.device)
            mu, _ = self.encode(batch)
            logp = self.decode(mu)
            labels.append(logp.exp())
        preds = torch.cat(labels, dim=0)
        idx = preds.argmax(dim=1)
        return F.one_hot(idx, num_classes=self.k)

    @torch.no_grad()
    def get_representative_z(self):
        return self.encode(self.representative_inputs)

    def reset_representative(self, rep_inputs: torch.Tensor):
        self.representative_inputs = rep_inputs.detach().clone()
        dim = rep_inputs.size(0)
        self.idle_input = torch.eye(dim, dim, device=self.device, requires_grad=False)
        self.representative_weights = nn.Sequential(
            nn.Linear(dim, 1, bias=False), nn.Softmax(dim=0)
        )
        nn.init.ones_(self.representative_weights[0].weight)

    def training_step(self, batch, batch_idx):
        x,y = batch['data'],batch['labels']
        w_batch = batch.get('weights', None)
        preds, z, mu, logvar = self.encode_decode(x)
        rep_mu, rep_logvar = self.get_representative_z()
        w = self.representative_weights(self.idle_input)
        loss, recon_err, kl = self.loss_fn(
            y.to(self.device), preds, z, mu, logvar, rep_mu, rep_logvar, w, w_batch
        )
        name = "train" if self.training else "valid"
        self.log(f'{name}_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{name}_recon', recon_err, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{name}_kl', kl, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    @torch.no_grad()
    def on_train_epoch_start(self):
        if self.trainer.current_epoch == 0:
            ds = self.trainer.datamodule.train_dataloader().dataset
            self.state_population0 = ds['labels'].float().mean(dim=0)
            self.representative_inputs = torch.eye(
                self.k, self.output_dim, device=self.device, requires_grad=False
            )

    @torch.no_grad()
    def on_train_epoch_end(self):
        ds = self.trainer.datamodule.train_dataloader().dataset
        new_labels = self.update_labels(ds['target'])
        if new_labels is None:
            return
        state_pop = new_labels.float().mean(dim=0)
        delta = torch.norm(state_pop - self.state_population0)
        self.log('state_population_change', delta)
        self.state_population0 = state_pop
        if delta < self.threshold:
            self.unchanged_epochs += 1
            if self.unchanged_epochs > self.patience:
                if torch.sum(state_pop > 0) < 2:
                    self.trainer.should_stop = True
                    return
                if self.UpdateLabel and self.update_times < self.refinements:
                    self.update_times += 1
                    self.unchanged_epochs = 0
                    ds['labels'] = new_labels
                    reps = self.estimate_representative_inputs(
                        ds['data'], getattr(ds, 'weights', None)
                    ).to(self.device)
                    self.reset_representative(reps)
                    self.log(f'refinement_{self.update_times}', 1)
                    # Force reload of the DataLoader to reflect updated labels
                    loop = self.trainer.fit_loop
                    loop._combined_loader = None
                    loop.setup_data()
                else:
                  self.trainer.should_stop = True
        else:
            self.unchanged_epochs = 0

    @torch.no_grad()
    def estimate_representative_inputs(
        self, inputs: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        self.eval()
        N = len(inputs)
        bs = self.trainer.datamodule.train_dataloader().batch_size
        preds = []
        for i in range(0, N, bs):
            batch = inputs[i:i + bs].to(self.device)
            mu, _ = self.encode(batch)
            logp = self.decoder(mu)
            preds.append(logp.exp())
        preds = torch.cat(preds, dim=0)
        labels = F.one_hot(preds.argmax(dim=1), num_classes=self.k)

        if bias is None:
            bias = torch.ones(N, device=self.device)

        data_shape = inputs.shape[1:]
        state_sums = torch.zeros(self.k, *data_shape, device=self.device)
        state_counts = torch.zeros(self.k, device=self.device)

        for state in range(self.k):
            mask = labels[:, state] == 1
            if mask.any():
                weights_expanded = bias[mask].view(-1, *([1] * len(data_shape)))
                weighted_inputs = inputs[mask] * weights_expanded
                state_sums[state] += weighted_inputs.sum(dim=0)
                state_counts[state] += bias[mask].sum()

        reps = torch.zeros(self.k, *data_shape, device=self.device)
        for state in range(self.k):
            if state_counts[state] > 0:
                reps[state] = state_sums[state] / state_counts[state]

        return reps