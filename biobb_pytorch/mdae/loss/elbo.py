#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Evidence Lower BOund (ELBO) loss functions used to train variational Autoencoders.
"""

__all__ = ["ELBOGaussiansLoss", "elbo_gaussians_loss", "ELBOLoss", "ELBOGaussianMixtureLoss"]


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from typing import Optional
import torch
import math
from torch import nn
from torch.nn import functional as F
from mlcolvar.core.loss.mse import mse_loss


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================


class ELBOGaussiansLoss(torch.nn.Module):
    """ELBO loss function assuming the latent and reconstruction distributions are Gaussian.

    The ELBO uses the MSE as the reconstruction loss (i.e., assumes that the
    decoder outputs the mean of a Gaussian distribution with variance 1), and
    the KL divergence between two normal distributions ``N(mean, var)`` and
    ``N(0, 1)``, where ``mean`` and ``var`` are the output of the encoder.
    """

    def forward(
        self,
        target: torch.Tensor,
        output: torch.Tensor,
        mean: torch.Tensor,
        log_variance: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the value of the loss function.

        Parameters
        ----------
        target : torch.Tensor
            Shape ``(n_batches, in_features)``. Data points (e.g. input of encoder
            or time-lagged features).
        output : torch.Tensor
            Shape ``(n_batches, in_features)``. Output of the decoder.
        mean : torch.Tensor
            Shape ``(n_batches, latent_features)``. The means of the Gaussian
            distributions associated to the inputs.
        log_variance : torch.Tensor
            Shape ``(n_batches, latent_features)``. The logarithm of the variances
            of the Gaussian distributions associated to the inputs.
        weights : torch.Tensor, optional
            Shape ``(n_batches,)`` or ``(n_batches,1)``. If given, the average over
            batches is weighted. The default (``None``) is unweighted.

        Returns
        -------
        loss: torch.Tensor
            The value of the loss function.
        """
        return elbo_gaussians_loss(target, output, mean, log_variance, weights)


def elbo_gaussians_loss(
    target: torch.Tensor,
    output: torch.Tensor,
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """ELBO loss function assuming the latent and reconstruction distributions are Gaussian.

    The ELBO uses the MSE as the reconstruction loss (i.e., assumes that the
    decoder outputs the mean of a Gaussian distribution with variance 1), and
    the KL divergence between two normal distributions ``N(mean, var)`` and
    ``N(0, 1)``, where ``mean`` and ``var`` are the output of the encoder.

    Parameters
    ----------
    target : torch.Tensor
        Shape ``(n_batches, in_features)``. Data points (e.g. input of encoder
        or time-lagged features).
    output : torch.Tensor
        Shape ``(n_batches, in_features)``. Output of the decoder.
    mean : torch.Tensor
        Shape ``(n_batches, latent_features)``. The means of the Gaussian
        distributions associated to the inputs.
    log_variance : torch.Tensor
        Shape ``(n_batches, latent_features)``. The logarithm of the variances
        of the Gaussian distributions associated to the inputs.
    weights : torch.Tensor, optional
        Shape ``(n_batches,)`` or ``(n_batches,1)``. If given, the average over
        batches is weighted. The default (``None``) is unweighted.

    Returns
    -------
    loss: torch.Tensor
        The value of the loss function.
    """
    # KL divergence between N(mean, variance) and N(0, 1).
    # See https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    kl = -0.5 * (log_variance - log_variance.exp() - mean**2 + 1).sum(dim=1)

    # Weighted mean over batches.
    if weights is None:
        kl = kl.mean()
    else:
        weights = weights.squeeze()
        if weights.shape != kl.shape:
            raise ValueError(
                f"weights should be a tensor of shape (n_batches,) or (n_batches,1), not {weights.shape}."
            )
        kl = (kl * weights).sum()

    # Reconstruction loss.
    reconstruction = mse_loss(output, target, weights=weights)

    return reconstruction + kl


class ELBOLoss(nn.Module):
    """
    Variational Autoencoder ELBO loss function.

    Implements the evidence lower bound (ELBO) objective:
        L = reconstruction_loss + beta * KL_divergence

    Reconstruction loss options:
      - Mean-squared error (MSE) -> assumes Gaussian decoder with unit variance
      - Binary cross-entropy (BCE) -> assumes Bernoulli decoder

    KL divergence is computed analytically between the approximate posterior
    q(z|x) = N(mu, diag(var)) and the prior p(z) = N(0, I):
        KL(q||p) = -0.5 * sum(1 + log(var) - mu^2 - var)

    Parameters
    ----------
    beta : float, default=1.0
        Scaling factor for the KL divergence term (beta-VAE).
    loss_type : {'mse', 'bce'}, default='mse'
        Type of reconstruction loss:
        - 'mse': use mean squared error
        - 'bce': use binary cross-entropy
    reduction : {'sum', 'mean', 'none'}, default='sum'
        How to reduce the reconstruction loss over elements:
        - 'sum': sum over all elements
        - 'mean': average over all elements
        - 'none': no reduction (returns per-element loss)
    """

    def __init__(
        self,
        beta: float = 1.0,
        reconstruction: str = 'mse',
        reduction: str = 'sum'
    ):
        super().__init__()
        if reconstruction not in {'mse', 'bce'}:
            raise ValueError(f"Unsupported reconstruction '{reconstruction}', choose 'mse' or 'bce'.")
        if reduction not in {'sum', 'mean', 'none'}:
            raise ValueError(f"Unsupported reduction '{reduction}', choose 'sum', 'mean', or 'none'.")

        self.beta = beta
        self.reconstruction = reconstruction
        self.reduction = reduction

    def forward(
        self,
        x: torch.Tensor,
        recon_x: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the combined ELBO loss.

        Parameters
        ----------
        x : Tensor
            Original input tensor (shape: [batch_size, ...]).
        recon_x : Tensor
            Reconstructed output tensor (same shape as x).
        mu : Tensor
            Mean of the approximate posterior q(z|x) (shape: [batch_size, latent_dim]).
        log_var : Tensor
            Log-variance of q(z|x) (same shape as mu).

        Returns
        -------
        loss : Tensor
            Scalar loss (if reduction!='none') or tensor of per-element losses.
        """
        # Reconstruction loss
        if self.reconstruction == 'bce':
            # For binary data, use BCE
            recon_loss = F.binary_cross_entropy(
                recon_x, x, reduction=self.reduction
            )
        else:
            # For continuous data, use MSE
            recon_loss = F.mse_loss(
                recon_x, x, reduction=self.reduction
            )

        # Analytic KL divergence between N(mu, var) and N(0, I)
        # var = exp(log_var)
        var = torch.exp(log_var)
        kl_div = -0.5 * torch.sum(
            1 + log_var - mu.pow(2) - var,
            dim=1  # sum over latent dimension for each sample
        )

        # Combine terms: sum or mean over batch
        if self.reduction == 'mean':
            kl_div = kl_div.mean()
        elif self.reduction == 'sum':
            kl_div = kl_div.sum()
        # else 'none': keep per-sample KL vector

        # Scale KL and add reconstruction
        return recon_loss + self.beta * kl_div


class ELBOGaussianMixtureLoss(nn.Module):
    """
    Gaussian Mixture VAE loss.

    Combines:
      1) Entropy regularization:  -∑_i q(y=i|x) log q(y=i|x)
      2) Reconstruction + KL:
         - E_{q(y|x)} [ log p(x|z,y) ]
         + E_{q(y|x)} [ KL( q(z|x,y) ‖ p(z|y) ) ]
    """
    def __init__(self, k: int, r_nent: float = 1.0):
        """
        Args:
            k       Number of mixture components.
            r_nent  Weight on the entropy term.
        """
        super().__init__()
        self.k = k
        self.r_nent = r_nent

    @staticmethod
    def log_normal(x: torch.Tensor,
                   mu: torch.Tensor,
                   var: torch.Tensor,
                   eps: float = 1e-10) -> torch.Tensor:
        """
        Compute log N(x; mu, var) summed over the last dim:
          -½ ∑ [ log(2π) + (x−μ)^2 / var + log var ]
        """
        const = math.log(2 * math.pi)
        return -0.5 * torch.sum(
            const + (x - mu).pow(2) / (var + eps) + var.log(),
            dim=-1
        )

    def forward(self,
                x: torch.Tensor,
                qy_logit: torch.Tensor,
                xm_list: list[torch.Tensor],
                xv_list: list[torch.Tensor],
                z_list:  list[torch.Tensor],
                zm_list: list[torch.Tensor],
                zv_list: list[torch.Tensor],
                zm_prior_list: list[torch.Tensor],
                zv_prior_list: list[torch.Tensor]
               ) -> torch.Tensor:
        """
        Args:
            x                [batch, n_features]                Input data
            qy_logit         [batch, k]                          Cluster logits
            xm_list, xv_list length-k lists of [batch, n_features]
            z_list, zm_list, zv_list       length-k lists of [batch, n_cvs]
            zm_prior_list, zv_prior_list   length-k lists of [batch, n_cvs]
        Returns:
            scalar loss = mean_batch( r_nent*nent + ∑_i qy_i * [rec_i + KL_i] )
        """
        # 1) cluster posteriors
        qy = F.softmax(qy_logit, dim=1)    # [batch, k]

        # 2) entropy regularization (cross-entropy of qy wrt itself)
        #    nent = -E[ log q(y|x) ]
        nent = -torch.sum(qy * F.log_softmax(qy_logit, dim=1), dim=1).mean()

        # 3) per-component reconstruction + KL
        comp_losses = []
        for i in range(self.k):
            # reconstruction:  - log p(x | z_i)
            rec_i = -self.log_normal(x, xm_list[i], xv_list[i])
            # KL divergence:   KL( q(z|x,y=i) ‖ p(z|y=i) )
            kl_i = (
                self.log_normal(z_list[i], zm_list[i], zv_list[i])
                - self.log_normal(z_list[i], zm_prior_list[i], zv_prior_list[i])
            )
            comp_losses.append(rec_i + kl_i)  # shape [batch]

        # 4) weight each comp by qy[:,i] and sum
        weighted = [qy[:, i] * comp_losses[i] for i in range(self.k)]
        total = self.r_nent * nent + sum(weighted)  # shape [batch]

        return total, nent