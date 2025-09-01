#!/usr/bin/env python

import torch
import torch.nn as nn
import math

class InformationBottleneckLoss(nn.Module):
    """
    Loss = reconstruction_error + beta * KL[q(z|x) || p(z)]

    Where p(z) is modeled as a mixture over representative_z (means/logvars),
    weighted by representative_weights(idle_input).
    """
    def __init__(
        self,
        beta: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.beta   = beta
        self.eps    = eps
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def log_p(
        self,
        z: torch.Tensor,                   # [B] or [B,1]
        rep_mean: torch.Tensor,            # [R]
        rep_logvar: torch.Tensor,          # [R]
        w: torch.Tensor,                   # [R] or [R,1]
        sum_up: bool = True
    ) -> torch.Tensor:
        """
        Compute log p(z) under the mixture prior.

        Args:
          z         (Tensor[B] or [B,1]): latent samples
          rep_mean  (Tensor[R])           : mixture means
          rep_logvar(Tensor[R])           : mixture log‐vars
          w         (Tensor[R] or [R,1])  : mixture weights
          sum_up    (bool)                : if True, returns [B] log‐density;
                                           if False, returns [B,R] component‐wise log‐probs
        Returns:
          Tensor[B] if sum_up else Tensor[B,R]
        """

        z_expand = z.unsqueeze(1)               
        mu    = rep_mean.unsqueeze(0)        
        lv    = rep_logvar.unsqueeze(0)      

        representative_log_q = -0.5 * torch.sum(lv + torch.pow(z_expand-mu, 2)
                                        / torch.exp(lv), dim=2 )
        
        if sum_up:
            log_p = torch.sum(torch.log(torch.exp(representative_log_q)@w + self.eps), dim=1)
        else:
            log_p = torch.log(torch.exp(representative_log_q)*w.T + self.eps)  
            
        return log_p              

    def forward(
        self,
        data_targets: torch.Tensor, # [B, C_out]
        outputs: torch.Tensor,      # [B, C_out], log‐probs
        z_sample: torch.Tensor,     # [B]
        z_mean: torch.Tensor,       # [B]
        z_logvar: torch.Tensor,     # [B]
        rep_mean: torch.Tensor,     # [R]
        rep_logvar: torch.Tensor,   # [R]
        w: torch.Tensor,            # [R] or [R,1]
        data_weights: torch.Tensor = None,
        sum_up: bool = True,
    ):
        """
        Computes:
          rec_err = E_q[−log p(x|z)]
          kld     = E_q[ log q(z|x) − log p(z) ]
        Returns:
          loss, rec_err (scalar), kl_term (scalar)
        """

        # --- RECONSTRUCTION ---
        # cross‐entropy per sample: [B]
        ce = torch.sum(-data_targets * outputs, dim=1)
        rec_err = torch.mean(ce * data_weights) if data_weights is not None else ce.mean()

        # --- KL TERM ---
        # log q(z|x): -½ ∑[logvar + (z−mean)² / exp(logvar)]
        log_q = -0.5 * (z_logvar + (z_sample - z_mean).pow(2).div(z_logvar.exp()))
        # log p(z): mixture prior
        log_p = self.log_p(z_sample, rep_mean, rep_logvar, w, sum_up=sum_up)

        # per‐sample KL
        kld   = log_q - log_p
        kl_term = (kld * data_weights).mean() if data_weights is not None else kld.mean()

        loss = rec_err + self.beta * kl_term
        return loss, rec_err, kl_term
# class InformationBottleneckLoss(nn.Module):
#     """
#     Loss = reconstruction_error + beta * KL[q(z|x) || p(z)]
#     Where p(z) is a mixture over representative_z (means/logvars),
#     weighted by representative_weights(idle_input).
#     """
#     def __init__(self, beta: float = 1.0, eps: float = 1e-8):
#         super().__init__()
#         self.beta = beta
#         self.eps = eps
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"

#     def log_p(
#         self,
#         z: torch.Tensor,                   # [B, n_cvs]
#         rep_mean: torch.Tensor,            # [k, n_cvs]
#         rep_logvar: torch.Tensor,          # [k, n_cvs]
#         w: torch.Tensor,                   # [k, 1]
#     ) -> torch.Tensor:
#         """
#         Compute log p(z) under the mixture prior.

#         Args:
#             z: (Tensor[B, n_cvs]) latent samples
#             rep_mean: (Tensor[k, n_cvs]) mixture means
#             rep_logvar: (Tensor[k, n_cvs]) mixture log-variances
#             w: (Tensor[k, 1]) mixture weights

#         Returns:
#             Tensor[B]
#         """
#         batch_size, n_cvs = z.shape
#         k = rep_mean.shape[0]

#         # Expand dimensions for broadcasting
#         z_expand = z.unsqueeze(1)  # [B, 1, n_cvs]
#         mu = rep_mean.unsqueeze(0)  # [1, k, n_cvs]
#         lv = rep_logvar.unsqueeze(0)  # [1, k, n_cvs]

#         var = torch.exp(lv)

#         # Log-probability per dimension per component
#         log_prob_per_dim = -0.5 * (math.log(2 * math.pi) + lv + ((z_expand - mu) ** 2) / var)  # [B, k, n_cvs]

#         # Sum over dimensions to get log_prob per component
#         log_prob_comp = log_prob_per_dim.sum(dim=2)  # [B, k]

#         # Add log-weights
#         log_w = torch.log(w + self.eps).squeeze(-1)  # [k]
#         log_prob_comp += log_w.unsqueeze(0)  # [B, k]

#         # Marginalize over components via logsumexp
#         log_p = torch.logsumexp(log_prob_comp, dim=1)  # [B]

#         return log_p

#     def forward(
#         self,
#         data_targets: torch.Tensor,  # [B, C_out]
#         outputs: torch.Tensor,       # [B, C_out], log-probs
#         z_sample: torch.Tensor,      # [B, n_cvs]
#         z_mean: torch.Tensor,        # [B, n_cvs]
#         z_logvar: torch.Tensor,      # [B, n_cvs]
#         rep_mean: torch.Tensor,      # [k, n_cvs]
#         rep_logvar: torch.Tensor,    # [k, n_cvs]
#         w: torch.Tensor,             # [k, 1]
#         data_weights: torch.Tensor = None,  # [B] or None
#     ):
#         """
#         Computes:
#             rec_err = E_q[-log p(x|z)]
#             kld = E_q[ log q(z|x) - log p(z) ]
#         Returns:
#             loss, rec_err (scalar), kl_term (scalar)
#         """
#         # Reconstruction error: cross-entropy per sample
#         ce = torch.sum(-data_targets * outputs, dim=1)  # [B]
#         rec_err = (ce * data_weights).mean() if data_weights is not None else ce.mean()

#         # KL term: log q(z|x) - log p(z)
#         # log q(z|x): full multivariate diagonal Gaussian log-prob
#         log_q_per_dim = -0.5 * (math.log(2 * math.pi) + z_logvar + ((z_sample - z_mean) ** 2) / z_logvar.exp())  # [B, n_cvs]
#         log_q = log_q_per_dim.sum(dim=1)  # [B]

#         # log p(z): mixture prior
#         log_p = self.log_p(z_sample, rep_mean, rep_logvar, w)  # [B]

#         # Per-sample KL
#         kld = log_q - log_p  # [B]
#         kl_term = (kld * data_weights).mean() if data_weights is not None else kld.mean()

#         loss = rec_err + self.beta * kl_term
#         return loss, rec_err, kl_term