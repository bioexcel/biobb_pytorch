import torch
import torch.nn as nn
import torch.nn.functional as F

def lrp_gmvae_single(model, x, latent_index, eps=1e-6):
    device = x.device
    batch_size = x.shape[0]
    in_features = model.in_features
    k = model.k

    has_norm = model.norm_in is not None
    if has_norm:
        x_input = model.norm_in(x)
    else:
        x_input = x

    qy_logit = model.encoder['qy_nn'](x_input)
    qy = torch.softmax(qy_logit, dim=1)

    y_ = torch.zeros(batch_size, k, device=device)

    zm_list = []
    intermediates_list = []

    for i in range(k):
        y = y_ + torch.eye(k, device=device)[i]

        # y_transform
        module = model.encoder['y_transform']
        z_h0 = y @ module.weight.t() + module.bias
        a_h0 = z_h0  # no activation

        intermediates = [('linear', module, y, z_h0)]

        xy = torch.cat([x_input, a_h0], dim=1)

        a = xy

        # qz_nn
        for sub_module in model.encoder['qz_nn'].children():
            if isinstance(sub_module, nn.Linear):
                z = a @ sub_module.weight.t() + sub_module.bias
                intermediates.append(('linear', sub_module, a, z))
                a = z
            elif isinstance(sub_module, nn.ReLU):
                a = F.relu(a)

        # zm_layer
        module = model.encoder['zm_layer']
        z = a @ module.weight.t() + module.bias
        intermediates.append(('linear', module, a, z))

        zm = z
        zm_list.append(zm)

        intermediates_list.append(intermediates)

    zm = torch.stack(zm_list, dim=1)  # [batch, k, n_cvs]

    term_k = qy * zm[:,:,latent_index]  # [batch, k]

    selected_a = torch.sum(term_k, dim=1)  # [batch]

    R_a = selected_a.unsqueeze(1)  # [batch,1]

    sign_a = selected_a.sign().unsqueeze(1)

    den = selected_a.unsqueeze(1) + eps * sign_a

    R_term_k = R_a * (term_k.unsqueeze(2) / den)  # [batch, k,1]

    R0 = torch.zeros(batch_size, in_features, device=device)

    for i in range(k):
        R = R_term_k[:,i,:]  # [batch,1]

        intermediates = intermediates_list[i][::-1]

        for op, module, a_prev, z_prev in intermediates:
            if op == 'linear':
                sign_z = z_prev.sign()
                Z = z_prev + eps * sign_z
                s = R / Z
                c = s @ module.weight
                R = a_prev * c

        # R is now R for xy
        R_xy = R
        R_x_k = R_xy[:, :in_features]
        R0 += R_x_k

    if has_norm:
        w = (1 / model.norm_in.std).view(1, -1).to(device)
        b = (-model.norm_in.mean / model.norm_in.std).view(1, -1).to(device)
        z = x * w + b
        sign_z = z.sign()
        Z = z + eps * sign_z
        s = R0 / Z
        c = s * w
        R0 = x * c

    return R0

def lrp_encoder(
    model: nn.Module,
    x: torch.Tensor,
    latent_index: int = None,
    eps: float = 1e-6
) -> torch.Tensor:
    if model.__class__.__name__ == 'GaussianMixtureVariationalAutoEncoder':
        if latent_index is None:
            R0 = 0
            for j in range(model.out_features):
                R0 += lrp_gmvae_single(model, x, j, eps)
            return R0
        else:
            return lrp_gmvae_single(model, x, latent_index, eps)

    # General case for MLP
    handles = []
    layers = []
    def collect_hook(module, inp, out):
        layers.append(module)

    for m in model.modules():
        if isinstance(m, nn.Linear):
            handle = m.register_forward_hook(collect_hook)
            handles.append(handle)

    model.eval()
    with torch.no_grad():
        _ = model.forward_cv(x[:1])

    for h in handles:
        h.remove()

    if len(layers) == 0:
        raise ValueError("No Linear layers found.")

    for layer in layers:
        if not isinstance(layer, nn.Linear):
            raise ValueError("LRP only supported for Linear layers in general case.")

    L = len(layers)

    has_norm = hasattr(model, 'norm_in') and model.norm_in is not None
    if has_norm:
        input_to_encoder = model.norm_in(x)
    else:
        input_to_encoder = x

    A = [input_to_encoder.clone()]
    Z = [None] * (L + 1)
    for l in range(L):
        lin = layers[l]
        z = A[l] @ lin.weight.t() + lin.bias
        sign_z = z.sign()
        Z[l+1] = z + eps * sign_z
        if l < L - 1:
            a = F.relu(z)
        else:
            a = z
        A.append(a)

    zL = A[L]
    if latent_index is None:
        R = [None] * (L + 1)
        R[L] = zL.sum(dim=1, keepdim=True)
    else:
        R = [None] * (L + 1)
        R[L] = zL[:, [latent_index]]

    for l in range(L-1, -1, -1):
        lin = layers[l]
        s = R[l+1] / Z[l+1]
        c = s @ lin.weight
        R[l] = A[l] * c

    R0 = R[0]

    if has_norm:
        w = (1 / model.norm_in.std).view(1, -1).to(x.device)
        b = (-model.norm_in.mean / model.norm_in.std).view(1, -1).to(x.device)
        z = x * w + b
        sign_z = z.sign()
        Z = z + eps * sign_z
        s = R0 / Z
        c = s * w
        R0 = x * c

    return R0



# Layer-wise Relevance Propagation
def lrp_encoder(
    encoder: nn.Module,
    x: torch.Tensor,
    latent_index: int = None,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Perform Layer‐Wise Relevance Propagation on `encoder` for input `x`.

    Arguments:
      encoder       -- an nn.Module mapping [batch, in_dim] → [batch, latent_dim],
                       built from Linear + ReLU layers.
      x             -- input tensor, shape [batch, in_dim].
      latent_index  -- which coordinate of the latent vector to explain.
                       If None, we explain the sum over all latent dims.
      eps           -- stabilization term to avoid division by zero.

    Returns:
      R0            -- relevance at the input layer, shape [batch, in_dim].
                       R0[b, i] is “how important feature i was for the chosen
                       latent coordinate (or sum).”
    """
    device = x.device

    # 1) Extract all Linear layers in execution order
    layers = []
    for module in encoder.modules():
        if isinstance(module, nn.Linear):
            layers.append(module)
    L = len(layers)

    # 2) FORWARD PASS: collect activations A[l] and pre-activations Z[l]
    A = [x.clone().to(device)]
    Z = [None] * (L + 1)
    for l, lin in enumerate(layers):
        z = A[l] @ lin.weight.t() + lin.bias  # shape [batch, out_dim]
        Z[l+1] = z + eps                       # add eps for numerical stability
        a = F.relu(z)
        A.append(a)

    # 3) INITIALIZE RELEVANCE at the top
    #    Let zL = A[L] be shape [batch, latent_dim]
    zL = A[L]
    if latent_index is None:
        # explain the sum of all latent coords
        R = [None] * (L + 1)
        R[L] = zL.sum(dim=1, keepdim=True)    # shape [batch, 1]
    else:
        # explain a single latent coordinate
        R = [None] * (L + 1)
        R[L] = zL[:, [latent_index]]          # shape [batch, 1]

    # 4) BACKWARD PASS (LRP) from layer L → 0
    #    At each step l, we have R[l+1] of shape [batch, out_dim].
    #    We want R[l] of shape [batch, in_dim].
    for l in range(L-1, -1, -1):
        lin = layers[l]
        w = lin.weight       # shape [out_dim, in_dim]
        # Z[l+1]: [batch, out_dim], R[l+1]: [batch, out_dim]
        s = R[l+1] / Z[l+1]               # [batch, out_dim]
        c = s @ w                         # [batch, in_dim], since w is [out, in]
        # multiply by the forward activation to zero‐out inactive neurons
        R[l] = A[l] * c                   # [batch, in_dim]

    # R[0] is now the relevance of each input feature
    return R[0]

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# X_batch1 = torch.tensor(data[2000:4000]).to(device)

# # explain the sum of *all* latent coords:
# R0_sum = lrp_encoder(model.encoder, X_batch1, latent_index=None)

# # To get a global feature ranking, average absolute relevance over the batch:
# R0_sum = R0_sum.reshape(R0_sum.size(0), -1, 3)

# R0_sum = R0_sum.mean(dim=2)     # [in_dim]
# global_importance = R0_sum.abs().mean(dim=0)    # [in_dim]
# global_importance = global_importance.cpu().detach().numpy()

# # Normalize
# global_importance = (global_importance - global_importance.min()) / (global_importance.max() - global_importance.min())

