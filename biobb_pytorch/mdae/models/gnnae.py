# import torch
# import lightning
# from mlcolvar.cvs import BaseCV
# from torch_geometric.nn import GCNConv, global_mean_pool
# from mlcolvar.core.transform.utils import Inverse
# from biobb_pytorch.mdae.featurization.normalization import Normalization
# from biobb_pytorch.mdae.loss import MSELoss

# __all__ = ["GNNAutoEncoder"]

# class GNNEncoder(torch.nn.Module):
#     def __init__(self, in_channels, hidden_dim, latent_dim):
#         super().__init__()
#         self.conv1 = GCNConv(in_channels, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         self.fc_mu = torch.nn.Linear(hidden_dim, latent_dim)

#     def forward(self, x, edge_index, batch):
#         x = torch.relu(self.conv1(x, edge_index))
#         x = torch.relu(self.conv2(x, edge_index))
#         x = global_mean_pool(x, batch)
#         z = self.fc_mu(x)
#         return z

# class GNNDecoder(torch.nn.Module):
#     def __init__(self, latent_dim, hidden_dim, out_features):
#         super().__init__()
#         self.mlp = torch.nn.Sequential(
#             torch.nn.Linear(latent_dim, hidden_dim),
#             torch.nn.ReLU(),
#             torch.nn.Linear(hidden_dim, hidden_dim),
#             torch.nn.ReLU()
#         )
#         self.out_layer = torch.nn.Linear(hidden_dim, out_features)

#     def forward(self, z, batch):
#         z_expanded = z[batch]
#         x_rec = self.out_layer(self.mlp(z_expanded))
#         return x_rec

# class GNNAutoEncoder(BaseCV, lightning.LightningModule):
#     BLOCKS = ["norm_in", "encoder", "decoder"]

#     def __init__(
#         self,
#         n_cvs: int,
#         encoder_layers: list,
#         decoder_layers: list = None,
#         edge_index: torch.Tensor = None,
#         options: dict = None,
#         **kwargs,
#     ):
#         super().__init__(in_features=encoder_layers[0], out_features=n_cvs, **kwargs)

#         self.loss_fn = MSELoss()
#         options = self.parse_options(options)

#         self.edge_index = edge_index
#         hidden_dim = encoder_layers[1] if len(encoder_layers) > 1 else 64

#         if decoder_layers is None:
#             decoder_layers = encoder_layers[::-1]

#         if options.get("norm_in", True):
#             self.norm_in = Normalization(self.in_features, **options.get("norm_in", {}))
#         else:
#             self.norm_in = None

#         self.encoder = GNNEncoder(in_channels=self.in_features, hidden_dim=hidden_dim, latent_dim=n_cvs)
#         self.decoder = GNNDecoder(latent_dim=n_cvs, hidden_dim=hidden_dim, out_features=self.in_features)

#     def forward_cv(self, x):
#         # x: [B, F], F = in_features
#         if self.norm_in is not None:
#             x = self.norm_in(x)
#         return self.encoder(x, self.edge_index, torch.zeros(x.shape[0], dtype=torch.long, device=x.device))

#     def encode_decode(self, x):
#         B, F = x.shape
#         if self.norm_in is not None:
#             x = self.norm_in(x)
#         # Simulate each frame as one node in a graph
#         z = self.encoder(x, self.edge_index, torch.zeros(B, dtype=torch.long, device=x.device))
#         x_rec = self.decoder(z, torch.arange(B, device=x.device))
#         if self.norm_in is not None:
#             x_rec = self.norm_in.inverse(x_rec)
#         return x_rec

#     def training_step(self, batch, batch_idx):
#         x = batch["data"]
#         target = batch.get("target", x)

#         x_hat = self.encode_decode(x)
#         loss = self.loss_fn(x_hat, target)

#         name = "train" if self.training else "valid"
#         self.log(f"{name}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
#         return loss

#     def get_decoder(self, return_normalization=False):
#         if return_normalization:
#             if self.norm_in is not None:
#                 inv_norm = Inverse(module=self.norm_in)
#                 decoder_model = torch.nn.Sequential(self.decoder, inv_norm)
#             else:
#                 raise ValueError("return_normalization is set to True but self.norm_in is None")
#         else:
#             decoder_model = self.decoder
#         return decoder_model
    
#     @property
#     def example_input_array(self):
#         return None