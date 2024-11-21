"""Module containing the MDAutoEncoder class and the command line interface."""

import torch


class MDAE(torch.nn.Module):
    def __init__(
        self,
        input_dimensions: int,
        num_layers: int,
        latent_dimensions: int,
        dropout: float = 0.0,
        leaky_relu: float = 0.0,
    ):
        super().__init__()
        self.input_dimensions: int = input_dimensions
        self.num_layers: int = num_layers
        self.latent_dimensions: int = latent_dimensions
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.delta: int = int((input_dimensions - latent_dimensions) / num_layers)
        self.dropout: float = dropout
        self.leaky_relu: float = leaky_relu

        # Encoder
        encoder: list = []
        nunits: int = self.input_dimensions
        for _ in range(self.num_layers - 1):
            encoder.append(torch.nn.Linear(nunits, nunits - self.delta))
            # encoder.append(torch.nn.ReLU())
            encoder.append(torch.nn.LeakyReLU(self.leaky_relu))
            encoder.append(torch.nn.Dropout(self.dropout))
            nunits = nunits - self.delta
        self.encoder = torch.nn.Sequential(*encoder)

        # Latent Space
        self.lv = torch.nn.Sequential(
            torch.nn.Linear(nunits, self.latent_dimensions), torch.nn.Sigmoid()
        )

        # Decoder
        decoder: list = []
        nunits = self.latent_dimensions
        for _ in range(self.num_layers - 1):
            decoder.append(torch.nn.Linear(nunits, nunits + self.delta))
            # decoder.append(torch.nn.ReLU())
            decoder.append(torch.nn.LeakyReLU(self.leaky_relu))
            decoder.append(torch.nn.Dropout(self.dropout))
            nunits = nunits + self.delta
        self.decoder = torch.nn.Sequential(*decoder)

        # Output
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(nunits, input_dimensions), torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        latent_space = self.lv(encoded)
        decoded = self.decoder(latent_space)
        output = self.output_layer(decoded)
        return latent_space, output
