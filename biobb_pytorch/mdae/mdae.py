"""Module containing the MDAutoEncoder class and the command line interface."""
import torch
from typing import List


class MDAE(torch.nn.Module):

    def __init__(self, input_dimensions: int, num_layers: int, latent_dimensions: int):
        super().__init__()
        self.input_dimensions: int = input_dimensions
        self.num_layers: int = num_layers
        self.latent_dimensions: int = latent_dimensions
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.delta: int = int((input_dimensions - latent_dimensions) / (num_layers + 1))

        # Encoder
        encoder: List = []
        nunits: int = self.input_dimensions
        for _ in range(self.num_layers):
            encoder.append(torch.nn.Linear(nunits, nunits - self.delta))
            encoder.append(torch.nn.ReLU())
            nunits = nunits - self.delta
        self.encoder = torch.nn.Sequential(*encoder)

        # Latent Space
        self.lv = torch.nn.Sequential(
            torch.nn.Linear(nunits, latent_dimensions),
            torch.nn.Sigmoid())

        # Decoder
        decoder: List = []
        nunits = self.latent_dimensions
        for _ in range(self.num_layers):
            decoder.append(torch.nn.Linear(nunits, nunits + self.delta))
            decoder.append(torch.nn.ReLU())
            nunits = nunits + self.delta
        self.decoder = torch.nn.Sequential(*decoder)

        # Output
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(nunits, input_dimensions),
            torch.nn.Sigmoid())

    def forward(self, x):
        encoded = self.encoder(x)
        latent_space = self.lv(encoded)
        decoded = self.decoder(latent_space)
        output = self.output_layer(decoded)
        return latent_space, output
