import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    """
    Matches the tuned checkpoint layout you have:
    30 -> 64 -> 32 -> BN(32) -> 4  (bottleneck)
    4  -> 32 -> 64 -> BN(64) -> 30
    """
    def __init__(self, input_dim: int = 30, latent_dim: int = 4):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),   # encoder.0.weight: [64, 30]
            nn.ReLU(),
            nn.Linear(64, 32),          # encoder.2.weight: [32, 64]
            nn.ReLU(),
            nn.BatchNorm1d(32),         # encoder.4.weight: [32]
            nn.ReLU(),
            nn.Linear(32, latent_dim),  # encoder.6.weight: [4, 32]
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),  # decoder.0.weight: [32, 4]
            nn.ReLU(),
            nn.Linear(32, 64),          # decoder.2.weight: [64, 32]
            nn.ReLU(),
            nn.BatchNorm1d(64),         # decoder.4.weight: [64]
            nn.ReLU(),
            nn.Linear(64, input_dim),   # decoder.6.weight: [30, 64]
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out
