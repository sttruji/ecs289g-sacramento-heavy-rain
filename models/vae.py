"""
Supervised variational autoencoder (VAE) for both outputs: the heavy-rain event and the 24-hour rain total.

A GRU reads the lag window (every chosen station's variables stacked into one
vector per day, as in models/gru.py) into the mean and log-variance of a small
Gaussian latent code z. Linear heavy-rain and rain-amount heads read z, so the
latent is supervised, and an MLP decoder rebuilds `decode` from z:
    "inputs": the lag window itself (an autoencoder)
    "today":  the chosen stations' variables on the forecast day itself
              (an extra forecasting task: known in training, never an input)

Loss = heavy-rain BCE + rain MSE (as in models/neural.py)
       + recon_weight (lambda) * MSE of the decoder on the observed, standardized values
       + kl_weight (beta) * KL divergence of z from a standard normal,
with beta ramped up from 0 over the first half of training so the heads learn
before the KL squeezes the latent. recon_weight = 0 is a variational
information bottleneck (Alemi et al. 2017); kl_weight = 0 is a supervised
autoencoder. Training samples z; prediction uses the mean, so it is deterministic.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from models.neural import NeuralModel


class VAENet(nn.Module):
    def __init__(self, n_inputs: int, hidden_size: int, latent_size: int, n_outputs: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(n_inputs, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.encode = nn.Linear(hidden_size, 2 * latent_size)  # mean and log-variance of z
        self.decode = nn.Sequential(nn.Linear(latent_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, n_outputs))
        self.heavy = nn.Linear(latent_size, 1)
        self.rain = nn.Linear(latent_size, 1)

    def forward(self, x, present):
        # x: (batch, stations, lags, variables) -> one vector of all stations per day
        b, n, lags, v = x.shape
        _, h = self.gru(x.permute(0, 2, 1, 3).reshape(b, lags, n * v))
        mean, log_var = self.encode(self.dropout(h[-1])).chunk(2, dim=-1)
        z = mean + torch.randn_like(mean) * (0.5 * log_var).exp() if self.training else mean
        return self.heavy(z).squeeze(-1), self.rain(z).squeeze(-1), self.decode(z), mean, log_var


class VAE(NeuralModel):
    name = "vae"

    def __init__(self, decode: str = "inputs", latent_size: int = 8, recon_weight: float = 1.0,
                 kl_weight: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.decode = decode
        self.latent_size = latent_size
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight

    def network(self, n_stations: int, n_variables: int) -> nn.Module:
        days = self.lags if self.decode == "inputs" else 1
        return VAENet(n_stations * n_variables, self.hidden_size, self.latent_size,
                      n_stations * days * n_variables, self.dropout)

    def _targets(self, df: pd.DataFrame) -> dict[str, torch.Tensor]:
        if self.decode == "inputs":
            raw = self._array(df)
        else:
            columns = [f"{s}_{v}" for s in self.stations for v in self.variables]
            raw = df[columns].to_numpy(np.float32).reshape(len(df), len(self.stations), 1, len(self.variables))
        decode = ((raw - self.mean) / self.std).reshape(len(df), -1)  # NaN where missing
        return {**super()._targets(df), "decode": torch.from_numpy(decode)}

    def loss(self, outputs: tuple, targets: dict[str, torch.Tensor], epoch: int) -> torch.Tensor:
        _, _, decoded, mean, log_var = outputs
        observed = ~targets["decode"].isnan()
        recon = F.mse_loss(decoded[observed], targets["decode"][observed])
        kl = -0.5 * (1 + log_var - mean ** 2 - log_var.exp()).sum(-1).mean()
        warmup = min(1.0, epoch / (self.epochs / 2))
        return super().loss(outputs, targets, epoch) + self.recon_weight * recon + warmup * self.kl_weight * kl
