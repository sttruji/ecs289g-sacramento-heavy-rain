"""
Graph attention network (GAT) over the station network, for both outputs.

Every station is a node. A GRU shared by all stations reads each station's lag
window into a state vector, plus a learned embedding saying which station it
is. One graph-attention layer (Velickovic et al. 2018) then lets every station
take a weighted mix of all stations' states, with weights the network
computes for each day; stations absent from the whole window get no weight.
The target station's (KSAC) final state feeds a heavy-rain head and a
rain-amount head. Uses all stations by default. Data preparation, loss and
training are shared with the GRU model (models/neural.py).
"""

import torch
import torch.nn.functional as F
from torch import nn

from models.features import ALL_STATIONS
from models.neural import NeuralModel


class GATLayer(nn.Module):
    def __init__(self, size: int, heads: int):
        super().__init__()
        self.heads, self.head_size = heads, size // heads
        self.project = nn.Linear(size, size, bias=False)
        self.attend_to = nn.Parameter(0.1 * torch.randn(heads, self.head_size))
        self.attend_from = nn.Parameter(0.1 * torch.randn(heads, self.head_size))

    def forward(self, h, present):
        # h: (batch, stations, size); present: (batch, stations)
        b, n, _ = h.shape
        z = self.project(h).view(b, n, self.heads, self.head_size)
        # score[i, j]: how much station i listens to station j, per head
        scores = (z * self.attend_to).sum(-1)[:, :, None] + (z * self.attend_from).sum(-1)[:, None, :]
        allowed = present[:, None, :] | torch.eye(n, dtype=torch.bool)  # absent stations only see themselves
        scores = F.leaky_relu(scores, 0.2).masked_fill(~allowed[..., None], float("-inf"))
        weights = scores.softmax(dim=2)
        mixed = torch.einsum("bijh,bjhd->bihd", weights, z).reshape(b, n, -1)
        return h + F.elu(mixed)


class GATNet(nn.Module):
    def __init__(self, n_stations: int, n_variables: int, hidden_size: int, heads: int, dropout: float,
                 target: int):
        super().__init__()
        self.gru = nn.GRU(n_variables, hidden_size, batch_first=True)
        self.station = nn.Embedding(n_stations, hidden_size)
        self.attention = GATLayer(hidden_size, heads)
        self.dropout = nn.Dropout(dropout)
        self.heavy = nn.Linear(hidden_size, 1)
        self.rain = nn.Linear(hidden_size, 1)
        self.target = target

    def forward(self, x, present):
        # x: (batch, stations, lags, variables) -> one state per station
        b, n, lags, v = x.shape
        _, h = self.gru(x.reshape(b * n, lags, v))
        h = h[-1].reshape(b, n, -1) + self.station.weight
        h = self.dropout(self.attention(h, present)[:, self.target])
        return self.heavy(h).squeeze(-1), self.rain(h).squeeze(-1)


class GAT(NeuralModel):
    name = "gat"

    def __init__(self, stations: list[str] = ALL_STATIONS, heads: int = 2, **kwargs):
        super().__init__(stations=stations, **kwargs)
        self.heads = heads

    def network(self, n_stations: int, n_variables: int) -> nn.Module:
        target = self.stations.index(self.rain_col.split("_")[0])
        return GATNet(n_stations, n_variables, self.hidden_size, self.heads, self.dropout, target)
