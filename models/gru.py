"""
GRU (gated recurrent unit) for both outputs: the heavy-rain event and the 24-hour rain total.

Reads the lag window one day at a time, oldest first, with every chosen
station's variables stacked into one vector per day. Its final hidden state
feeds a heavy-rain head and a rain-amount head. Data preparation, loss and
training are shared with the GAT model (models/neural.py).
"""

from torch import nn

from models.neural import NeuralModel


class GRUNet(nn.Module):
    def __init__(self, n_inputs: int, hidden_size: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(n_inputs, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.heavy = nn.Linear(hidden_size, 1)
        self.rain = nn.Linear(hidden_size, 1)

    def forward(self, x, present):
        # x: (batch, stations, lags, variables) -> one vector of all stations per day
        b, n, lags, v = x.shape
        _, h = self.gru(x.permute(0, 2, 1, 3).reshape(b, lags, n * v))
        h = self.dropout(h[-1])
        return self.heavy(h).squeeze(-1), self.rain(h).squeeze(-1)


class GRU(NeuralModel):
    name = "gru"

    def network(self, n_stations: int, n_variables: int) -> nn.Module:
        return GRUNet(n_stations * n_variables, self.hidden_size, self.dropout)
