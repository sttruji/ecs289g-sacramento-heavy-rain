"""
Shared data preparation and training for the PyTorch models (models/gru.py, models/gat.py, models/vae.py).

Each day's input is the previous `lags` days of every chosen station's daily
variables, shaped (stations, lags, variables) with the oldest day first. Each
station-variable is standardized with its training mean and spread. Missing
values (a dropped day, early December, an optional station's gap) become 0,
the training mean, and a station with nothing in the whole window is marked
absent.

The network has two heads trained together: a heavy-rain logit (binary
cross-entropy, heavy days weighted by the training ratio of other days to
heavy days) and log(1 + rain mm) (mean squared error). The weighting balances
the classes, so the default cutoff is 0.5; train.py tunes it on val. SEED fixes
the initial weights, dropout and batch order, so retraining gives identical models.

Early stopping: given the val winters, fit() checks that shared two-head loss
on val after every epoch, stops once it has not improved for `patience`
epochs (`epochs` is the cap) and keeps the weights from the best epoch
(`best_epoch`). Without val, or with patience=None, it trains for all `epochs`.

Expects the clean daily table indexed by date, e.g.
    df = pd.read_csv("data/train_data/train.csv", parse_dates=["date"], index_col="date")
"""

from pathlib import Path

import copy

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from models.features import LAGS, STATIONS, lag_features

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
SEED = 42

# One thread: the models are tiny, and on macOS PyTorch's OpenMP runtime crashes
# when scikit-learn's copy was loaded first
torch.set_num_threads(1)


class NeuralModel:
    name = "neural"

    def __init__(self, rain_col: str = "KSAC_prcp", heavy_col: str = "heavy_rain", lags: int = LAGS,
                 stations: list[str] = STATIONS, hidden_size: int = 32, dropout: float = 0.2, epochs: int = 200,
                 patience: int | None = 10, learning_rate: float = 1e-3, weight_decay: float = 1e-4,
                 batch_size: int = 64, threshold: float | None = None):
        self.rain_col = rain_col
        self.heavy_col = heavy_col
        self.lags = lags
        self.stations = list(stations)
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.epochs = epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.threshold = threshold

    def network(self, n_stations: int, n_variables: int) -> torch.nn.Module:
        """The network: forward(x, present) -> (heavy-rain logits, log(1 + rain mm), anything loss() needs)."""
        raise NotImplementedError

    def _array(self, df: pd.DataFrame) -> np.ndarray:
        """(days, stations, lags, variables), oldest lag first, NaN where missing."""
        X = lag_features(df, self.lags, self.stations)
        columns = [f"{s}_{v}_lag{k}" for s in self.stations for k in range(self.lags, 0, -1) for v in self.variables]
        return X[columns].to_numpy(np.float32).reshape(len(df), len(self.stations), self.lags, len(self.variables))

    def _inputs(self, df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        X = self._array(df)
        present = ~np.isnan(X).all(axis=(2, 3))
        X = np.nan_to_num((X - self.mean) / self.std)
        return torch.from_numpy(X), torch.from_numpy(present)

    def _targets(self, df: pd.DataFrame) -> dict[str, torch.Tensor]:
        """Training targets, one row per day."""
        return {"heavy": torch.from_numpy(df[self.heavy_col].to_numpy(np.float32)),
                "rain": torch.from_numpy(np.log1p(df[self.rain_col].to_numpy(np.float32)))}

    def loss(self, outputs: tuple, targets: dict[str, torch.Tensor], epoch: int) -> torch.Tensor:
        """Weighted binary cross-entropy on the heavy-rain logit plus squared error on log(1 + rain mm)."""
        heavy, rain = outputs[:2]
        return (F.binary_cross_entropy_with_logits(heavy, targets["heavy"], pos_weight=self.pos_weight)
                + F.mse_loss(rain, targets["rain"]))

    def fit(self, df: pd.DataFrame, val: pd.DataFrame | None = None) -> "NeuralModel":
        torch.manual_seed(SEED)
        first = self.stations[0]
        self.variables = [c.removeprefix(f"{first}_") for c in df if c.startswith(f"{first}_")]
        raw = self._array(df)
        self.mean = np.nan_to_num(np.nanmean(raw, axis=(0, 2), keepdims=True))
        std = np.nanstd(raw, axis=(0, 2), keepdims=True)
        self.std = np.where(std > 0, std, 1).astype(np.float32)

        X, present = self._inputs(df)
        targets = self._targets(df)
        self.pos_weight = (1 - targets["heavy"].mean()) / targets["heavy"].mean()

        self.net = self.network(len(self.stations), len(self.variables))
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        order = torch.Generator().manual_seed(SEED)
        stop_early = val is not None and self.patience is not None
        if stop_early:
            val_inputs, val_targets = self._inputs(val), self._targets(val)
        best_loss, best_weights, self.best_epoch = float("inf"), None, self.epochs
        for epoch in range(self.epochs):
            self.net.train()
            for batch in torch.randperm(len(X), generator=order).split(self.batch_size):
                outputs = self.net(X[batch], present[batch])
                loss = self.loss(outputs, {name: target[batch] for name, target in targets.items()}, epoch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if not stop_early:
                continue
            self.net.eval()
            with torch.no_grad():
                val_loss = float(NeuralModel.loss(self, self.net(*val_inputs), val_targets, epoch))
            if val_loss < best_loss:
                best_loss, best_weights, self.best_epoch = val_loss, copy.deepcopy(self.net.state_dict()), epoch + 1
            elif epoch + 1 - self.best_epoch >= self.patience:
                break
        if best_weights is not None:
            self.net.load_state_dict(best_weights)
        self.net.eval()

        self.cutoff = self.threshold if self.threshold is not None else 0.5
        return self

    def _forward(self, df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.net(*self._inputs(df))

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """24-hour rain total (mm) for each day."""
        rain = self._forward(df)[1].double().numpy()
        return pd.Series(np.expm1(rain).clip(min=0), index=df.index)

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        """Heavy-rain probability for each day."""
        return pd.Series(torch.sigmoid(self._forward(df)[0]).double().numpy(), index=df.index)

    def predict_heavy(self, df: pd.DataFrame) -> pd.Series:
        """Heavy-rain event (1/0) for each day."""
        return (self.predict_proba(df) >= self.cutoff).astype("Int64")

    def save(self, directory: Path | None = None) -> Path:
        directory = Path(directory or WEIGHTS_DIR / self.name)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, directory / "model.joblib")
        return directory

    @classmethod
    def load(cls, directory: Path | None = None) -> "NeuralModel":
        return joblib.load(Path(directory or WEIGHTS_DIR / cls.name) / "model.joblib")
