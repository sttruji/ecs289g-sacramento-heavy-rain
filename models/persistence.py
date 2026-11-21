"""
Persistence baseline: the next 24 hours look like the last 24 hours.

The forecast for day D is what was observed on day D-1 at the target station:
its rain total for the 24-hour rainfall output and its heavy_rain flag for the
event output. Days whose previous day is not in the data (Nov 1, or the day
after a dropped day) get NaN.

Expects the clean daily table indexed by date, e.g.
    df = pd.read_csv("data/train_data/train.csv", parse_dates=["date"], index_col="date")
"""

import json
from pathlib import Path

import pandas as pd

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"


class Persistence:
    name = "persistence"
    trainable = False

    def __init__(self, rain_col: str = "KSAC_prcp", heavy_col: str = "heavy_rain"):
        self.rain_col = rain_col
        self.heavy_col = heavy_col

    def fit(self, df: pd.DataFrame) -> "Persistence":
        """Nothing to learn; kept so every model trains the same way."""
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """24-hour rain total (mm) for each day."""
        return self._previous_day(df, self.rain_col)

    def predict_heavy(self, df: pd.DataFrame) -> pd.Series:
        """Heavy-rain event (1/0) for each day."""
        return self._previous_day(df, self.heavy_col).astype("Int64")

    def save(self, directory: Path | None = None) -> Path:
        directory = Path(directory or WEIGHTS_DIR / self.name)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "model.json").write_text(json.dumps(vars(self), indent=2))
        return directory

    @classmethod
    def load(cls, directory: Path | None = None) -> "Persistence":
        directory = Path(directory or WEIGHTS_DIR / cls.name)
        return cls(**json.loads((directory / "model.json").read_text()))

    @staticmethod
    def _previous_day(df: pd.DataFrame, col: str) -> pd.Series:
        return df[col].reindex(df.index - pd.Timedelta(days=1)).set_axis(df.index)
