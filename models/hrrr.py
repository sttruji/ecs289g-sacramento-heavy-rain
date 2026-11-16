"""
HRRR reference: NOAA's operational 3 km forecast model, run by NOAA, not trained here.

predict() looks up the 24-hour KSAC rain forecast in data/hrrr_daily.csv (built
by scripts/hrrr_downloader.py from the 06 UTC run issued just before each
Pacific day). predict_heavy() flags days whose forecast reaches the heavy-rain
threshold, which fit() recomputes from the training winters exactly as
data_preprocess.py does. Days without an HRRR forecast (winters before 2018)
get NaN.

Expects the clean daily table indexed by date, e.g.
    df = pd.read_csv("data/train_data/train.csv", parse_dates=["date"], index_col="date")
"""

import json
from pathlib import Path

import pandas as pd

from scripts.data_preprocess import heavy_threshold

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "weights"
FORECASTS = ROOT / "data" / "hrrr_daily.csv"


class HRRR:
    name = "hrrr"
    trainable = False

    def fit(self, df: pd.DataFrame) -> "HRRR":
        """Nothing to learn except the heavy-rain threshold (mm) from the training winters."""
        self.cutoff = float(heavy_threshold(df))
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """24-hour rain total (mm) for each day."""
        forecasts = pd.read_csv(FORECASTS, parse_dates=["date"], index_col="date")["hrrr_prcp"]
        return forecasts.reindex(df.index)

    def predict_heavy(self, df: pd.DataFrame) -> pd.Series:
        """Heavy-rain event (1/0) for each day."""
        rain = self.predict(df)
        return (rain >= self.cutoff).astype("Int64").mask(rain.isna())

    def save(self, directory: Path | None = None) -> Path:
        directory = Path(directory or WEIGHTS_DIR / self.name)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "model.json").write_text(json.dumps(vars(self), indent=2))
        return directory

    @classmethod
    def load(cls, directory: Path | None = None) -> "HRRR":
        directory = Path(directory or WEIGHTS_DIR / cls.name)
        model = cls()
        model.__dict__.update(json.loads((directory / "model.json").read_text()))
        return model
