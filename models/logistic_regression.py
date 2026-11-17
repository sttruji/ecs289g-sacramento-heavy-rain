"""
Logistic regression for the heavy-rain event.

Inputs are the previous 1-5 days of the chosen stations' variables (models/features.py),
median-imputed where a lag day is missing, then standardized. predict_proba()
returns the heavy-rain probability and predict_heavy() flags days at or above
the cutoff. By default the cutoff is the training heavy-day rate, the same rule
Climatology uses. This model forecasts the event only; the 24-hour rain total
needs a regression model.

Expects the clean daily table indexed by date, e.g.
    df = pd.read_csv("data/train_data/train.csv", parse_dates=["date"], index_col="date")
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn import linear_model
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models.features import LAGS, STATIONS, lag_features

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"


class LogisticRegression:
    name = "logistic_regression"

    def __init__(self, heavy_col: str = "heavy_rain", lags: int = LAGS, stations: list[str] = STATIONS,
                 C: float = 1.0, threshold: float | None = None):
        self.heavy_col = heavy_col
        self.lags = lags
        self.stations = list(stations)
        self.threshold = threshold
        self.pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            linear_model.LogisticRegression(C=C, max_iter=1000),
        )

    def fit(self, df: pd.DataFrame) -> "LogisticRegression":
        self.pipeline.fit(lag_features(df, self.lags, self.stations), df[self.heavy_col])
        self.cutoff = self.threshold if self.threshold is not None else float(df[self.heavy_col].mean())
        return self

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        """Heavy-rain probability for each day."""
        return pd.Series(self.pipeline.predict_proba(lag_features(df, self.lags, self.stations))[:, 1], index=df.index)

    def predict_heavy(self, df: pd.DataFrame) -> pd.Series:
        """Heavy-rain event (1/0) for each day."""
        return (self.predict_proba(df) >= self.cutoff).astype("Int64")

    def save(self, directory: Path | None = None) -> Path:
        directory = Path(directory or WEIGHTS_DIR / self.name)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, directory / "model.joblib")
        return directory

    @classmethod
    def load(cls, directory: Path | None = None) -> "LogisticRegression":
        return joblib.load(Path(directory or WEIGHTS_DIR / cls.name) / "model.joblib")
