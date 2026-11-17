"""
Random forest for both outputs: the heavy-rain event and the 24-hour rain total.

Inputs are the previous 1-5 days of the chosen stations' variables (models/features.py);
missing lag days are left as NaN, which scikit-learn forests handle natively.
A classifier gives predict_proba() / predict_heavy() and a regressor gives
predict(). predict_heavy() flags days at or above the cutoff. By default the
cutoff is the training heavy-day rate, the same rule as the other models.
SEED fixes the forests so retraining gives identical models.

Expects the clean daily table indexed by date, e.g.
    df = pd.read_csv("data/train_data/train.csv", parse_dates=["date"], index_col="date")
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from models.features import LAGS, STATIONS, lag_features

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
SEED = 42


class RandomForest:
    name = "random_forest"

    def __init__(self, rain_col: str = "KSAC_prcp", heavy_col: str = "heavy_rain", lags: int = LAGS,
                 stations: list[str] = STATIONS, n_estimators: int = 500, min_samples_leaf: int = 5,
                 threshold: float | None = None):
        self.rain_col = rain_col
        self.heavy_col = heavy_col
        self.lags = lags
        self.stations = list(stations)
        self.threshold = threshold
        forest = dict(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                      max_features="sqrt", n_jobs=-1, random_state=SEED)
        self.classifier = RandomForestClassifier(**forest)
        self.regressor = RandomForestRegressor(**forest)

    def fit(self, df: pd.DataFrame) -> "RandomForest":
        X = lag_features(df, self.lags, self.stations)
        self.classifier.fit(X, df[self.heavy_col])
        self.regressor.fit(X, df[self.rain_col])
        self.cutoff = self.threshold if self.threshold is not None else float(df[self.heavy_col].mean())
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """24-hour rain total (mm) for each day."""
        return pd.Series(self.regressor.predict(lag_features(df, self.lags, self.stations)), index=df.index)

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        """Heavy-rain probability for each day."""
        return pd.Series(self.classifier.predict_proba(lag_features(df, self.lags, self.stations))[:, 1], index=df.index)

    def predict_heavy(self, df: pd.DataFrame) -> pd.Series:
        """Heavy-rain event (1/0) for each day."""
        return (self.predict_proba(df) >= self.cutoff).astype("Int64")

    def save(self, directory: Path | None = None) -> Path:
        directory = Path(directory or WEIGHTS_DIR / self.name)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, directory / "model.joblib")
        return directory

    @classmethod
    def load(cls, directory: Path | None = None) -> "RandomForest":
        return joblib.load(Path(directory or WEIGHTS_DIR / cls.name) / "model.joblib")
