"""
XGBoost (gradient-boosted trees) for both outputs: the heavy-rain event and the 24-hour rain total.

Inputs are the previous 1-5 days of every station variable (models/features.py);
missing lag days are left as NaN, which XGBoost handles natively. A classifier
gives predict_proba() / predict_heavy() and a regressor gives predict().
predict_heavy() flags days at or above the cutoff. By default the cutoff is the
training heavy-day rate, the same rule as the other models. SEED fixes the
row/column subsampling so retraining gives identical models.

Expects the clean daily table indexed by date, e.g.
    df = pd.read_csv("data/train_data/train.csv", parse_dates=["date"], index_col="date")
"""

from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from models.features import LAGS, lag_features

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
SEED = 42


class XGBoost:
    name = "xgboost"

    def __init__(self, rain_col: str = "KSAC_prcp", heavy_col: str = "heavy_rain", lags: int = LAGS,
                 n_estimators: int = 300, learning_rate: float = 0.05, max_depth: int = 3,
                 subsample: float = 0.8, colsample_bytree: float = 0.8, threshold: float | None = None):
        self.rain_col = rain_col
        self.heavy_col = heavy_col
        self.lags = lags
        self.threshold = threshold
        boosting = dict(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                        subsample=subsample, colsample_bytree=colsample_bytree, tree_method="hist",
                        n_jobs=-1, random_state=SEED)
        self.classifier = XGBClassifier(**boosting)
        self.regressor = XGBRegressor(**boosting)

    def fit(self, df: pd.DataFrame) -> "XGBoost":
        X = lag_features(df, self.lags)
        self.classifier.fit(X, df[self.heavy_col])
        self.regressor.fit(X, df[self.rain_col])
        self.cutoff = self.threshold if self.threshold is not None else float(df[self.heavy_col].mean())
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """24-hour rain total (mm) for each day."""
        return pd.Series(self.regressor.predict(lag_features(df, self.lags)), index=df.index)

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        """Heavy-rain probability for each day."""
        return pd.Series(self.classifier.predict_proba(lag_features(df, self.lags))[:, 1], index=df.index)

    def predict_heavy(self, df: pd.DataFrame) -> pd.Series:
        """Heavy-rain event (1/0) for each day."""
        return (self.predict_proba(df) >= self.cutoff).astype("Int64")

    def save(self, directory: Path | None = None) -> Path:
        directory = Path(directory or WEIGHTS_DIR / self.name)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, directory / "model.joblib")
        return directory

    @classmethod
    def load(cls, directory: Path | None = None) -> "XGBoost":
        return joblib.load(Path(directory or WEIGHTS_DIR / cls.name) / "model.joblib")
