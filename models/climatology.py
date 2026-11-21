"""
Climatology baseline: forecast what is typical for this point in the winter.

fit() learns, from the training winters, the average 24-hour rain total and the
heavy-rain frequency for each day of the season (days since Nov 1), smoothed
with a centered WINDOW_DAYS rolling mean. predict() returns the average rain,
predict_proba() the heavy-rain frequency, and predict_heavy() flags days whose
frequency is at or above the cutoff. By default the cutoff is the overall
training heavy-day rate, so it flags the part of the season where heavy rain is
more common than average.

Expects the clean daily table indexed by date, e.g.
    df = pd.read_csv("data/train_data/train.csv", parse_dates=["date"], index_col="date")
"""

import json
from pathlib import Path

import pandas as pd

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"

SEASON_DAYS = 152  # Nov 1 (day 0) through Mar 31 (day 150, or 151 in a leap year)
WINDOW_DAYS = 31


def season_day(dates: pd.DatetimeIndex) -> pd.Index:
    """Days since Nov 1 of each date's winter."""
    nov1 = pd.to_datetime(pd.DataFrame({"year": dates.year - (dates.month < 11), "month": 11, "day": 1}))
    return pd.Index((dates - pd.DatetimeIndex(nov1)).days)


class Climatology:
    name = "climatology"

    def __init__(self, rain_col: str = "KSAC_prcp", heavy_col: str = "heavy_rain",
                 window_days: int = WINDOW_DAYS, threshold: float | None = None):
        self.rain_col = rain_col
        self.heavy_col = heavy_col
        self.window_days = window_days
        self.threshold = threshold

    def fit(self, df: pd.DataFrame) -> "Climatology":
        by_day = df.groupby(season_day(df.index))[[self.rain_col, self.heavy_col]].mean()
        smooth = by_day.reindex(range(SEASON_DAYS)).rolling(self.window_days, center=True, min_periods=1).mean()
        self.rain_by_day = smooth[self.rain_col].round(3).tolist()
        self.heavy_by_day = smooth[self.heavy_col].round(4).tolist()
        self.cutoff = self.threshold if self.threshold is not None else round(float(df[self.heavy_col].mean()), 4)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """24-hour rain total (mm) for each day."""
        return self._lookup(df, self.rain_by_day)

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        """Heavy-rain probability for each day."""
        return self._lookup(df, self.heavy_by_day)

    def predict_heavy(self, df: pd.DataFrame) -> pd.Series:
        """Heavy-rain event (1/0) for each day."""
        return (self.predict_proba(df) >= self.cutoff).astype("Int64")

    def save(self, directory: Path | None = None) -> Path:
        directory = Path(directory or WEIGHTS_DIR / self.name)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "model.json").write_text(json.dumps(vars(self), indent=2))
        return directory

    @classmethod
    def load(cls, directory: Path | None = None) -> "Climatology":
        directory = Path(directory or WEIGHTS_DIR / cls.name)
        model = cls()
        model.__dict__.update(json.loads((directory / "model.json").read_text()))
        return model

    @staticmethod
    def _lookup(df: pd.DataFrame, values: list[float]) -> pd.Series:
        return pd.Series(values, dtype=float).reindex(season_day(df.index)).set_axis(df.index)
