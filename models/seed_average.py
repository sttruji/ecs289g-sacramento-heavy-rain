"""
Average of one neural model trained with several random seeds.

crossval.py scores neural models on their seed-averaged probabilities, and the
heavy-rain cutoff it selects is on that scale, so `train.py --cv` saves the
final neural models the same way: several copies trained with consecutive
seeds, whose heavy-rain probabilities and rain totals are averaged.
"""

from pathlib import Path

import joblib
import pandas as pd

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"


class SeedAverage:
    name = "seed_average"

    def __init__(self, members: list, cutoff: float = 0.5):
        self.members = members
        self.cutoff = cutoff

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """24-hour rain total (mm) for each day."""
        return pd.concat([m.predict(df) for m in self.members], axis=1).mean(axis=1)

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        """Heavy-rain probability for each day."""
        return pd.concat([m.predict_proba(df) for m in self.members], axis=1).mean(axis=1)

    def predict_heavy(self, df: pd.DataFrame) -> pd.Series:
        """Heavy-rain event (1/0) for each day."""
        return (self.predict_proba(df) >= self.cutoff).astype("Int64")

    def save(self, directory: Path | None = None) -> Path:
        directory = Path(directory or WEIGHTS_DIR / self.name)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, directory / "model.joblib")
        return directory

    @classmethod
    def load(cls, directory: Path | None = None) -> "SeedAverage":
        return joblib.load(Path(directory or WEIGHTS_DIR / cls.name) / "model.joblib")
