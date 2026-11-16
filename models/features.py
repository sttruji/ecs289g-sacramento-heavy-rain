"""
Lag features shared by the ML models.

Row D of the clean daily table is the forecast for day D, so features come only
from the previous days D-1 ... D-lags, never from day D itself. A lag whose day
is not in the data (early December, or after a dropped day) is NaN.
"""

import pandas as pd

LAGS = 5
NON_FEATURES = ["winter", "split", "heavy_rain"]


def lag_features(df: pd.DataFrame, lags: int = LAGS) -> pd.DataFrame:
    daily = df.drop(columns=NON_FEATURES, errors="ignore")
    return pd.concat(
        [
            daily.reindex(df.index - pd.Timedelta(days=k)).set_axis(df.index).add_suffix(f"_lag{k}")
            for k in range(1, lags + 1)
        ],
        axis=1,
    )
