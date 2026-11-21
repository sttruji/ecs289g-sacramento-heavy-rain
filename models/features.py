"""
Lag features shared by the ML models.

Row D of the clean daily table is the forecast for day D, so features come only
from the previous days D-1 ... D-lags, never from day D itself. A lag whose day
is not in the data (early November, or after a dropped day) is NaN.

Only the columns of the chosen stations are used: STATIONS (KSAC, KSFO) by
default, or ALL_STATIONS to add the optional nearby stations.
"""

import pandas as pd

from scripts.data_preprocess import OPTIONAL_STATIONS, STATIONS

LAGS = 5
ALL_STATIONS = STATIONS + OPTIONAL_STATIONS
NON_FEATURES = ["winter", "split", "heavy_rain", "rain_source"]


def lag_features(df: pd.DataFrame, lags: int = LAGS, stations: list[str] = STATIONS) -> pd.DataFrame:
    daily = df.drop(columns=NON_FEATURES, errors="ignore")
    daily = daily[[c for c in daily if c.split("_")[0] in stations]]
    return pd.concat(
        [
            daily.reindex(df.index - pd.Timedelta(days=k)).set_axis(df.index).add_suffix(f"_lag{k}")
            for k in range(1, lags + 1)
        ],
        axis=1,
    )
