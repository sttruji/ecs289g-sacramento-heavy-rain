"""
Build the clean daily table and the train/val/test split from data/raw_data/.

One row per Dec-Feb day (Pacific Standard Time, midnight to midnight) with daily
values for each station in STATIONS:
    {ICAO}_prcp          24-hour rain total (mm), trace counted as 0
    {ICAO}_pres          mean sea-level pressure (hPa)
    {ICAO}_pres_change   last minus first pressure reading of the day (hPa)
    {ICAO}_dwpt          mean dew point (degC)
    {ICAO}_temp          mean temperature (degC)
    {ICAO}_wind_u/v      mean wind vector, toward east/north (km/h)
plus heavy_rain: 1 if the target station's rain total is at or above the
training-period 90th percentile of wet-day totals.

A station-day counts only if every variable has at least MIN_HOURS of 24 hourly
readings, and a day is dropped if any station fails. Winters left with fewer
than MIN_WINTER_DAYS days are dropped. Whole winters are then split
chronologically 70/15/15, so the split is deterministic with no random seed.

Outputs:
    data/clean_daily.csv       all days, with a split column
    data/train_data/train.csv
    data/train_data/val.csv
    data/test_data/test.csv

Usage:
    python scripts/data_preprocess.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw_data"

# Regions of interest: Sacramento (target) and the upwind coast. KSMF and KOAK
# have almost no hourly rainfall before 2004; adding them drops ~30 winters.
STATIONS = ["KSAC", "KSFO"]
TARGET = "KSAC"

MIN_HOURS = 20
MIN_WINTER_DAYS = 45
WET_DAY_MM = 1.0
HEAVY_PERCENTILE = 90
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}

# Raw stamps are UTC and rainfall is for the hour ending at the stamp, so
# shifting by 9 h groups hours into PST (UTC-8) midnight-to-midnight days.
# TODO: confirm the day window with the meteorology lead.
DAY_SHIFT = pd.Timedelta(hours=9)


def load_hourly(icao: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / f"{icao}.csv", parse_dates=["time_utc"], index_col="time_utc", low_memory=False)

    # Empty prcp with a source is NOAA trace rainfall (< 0.1 mm)
    df["prcp_trace"] = df["prcp"].isna() & df["prcp_source"].notna()
    df["prcp"] = df["prcp"].mask(df["prcp_trace"], 0.0)

    # Wind direction is where the wind blows from; calm hours have no direction
    rad = np.deg2rad(df["wdir"])
    calm = df["wspd"] == 0
    df["wind_u"] = (-df["wspd"] * np.sin(rad)).mask(calm, 0.0)
    df["wind_v"] = (-df["wspd"] * np.cos(rad)).mask(calm, 0.0)
    return df


def daily_station(icao: str) -> pd.DataFrame:
    """Daily values for one station, keeping only complete Dec-Feb days."""
    df = load_hourly(icao)
    g = df.groupby((df.index - DAY_SHIFT).normalize().rename("date"))

    daily = pd.DataFrame({
        "prcp": g["prcp"].sum(),
        "pres": g["pres"].mean(),
        "pres_change": g["pres"].last() - g["pres"].first(),
        "dwpt": g["dwpt"].mean(),
        "temp": g["temp"].mean(),
        "wind_u": g["wind_u"].mean(),
        "wind_v": g["wind_v"].mean(),
    })
    complete = (g[["prcp", "pres", "dwpt", "temp", "wind_u"]].count() >= MIN_HOURS).all(axis=1)
    winter_day = daily.index.month.isin([12, 1, 2])
    return daily[complete & winter_day].round(2).add_prefix(f"{icao}_")


def build_daily(stations: list[str] = STATIONS) -> pd.DataFrame:
    """Join stations on days they are all complete and drop sparse winters."""
    df = pd.concat([daily_station(icao) for icao in stations], axis=1, join="inner")
    df.insert(0, "winter", df.index.year - (df.index.month <= 2))
    days = df.groupby("winter").size()
    return df[df["winter"].isin(days.index[days >= MIN_WINTER_DAYS])]


def split_winters(winters) -> dict[str, list[int]]:
    winters = sorted(winters)
    n_train = round(SPLIT["train"] * len(winters))
    n_val = round(SPLIT["val"] * len(winters))
    return {
        "train": winters[:n_train],
        "val": winters[n_train:n_train + n_val],
        "test": winters[n_train + n_val:],
    }


def heavy_threshold(train: pd.DataFrame, percentile: float = HEAVY_PERCENTILE) -> float:
    rain = train[f"{TARGET}_prcp"]
    return rain[rain >= WET_DAY_MM].quantile(percentile / 100)


def main() -> None:
    df = build_daily()
    splits = split_winters(df["winter"].unique())
    df.insert(1, "split", df["winter"].map({w: name for name, ws in splits.items() for w in ws}))

    threshold = heavy_threshold(df[df["split"] == "train"])
    df["heavy_rain"] = (df[f"{TARGET}_prcp"] >= threshold).astype(int)

    df.to_csv(DATA_DIR / "clean_daily.csv")
    print(f"{len(df):,} days, {df['winter'].nunique()} winters, stations {', '.join(STATIONS)} -> data/clean_daily.csv")
    print(f"Heavy rain: {TARGET} >= {threshold:.1f} mm (train {HEAVY_PERCENTILE}th percentile of wet days)")

    for name, folder in [("train", "train_data"), ("val", "train_data"), ("test", "test_data")]:
        part = df[df["split"] == name].drop(columns="split")
        path = DATA_DIR / folder / f"{name}.csv"
        path.parent.mkdir(exist_ok=True)
        part.to_csv(path)
        winters = splits[name]
        print(f"  {name:5s} winters {winters[0]}-{winters[-1]} ({len(winters)}): "
              f"{len(part):,} days, {part['heavy_rain'].sum()} heavy -> {path.relative_to(DATA_DIR.parent)}")


if __name__ == "__main__":
    main()
