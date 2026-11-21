"""
Build the clean daily table and the train/val/test split from data/raw_data/
(hourly, scripts/dataset_downloader.py) and data/ghcnd_daily.csv (official
daily rain, scripts/ghcnd_downloader.py). Rationale for the choices below:
README_Sarvesh.md.

One row per Nov-Mar day (Pacific Standard Time, midnight to midnight) with daily
values for each station in STATIONS + OPTIONAL_STATIONS:
    {ICAO}_prcp          24-hour rain total (mm): GHCN-Daily, else the sum of a
                         complete day of hourly reports (trace counted as 0)
    {ICAO}_pres          mean sea-level pressure (hPa)
    {ICAO}_pres_change   last minus first pressure reading of the day (hPa)
    {ICAO}_pres_range    highest minus lowest pressure reading of the day (hPa)
    {ICAO}_dwpt          mean dew point (degC)
    {ICAO}_temp          mean temperature (degC)
    {ICAO}_temp_range    highest minus lowest temperature of the day (degC)
    {ICAO}_wind_speed    mean wind speed, calm counted as 0 (km/h)
    {ICAO}_wind_u/v      mean wind vector, toward east/north (km/h)
  Late-evening values, from the last LATE_HOURS hours of the day (18:00-24:00 PST),
  closest to when the forecast for the next day is made:
    {ICAO}_pres_late         mean pressure (hPa)
    {ICAO}_pres_late_change  mean pressure 21-24h minus mean 15-18h PST (hPa):
                             the 6-hour pressure tendency into midnight
    {ICAO}_dwpt_late         mean dew point (degC)
    {ICAO}_wind_u/v_late     mean wind vector (km/h)
    {ICAO}_prcp_late         rain total (mm, hourly reports)
plus rain_source (where the target's rain came from: ghcnd or hourly) and
heavy_rain: 1 if the target's rain total is at or above the training-period
90th percentile of wet-day totals.

Missing values are blank per variable: a daily variable needs at least
MIN_HOURS of its 24 hourly readings, a late-evening mean LATE_MIN_HOURS of its
LATE_HOURS. A day is kept if the target's rain total is known and every other
station in STATIONS (KSFO) has all its daily variables. Winters with fewer than
MIN_WINTER_DAYS days are dropped. The split is by whole winters with fixed
boundaries (VAL_FIRST_WINTER, TEST_FIRST_WINTER) so the val and test winters stay
the same as the original Dec-Feb pipeline's, and no random seed is involved.

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
GHCND_PATH = DATA_DIR / "ghcnd_daily.csv"

# Regions of interest: Sacramento (target) and the upwind coast. KSMF and KOAK
# have almost no hourly rainfall before 2004; adding them drops ~30 winters.
STATIONS = ["KSAC", "KSFO"]
TARGET = "KSAC"

# Nearby stations that add columns but never drop a day: their values are blank
# where incomplete (e.g. KSJC pressure before 2004). KEKA has no complete days.
OPTIONAL_STATIONS = ["KSCK", "KSMX", "KSJC"]

# Nov-Mar; a winter is labelled by its December's year (winter 2016 = Nov 2016 - Mar 2017)
SEASON_MONTHS = [11, 12, 1, 2, 3]
SEASON_START_MONTH = 11

MIN_HOURS = 20
LATE_HOURS = 6
LATE_MIN_HOURS = 4
MIN_WINTER_DAYS = 75  # half a Nov-Mar season (the Dec-Feb pipeline used 45 of 90)
WET_DAY_MM = 1.0
HEAVY_PERCENTILE = 90
VAL_FIRST_WINTER = 2012
TEST_FIRST_WINTER = 2019

# Raw stamps are UTC and rainfall is for the hour ending at the stamp, so
# shifting by 9 h groups hours into PST (UTC-8) midnight-to-midnight days and
# makes the shifted hour h the PST hour h:00-(h+1):00.
DAY_SHIFT = pd.Timedelta(hours=9)

DAILY_VARIABLES = ["prcp", "pres", "pres_change", "pres_range", "dwpt", "temp", "temp_range",
                   "wind_speed", "wind_u", "wind_v"]


def winter_of(dates: pd.DatetimeIndex) -> pd.Index:
    """Winter label: the year of the season's December (Jan-Mar belong to the previous year's winter)."""
    return pd.Index(dates.year - (dates.month < SEASON_START_MONTH))


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


def late_evening(df: pd.DataFrame) -> pd.DataFrame:
    """Late-evening values per PST day from hourly rows whose index is already shifted to PST."""
    hour = df.index.hour
    day = df.index.normalize().rename("date")
    late = df[hour >= 24 - LATE_HOURS].groupby(day[hour >= 24 - LATE_HOURS])
    enough = lambda col: late[col].count() >= LATE_MIN_HOURS
    end = df[hour >= 21].groupby(day[hour >= 21])["pres"]
    start = df[(hour >= 15) & (hour < 18)].groupby(day[(hour >= 15) & (hour < 18)])["pres"]
    blocks_ok = (end.count() >= 2) & (start.count() >= 2)
    return pd.DataFrame({
        "pres_late": late["pres"].mean().where(enough("pres")),
        "pres_late_change": (end.mean() - start.mean()).where(blocks_ok),
        "dwpt_late": late["dwpt"].mean().where(enough("dwpt")),
        "wind_u_late": late["wind_u"].mean().where(enough("wind_u")),
        "wind_v_late": late["wind_v"].mean().where(enough("wind_v")),
        "prcp_late": late["prcp"].sum().where(late["prcp"].count() >= LATE_HOURS - 1),
    })


def daily_station(icao: str, ghcnd: pd.DataFrame | None, months: list[int] = SEASON_MONTHS) -> pd.DataFrame:
    """Daily and late-evening values for one station on season days, blank where incomplete.

    ghcnd=None uses only the hourly reports for rain (the original pipeline's source).
    """
    df = load_hourly(icao)
    df.index = df.index - DAY_SHIFT
    g = df.groupby(df.index.normalize().rename("date"))
    n = g[["prcp", "pres", "dwpt", "temp", "wspd", "wind_u"]].count() >= MIN_HOURS

    daily = pd.DataFrame({
        "prcp": g["prcp"].sum().where(n["prcp"]),
        "pres": g["pres"].mean().where(n["pres"]),
        "pres_change": (g["pres"].last() - g["pres"].first()).where(n["pres"]),
        "pres_range": (g["pres"].max() - g["pres"].min()).where(n["pres"]),
        "dwpt": g["dwpt"].mean().where(n["dwpt"]),
        "temp": g["temp"].mean().where(n["temp"]),
        "temp_range": (g["temp"].max() - g["temp"].min()).where(n["temp"]),
        "wind_speed": g["wspd"].mean().where(n["wspd"]),
        "wind_u": g["wind_u"].mean().where(n["wind_u"]),
        "wind_v": g["wind_v"].mean().where(n["wind_u"]),  # u and v come from the same readings
    }).join(late_evening(df))

    # Official daily totals first; the hourly sum only where GHCN-Daily has none
    official = ghcnd[icao].reindex(daily.index.union(ghcnd.index)) if ghcnd is not None and icao in ghcnd else None
    if official is not None:
        daily = daily.reindex(official.index)
        daily["prcp_source"] = np.where(official.notna(), "ghcnd", np.where(daily["prcp"].notna(), "hourly", None))
        daily["prcp"] = official.fillna(daily["prcp"])
    else:
        daily["prcp_source"] = np.where(daily["prcp"].notna(), "hourly", None)

    daily = daily[daily.index.month.isin(months)]
    source = daily.pop("prcp_source")
    out = daily.round(2).add_prefix(f"{icao}_")
    if icao == TARGET:
        out["rain_source"] = source
    return out


def build_daily(stations: list[str] = STATIONS, optional: list[str] = OPTIONAL_STATIONS,
                months: list[int] = SEASON_MONTHS, use_ghcnd: bool = True,
                min_winter_days: int = MIN_WINTER_DAYS) -> pd.DataFrame:
    """Season days where the target's rain is known and the other required stations are complete.

    months, use_ghcnd and min_winter_days exist so crossval_ablation.py can rebuild
    the table without each change (e.g. Dec-Feb, hourly rain only).
    """
    ghcnd = pd.read_csv(GHCND_PATH, parse_dates=["date"], index_col="date") if use_ghcnd else None
    df = pd.concat([daily_station(icao, ghcnd, months) for icao in stations + optional], axis=1)
    keep = df[f"{TARGET}_prcp"].notna()
    for icao in stations:
        if icao != TARGET:
            keep &= df[[f"{icao}_{v}" for v in DAILY_VARIABLES]].notna().all(axis=1)
    df = df[keep].sort_index()
    df.insert(0, "winter", winter_of(df.index))
    days = df.groupby("winter").size()
    return df[df["winter"].isin(days.index[days >= min_winter_days])]


def split_winters(winters) -> dict[str, list[int]]:
    winters = sorted(winters)
    return {
        "train": [w for w in winters if w < VAL_FIRST_WINTER],
        "val": [w for w in winters if VAL_FIRST_WINTER <= w < TEST_FIRST_WINTER],
        "test": [w for w in winters if w >= TEST_FIRST_WINTER],
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
    print(f"{len(df):,} days, {df['winter'].nunique()} winters, months {SEASON_MONTHS} -> data/clean_daily.csv")
    print(f"  target rain source: {df['rain_source'].value_counts().to_dict()}")
    for icao in STATIONS + OPTIONAL_STATIONS:
        print(f"  {icao}: rain on {df[f'{icao}_prcp'].notna().mean():.0%} of days, "
              f"pressure {df[f'{icao}_pres'].notna().mean():.0%}, late-evening pressure change "
              f"{df[f'{icao}_pres_late_change'].notna().mean():.0%}")
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
