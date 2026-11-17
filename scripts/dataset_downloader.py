"""
Download raw hourly station observations for the Sacramento heavy-rain project.

Reads Meteostat's hourly files (NOAA ISD-Lite through Aug 2025, NOAA GHCN-Hourly
and METAR after) for Sacramento and nearby/upwind stations, keeps station
observations only (no model gap-filling), and writes one CSV per station plus a
station metadata table to data/raw_data/.

The files are read directly instead of through meteostat.hourly(), which drops
any value that is empty but has a source. Meteostat stores NOAA trace rainfall
(< 0.1 mm) exactly that way, so prcp empty + prcp_source set means trace.

Each winter is labelled by the year of its December (winter=2016 is Dec 2016 -
Feb 2017) and padded with LOOKBACK_DAYS before Dec 1 (history for the lookback
features) and all of Mar 1 (so a Pacific-time day window on the last day of Feb
is complete). Timestamps are UTC. Units: degC, %, mm, hPa, degrees, km/h.

Meteostat's hourly product carries relative humidity but not dew point, so dwpt
is recovered by inverting the Magnus formula Meteostat uses to derive rhum.

Usage:
    python scripts/dataset_downloader.py
    python scripts/dataset_downloader.py --start-year 2000 --stations KSAC KSFO
"""

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError

import numpy as np
import pandas as pd
import meteostat as ms

# ICAO -> Meteostat station ID
STATIONS = {
    "KSAC": "72483",  # Sacramento Executive Airport (target)
    "KSMF": "KSMF0",  # Sacramento International Airport
    "KSFO": "72494",  # San Francisco International Airport (upwind, coastal)
    "KOAK": "72493",  # Oakland International Airport (upwind, coastal)
    "KSCK": "72492",  # Stockton Metropolitan Airport (Central Valley, south)
    "KSJC": "KSJC0",  # San Jose International Airport (South Bay)
    "KSMX": "72394",  # Santa Maria Public Airport (central coast, south)
    "KEKA": "72594",  # Eureka, Woodley Island (north coast)
}

COLUMNS = ["temp", "rhum", "prcp", "pres", "wdir", "wspd", "wpgt"]

# NOAA station observations; values from any other source (e.g. models) are dropped
OBSERVATION_SOURCES = {"isd_lite", "ghcnh", "metar"}

# First winter with hourly records (Eureka's Meteostat record only starts in 2005)
FIRST_WINTER = 1973
LOOKBACK_DAYS = 5

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw_data"


def last_complete_winter(today: date) -> int:
    return today.year - 1 if today.month >= 4 else today.year - 2


def dew_point(temp: pd.Series, rhum: pd.Series) -> pd.Series:
    """Invert Meteostat's Magnus formula (a=17.625, b=243.04) to get dew point."""
    a, b = 17.625, 243.04
    gamma = np.log(rhum.astype(float) / 100) + a * temp.astype(float) / (b + temp.astype(float))
    return (b * gamma / (a - gamma)).round(1)


def load_year(station_id: str, year: int) -> pd.DataFrame | None:
    url = ms.config.hourly_endpoint.format(station=station_id, year=year)
    try:
        df = pd.read_csv(url, compression="gzip")
    except HTTPError as error:
        if error.code == 404:
            return None
        raise
    df.index = pd.to_datetime(df[["year", "month", "day", "hour"]])
    return df


def download_station(station_id: str, first_winter: int, last_winter: int) -> pd.DataFrame:
    years = [load_year(station_id, year) for year in range(first_winter, last_winter + 2)]
    years = [df for df in years if df is not None]
    if not years:
        return pd.DataFrame()
    # Some stations never report a variable (e.g. gusts); its columns come back empty
    df = pd.concat(years).reindex(columns=[c for col in COLUMNS for c in (col, f"{col}_source")])

    obs = pd.DataFrame(index=df.index)
    for col in COLUMNS:
        source = df[f"{col}_source"].where(df[f"{col}_source"].isin(OBSERVATION_SOURCES))
        obs[col] = df[col].where(source.notna())
        obs[f"{col}_source"] = source
    obs.insert(obs.columns.get_loc("rhum"), "dwpt", dew_point(obs["temp"], obs["rhum"]))

    winters = []
    for winter in range(first_winter, last_winter + 1):
        start = datetime(winter, 12, 1) - timedelta(days=LOOKBACK_DAYS)
        end = datetime(winter + 1, 3, 1, 23)
        chunk = obs.loc[start:end]
        if not chunk.empty:
            winters.append(chunk.assign(winter=winter))

    if not winters:
        return pd.DataFrame()

    out = pd.concat(winters)
    out.index.name = "time_utc"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start-year", type=int, default=FIRST_WINTER, help="first winter (December year)")
    parser.add_argument("--end-year", type=int, default=last_complete_winter(date.today()), help="last winter (December year)")
    parser.add_argument("--stations", nargs="+", default=list(STATIONS), choices=list(STATIONS), help="ICAO codes")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    meta = []
    for icao in args.stations:
        station_id = STATIONS[icao]
        info = ms.stations.meta(station_id)
        print(f"{icao} ({info.name}): winters {args.start_year}-{args.end_year} ...", flush=True)

        df = download_station(station_id, args.start_year, args.end_year)
        path = OUT_DIR / f"{icao}.csv"
        df.to_csv(path)

        n_winters = df["winter"].nunique() if not df.empty else 0
        print(f"  {len(df):,} rows, {n_winters} winters -> {path.relative_to(OUT_DIR.parent.parent)}")

        meta.append({
            "icao": icao,
            "meteostat_id": station_id,
            "name": info.name,
            "latitude": info.latitude,
            "longitude": info.longitude,
            "elevation_m": info.elevation,
            "first_winter": df["winter"].min() if n_winters else None,
            "last_winter": df["winter"].max() if n_winters else None,
            "winters": n_winters,
            "rows": len(df),
        })

    pd.DataFrame(meta).to_csv(OUT_DIR / "stations.csv", index=False)
    print(f"Station metadata -> {(OUT_DIR / 'stations.csv').relative_to(OUT_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
