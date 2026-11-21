"""
Download official daily rain totals (NOAA GHCN-Daily) for the project stations.

GHCN-Daily is NOAA's quality-controlled archive of daily station summaries. For
these first-order (NWS/FAA) stations the daily total covers midnight to midnight
local standard time, the same Pacific-time day window data_preprocess.py uses,
and is published as PRCP in tenths of a mm.

Why we need it (see README_Sarvesh.md): the hourly ISD/Meteostat record has no
rain at KSAC for winters 1994-97, no rain at KSJC/KSMF before ~2004, and on
some storm days it misses hours, so summing hourly reports undercounts heavy
days. data_preprocess.py uses these daily totals as each station's {ICAO}_prcp
(and so as the heavy-rain target) and falls back to the hourly sum only where
GHCN-Daily has no value.

Values with a GHCN quality flag (failed a NOAA QC check) are dropped. Only
Oct-Apr days are kept: the Nov-Mar season plus lookback/lookahead padding.

Writes data/ghcnd_daily.csv: one row per date, one column per ICAO (mm).

Usage:
    python scripts/ghcnd_downloader.py
"""

from pathlib import Path

import pandas as pd

# ICAO -> GHCN-Daily station ID (checked against each station's coordinates)
STATIONS = {
    "KSAC": "USW00023232",  # Sacramento Executive Airport (target)
    "KSFO": "USW00023234",  # San Francisco International Airport
    "KSCK": "USW00023237",  # Stockton Metropolitan Airport
    "KSJC": "USW00023293",  # San Jose International Airport
    "KSMX": "USW00023273",  # Santa Maria Public Airport
}

URL = "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/{id}.csv"
FIRST_YEAR = 1973
MONTHS = [10, 11, 12, 1, 2, 3, 4]

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "ghcnd_daily.csv"


def daily_rain(ghcn_id: str) -> pd.Series:
    df = pd.read_csv(URL.format(id=ghcn_id), usecols=["DATE", "PRCP", "PRCP_ATTRIBUTES"],
                     parse_dates=["DATE"], index_col="DATE", low_memory=False)
    # PRCP_ATTRIBUTES is "measurement flag,quality flag,source flag,obs time"
    quality = df["PRCP_ATTRIBUTES"].fillna("").str.split(",").str[1].fillna("")
    rain = (df["PRCP"] / 10).where(quality == "")
    rain = rain[(rain.index.year >= FIRST_YEAR) & rain.index.month.isin(MONTHS)]
    print(f"  {ghcn_id}: {rain.notna().sum():,} days, {int((quality != '').sum())} QC-flagged values dropped")
    return rain


def main() -> None:
    columns = {}
    for icao, ghcn_id in STATIONS.items():
        print(f"{icao} ({ghcn_id}) ...", flush=True)
        columns[icao] = daily_rain(ghcn_id)
    out = pd.DataFrame(columns).rename_axis("date").sort_index()
    out.to_csv(OUT_PATH)
    print(f"{len(out):,} days -> {OUT_PATH.relative_to(OUT_PATH.parent.parent)}")


if __name__ == "__main__":
    main()
