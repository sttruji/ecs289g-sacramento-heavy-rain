"""
Download HRRR 24-hour rain forecasts for Sacramento Executive (KSAC).

For each Dec-Feb day D (Pacific Standard Time midnight to midnight, i.e. 08 UTC
to 08 UTC) this takes the 06 UTC HRRR run on D, issued just before the day
starts, and computes the forecast total APCP(0-26 h) - APCP(0-2 h) at the grid
point nearest KSAC. 26-hour forecasts only exist since HRRRv3 (July 2018), so
the first winter is 2018. Writes data/hrrr_daily.csv.

Fetches two ~1 MB GRIB2 fields per day from NOAA's AWS archive via Herbie.

Usage:
    python scripts/hrrr_downloader.py
    python scripts/hrrr_downloader.py --start-year 2019 --end-year 2019
"""

import argparse
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from herbie import Herbie

from dataset_downloader import last_complete_winter

KSAC = (38.507, -121.496)  # NOAA ISD station history
FIRST_WINTER = 2018
RUN_HOUR = 6
START_FXX, END_FXX = 2, 26  # 08 UTC on D to 08 UTC on D+1
WORKERS = 8
ATTEMPTS = 3

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "hrrr_daily.csv"


def apcp_at_ksac(run: pd.Timestamp, fxx: int) -> float:
    """Precipitation (mm) accumulated from the start of the run to forecast hour fxx."""
    H = Herbie(run, model="hrrr", product="sfc", fxx=fxx, save_dir=tempfile.gettempdir(), verbose=False)
    ds = H.xarray(f":APCP:surface:0-{fxx} hour acc")
    lon = (ds.longitude.values + 180) % 360 - 180
    iy, ix = np.unravel_index(((ds.latitude.values - KSAC[0]) ** 2 + (lon - KSAC[1]) ** 2).argmin(), lon.shape)
    return float(ds["tp"].values[iy, ix])


def forecast_day(day: pd.Timestamp) -> float:
    run = day + pd.Timedelta(hours=RUN_HOUR)
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return round(apcp_at_ksac(run, END_FXX) - apcp_at_ksac(run, START_FXX), 1)
        except Exception as error:  # network hiccup, or run/field missing from the archive
            if attempt == ATTEMPTS:
                print(f"  {day.date()}: no forecast ({error})", flush=True)
    return np.nan


def winter_days(winter: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{winter}-12-01", f"{winter + 1}-03-01", inclusive="left", name="date")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start-year", type=int, default=FIRST_WINTER, help="first winter (December year)")
    parser.add_argument("--end-year", type=int, default=last_complete_winter(date.today()), help="last winter (December year)")
    args = parser.parse_args()

    frames = []
    with ProcessPoolExecutor(WORKERS) as pool:
        for winter in range(args.start_year, args.end_year + 1):
            days = winter_days(winter)
            prcp = list(pool.map(forecast_day, days))
            frames.append(pd.DataFrame({"winter": winter, "hrrr_prcp": prcp}, index=days))
            print(f"winter {winter}: {np.isfinite(prcp).sum()}/{len(days)} days", flush=True)

    out = pd.concat(frames)
    out.to_csv(OUT_PATH)
    print(f"{out['hrrr_prcp'].notna().sum():,}/{len(out):,} days -> {OUT_PATH.relative_to(OUT_PATH.parent.parent)}")


if __name__ == "__main__":
    main()
