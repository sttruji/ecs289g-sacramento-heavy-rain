"""
Report data coverage and the plan's heavy-rain risk check.

Reads data/raw_data/ (run dataset_downloader.py first) and prints:
    1. Hourly coverage per station: % of Dec-Feb hours with each plan variable
    2. Usable days per station under the preprocessing completeness rule
    3. The clean table and its chronological split (data_preprocess.py settings)
    4. Risk check: heavy-rain days per winter at the 90th and 75th percentiles

Usage:
    python scripts/data_report.py
"""

from datetime import datetime

import pandas as pd

import data_preprocess as dp

VARIABLES = ["prcp", "pres", "dwpt", "wind_u", "temp"]

# Plan risk check: below this, widen to Nov-Mar or add a 75th percentile threshold
MIN_HEAVY_PER_WINTER = 3


def winter_days(winter: int) -> int:
    return (datetime(winter + 1, 3, 1) - datetime(winter, 12, 1)).days


def station_coverage(icao: str, winters: range) -> dict:
    df = dp.load_hourly(icao)
    df = df[df.index.month.isin([12, 1, 2])]
    hours = 24 * sum(winter_days(w) for w in winters)

    row = {v: round(100 * df[v].notna().sum() / hours, 1) for v in VARIABLES}
    row["trace_hours"] = int(df["prcp_trace"].sum())

    usable = dp.daily_station(icao)
    usable_winters = usable.groupby(usable.index.year - (usable.index.month <= 2)).size()
    row["usable_days_%"] = round(100 * len(usable) / (hours / 24), 1)
    row["usable_winters"] = int((usable_winters >= dp.MIN_WINTER_DAYS).sum())
    return row


def heavy_per_winter(df: pd.DataFrame, threshold: float) -> pd.Series:
    return (df[f"{dp.TARGET}_prcp"] >= threshold).groupby(df["winter"]).sum()


def main() -> None:
    stations = pd.read_csv(dp.RAW_DIR / "stations.csv")
    winters = range(stations["first_winter"].min(), stations["last_winter"].max() + 1)

    print(f"== 1-2. Coverage per station, winters {winters[0]}-{winters[-1]} ({len(winters)}) ==")
    print("Variables: % of Dec-Feb hours with a value (trace rain counts as a value, calm wind as 0).")
    print(f"Usable day: every variable has >= {dp.MIN_HOURS}/24 hours. "
          f"Usable winter: >= {dp.MIN_WINTER_DAYS} usable days.\n")
    coverage = pd.DataFrame.from_dict({icao: station_coverage(icao, winters) for icao in stations["icao"]}, orient="index")
    print(coverage.to_string())

    df = dp.build_daily()
    splits = dp.split_winters(df["winter"].unique())
    dropped = sorted(set(winters) - set(df["winter"]))
    print(f"\n== 3. Clean table: stations {', '.join(dp.STATIONS)}, optional {', '.join(dp.OPTIONAL_STATIONS)} ==")
    print(f"{len(df):,} days in {df['winter'].nunique()} winters. Dropped winters: {dropped}")
    print("Optional stations, % of days complete: "
          + ", ".join(f"{icao} {df[f'{icao}_prcp'].notna().mean():.0%}" for icao in dp.OPTIONAL_STATIONS))
    for name, ws in splits.items():
        print(f"  {name:5s} {ws[0]}-{ws[-1]} ({len(ws)} winters, {df['winter'].isin(ws).sum():,} days)")

    train = df[df["winter"].isin(splits["train"])]
    print(f"\n== 4. Risk check: {dp.TARGET} heavy-rain days per winter "
          f"(threshold = train percentile of wet days >= {dp.WET_DAY_MM} mm) ==")
    for percentile in [dp.HEAVY_PERCENTILE, 75]:
        threshold = dp.heavy_threshold(train, percentile)
        per_winter = heavy_per_winter(df, threshold)
        print(f"\n{percentile}th percentile: >= {threshold:.1f} mm")
        table = pd.DataFrame.from_dict({
            name: {"heavy_days": per_winter[ws].sum(), "per_winter": round(per_winter[ws].mean(), 2),
                   "min": per_winter[ws].min(), "max": per_winter[ws].max()}
            for name, ws in {**splits, "all": list(per_winter.index)}.items()
        }, orient="index")
        print(table.to_string())

    mean_heavy = heavy_per_winter(df, dp.heavy_threshold(train)).mean()
    verdict = "OK" if mean_heavy >= MIN_HEAVY_PER_WINTER else "LOW: decide on widening to Nov-Mar or adding the 75th percentile threshold"
    print(f"\nVerdict: {mean_heavy:.2f} heavy days/winter at the {dp.HEAVY_PERCENTILE}th percentile "
          f"(plan needs ~{MIN_HEAVY_PER_WINTER}) -> {verdict}")


if __name__ == "__main__":
    main()
