"""
Is the v2 data better? A like-for-like comparison of the data versions.

Each step adds one change from README_Sarvesh.md:
    A  main               main's committed table: Dec-Feb, hourly rain, daily features
    B  + Nov-Mar          Nov-Mar season (and v2's per-variable missing-data rule), hourly rain
    C  + GHCN-Daily       official daily rain totals as rain and target
    D  + late-evening     last-6-hour features (the full v2 table)

Each version trains on its own table, with its own features and its own
heavy_rain labels (threshold = its own training-winter 90th percentile), exactly
as that pipeline would. All versions are then scored on the SAME days against
the SAME truth, so the scores are comparable:
    truth   GHCN-Daily KSAC rain >= version D's threshold (official totals)
    E1      Dec-Feb days present in every version (A-D)
    E2      Nov-Mar days present in B-D (A cannot forecast Nov/Mar)
Scoring is leave-one-winter-out over the train+val winters (crossval.py); the
test winters are never used. AP differences are against the first version in
each evaluation set, with 90% winter-bootstrap intervals.

Needs data/raw_data/ (scripts/dataset_downloader.py) and the main branch in git.
Writes results/cv/ablation.csv, or ablation_p<percentile>.csv with --percentile.

Usage:
    python crossval_ablation.py
    python crossval_ablation.py --models logistic_regression xgboost gru --seeds 5
"""

import argparse
import io
import subprocess
import time
from pathlib import Path

import pandas as pd

from crossval import ROOT, OUT_DIR, csi, oof_seeds, summarize, winter_folds
from evaluate import HEAVY
from models import Persistence
from scripts import data_preprocess as dp
from train import MODELS

LATE = ("_late", "_late_change")


def main_table() -> pd.DataFrame:
    text = subprocess.run(["git", "show", "main:data/clean_daily.csv"], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout
    return pd.read_csv(io.StringIO(text), parse_dates=["date"], index_col="date")


def labelled(df: pd.DataFrame, percentile: float = dp.HEAVY_PERCENTILE) -> tuple[pd.DataFrame, float]:
    """Add the split's heavy_rain label with this table's own training threshold."""
    threshold = dp.heavy_threshold(df[df["winter"] < dp.VAL_FIRST_WINTER], percentile)
    return df.assign(heavy_rain=(df[f"{dp.TARGET}_prcp"] >= threshold).astype(int)), threshold


def versions(percentile: float = dp.HEAVY_PERCENTILE) -> dict[str, tuple[pd.DataFrame, float]]:
    drop_late = lambda df: df[[c for c in df if not c.endswith(LATE)]]
    a = main_table().drop(columns="split")
    print("  building B (Nov-Mar, hourly rain) ...", flush=True)
    b = drop_late(dp.build_daily(use_ghcnd=False))
    print("  building C/D (Nov-Mar, GHCN-Daily) ...", flush=True)
    d = dp.build_daily()
    # A is relabelled with the same rule (at the 90th percentile this reproduces main's own labels)
    out = {}
    for name, df in [("A main", a), ("B + Nov-Mar", b), ("C + GHCN-Daily", drop_late(d)), ("D + late-evening", d)]:
        out[name] = labelled(df, percentile)
    # Never touch the test winters
    return {k: (df[df["winter"] < dp.TEST_FIRST_WINTER], t) for k, (df, t) in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["logistic_regression", "xgboost"],
                        choices=[m for m in MODELS if m not in ("hrrr", "persistence")])
    parser.add_argument("--seeds", type=int, default=5, help="seeds averaged for the neural models")
    parser.add_argument("--percentile", type=float, default=dp.HEAVY_PERCENTILE,
                        help="heavy-rain percentile of training wet days (sensitivity check, e.g. 75)")
    args = parser.parse_args()
    out_name = "ablation.csv" if args.percentile == dp.HEAVY_PERCENTILE else f"ablation_p{args.percentile:g}.csv"

    print("Building data versions:")
    tables = versions(args.percentile)
    for name, (df, t) in tables.items():
        print(f"  {name:18s} {len(df):,} days, {df['winter'].nunique()} winters, "
              f"{int(df[HEAVY].sum())} heavy at >= {t:.1f} mm (own labels)")

    ghcnd = pd.read_csv(dp.GHCND_PATH, parse_dates=["date"], index_col="date")[dp.TARGET]
    truth_mm = tables["D + late-evening"][1]
    common = lambda names: sorted(set.intersection(*(set(tables[n][0].index) for n in names)))
    e1 = pd.DatetimeIndex([d for d in common(list(tables)) if d.month in (12, 1, 2)])
    e2 = pd.DatetimeIndex(common([n for n in tables if not n.startswith("A")]))
    evals = {"E1 Dec-Feb, all versions": (e1, list(tables)),
             "E2 Nov-Mar, B-D": (e2, [n for n in tables if not n.startswith("A")])}
    folds = winter_folds(sorted(set().union(*(set(df["winter"]) for df, _ in tables.values()))))
    print(f"Truth: GHCN-Daily {dp.TARGET} >= {truth_mm:.1f} mm. {len(folds)} leave-one-winter-out folds.")

    # Out-of-fold probabilities for every version and model, once
    oof = {}
    for model in args.models:
        for name, (df, _) in tables.items():
            start = time.perf_counter()
            oof[model, name] = oof_seeds(MODELS[model], df, folds, args.seeds)
            print(f"  {model} on {name}: {time.perf_counter() - start:.0f}s", flush=True)

    rows = []
    for eval_name, (days, names) in evals.items():
        y = (ghcnd.reindex(days) >= truth_mm).astype(int)
        winters = pd.Series(dp.winter_of(days), index=days)
        print(f"\n== {eval_name}: {len(days):,} days, {int(y.sum())} heavy (truth) ==")
        for name in names:
            df = tables[name][0]
            pers = Persistence().predict_heavy(df).reindex(days)
            has = pers.notna()
            rows.append({"eval": eval_name, "version": name, "model": "persistence",
                         "csi_best": round(csi(y[has], pers[has]), 4)})
            print(f"  {name:18s} persistence           CSI {rows[-1]['csi_best']:.3f}")
        for model in args.models:
            reference = oof[model, names[0]].reindex(days)
            for name in names:
                p = oof[model, name].reindex(days)
                s = summarize(y, winters, p, reference if name != names[0] else None)
                rows.append({"eval": eval_name, "version": name, "model": model,
                             **{k: (round(v, 4) if isinstance(v, float) else
                                    [round(x, 4) for x in v] if isinstance(v, tuple) else v) for k, v in s.items()}})
                diff = (f"  AP vs {names[0].split()[0]} {s['ap_diff']:+.3f} [{s['ap_diff_ci90'][0]:+.3f}, "
                        f"{s['ap_diff_ci90'][1]:+.3f}]") if "ap_diff" in s else ""
                print(f"  {name:18s} {model:20s}  AP {s['ap']:.3f}  ROC-AUC {s['roc_auc']:.3f}  "
                      f"CSI {s['csi_best']:.3f}  Brier {s['brier']:.4f}{diff}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / out_name, index=False)
    print(f"\n-> {(OUT_DIR / out_name).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
