"""
Score saved models on the validation or test winters.

Shared metrics (the plan's evaluate()):
    CSI, precision, hit rate   heavy-rain event, on the days the model has a forecast
    MAE                        24-hour rain total (mm), only for models that forecast amounts
    CSI diff vs persistence    on days both have a forecast, with a 90% CI from a
                               paired block bootstrap that resamples whole winters

Writes results/<tag>/metrics_<split>.json for every model saved in weights/.
Use val while developing and run test only for the final numbers; the plan
keeps the test winters untouched until then.

Usage:
    python evaluate.py
    python evaluate.py --split test
    python evaluate.py --models logistic_regression random_forest
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import models
from models import Persistence

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
RESULTS_DIR = ROOT / "results"
SPLITS = {
    "train": ROOT / "data" / "train_data" / "train.csv",
    "val": ROOT / "data" / "train_data" / "val.csv",
    "test": ROOT / "data" / "test_data" / "test.csv",
}

HEAVY, RAIN = "heavy_rain", "KSAC_prcp"
N_BOOT = 1000
SEED = 42


def load_split(split: str) -> pd.DataFrame:
    return pd.read_csv(SPLITS[split], parse_dates=["date"], index_col="date")


def ratio(a: int, b: int) -> float:
    return a / b if b else float("nan")


def contingency(observed: pd.Series, predicted: pd.Series) -> dict:
    """Hits, misses and false alarms on days with a forecast."""
    has = predicted.notna()
    obs, pred = observed[has].astype(bool), predicted[has].astype(bool)
    return {"hits": int((obs & pred).sum()), "misses": int((obs & ~pred).sum()), "false_alarms": int((~obs & pred).sum())}


def csi(observed: pd.Series, predicted: pd.Series) -> float:
    c = contingency(observed, predicted)
    return ratio(c["hits"], c["hits"] + c["misses"] + c["false_alarms"])


def evaluate(df: pd.DataFrame, heavy: pd.Series, rain: pd.Series | None = None) -> dict:
    """Scores for one model's heavy-rain flags (and rain totals, if it forecasts them)."""
    c = contingency(df[HEAVY], heavy)
    return {
        "forecast_days": int(heavy.notna().sum()),
        **c,
        "csi": ratio(c["hits"], c["hits"] + c["misses"] + c["false_alarms"]),
        "precision": ratio(c["hits"], c["hits"] + c["false_alarms"]),
        "hit_rate": ratio(c["hits"], c["hits"] + c["misses"]),
        "mae": None if rain is None else float((rain - df[RAIN]).abs().mean()),
    }


def csi_diff(df: pd.DataFrame, heavy: pd.Series, reference: pd.Series) -> tuple[float, float, float]:
    """CSI(model) - CSI(reference) and its 90% paired block-bootstrap CI over winters."""
    both = heavy.notna() & reference.notna()
    days = pd.DataFrame({"winter": df["winter"], "obs": df[HEAVY], "model": heavy, "ref": reference})[both]
    diff = lambda d: csi(d["obs"], d["model"]) - csi(d["obs"], d["ref"])

    by_winter = dict(list(days.groupby("winter")))
    winters = list(by_winter)
    rng = np.random.default_rng(SEED)
    boot = [diff(pd.concat([by_winter[w] for w in rng.choice(winters, len(winters))])) for _ in range(N_BOOT)]
    low, high = np.nanpercentile(boot, [5, 95])
    return diff(days), float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="val", choices=list(SPLITS))
    parser.add_argument("--models", nargs="+", help="model tags (default: everything in weights/)")
    args = parser.parse_args()

    tags = args.models or sorted(p.parent.name for p in WEIGHTS_DIR.glob("*/config.json"))
    df = load_split(args.split)
    reference = Persistence().predict_heavy(df)

    print(f"{args.split}: {len(df):,} days, {int(df[HEAVY].sum())} heavy")
    for tag in tags:
        config = json.loads((WEIGHTS_DIR / tag / "config.json").read_text())
        model = getattr(models, config["model"]).load(WEIGHTS_DIR / tag)
        heavy = model.predict_heavy(df)
        rain = model.predict(df) if hasattr(model, "predict") else None

        scores = {"split": args.split, "days": len(df), "heavy_days": int(df[HEAVY].sum()), **evaluate(df, heavy, rain)}
        if config["model"] != "Persistence":
            d, low, high = csi_diff(df, heavy, reference)
            scores.update(csi_diff_vs_persistence=d, csi_diff_ci90_low=low, csi_diff_ci90_high=high)
        scores = {k: round(v, 4) if isinstance(v, float) else v for k, v in scores.items()}
        scores["evaluated_at"] = datetime.now().isoformat(timespec="seconds")

        path = RESULTS_DIR / tag / f"metrics_{args.split}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(scores, indent=2))

        mae = "n/a" if scores["mae"] is None else f"{scores['mae']:.2f}"
        ci = f"  CSI diff {scores['csi_diff_vs_persistence']:+.3f} [{scores['csi_diff_ci90_low']:+.3f}, {scores['csi_diff_ci90_high']:+.3f}]" \
            if "csi_diff_vs_persistence" in scores else ""
        print(f"  {tag:20s} days {scores['forecast_days']:4d}  CSI {scores['csi']:.3f}  precision {scores['precision']:.3f}  "
              f"hit rate {scores['hit_rate']:.3f}  MAE {mae}{ci}")


if __name__ == "__main__":
    main()
