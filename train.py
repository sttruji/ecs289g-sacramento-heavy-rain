"""
Train every model on the training winters and save weights, config and timings.

For each model tag in MODELS this:
    1. fits on the training winters and times it
    2. tunes the heavy-rain cutoff to maximize CSI on the validation winters
       (models with a heavy-rain probability only; the plan tunes on validation only)
    3. times prediction on the validation winters
    4. saves weights/<tag>/: the model files plus config.json with the exact
       hyperparameters, cutoff, data, git commit and package versions
    5. saves results/<tag>/train.json with the training and prediction times

Usage:
    python train.py
    python train.py --models logistic_regression random_forest
"""

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import HEAVY, csi, load_split
from models import HRRR, Climatology, LogisticRegression, Persistence, RandomForest, XGBoost

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
RESULTS_DIR = ROOT / "results"

# Tag -> model. Add variants under new tags, e.g.
#     "random_forest_leaf10": lambda: RandomForest(min_samples_leaf=10),
MODELS = {
    "persistence": Persistence,
    "climatology": Climatology,
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForest,
    "xgboost": XGBoost,
    "hrrr": HRRR,
}

PACKAGES = ["numpy", "pandas", "scikit-learn", "xgboost"]


def is_plain(value) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def hyperparameters(model) -> dict:
    """The wrapper's settings plus every setting of its scikit-learn estimators. Call before fit()."""
    params = {}
    for name, value in vars(model).items():
        if is_plain(value):
            params[name] = value
        elif hasattr(value, "get_params"):
            params[name] = {k: v for k, v in value.get_params().items() if is_plain(v)}
    return params


def tune_cutoff(model, val: pd.DataFrame) -> float:
    """Probability cutoff with the highest validation CSI (lowest cutoff on ties)."""
    proba = model.predict_proba(val)
    candidates = np.unique(proba.dropna())
    scores = [csi(val[HEAVY], (proba >= c).astype(int)) for c in candidates]
    return float(candidates[int(np.nanargmax(scores))])


def git_commit() -> str:
    git = lambda *args: subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return git("rev-parse", "--short", "HEAD") + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def train(tag: str, train_df: pd.DataFrame, val: pd.DataFrame) -> None:
    model = MODELS[tag]()
    params = hyperparameters(model)
    trainable = getattr(model, "trainable", True)

    start = time.perf_counter()
    model.fit(train_df)
    train_seconds = time.perf_counter() - start

    tuned = hasattr(model, "predict_proba")
    if tuned:
        model.cutoff = tune_cutoff(model, val)

    start = time.perf_counter()
    model.predict_heavy(val)
    if hasattr(model, "predict"):
        model.predict(val)
    predict_seconds = time.perf_counter() - start

    weights_dir = model.save(WEIGHTS_DIR / tag)
    write_json(weights_dir / "config.json", {
        "tag": tag,
        "model": type(model).__name__,
        "trainable": trainable,
        "epochs": getattr(model, "epochs", None),
        "hyperparameters": params,
        "cutoff": getattr(model, "cutoff", None),
        "cutoff_tuned_on": "val" if tuned else None,
        "data": {
            "train_winters": [int(train_df["winter"].min()), int(train_df["winter"].max())],
            "train_days": len(train_df),
            "val_winters": [int(val["winter"].min()), int(val["winter"].max())],
            "val_days": len(val),
        },
        "git_commit": git_commit(),
        "versions": {"python": platform.python_version(), **{p: version(p) for p in PACKAGES}},
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    })
    write_json(RESULTS_DIR / tag / "train.json", {
        "train_seconds": round(train_seconds, 4) if trainable else None,
        "train_days": len(train_df),
        "predict_seconds": round(predict_seconds, 4),
        "predict_days": len(val),
    })

    cutoff = getattr(model, "cutoff", None)
    print(f"  {tag:20s} train {f'{train_seconds:.2f}s' if trainable else 'n/a':>7s}  "
          f"predict {predict_seconds:.3f}s  cutoff {'n/a' if cutoff is None else f'{cutoff:.4g}'}"
          f"{' (tuned on val)' if tuned else ''}  -> {weights_dir.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS), help="model tags")
    args = parser.parse_args()

    train_df, val = load_split("train"), load_split("val")
    print(f"train {len(train_df):,} days, val {len(val):,} days")
    for tag in args.models:
        train(tag, train_df, val)


if __name__ == "__main__":
    main()
