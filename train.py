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

With --grid, each model in GRIDS instead tries every combination in its grid
(steps 1-2 per combination), keeps the one with the highest validation CSI, and
saves it as "<tag>_tuned" with every combination's score in
results/<tag>_tuned/grid_search.csv.

Usage:
    python train.py
    python train.py --models logistic_regression random_forest
    python train.py --grid
"""

import argparse
import itertools
import json
import platform
import subprocess
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import HEAVY, RAIN, csi, load_split
from models import GAT, GRU, HRRR, Climatology, LogisticRegression, Persistence, RandomForest, XGBoost
from models.features import ALL_STATIONS

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
RESULTS_DIR = ROOT / "results"

# Tag -> model. Add variants under new tags, e.g.
#     "random_forest_leaf10": lambda: RandomForest(min_samples_leaf=10),
# The ML models use KSAC + KSFO by default; the _all_stations variants add the
# optional nearby stations (not grid-searched: they look good on val but not test).
MODELS = {
    "persistence": Persistence,
    "climatology": Climatology,
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForest,
    "xgboost": XGBoost,
    "gru": GRU,
    "gat": GAT,  # graph over all stations by design
    "hrrr": HRRR,
    "logistic_regression_all_stations": lambda: LogisticRegression(stations=ALL_STATIONS),
    "random_forest_all_stations": lambda: RandomForest(stations=ALL_STATIONS),
    "xgboost_all_stations": lambda: XGBoost(stations=ALL_STATIONS),
    "gru_all_stations": lambda: GRU(stations=ALL_STATIONS),
}

# Hyperparameter grids for --grid (constructor arguments of each model)
GRIDS = {
    "climatology": {"window_days": [15, 31, 61]},
    "logistic_regression": {"C": [0.001, 0.01, 0.1, 1.0, 10.0], "lags": [1, 2, 3, 5]},
    "random_forest": {"min_samples_leaf": [1, 5, 10, 20], "lags": [1, 3, 5]},
    "xgboost": {"max_depth": [2, 3, 5], "learning_rate": [0.03, 0.1], "n_estimators": [100, 300], "lags": [3, 5]},
    "gru": {"hidden_size": [16, 32], "epochs": [20, 50, 100], "lags": [3, 5]},
    "gat": {"hidden_size": [16, 32], "epochs": [20, 50, 100], "heads": [1, 2]},
}

PACKAGES = ["numpy", "pandas", "scikit-learn", "xgboost", "torch"]


def is_plain(value) -> bool:
    if isinstance(value, (list, tuple)):
        return all(is_plain(v) for v in value)
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
    """Probability cutoff with the highest validation CSI (lowest cutoff on ties).

    Cutoffs sit halfway between neighbouring validation probabilities, so float
    noise in a repeated prediction (parallel random forests) cannot move a day
    across the cutoff.
    """
    proba = model.predict_proba(val)
    values = np.unique(proba.dropna())
    candidates = np.append(0.0, (values[:-1] + values[1:]) / 2)
    scores = [csi(val[HEAVY], (proba >= c).astype(int)) for c in candidates]
    return float(candidates[int(np.nanargmax(scores))])


def git_commit() -> str:
    """Current commit, marked -dirty if tracked code (not results/ or weights/ outputs) has uncommitted changes."""
    git = lambda *args: subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = git("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)results", ":(exclude)weights")
    return git("rev-parse", "--short", "HEAD") + ("-dirty" if dirty else "")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def fit(model, train_df: pd.DataFrame, val: pd.DataFrame) -> float:
    """Fit on train, tune the cutoff on val if the model gives probabilities, and return the fit time."""
    start = time.perf_counter()
    model.fit(train_df)
    train_seconds = time.perf_counter() - start
    if hasattr(model, "predict_proba"):
        model.cutoff = tune_cutoff(model, val)
    return train_seconds


def save(tag: str, model, params: dict, train_seconds: float, train_df: pd.DataFrame, val: pd.DataFrame,
         extra: dict | None = None) -> None:
    """Time prediction on val, then write the weights, config.json and train.json."""
    trainable = getattr(model, "trainable", True)
    tuned = hasattr(model, "predict_proba")

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
        **(extra or {}),
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
    print(f"  {tag:26s} train {f'{train_seconds:.2f}s' if trainable else 'n/a':>7s}  "
          f"predict {predict_seconds:.3f}s  cutoff {'n/a' if cutoff is None else f'{cutoff:.4g}'}"
          f"{' (tuned on val)' if tuned else ''}  -> {weights_dir.relative_to(ROOT)}")


def train(tag: str, train_df: pd.DataFrame, val: pd.DataFrame) -> None:
    model = MODELS[tag]()
    params = hyperparameters(model)
    save(tag, model, params, fit(model, train_df, val), train_df, val)


def grid_search(tag: str, train_df: pd.DataFrame, val: pd.DataFrame) -> None:
    """Try every combination in GRIDS[tag]; save the best by validation CSI as <tag>_tuned."""
    grid = GRIDS[tag]
    rows, best = [], None
    for values in itertools.product(*grid.values()):
        settings = dict(zip(grid, values))
        model = MODELS[tag](**settings)
        params = hyperparameters(model)
        train_seconds = fit(model, train_df, val)

        score = csi(val[HEAVY], model.predict_heavy(val))
        mae = float((model.predict(val) - val[RAIN]).abs().mean()) if hasattr(model, "predict") else None
        rows.append({**settings, "val_csi": round(score, 4), "val_mae": mae and round(mae, 4),
                     "cutoff": model.cutoff, "train_seconds": round(train_seconds, 4)})
        if best is None or score > best[0]:
            best = (score, model, params, train_seconds, settings)

    score, model, params, train_seconds, settings = best
    tuned_tag = f"{tag}_tuned"
    (RESULTS_DIR / tuned_tag).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / tuned_tag / "grid_search.csv", index=False)
    print(f"  {tag}: {len(rows)} combinations, best {settings} (val CSI {score:.3f})")
    save(tuned_tag, model, params, train_seconds, train_df, val, extra={
        "grid_search": {"grid": grid, "selected": settings, "selected_by": "highest val CSI", "combinations": len(rows)},
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS), help="model tags")
    parser.add_argument("--grid", action="store_true", help="grid-search the models in GRIDS, save as <tag>_tuned")
    args = parser.parse_args()

    train_df, val = load_split("train"), load_split("val")
    print(f"train {len(train_df):,} days, val {len(val):,} days")
    for tag in args.models:
        if not args.grid:
            train(tag, train_df, val)
        elif tag in GRIDS:
            grid_search(tag, train_df, val)


if __name__ == "__main__":
    main()
