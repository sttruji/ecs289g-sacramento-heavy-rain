"""
Compile every model's config, timings and scores into results/leaderboard.csv.

One row per model tag: model class, stations used as features, epochs, cutoff,
training and prediction times (results/<tag>/train.json), each evaluated
split's scores (results/<tag>/metrics_<split>.json), and the exact
hyperparameters (weights/<tag>/config.json). "n/a" marks values that do not
apply, e.g. no station features or training for persistence and HRRR, or no
MAE for a model without rain amounts.
Empty cells mean that split has not been evaluated yet. Rows are not ranked.

Usage:
    python leaderboard.py
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
RESULTS_DIR = ROOT / "results"

SPLITS = ["val", "test"]
METRICS = ["days", "forecast_days", "heavy_days", "csi", "precision", "hit_rate", "mae",
           "csi_diff_vs_persistence", "csi_diff_ci90_low", "csi_diff_ci90_high"]
NA = "n/a"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def model_row(tag: str) -> dict:
    config = read_json(WEIGHTS_DIR / tag / "config.json")
    timing = read_json(RESULTS_DIR / tag / "train.json")
    stations = config["hyperparameters"].get("stations")
    row = {
        "tag": tag,
        "model": config["model"],
        "stations": "+".join(stations) if stations else None,
        "trainable": config["trainable"],
        "epochs": config["epochs"],
        "cutoff": config["cutoff"],
        "train_seconds": timing.get("train_seconds"),
        "predict_seconds": timing.get("predict_seconds"),
    }
    for split in SPLITS:
        metrics = read_json(RESULTS_DIR / tag / f"metrics_{split}.json")
        if metrics:
            row.update({f"{split}_{m}": metrics.get(m) for m in METRICS})
    row["hyperparameters"] = json.dumps(config["hyperparameters"])
    row["git_commit"] = config["git_commit"]
    row["trained_at"] = config["trained_at"]
    return {k: NA if v is None else v for k, v in row.items()}


def main() -> None:
    tags = sorted(p.parent.name for p in WEIGHTS_DIR.glob("*/config.json"))
    board = pd.DataFrame([model_row(tag) for tag in tags]).set_index("tag")

    columns = ["model", "stations", "trainable", "epochs", "cutoff", "train_seconds", "predict_seconds"]
    columns += [f"{s}_{m}" for s in SPLITS for m in METRICS] + ["hyperparameters", "git_commit", "trained_at"]
    board = board.reindex(columns=columns).dropna(axis=1, how="all")

    path = RESULTS_DIR / "leaderboard.csv"
    board.to_csv(path)
    print(board.drop(columns=["hyperparameters", "trained_at"]).to_string())
    print(f"\n{len(board)} models -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
