"""
Cross-validated model scores over the train + val winters. The test winters are never used.

Why: the original pipeline picks hyperparameters, tunes the heavy-rain cutoff and
early-stops the neural models on 7 validation winters (31 heavy days), so model
rankings there are mostly noise. Here every train/val winter takes a turn as the
held-out winter (leave-one-winter-out): the model is fit on all the other
winters and predicts the held-out one. Pooling those out-of-fold predictions
scores each model on every train+val day (~175 heavy days) instead of 31.

Metrics on the pooled out-of-fold predictions:
    ap          average precision (area under the precision-recall curve): main
                ranking metric, threshold-free and suited to rare events
    brier       mean squared error of the probability (lower is better)
    roc_auc     area under the ROC curve
    csi_best    CSI at the single cutoff that maximizes it on the pooled
                predictions (slightly optimistic, but the same rule for every
                model, so fair for comparing them)
    *_ci90      90% interval from a bootstrap that resamples whole winters
Persistence has no probability, so it gets CSI only.

Neural models: early stopping uses the INNER_VAL_WINTERS most recent training
winters of each fold (never the held-out winter), and scores are averaged over
--seeds random seeds (each seed's predictions are averaged before scoring).

Writes results/cv/summary.csv and results/cv/oof.csv (out-of-fold
probabilities, one column per model, useful for case studies).

CV grid search (--grid): each model in CV_GRIDS tries every combination, scored
the same way; the highest pooled AP wins. Writes results/cv/grid_<tag>.csv and
results/cv/selected.json (settings, CV scores, the best-CSI cutoff from the
pooled out-of-fold predictions and, for neural models, the median early-stopping
epoch), which `train.py --cv` uses to train the final models on all train+val winters.

Usage:
    python crossval.py
    python crossval.py --grid                # ~45 min
    python crossval.py --models logistic_regression xgboost gru --seeds 5
    python crossval.py --folds 10          # 10 contiguous blocks of winters instead
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import models.neural as neural
from evaluate import HEAVY, csi, load_split
from models import Persistence
from train import MODELS

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results" / "cv"
INNER_VAL_WINTERS = 5
N_BOOT = 1000
SEED = 42
DEFAULT_MODELS = ["climatology", "logistic_regression", "xgboost", "gru"]
SELECTED_PATH = OUT_DIR / "selected.json"

# CV hyperparameter grids (constructor arguments). Neural models pick epochs by early stopping.
CV_GRIDS = {
    "logistic_regression": {"C": [0.001, 0.01, 0.1, 1.0, 10.0], "lags": [1, 2, 3, 5]},
    "xgboost": {"max_depth": [2, 3, 5], "learning_rate": [0.03, 0.1], "n_estimators": [100, 300], "lags": [1, 3]},
    "gru": {"hidden_size": [16, 32], "lags": [1, 3, 5]},
}
GRID_SEEDS = 3


def winter_folds(winters, k: int | None = None) -> list[list[int]]:
    """Leave-one-winter-out, or k contiguous blocks of winters."""
    winters = sorted(winters)
    if k is None:
        return [[w] for w in winters]
    return [list(block) for block in np.array_split(winters, k)]


def oof_proba(make_model, df: pd.DataFrame, folds: list[list[int]], predict_on: pd.DataFrame | None = None,
              epochs: list | None = None) -> pd.Series:
    """Out-of-fold heavy-rain probabilities.

    df is the training table (its own features and heavy_rain labels).
    predict_on, if given, is the table whose rows are predicted for each held-out
    winter (defaults to df); it lets two data versions be scored on the same days.
    epochs, if given, collects each neural fold's early-stopping epoch.
    """
    predict_on = df if predict_on is None else predict_on
    out = pd.Series(np.nan, index=predict_on.index)
    for held in folds:
        train = df[~df["winter"].isin(held)]
        test = predict_on[predict_on["winter"].isin(held)]
        if test.empty or train.empty:
            continue
        model = make_model()
        if isinstance(model, neural.NeuralModel):
            recent = sorted(train["winter"].unique())[-INNER_VAL_WINTERS:]
            inner = train["winter"].isin(recent)
            model.fit(train[~inner], train[inner])
            if epochs is not None:
                epochs.append(model.best_epoch)
        else:
            model.fit(train)
        out.loc[test.index] = model.predict_proba(test).to_numpy()
    return out


def oof_seeds(make_model, df, folds, seeds: int, predict_on=None, epochs: list | None = None) -> pd.Series:
    """oof_proba averaged over seeds for neural models (one run for everything else)."""
    if not isinstance(make_model(), neural.NeuralModel) or seeds <= 1:
        return oof_proba(make_model, df, folds, predict_on, epochs)
    runs = []
    original = neural.SEED
    try:
        for s in range(seeds):
            neural.SEED = original + s
            runs.append(oof_proba(make_model, df, folds, predict_on, epochs))
    finally:
        neural.SEED = original
    return pd.concat(runs, axis=1).mean(axis=1)


def best_csi(y: pd.Series, p: pd.Series) -> tuple[float, float]:
    """Highest CSI over cutoffs halfway between neighbouring probabilities, and that cutoff."""
    values = np.unique(p)
    candidates = np.append(values[0] - 1e-9, (values[:-1] + values[1:]) / 2)
    yb = y.to_numpy().astype(bool)
    pv = p.to_numpy()
    scores = []
    for c in candidates:
        pred = pv >= c
        hits = (yb & pred).sum()
        scores.append(hits / max(hits + (yb & ~pred).sum() + (~yb & pred).sum(), 1))
    i = int(np.argmax(scores))
    return float(scores[i]), float(candidates[i])


def scores(y: pd.Series, p: pd.Series) -> dict:
    s = {"ap": average_precision_score(y, p), "brier": brier_score_loss(y, p.clip(0, 1)), "roc_auc": roc_auc_score(y, p)}
    s["csi_best"], s["cutoff"] = best_csi(y, p)
    return s


def bootstrap(winters: pd.Series, metric, n: int = N_BOOT) -> tuple[float, float]:
    """90% interval of metric(positions) over resamples of whole winters."""
    groups = list(pd.Series(winters.to_numpy()).groupby(winters.to_numpy()).indices.values())
    rng = np.random.default_rng(SEED)
    values = [metric(np.concatenate([groups[g] for g in rng.integers(len(groups), size=len(groups))]))
              for _ in range(n)]
    low, high = np.nanpercentile(values, [5, 95])
    return float(low), float(high)


def summarize(y: pd.Series, winters: pd.Series, p: pd.Series, reference: pd.Series | None = None) -> dict:
    """Scores, their winter-bootstrap intervals, and the AP difference vs reference probabilities."""
    yv, pv = y.to_numpy().astype(int), p.to_numpy()
    ap = lambda i, q: average_precision_score(yv[i], q[i]) if yv[i].any() else np.nan
    s = scores(y, p)
    s["ap_ci90"] = bootstrap(winters, lambda i: ap(i, pv))
    s["csi_best_ci90"] = bootstrap(winters, lambda i: csi(pd.Series(yv[i]), pd.Series((pv[i] >= s["cutoff"]).astype(int))))
    if reference is not None:
        rv = reference.to_numpy()
        diff = lambda i: ap(i, pv) - ap(i, rv)
        s["ap_diff"] = diff(np.arange(len(yv)))
        s["ap_diff_ci90"] = bootstrap(winters, diff)
    return s


def fmt_ci(ci) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def clean(v):
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, tuple):
        return [round(x, 4) for x in v]
    return v


def grid_search(tag: str, df: pd.DataFrame, folds) -> dict:
    """Try every combination in CV_GRIDS[tag]; return the best by pooled out-of-fold AP."""
    grid = CV_GRIDS[tag]
    y = df[HEAVY]
    rows, best = [], None
    for values in itertools.product(*grid.values()):
        settings = dict(zip(grid, values))
        start = time.perf_counter()
        epochs = []
        p = oof_seeds(lambda: MODELS[tag](**settings), df, folds, GRID_SEEDS, epochs=epochs)
        s = scores(y, p)
        epoch = int(np.median(epochs)) if epochs else None
        rows.append({**settings, **{k: clean(v) for k, v in s.items()}, "median_epoch": epoch,
                     "seconds": round(time.perf_counter() - start, 1)})
        print(f"    {settings}  AP {s['ap']:.3f}  ROC-AUC {s['roc_auc']:.3f}  CSI {s['csi_best']:.3f}"
              + (f"  epoch {epoch}" if epoch else ""), flush=True)
        if best is None or s["ap"] > best[0]:
            best = (s["ap"], settings, p, epoch)
    pd.DataFrame(rows).to_csv(OUT_DIR / f"grid_{tag}.csv", index=False)
    _, settings, p, epoch = best
    s = summarize(y, df["winter"], p)
    return {"settings": settings, "seeds": GRID_SEEDS if epoch else None, "epochs": epoch, "cutoff": s["cutoff"],
            "cv": {k: clean(v) for k, v in s.items() if k != "cutoff"},
            "combinations": len(rows), "selected_by": "highest leave-one-winter-out average precision"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=[m for m in MODELS if m != "hrrr"])
    parser.add_argument("--seeds", type=int, default=5, help="seeds averaged for the neural models")
    parser.add_argument("--folds", type=int, default=None, help="contiguous winter blocks (default: one per winter)")
    parser.add_argument("--grid", action="store_true", help="CV grid search for the models in CV_GRIDS")
    args = parser.parse_args()

    df = pd.concat([load_split("train"), load_split("val")]).sort_index()
    folds = winter_folds(df["winter"].unique(), args.folds)
    y, winters = df[HEAVY], df["winter"]
    print(f"train+val: {len(df):,} days, {int(y.sum())} heavy, {df['winter'].nunique()} winters, {len(folds)} folds")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.grid:
        selected = json.loads(SELECTED_PATH.read_text()) if SELECTED_PATH.exists() else {}
        tags = list(CV_GRIDS) if args.models == DEFAULT_MODELS else [m for m in args.models if m in CV_GRIDS]
        for tag in tags:
            print(f"  {tag}: grid {CV_GRIDS[tag]}", flush=True)
            selected[tag] = choice = grid_search(tag, df, folds)
            SELECTED_PATH.write_text(json.dumps(selected, indent=2))
            print(f"  -> {tag}: {choice['settings']}  AP {choice['cv']['ap']:.3f} {fmt_ci(choice['cv']['ap_ci90'])}  "
                  f"cutoff {choice['cutoff']:.4f}", flush=True)
        print(f"-> {SELECTED_PATH.relative_to(ROOT)}")
        return

    persistence = Persistence().predict_heavy(df)
    has = persistence.notna()
    p_csi = csi(y[has], persistence[has])
    rows = [{"model": "persistence", "days": int(has.sum()), "csi_best": round(p_csi, 4)}]
    print(f"  {'persistence':22s} CSI {p_csi:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    oof = pd.DataFrame({"winter": winters, HEAVY: y})
    for tag in args.models:
        start = time.perf_counter()
        p = oof_seeds(MODELS[tag], df, folds, args.seeds)
        s = summarize(y, winters, p)
        oof[tag] = p.round(5)
        rows.append({"model": tag, "days": len(df), **{k: (round(v, 4) if isinstance(v, float) else
                                                         [round(x, 4) for x in v] if isinstance(v, tuple) else v)
                                                      for k, v in s.items()},
                     "seconds": round(time.perf_counter() - start, 1)})
        print(f"  {tag:22s} AP {s['ap']:.3f} {fmt_ci(s['ap_ci90'])}  Brier {s['brier']:.4f}  ROC-AUC {s['roc_auc']:.3f}  "
              f"CSI {s['csi_best']:.3f} {fmt_ci(s['csi_best_ci90'])}  ({time.perf_counter() - start:.0f}s)")

    pd.DataFrame(rows).to_csv(OUT_DIR / "summary.csv", index=False)
    oof.to_csv(OUT_DIR / "oof.csv")
    print(f"-> {(OUT_DIR / 'summary.csv').relative_to(ROOT)}, {(OUT_DIR / 'oof.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
