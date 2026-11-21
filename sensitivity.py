"""
Sensitivity check: do the conclusions hold at a lower heavy-rain threshold?

The main results use the 90th percentile of training wet days (>= 23.4 mm, 195
events). This reruns the final-model pipeline with heavy rain relabelled at
another percentile (default 75th: >= 14.2 mm, about 2.3x the events), keeping
everything else fixed:
    - the same v2 data, splits and features
    - the same model settings crossval.py --grid selected at the 90th
      percentile (results/cv/selected.json), so only the threshold changes
    - leave-one-winter-out CV over train+val; cutoff from the pooled
      out-of-fold predictions
    - final fit on train+val, test winters scored once
    - baselines: persistence (yesterday's label), climatology, and HRRR
      (heavy when its forecast reaches the same mm threshold)

Labels are changed in memory only: data/ and weights/ are not touched.
Writes results/sensitivity_p<percentile>/cv.csv and test.csv.

Usage:
    python sensitivity.py
    python sensitivity.py --percentile 75 --seeds 3
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

import models.neural as neural
from crossval import SELECTED_PATH, csi, oof_seeds, summarize, winter_folds
from evaluate import HEAVY, RAIN, csi_diff, load_split
from models import HRRR, Climatology, Persistence, SeedAverage
from scripts import data_preprocess as dp
from train import MODELS

ROOT = Path(__file__).resolve().parent


def relabel(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return df.assign(**{HEAVY: (df[RAIN] >= threshold).astype(int)})


def fit_final(tag: str, choice: dict, train_val: pd.DataFrame, seeds: int):
    """Refit the selected settings on train+val, as train.py --cv does."""
    settings, epochs = choice["settings"], choice["epochs"]
    if not epochs:
        return MODELS[tag](**settings).fit(train_val)
    members, original = [], neural.SEED
    try:
        for s in range(seeds):
            neural.SEED = original + s
            members.append(MODELS[tag](**settings, epochs=epochs, patience=None).fit(train_val))
    finally:
        neural.SEED = original
    return SeedAverage(members)


def clean(v):
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, tuple):
        return [round(x, 4) for x in v]
    return v


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--percentile", type=float, default=75)
    parser.add_argument("--seeds", type=int, default=3, help="seeds averaged for the neural models")
    args = parser.parse_args()
    out_dir = ROOT / "results" / f"sensitivity_p{args.percentile:g}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train, val, test = load_split("train"), load_split("val"), load_split("test")
    threshold = dp.heavy_threshold(train, args.percentile)
    train, val, test = (relabel(df, threshold) for df in (train, val, test))
    train_val = pd.concat([train, val]).sort_index()
    selected = json.loads(SELECTED_PATH.read_text())
    print(f"Heavy rain at the {args.percentile:g}th percentile: >= {threshold:.1f} mm. "
          f"train+val {int(train_val[HEAVY].sum())} heavy, test {int(test[HEAVY].sum())} heavy")

    # 1. Cross-validation over train+val with the 90th-percentile settings
    folds = winter_folds(train_val["winter"].unique())
    y, winters = train_val[HEAVY], train_val["winter"]
    persistence = Persistence().predict_heavy(train_val)
    has = persistence.notna()
    cv_rows = [{"model": "persistence", "csi_best": round(csi(y[has], persistence[has]), 4)}]
    print(f"\n== Leave-one-winter-out CV ({len(folds)} folds) ==\n  {'persistence':22s} CSI {cv_rows[0]['csi_best']:.3f}")
    cutoffs = {}
    for tag in ["climatology", *selected]:
        start = time.perf_counter()
        make = Climatology if tag == "climatology" else (lambda t=tag: MODELS[t](**selected[t]["settings"]))
        p = oof_seeds(make, train_val, folds, args.seeds)
        s = summarize(y, winters, p)
        cutoffs[tag] = s["cutoff"]
        cv_rows.append({"model": tag, **{k: clean(v) for k, v in s.items()}})
        print(f"  {tag:22s} AP {s['ap']:.3f} [{s['ap_ci90'][0]:.3f}, {s['ap_ci90'][1]:.3f}]  ROC-AUC {s['roc_auc']:.3f}  "
              f"CSI {s['csi_best']:.3f} [{s['csi_best_ci90'][0]:.3f}, {s['csi_best_ci90'][1]:.3f}]  "
              f"({time.perf_counter() - start:.0f}s)", flush=True)
    pd.DataFrame(cv_rows).to_csv(out_dir / "cv.csv", index=False)

    # 2. Final fit on train+val, test scored once
    hrrr = HRRR()
    hrrr.cutoff = float(threshold)
    h_rain, h_heavy = hrrr.predict(test), hrrr.predict_heavy(test)
    yt, wt = test[HEAVY], test["winter"]
    ref = Persistence().predict_heavy(test)

    def contingency(heavy: pd.Series) -> dict:
        obs, pred = yt.astype(bool), heavy.fillna(0).astype(bool)
        return {"hits": int((obs & pred).sum()), "misses": int((obs & ~pred).sum()),
                "false_alarms": int((~obs & pred).sum())}

    test_rows = []
    has = ref.notna()
    test_rows.append({"model": "persistence", "csi": round(csi(yt[has], ref[has]), 4), **contingency(ref)})
    hs = summarize(yt, wt, h_rain)
    test_rows.append({"model": "hrrr", "csi": round(csi(yt, h_heavy), 4), **contingency(h_heavy),
                      "ap": clean(hs["ap"]), "ap_ci90": clean(hs["ap_ci90"]), "roc_auc": clean(hs["roc_auc"])})
    print(f"\n== Test winters ({len(test):,} days, {int(yt.sum())} heavy), scored once ==")
    print(f"  {'persistence':22s} CSI {test_rows[0]['csi']:.3f}")
    print(f"  {'hrrr':22s} CSI {test_rows[1]['csi']:.3f}  AP {hs['ap']:.3f} [{hs['ap_ci90'][0]:.3f}, {hs['ap_ci90'][1]:.3f}]  "
          f"hits {test_rows[1]['hits']} misses {test_rows[1]['misses']} false alarms {test_rows[1]['false_alarms']}")
    for tag in ["climatology", *selected]:
        model = Climatology().fit(train_val) if tag == "climatology" else fit_final(tag, selected[tag], train_val,
                                                                                  args.seeds)
        model.cutoff = cutoffs[tag]
        p, heavy = model.predict_proba(test), model.predict_heavy(test)
        s = summarize(yt, wt, p, reference=h_rain)
        d_p, lo_p, hi_p = csi_diff(test, heavy, ref)
        d_h, lo_h, hi_h = csi_diff(test, heavy, h_heavy)
        test_rows.append({"model": tag, "csi": round(csi(yt, heavy), 4), **contingency(heavy),
                          "ap": clean(s["ap"]), "ap_ci90": clean(s["ap_ci90"]), "roc_auc": clean(s["roc_auc"]),
                          "csi_minus_persistence": clean(d_p), "csi_minus_persistence_ci90": [clean(lo_p), clean(hi_p)],
                          "csi_minus_hrrr": clean(d_h), "csi_minus_hrrr_ci90": [clean(lo_h), clean(hi_h)],
                          "ap_minus_hrrr": clean(s["ap_diff"]), "ap_minus_hrrr_ci90": clean(s["ap_diff_ci90"])})
        r = test_rows[-1]
        print(f"  {tag:22s} CSI {r['csi']:.3f}  AP {s['ap']:.3f} [{s['ap_ci90'][0]:.3f}, {s['ap_ci90'][1]:.3f}]  "
              f"hits {r['hits']} misses {r['misses']} false alarms {r['false_alarms']}  "
              f"CSI-pers {d_p:+.3f} [{lo_p:+.3f}, {hi_p:+.3f}]  CSI-HRRR {d_h:+.3f} [{lo_h:+.3f}, {hi_h:+.3f}]  "
              f"AP-HRRR {s['ap_diff']:+.3f} [{s['ap_diff_ci90'][0]:+.3f}, {s['ap_diff_ci90'][1]:+.3f}]", flush=True)
    pd.DataFrame(test_rows).to_csv(out_dir / "test.csv", index=False)
    print(f"\n-> {(out_dir / 'cv.csv').relative_to(ROOT)}, {(out_dir / 'test.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
