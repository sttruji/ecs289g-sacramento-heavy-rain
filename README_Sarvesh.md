# Data pipeline v2: more events, official rain totals, late-evening features

Branch `Sarvesh` · Sarvesh Halbe

This branch makes three changes to the shared daily table, and each one answers a weakness found while reviewing
the original pipeline. It then re-evaluates the models on the new table: leave-one-winter-out cross-validation, CV
tuning, a final fit on train+val, a single scoring of the test winters, and a sensitivity check at a lower
threshold. The sections below give the reason, the evidence, what changed, and the caveats for each.

## Key findings (for the report)

1. **Station-only ML clearly beats persistence.** On the test winters, every model gains +0.20 to +0.27 CSI over
   persistence, and every interval excludes zero. This answers the project's main question.
2. **The biggest single gain is the late-evening (last 6 h) features**, not model choice. They add about +0.06 AP at
   the 90th percentile and +0.06 to +0.09 at the 75th, with intervals that exclude zero. The Nov–Mar season and the
   GHCN-Daily totals add events and fix the labels, but alone they are roughly neutral for skill.
3. **Logistic regression, XGBoost and the GRU are statistically tied** in cross-validation at both thresholds. They
   all choose 1 day of history.
4. **HRRR is still better for the heaviest storms.** At ≥ 23.4 mm (90th percentile, 20 test events), HRRR scores CSI
   0.32 / AP 0.54, against 0.20–0.27 / 0.33–0.43 for our models.
5. **For moderate-to-heavy rain the gap almost closes.** At ≥ 14.2 mm (75th percentile, 46 test events), the best
   station model (XGBoost) scores CSI 0.37 / AP 0.52 against HRRR's 0.40 / 0.53. The difference is −0.024
   [−0.093, +0.048], so it is not significant. The other two models are slightly but significantly behind.

**Suggested wording:** *"Using only hourly observations from five airport stations, ML models come close to NOAA's
HRRR for moderate-to-heavy rain (≥ 14 mm), but HRRR remains clearly better for the most extreme storms
(≥ 23 mm). Recent late-evening observations matter more than the choice of model."*

**Caveats to state alongside these:**
- The test set is small (20 or 46 events), so one hit moves CSI by about 0.02–0.03.
- "Close to HRRR" refers to the best of three CV-tied models, so report all three.
- The models were tuned at the 90th percentile and not re-tuned for the 75th.
- Model choice was made by cross-validation before the test winters were scored. Don't re-rank models from the
  test table.

## Data summary

| | `main` (Dec–Feb, hourly rain) | This branch |
|---|---|---|
| Season | Dec–Feb (90 days) | **Nov–Mar** (151 days) |
| Target rain source | Sum of hourly ISD reports | **NOAA GHCN-Daily** official daily total (hourly sum only as a fallback) |
| Features | Daily means and ranges | Daily values **plus the last 6 hours of each day** |
| Winters | 48 (1993–97 missing) | **53** (none missing) |
| Days | 4,076 | **7,993** |
| Heavy-rain threshold | 22.3 mm | 23.4 mm |
| Heavy days train / val / test | 72 / 22 / 12 | **144 / 31 / 20** |
| Heavy days per winter | 2.2 (below the plan's ~3) | **3.7** (risk check passes) |
| Val / test winters | 2012–18 / 2019–25 | **Same winters** (2012–18 / 2019–25) |

## 1. Season widened to Nov–Mar

**Why.** With only about 2.2 heavy days per winter, the test set held 12 events. One extra hit or miss moved CSI
by 0.03–0.08, so no two ML models could be told apart. The README's own risk check flagged this.

**Evidence that Nov and Mar are real rain months.** KSAC averages for 1998–2025:

| Month | Rain (mm) | Days ≥ 22.3 mm |
|---|---|---|
| Nov | 38 | 0.32 |
| Dec | 83 | 0.89 |
| Jan | 69 | 0.68 |
| Feb | 74 | 0.68 |
| Mar | 56 | 0.46 |
| Apr | 30 | 0.21 |

March is a real storm month and November is weaker. April and October are clearly drier, so the season stops at
Nov–Mar. In the new table, heavy days by month are Nov 22, Dec 46, Jan 49, Feb 53, Mar 25.

**What changed.**
- `SEASON_MONTHS` in `scripts/data_preprocess.py`.
- The download window in `scripts/dataset_downloader.py`: Nov 1 minus 5 lookback days, through Apr 1.
- Climatology's season calendar, which now counts days since Nov 1 over 152 days.
- `scripts/data_report.py`.
- `scripts/hrrr_downloader.py`, which now also fetches Nov/Mar and reuses forecasts it already has.
  `data/hrrr_daily.csv` now covers all 1,210 Nov–Mar days of winters 2018–25: 488 new days, none missing, and
  the existing Dec–Feb forecasts are unchanged.
- A winter is still labelled by its December's year (winter 2016 = Nov 2016 – Mar 2017).
- `MIN_WINTER_DAYS` went from 45 to 75, which is still half a season.

**Caveat.** Early- and late-season storms can differ from midwinter ones. The day-of-season (which climatology
already uses) or the month could be added as a model feature.

## 2. Official daily rain totals (GHCN-Daily) as the target

**Why.** Every rain value in `main` came from summing hourly ISD reports, which come through Meteostat. That record
has real gaps in the source, not pipeline bugs. We checked the raw Meteostat files:

- **KSAC, winters 1994–97: no hourly rain at all.** The rain source column is empty for every hour, while
  temperature, humidity and wind are present. This matches the 1990s switch to automated (ASOS) stations, when
  many sites stopped reporting hourly rain. Pressure is also missing for 1996–97.
- **KSAC, winter 1993:** about 25% of hours are missing outright, so most days failed the 20-of-24-hours rule and
  the whole winter was dropped.
- **KSJC and KSMF before ~2004:** about 3% hourly rain coverage, with temperature and wind at about 99%. Meteostat
  has only one record for each station, so there was no alternative station ID that was missed.

So 1993–97 dropped out of `main`. Those include some of the wettest winters on record, such as the January 1997
flood storm.

**Hourly sums also undercount heavy days.** On the overlapping Nov–Mar days the two sources agree well
(correlation 0.984, mean absolute difference 0.16 mm). But at a 22.3 mm threshold, GHCN-Daily has **177** heavy days
where the hourly sums have **151**. In the 30 days where they disagree:
- GHCN-Daily is higher in 28, by a median of 8.8 mm.
- Those days are 3× more likely to have hours missing from the hourly record.
- Some are clear dropouts. For example, on 2025-11-13 the hourly sum is 0.7 mm and the official total is 36.3 mm.

**The fix.** NOAA **GHCN-Daily** is the quality-controlled archive of official daily station totals. For these
NWS/FAA stations the day runs midnight to midnight local standard time, the same window our pipeline uses. Each
station ID was checked against the airport's coordinates; USW00023232 is exactly KSAC.

**What changed.**
- **New script: `scripts/ghcnd_downloader.py`.** It writes `data/ghcnd_daily.csv` with daily totals for KSAC,
  KSFO, KSCK, KSJC and KSMX, and drops values that failed NOAA's quality check (none did).
- **Rain columns.** Every station's `{ICAO}_prcp` now uses GHCN-Daily. A complete day of hourly reports is used only
  where GHCN has no value. A new `rain_source` column records which source the target's rain came from: 7,992
  ghcnd and 1 hourly.
- **Missing data handling.** A missing hourly variable now blanks only that variable (`NaN`) instead of dropping the
  whole day. The models already handle this: median imputation for the scikit-learn models, mean-fill plus a
  missing-station flag for the neural ones.
- **Day-keeping rule.** A day is now kept when the target's rain is known and KSFO is complete. KSFO is complete in
  every winter, so every day keeps its upstream signal.
- **Result:** winters 1993–97 are back, with 27 heavy days. KSAC pressure is blank for those winters, and KSFO fills
  in for it.

**Caveats.**
- San Jose's GHCN-Daily record also starts only in Oct 1998, so KSJC is still blank before then.
- KSAC hourly-derived features remain partly missing in 1993–97.

## 3. Late-evening (last 6 hours) features

**Why.** The forecast for day D is effectively made at midnight. The most useful information is the last hours of
D-1: falling pressure, wind turning southerly, moisture arriving. A midnight-to-midnight mean dilutes that. It
matters because **29% of heavy days had no rain at KSAC the day before**, so for those the only warning is the
approaching storm's pressure, wind and humidity. The original plan specified daily inputs, and nothing in the repo
shows sub-daily features were considered.

**What changed.** For every station, six new columns, computed from 18:00–24:00 PST:

| Column | Meaning |
|---|---|
| `pres_late` | Mean pressure |
| `pres_late_change` | Mean pressure 21–24h minus mean 15–18h: the 6-hour tendency into midnight |
| `dwpt_late` | Mean dew point |
| `wind_u_late`, `wind_v_late` | Mean wind vector (east-west and north-south parts) |
| `prcp_late` | Rain total from hourly reports |

Means need at least 4 of the 6 hours, and the tendency needs at least 2 readings in each 3-hour block. These
columns are still lagged like every other feature (`models/features.py`), so they only ever come from days before
the forecast day. There is no leakage.

**Evidence.** Each late-evening feature correlates more strongly with next-day heavy rain than its daily
counterpart:

| Feature | Daily | Late evening |
|---|---|---|
| KSFO north-south wind (`wind_v`) | +0.203 | **+0.243** |
| KSFO pressure change | −0.116 | **−0.145** |
| KSAC pressure | −0.122 | **−0.148** |
| KSAC pressure change | −0.094 | **−0.123** |
| KSAC dew point | +0.081 | **+0.105** |
| KSAC rain | +0.151 | **+0.161** |

**Model check.** On the single validation set (31 events) the result looked mixed: logistic regression improved
slightly and the GRU got worse. The cross-validated comparison below settles it: the late-evening features are
the largest and clearest gain of the three changes, for every model tested.

## Split

The original split was 70/15/15 of whatever winters survived. Recovering 1993–97 would have changed the boundaries,
so this branch **fixes them** instead: `VAL_FIRST_WINTER = 2012` and `TEST_FIRST_WINTER = 2019`. The validation and
test winters are therefore the same ones the team has been using. The proportions are now 74 / 13 / 13 by winter.
The test winters are unchanged, but their *days* and *labels* are not: Nov/Mar days were added, the threshold
moved, and the rain source changed. **Scores on `main`'s leaderboard are not directly comparable with scores from
this branch.**

## Cross-validation instead of one validation set (`crossval.py`)

**Why.** `main` uses the 7 validation winters for three things at once: picking hyperparameters, tuning each
model's cutoff, and early stopping. Before this branch that meant 22 events (31 now), so model rankings there were
mostly noise. For example, the GRU's response to the late-evening features *reversed* between the single
validation set and cross-validation.

**How it works.** Each of the 46 train+val winters takes a turn as the held-out winter (leave-one-winter-out). The
model is fit on the other 45 and predicts the held-out one. The pooled predictions score every model on all 175
heavy days. **The test winters are never used.**

- **Main metric: average precision (AP).** AP is the area under the precision-recall curve. It doesn't depend on a
  cutoff and suits rare events: guessing at random scores about 0.025.
- **Also reported:** ROC-AUC, Brier score, and CSI at the single best cutoff. That CSI is slightly optimistic, but
  it uses the same rule for every model, so comparisons stay fair.
- **Uncertainty:** 90% intervals come from a bootstrap that resamples whole winters.
- **Neural models:** they early-stop on the 5 most recent *training* winters of each fold (never the held-out one),
  and their scores are averaged over several seeds.

Results on the v2 table (`results/cv/summary.csv`):

| Model | AP [90% CI] | ROC-AUC | CSI (best cutoff) |
|---|---|---|---|
| Persistence | — | — | 0.055 |
| Climatology | 0.029 [0.023, 0.036] | 0.572 | 0.033 |
| Logistic regression | 0.188 [0.150, 0.241] | 0.869 | 0.156 |
| XGBoost | 0.201 [0.161, 0.251] | 0.898 | 0.165 |
| GRU (5 seeds) | 0.211 [0.165, 0.274] | 0.911 | 0.170 |

All three ML models are far above persistence and climatology, and **their intervals overlap almost entirely**, so
none is clearly the best yet. The GRU's Brier score (0.126) is poor because its class-weighted loss inflates its
probabilities. It ranks days well but would need calibrating before its probabilities could be used as
probabilities.

## Is v2 better than `main`? (`crossval_ablation.py`)

`main`'s scores and this branch's can't be compared directly, because they use different days, labels and
threshold. So this script builds every version of the data:

| Version | What it adds |
|---|---|
| A | `main` as committed |
| B | Nov–Mar season |
| C | GHCN-Daily rain |
| D | Late-evening features |

Each version trains on its own table and its own labels, exactly as its pipeline would. All versions are then
scored on **the same days against the same truth**: the GHCN-Daily rain total at or above 23.4 mm. Scoring uses
leave-one-winter-out over train+val, and the test winters are never used.

**E1: the Dec–Feb days every version can forecast** (3,527 days, 97 heavy). AP, with the change vs A and its 90% CI:

| Version | Logistic regression | XGBoost | GRU (3 seeds) |
|---|---|---|---|
| A `main` | 0.149 | 0.153 | 0.123 |
| B + Nov–Mar | 0.142 (−0.006 [−0.027, +0.012]) | 0.132 (−0.021 [−0.056, +0.014]) | 0.148 (+0.025 [+0.002, +0.061]) |
| C + GHCN-Daily | 0.155 (+0.007 [−0.021, +0.035]) | 0.161 (+0.008 [−0.030, +0.054]) | 0.158 (+0.035 [+0.016, +0.060]) |
| **D + late evening** | **0.210 (+0.061 [+0.026, +0.101])** | **0.214 (+0.061 [+0.010, +0.123])** | **0.192 (+0.069 [+0.038, +0.106])** |

ROC-AUC rises at every step for every model: from 0.82–0.86 for A to 0.88–0.92 for D.

**E2: all Nov–Mar days, versions B–D** (6,100 days, 139 heavy). Adding the late-evening features to version C
raises AP by **+0.046 (logistic regression), +0.063 (XGBoost) and +0.038 (GRU)**, and all three intervals
exclude zero.

**What this shows.**
1. **v2 is better than `main`.** For every model, the full v2 table (D) beats `main` (A) on `main`'s own Dec–Feb
   days, with intervals that exclude zero. AP rises by about 40% (0.15 → 0.21).
2. **Most of the gain comes from the late-evening features.** This was the change the single validation set had
   made look harmful for the GRU.
3. **Nov–Mar and GHCN-Daily do not change skill on Dec–Feb days by themselves**, except for a small, consistent
   gain for the GRU. Their value is different:
   - They make the evaluation itself reliable: 175 events instead of 94 in train+val, and 20 instead of 12 in test.
   - They correct about 1 in 6 heavy-day labels (30 of the ~177 heavy days on which the two rain sources disagree).
   - They let the model forecast November and March.

## Fix: XGBoost crashed on macOS

On this Mac, `train.py --models xgboost` crashed with a segmentation fault (exit 139). PyTorch loads its own copy of
the OpenMP library (`libomp`), and fitting XGBoost afterwards, with Homebrew's copy, crashes the process.
`models/__init__.py` now imports XGBoost before the PyTorch models, and both train normally.

## Final models: tuned by cross-validation, then tested once

**Tuning (`crossval.py --grid`).** A leave-one-winter-out grid search over the 46 train+val winters, selected by
pooled AP. Neural models are averaged over 3 seeds. The full grids are in `results/cv/grid_<model>.csv` and the
choices in `results/cv/selected.json`.

| Model | Selected settings | CV AP [90% CI] | CV ROC-AUC | Cutoff (from CV) |
|---|---|---|---|---|
| Logistic regression | C = 10, 1 day of history | 0.220 [0.176, 0.276] | 0.903 | 0.180 |
| XGBoost | depth 5, learning rate 0.03, 100 trees, 1 day of history | 0.231 [0.176, 0.292] | 0.907 | 0.120 |
| GRU | 32 hidden units, 1 day of history, 8 epochs | 0.229 [0.180, 0.281] | 0.914 | 0.914 |

**All three choose 1 day of history.** Adding more days lowers AP in every grid, which fits the ablation result
that the latest observations matter most. For logistic regression, C = 10 is at the edge of the grid, but it beats
C = 1 by only 0.002 AP.

**Final training (`train.py --cv`).** Each selected setting is refit on **all train+val winters** and saved as
`<model>_cv`.
- The cutoff comes from the cross-validated predictions, not from the validation set.
- The GRU trains for the median early-stopping epoch, averaged over 3 seeds (`models/seed_average.py`).
- Because val is part of their training data, `evaluate.py` scores the `_cv` models on test only.

**Test winters 2019–25, scored once** (1,054 days, 20 heavy). HRRR forecasts every test day.

| Model | Test CSI | Test AP [90% CI] | AP − HRRR [90% CI] | CSI − HRRR [90% CI] |
|---|---|---|---|---|
| Persistence | 0.000 | — | — | — |
| Climatology | 0.030 | — | — | — |
| Logistic regression (CV) | 0.222 | 0.329 [0.230, 0.527] | −0.210 [−0.380, +0.017] | −0.101 [−0.265, +0.064] |
| XGBoost (CV) | 0.200 | 0.380 [0.233, 0.644] | −0.158 [−0.317, +0.019] | −0.124 [−0.241, −0.010] |
| **GRU (CV)** | **0.273** | **0.425 [0.277, 0.653]** | −0.114 [−0.320, +0.137] | −0.051 [−0.243, +0.158] |
| **HRRR** (NOAA's weather model) | **0.324** | **0.539 [0.400, 0.716]** | — | — |

**How to read this.**
1. **Every station-only model beats persistence by a wide margin.** The gain over persistence is +0.20 to +0.27
   CSI, and every interval excludes zero. That answers the project's core question: yes.
2. **HRRR is still better.** Its AP is clearly higher (0.54 vs 0.33–0.43). Only XGBoost's CSI gap is outside noise,
   though; the GRU's interval for the CSI gap includes zero.
3. **Test AP is higher than CV AP** (0.33–0.43 vs about 0.22). The 2019–25 winters seem easier, or have more
   complete data, than the 1970s–2010s average. This is one reason to report CV and test together.
4. **Don't pick models from the test table.** Among all retrained models, the val-tuned VAE scores 0.288 and the
   default VAE 0.294 on test, above the CV models. But 20 events can't rank models: one more hit moves CSI by about
   0.03. The same applies within the CV models: the GRU's higher test score is not evidence that it is better,
   because the three were tied in CV. **Recommended for the report: present all three CV models as tied, with
   logistic regression as the simple, interpretable baseline.**

## Sensitivity check: 75th-percentile threshold (`sensitivity.py`, `crossval_ablation.py --percentile 75`)

**Why.** The main results rest on 195 heavy days (20 in test). Lowering the bar to the 75th percentile of training
wet days (≥ 14.2 mm, "moderate-to-heavy" rain) gives **453 events (46 in test)**. This checks whether the
conclusions hold with 2.3× the data.

**What stays fixed:** the data, splits and features, plus the model settings selected at the 90th percentile. Only
the labels change, and they are relabelled in memory: `data/` and `weights/` are untouched. Outputs are in
`results/sensitivity_p75/` and `results/cv/ablation_p75.csv`. The 90th percentile remains the main result.

**1. The late-evening features are still the main gain.** AP on the same Dec–Feb days, against the same truth:

| Version | Logistic regression | XGBoost | GRU |
|---|---|---|---|
| A `main` | 0.301 | 0.297 | 0.288 |
| B + Nov–Mar | 0.295 | 0.281 | 0.310 |
| C + GHCN-Daily | 0.312 | 0.303 | 0.322 |
| **D + late evening** | **0.370** (+0.069 [+0.033, +0.110]) | **0.359** (+0.062 [+0.025, +0.105]) | **0.381** (+0.093 [+0.055, +0.132]) |

The pattern is the same as at the 90th percentile. The late-evening features add +0.06 to +0.09 AP over `main`,
with intervals that exclude zero. Nov–Mar and GHCN alone are roughly neutral, except for small gains for the GRU.

**2. The three models are still tied in cross-validation**, now with tighter intervals:

| Model | CV AP [90% CI] | CV CSI (best cutoff) |
|---|---|---|
| Persistence | — | 0.131 |
| Climatology | 0.064 | 0.067 |
| Logistic regression | 0.385 [0.345, 0.432] | 0.286 |
| XGBoost | 0.393 [0.354, 0.437] | 0.284 |
| GRU (3 seeds) | 0.380 [0.342, 0.422] | 0.285 |

**3. Test (46 events): the gap to HRRR nearly closes at this threshold.**

| Model | CSI | Hits / misses / false alarms | AP | CSI − HRRR [90% CI] |
|---|---|---|---|---|
| Persistence | 0.046 | — | — | — |
| Logistic regression | 0.315 | 29 / 17 / 46 | 0.463 | −0.080 [−0.197, −0.004] |
| **XGBoost** | **0.371** | 33 / 13 / 43 | **0.518** | **−0.024 [−0.093, +0.048]** |
| GRU | 0.342 | 26 / 20 / 30 | 0.499 | −0.053 [−0.124, −0.006] |
| **HRRR** | **0.395** | 32 / 14 / 35 | **0.526** | — |

**What this adds.**
- At the 75th percentile XGBoost is statistically indistinguishable from HRRR on both CSI and AP: it catches 33 of
  46 events against HRRR's 32, with 8 more false alarms. Logistic regression and the GRU are slightly behind.
  XGBoost was not chosen in advance (all three are tied in CV), so the claim is about the best of the three.
- At the 90th percentile HRRR's lead was clearer (AP 0.54 vs 0.33–0.43).
- So **HRRR's advantage is concentrated in the most extreme storms.** For moderate-to-heavy rain, a model using
  only cheap station data comes close to NOAA's operational model.
- The models were not re-tuned at the 75th percentile, so these numbers are, if anything, slightly conservative.

## Status

- All data files are rebuilt for v2.
- **Every model on the leaderboard is retrained on the v2 data:** the team's defaults and val-tuned `_tuned`
  versions, plus the new CV-tuned `_cv` models. All are scored on val (except `_cv`) and on test.
- `results/leaderboard.csv` includes `cv_*` columns for the `_cv` models.
- Not done: the two extra features (SFO−SAC pressure difference, day of season), probability calibration for the
  GRU, and saving per-day test predictions for the case studies.

## Reproduce

```bash
.venv/bin/python scripts/dataset_downloader.py   # hourly, Nov-Mar -> data/raw_data/ (gitignored)
.venv/bin/python scripts/ghcnd_downloader.py     # official daily rain -> data/ghcnd_daily.csv
.venv/bin/python scripts/hrrr_downloader.py      # HRRR, adds Nov/Mar days -> data/hrrr_daily.csv
.venv/bin/python scripts/data_preprocess.py      # clean table + split
.venv/bin/python scripts/data_report.py          # coverage + risk check (run from scripts/ or repo root)
.venv/bin/python crossval.py                     # leave-one-winter-out scores -> results/cv/summary.csv (~5 min)
.venv/bin/python crossval_ablation.py --models logistic_regression xgboost gru --seeds 3   # -> results/cv/ablation.csv (~15 min)
.venv/bin/python crossval.py --grid              # CV tuning -> results/cv/selected.json (~45 min)
.venv/bin/python train.py && .venv/bin/python train.py --grid   # team's models on v2 data
.venv/bin/python train.py --cv                   # final <model>_cv models on train+val
.venv/bin/python evaluate.py --split val && .venv/bin/python evaluate.py --split test
.venv/bin/python leaderboard.py
```

On macOS, XGBoost needs OpenMP: `brew install libomp`.
