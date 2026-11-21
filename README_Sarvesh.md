# Data pipeline v2: more events, official rain totals, late-evening features

Branch `Sarvesh` · Sarvesh Halbe

This branch changes the **data**, not the models. It makes three changes to the shared daily table, and each one
answers a weakness found while reviewing the original pipeline. The sections below give the reason, the evidence,
what changed, and the caveats for each.

## Summary

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

**First model check (validation only, test not touched).** We trained default models with and without these
columns. Each cutoff was tuned on validation.

| Model | Daily only | + late evening |
|---|---|---|
| Logistic regression, 1 day of history | 0.271 | **0.288** |
| Logistic regression, 5 days of history | 0.182 | **0.193** |
| GRU (default settings) | **0.309** | 0.244 |

Persistence scores CSI 0.033 and climatology 0.062 on the same 1,057 validation days. With 31 validation events
and a single seed for the GRU, **these differences are within noise**. The features are physically motivated and
carry a stronger signal, but whether the models make use of them still needs proper evaluation: several seeds,
cross-validation over winters, and feature selection. That is the next step.

## Split

The original split was 70/15/15 of whatever winters survived. Recovering 1993–97 would have changed the boundaries,
so this branch **fixes them** instead: `VAL_FIRST_WINTER = 2012` and `TEST_FIRST_WINTER = 2019`. The validation and
test winters are therefore the same ones the team has been using. The proportions are now 74 / 13 / 13 by winter.
The test winters are unchanged, but their *days* and *labels* are not: Nov/Mar days were added, the threshold
moved, and the rain source changed. **Scores on `main`'s leaderboard are not directly comparable with scores from
this branch.**

## Status and what is stale

- `data/clean_daily.csv`, `data/train_data/*`, `data/test_data/test.csv`, `data/ghcnd_daily.csv` and
  `data/hrrr_daily.csv` are rebuilt.
- `weights/*/config.json`, `results/*` and `results/leaderboard.csv` **still describe models trained on `main`'s
  data**. They need retraining (`train.py`, `train.py --grid`, `evaluate.py`, `leaderboard.py`) once we decide which
  models to carry forward.

## Reproduce

```bash
.venv/bin/python scripts/dataset_downloader.py   # hourly, Nov-Mar -> data/raw_data/ (gitignored)
.venv/bin/python scripts/ghcnd_downloader.py     # official daily rain -> data/ghcnd_daily.csv
.venv/bin/python scripts/hrrr_downloader.py      # HRRR, adds Nov/Mar days -> data/hrrr_daily.csv
.venv/bin/python scripts/data_preprocess.py      # clean table + split
.venv/bin/python scripts/data_report.py          # coverage + risk check (run from scripts/ or repo root)
```

On macOS, XGBoost needs OpenMP: `brew install libomp`.
