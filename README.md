# Forecasting Heavy Winter Rainfall in Sacramento

ECS 289G · Group 5 — Lucas Hornung, Sarvesh Halbe, Steven Trujillo, Vikhas Somasundaram Giri, Willian Klippel-Huber

**Due:** 4-page report and recorded presentation Mon Nov 30. Live presentations Dec 1 and 3.

## Goal

Predict heavy winter rain in Sacramento for the next 24 hours using only low-cost local
weather-station data, and test whether a lightweight ML model beats persistence ("the next
24 hours will look like the last 24 hours").

| | |
|---|---|
| **Input** | Previous 1–5 days of station readings: pressure, dew point, wind direction, rainfall (Sacramento + nearby upwind stations) |
| **Output** | (1) Heavy-rain event in the next 24 h: yes/no. (2) 24-hour rainfall total in mm |
| **Data** | Hourly NOAA observations (ISD/LCD, e.g. via `meteostat`), Dec–Feb, as many winters as records allow |
| **Heavy rain** | 90th percentile of wet-day totals in the training period |
| **Split** | Whole winters, chronological: 70% train / 15% validation / 15% test |
| **Baselines** | Persistence (required), climatology, logistic regression; HRRR as optional professional reference |
| **Metrics** | CSI (main), precision, hit rate, MAE; paired block-bootstrap 90% CI for the CSI difference |

## Pipeline

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/dataset_downloader.py  # raw hourly station data -> data/raw_data/ (gitignored)
.venv/bin/python scripts/data_report.py         # coverage + heavy-rain risk check
.venv/bin/python scripts/data_preprocess.py     # clean daily table + train/val/test split
.venv/bin/python scripts/hrrr_downloader.py     # HRRR forecasts (already in data/hrrr_daily.csv)
.venv/bin/python train.py                       # fit models -> weights/<tag>/, results/<tag>/train.json
.venv/bin/python train.py --grid                # grid search on val -> weights/<tag>_tuned/
.venv/bin/python evaluate.py --split val        # scores -> results/<tag>/metrics_<split>.json
.venv/bin/python leaderboard.py                 # all configs, times, scores -> results/leaderboard.csv
```

## Roles

| # | Role | Owner | Owns |
|---|------|-------|------|
| 1 | Data lead | TBD | Station data download, cleaning, 24-hour rainfall table |
| 2 | Modeling lead | TBD | Features and ML models |
| 3 | Evaluation lead | TBD | Split, baselines, scores, confidence intervals, result figures |
| 4 | Meteorology lead | Willian | Stations, physical features, heavy-rain definition, storm case studies |
| 5 | Science & report lead | TBD | Literature review, report assembly, presentation story, submission |

## To Do

### All
- [ ] Risk check: only 2.2 heavy-rain days/winter at the 90th percentile (plan needs ~3). Decide: widen to Nov–Mar or add the 75th percentile (~5/winter)

### 3. Evaluation lead
- [ ] Save test-winter predictions per day (needed for storm case studies)
- [ ] Produce main result figures

### 4. Meteorology lead — case studies Mon Nov 23
- [ ] Justify stations: KSAC + KSFO required; KSCK, KSMX, KSJC optional (blank on days they are incomplete, mostly before ~2000). Downloaded but unused: KSMF, KOAK, KEKA (almost no data)
- [ ] Review the physical features in `scripts/data_preprocess.py` (pressure, pressure change and range, wind u/v and speed, dew point, temperature and its range, upstream rain)
- [ ] Confirm heavy-rain definition (wet day ≥ 1 mm) and day window (PST midnight–midnight) in `scripts/data_preprocess.py`
- [ ] After results: case studies of 2–3 storms (caught and missed) explaining model success/failure

### 5. Science & report lead — lit review/skeleton Wed Nov 18, full draft Wed Nov 25
- [ ] Literature review: ML rainfall nowcasting, California atmospheric rivers, low-cost forecasting for rural/developing regions
- [ ] Set up shared report doc with required sections; enforce page limits
- [ ] Draft intro, data description, conclusion; edit report into one voice
- [ ] Outline presentation, coordinate recording, collect GenAI acknowledgments, submit

## Timeline

| Dates | Milestone | Lead |
|-------|-----------|------|
| Nov 18–20 | Lit review and report skeleton ready | #5 |
| Nov 21–23 | Figures; storm case studies | #3, #4 |
| Nov 24–25 | Full report draft (before Thanksgiving) | All |
| Nov 26–28 | Light edits, slides, recording | #5 + all |
| Nov 29 | Final check: page limit, PDF name, names/emails, code link | #5 |
| Nov 30 | Submit report and recording | #5 |

## Report (4 pages, excl. references and contribution statements)

| Section | Length | Writer | Reviewer |
|---------|--------|--------|----------|
| Introduction & literature review | ¾ p | #5 | #4 |
| Data & model description | ¼ p | #1 + #4 | #5 |
| Method & results | 2 p | #2 + #3 | #4 |
| Conclusion & discussion (incl. case studies) | 1 p | #4 + #5 | #3 |
| Individual contribution statements | — | Everyone | — |
| GenAI acknowledgment | — | #5 collects | — |

Submission checklist:
- [ ] 1 column, single spaced, 11 pt
- [ ] One PDF with "Group5" in the file name
- [ ] All names and student emails
- [ ] 4–5 captioned figures/tables referenced in the text
- [ ] Link to this GitHub repo

## Presentation (~1–2 min per segment, rehearse once with a timer)

1. Problem and motivation — #5
2. Stations and why these features — #4
3. Data and preprocessing — #1
4. Models — #2
5. Results against baselines — #3
6. Storm case studies — #4
7. Conclusion and limitations — #5

## Ground Rules

- One shared repo, one shared data file and split — everyone trains and tests on identical data
- Check-ins after Tuesday and Thursday class, plus the group chat
- Keep brief notes on your own work for contribution statements
- Raise blockers early in the group chat
