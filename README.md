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

## Roles

| # | Role | Owner | Owns |
|---|------|-------|------|
| 1 | Data lead | TBD | Station data download, cleaning, 24-hour rainfall table |
| 2 | Modeling lead | TBD | Features and ML models |
| 3 | Evaluation lead | TBD | Split, baselines, scores, confidence intervals, result figures |
| 4 | Meteorology lead | Willian | Stations, physical features, heavy-rain definition, storm case studies |
| 5 | Science & report lead | TBD | Literature review, report assembly, presentation story, submission |

## To Do

### 1. Data lead — due Mon Nov 16
- [ ] Download hourly NOAA observations for Sacramento and the chosen nearby stations
- [ ] Check coverage: missing hours, gaps, station changes, "trace" rainfall values
- [ ] Build one clean table: one row per day, 24-hour rain totals, fixed day window (agreed with #4)
- [ ] Count heavy-rain events per winter and share with the group (risk check)
- [ ] Deliver clean shared CSV + event count

### 2. Modeling lead — due Fri Nov 20
- [ ] Build input features from the previous 1–5 days using #4's feature list
- [ ] Train logistic regression first, then random forest or gradient boosting
- [ ] Tune on validation winters only — never touch test winters while tuning
- [ ] Record training and prediction runtime (supports the low-cost pitch)
- [ ] Deliver trained models + test-winter predictions

### 3. Evaluation lead — split/baselines Tue Nov 17, results Sat Nov 21
- [ ] Fix and share the chronological 70/15/15 split by whole winters
- [ ] Implement baselines: persistence, climatology, HRRR if time allows
- [ ] Write one shared `evaluate()`: CSI, precision, hit rate, MAE, paired block-bootstrap 90% CI on CSI difference
- [ ] Produce main result figures and tables

### 4. Meteorology lead — stations/features Sun Nov 15, case studies Mon Nov 23
- [ ] Choose nearby/upwind stations (e.g. Sacramento Executive, Sacramento International, SFO, Oakland) and justify
- [ ] Propose physical features: 24-h pressure change, wind direction, dew point, upstream rainfall
- [ ] Agree heavy-rain definition and 24-hour day window with #1
- [ ] After results: case studies of 2–3 storms (caught and missed) explaining model success/failure

### 5. Science & report lead — lit review/skeleton Wed Nov 18, full draft Wed Nov 25
- [ ] Literature review: ML rainfall nowcasting, California atmospheric rivers, low-cost forecasting for rural/developing regions
- [ ] Set up shared report doc with required sections; enforce page limits
- [ ] Draft intro, data description, conclusion; edit report into one voice
- [ ] Outline presentation, coordinate recording, collect GenAI acknowledgments, submit

## Timeline

| Dates | Milestone | Lead |
|-------|-----------|------|
| Nov 13–15 | Kickoff; stations and features chosen; data download starts | All, #4, #1 |
| Nov 16–17 | Clean data shared; event count checked; split and baselines locked | #1, #3 |
| Nov 18–20 | Models trained; lit review and report skeleton ready | #2, #5 |
| Nov 21–23 | Results with CIs; figures; storm case studies | #3, #4 |
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
- **Risk check Mon Nov 16:** if fewer than ~3 heavy-rain days per winter, decide whether to widen to Nov–Mar or add a lower threshold (e.g. 75th percentile)
- Keep brief notes on your own work for contribution statements
- Raise blockers early in the group chat
