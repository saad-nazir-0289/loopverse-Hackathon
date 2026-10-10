# Sprint 1: Next-day PM2.5 forecast

## Reproduce
```bash
pip install -r requirements.txt
python src/clean.py      # -> data/processed/daily_pm25.csv, prints the data audit
python src/forecast.py   # -> walk-forward report, outputs/*.csv, predictions.csv (checked against SPEC section 4)
python tests/test_leakage.py   # proves no future data reaches the model
python src/audit_traps.py      # evidence for the 4 handbook traps -> outputs/trap_audit.md + outputs/figures/
```
Optional: `python src/forecast.py --model ridge` forces a specific model. Runs are deterministic (fixed seeds).

---

Trap-by-trap evidence (date ranges, feature timestamps, before/after network plots, sentinel and stuck-sensor plots, accuracy vs recall table): [outputs/trap_audit.md](outputs/trap_audit.md).

## 1. Units and timezones: what everything is converted to

**Working unit: PM2.5 concentration in µg/m³. Working clock: Pakistan Standard Time (UTC+5), calendar date.**

We use PM2.5 because the submission column is `predicted_pm25` and the hazardous rule (`PM2.5 >= 165`, SPEC section 3) is a concentration threshold, not an AQI value. AQI is an index on a different scale (for example, AQI 200 ≈ 150 µg/m³), so mixing the two would make averages and the 165 cut-off meaningless.

| File | As supplied | Conversion |
|---|---|---|
| `batch_1_sensor_data.csv` | PM2.5 µg/m³, UTC timestamps at 19:00 | Add 5 h. 19:00 UTC becomes 00:00 PKT on the **next** calendar day. Units unchanged. |
| `batch_2_sensor_data.csv` | AQI (integer), PKT dates | Invert the US EPA PM2.5 AQI formula to get µg/m³. Dates unchanged. |

**Which AQI table:** we tested both EPA breakpoint tables against the overlapping readings.

| Table | AQI(batch_1 PM2.5) == batch_2 exactly | Off by ±1 (rounding) |
|---|---|---|
| **Pre-2024 EPA** (0–12.0 → 0–50, 12.1–35.4 → 51–100, 35.5–55.4 → 101–150, 55.5–150.4 → 151–200, 150.5–250.4 → 201–300, ...) | **1714 / 1744** | 29 (plus 1 reading off by 5) |
| 2024 EPA (0–9.0 → 0–50, ...) | 171 / 1744 | n/a (systematic offset) |

The pre-2024 table is used (`AQI_BREAKPOINTS` in `src/clean.py`). After inversion the two networks agree to a median of 0.41 µg/m³ (p99 0.96). The remaining gap comes from AQI being rounded to an integer.

Order of operations (per the handbook): parse time → shift to PKT → convert AQI to PM2.5 → replace -999 → handle the stuck sensor → one value per sensor-day → build lagged features.

## 2. How the two batches are combined (and what is dropped)

Once timezone and units are aligned, **the two files are the same 1800 sensor-days recorded twice**: all 15 sensors appear in both, on identical PKT dates. Nothing is thrown away blindly:

| Situation | Rows | What we do | Why |
|---|---|---|---|
| batch_1 valid | 1770 | Use batch_1 PM2.5 | Native unit at 0.1 µg/m³ precision; the inverted AQI carries about ±0.5 µg/m³ of rounding error |
| batch_1 = -999, batch_2 valid | 27 | Use inverted batch_2 AQI | The -999 gaps never overlap between files, so the second copy recovers every gap |
| Both -999 | 0 | n/a | n/a |
| **Stuck sensor S07, 2026-08-25..27** | **3** | **Set to missing** | See below |

**Why the stuck days are dropped even though there are two batches.** The second batch is not an independent sensor. It is the same faulty reading re-encoded as AQI. On those three days batch_1 says 71.8, 71.8, 71.8, and batch_2 says AQI 159, 159, 159, which is exactly AQI(71.8). So there is no clean copy to fall back on, and filling from batch_2 would reinsert the fault. The true values are unknown, so we:
- exclude those 3 days as **training targets**, so the model never learns to predict a hardware fault, and
- for **features**, forward-fill from the last good reading (max 3 days). We never interpolate, because interpolation would use future readings (leakage).

The stuck run is detected on the 0.1-precision batch_1 series. The integer AQI series has 4 runs of 3 equal values (S01, S07 ×2, S14), but 3 are coincidences of integer rounding (the underlying PM2.5 values differ). Only S07 Aug 25–27 is flat in both networks, which matches DOC-06 ("one sensor ... stuck ... for three consecutive days").

### 2.1 Further data checks (no change needed, recorded for transparency)
| Check | Result | Decision |
|---|---|---|
| Other sentinel or impossible values | Only -999. batch_1 18.9..276.9 µg/m³, batch_2 AQI 65..322; numeric dtype, clean IDs, 120 rows per sensor | none |
| Weather files | No NaN, duplicates or date gaps; plausible ranges; covers every sensor date. One odd day: 2026-10-20 temp 13.3 °C between 19.8 and 22.1 (within 3σ) | kept; weather barely affects the model |
| Holdout files | 150 rows, no duplicates/NaN, areas match the metadata | none |
| Near-constant runs beyond exact repeats | S01 2026-07-26..28 = 71.3 / 71.0 / 71.4 (3-day range 0.4, flatter than 99.9% of windows; AQI 159 ×3 in batch_2) | **kept**: not exactly constant, and DOC-06 reports a single stuck sensor (S07). Ambiguous, flagged here. |
| Cross-network disagreements | Only S03 2026-09-18: batch_1 276.9 vs batch_2 AQI 322 (→ 271.7; AQI(276.9) would be 327). All other readings agree within ±1 | batch_1 value used; AQI > 300 conversions are slightly less certain |
| One-day spikes | 61 isolated spike-and-return events, identical in both networks | kept as real; indistinguishable from faults, and the hidden answers come from the same data |

## 3. Forecasting setup

- **Horizon:** observations end **2026-10-28**. Holdout targets run 2026-10-29..11-07, so these are **1- to 10-day-ahead forecasts from 10-28**. The `forecast_origin_date` values after 10-28 fall in the hidden-answer period, so no readings exist for them.
- **Direct multi-horizon model:** one training row per (sensor, origin t, horizon h), target = PM2.5 on t+h. The model learns the residual `y − mean7(t)`, so it follows the seasonal level instead of being capped at training values.
- **Features (30), all known at the end of origin day t:**
  - PM2.5 history: lags 0–6, rolling mean 3/7/14, std/min/max 7, 7-day trend, the sensor's expanding mean and offset from the city, city mean (today and 7-day)
  - **Spike history:** `spike_rate30`, the share of the last 30 days on which the sensor jumped more than 45 µg/m³ above its previous 14-day median
  - **Calendar:** weekday of the target date (one-hot). The calendar is known in advance, so this is not future data.
  - Horizon h, and weather **on the origin day**
- **Not used by default: weather on the target day.** That is after the origin, i.e. future data. SPEC says `holdout_weather.csv` "MAY" be used, but the rule it refers to is missing, so we take the strict reading. The ablation (section 5.4) shows it makes the models worse anyway. It can be switched on with `--target-weather`.
- **Preprocessing** (imputers, scalers) sits inside each sklearn pipeline, so it is fitted on training rows only, per fold.

## 4. No future data: how it is enforced and tested

| Where the future could leak in | Guard | Proof |
|---|---|---|
| Random train/test split | Walk-forward folds only: train on target dates ≤ origin, test after it | Assertion in `walk_forward()`; `test_walk_forward_split` |
| Lag, rolling and spike features | Every window ends at the origin day; forward-fill only, never interpolate (interpolation reads the next reading) | `test_past_features_ignore_the_future`: replace **all** PM2.5 and weather after the origin with random noise and check that all 30 features are bit-identical (7 origins) |
| Sensor "typical level" | Expanding mean up to t, not a mean over the whole dataset | Same test |
| Weather after the origin | Excluded by default | Control test shows the detector **does** flag target-day weather, so it is sensitive |
| Cleaning (stuck sensor, gap fill) | Gap fill uses the same day's reading from the other network | `test_cleaning_is_causal`: cleaning data truncated at each origin gives identical values to cleaning the full file |
| Imputer / scaler statistics | Fitted inside each fold's pipeline on training rows | Code structure (`_linear`, `_imputed` in `src/forecast.py`) |
| Picking the best model on the test data | Choose on folds 1–5, report fold 6 untouched | Section 5.2 |

Run with `python tests/test_leakage.py`. All 4 tests pass.

## 5. Evaluation

### 5.1 Protocol and metrics
Walk-forward, 6 folds whose origins are 7 days apart (09-13 … 10-18). Each fold forecasts all 15 sensors for origin+1..origin+10 (150 rows, the same shape as the holdout).

| Metric | What it answers |
|---|---|
| MAE | Typical error in µg/m³ |
| RMSE | Punishes large misses more |
| MAPE % | Error relative to the level |
| Bias | Systematic over (+) or under (−) forecasting |
| R² | Share of variance explained |
| MAE_high, bias_high | Error on days with true PM2.5 ≥ 150 (see 5.6 for how to read this) |
| Skill vs mean7 | % MAE improvement over the 7-day-average baseline |
| Fold std | Stability of MAE across folds |
| Recall, precision, F1, false alarms, FA rate | Hazardous alarm quality |
| PR-AUC | Threshold-free: does a higher forecast rank hazardous days higher? Chance = 0.032 |

### 5.2 Model selection rule
**Choose by mean rank across MAE, RMSE, MAE_high, |bias| and PR-AUC on folds 1–5 (750 rows).** Fold 6 (origin 10-18, targets 10-19..10-28, 150 rows) is never used for choosing and serves as an untouched confirmation. The rule was fixed before looking at results.

### 5.3 Results (submitted configuration: 30 features)

Selection folds, sorted by mean rank:

| Model | Rank | MAE | RMSE | MAPE % | Bias | R² | Skill % | Fold std | PR-AUC |
|---|---|---|---|---|---|---|---|---|---|
| **Ridge α=300 (submitted)** | **3.6** | 13.16 | 21.71 | 11.7 | −1.64 | 0.433 | 16.2 | 1.29 | 0.091 |
| Ridge α=10 | 4.2 | 13.17 | 21.74 | 11.7 | −1.57 | 0.432 | 16.2 | 1.22 | 0.090 |
| Ridge + XGBoost | 4.2 | 13.20 | 21.76 | 11.6 | −1.84 | 0.430 | 16.0 | 1.33 | 0.090 |
| Ridge + LightGBM | 5.2 | 13.27 | 21.81 | 11.7 | −2.01 | 0.428 | 15.6 | 1.37 | 0.092 |
| Huber | 5.4 | **12.96** | **21.65** | **11.3** | −2.78 | **0.436** | **17.5** | 1.18 | **0.093** |
| Ridge + ExtraTrees | 6.2 | 13.28 | 21.83 | 11.7 | −2.01 | 0.427 | 15.5 | 1.27 | 0.070 |
| XGBoost | 6.6 | 13.71 | 22.13 | 12.0 | −2.10 | 0.411 | 12.8 | 1.47 | 0.062 |
| ElasticNet | 6.8 | 13.47 | 21.91 | 11.9 | −1.73 | 0.422 | 14.3 | 1.42 | 0.089 |
| RandomForest | 7.6 | 13.84 | 22.17 | 12.1 | −2.34 | 0.409 | 11.9 | 1.37 | 0.060 |
| ExtraTrees | 9.0 | 13.79 | 22.22 | 12.1 | −2.45 | 0.406 | 12.2 | 1.31 | 0.054 |
| HistGradientBoosting | 9.4 | 14.08 | 22.48 | 12.3 | −2.34 | 0.392 | 10.4 | 1.35 | 0.052 |
| LightGBM | 9.8 | 13.87 | 22.30 | 12.1 | −2.46 | 0.402 | 11.7 | 1.52 | 0.060 |
| *7-day mean (baseline)* | 13.0 | 15.71 | 24.05 | 13.6 | −4.11 | 0.304 | 0.0 | 1.77 | 0.043 |
| *Persistence (baseline)* | 14.0 | 20.77 | 31.99 | 18.4 | −4.93 | −0.231 | −32.2 | 5.81 | 0.040 |

Untouched confirmation fold (origin 10-18):

| Model | MAE | RMSE | Bias | PR-AUC |
|---|---|---|---|---|
| **Ridge α=300 (submitted)** | 13.14 | 21.28 | −1.68 | 0.064 |
| Ridge α=10 | 12.99 | 21.17 | −1.87 | 0.064 |
| Huber | 12.93 | 21.25 | −2.71 | 0.050 |
| Ridge + XGBoost | 13.15 | 21.43 | −0.59 | 0.057 |
| XGBoost | 13.90 | 22.00 | +0.69 | 0.049 |
| *7-day mean* | 15.72 | 24.29 | −2.88 | 0.037 |

### 5.4 Ablations: which features earn their place
Best model in each configuration, same folds:

| Configuration | Best MAE | Rank winner MAE | Best PR-AUC | Confirm MAE (rank winner) |
|---|---|---|---|---|
| History + origin weather only (first version) | 14.35 | 14.48 | 0.059 | 14.58 |
| + spike history | 14.30 | 14.49 | 0.053 | 14.31 |
| + weekday | 13.09 | 13.28 | 0.093 | 13.43 |
| **+ weekday + spike history (submitted)** | **12.96** | **13.16** | **0.093** | **13.14** |
| first version + **target-day weather** | 14.51 | 14.64 | n/a | 14.38 |

- **Weekday is the biggest single gain:** −1.3 MAE (−9%), and PR-AUC nearly doubles. The data has a weekly cycle: Fri +7, Thu/Sat +3–4, Mon −8, Tue −7 µg/m³ around the recent level, measured with spikes excluded. That's consistent with a traffic or work-week pattern. Holdout forecasts reproduce it: Friday mean 140, Monday 122.
- **Spike history** adds a little on top of weekday.
- **Target-day weather made every tree model worse** (XGBoost +1.2 MAE) and left linear models unchanged, so dropping it is both the strict and the better choice.

### 5.5 What the numbers say
1. **Every model beats both baselines on every error metric**, on the selection folds and the untouched fold. The submitted model is 16% better than the 7-day average in MAE, removes most of its −4 bias, and raises R² from 0.30 to 0.43.
2. **The top 6 models are statistically tied** (MAE 12.96–13.28 vs fold std ≈ 1.3). Ridge α=300 wins the pre-declared multi-metric rank. Huber has the lowest MAE/RMSE but the largest negative bias. Either is defensible; we keep the pre-declared rule.
3. Linear models beat boosted trees here: 120 days × 15 sensors is little data, and the trees fit spike noise.

### 5.6 "High days are under-forecast by about 60": what it means
MAE_high is about 60 with bias_high = −MAE_high for every model, which looks alarming. It is a **selection effect, not a model bias**:

| Grouped by actual PM2.5 | Bias | | Grouped by predicted PM2.5 | Bias |
|---|---|---|---|---|
| ≤ 90 | +17.7 | | ≤ 110 | +1.5 |
| 90–120 | +4.6 | | 110–125 | −1.4 |
| 120–150 | −4.5 | | 125–140 | −0.4 |
| > 150 | −55.8 | | > 140 | −1.0 |

(First-version model. The pattern is the same for the submitted one.)

- Grouped by what the model **said**, the bias is about 0: the forecasts are calibrated.
- Grouped by what **happened**, high days look under-forecast and low days over-forecast. Days that turned out ≥ 150 are mostly days where an unpredictable spike landed on top of the level. Any honest forecast of the expected value shows this.
- The spikes are close to unpredictable: residual autocorrelation at lag 1 is 0.01, the day after a spike returns to normal (+2.5 on average), and the residual distribution is heavily right-skewed (1st percentile −33, 99th +114).

## 6. Hazardous alarm (decision point)

**Team decision: `hazardous = predicted_pm25 >= 165`, the SPEC rule as written. 0 of 150 holdout rows are flagged.** Forecasts range 103–164. The alternatives were tested (section 8). A lower alarm threshold (e.g. 145–150) has a better *expected* recall/F1, but it breaks SPEC's "apply to your predicted PM2.5" and its gain rests on few validation rows. We chose literal SPEC compliance and disclose the 0-recall consequence here and in the recommendation.

**Why it raises no alarms.** The point forecast is an *expected value*. Hazardous days are isolated spikes about 100 µg/m³ above a level of about 130 (55 of 1800 sensor-days, every month, 14 of 15 sensors). An expected value only reaches 165 when the level itself does, which has not happened yet. This is the handbook's misleading-accuracy trap: "never hazardous" scores about 97% accuracy with 0 recall.

**Is there alarm skill?** Some. In the selection folds the true hazardous rate is 1.3% in the bottom half of forecasts and about 5% in the top half (≈ 4× lift), and PR-AUC is 0.09 vs a chance level of 0.03. Calibrating the alarm threshold (Ridge α=300):

| Alarm when predicted ≥ | Selection folds (24 hazardous): flagged / caught / false alarms | Confirmation fold (4 hazardous): flagged / caught / false alarms | Holdout rows flagged (of 150) |
|---|---|---|---|
| **165 (SPEC)** | 0 / 0 / 0 | 0 / 0 / 0 | **0** |
| 150 | 1 / 1 / 0 | 7 / 0 / 7 | 23 |
| 145 | 7 / 1 / 6 | 14 / 1 / 13 | 43 |
| 140 | 24 / 1 / 23 | 30 / 2 / 28 | 60 |
| 135 | 60 / 3 / 57 | 44 / 2 / 42 | 72 |

Precision stays at 5–9% at every lower threshold: most alarms would be false. Hazardous counts are small (24 and 4), so these numbers are noisy. The holdout flags far more rows than validation because the level is still rising into November.

Switch with `python src/forecast.py --alarm-threshold 145`. The team must choose between SPEC consistency with zero recall, and a calibrated alarm with some recall and many false alarms. Record the choice and the reason in the recommendation document.

## 7. Known limitations
- 120 days of history (Jul–Oct) do not include the November peak smog season, so any rise beyond the October trend is not learned.
- Individual hazardous spikes are not forecastable from the supplied inputs. The Sprint 2 assistant must not present "not hazardous" as a guarantee of safe air.
- Multi-step error grows only slowly with horizon (see MAE by horizon in `outputs/log_final.txt`), because most of the signal is the sensor's level and the weekday.

## 8. Method bake-off (`python src/experiments.py`)

Protocol fixed before running: 11 weekly walk-forward folds (origins 08-09 … 10-18). Folds 1–3 are warm-up and 4–11 are scored (1200 rows, 32 hazardous). Every alarm threshold is tuned **nested**: the threshold applied to fold k is chosen only on earlier folds.

### 8.1 Point-forecast variants (scored folds)
| Variant | MAE | RMSE | Bias | Folds beating current |
|---|---|---|---|---|
| **Current: Ridge α=300 on mean7** | **12.90** | 21.06 | −1.23 | n/a |
| + robust features (median7/14, days since spike) | 12.94 | 21.05 | −0.93 | 3 / 8 |
| Ridge on median7, spikes clipped from target | 13.01 | 21.19 | −1.88 | 3 / 8 |
| Ridge on median7 + robust features | 13.14 | 21.22 | −0.95 | 3 / 8 |
| Blend Ridge + LightGBM (Huber loss) | 13.17 | 21.36 | −2.40 | 2 / 8 |
| LightGBM Huber loss | 13.50 | 21.73 | −3.85 | 1 / 8 |
| Separate model per horizon | 13.73 | 21.69 | −0.56 | 1 / 8 |

No variant beats the current model, so the current model is kept.

### 8.2 Hazardous-alarm methods (scored folds; chance PR-AUC = 0.027)
| Method | PR-AUC | Nested F1 threshold: caught / false alarms | Nested F2 threshold: caught / false alarms |
|---|---|---|---|
| **Point forecast, current model** | **0.051** | 15 / 457 | 18 / 536 |
| Point forecast, median7 model | 0.046 | 12 / 436 | 19 / 522 |
| Residual probability by sensor spikiness | 0.044 | 9 / 291 | 12 / 317 |
| Quantile q0.90 (LightGBM) | 0.038 | 14 / 283 | 19 / 474 |
| Quantile q0.95 (LightGBM) | 0.031 | 13 / 370 | 17 / 440 |
| Logistic regression, class-balanced | 0.026 | 2 / 79 | 4 / 153 |
| Sensor spike rate only (naive) | 0.025 | 17 / 623 | 17 / 623 |
| LightGBM classifier, class-balanced | 0.023 | 10 / 484 | 11 / 492 |

Dedicated classifiers do **worse than chance**: with 32 positives they overfit. The best ranking of hazardous days comes from the point forecast itself, at about 2× chance.

### 8.3 Is there any hidden structure in the spikes? (spike = > 45 above the previous 14-day median; 62 events)
| Possible pattern | Result |
|---|---|
| Spikes on certain weekdays | No (χ² p = 0.66). The weekly cycle is in the level, not in the spikes. |
| Some sensors spike more | No (χ² p = 0.69). All sensors are at about 3.4% of days. |
| Spike spreads to other sensors the next day | No (3.5% vs 3.4%) |
| Weather 1–2 days before | No (all p > 0.06) |
| Rising toward smog season | No (Jul 3.4%, Aug 4.9%, Sep 2.9%, Oct 2.4%) |

Conclusion: the data behaves like **seasonal level + weekly cycle + random spikes at a constant ≈ 3.4% rate**. The model captures the first two. The spikes are not predictable from the inputs, so the only lever is how the alarm turns the forecast into a yes/no.

### 8.4 Expected holdout outcome per alarm policy
P(hazardous | forecast) was calibrated on the scored folds (isotonic): 0.017 at a forecast of 100, about 0.04 at 120–140, 0.06 at 145, and 0.17 at ≥ 150. The ≥ 150 estimate rests on few validation rows, so it is uncertain. Expected hazardous rows in the holdout: about 10.6 of 150.

| Policy | Flagged | Expected caught | Expected false alarms | Expected recall | Expected precision | Expected F1 |
|---|---|---|---|---|---|---|
| SPEC: alarm ≥ 165 | 0 | 0 | 0 | 0.00 | n/a | 0.00 |
| alarm ≥ 155 | 9 | 1.5 | 7.5 | 0.14 | 0.17 | 0.15 |
| alarm ≥ 150 | 23 | 3.8 | 19.2 | 0.36 | 0.17 | 0.23 |
| alarm ≥ 145 | 43 | 6.5 | 36.5 | 0.62 | 0.15 | **0.24** |
| alarm ≥ 140 | 60 | 7.4 | 52.6 | 0.70 | 0.12 | 0.21 |
| flag all 150 | 150 | 10.6 | 139.4 | 1.00 | 0.07 | 0.13 |

### 8.5 Decision
- **Point forecast:** keep the current model (Ridge α=300 on mean7, 30 features). No variant beat it.
- **Alarm:** SPEC rule `predicted >= 165` (team decision). Known consequence: expected recall 0 on the holdout. Under F1, `--alarm-threshold 145` or `150` would score better in expectation (table 8.4). Recorded so the trade-off is visible to judges.

## 9. Handbook "stronger version" checklist

| Step | Stronger version (handbook) | Where it is done |
|---|---|---|
| Inspect | Coverage heatmap and same-area network comparison | `src/audit_traps.py`: `outputs/figures/trap3_coverage_heatmap.png`, `trap2_networks_before_after.png` |
| Standardize | Automated unit and timezone assertions, raw columns retained | `src/clean.py` asserts: batch_1 at 19:00 UTC = 00:00 PKT; batch_2 integer AQI on 0..500; batch_1 plausible PM2.5; cross-network median gap < 1 µg/m³ (actual 0.41); the same comparison **without** the +5 h shift is > 5× worse (13.4), so the shift is proven necessary. Raw `pm25_b1`, `aqi_b2`, `pm25_b2` kept in `daily_pm25.csv`. |
| Clean | Quality flags, coverage threshold, documented imputation | `pm25_source` and `stuck` flags; asserted coverage ≥ 90% per sensor (minimum 97.5%, S07); imputation documented in section 2 |
| Aggregate | Minimum number of valid readings per day | `n_valid` = valid, non-faulty readings per sensor-day (2 networks): 1741 days have 2, 56 have 1, 3 have 0 (stuck). Rule: ≥ 1, else missing. |
| Features | Several shifted lags, rolling spread, weather lags, area and network | Lags 0–6, std/min/max 7 in the model. Weather lags (t-1, t-2, 3-day mean) and area/network (one-hot area, AQI-network flag) were **built and tested**, and rejected by the tuning-fold rule (table below). |
| Model | Regularized regression, trees or boosting, validated ensemble | Ridge (submitted), plus Huber, ElasticNet, RandomForest, ExtraTrees, XGBoost, LightGBM, HGB, blends and LSTM, all walk-forward validated |
| Validate | Walk-forward across multiple cutoffs | 6 folds (model selection) and 11 folds (experiments) |
| Alarm | Calibrate the threshold using hazardous recall and false alarms | Calibrated in section 8.4; team chose the SPEC threshold (section 6) |

### 9.1 Feature candidates (11 folds; tune = folds 4–7, test = folds 8–11)
| Feature set | Tune MAE | Test MAE | Adopted |
|---|---|---|---|
| **Current (30 features)** | **13.07** | 12.74 | yes |
| + weather lags | 13.09 | 12.77 | no: worse on both |
| + area & network | 13.21 | 12.64 | no: worse on tune folds (the selection rule) |
| + both | 13.21 | 12.68 | no |

Sensor identity is already captured by `sensor_level` and `sensor_offset`, so area one-hots add little.

### 9.2 Random Forest, XGBoost, LSTM with honest tuning (`python src/experiments_ml.py`)
Hyperparameters are chosen on the tune folds 4–7 only. The chosen setting is then scored on the untouched test folds 8–11 (600 rows each).

| Family | Settings tried | Chosen (on tune folds) | Tune MAE | Test MAE | Test RMSE | Test folds beating Ridge |
|---|---|---|---|---|---|---|
| **Ridge α=300 (submitted)** | n/a | n/a | **13.07** | **12.74** | **20.43** | n/a |
| XGBoost | 13 (depth 2/3/5, 200/600 trees, min child 20/80, squared and pseudo-Huber loss) | depth=2, 600 trees, child=80 | 13.20 | 13.12 | 20.68 | 1 / 4 |
| RandomForest | 6 (min leaf 10/30/80, max features 0.3/0.6) | leaf=80, features=0.3 | 13.56 | 13.26 | 20.70 | 0 / 4 |
| LSTM (PyTorch; 28-day window of PM2.5, weather and weekday, sensor embedding, target weekdays; 10 horizons at once; Huber loss; 3-seed average) | 4 (hidden 16/32 × 40/120 epochs) | hidden=32, 40 epochs | 14.97 | 14.37 | 22.29 | 0 / 4 |

**Verdict: Ridge stays.** With tuning done honestly on separate folds:
- XGBoost comes closest (+0.38 MAE) and beats Ridge on 1 of 4 test folds.
- Random Forest is +0.52 worse.
- LSTM is +1.63 worse and loses every fold. With about 1,300 training sequences, it cannot learn more than a regularized linear model on well-engineered lags, weekday and level features. More epochs made it worse, which points to overfitting.

Full grids: `outputs/experiments_ml_all_configs_*.csv`.

## 10. Spike prediction (can the model find real spikes?)

Hazardous days are almost all **spikes** (47 of 55: more than 45 µg/m³ above the sensor's recent median). Two experiments test whether any model can time them, with the threshold kept at 165.

**10.1 Spike detectors** (`python src/experiments_spike_detectors.py`; 11 walk-forward folds, scored on folds 4–11: 1200 rows, 37 spikes; ROC-AUC 0.5 = coin flip)

| Detector (handbook "simple models" and feature families) | ROC-AUC | Fold range | Top 10% flagged: spikes caught (chance ≈ 3.7) |
|---|---|---|---|
| Warm target day (temperature, from holdout weather) | **0.65** | 0.51–0.75 | **7** |
| Boosting, + target-day weather | 0.54 | 0.43–0.62 | 3 |
| Random forest, + target-day weather | 0.54 | 0.38–0.77 | 6 |
| Random forest, all past features (lags, spread, weather lags, area, network, spike history) | 0.51 | 0.32–0.78 | 2 |
| Logistic, + target-day weather | 0.51 | 0.28–0.67 | 5 |
| Persistence (spiked on the origin day) | 0.49 | n/a | n/a |
| Boosting, all past features | 0.48 | 0.28–0.80 | 3 |
| Recent average level | 0.47 | 0.23–0.70 | 2 |
| Rolling max / spread | 0.45–0.47 | 0.16–0.65 | 1 |
| Logistic, all past features | 0.46 | 0.28–0.64 | 2 |
| Days since last spike / sensor spike rate | 0.45 | 0.22–0.78 | 2–7 |

**10.2 Spike-aware forecast** (`python src/experiments_spikes.py`): classifier + "level + typical spike size" on flagged rows, so that SPEC's 165 rule produces the alarm. Best variant: 2 of 32 hazardous days caught for 33 false spikes, and MAE rose from 12.9 to 15.3. Past-only variants scored exactly at chance (spike PR-AUC 0.031 = base rate).

**10.3 Pattern search** (found on Jul–Sep, checked on Oct): per-sensor periodic schedules (1 caught vs 1.1 by chance), calendar cycles (all p > 0.18), spikes spreading between sensors (0 caught), calm/humid weather (no effect). The 3 *level-driven* hazardous days (no spike: S02 Oct 16, S03 and S08 Oct 17) show that rising November levels can produce hazardous days without spikes. A nested bias correction of the level forecast did not flag any of them in validation.

**Conclusion.** With past data, every model from the handbook ranks spike days at chance level. The only signal is target-day temperature, and it is weak (about 2× chance). Inventing spikes would raise the forecast error and fill the alarm with false positives, so `predictions.csv` keeps the level forecast and the SPEC 165 rule. Spikes are disclosed as the main limitation (RECOMMENDATION.md).

## 11. Two-part (hurdle) spike model (`python src/experiments_hurdle.py`)
Stage 1 classifier P(spike): ridge-style L2 logistic or gradient boosting, each with no imbalance handling, class weights, or 5:1 undersampling. Stage 2 spike size: ridge regression on spike rows, or a constant median. Combined *hard* (level + size when P ≥ t, with t tuned on earlier folds only) or *soft* (level + P × size). Scored on folds 4–11: 1200 rows, 32 hazardous, 37 spikes (base rate 3.1%).

| Variant (best of each kind) | Spike ROC-AUC | Mean P (true rate 0.031) | Flagged | Caught | False alarms | MAE |
|---|---|---|---|---|---|---|
| Level only (submitted) | n/a | n/a | 0 | 0/32 | 0 | **12.9** |
| Ridge-logistic, no weighting, 30+ past features | 0.39 | 0.027 | 14 | 1/32 | 13 | 13.9 |
| Ridge-logistic, class weights, past+temp | 0.50 | 0.339 | 124 | 5/32 | 119 | 21.3 |
| Boosting, class weights, past+temp | 0.48 | 0.223 | 424 | 9/32 | 415 | 43.3 |
| **Lean ridge-logistic (temp, Δtemp, level), class weights** | **0.63** | 0.351 | 161 | **9/32** | 152 | 23.5 |
| Lean ridge-logistic, no weighting | 0.63 | 0.025 | 73 | 5/32 | 68 | 17.6 |

Stage 2 (size, given a spike): ridge size MAE 26.5 vs **constant median 24.1** (true size 90 ± 29 µg/m³). The size is stable, but the timing is not.

**Findings**
1. **Stage 1 is the bottleneck.** With past-only data the classifier is at or below chance. Only target-day temperature adds signal (≈ 0.63).
2. **Fewer features beat many:** 3 features give 0.63 ROC-AUC vs 0.39–0.50 for 30+ features. With about 50 training spikes, a large model learns noise.
3. **Imbalance handling breaks calibration.** Class weights or undersampling raise the average predicted P from 0.03 to 0.22–0.37, so the *soft* expected value overshoots (MAE 26–39). Weighting may only be used for *ranking*, with the threshold tuned separately.
4. **Soft combination never alarms.** A calibrated P (≈ 0.03) × size 90 adds about 3 µg/m³, so it never crosses 165; an alarm needs the *hard* decision.
5. **Best trade-off: 9 of 32 hazardous days caught for 152 false alarms** (precision 5.6%, about 2× chance) and MAE 12.9 → 23.5. The submission keeps the level forecast and the SPEC 165 rule.
