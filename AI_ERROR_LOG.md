# AI Error Log

Concrete mistakes an AI tool (Claude Code) made while building Sprint 1, and how they were caught and fixed. Pick the strongest 3 for the final submission (suggested: 4 future weather, 8 missed weekly cycle, 1 useless alarm). Add Sprint 2 entries as they happen.

## Entry 1: model "beat the baseline" but the alarm was useless
- **Context + wrong output:** The first Ridge model was reported as the winner because its MAE (14.6) beat the 7-day mean (15.7). The AI's summary focused on MAE and did not flag the alarm column.
- **Risk:** Hazardous recall was 0. Every truly hazardous day in validation was missed. Overall alarm accuracy still read about 97% because hazardous days are only 3% of rows. This is exactly the handbook's misleading-score trap, and it is the most safety-relevant output of the system.
- **Detection:** The walk-forward report prints recall, precision and false alarms next to MAE. All models showed recall 0.0 against 28 true hazardous rows.
- **Correction + verification:** Analysed the hazardous days (isolated single-sensor spikes about 100 µg/m³ above the level). Explained why an expected-value forecast never reaches 165 even with skill. After adding weekday features (Entry 8), PR-AUC rose to 0.09 vs a chance level of 0.03, so the alarm has some skill. SPRINT1.md section 6 lays out the threshold trade-off (recall vs false alarms, validated on the selection and confirmation folds) so the team makes an explicit, documented choice instead of an unexamined 0-recall default.

## Entry 2: stuck-sensor detector found 4 faults instead of 1
- **Context + wrong output:** The first audit scanned both files for 3 or more identical consecutive readings and reported flat lines at S01 (Jul 26–28), S07 (Aug 25–27), S07 (Sep 18–20) and S14 (Jul 13–15).
- **Risk:** Excluding 9 extra valid sensor-days would discard real high readings and bias the training data. DOC-06 states there is exactly one stuck sensor for three days.
- **Detection:** Compared each run with the other network. Only S07 Aug 25–27 was flat in batch_1 too (71.8 ×3). The other runs were integer AQI values that repeat by chance, while the underlying 0.1-precision PM2.5 values differed.
- **Correction + verification:** Stuck detection now runs only on the precise batch_1 series (`flag_stuck` in `src/clean.py`). The audit prints exactly one fault, `stuck: S07 2026-08-25..2026-08-27 value=71.8`, which matches DOC-06.

## Entry 3: model selection on the same folds that were reported
- **Context + wrong output:** After adding XGBoost, LightGBM, RandomForest, ExtraTrees, Huber, ElasticNet and blends (14 candidates), the first version picked the lowest MAE over all 6 folds and reported that same number as the expected score.
- **Risk:** Picking the best of 14 on the reporting data overstates the winner's accuracy (selection bias). That is a form of leakage from the evaluation set into the choice of model.
- **Detection:** Reviewed the protocol. The reported score and the selection score came from the same 900 rows, and the top linear models differed by less than the fold-to-fold std (≈ 1.2 MAE).
- **Correction + verification:** `report()` in `src/forecast.py` now selects on the first 5 folds only and reports the latest fold (origin 10-18) as an untouched confirmation. The final model (Ridge α=300) scores 13.16 MAE on selection and 13.14 on confirmation, vs the 7-day mean at 15.71 / 15.72, so the gain holds out of sample.

## Entry 4: weather from the target day (the future) used as a feature
- **Context + wrong output:** The AI built features from weather **on the target date** (`w_*`, `dw_*`), reasoning that SPEC supplies `holdout_weather.csv` and says it "MAY" be used. The SPEC sentence points to a rule ("see rule below") that does not exist, and the AI did not flag that ambiguity. A team member challenged it: "we can't use future data".
- **Risk:** Target-day weather is not known at the forecast origin. If the judges read the rule strictly, this is future leakage. The model would also depend on inputs that real deployment would only have as an uncertain forecast.
- **Detection:** Review of the feature list against the origin date, then an ablation: the same 14 models with and without target-day weather.
- **Correction + verification:** Target-day weather is now off by default (`PAST_FEATURES` in `src/features.py`; opt in with `--target-weather`). `tests/test_leakage.py` randomises every PM2.5 and weather value after the origin and checks that all default features (now 30) are unchanged; a control test confirms the detector does catch target-day weather. The ablation showed the future weather had **hurt** every tree model (XGBoost 15.92 → 14.74 MAE without it) and did nothing for the linear ones.

## Entry 5: model chosen on MAE alone
- **Context + wrong output:** The first comparison ranked and chose models by MAE only, although RMSE, recall, precision and false alarms were printed.
- **Risk:** MAE hides systematic bias and how badly high-pollution days are missed, which are the days that matter for health advice. A model can win MAE while under-forecasting every bad day.
- **Detection:** Added bias, R², MAPE, error on days with true PM2.5 ≥ 150 (MAE_high, bias_high), F1, false-alarm rate and PR-AUC. MAE_high came out at about 60 µg/m³ with bias_high = −MAE_high for every model, so every high day is under-forecast, which MAE alone never showed.
- **Correction + verification:** Selection is now a mean rank across MAE, RMSE, MAE_high, |bias| and PR-AUC, fixed before looking at the results, and Huber (the MAE winner, with the largest negative bias) no longer wins automatically. **Follow-up error in the same step:** the AI first called the about −60 bias on high days "the main weakness". Grouping errors by the *predicted* value instead showed a bias of about 0 in every band, so the forecasts are calibrated. Grouping by the *actual* value builds in a selection effect: days that turned out high are mostly unpredictable spikes. Documented correctly in SPRINT1.md section 5.6.

## Entry 6 (spare): wrong AQI table is the obvious default
- **Context + wrong output:** The natural assumption is the current (2024) US EPA AQI table, where 0–9.0 µg/m³ maps to AQI 0–50.
- **Risk:** Every batch_2 sensor-day would be converted with a systematic error. The two networks would disagree, gap-filling from batch_2 would be biased, and the 165 threshold would be applied to wrong values.
- **Detection:** Converted batch_1 PM2.5 to AQI under both tables and compared it with batch_2: 2024 table 171/1744 exact matches, pre-2024 table 1714/1744.
- **Correction + verification:** `AQI_BREAKPOINTS` uses the pre-2024 table. After inversion the median cross-network difference is 0.41 µg/m³, printed by `python src/clean.py`.

## Entry 7 (spare): README filenames do not match the repository
- **Context + wrong output:** The supplied README lists `sensors/network_a_utc_pm25.csv` and `network_b_pkt_aqi.csv` and mentions a "3-attempt" limit. The actual files are `batch_1_sensor_data.csv` / `batch_2_sensor_data.csv`, and SPEC says one submission.
- **Risk:** Code written from the README would fail to load the data, and the team could misplan the submission.
- **Detection:** Listed the repository files and compared them with the README and SPEC.md.
- **Correction + verification:** Followed SPEC.md, which is authoritative on mechanical details. The scripts load the real filenames and run end to end.

## Entry 8: hazardous spikes declared "random" without checking the calendar
- **Context + wrong output:** After finding no correlation between spikes and weather, the AI concluded the spikes were "close to random noise" and that "no model will catch them reliably". It stopped there and never tested calendar effects. A team member pushed back ("there are traps we must be hitting").
- **Risk:** A real, predictable signal was left out of the model, costing about 1.3 µg/m³ MAE, and the alarm was written off as hopeless. The submission would have been weaker than the data allowed.
- **Detection:** Grouped residuals (PM2.5 minus the sensor's recent level) by weekday. There is a clear weekly cycle even with spikes excluded: Fri +7, Thu/Sat +3–4, Mon −8, Tue −7 µg/m³. The residual autocorrelation at lags 3–4 is negative (≈ −0.09), which fits a 7-day cycle.
- **Correction + verification:** Added weekday-of-target features (calendar, so no future data) and a past-only spike-frequency feature. The leakage test covers both. Walk-forward MAE fell from 14.35 to 12.96 (best model) and PR-AUC rose from 0.059 to 0.093; on the untouched confirmation fold, MAE improved from 14.58 to 13.14. Holdout forecasts show the expected cycle (Friday mean 140, Monday 122).

## Entry 9 (Sprint 2): batch loader silently dropped questions
- **Context + wrong output:** The AI-written `load_questions()` kept a line only if it had a numeric label ("1.", "Q1:") or ended with "?". Test lines labeled "A1." and "A3." that ended with "." were dropped without any message, and they were exactly the prompt-injection and "best guess" tests.
- **Risk:** Questions in the released set could be silently unanswered. That costs points directly, and nobody would notice before submission.
- **Detection:** The run reported 6 answers for 8 input lines; the IDs were renumbered (Q4..Q6) instead of A1..A5.
- **Correction + verification:** The loader accepts any alphanumeric label, bullets and markdown tables, and prints a WARNING for every non-blank line it skips. A format test with 9 lines in 7 styles loaded all 7 questions and flagged the 1 intro sentence.

## Entry 10 (Sprint 2): output validator rejected correct answers
- **Context + wrong output:** The numeric-claim validator extracted every number in the LLM answer, including "12" from "DOC-12" and "8" from "DOC-08", and rejected those correct answers as "unsupported numbers [12.0]".
- **Risk:** Good LLM answers were replaced by longer extractive fallbacks. A validator that is too strict quietly degrades quality; one that is too loose lets invented numbers through.
- **Detection:** The live run trace (`--trace`) showed `validation: unsupported numbers [12.0]` and `[8.0]` for the mask and brick-kiln questions.
- **Correction + verification:** Document IDs are removed before number extraction. On the rerun, 16/16 practice answers passed. The mock-LLM test still rejects a genuinely invented number (42 µg/m³).

