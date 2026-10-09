# Lahore Smog Intelligence: Loopverse 3.0 AI/ML

Next-day PM2.5 forecasts and a hazardous alarm for 15 Lahore sensors (Sprint 1), and a grounded advisory assistant that calls the forecast and cites the supplied guidance (Sprint 2).

| Deliverable | File |
|---|---|
| Forecast submission | [`predictions.csv`](predictions.csv) (150 rows, SPEC section 4) |
| Assistant | `ask(question)` and `forecast(location, target_date)` in [`src/assistant.py`](src/assistant.py) |
| Answers to the question set | `answers.json` / `answers.csv` (generated, see below) |
| Recommendation | [`RECOMMENDATION.md`](RECOMMENDATION.md) |
| AI error log | [`AI_ERROR_LOG.md`](AI_ERROR_LOG.md) (3 required entries) + [`AI_ERROR_LOG_ADDITIONAL.md`](AI_ERROR_LOG_ADDITIONAL.md) |
| Method and evidence | [`SPRINT1.md`](SPRINT1.md), [`SPRINT2.md`](SPRINT2.md), [`outputs/trap_audit.md`](outputs/trap_audit.md) |
| Original challenge README | [`CHALLENGE_README.md`](CHALLENGE_README.md) |

## Quick start
Python 3.11+.
```bash
pip install -r requirements.txt
python src/run_all.py                                            # clean -> forecast -> tests -> audit
python src/run_all.py --questions spec/25_unseen_questions.md    # ...and answer the question set
streamlit run app.py                                             # simple UI
```
**Only the submitted model** (fast, about 5 s, same `predictions.csv`):
```bash
python src/clean.py
python src/forecast.py --model ridge_strong
```
Without `--model`, `forecast.py` re-validates all 14 candidates and picks the best by the pre-declared rule (currently `ridge_strong`).

`run_all.py` stops at the first failing step. Each step can also run on its own:

| Step | Command | Output |
|---|---|---|
| 1. Clean and standardise | `python src/clean.py` | `data/processed/daily_pm25.csv` + audit printout + assertions |
| 2. Forecast | `python src/forecast.py` | walk-forward report, `outputs/walk_forward_*.csv`, **`predictions.csv`** |
| 3. Leakage proof | `python tests/test_leakage.py` | 4 tests |
| 4. Trap evidence | `python src/audit_traps.py` | `outputs/trap_audit.md`, `outputs/figures/` |
| 5. Assistant tests | `python tests/test_assistant.py` | 11 tests, no API key needed |
| 6. Answer questions | `python src/run_questions.py <file> --trace` | `answers.json`, `answers.csv`, `answers_trace.json` |
| One question | `python src/assistant.py "Should schools in DHA close tomorrow?"` | JSON in the SPEC schema |
| Experiments (optional) | `python src/experiments.py`, `python src/experiments_ml.py` | model and alarm comparisons |

## Inference only: predict new data with the saved model (no retraining)
`python src/forecast.py` saves the selected fitted model to `models/model.joblib` (with `models/model_info.json`). Any folder in the challenge layout can then be predicted without training:
```bash
python src/inference.py --data C:/path/to/judges_folder                # -> outputs/inference_predictions.csv
python src/inference.py --data C:/path/to/judges_folder --out predictions.csv --alarm-threshold 165
```
In the UI: **Run on new data** tab. Enter a folder path (or upload the CSV files), click **Run inference**, then download `predictions.csv`. The folder's data goes through the same cleaning checks (timezone, units, coverage); a wrong or incomplete folder gives a clear error. Running `python src/inference.py --data .` on this repository reproduces the committed `predictions.csv` exactly.

## Running on the judges' own files
Put files with the **same names and columns** in the same folders, then rerun `python src/run_all.py`. Nothing is hard-coded to the supplied dates, sensors or row counts.

| Folder / file | Columns | Notes |
|---|---|---|
| `sensors/batch_1_sensor_data.csv` | `sensor_id, timestamp, reading_value` | PM2.5 µg/m³, UTC. Several readings per day are averaged per Pakistan-time day. `-999` = missing |
| `sensors/batch_2_sensor_data.csv` | `sensor_id, timestamp, reading_value` | AQI (integer 0–500), Pakistan time. Converted to PM2.5 with the US EPA (pre-2024) breakpoints |
| `weather/weather_history.csv` | `date, temp_c, humidity_pct, wind_kmh` | daily |
| `weather/sensor_metadata.csv` | `sensor_id, area, batch, timezone, unit_type` | defines the covered areas for the assistant |
| `holdout/holdout_inputs.csv` | `sensor_id, forecast_origin_date, target_date, area` | any number of rows and days ahead; `predictions.csv` gets exactly these pairs |
| `holdout/holdout_weather.csv` | `date, temp_c, humidity_pct, wind_kmh` | not used by default (it is after the forecast origin) |
| `docs/*.md` | metadata block with `document_id, title, authority, published_date, status[, supersedes, superseded_by]` | any number of documents; files without `document_id` are skipped with a warning |
| question set | `.md`, `.txt`, `.csv` (`question_id, question`) or `.json` | `python src/run_questions.py <file>`; labels like `1.`, `Q1:`, `A3.` and markdown tables are read; skipped lines are reported |
| reference forecast (if released) | `sensor_id, target_date, predicted_pm25[, hazardous]` | save as `reference/reference_forecast.csv` and set `ADVISOR_FORECAST_SOURCE=reference` |

What adapts automatically:
- the forecast horizon follows the holdout;
- the assistant's "today" is the day before the first holdout target (override with `ADVISOR_TODAY=YYYY-MM-DD`);
- covered areas come from the metadata, and the supported dates from the holdout;
- the "typical error" quoted in answers comes from the latest validation run;
- the coded policy rules (school, odd-even, construction, health bands) are used only if their documents exist and still state the same thresholds.

The cleaning step **fails loudly** if timezone or units do not line up (cross-network gap ≥ 1 µg/m³, coverage < 90%, non-integer AQI). `tests/test_assistant.py` asks about specific supplied areas and dates, so `run_all.py` treats it as a warning on other data.

## Hazardous alarm threshold
SPEC's rule is `hazardous = predicted_pm25 >= 165`, which is the default everywhere. To use another alarm threshold:

| Where | How |
|---|---|
| `predictions.csv` | `python src/forecast.py --alarm-threshold 150` (or `python src/run_all.py --alarm-threshold 150`) |
| Streamlit UI | sidebar **Alarm threshold**: updates the forecast flag, the assistant's answers, the charts, and the downloadable `predictions.csv`, and shows what that threshold would have caught and falsely flagged in validation |
| Assistant / forecast tool | `ADVISOR_ALARM_THRESHOLD=150` in `.env` or the environment |

The alarm only changes the `hazardous` yes/no flag. The documents' own thresholds (DOC-01 health bands, DOC-03 school rules, DOC-05 odd-even, DOC-08 construction) are policy facts and stay as written.

## Streamlit UI
```bash
streamlit run app.py        # http://localhost:8501
```
Tabs: **Ask the assistant** (answer, sources, forecast_called, cited documents, raw SPEC JSON), **Forecast** (area/date lookup and an all-areas chart), **All predictions** (table, download, daily trend). For a public link, deploy the GitHub repo on share.streamlit.io with main file `app.py`; add `OPENAI_API_KEY` under Secrets only if LLM answers are wanted.

## LLM configuration
The assistant works with no LLM: deterministic, extractive, cited answers. With an LLM it writes shorter natural answers, which are validated (citations, numbers, injection) before use. Settings live in `.env` (copy `.env.example`; `.env` is git-ignored) or in environment variables:

| Variable | Meaning |
|---|---|
| `OPENAI_API_KEY` | key for the provider (any non-empty value for local Ollama) |
| `OPENAI_BASE_URL` | any OpenAI-compatible endpoint (default: OpenAI) |
| `OPENAI_MODEL`, `OPENAI_FALLBACK_MODEL` | model names for that provider |
| `ADVISOR_LLM_API` | `auto` (default), `responses` or `chat`; `chat` is what most compatible providers offer |
| `ADVISOR_LLM=none` | force deterministic answers |

## Repository layout
```
src/clean.py            cleaning, timezone/unit conversion, assertions
src/features.py         leak-free features (lags, rolling, weekday, spike history)
src/forecast.py         model comparison, walk-forward validation, predictions.csv
src/audit_traps.py      evidence for the four handbook traps
src/experiments*.py     method bake-offs (alarm methods, RF/XGBoost/LSTM)
src/advisor/            assistant: documents, retrieval, understanding, forecast tool, policy rules, LLM, ask()
src/assistant.py        ask() / forecast() entry point and CLI
src/run_questions.py    batch answering
src/run_all.py          the whole pipeline in one command
app.py                  Streamlit UI
tests/                  leakage tests, assistant tests, practice questions
```
