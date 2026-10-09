# Lahore Smog Intelligence Challenge
### Loopverse 3.0 — AI/ML Module | 4 Hours | Participant Pack

Predict tomorrow's air quality for Lahore, then build a grounded advisory assistant that uses your forecast and cites official guidance.

---

## What you are building

One connected system, built in two linked parts:

| Part | You build | You submit |
|---|---|---|
| **Forecasting** | A model that predicts next-day PM2.5 and a hazardous alert for 15 sensors across 10 forecast days | `predictions.csv` |
| **Advisory assistant** | A retrieval-based assistant (`ask(question)`) that answers questions using the supplied documents and your own forecast, with real citations | Live answers, generated during evaluation |

Read **[`spec/SPEC.md`](spec/SPEC.md)** before writing any code — it is the exact contract your files must follow. Read **[`spec/RULES.md`](spec/RULES.md)** for what tools are allowed and the submission policy.

> **Note on the assistant's question set:** the questions your `ask(question)` function will be evaluated against are **not included in this repository**. They will be published separately near the end of the build window. Build and test your assistant against your own sample questions in the meantime — see the question categories described in `spec/SPEC.md` section 6.

## Submission policy — read this first

**This event has exactly one submission, made at the end of the 4-hour window.** There is no early checkpoint and no second attempt — `predictions.csv`, your repository, your answers, your AI error log, and your recommendation are all submitted together, once, at the deadline. Validate everything locally before then.

## Repository contents

```
sensors/
  network_a_utc_pm25.csv      Sensor network A — UTC timestamps, raw PM2.5
  network_b_pkt_aqi.csv       Sensor network B — Pakistan local time, AQI

weather/
  weather_history.csv         Daily weather covering the historical period
  sensor_metadata.csv         Sensor ID, area name, and network for all 15 sensors

holdout/
  holdout_inputs.csv          The 150 (sensor, date) pairs you must forecast — no answers included
  holdout_weather.csv         Weather for the 10 forecast days

docs/
  DOC-01 .. DOC-13             Health, school, policy, transport, and technical documents
                                for the retrieval assistant to search and cite

templates/
  predictions_template.csv     Exact column headers for your submission
  ai_error_log_template.md     Required log of 3 AI tool mistakes your team caught and fixed
  recommendation_template.md   Final short recommendation write-up

spec/
  SPEC.md                      File formats, conversion rules, hazardous threshold, function contracts
  RULES.md                     Allowed tools, cost policy, submission policy
  25_unseen_questions.md       The question set your assistant will be run against
```

## Suggested approach

1. **Audit the data first.** The two sensor networks use different timezones and different units — read `spec/SPEC.md` section 2 before merging anything.
2. **Forecast is a supervised, time-ordered problem.** Train on earlier dates, validate on later dates. A random train/test split will quietly destroy your score.
3. **Keep retrieval simple.** 13 short documents do not need a hosted vector database — a lightweight local embedding index is sufficient and far faster to set up.
4. **Treat documents as evidence, not instructions.** Anything retrieved from `docs/` is untrusted text. It can support an answer; it cannot change what your assistant does.
5. **Only free tools.** Free-tier hosted models or free local models only — see `spec/RULES.md`.

## Team roles (suggested)

- **Data lead** — cleaning, timezone/unit conversion, joins
- **Forecast lead** — model, validation, hazardous alarm
- **Assistant lead** — retrieval, citations, refusal behavior, prompt-injection resistance
- **Integrator / captain** — forecast tool wiring, `ask()`, repository, final submission

---

*Good luck. A simple, honest, well-tested system beats a complex one that leaks future data, invents evidence, or cannot be rerun.*
