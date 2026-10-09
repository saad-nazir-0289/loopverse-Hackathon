"""Step 2: leak-free features for direct multi-horizon forecasting.

Each row is (sensor, origin date t, horizon h) with target PM2.5 on t+h.
Every PM2.5 feature uses only readings dated <= t. Weather on the target date
is allowed: SPEC supplies holdout_weather.csv for the forecast days.

Why multi-horizon: observations end on 2026-10-28, but the holdout targets run
to 2026-11-07. Readings after 10-28 are the hidden answers, so a target on
11-07 is really a 10-day-ahead forecast from 10-28, whatever the nominal
forecast_origin_date says.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = range(1, 11)
WEATHER_COLS = ["temp_c", "humidity_pct", "wind_kmh"]
FFILL_LIMIT = 3  # carry a reading forward at most 3 days over gaps
SPIKE_JUMP = 45.0  # ug/m3 above the recent median that counts as a spike


def load_daily(path=None):
    d = pd.read_csv(path or ROOT / "data" / "processed" / "daily_pm25.csv", parse_dates=["date"])
    return d[["sensor_id", "area", "date", "pm25"]]


def load_weather(data_dir=ROOT):
    hist = pd.read_csv(Path(data_dir) / "weather" / "weather_history.csv", parse_dates=["date"])
    hold_path = Path(data_dir) / "holdout" / "holdout_weather.csv"
    hold = pd.read_csv(hold_path, parse_dates=["date"]) if hold_path.exists() else hist.iloc[:0]
    w = pd.concat([hist, hold]).drop_duplicates("date").sort_values("date")
    assert w["date"].is_unique
    return w.set_index("date")[WEATHER_COLS]


def _days_since(flags, cap=60):
    """Days since the last True in each column, counting only the past (0 = today)."""
    out = pd.DataFrame(index=flags.index, columns=flags.columns, dtype=float)
    for c in flags.columns:
        last, vals = None, []
        for i, f in enumerate(flags[c].to_numpy()):
            if f:
                last = i
            vals.append(cap if last is None else min(i - last, cap))
        out[c] = vals
    return out


def origin_features(daily):
    """Features known at the end of each origin day t, per sensor."""
    wide = daily.pivot(index="date", columns="sensor_id", values="pm25").sort_index()
    filled = wide.ffill(limit=FFILL_LIMIT)  # past-only fill; never interpolate
    feats = {}
    for k in range(7):
        feats[f"lag{k}"] = filled.shift(k)
    feats["mean3"] = filled.rolling(3, min_periods=2).mean()
    feats["mean7"] = filled.rolling(7, min_periods=4).mean()
    feats["mean14"] = filled.rolling(14, min_periods=7).mean()
    feats["std7"] = filled.rolling(7, min_periods=4).std()
    feats["max7"] = filled.rolling(7, min_periods=4).max()
    feats["min7"] = filled.rolling(7, min_periods=4).min()
    feats["trend7"] = feats["mean7"] - filled.shift(7).rolling(7, min_periods=4).mean()
    feats["sensor_level"] = wide.expanding(min_periods=7).mean()  # expanding = past only
    # Spike history: a day more than SPIKE_JUMP above the median of the previous
    # 14 days. Hazardous days are such spikes; some sensors spike more often.
    spike = (filled - filled.shift(1).rolling(14, min_periods=7).median()) > SPIKE_JUMP
    feats["spike_rate30"] = spike.astype(float).rolling(30, min_periods=10).mean()
    # Robust level: a single spike moves mean7 by ~15 for a week, the median barely
    feats["median7"] = filled.rolling(7, min_periods=4).median()
    feats["median14"] = filled.rolling(14, min_periods=7).median()
    feats["days_since_spike"] = _days_since(spike)
    city = filled.mean(axis=1)
    long = pd.concat({k: v.stack(future_stack=True) for k, v in feats.items()}, axis=1)
    long.index.names = ["origin", "sensor_id"]
    long = long.reset_index()
    long["city_lag0"] = long["origin"].map(city)
    long["city_mean7"] = long["origin"].map(city.rolling(7, min_periods=4).mean())
    long["sensor_offset"] = long["sensor_level"] - long["origin"].map(wide.expanding(min_periods=7).mean().mean(axis=1))
    return long


def build_rows(daily, weather, origins=None, horizons=HORIZONS, data_dir=ROOT):
    """Cross origin features with horizons, attach weather and (if known) target."""
    of = origin_features(daily)
    if origins is not None:
        of = of[of["origin"].isin(origins)]
    rows = []
    for h in horizons:
        r = of.copy()
        r["h"] = h
        r["target_date"] = r["origin"] + pd.Timedelta(days=h)
        rows.append(r)
    X = pd.concat(rows, ignore_index=True)
    w3 = weather.rolling(3, min_periods=1).mean()  # trailing: days t-2..t
    for c in WEATHER_COLS:
        X[f"w_{c}"] = X["target_date"].map(weather[c])
        X[f"w0_{c}"] = X["origin"].map(weather[c])
        X[f"dw_{c}"] = X[f"w_{c}"] - X[f"w0_{c}"]
        # weather lags: the days before the origin
        X[f"w1_{c}"] = (X["origin"] - pd.Timedelta(days=1)).map(weather[c])
        X[f"w2_{c}"] = (X["origin"] - pd.Timedelta(days=2)).map(weather[c])
        X[f"w3mean_{c}"] = X["origin"].map(w3[c])
    target = daily.set_index(["sensor_id", "date"])["pm25"]
    X["y"] = target.reindex(pd.MultiIndex.from_frame(X[["sensor_id", "target_date"]])).to_numpy()
    X["sensor_code"] = X["sensor_id"].str[1:].astype(int)
    # Calendar of the target day is known in advance, so it is not future data.
    # Weekly cycle in the data: Fri about +7, Mon about -8 ug/m3 vs the recent level.
    # Area and network (sensor_metadata.csv): one-hot area, flag for the AQI/PKT network
    meta = pd.read_csv(Path(data_dir) / "weather" / "sensor_metadata.csv").set_index("sensor_id")
    for sid in meta.index:
        X[f"area_{sid}"] = (X["sensor_id"] == sid).astype(float)
    X["net_b2"] = (X["sensor_id"].map(meta["batch"]) == "batch_2").astype(float)
    dow = X["target_date"].dt.dayofweek
    for k in range(7):
        X[f"dow{k}"] = (dow == k).astype(float)
    return X


# Default: strictly past-only. Everything here was observed on or before the origin day.
PAST_FEATURES = (
    [f"lag{k}" for k in range(7)]
    + ["mean3", "mean7", "mean14", "std7", "max7", "min7", "trend7",
       "sensor_level", "sensor_offset", "city_lag0", "city_mean7", "h"]
    + [f"w0_{c}" for c in WEATHER_COLS]
)
CALENDAR_FEATURES = [f"dow{k}" for k in range(7)]  # weekday of the target date
SPIKE_FEATURES = ["spike_rate30"]
ROBUST_FEATURES = ["median7", "median14", "days_since_spike"]  # candidates, see src/experiments.py
WEATHER_LAG_FEATURES = [f"{p}_{c}" for p in ("w1", "w2", "w3mean") for c in WEATHER_COLS]  # candidates
AREA_NETWORK_FEATURES = [f"area_S{i:02d}" for i in range(1, 16)] + ["net_b2"]  # candidates
# Opt-in (--target-weather): weather ON the target day, i.e. after the origin.
# SPEC says holdout_weather.csv "MAY" be used but the referenced rule is
# missing, so it is excluded by default. The ablation in SPRINT1.md shows
# it adds nothing measurable.
TARGET_WEATHER_FEATURES = [f"{p}_{c}" for p in ("w", "dw") for c in WEATHER_COLS]
