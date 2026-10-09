"""Leakage tests: prove that nothing after the forecast origin reaches the model.

Run:  python tests/test_leakage.py      (or: pytest tests/)

Test idea: replace every PM2.5 reading and weather value AFTER origin t with
random noise. If any feature at origin t changes, that feature used the future.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import clean  # noqa: E402
from features import (AREA_NETWORK_FEATURES, CALENDAR_FEATURES, PAST_FEATURES, ROBUST_FEATURES,  # noqa: E402
                      SPIKE_FEATURES, TARGET_WEATHER_FEATURES, WEATHER_LAG_FEATURES, build_rows, load_daily,
                      load_weather)

# default model features plus every past-only candidate tried in the experiments
DEFAULT_FEATURES = (PAST_FEATURES + CALENDAR_FEATURES + SPIKE_FEATURES + ROBUST_FEATURES
                    + WEATHER_LAG_FEATURES + AREA_NETWORK_FEATURES)

DAILY, WEATHER = load_daily(), load_weather()
LAST = DAILY["date"].max()
# The 6 walk-forward origins plus the final origin used for predictions.csv
ORIGINS = [LAST - pd.Timedelta(days=10 + 7 * i) for i in range(6)] + [LAST]
rng = np.random.default_rng(0)


def _scramble_after(t):
    d, w = DAILY.copy(), WEATHER.copy()
    fut = d["date"] > t
    d.loc[fut, "pm25"] = rng.uniform(0, 500, fut.sum())
    w.loc[w.index > t] = rng.uniform(0, 100, w.loc[w.index > t].shape)
    return d, w


def _features_at(daily, weather, t, cols):
    r = build_rows(daily, weather, origins=[t])
    return r.sort_values(["sensor_id", "h"])[cols].reset_index(drop=True)


def test_past_features_ignore_the_future():
    for t in ORIGINS:
        before = _features_at(DAILY, WEATHER, t, DEFAULT_FEATURES)
        after = _features_at(*_scramble_after(t), t, DEFAULT_FEATURES)
        pd.testing.assert_frame_equal(before, after, obj=f"past-only features at origin {t.date()}")
    print(f"PASS  {len(DEFAULT_FEATURES)} features (default + candidates) unchanged when all data after the origin is randomised "
          f"({len(ORIGINS)} origins)")


def test_detector_is_sensitive():
    """Control: the test must catch a feature that DOES look ahead (target-day weather)."""
    t = ORIGINS[0]
    before = _features_at(DAILY, WEATHER, t, TARGET_WEATHER_FEATURES)
    after = _features_at(*_scramble_after(t), t, TARGET_WEATHER_FEATURES)
    assert not before.equals(after), "control failed: leakage test cannot detect look-ahead"
    print("PASS  control: target-day weather features DO change, so the test can detect look-ahead")


def test_cleaning_is_causal():
    """Cleaning history only up to t must give the same values as cleaning the full file."""
    full = clean.build_daily().set_index(["sensor_id", "date"])["pm25"]
    orig1, orig2 = clean.load_batch1, clean.load_batch2
    try:
        for t in ORIGINS:
            clean.load_batch1 = lambda t=t: (lambda d: d[d["date"] <= t])(orig1())
            clean.load_batch2 = lambda t=t: (lambda d: d[d["date"] <= t])(orig2())
            part = clean.build_daily().set_index(["sensor_id", "date"])["pm25"]
            pd.testing.assert_series_equal(part, full.loc[part.index], obj=f"cleaned data up to {t.date()}")
    finally:
        clean.load_batch1, clean.load_batch2 = orig1, orig2
    print(f"PASS  cleaning data truncated at each origin gives identical values ({len(ORIGINS)} origins)")


def test_walk_forward_split():
    rows = build_rows(DAILY, WEATHER)
    for t in ORIGINS[:-1]:
        train = rows[(rows.target_date <= t) & rows.y.notna()]
        test = rows[rows.origin == t]
        assert train.target_date.max() <= t < test.target_date.min()
        assert train.origin.max() < t, "a training row's origin is not before the test origin"
    print("PASS  every training target date <= fold origin < every test target date")


if __name__ == "__main__":
    test_past_features_ignore_the_future()
    test_detector_is_sensitive()
    test_cleaning_is_causal()
    test_walk_forward_split()
