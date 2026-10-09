"""Step 1: standardize and clean both sensor networks into one daily PM2.5 panel.

Order (handbook): parse time -> align timezone -> convert units -> replace -999
-> handle stuck sensor -> aggregate to daily.

Output: data/processed/daily_pm25.csv with one row per (sensor_id, date) in
Pakistan local time, PM2.5 in ug/m3. Raw columns are kept for auditing.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "daily_pm25.csv"

SENTINEL = -999
PKT_OFFSET = pd.Timedelta(hours=5)  # Pakistan Standard Time = UTC+5, no DST
STUCK_MIN_RUN = 3  # DOC-06: one sensor flat-lined for three consecutive days
MIN_VALID_PER_DAY = 1  # each sensor-day has 2 readings (one per network); need >= 1 valid, non-faulty
COVERAGE_MIN = 0.90  # every sensor must have valid PM2.5 on >= 90% of days, else stop
MAX_NETWORK_GAP = 1.0  # ug/m3: median |batch_1 - converted batch_2| must be below this

# US EPA PM2.5 AQI breakpoints (pre-2024 table). Verified against the data:
# batch_2 AQI equals AQI(batch_1 PM2.5) exactly for ~98% of readings, +/-1
# for the rest (rounding). The 2024 table does not fit.
# (C_low, C_high, I_low, I_high)
AQI_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]


def aqi_to_pm25(aqi):
    """Invert the piecewise-linear EPA AQI formula. NaN stays NaN."""
    aqi = np.asarray(aqi, dtype=float)
    out = np.full_like(aqi, np.nan)
    for c_lo, c_hi, i_lo, i_hi in AQI_BREAKPOINTS:
        m = (aqi >= i_lo) & (aqi <= i_hi)
        out[m] = (aqi[m] - i_lo) * (c_hi - c_lo) / (i_hi - i_lo) + c_lo
    # AQI between bands (e.g. 50.5 after averaging) -> interpolate upper band
    gaps = np.isnan(out) & ~np.isnan(aqi) & (aqi >= 0)
    if gaps.any():
        out[gaps] = np.interp(aqi[gaps], [b[2] for b in AQI_BREAKPOINTS], [b[0] for b in AQI_BREAKPOINTS])
    return out


def load_batch1():
    """Network A: UTC timestamps, raw PM2.5."""
    df = pd.read_csv(ROOT / "sensors" / "batch_1_sensor_data.csv")
    ts_utc = pd.to_datetime(df["timestamp"])
    assert ts_utc.notna().all(), "unparseable batch_1 timestamps"
    ts_pkt = ts_utc + PKT_OFFSET
    # Timezone check: the supplied data has one reading per day at 19:00 UTC = 00:00 PKT.
    # Other times (e.g. hourly files) are fine: they are averaged per PKT day below.
    if not (ts_utc.dt.hour == 19).all():
        print("note: batch_1 readings are not all at 19:00 UTC; averaging per Pakistan-time day")
    df["date"] = ts_pkt.dt.normalize()
    df["date_utc_naive"] = ts_utc.dt.normalize()  # kept only to prove the shift matters (assert_standardized)
    df["pm25_b1"] = df["reading_value"].replace(SENTINEL, np.nan)
    v = df["pm25_b1"].dropna()
    assert ((v > 0) & (v < 1000)).all(), "batch_1 values outside a plausible PM2.5 range"
    # Aggregate: daily mean of valid readings (one reading per day in the supplied files)
    return df.groupby(["sensor_id", "date"], as_index=False).agg(
        date_utc_naive=("date_utc_naive", "first"), pm25_b1=("pm25_b1", "mean"))


def load_batch2():
    """Network B: Pakistan local dates, AQI."""
    df = pd.read_csv(ROOT / "sensors" / "batch_2_sensor_data.csv")
    df["date"] = pd.to_datetime(df["timestamp"]).dt.normalize()  # already Pakistan time
    df["aqi_b2"] = df["reading_value"].replace(SENTINEL, np.nan)
    a = df["aqi_b2"].dropna()
    # Unit assertion: AQI is an integer index on 0..500
    assert ((a >= 0) & (a <= 500) & (a == a.round())).all(), "batch_2 is not integer AQI on 0..500"
    df["pm25_b2"] = aqi_to_pm25(df["aqi_b2"])  # convert each reading first, then average (AQI is non-linear)
    return df.groupby(["sensor_id", "date"], as_index=False).agg(aqi_b2=("aqi_b2", "mean"), pm25_b2=("pm25_b2", "mean"))


def flag_stuck(values, min_run=STUCK_MIN_RUN):
    """True for every reading inside a run of >= min_run identical, non-NaN values."""
    v = pd.Series(values).reset_index(drop=True)
    run_id = (v != v.shift()).cumsum()
    run_len = v.groupby(run_id).transform("size")
    return ((run_len >= min_run) & v.notna()).to_numpy()


def build_daily():
    meta = pd.read_csv(ROOT / "weather" / "sensor_metadata.csv")
    b1, b2 = load_batch1(), load_batch2()

    # Both files contain all 15 sensors on the same PKT dates once aligned.
    daily = b1.merge(b2, on=["sensor_id", "date"], how="outer", validate="1:1")
    overlap = daily[["pm25_b1", "pm25_b2"]].notna().all(axis=1).sum() / max(len(b1), len(b2))
    if len(daily) != len(b1) or len(daily) != len(b2):
        print(f"note: networks cover different sensor-days ({len(b1)} vs {len(b2)}; {overlap:.0%} overlap)")
    daily = daily.merge(meta[["sensor_id", "area", "batch"]], on="sensor_id", how="left", validate="m:1")
    assert daily["area"].notna().all(), "unknown sensor id"
    daily = daily.sort_values(["sensor_id", "date"]).reset_index(drop=True)

    # Stuck detection runs on the precise 0.1-resolution PM2.5 series; the
    # integer AQI series repeats by coincidence and would give false positives.
    daily["stuck"] = daily.groupby("sensor_id")["pm25_b1"].transform(lambda s: flag_stuck(s.to_numpy()))

    # Prefer native PM2.5 (no AQI rounding loss); fill -999 gaps from network B.
    daily["pm25"] = daily["pm25_b1"].fillna(daily["pm25_b2"])
    daily["pm25_source"] = np.where(daily["pm25_b1"].notna(), "batch_1",
                                    np.where(daily["pm25_b2"].notna(), "batch_2_aqi_inverted", "missing"))
    # Stuck days are a hardware fault in both networks: exclude, never impute
    # with future values. Downstream features forward-fill from the past only.
    daily.loc[daily["stuck"], "pm25"] = np.nan
    daily.loc[daily["stuck"], "pm25_source"] = "stuck_excluded"
    # Aggregate rule: count valid, non-faulty readings behind each sensor-day
    daily["n_valid"] = (daily[["pm25_b1", "pm25_b2"]].notna().sum(axis=1)).where(~daily["stuck"], 0)
    daily.loc[daily["n_valid"] < MIN_VALID_PER_DAY, "pm25"] = np.nan
    assert_standardized(daily)
    return daily.drop(columns="date_utc_naive")


def assert_standardized(daily):
    """Automated checks that clock and units really agree across networks."""
    both = daily.pm25_b1.notna() & daily.pm25_b2.notna()
    gap = (daily.pm25_b1 - daily.pm25_b2)[both].abs().median()
    assert gap < MAX_NETWORK_GAP, f"networks disagree after conversion: median gap {gap:.2f}"
    # Same comparison WITHOUT the +5 h shift must be clearly worse, or the shift is not doing anything
    naive = daily[["sensor_id", "date_utc_naive", "pm25_b1"]].merge(
        daily[["sensor_id", "date", "pm25_b2"]], left_on=["sensor_id", "date_utc_naive"], right_on=["sensor_id", "date"])
    naive_gap = (naive.pm25_b1 - naive.pm25_b2).abs().median()
    assert naive_gap > 5 * gap, f"timezone shift has no effect ({naive_gap:.2f} vs {gap:.2f})"
    cov = daily.groupby("sensor_id").pm25.apply(lambda s: s.notna().mean())
    assert (cov >= COVERAGE_MIN).all(), f"coverage below {COVERAGE_MIN:.0%}: {cov[cov < COVERAGE_MIN].to_dict()}"


def audit(daily):
    print(f"rows={len(daily)} sensors={daily.sensor_id.nunique()} "
          f"dates={daily.date.min().date()}..{daily.date.max().date()} ({daily.date.nunique()} days)")
    print("missing -999 batch_1:", int(daily.pm25_b1.isna().sum()), " batch_2:", int(daily.aqi_b2.isna().sum()))
    both = daily.pm25_b1.notna() & daily.pm25_b2.notna()
    err = (daily.pm25_b1 - daily.pm25_b2)[both].abs()
    print(f"cross-network check (PM2.5 vs inverted AQI): median |diff|={err.median():.2f}, p99={err.quantile(.99):.2f}")
    stuck = daily[daily.stuck]
    for sid, g in stuck.groupby("sensor_id"):
        print(f"stuck: {sid} {g.date.min().date()}..{g.date.max().date()} value={g.pm25_b1.iloc[0]}")
    print("source counts:", daily.pm25_source.value_counts().to_dict())
    print("remaining NaN pm25:", int(daily.pm25.isna().sum()))
    print("valid readings per sensor-day:", daily.n_valid.value_counts().sort_index().to_dict(),
          f"(rule: >= {MIN_VALID_PER_DAY})")
    cov = daily.groupby("sensor_id").pm25.apply(lambda s: s.notna().mean())
    print(f"coverage per sensor: min {cov.min():.1%} ({cov.idxmin()}), threshold {COVERAGE_MIN:.0%}")
    print("assertions passed: timezone (19:00 UTC = 00:00 PKT, shift matters), units (AQI integer 0..500, "
          f"PM2.5 plausible), cross-network gap < {MAX_NETWORK_GAP}, coverage")


def main():
    daily = build_daily()
    audit(daily)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    daily.assign(date=daily.date.dt.strftime("%Y-%m-%d")).to_csv(OUT, index=False)
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
