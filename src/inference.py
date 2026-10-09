"""Predict a data folder with the SAVED model: no retraining.

    python src/inference.py --data C:/path/to/judges_folder
    python src/inference.py --data C:/path/to/judges_folder --alarm-threshold 165 --out my_predictions.csv

The folder must use the challenge layout:
    sensors/batch_1_sensor_data.csv   sensors/batch_2_sensor_data.csv
    weather/weather_history.csv       weather/sensor_metadata.csv
    holdout/holdout_inputs.csv        holdout/holdout_weather.csv (optional)
The saved model (models/model.joblib) is written by `python src/forecast.py`.
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import clean
from features import build_rows, load_weather

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "model.joblib"
REQUIRED = ["sensors/batch_1_sensor_data.csv", "sensors/batch_2_sensor_data.csv", "weather/weather_history.csv",
            "weather/sensor_metadata.csv", "holdout/holdout_inputs.csv"]
COLUMNS = ["sensor_id", "target_date", "predicted_pm25", "hazardous"]


def check_folder(data_dir):
    missing = [f for f in REQUIRED if not (Path(data_dir) / f).exists()]
    if missing:
        raise FileNotFoundError("missing in data folder: " + ", ".join(missing))


def predict_folder(data_dir, alarm_threshold=165.0, model_path=MODEL, log=print):
    """Clean the folder's data (same checks), build features at its last observation, predict."""
    data_dir = Path(data_dir)
    check_folder(data_dir)
    if not Path(model_path).exists():
        raise FileNotFoundError(f"{model_path} not found: run `python src/forecast.py` once to train and save it")
    bundle = joblib.load(model_path)
    daily = clean.build_daily(data_dir)
    holdout = pd.read_csv(data_dir / "holdout" / "holdout_inputs.csv", parse_dates=["target_date"])
    last_obs = daily["date"].max()
    h = (holdout["target_date"] - last_obs).dt.days
    if h.min() < 1:
        raise ValueError(f"holdout targets must be after the last observation ({last_obs.date()})")
    if h.max() > bundle["max_horizon_trained"]:
        log(f"note: holdout needs up to {h.max()} days ahead; the model was trained up to "
            f"{bundle['max_horizon_trained']} (longer horizons are extrapolated)")
    rows = build_rows(daily[["sensor_id", "area", "date", "pm25"]], load_weather(data_dir), origins=[last_obs],
                      horizons=range(1, int(h.max()) + 1), data_dir=data_dir)
    rows["predicted_pm25"] = np.clip(rows["mean7"].fillna(rows["lag0"]).to_numpy()
                                     + bundle["model"].predict(rows[bundle["features"]]), 0, None)
    sub = holdout[["sensor_id", "target_date"]].merge(
        rows[["sensor_id", "target_date", "predicted_pm25"]], on=["sensor_id", "target_date"], how="left",
        validate="1:1")
    if sub["predicted_pm25"].isna().any():
        bad = sub[sub.predicted_pm25.isna()].sensor_id.unique()[:5]
        raise ValueError(f"no prediction for some holdout rows (sensors without history?): {list(bad)}")
    sub["predicted_pm25"] = sub["predicted_pm25"].round(2)
    sub["hazardous"] = (sub["predicted_pm25"] >= alarm_threshold).astype(int)
    sub["target_date"] = sub["target_date"].dt.strftime("%Y-%m-%d")
    sub = sub[COLUMNS]
    assert len(sub) == len(holdout) and not sub.duplicated(["sensor_id", "target_date"]).any()
    info = {"model": bundle["model_name"], "trained_through": bundle["trained_through"],
            "data_last_observation": str(last_obs.date()), "rows": len(sub), "hazardous": int(sub.hazardous.sum()),
            "horizons": f"{h.min()}..{h.max()} days", "alarm_threshold": alarm_threshold}
    return sub, info


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with sensors/, weather/, holdout/")
    ap.add_argument("--alarm-threshold", type=float, default=165.0)
    ap.add_argument("--out", help="output CSV (default: outputs/inference_predictions.csv)")
    args = ap.parse_args()
    sub, info = predict_folder(args.data, args.alarm_threshold)
    out = Path(args.out) if args.out else ROOT / "outputs" / "inference_predictions.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    print(info)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
