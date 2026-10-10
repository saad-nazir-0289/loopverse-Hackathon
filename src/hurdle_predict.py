"""Predict the holdout with the two-part (hurdle) spike model and report its validation scores.

    python src/hurdle_predict.py            -> outputs/predictions_hurdle.csv (SPEC format)
    python src/hurdle_predict.py --out predictions.csv     # replace the submission (only if you decide to)

Model (best variant from experiments_hurdle.py, SPRINT1.md section 11):
  level    = submitted Ridge forecast (normal-day PM2.5)
  P(spike) = L2 logistic on [target-day temp, temp change, mean7], class weights "balanced"
  size     = median spike excess over the level in training spikes
  forecast = level + size if P(spike) >= t else level;  hazardous = forecast >= 165 (SPEC rule)
The threshold t is chosen on walk-forward folds; the reported validation scores are NESTED
(each fold's threshold chosen on earlier folds only), so they are honest out-of-sample numbers.
Uses target-day weather from holdout/holdout_weather.csv (SPEC: "MAY" be used).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

import experiments as E
import experiments_hurdle as H
from features import build_rows, load_daily, load_weather
from forecast import validate_submission

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
FEATS = ["w_temp_c", "dw_temp_c", "mean7"]
HAZ = 165.0
model = H.hurdle("ridge-logistic", "weights", FEATS, "constant")


def decide(df, t):
    return df["level"] + (df["p"] >= t) * df["size"]


def best_threshold(df):
    cand = np.unique(np.quantile(df["p"], np.linspace(0.5, 0.999, 150)))
    f1 = [f1_score(df.y >= HAZ, decide(df, t) >= HAZ, zero_division=0) for t in cand]
    return float(cand[int(np.argmax(f1))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs" / "predictions_hurdle.csv"))
    args = ap.parse_args()

    daily, weather = load_daily(), load_weather()
    last_obs = daily.date.max()
    holdout = pd.read_csv(ROOT / "holdout" / "holdout_inputs.csv", parse_dates=["target_date"])
    h_max = int((holdout.target_date - last_obs).dt.days.max())
    rows = build_rows(daily, weather, horizons=range(1, max(10, h_max) + 1))
    rows["dw_temp_c"] = rows["w_temp_c"] - rows["w0_temp_c"]

    # ---- walk-forward validation (11 folds; folds 4-11 scored with nested thresholds)
    parts = []
    for c, tr, te in E.folds(rows, last_obs):
        te = te.dropna(subset=["median7"]).copy()
        te["level"], te["p"], te["size"], _ = model(tr, te)
        te["cutoff"] = c
        parts.append(te[["sensor_id", "target_date", "y", "median7", "level", "p", "size", "cutoff"]])
    cv = pd.concat(parts, ignore_index=True)
    cut = sorted(cv.cutoff.unique())
    nested = []
    for k in range(E.WARMUP, len(cut)):
        t = best_threshold(cv[cv.cutoff < cut[k]])
        cur = cv[cv.cutoff == cut[k]].copy()
        cur["pred"] = decide(cur, t)
        nested.append(cur)
    sc = pd.concat(nested)
    yh, fh = sc.y >= HAZ, sc.pred >= HAZ
    ys = H.spike_label(sc)
    tp, fp, fn = int((yh & fh).sum()), int((~yh & fh).sum()), int((yh & ~fh).sum())
    base = sc.level
    print(f"VALIDATION (walk-forward, folds 4-11, {len(sc)} rows, {int(yh.sum())} truly hazardous, nested thresholds)")
    print(f"{'metric':32s} {'hurdle model':>14s} {'submitted level model':>22s}")
    for name, a, b in [
        ("MAE (ug/m3)", (sc.pred - sc.y).abs().mean(), (base - sc.y).abs().mean()),
        ("RMSE (ug/m3)", np.sqrt(((sc.pred - sc.y) ** 2).mean()), np.sqrt(((base - sc.y) ** 2).mean())),
        ("hazardous accuracy", accuracy_score(yh, fh), accuracy_score(yh, base >= HAZ)),
        ("hazardous recall (caught)", tp / yh.sum(), 0.0),
        ("hazardous precision", tp / fh.sum() if fh.sum() else float("nan"), float("nan")),
        ("hazardous F1", f1_score(yh, fh, zero_division=0), 0.0),
        ("flagged / caught / false alarms", f"{int(fh.sum())} / {tp} / {fp}", "0 / 0 / 0"),
        ("missed hazardous days", fn, int(yh.sum())),
        ("spike ranking ROC-AUC (0.5=chance)", roc_auc_score(ys, sc.p), "n/a"),
        ("spike ranking PR-AUC (chance %.3f)" % ys.mean(), average_precision_score(ys, sc.p), "n/a"),
    ]:
        fa = f"{a:14.3f}" if isinstance(a, float) else f"{a!s:>14s}"
        fb = f"{b:22.3f}" if isinstance(b, float) else f"{b!s:>22s}"
        print(f"{name:32s} {fa} {fb}")

    # ---- final model: threshold from all scored folds, fit on all data up to the last observation
    t_final = best_threshold(cv[cv.cutoff >= cut[E.WARMUP]])
    train = rows[(rows.target_date <= last_obs) & rows.y.notna()]
    test = rows[rows.origin == last_obs].copy()
    test["level"], test["p"], test["size"], _ = model(train, test)
    test["predicted_pm25"] = np.clip(decide(test, t_final), 0, None)
    sub = holdout[["sensor_id", "target_date"]].merge(
        test[["sensor_id", "target_date", "predicted_pm25", "p", "level"]], on=["sensor_id", "target_date"],
        how="left", validate="1:1")
    sub["predicted_pm25"] = sub["predicted_pm25"].round(2)
    sub["hazardous"] = (sub["predicted_pm25"] >= HAZ).astype(int)
    sub["target_date"] = sub["target_date"].dt.strftime("%Y-%m-%d")
    flagged = sub[sub.hazardous == 1]
    out = sub[["sensor_id", "target_date", "predicted_pm25", "hazardous"]]
    validate_submission(out, holdout)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nHOLDOUT: threshold P(spike) >= {t_final:.3f}, spike size +{test['size'].iloc[0]:.1f} ug/m3")
    print(f"wrote {args.out}: {len(out)} rows, {len(flagged)} hazardous, PM2.5 {out.predicted_pm25.min():.1f}.."
          f"{out.predicted_pm25.max():.1f}")
    if len(flagged):
        print("flagged rows (expected ~5-6% of these to be real spikes, per validation precision):")
        print(flagged[["sensor_id", "target_date", "level", "p", "predicted_pm25"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
