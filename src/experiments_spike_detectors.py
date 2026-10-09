"""Can any simple model from the handbook detect spike days? Threshold-free ranking test.

Usage:  python src/experiments_spike_detectors.py   -> outputs/experiments_spike_detectors.csv

Spike = PM2.5 more than 45 ug/m3 above the sensor's 7-day median at the origin (these
cause almost all hazardous days). Same 11 walk-forward folds; models fit on training rows
only; scored on folds 4-11. ROC-AUC 0.5 = coin flip; PR-AUC chance = spike base rate.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import experiments as E
from features import (AREA_NETWORK_FEATURES, CALENDAR_FEATURES, PAST_FEATURES, ROBUST_FEATURES, SPIKE_FEATURES,
                      WEATHER_LAG_FEATURES, build_rows, load_daily, load_weather)

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
JUMP = 45.0
ALL_PAST = PAST_FEATURES + CALENDAR_FEATURES + SPIKE_FEATURES + ROBUST_FEATURES + WEATHER_LAG_FEATURES \
    + AREA_NETWORK_FEATURES + ["spike_today", "jump_today"]
WITH_TARGET_WEATHER = ALL_PAST + ["w_temp_c", "w_humidity_pct", "w_wind_kmh", "dw_temp_c", "dw_humidity_pct",
                                  "dw_wind_kmh"]


def label(df):
    return ((df["y"] - df["median7"]) > JUMP).astype(int)


def rule(col, sign=1):
    return lambda tr, te: sign * te[col].fillna(te[col].median()).to_numpy()


def clf(make, feats):
    def fit_predict(tr, te):
        tr = tr.dropna(subset=["median7"])
        m = make()
        m.fit(tr[feats], label(tr))
        return m.predict_proba(te[feats])[:, 1]
    return fit_predict


logit = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),  # noqa: E731
                              LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000))
forest = lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(  # noqa: E731
    n_estimators=300, min_samples_leaf=40, max_features=0.4, class_weight="balanced", random_state=0, n_jobs=4))
boost = lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.03, max_depth=3,  # noqa: E731
                                               min_samples_leaf=60, class_weight="balanced", random_state=0)

DETECTORS = {
    "persistence: spiked on origin day": rule("spike_today"),
    "recent average level (mean7)": rule("mean7"),
    "rolling spread (std7)": rule("std7"),
    "rolling max above median": rule("jump_max7"),
    "sensor spike rate (30 d)": rule("spike_rate30"),
    "days since last spike (fewer = higher)": rule("days_since_spike", -1),
    "warm target day (temp)": rule("w_temp_c"),
    "logistic, all past features": clf(logit, ALL_PAST),
    "random forest, all past features": clf(forest, ALL_PAST),
    "boosting, all past features": clf(boost, ALL_PAST),
    "logistic, + target-day weather": clf(logit, WITH_TARGET_WEATHER),
    "random forest, + target-day weather": clf(forest, WITH_TARGET_WEATHER),
    "boosting, + target-day weather": clf(boost, WITH_TARGET_WEATHER),
}


def main():
    daily, weather = load_daily(), load_weather()
    rows = build_rows(daily, weather)
    rows["jump_today"] = rows["lag0"] - rows["median7"]
    rows["spike_today"] = (rows["jump_today"] > JUMP).astype(float)
    rows["jump_max7"] = rows["max7"] - rows["median7"]
    out = []
    for c, tr, te in E.folds(rows, daily.date.max()):
        te = te.dropna(subset=["median7"])
        o = te[["sensor_id", "target_date", "y", "median7"]].copy()
        o["cutoff"] = c
        for name, fn in DETECTORS.items():
            o[name] = fn(tr, te)
        out.append(o)
        print(f"  fold {c.date()} done", flush=True)
    cv = pd.concat(out, ignore_index=True)
    cut = sorted(cv.cutoff.unique())
    sc = cv[cv.cutoff >= cut[E.WARMUP]]
    yb = label(sc)
    res = []
    for name in DETECTORS:
        per_fold = [roc_auc_score(label(g), g[name]) for _, g in sc.groupby("cutoff") if 0 < label(g).sum() < len(g)]
        top = sc[name] >= np.quantile(sc[name], 0.9)
        res.append({"detector": name, "ROC_AUC": roc_auc_score(yb, sc[name]), "PR_AUC": average_precision_score(yb, sc[name]),
                    "AUC_fold_min": min(per_fold), "AUC_fold_max": max(per_fold),
                    "top10%_caught": int((top & (yb == 1)).sum()), "top10%_flagged": int(top.sum())})
    R = pd.DataFrame(res).sort_values("ROC_AUC", ascending=False)
    pd.set_option("display.width", 250)
    print(f"\nScored folds 4-11: {len(sc)} rows, {int(yb.sum())} spike rows (base rate {yb.mean():.3f} = PR-AUC chance)")
    print(R.round(3).to_string(index=False))
    R.to_csv(ROOT / "outputs" / "experiments_spike_detectors.csv", index=False)


if __name__ == "__main__":
    main()
