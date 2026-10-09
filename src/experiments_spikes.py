"""Spike-aware forecast: predict WHEN a spike happens instead of averaging it away.

Two-part model:
  1. level  = current Ridge forecast (expected value)
  2. P(spike) from a classifier; rows above a cut-off get  level + typical spike size,
     so they cross 165 and hazardous = 1 follows the SPEC rule.

Usage:  python src/experiments_spikes.py      -> outputs/experiments_spikes.csv

Protocol: same 11 weekly walk-forward folds as experiments.py. The classifier and the
typical spike size are fitted on training rows only; the cut-off for fold k is chosen on
earlier folds only (nested). Folds 1-3 are warm-up; folds 4-11 are scored.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, fbeta_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import experiments as E
from features import CALENDAR_FEATURES, build_rows, load_daily, load_weather

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
HAZ, JUMP = 165.0, 45.0
PAST = ["mean7", "median7", "sensor_level", "spike_rate30", "days_since_spike", "std7", "max7", "h"] + CALENDAR_FEATURES
WITH_TEMP = PAST + ["w_temp_c", "dw_temp_c"]  # target-day temperature (holdout_weather; SPEC says "MAY" use)
level_model = E.residual_ridge(E.BASE, "mean7")


def spike_label(df):
    return ((df["y"] - df["median7"]) > JUMP).astype(int)


def spike_model(feats, kind):
    def fit_predict(train, test):
        tr = train.dropna(subset=["y", "median7"])
        yb = spike_label(tr)
        if kind == "logistic":
            m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000))
        else:
            m = LGBMClassifier(n_estimators=150, learning_rate=0.03, num_leaves=5, min_child_samples=80,
                               subsample=0.8, subsample_freq=1, reg_lambda=20, verbose=-1, random_state=0,
                               scale_pos_weight=(1 - yb.mean()) / max(yb.mean(), 1e-3))
        m.fit(tr[feats], yb)
        level = level_model(train, test)
        # typical spike size above the LEVEL forecast, learned on training spikes only
        lvl_tr = level_model(train, tr)
        size = float(np.median((tr["y"] - lvl_tr)[yb == 1])) if yb.sum() else 100.0
        return level, m.predict_proba(test[feats])[:, 1], size
    return fit_predict


VARIANTS = {
    "spike logistic, past-only": spike_model(PAST, "logistic"),
    "spike LightGBM, past-only": spike_model(PAST, "lgbm"),
    "spike logistic, + target-day temperature": spike_model(WITH_TEMP, "logistic"),
    "spike LightGBM, + target-day temperature": spike_model(WITH_TEMP, "lgbm"),
}


def main():
    daily, weather = load_daily(), load_weather()
    rows = build_rows(daily, weather)
    rows["dw_temp_c"] = rows["w_temp_c"] - rows["w0_temp_c"]
    out = []
    for c, train, test in E.folds(rows, daily.date.max()):
        o = test[["sensor_id", "target_date", "h", "y", "median7"]].copy()
        o["cutoff"] = c
        for name, fn in VARIANTS.items():
            lvl, p, size = fn(train, test)
            o["level"], o[f"p|{name}"], o[f"size|{name}"] = lvl, p, size
        out.append(o)
        print(f"  fold {c.date()} done", flush=True)
    cv = pd.concat(out, ignore_index=True)
    cut = sorted(cv.cutoff.unique())
    scored = cv[cv.cutoff >= cut[E.WARMUP]]
    yh = scored.y >= HAZ

    def summarise(label, pred):
        e = pred - scored.y
        flag = pred >= HAZ
        tp, fp = int((flag & yh).sum()), int((flag & ~yh).sum())
        return {"variant": label, "MAE": e.abs().mean(), "RMSE": np.sqrt((e ** 2).mean()), "flagged": int(flag.sum()),
                "caught": tp, "of": int(yh.sum()), "false_alarms": fp, "recall": tp / yh.sum(),
                "precision": tp / flag.sum() if flag.sum() else np.nan,
                "F1": fbeta_score(yh, flag, beta=1, zero_division=0)}

    res = [summarise("current (level only, SPEC rule)", scored["level"])]
    for name in VARIANTS:
        pcol, scol = f"p|{name}", f"size|{name}"
        res_ap = average_precision_score(spike_label(scored.assign(y=scored.y)), scored[pcol])
        for beta in (1, 2):
            preds = []
            for k in range(E.WARMUP, len(cut)):
                past = cv[cv.cutoff < cut[k]]
                cand = np.unique(np.quantile(past[pcol], np.linspace(0.80, 0.999, 120)))
                score = [fbeta_score(past.y >= HAZ, (past.level + (past[pcol] >= t) * past[scol]) >= HAZ,
                                     beta=beta, zero_division=0) for t in cand]
                t = cand[int(np.argmax(score))]
                cur = cv[cv.cutoff == cut[k]]
                preds.append(cur.level + (cur[pcol] >= t) * cur[scol])
            r = summarise(f"{name} (cut-off tuned for F{beta})", pd.concat(preds).reindex(scored.index))
            r["spike PR-AUC"] = res_ap
            res.append(r)
    R = pd.DataFrame(res)
    pd.set_option("display.width", 250)
    print(f"\nScored folds 4-11: {len(scored)} rows, {int(yh.sum())} hazardous; spike base rate "
          f"{spike_label(scored).mean():.3f}")
    print(R.round(3).to_string(index=False))
    R.to_csv(ROOT / "outputs" / "experiments_spikes.csv", index=False)


if __name__ == "__main__":
    main()
