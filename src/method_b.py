"""Method B from the blog post: quantile regression (predict the high end, not the middle).

    python src/method_b.py      -> outputs/method_b_scores.csv, outputs/predictions_method_b.csv

For each walk-forward fold, fit the 0.5 and 0.9 quantiles of next-days PM2.5 and test:
  1. q50 as the forecast            (MAE vs the submitted ridge level forecast)
  2. q90 as an upper bound          (coverage: share of days at or below it, target 90%)
  3. alarm when q90 >= 165          (the SPEC threshold applied to the high end, no tuning)
  4. the gap q90 - q50 as a spike risk score (ROC-AUC, PR-AUC vs base rate)
Models: LightGBM quantile (as in the post) and linear quantile regression (sklearn), both on y - mean7.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import experiments as E
from features import build_rows, load_daily, load_weather
from forecast import validate_submission

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
HAZ, JUMP = 165.0, 45.0
FEATS = E.BASE


def lgbm_q(alpha):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=300, learning_rate=0.03, num_leaves=7,
                         min_child_samples=60, subsample=0.8, subsample_freq=1, verbose=-1, random_state=0)


def linear_q(alpha):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         QuantileRegressor(quantile=alpha, alpha=0.01, solver="highs"))


MAKERS = {"LightGBM quantile": lgbm_q, "linear quantile": linear_q}


def fit_quantiles(make, train, test, sample=None):
    tr = train.dropna(subset=["y", "mean7"])
    if sample and len(tr) > sample:  # the linear LP solver is slow on all rows; a random subset is plenty
        tr = tr.sample(sample, random_state=0)
    base = test["mean7"].fillna(test["lag0"]).to_numpy()
    out = {}
    for a in (0.5, 0.9):
        m = make(a)
        m.fit(tr[FEATS], tr["y"] - tr["mean7"])
        out[a] = base + m.predict(test[FEATS])
    return out[0.5], np.maximum(out[0.9], out[0.5])


def main():
    daily, weather = load_daily(), load_weather()
    last_obs = daily.date.max()
    rows = build_rows(daily, weather)
    level = E.residual_ridge(E.BASE, "mean7")
    parts = []
    for c, tr, te in E.folds(rows, last_obs):
        te = te.dropna(subset=["median7"]).copy()
        te["cutoff"], te["level"] = c, level(tr, te)
        for name, make in MAKERS.items():
            te[f"q50|{name}"], te[f"q90|{name}"] = fit_quantiles(make, tr, te,
                                                                  sample=4000 if name.startswith("linear") else None)
        parts.append(te[["sensor_id", "target_date", "y", "median7", "level", "cutoff"]
                        + [c_ for c_ in te.columns if "|" in c_]])
        print(f"  fold {c.date()} done", flush=True)
    cv = pd.concat(parts, ignore_index=True)
    sc = cv[cv.cutoff >= sorted(cv.cutoff.unique())[E.WARMUP]]
    yh, ys = sc.y >= HAZ, (sc.y - sc.median7) > JUMP

    def alarm_stats(flag):
        tp, fp = int((flag & yh).sum()), int((flag & ~yh).sum())
        return tp, fp, (tp / flag.sum() if flag.sum() else float("nan"))

    res = [{"model": "submitted ridge level (SPEC 165)", "MAE": (sc.level - sc.y).abs().mean(),
            "q90 coverage": np.nan, "alarm flagged": int((sc.level >= HAZ).sum()),
            "caught (of %d)" % yh.sum(): 0, "false alarms": 0, "precision": np.nan,
            "gap ROC-AUC (spikes)": np.nan, "gap PR-AUC": np.nan}]
    for name in MAKERS:
        q50, q90 = sc[f"q50|{name}"], sc[f"q90|{name}"]
        tp, fp, prec = alarm_stats(q90 >= HAZ)
        gap = q90 - q50
        res.append({"model": f"Method B, {name}", "MAE": (q50 - sc.y).abs().mean(),
                    "q90 coverage": float((sc.y <= q90).mean()), "alarm flagged": int((q90 >= HAZ).sum()),
                    "caught (of %d)" % yh.sum(): tp, "false alarms": fp, "precision": prec,
                    "gap ROC-AUC (spikes)": roc_auc_score(ys, gap), "gap PR-AUC": average_precision_score(ys, gap)})
    R = pd.DataFrame(res)
    pd.set_option("display.width", 250)
    print(f"\nScored folds 4-11: {len(sc)} rows, {int(yh.sum())} hazardous, {int(ys.sum())} spikes "
          f"(spike base rate {ys.mean():.3f} = PR-AUC chance)")
    print(R.round(3).to_string(index=False))
    R.to_csv(ROOT / "outputs" / "method_b_scores.csv", index=False)

    # holdout with the better Method B variant (by MAE of q50)
    best = min(MAKERS, key=lambda n: (sc[f"q50|{n}"] - sc.y).abs().mean())
    holdout = pd.read_csv(ROOT / "holdout" / "holdout_inputs.csv", parse_dates=["target_date"])
    train = rows[(rows.target_date <= last_obs) & rows.y.notna()]
    test = rows[rows.origin == last_obs].copy()
    test["q50"], test["q90"] = fit_quantiles(MAKERS[best], train, test,
                                             sample=4000 if best.startswith("linear") else None)
    sub = holdout[["sensor_id", "target_date"]].merge(test[["sensor_id", "target_date", "q50", "q90"]],
                                                      on=["sensor_id", "target_date"], how="left", validate="1:1")
    sub["predicted_pm25"] = sub.q50.clip(lower=0).round(2)
    sub["hazardous"] = (sub.q90 >= HAZ).astype(int)  # alarm on the high end
    sub["target_date"] = sub.target_date.dt.strftime("%Y-%m-%d")
    out = sub[["sensor_id", "target_date", "predicted_pm25", "hazardous"]]
    validate_submission(out, holdout)
    out.to_csv(ROOT / "outputs" / "predictions_method_b.csv", index=False)
    print(f"\nHOLDOUT ({best}): {int(out.hazardous.sum())} of {len(out)} rows flagged (q90 >= 165); "
          f"q90 range {sub.q90.min():.1f}..{sub.q90.max():.1f}; wrote outputs/predictions_method_b.csv")


if __name__ == "__main__":
    main()
