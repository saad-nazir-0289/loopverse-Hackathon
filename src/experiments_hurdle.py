"""Two-part (hurdle) spike model, with explicit class-imbalance handling.

    Stage 1  classifier : P(spike)          (ridge-style L2 logistic  |  gradient boosting)
    Stage 2  regressor  : spike size        (ridge on spike rows      |  constant median size)
    Combine             : hard  -> level + size if P >= t  (t tuned on EARLIER folds only)
                          soft  -> level + P * size        (expected value)

Imbalance strategies for stage 1: none, class weights ("balanced"), undersampling normal rows to 5:1.

Usage:  python src/experiments_hurdle.py      -> outputs/experiments_hurdle*.csv
Same 11 weekly walk-forward folds as experiments.py; folds 4-11 scored.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, fbeta_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import experiments as E
from features import CALENDAR_FEATURES, ROBUST_FEATURES, SPIKE_FEATURES, WEATHER_LAG_FEATURES, build_rows, load_daily, load_weather

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
HAZ, JUMP = 165.0, 45.0
PAST = ["mean7", "median7", "std7", "max7", "sensor_level", "lag0", "lag1", "trend7", "h", "city_lag0"] \
    + SPIKE_FEATURES + ROBUST_FEATURES + WEATHER_LAG_FEATURES + CALENDAR_FEATURES
PAST = list(dict.fromkeys(PAST))  # median7 is in two groups
TEMP = PAST + ["w_temp_c", "w_humidity_pct", "w_wind_kmh", "dw_temp_c"]
level_model = E.residual_ridge(E.BASE, "mean7")  # the submitted model = the "normal day" forecast


def spike_label(df):
    return ((df["y"] - df["median7"]) > JUMP).astype(int)


def make_clf(kind, imbalance):
    w = "balanced" if imbalance == "weights" else None
    if kind == "ridge-logistic":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.05, class_weight=w, max_iter=3000))
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.03, max_depth=3, min_samples_leaf=60,
                                          l2_regularization=1.0, class_weight=w, random_state=0)


def undersample(tr, yb, ratio=5, seed=0):
    pos, neg = tr[yb == 1], tr[yb == 0]
    neg = neg.sample(min(len(neg), ratio * max(len(pos), 1)), random_state=seed)
    out = pd.concat([pos, neg])
    return out, spike_label(out)


def hurdle(kind, imbalance, feats, size_model):
    def fit_predict(train, test):
        tr = train.dropna(subset=["y", "median7"])
        yb = spike_label(tr)
        fit_rows, fit_y = undersample(tr, yb) if imbalance == "undersample" else (tr, yb)
        clf = make_clf(kind, imbalance).fit(fit_rows[feats], fit_y)
        p = clf.predict_proba(test[feats])[:, 1]
        level_te, level_tr = level_model(train, test), level_model(train, tr)
        excess = (tr["y"] - level_tr)[yb == 1]  # stage 2 target: how far above the normal-day forecast
        if size_model == "ridge" and yb.sum() >= 10:
            reg = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=50.0))
            reg.fit(tr.loc[yb == 1, feats], excess)
            size = np.clip(reg.predict(test[feats]), 0, None)
            size_tr_fit = reg.predict(tr.loc[yb == 1, feats])
        else:
            size = np.full(len(test), float(excess.median()) if len(excess) else 100.0)
            size_tr_fit = np.full(len(excess), float(excess.median()) if len(excess) else 100.0)
        return level_te, p, size, float(np.mean(np.abs(size_tr_fit - excess))) if len(excess) else np.nan
    return fit_predict


VARIANTS = {}
for feats_name, feats in [("past", PAST), ("past+temp", TEMP)]:
    for kind in ("ridge-logistic", "boosting"):
        for imb in ("none", "weights", "undersample"):
            VARIANTS[f"{kind} | {imb} | {feats_name} | ridge size"] = hurdle(kind, imb, feats, "ridge")
    VARIANTS[f"ridge-logistic | weights | {feats_name} | constant size"] = hurdle("ridge-logistic", "weights", feats,
                                                                                  "constant")


def main():
    daily, weather = load_daily(), load_weather()
    rows = build_rows(daily, weather)
    rows["dw_temp_c"] = rows["w_temp_c"] - rows["w0_temp_c"]
    parts = []
    for c, train, test in E.folds(rows, daily.date.max()):
        test = test.dropna(subset=["median7"])
        o = test[["sensor_id", "target_date", "h", "y", "median7"]].copy()
        o["cutoff"] = c
        for name, fn in VARIANTS.items():
            lvl, p, size, _ = fn(train, test)
            o["level"], o[f"p|{name}"], o[f"s|{name}"] = lvl, p, size
        parts.append(o)
        print(f"  fold {c.date()} done", flush=True)
    cv = pd.concat(parts, ignore_index=True)
    cut = sorted(cv.cutoff.unique())
    sc = cv[cv.cutoff >= cut[E.WARMUP]].copy()
    yh, ys = sc.y >= HAZ, spike_label(sc)

    def scores(label, pred, p=None):
        e = pred - sc.y
        f = pred >= HAZ
        tp, fp = int((f & yh).sum()), int((f & ~yh).sum())
        r = {"variant": label, "MAE": e.abs().mean(), "flagged": int(f.sum()), "caught": tp, "of": int(yh.sum()),
             "false_alarms": fp, "precision": tp / f.sum() if f.sum() else np.nan,
             "F1": fbeta_score(yh, f, beta=1, zero_division=0)}
        if p is not None:
            r.update({"spike_ROC_AUC": roc_auc_score(ys, p), "spike_PR_AUC": average_precision_score(ys, p),
                      "mean_P": float(np.mean(p)), "Brier": brier_score_loss(ys, p)})
        return r

    res = [scores("level only (submitted model, SPEC 165)", sc["level"])]
    for name in VARIANTS:
        pc, scol = f"p|{name}", f"s|{name}"
        hard = []
        for k in range(E.WARMUP, len(cut)):  # nested: threshold from earlier folds only
            past = cv[cv.cutoff < cut[k]]
            cand = np.unique(np.quantile(past[pc], np.linspace(0.5, 0.999, 150)))
            f1 = [fbeta_score(past.y >= HAZ, (past.level + (past[pc] >= t) * past[scol]) >= HAZ, beta=1, zero_division=0)
                  for t in cand]
            t = cand[int(np.argmax(f1))]
            cur = cv[cv.cutoff == cut[k]]
            hard.append(cur.level + (cur[pc] >= t) * cur[scol])
        res.append(scores(f"HARD  {name}", pd.concat(hard).reindex(sc.index), sc[pc]))
        res.append(scores(f"SOFT  {name}", sc.level + sc[pc] * sc[scol]))
    R = pd.DataFrame(res)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 70)
    print(f"\nScored folds 4-11: {len(sc)} rows | hazardous {int(yh.sum())} | spikes {int(ys.sum())} "
          f"(spike base rate {ys.mean():.3f} = PR-AUC chance; ROC-AUC 0.5 = chance)")
    print(R.round(3).to_string(index=False))
    R.to_csv(ROOT / "outputs" / "experiments_hurdle.csv", index=False)

    # stage 2 on its own: how well is spike SIZE predicted, given that a spike happened?
    print("\nStage 2 (size | spike happened), scored folds:")
    real = sc[ys == 1]
    excess = real.y - real.level
    for name in [n for n in VARIANTS if n.startswith("ridge-logistic | weights")]:
        err = (real[f"s|{name}"] - excess).abs()
        print(f"   {name:60s} size MAE {err.mean():6.1f}  (true size mean {excess.mean():.1f}, sd {excess.std():.1f})")


if __name__ == "__main__":
    main()
