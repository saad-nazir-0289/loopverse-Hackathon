"""Method bake-off: point-forecast variants (A) and hazardous-alarm methods (B).

Usage:  python src/experiments.py      -> outputs/experiments_*.csv, outputs/experiments_log.txt

Protocol (fixed before running):
- 11 walk-forward folds, weekly origins 2026-08-09 .. 2026-10-18, each forecasting
  all 15 sensors 1..10 days ahead (150 rows), training only on targets <= origin.
- Alarm thresholds are tuned NESTED: the threshold applied to fold k is chosen
  only on out-of-sample scores from folds < k. Folds 1-3 are warm-up and not scored.
- Scored folds 4..11 (origins 2026-08-30 .. 2026-10-18).
- Point forecasts: MAE, RMSE, bias. Alarms: PR-AUC (threshold-free), and at the
  nested threshold recall / precision / F1 / F2 / false alarms.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, fbeta_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import forecast
from features import CALENDAR_FEATURES, PAST_FEATURES, ROBUST_FEATURES, SPIKE_FEATURES, build_rows, load_daily, load_weather

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
HAZ = 165.0
BASE = PAST_FEATURES + CALENDAR_FEATURES + SPIKE_FEATURES
ROBUST = BASE + ROBUST_FEATURES
N_FOLDS, WARMUP = 11, 3


# ---------- A. point-forecast variants: fn(train, test) -> PM2.5 ----------

def residual_ridge(feats, base_col, alpha=300.0):
    def fit_predict(train, test):
        tr = train.dropna(subset=["y", base_col])
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))
        m.fit(tr[feats], tr["y"] - tr[base_col])
        return test[base_col].fillna(test["lag0"]).to_numpy() + m.predict(test[feats])
    return fit_predict


def residual_ridge_robust_target(feats, base_col, clip=60.0, alpha=300.0):
    """Train on residuals clipped at +clip, so the model learns the level, not the spikes."""
    def fit_predict(train, test):
        tr = train.dropna(subset=["y", base_col])
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))
        m.fit(tr[feats], (tr["y"] - tr[base_col]).clip(upper=clip))
        return test[base_col].fillna(test["lag0"]).to_numpy() + m.predict(test[feats])
    return fit_predict


def per_horizon(make):
    """Separate model per horizon instead of one pooled model with h as a feature."""
    def fit_predict(train, test):
        out = np.empty(len(test))
        for h in test["h"].unique():
            mt = (test["h"] == h).to_numpy()
            out[mt] = make(train[train["h"] == h], test[mt])
        return out
    return fit_predict


def lgbm_residual(feats, base_col):
    def fit_predict(train, test):
        tr = train.dropna(subset=["y", base_col])
        m = LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=60,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10, verbose=-1,
                          random_state=0, objective="huber", alpha=30)
        m.fit(tr[feats], tr["y"] - tr[base_col])
        return test[base_col].fillna(test["lag0"]).to_numpy() + m.predict(test[feats])
    return fit_predict


POINT = {
    "current: ridge300 on mean7, base feats": residual_ridge(BASE, "mean7"),
    "ridge300 on mean7 + robust feats": residual_ridge(ROBUST, "mean7"),
    "ridge300 on median7 + robust feats": residual_ridge(ROBUST, "median7"),
    "ridge300 on median7, clipped target": residual_ridge_robust_target(ROBUST, "median7"),
    "per-horizon ridge300 on median7": per_horizon(residual_ridge(ROBUST, "median7")),
    "lgbm huber-loss on median7": lgbm_residual(ROBUST, "median7"),
    "blend ridge(median7) + lgbm huber": lambda tr, te: (residual_ridge(ROBUST, "median7")(tr, te)
                                                        + lgbm_residual(ROBUST, "median7")(tr, te)) / 2,
}


# ---------- B. alarm methods: fn(train, test) -> score (higher = more likely hazardous) ----------

def score_point(point_fn):
    return point_fn  # the PM2.5 forecast itself, thresholded


def quantile_lgbm(alpha, feats=ROBUST):
    """Upper quantile of tomorrow's PM2.5: alarm if the q-th percentile reaches the threshold."""
    def fit_predict(train, test):
        tr = train.dropna(subset=["y", "median7"])
        m = LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=300, learning_rate=0.03, num_leaves=7,
                          min_child_samples=60, subsample=0.8, subsample_freq=1, verbose=-1, random_state=0)
        m.fit(tr[feats], tr["y"] - tr["median7"])
        return test["median7"].fillna(test["lag0"]).to_numpy() + m.predict(test[feats])
    return fit_predict


def logistic(feats=ROBUST, C=0.1):
    def fit_predict(train, test):
        tr = train.dropna(subset=["y"])
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(C=C, class_weight="balanced", max_iter=2000))
        m.fit(tr[feats], tr["y"] >= HAZ)
        return m.predict_proba(test[feats])[:, 1]
    return fit_predict


def lgbm_classifier(feats=ROBUST):
    def fit_predict(train, test):
        tr = train.dropna(subset=["y"])
        yb = tr["y"] >= HAZ
        m = LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=40,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10,
                           scale_pos_weight=(~yb).sum() / max(yb.sum(), 1), verbose=-1, random_state=0)
        m.fit(tr[feats], yb)
        return m.predict_proba(test[feats])[:, 1]
    return fit_predict


def residual_probability(point_fn, by="spike_bin"):
    """P(y >= 165) = P(residual >= 165 - forecast), residual distribution taken from
    TRAINING rows only, separately for spiky and calm sensors (spike_rate30 tercile)."""
    def fit_predict(train, test):
        tr = train.dropna(subset=["y"]).copy()
        # in-sample residuals of a 7-day-median forecast; cheap and leak-free (training rows only)
        tr["res"] = tr["y"] - tr["median7"].fillna(tr["lag0"])
        edges = np.nanquantile(tr["spike_rate30"], [1 / 3, 2 / 3])
        bins_tr = np.digitize(tr["spike_rate30"].fillna(0), edges)
        bins_te = np.digitize(test["spike_rate30"].fillna(0), edges)
        pred = point_fn(train, test)
        out = np.empty(len(test))
        for b in range(3):
            r = np.sort(tr["res"][bins_tr == b].dropna().to_numpy())
            mt = bins_te == b
            need = HAZ - pred[mt]
            out[mt] = 1 - np.searchsorted(r, need) / max(len(r), 1)
        return out
    return fit_predict


def climatology_spike_rate(train, test):
    return test["spike_rate30"].fillna(0).to_numpy()


ALARM = {
    "point forecast (current model)": score_point(residual_ridge(BASE, "mean7")),
    "point forecast (median7 model)": score_point(residual_ridge(ROBUST, "median7")),
    "quantile q0.90 (LightGBM)": quantile_lgbm(0.90),
    "quantile q0.95 (LightGBM)": quantile_lgbm(0.95),
    "logistic regression (balanced)": logistic(),
    "LightGBM classifier (balanced)": lgbm_classifier(),
    "residual probability by spikiness": residual_probability(residual_ridge(ROBUST, "median7")),
    "sensor spike rate only (naive)": climatology_spike_rate,
}


# ---------- harness ----------

def folds(rows, last_obs):
    cutoffs = [last_obs - pd.Timedelta(days=10 + 7 * i) for i in range(N_FOLDS)][::-1]
    for c in cutoffs:
        train = rows[(rows.target_date <= c) & rows.y.notna()]
        test = rows[(rows.origin == c) & rows.y.notna()]
        assert train.target_date.max() <= c < test.target_date.min()
        yield c, train, test


def run(methods, rows, last_obs):
    out = []
    for c, train, test in folds(rows, last_obs):
        o = test[["sensor_id", "target_date", "h", "y"]].copy()
        o["cutoff"] = c
        for name, fn in methods.items():
            o[name] = fn(train, test)
        out.append(o)
        print(f"  fold {c.date()} done", flush=True)
    return pd.concat(out, ignore_index=True)


def nested_threshold_eval(cv, name, beta):
    """For each scored fold, pick the threshold maximising F-beta on earlier folds only."""
    cut = sorted(cv.cutoff.unique())
    flags = []
    for k in range(WARMUP, len(cut)):
        past = cv[cv.cutoff < cut[k]]
        yb = past.y >= HAZ
        cands = np.unique(np.quantile(past[name], np.linspace(0.5, 0.999, 200)))
        f = [fbeta_score(yb, past[name] >= t, beta=beta, zero_division=0) for t in cands]
        t = cands[int(np.argmax(f))]
        cur = cv[cv.cutoff == cut[k]]
        flags.append(pd.Series(cur[name].to_numpy() >= t, index=cur.index))
    return pd.concat(flags)


def main():
    daily, weather = load_daily(), load_weather()
    last_obs = daily["date"].max()
    rows = build_rows(daily, weather)
    forecast.FEATURES[:] = BASE

    print("A. point-forecast variants")
    cvA = run(POINT, rows, last_obs)
    cvA.to_csv(ROOT / "outputs" / "experiments_point_predictions.csv", index=False)
    scored = cvA[cvA.cutoff >= sorted(cvA.cutoff.unique())[WARMUP]]
    A = pd.DataFrame({n: {"MAE": np.mean(np.abs(scored[n] - scored.y)),
                          "RMSE": np.sqrt(np.mean((scored[n] - scored.y) ** 2)),
                          "bias": np.mean(scored[n] - scored.y),
                          "MAE_last3_folds": np.mean(np.abs(scored[n] - scored.y)[scored.cutoff >= sorted(scored.cutoff.unique())[-3]]),
                          "folds_won_vs_current": int(sum(
                              np.mean(np.abs(g[n] - g.y)) < np.mean(np.abs(g[list(POINT)[0]] - g.y))
                              for _, g in scored.groupby("cutoff")))}
                      for n in POINT}).T
    print(f"\nA. point forecasts, scored folds ({scored.cutoff.nunique()} folds, {len(scored)} rows):")
    print(A.round(3).sort_values("MAE").to_string())
    A.to_csv(ROOT / "outputs" / "experiments_point_scores.csv")

    print("\nB. alarm methods")
    cvB = run(ALARM, rows, last_obs)
    cvB.to_csv(ROOT / "outputs" / "experiments_alarm_scores_raw.csv", index=False)
    scoredB = cvB[cvB.cutoff >= sorted(cvB.cutoff.unique())[WARMUP]]
    yb = scoredB.y >= HAZ
    res = {}
    for n in ALARM:
        r = {"PR_AUC": average_precision_score(yb, scoredB[n])}
        for beta in (1, 2):
            fl = nested_threshold_eval(cvB, n, beta).reindex(scoredB.index)
            r[f"F{beta}_recall"] = recall_score(yb, fl, zero_division=0)
            r[f"F{beta}_precision"] = precision_score(yb, fl, zero_division=0)
            r[f"F{beta}_score"] = fbeta_score(yb, fl, beta=beta, zero_division=0)
            r[f"F{beta}_caught"] = int((fl & yb).sum())
            r[f"F{beta}_false_alarms"] = int((fl & ~yb).sum())
        # recall when allowed to flag the top 10% of rows
        top = scoredB[n] >= np.quantile(scoredB[n], 0.90)
        r["recall@top10%"] = (top & yb).sum() / yb.sum()
        res[n] = r
    B = pd.DataFrame(res).T.sort_values("PR_AUC", ascending=False)
    pd.set_option("display.width", 250)
    print(f"\nB. alarms, scored folds: {len(scoredB)} rows, {int(yb.sum())} hazardous (chance PR-AUC {yb.mean():.3f})")
    print(B.round(3).to_string())
    B.to_csv(ROOT / "outputs" / "experiments_alarm_scores.csv")


if __name__ == "__main__":
    main()
