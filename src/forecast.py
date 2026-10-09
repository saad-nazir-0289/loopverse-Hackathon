"""Step 3: walk-forward validation against baselines, then write predictions.csv.

Usage:
    python src/clean.py
    python src/forecast.py            # picks the model with best walk-forward MAE
    python src/forecast.py --model ridge

Validation mirrors the holdout exactly: at each cutoff c, train only on rows
whose TARGET date is <= c, then forecast all 15 sensors for c+1..c+10 from
origin c (150 rows per fold).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from sklearn.metrics import average_precision_score, r2_score

from features import (CALENDAR_FEATURES, HORIZONS, PAST_FEATURES, SPIKE_FEATURES, TARGET_WEATHER_FEATURES,
                      build_rows, load_daily, load_weather)

FEATURES = list(PAST_FEATURES)  # set in main() from the command-line switches

ROOT = Path(__file__).resolve().parents[1]
HAZARD_THRESHOLD = 165.0  # SPEC section 3: a TRUE day is hazardous if PM2.5 >= 165
N_FOLDS = 6
FOLD_STEP_DAYS = 7


# ---------- models: each takes (train, test) and returns PM2.5 predictions ----------

def persistence(train, test):
    return test["lag0"].fillna(test["mean7"]).to_numpy()


def mean7(train, test):
    return test["mean7"].fillna(test["lag0"]).to_numpy()


def _residual_model(make_model):
    """Learn y - mean7 so tree models can follow the level outside the training range."""
    def fit_predict(train, test):
        tr = train.dropna(subset=["y", "mean7"])
        model = make_model()  # imputer/scaler live inside the pipeline: fit on train only
        model.fit(tr[FEATURES], tr["y"] - tr["mean7"])
        base = test["mean7"].fillna(test["lag0"])
        return base.to_numpy() + model.predict(test[FEATURES])
    return fit_predict


def _linear(estimator):
    return lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)


def _imputed(estimator):
    return lambda: make_pipeline(SimpleImputer(strategy="median"), estimator)


ridge = _residual_model(_linear(Ridge(alpha=10.0)))
ridge_strong = _residual_model(_linear(Ridge(alpha=300.0)))
elasticnet = _residual_model(_linear(ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=5000)))
huber = _residual_model(_linear(HuberRegressor(alpha=1.0, epsilon=1.5, max_iter=2000)))
hgb = _residual_model(lambda: HistGradientBoostingRegressor(
    max_iter=300, learning_rate=0.04, max_depth=4, min_samples_leaf=30, l2_regularization=1.0, random_state=0))
xgb = _residual_model(lambda: XGBRegressor(
    n_estimators=400, learning_rate=0.03, max_depth=3, min_child_weight=20, subsample=0.8,
    colsample_bytree=0.8, reg_lambda=5.0, random_state=0, n_jobs=4))
lgbm = _residual_model(lambda: LGBMRegressor(
    n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_lambda=5.0, random_state=0, verbose=-1))
extra_trees = _residual_model(_imputed(ExtraTreesRegressor(
    n_estimators=300, min_samples_leaf=30, max_features=0.5, random_state=0, n_jobs=4)))
random_forest = _residual_model(_imputed(RandomForestRegressor(
    n_estimators=300, min_samples_leaf=30, max_features=0.5, random_state=0, n_jobs=4)))


def _blend(*models):
    return lambda train, test: np.mean([m(train, test) for m in models], axis=0)


MODELS = {
    "persistence": persistence, "mean7": mean7,
    "ridge": ridge, "ridge_strong": ridge_strong, "elasticnet": elasticnet, "huber": huber,
    "hgb": hgb, "xgb": xgb, "lgbm": lgbm, "extra_trees": extra_trees, "random_forest": random_forest,
    "ridge+xgb": _blend(ridge, xgb), "ridge+lgbm": _blend(ridge, lgbm), "ridge+extra_trees": _blend(ridge, extra_trees),
}
BASELINES = {"persistence", "mean7"}


# ---------- evaluation ----------

HIGH_PM = 150.0  # "high pollution" band just below the hazardous line


def scores(y, p):
    """Regression error, error where it matters (high days), and alarm quality."""
    err = p - y
    y_h, p_h = y >= HAZARD_THRESHOLD, p >= HAZARD_THRESHOLD
    tp, fp, fn = int((y_h & p_h).sum()), int((~y_h & p_h).sum()), int((y_h & ~p_h).sum())
    recall = tp / (tp + fn) if tp + fn else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    high = y >= HIGH_PM
    return {
        "MAE": np.mean(np.abs(err)),
        "RMSE": np.sqrt(np.mean(err ** 2)),
        "MAPE%": 100 * np.mean(np.abs(err) / y),
        "bias": np.mean(err),                       # >0 over-forecasts, <0 under-forecasts
        "R2": r2_score(y, p),
        "MAE_high": np.mean(np.abs(err[high])),     # error on days with true PM2.5 >= 150
        "bias_high": np.mean(err[high]),
        "haz_true": int(y_h.sum()),
        "recall": recall,
        "precision": precision,
        "F1": 2 * precision * recall / (precision + recall) if tp else 0.0,
        "false_alarms": fp,
        "FA_rate": fp / int((~y_h).sum()),
        # Threshold-free: does a higher forecast rank hazardous days higher?
        # Chance level = share of hazardous rows (~0.03).
        "PR_AUC": average_precision_score(y_h, p) if y_h.any() else np.nan,
    }


# Model ranking: average rank over several criteria, not MAE alone.
# (column, higher_is_better)
RANK_CRITERIA = [("MAE", False), ("RMSE", False), ("MAE_high", False), ("abs_bias", False), ("PR_AUC", True)]


def walk_forward(rows, last_obs):
    cutoffs = [last_obs - pd.Timedelta(days=10 + FOLD_STEP_DAYS * i) for i in range(N_FOLDS)][::-1]
    preds = []
    for c in cutoffs:
        train = rows[(rows["target_date"] <= c) & rows["y"].notna()]
        test = rows[(rows["origin"] == c) & rows["y"].notna()]
        assert train["target_date"].max() <= c < test["target_date"].min(), "leakage: train overlaps test"
        print(f"fold origin={c.date()}  train targets {train.target_date.min().date()}..{train.target_date.max().date()}"
              f" ({len(train)} rows)  test targets {test.target_date.min().date()}..{test.target_date.max().date()} ({len(test)} rows)")
        out = test[["sensor_id", "target_date", "h", "y"]].copy()
        out["cutoff"] = c
        for name, fn in MODELS.items():
            out[name] = fn(train, test)
        preds.append(out)
    return pd.concat(preds, ignore_index=True)


def score_table(cv):
    return pd.DataFrame({m: scores(cv["y"].to_numpy(), cv[m].to_numpy()) for m in MODELS}).T


def report(cv):
    """Select on the earlier folds; keep the latest fold as an untouched confirmation.

    With 14 candidates, picking the best on the same folds we report would
    overstate the winner. The confirmation fold is never used for selection.
    """
    last = cv["cutoff"].max()
    select, confirm = cv[cv["cutoff"] < last], cv[cv["cutoff"] == last]
    table = score_table(select)
    table["MAE_fold_std"] = select.groupby("cutoff").apply(
        lambda g: pd.Series({m: np.mean(np.abs(g.y - g[m])) for m in MODELS}), include_groups=False).std()
    table["abs_bias"] = table["bias"].abs()
    table["mean_rank"] = np.mean([table[c].rank(ascending=not hi) for c, hi in RANK_CRITERIA], axis=0)
    table["skill_vs_mean7%"] = 100 * (1 - table["MAE"] / table.loc["mean7", "MAE"])
    conf = score_table(confirm)
    for c in ("MAE", "RMSE", "bias", "MAE_high", "PR_AUC"):
        table[f"confirm_{c}"] = conf[c]
    table = table.sort_values(["mean_rank", "MAE"])
    pd.set_option("display.width", 250)
    print(f"\nSelection folds (origins before {last.date()}, {len(select)} rows), sorted by mean rank over "
          f"{[c for c, _ in RANK_CRITERIA]}; confirm_* = untouched fold at origin {last.date()}:")
    print("\n-- point-forecast error --")
    print(table[["mean_rank", "MAE", "RMSE", "MAPE%", "bias", "R2", "MAE_high", "bias_high",
                 "skill_vs_mean7%", "MAE_fold_std"]].round(3).to_string())
    print("\n-- hazardous alarm (SPEC rule: predicted >= 165) --")
    print(table[["haz_true", "recall", "precision", "F1", "false_alarms", "FA_rate", "PR_AUC"]].round(3).to_string())
    print(f"(PR_AUC chance level = {select.y.ge(HAZARD_THRESHOLD).mean():.3f})")
    print("\n-- untouched confirmation fold --")
    print(table[[c for c in table.columns if c.startswith("confirm_")]].round(3).to_string())
    by_fold = cv.groupby("cutoff").apply(lambda g: pd.Series({m: np.mean(np.abs(g.y - g[m])) for m in MODELS}),
                                         include_groups=False)
    print("\nMAE by fold:")
    print(by_fold.round(1).to_string())
    by_h = cv.groupby("h").apply(lambda g: pd.Series({m: np.mean(np.abs(g.y - g[m])) for m in MODELS}),
                                 include_groups=False)
    print("\nMAE by horizon (all folds):")
    print(by_h.round(1).to_string())
    return table


# ---------- final predictions ----------

def write_predictions(rows, last_obs, model_name, alarm_threshold=HAZARD_THRESHOLD):
    holdout = pd.read_csv(ROOT / "holdout" / "holdout_inputs.csv", parse_dates=["target_date"])
    train = rows[(rows["target_date"] <= last_obs) & rows["y"].notna()]
    test = rows[rows["origin"] == last_obs].copy()
    test["predicted_pm25"] = np.clip(MODELS[model_name](train, test), 0, None)

    sub = holdout[["sensor_id", "target_date"]].merge(
        test[["sensor_id", "target_date", "predicted_pm25"]], on=["sensor_id", "target_date"], how="left", validate="1:1")
    sub["predicted_pm25"] = sub["predicted_pm25"].round(2)
    # Default = SPEC rule on the predicted value. A lower alarm threshold trades
    # false alarms for recall; see SPRINT1.md section 6 before changing it.
    sub["hazardous"] = (sub["predicted_pm25"] >= alarm_threshold).astype(int)
    sub["target_date"] = sub["target_date"].dt.strftime("%Y-%m-%d")

    template_cols = list(pd.read_csv(ROOT / "templates" / "predictions_template.csv").columns)
    sub = sub[template_cols]
    validate_submission(sub, holdout)
    sub.to_csv(ROOT / "predictions.csv", index=False)
    print(f"\nwrote predictions.csv with model={model_name}, alarm at predicted >= {alarm_threshold:g}: {len(sub)} rows, "
          f"{sub.hazardous.sum()} hazardous, PM2.5 range {sub.predicted_pm25.min():.1f}..{sub.predicted_pm25.max():.1f}")


def validate_submission(sub, holdout):
    """The SPEC section 4 gate. Fails loudly instead of writing a bad file."""
    assert list(sub.columns) == ["sensor_id", "target_date", "predicted_pm25", "hazardous"]
    assert len(sub) == len(holdout) and not sub.duplicated(["sensor_id", "target_date"]).any()
    expected = set(zip(holdout.sensor_id, holdout.target_date.dt.strftime("%Y-%m-%d")))
    assert set(zip(sub.sensor_id, sub.target_date)) == expected, "pairs differ from holdout_inputs.csv"
    assert sub.notna().all().all(), "missing cells"
    assert np.isfinite(sub.predicted_pm25).all() and (sub.predicted_pm25 >= 0).all()
    assert sub.hazardous.isin([0, 1]).all()
    assert sub.target_date.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), help="override the automatic choice")
    ap.add_argument("--target-weather", action="store_true",
                    help="also use weather ON the target day (after the origin). Off by default: past-only features.")
    ap.add_argument("--no-calendar", action="store_true", help="ablation: drop target weekday features")
    ap.add_argument("--no-spike", action="store_true", help="ablation: drop spike-history feature")
    ap.add_argument("--alarm-threshold", type=float, default=HAZARD_THRESHOLD,
                    help="flag hazardous when predicted PM2.5 >= this (default: SPEC value 165)")
    ap.add_argument("--tag", default="", help="suffix for outputs/ files, e.g. for ablation runs")
    args = ap.parse_args()
    if args.model:  # fast path: validate and fit only this model (+ the two baselines for comparison)
        for name in [m for m in MODELS if m not in {args.model, *BASELINES}]:
            del MODELS[name]
        print(f"single-model run: {args.model} (+ baselines {sorted(BASELINES)})")
    if not args.no_calendar:
        FEATURES.extend(CALENDAR_FEATURES)
    if not args.no_spike:
        FEATURES.extend(SPIKE_FEATURES)
    if args.target_weather:
        FEATURES.extend(TARGET_WEATHER_FEATURES)
    print(f"features ({len(FEATURES)}): calendar={not args.no_calendar} spike={not args.no_spike} "
          f"target_weather={args.target_weather}")

    daily, weather = load_daily(), load_weather()
    last_obs = daily["date"].max()
    holdout = pd.read_csv(ROOT / "holdout" / "holdout_inputs.csv", parse_dates=["target_date"])
    h_needed = (holdout["target_date"] - last_obs).dt.days
    print(f"last observation {last_obs.date()}; holdout needs horizons {h_needed.min()}..{h_needed.max()} days ahead")
    assert h_needed.min() >= 1, "holdout targets must be after the last observation"
    horizons = range(1, max(max(HORIZONS), int(h_needed.max())) + 1)  # follows the holdout file
    rows = build_rows(daily, weather, horizons=horizons)

    cv = walk_forward(rows, last_obs)
    table = report(cv)
    (ROOT / "outputs").mkdir(exist_ok=True)
    cv.to_csv(ROOT / "outputs" / f"walk_forward_predictions{args.tag}.csv", index=False)
    table.to_csv(ROOT / "outputs" / f"walk_forward_scores{args.tag}.csv")

    # Best mean rank across MAE, RMSE, MAE_high, |bias|, PR_AUC (table is sorted by it).
    best = args.model or table.drop(index=list(BASELINES)).index[0]
    b, m7 = table.loc[best], table.loc["mean7"]
    print(f"\nchosen model: {best} (mean rank {b.mean_rank:.1f})")
    for c in ("MAE", "RMSE", "MAE_high", "bias", "PR_AUC"):
        print(f"  {c:9s} model {b[c]:7.3f}   mean7 baseline {m7[c]:7.3f}   confirm: model {b.get('confirm_' + c, np.nan):7.3f}"
              f" / mean7 {m7.get('confirm_' + c, np.nan):7.3f}")
    if b.MAE >= table.loc[list(BASELINES), "MAE"].min():
        print("WARNING: model does not beat the baseline; consider --model mean7")
    if args.tag:
        print("tagged run: predictions.csv not written")
        return
    write_predictions(rows, last_obs, best, args.alarm_threshold)


if __name__ == "__main__":
    main()
