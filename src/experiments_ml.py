"""Random Forest, XGBoost and LSTM vs the current Ridge model, with honest tuning.

Usage:  python src/experiments_ml.py      -> outputs/experiments_ml_*.csv

Protocol (fixed before running), same 11 weekly walk-forward folds as experiments.py:
- folds 1-3   warm-up (not scored)
- folds 4-7   TUNE: hyperparameters are chosen here, by MAE
- folds 8-11  TEST: the chosen setting of each model is scored here, never used for choosing
"""
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

import experiments as E
from features import WEATHER_COLS, build_rows, load_daily, load_weather

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
FEATS = E.BASE
HORIZON = 10
REF = "ridge (current)"


def rf(min_leaf, max_feat):
    def fit_predict(train, test):
        tr = train.dropna(subset=["y", "mean7"])
        m = make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(
            n_estimators=300, min_samples_leaf=min_leaf, max_features=max_feat, random_state=0, n_jobs=4))
        m.fit(tr[FEATS], tr.y - tr.mean7)
        return test.mean7.fillna(test.lag0).to_numpy() + m.predict(test[FEATS])
    return fit_predict


def xgb(depth, n_est, min_child, objective="reg:squarederror"):
    def fit_predict(train, test):
        tr = train.dropna(subset=["y", "mean7"])
        m = XGBRegressor(n_estimators=n_est, learning_rate=0.03, max_depth=depth, min_child_weight=min_child,
                         subsample=0.8, colsample_bytree=0.8, reg_lambda=10, objective=objective,
                         huber_slope=30.0, random_state=0, n_jobs=4)
        m.fit(tr[FEATS], tr.y - tr.mean7)
        return test.mean7.fillna(test.lag0).to_numpy() + m.predict(test[FEATS])
    return fit_predict


# ---------- LSTM: one sequence per (sensor, origin) -> 10 horizons at once ----------

class SeqData:
    """Daily panel as arrays: pm25 (filled from the past only), weather, weekday."""

    def __init__(self, daily, weather, window):
        wide = daily.pivot(index="date", columns="sensor_id", values="pm25").sort_index()
        self.dates = wide.index
        self.sensors = list(wide.columns)
        # past-only fill; a gap at the very start becomes 1.0 (=100 ug/m3), never a later reading
        self.pm = (wide.ffill() / 100.0).fillna(1.0).to_numpy()
        # fixed scaling constants: statistics over all dates would include holdout weather
        w = weather.reindex(self.dates)[WEATHER_COLS]
        self.w = (w / np.array([30.0, 100.0, 10.0])).fillna(0).to_numpy()
        self.dow = np.eye(7)[self.dates.dayofweek]
        self.window = window
        self.raw = wide.to_numpy() / 100.0  # targets: real readings only (NaN = unknown)

    def sample(self, s, t):
        """Input window ending at origin index t (inclusive) for sensor s, and 10 target residuals."""
        L = self.window
        x = np.column_stack([self.pm[t - L + 1:t + 1, s], self.w[t - L + 1:t + 1], self.dow[t - L + 1:t + 1]])
        level = np.nanmean(self.pm[t - 6:t + 1, s])  # mean7, same base as the tabular models
        # weekday of each target day, so the network knows which day it is forecasting
        tdow = np.eye(7)[(self.dates[t].dayofweek + np.arange(1, HORIZON + 1)) % 7]  # calendar, known in advance
        y = np.full(HORIZON, np.nan)
        end = min(t + HORIZON, len(self.dates) - 1)
        y[:end - t] = self.raw[t + 1:end + 1, s] - level
        return x - np.r_[level, np.zeros(x.shape[1] - 1)], tdow.ravel(), level, y


def lstm(hidden, epochs, window=28, seeds=(0, 1, 2)):
    import torch
    from torch import nn
    torch.set_num_threads(4)

    class Net(nn.Module):
        def __init__(self, n_in, n_sensors):
            super().__init__()
            self.emb = nn.Embedding(n_sensors, 4)
            self.rnn = nn.LSTM(n_in, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden + 4 + 7 * HORIZON, 64), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(64, HORIZON))

        def forward(self, x, sid, tdow):
            _, (h, _) = self.rnn(x)
            return self.head(torch.cat([h[-1], self.emb(sid), tdow], 1))

    def fit_predict(train, test, _cache={}):
        sd = _cache.get("sd")
        if sd is None:
            sd = _cache["sd"] = SeqData(load_daily(), load_weather(), window)
        cutoff = test.origin.iloc[0]
        t_cut = sd.dates.get_loc(cutoff)
        X, S, D, Y = [], [], [], []
        for s in range(len(sd.sensors)):
            for t in range(window - 1, t_cut):
                x, tdow, _, y = sd.sample(s, t)
                # keep only targets dated <= cutoff (no future in training)
                y = y.copy()
                y[np.arange(t + 1, t + 1 + HORIZON) > t_cut] = np.nan
                if np.isnan(y).all():
                    continue
                X.append(x); S.append(s); D.append(tdow); Y.append(y)
        X, S, D, Y = (torch.tensor(np.array(a), dtype=torch.float32) for a in (X, S, D, Y))
        S = S.long()
        mask = ~torch.isnan(Y)
        Y = torch.nan_to_num(Y)
        Xt, St, Dt, levels = [], [], [], []
        for s in range(len(sd.sensors)):
            x, tdow, level, _ = sd.sample(s, t_cut)
            Xt.append(x); St.append(s); Dt.append(tdow); levels.append(level)
        Xt, Dt = (torch.tensor(np.array(a), dtype=torch.float32) for a in (Xt, Dt))
        St = torch.tensor(St)
        preds = []
        for seed in seeds:
            torch.manual_seed(seed)
            net = Net(X.shape[2], len(sd.sensors))
            opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
            loss_fn = nn.HuberLoss(delta=0.3, reduction="none")  # robust to spikes (0.3 = 30 ug/m3)
            for _ in range(epochs):
                net.train()
                perm = torch.randperm(len(X))
                for i in range(0, len(X), 128):
                    b = perm[i:i + 128]
                    opt.zero_grad()
                    out = net(X[b], S[b], D[b])
                    loss = (loss_fn(out, Y[b]) * mask[b]).sum() / mask[b].sum()
                    loss.backward()
                    opt.step()
            net.eval()
            with torch.no_grad():
                preds.append(net(Xt, St, Dt).numpy())
        p = (np.mean(preds, 0) + np.array(levels)[:, None]) * 100.0  # (sensor, horizon)
        lookup = {(sd.sensors[s], h + 1): p[s, h] for s in range(len(sd.sensors)) for h in range(HORIZON)}
        return np.array([lookup[(r.sensor_id, r.h)] for r in test.itertuples()])
    return fit_predict


def candidates(have_torch):
    c = {"ridge (current)": ("Ridge", E.residual_ridge(FEATS, "mean7"))}
    for leaf, mf in itertools.product([10, 30, 80], [0.3, 0.6]):
        c[f"rf leaf={leaf} feat={mf}"] = ("RandomForest", rf(leaf, mf))
    for d, n, mc in itertools.product([2, 3, 5], [200, 600], [20, 80]):
        c[f"xgb depth={d} n={n} child={mc}"] = ("XGBoost", xgb(d, n, mc))
    c["xgb pseudo-huber depth=3 n=600 child=80"] = ("XGBoost", xgb(3, 600, 80, "reg:pseudohubererror"))
    if have_torch:
        for hdim, ep in itertools.product([16, 32], [40, 120]):
            c[f"lstm hidden={hdim} epochs={ep}"] = ("LSTM", lstm(hdim, ep))
    return c


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="Ridge,RandomForest,XGBoost,LSTM", help="comma list to run")
    ap.add_argument("--tag", default="", help="suffix for output files")
    args = ap.parse_args()
    fams = set(args.families.split(","))
    try:
        import torch  # noqa: F401
        have_torch = True
    except ImportError:
        have_torch = False
        print("torch not installed: skipping LSTM")
    daily, weather = load_daily(), load_weather()
    rows = build_rows(daily, weather)
    cands = {k: v for k, v in candidates(have_torch).items() if v[0] in fams}
    cands.setdefault(REF, ("Ridge", E.residual_ridge(FEATS, "mean7")))
    cv = E.run({k: v[1] for k, v in cands.items()}, rows, daily.date.max())
    cv.to_csv(ROOT / "outputs" / f"experiments_ml_predictions{args.tag}.csv", index=False)

    cut = sorted(cv.cutoff.unique())
    tune, test = cv[cv.cutoff.isin(cut[3:7])], cv[cv.cutoff.isin(cut[7:])]
    rows_out = []
    for k, (fam, _) in cands.items():
        et, es = tune[k] - tune.y, test[k] - test.y
        rows_out.append({"family": fam, "config": k, "tune_MAE": et.abs().mean(), "test_MAE": es.abs().mean(),
                         "test_RMSE": np.sqrt((es ** 2).mean()), "test_bias": es.mean(),
                         "test_folds_beating_ridge": int(sum(
                             (g[k] - g.y).abs().mean() < (g[REF] - g.y).abs().mean()
                             for _, g in test.groupby("cutoff")))})
    allr = pd.DataFrame(rows_out)
    allr.to_csv(ROOT / "outputs" / f"experiments_ml_all_configs{args.tag}.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"\nAll configs (tune = folds 4-7, {len(tune)} rows; test = folds 8-11, {len(test)} rows):")
    print(allr.sort_values("tune_MAE").round(3).to_string(index=False))
    best = allr.loc[allr.groupby("family").tune_MAE.idxmin()].sort_values("test_MAE")
    print("\nBest config per family (chosen on TUNE folds), scored on untouched TEST folds:")
    print(best.round(3).to_string(index=False))
    best.to_csv(ROOT / "outputs" / f"experiments_ml_best{args.tag}.csv", index=False)


if __name__ == "__main__":
    main()
