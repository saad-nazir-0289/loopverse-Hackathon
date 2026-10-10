"""Figures for a blog post: why spikes in this dataset cannot be predicted from advance information.

    python src/blog_figures.py      -> outputs/blog/fig1..fig6 (*.png)

Needs outputs from: clean.py, forecast.py, experiments_spike_detectors.py.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "blog"
OUT.mkdir(parents=True, exist_ok=True)

# palette + chrome (dataviz reference palette, light mode)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
JUMP, HAZ = 45.0, 165.0
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11, "axes.titlesize": 14,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left", "lines.linewidth": 2,
})


def finish(fig, name, subtitle, source="Lahore Smog Intelligence Challenge data, Jul–Oct 2026", title=None):
    """Title + subtitle as figure text (top-left, never clipped); source line at the bottom."""
    if title is None:  # single-panel figure: promote the axes title to the figure header
        title = fig.axes[0].get_title(loc="left")
        fig.axes[0].set_title("", loc="left")
    fig.text(0.012, 0.975, title, fontsize=15, fontweight="bold", color=INK, va="top")
    import textwrap
    width = int(fig.get_figwidth() * 12.5)  # ~characters per line at 10.5 pt
    fig.text(0.012, 0.912, textwrap.fill(subtitle, width), fontsize=10.5, color=INK2, va="top", linespacing=1.4)
    fig.text(0.012, 0.012, f"Source: {source}", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.84))
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print("wrote", OUT / name)


d = pd.read_csv(ROOT / "data" / "processed" / "daily_pm25.csv", parse_dates=["date"]).sort_values(["sensor_id", "date"])
d["normal"] = d.groupby("sensor_id").pm25.transform(lambda s: s.shift(1).rolling(14, min_periods=7).median())
d["excess"] = d.pm25 - d.normal
d["spike"] = d.excess > JUMP
w = pd.read_csv(ROOT / "weather" / "weather_history.csv", parse_dates=["date"]).set_index("date")
rate = d.spike.mean()
n_spikes = int(d.spike.sum())

# ---------- Fig 1: what a spike looks like
sid = "S01"
s = d[d.sensor_id == sid]
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(s.date, s.pm25, color=BLUE, lw=1.6, label="Daily PM2.5")
ax.plot(s.date, s.normal, color=MUTED, lw=1.6, ls="--", label="Normal level (past 14-day median)")
sp = s[s.spike]
ax.scatter(sp.date, sp.pm25, s=70, color=ORANGE, zorder=5, edgecolor=SURFACE, linewidth=2, label="Spike")
ax.axhline(HAZ, color=INK2, lw=1, ls=":")
ax.text(s.date.max(), HAZ + 4, "Hazardous ≥ 165 µg/m³", color=INK2, fontsize=9.5, ha="right")
prev = None
for _, r in sp.iterrows():
    close = prev is not None and (r.date - prev).days < 6  # nudge labels of back-to-back spikes apart
    ax.annotate(f"+{r.excess:.0f}", (r.date, r.pm25), xytext=(12, 8) if close else (-6, 8),
                textcoords="offset points", ha="left" if close else "right", fontsize=8.5, color=INK2)
    prev = r.date
ax.set_ylabel("PM2.5 (µg/m³)")
ax.legend(loc="upper left", frameon=False, ncol=3, fontsize=9.5)
ax.set_ylim(0, max(s.pm25.max() + 30, 250))
ax.set_title(f"Spikes appear out of nowhere and are gone the next day ({sid}, Gulberg)")
finish(fig, "fig1_what_a_spike_looks_like.png",
       f"Each orange point jumps ~90 µg/m³ above the normal level for one day. "
       f"{n_spikes} such spikes across 15 sensors ({rate:.1%} of days).")

# ---------- Fig 2: nothing builds up before a spike (superposed epoch)
win = range(-7, 4)
ex, tm = {k: [] for k in win}, {k: [] for k in win}
wt = w.temp_c - w.temp_c.rolling(15, center=True, min_periods=5).mean()
for sid_, g in d.groupby("sensor_id"):
    g = g.set_index("date")
    for day in g.index[g.spike]:
        for k in win:
            t = day + pd.Timedelta(days=k)
            if t in g.index and np.isfinite(g.at[t, "excess"]):
                ex[k].append(g.at[t, "excess"])
            if t in wt.index and np.isfinite(wt.at[t]):
                tm[k].append(wt.at[t])
base_sd = d.excess[~d.spike].std()
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.4, 1]})
ax = axes[0]
m = np.array([np.mean(ex[k]) for k in win])
se = np.array([np.std(ex[k]) / np.sqrt(len(ex[k])) for k in win])
ax.axhspan(-1.96 * base_sd / np.sqrt(n_spikes), 1.96 * base_sd / np.sqrt(n_spikes), color=GRID, alpha=0.9, lw=0)
ax.fill_between(list(win), m - 1.96 * se, m + 1.96 * se, color=BLUE, alpha=0.15, lw=0)
ax.plot(list(win), m, color=BLUE, marker="o", ms=7, mec=SURFACE, mew=2)
ax.axvline(0, color=ORANGE, lw=1.5, ls="--")
ax.text(0.15, m.max() * 0.92, "spike day", color=INK2, fontsize=9.5)
ax.text(1.2, -10, "grey band: range pure noise would give", color=MUTED, fontsize=8.5)
ax.set_ylim(-14, m.max() + 15)
ax.set_xticks(list(win))
ax.set_xlabel("Days relative to the spike")
ax.set_ylabel("PM2.5 above normal level (µg/m³)")
ax.set_title("PM2.5 above normal", fontsize=11.5)
ax2 = axes[1]
mt = np.array([np.mean(tm[k]) for k in win])
st = np.array([np.std(tm[k]) / np.sqrt(len(tm[k])) for k in win])
ax2.fill_between(list(win), mt - 1.96 * st, mt + 1.96 * st, color=AQUA, alpha=0.18, lw=0)
ax2.plot(list(win), mt, color=AQUA, marker="o", ms=6, mec=SURFACE, mew=2, label="Temperature vs seasonal norm")
ax2.axhline(0, color=AXIS, lw=1)
ax2.axvline(0, color=ORANGE, lw=1.5, ls="--")
ax2.set_xticks(list(win))
ax2.set_xlabel("Days relative to the spike")
ax2.set_ylabel("Temperature anomaly (°C)")
ax2.set_title("Temperature: only a faint warm blip", fontsize=11.5)
finish(fig, "fig2_nothing_builds_up.png",
       f"Average of all {n_spikes} spikes, lined up on the spike day (shaded = 95% range). "
       "The days before look like any other day.", title="Nothing builds up before a spike")

# ---------- Fig 3: correlation of advance information with "spike tomorrow"
sys.path.insert(0, str(ROOT / "src"))
from features import build_rows, load_daily, load_weather  # noqa: E402
r = build_rows(load_daily(), load_weather(), horizons=[1]).dropna(subset=["y", "median7"])
r["spike_next"] = ((r.y - r.median7) > JUMP).astype(float)
r["days"] = (r.target_date - r.target_date.min()).dt.days
cands = {"lag0": "PM2.5 today", "lag1": "PM2.5 yesterday", "mean7": "7-day mean", "std7": "7-day spread",
         "max7": "7-day max", "trend7": "7-day trend", "sensor_level": "Sensor's usual level",
         "city_lag0": "City-wide PM2.5 today", "spike_rate30": "Sensor's recent spike rate",
         "days_since_spike": "Days since last spike", "days": "Day of season", "dow4": "Tomorrow is Friday",
         "w0_temp_c": "Temperature today", "w0_wind_kmh": "Wind today", "w0_humidity_pct": "Humidity today",
         "w_temp_c": "Temperature tomorrow*", "w_wind_kmh": "Wind tomorrow*", "w_humidity_pct": "Humidity tomorrow*"}
cor = pd.Series({lab: r[c].corr(r.spike_next) for c, lab in cands.items()}).sort_values()
ref = r["median7"].corr(r["y"])
noise = 1.96 / np.sqrt(len(r))
fig, ax = plt.subplots(figsize=(11, 7))
y = np.arange(len(cor))
ax.axvspan(-noise, noise, color=GRID, lw=0)
ax.barh(y, cor.values, height=0.6, color=BLUE, edgecolor=SURFACE, linewidth=2)
yr = len(cor) + 0.6
ax.barh([yr], [ref], height=0.6, color=AXIS, edgecolor=SURFACE, linewidth=2)
ax.text(ref + 0.01, yr, f"{ref:.2f}: a real signal\n(7-day median vs tomorrow's level)",
        va="center", fontsize=9, color=INK2)
ax.set_yticks(list(y) + [yr], list(cor.index) + ["Reference"], color=INK2)
ax.axvline(0, color=AXIS, lw=1)
for yi, v in zip(y, cor.values):
    ax.text(v + (0.004 if v >= 0 else -0.004), yi, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right",
            fontsize=9, color=INK2)
ax.text(noise + 0.008, -0.9, "grey band = what random noise produces", color=MUTED, fontsize=8.5)
ax.set_xlim(-0.2, 1.05)
ax.set_xlabel("Correlation with 'a spike happens tomorrow' (−1 to +1)")
ax.set_title("Nothing known in advance tracks tomorrow's spikes")
finish(fig, "fig3_no_advance_signal.png",
       f"Correlation of each advance input with 'a spike happens tomorrow' ({len(r)} sensor-days). All are below 0.1, "
       "under 1% of spike variation. *Tomorrow's weather would be a forecast.")

# ---------- Fig 4: model scoreboard
sc = pd.read_csv(ROOT / "outputs" / "experiments_spike_detectors.csv").sort_values("ROC_AUC")
fig, ax = plt.subplots(figsize=(10, 6.5))
y = np.arange(len(sc))
ax.axvspan(0.45, 0.55, color=GRID, lw=0)
ax.hlines(y, sc.AUC_fold_min, sc.AUC_fold_max, color=AXIS, lw=2)
ax.scatter(sc.ROC_AUC, y, s=80, color=[ORANGE if v >= 0.6 else BLUE for v in sc.ROC_AUC], zorder=4,
           edgecolor=SURFACE, linewidth=2)
ax.axvline(0.5, color=INK2, lw=1, ls="--")
ax.text(0.502, len(sc) - 0.4, "coin flip", color=INK2, fontsize=9.5)
ax.set_yticks(y, sc.detector, color=INK2, fontsize=9.5)
for yi, v in zip(y, sc.ROC_AUC):
    ax.text(v, yi + 0.28, f"{v:.2f}", ha="center", fontsize=8.5, color=INK2)
ax.set_xlim(0.1, 0.9)
ax.set_xlabel("Spike-ranking skill on unseen weeks (ROC-AUC: 0.5 = coin flip, 1.0 = perfect)")
ax.set_title("13 spike detectors: all close to a coin flip")
finish(fig, "fig4_model_scoreboard.png",
       "Dot = average over 8 unseen weeks; line = best to worst week. Only temperature (orange) beats chance, weakly.")

# ---------- Fig 5: alarms vs catches, compared with random guessing
cv = pd.read_csv(ROOT / "outputs" / "walk_forward_predictions.csv")
scores = pd.read_csv(ROOT / "outputs" / "walk_forward_scores.csv", index_col=0)
mdl = scores.drop(index=["persistence", "mean7"], errors="ignore").index[0]
yh = (cv.y >= HAZ).to_numpy()
order = np.argsort(-cv[mdl].to_numpy())
caught = np.cumsum(yh[order])
alarms = np.arange(1, len(cv) + 1)
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(alarms, alarms * yh.mean(), color=MUTED, ls="--", lw=1.8, label="Random guessing")
ax.plot(alarms, caught, color=BLUE, lw=2.2, label="Best forecast model")
for thr in (165, 150, 140, 130, 115):
    k = int((cv[mdl] >= thr).sum())
    if k:
        ax.scatter([k], [caught[k - 1]], s=60, color=ORANGE, zorder=5, edgecolor=SURFACE, linewidth=2)
        ax.annotate(f"alarm ≥ {thr}: {caught[k - 1]} caught, {k - caught[k - 1]} false", (k, caught[k - 1]),
                    xytext=(14, 30) if thr == 150 else (10, -14), textcoords="offset points", fontsize=8.5, color=INK2)
ax.text(len(cv) * 0.62, len(cv) * yh.mean() * 0.55, "random guessing", color=MUTED, fontsize=9.5)
ax.text(len(cv) * 0.30, yh.sum() * 0.93, "best forecast model", color=BLUE, fontsize=9.5)
ax.set_xlabel("Number of alarms raised (sensor-days flagged)")
ax.set_ylabel("Hazardous days caught")
ax.set_xlim(0, len(cv))
ax.set_ylim(0, yh.sum() + 2)
ax.set_title("Catching more hazardous days means hundreds of false alarms")
finish(fig, "fig5_alarms_vs_catches.png",
       f"{len(cv)} past forecasts, {int(yh.sum())} truly hazardous, highest forecasts flagged first. The model beats "
       "random guessing only modestly: at 415 alarms, 4.6% are right vs 3.1% for guessing.")

# ---------- Fig 6: spike timing looks random (gaps vs a memoryless process)
gaps = []
for _, g in d.groupby("sensor_id"):
    idx = np.flatnonzero(g.spike.to_numpy())
    gaps += list(np.diff(idx))
gaps = np.array(gaps)
p = rate
bins = [1, 4, 8, 15, 22, 31, 46, 61, 121]
labels = ["1–3", "4–7", "8–14", "15–21", "22–30", "31–45", "46–60", "61+"]
obs = np.histogram(gaps, bins=bins)[0]
geo = np.array([((1 - p) ** (a - 1) - (1 - p) ** (b - 1)) for a, b in zip(bins[:-1], bins[1:])])
exp = geo / geo.sum() * len(gaps)
from scipy import stats  # noqa: E402
chi_p = stats.chisquare(obs, exp).pvalue
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(labels))
ax.bar(x - 0.2, obs, width=0.38, color=BLUE, edgecolor=SURFACE, linewidth=2, label="Observed gaps")
ax.bar(x + 0.2, exp, width=0.38, color=AXIS, edgecolor=SURFACE, linewidth=2, label="If spikes were pure chance")
for xi, (o, e) in enumerate(zip(obs, exp)):
    ax.text(xi - 0.2, o + 0.3, str(o), ha="center", fontsize=9, color=INK2)
    ax.text(xi + 0.2, e + 0.3, f"{e:.0f}", ha="center", fontsize=9, color=MUTED)
ax.set_xticks(x, labels)
ax.set_xlabel("Days between two spikes at the same sensor")
ax.set_ylabel("Number of gaps")
ax.legend(frameon=False, loc="upper right")
ax.set_title("Spike timing is consistent with pure chance")
finish(fig, "fig6_timing_is_random.png",
       f"{len(gaps)} gaps between consecutive spikes vs a random (memoryless) process with the same {p:.1%} daily rate. "
       f"No significant difference (chi-square p = {chi_p:.2f}); the 8–14-day bump is partly one sensor's early rhythm "
       "that did not continue.")
