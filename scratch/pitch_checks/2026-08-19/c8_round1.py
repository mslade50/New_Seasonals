"""C8 round 1: the base-breakout cross-section.

Trigger, per NAME per DAY (all PIT):
  rank63 = trailing-252d percentile of the 63d return >= 95
  AND close <= 0.85 * trailing-252d rolling max (>= 15% below the 52w high)

Kill order (placebo FIRST, per brief):
  0. denominator roll + tape over-selection diagnostics
  1. ALPHABETICAL PLACEBO: deepest-4 vs alphabetically-first-4 per trigger day
  2. pooled excess vs three controls
  3. gate attribution (each leg alone)
  4. cross-name heterogeneity (per-name means, permutation max-of-K)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import pandas as pd, numpy as np
import strategy_config as sc

UNIV = sorted(set(sc.LIQUID_PLUS_COMMODITIES))
px = close_panel(UNIV + ["SPY"])
spy = px["SPY"].dropna()
dates = px.index
print("panel", dates[0].date(), "->", dates[-1].date(), "names", px.shape[1])

RK, DD = 95.0, -0.15
H = 5

rk63 = pd.DataFrame({t: pct_rank(px[t], 63) for t in UNIV})
hi252 = pd.DataFrame({t: px[t].rolling(252, min_periods=200).max() for t in UNIV})
dd = pd.DataFrame({t: px[t] / hi252[t] - 1.0 for t in UNIV})
sma200 = pd.DataFrame({t: px[t].rolling(200).mean() for t in UNIV})

trig = (rk63 >= RK) & (dd <= DD)
trig = trig.fillna(False)
print("total name-days qualifying:", int(trig.values.sum()),
      " days with >=1 qualifier:", int(trig.any(axis=1).sum()))

# today's slate
today = dates[-1]
live = [t for t in UNIV if trig.loc[today, t]]
print("live slate", today.date(), ":", live)
print("  live rk63/dd:", {t: (round(rk63.loc[today, t], 1), round(100*dd.loc[today, t], 1))
                          for t in live})

# ---------- 0a. DENOMINATOR ROLL ----------
# the 63d return changes day to day by (today's move) - (the t-63 bar rolling off).
r63 = pd.DataFrame({t: px[t].pct_change(63) for t in UNIV})
own = pd.DataFrame({t: px[t].pct_change(1) for t in UNIV})
roll_off = pd.DataFrame({t: px[t].shift(63).pct_change(1) for t in UNIV})
m = trig.values
d63 = (r63 - r63.shift(1)).values
own_v, roll_v = own.values, roll_off.values
sel = m & ~np.isnan(own_v) & ~np.isnan(roll_v)
dom = (np.abs(roll_v) > np.abs(own_v))[sel]
print("\n0a. denominator roll: |t-63 roll-off| > |today's own move| on "
      f"{100*dom.mean():.1f}% of {sel.sum()} trigger name-days")
# magnitude gate: what 63d return does rank>=95 buy?
print("    63d return on trigger days: median %+0.1f%%, 10th pct %+0.1f%%; "
      "today's slate: %s" % (100*np.nanmedian(r63.values[sel]),
                             100*np.nanpercentile(r63.values[sel], 10),
                             {t: round(100*r63.loc[today, t], 1) for t in live}))

# ---------- 0b. TAPE OVER-SELECTION ----------
spy200 = spy.rolling(200).mean()
above = (spy > spy200).reindex(dates)
trig_days = dates[trig.any(axis=1).values]
base_days = dates[above.notna().values]
print("0b. SPY above its 200d on %.1f%% of trigger days vs base rate %.1f%%"
      % (100*above.reindex(trig_days).mean(), 100*above.loc[base_days].mean()))

# ---------- forward returns ----------
def fwd_panel(h):
    f = {}
    for t in UNIV:
        f[t] = fwd_lag(px[t], h, 1)
    f = pd.DataFrame(f)
    b = fwd_lag(spy, h, 1).reindex(dates)
    return f, b

fwd, bfwd = fwd_panel(H)
exc = fwd.sub(bfwd, axis=0)

# ---------- 1. ALPHABETICAL PLACEBO (run FIRST) ----------
K = 4
rows_deep, rows_alpha, rows_rank, rows_all = [], [], [], []
for d in trig_days:
    q = [t for t in UNIV if trig.loc[d, t] and not np.isnan(exc.loc[d, t])]
    if not q:
        continue
    deep = sorted(q, key=lambda t: dd.loc[d, t])[:K]          # deepest below 52wh
    rnk = sorted(q, key=lambda t: -rk63.loc[d, t])[:K]        # highest 63d rank
    alpha = sorted(q)[:K]                                     # ignorant rule
    rows_deep.append(np.mean([exc.loc[d, t] for t in deep]))
    rows_rank.append(np.mean([exc.loc[d, t] for t in rnk]))
    rows_alpha.append(np.mean([exc.loc[d, t] for t in alpha]))
    rows_all.append(np.mean([exc.loc[d, t] for t in q]))

show([summarize(np.array(rows_deep), f"deepest-{K} (signal-most)"),
      summarize(np.array(rows_rank), f"highest-rank-{K}"),
      summarize(np.array(rows_alpha), f"alphabetical-{K} (IGNORANT)"),
      summarize(np.array(rows_all), "all qualifiers, equal weight")],
     f"1. ALPHABETICAL PLACEBO, h={H} excess vs SPY, day-level baskets "
     f"(N days={len(rows_all)})")
print("   deepest minus alphabetical = %+0.3fpp   highest-rank minus alphabetical"
      " = %+0.3fpp" % (100*(np.mean(rows_deep)-np.mean(rows_alpha)),
                       100*(np.mean(rows_rank)-np.mean(rows_alpha))))

# ---------- 2. pooled name-episodes vs controls ----------
def episodes_for(mask_df, h):
    """Per-name declustered episodes; returns (dates, names, excess vals)."""
    f = pd.DataFrame({t: fwd_lag(px[t], h, 1) for t in UNIV})
    e = f.sub(fwd_lag(spy, h, 1).reindex(dates), axis=0)
    ds, ns, vs = [], [], []
    for t in UNIV:
        s = mask_df[t]
        valid = e[t].dropna().index
        idx = dates[s.values & e[t].notna().values]
        if len(idx) == 0:
            continue
        epi = declusters(pd.DatetimeIndex(idx), 21, valid)
        for d in epi:
            ds.append(d); ns.append(t); vs.append(e.loc[d, t])
    return pd.DatetimeIndex(ds), np.array(ns), np.array(vs)

ed, en, ev = episodes_for(trig, H)
print("\n2. pooled name-level episodes (min_gap 21td):", len(ev))
# controls
allf = exc.values.ravel()
allf = allf[~np.isnan(allf)]
span = (ed.min(), ed.max())
inspan = exc.loc[(dates >= span[0]) & (dates <= span[1])].values.ravel()
inspan = inspan[~np.isnan(inspan)]
# local +/-126td ex-trigger, same names
locv = []
for t in UNIV:
    s = trig[t]
    idx = dates[s.values]
    if len(idx) == 0:
        continue
    loc = local_control(exc[t].dropna().index, pd.DatetimeIndex(idx))
    locv.append(exc.loc[loc, t].values)
locv = np.concatenate(locv) if locv else np.array([])
locv = locv[~np.isnan(locv)]
show([summarize(ev, f"COND episodes (N={len(ev)})"),
      summarize(inspan, "CTRL-a all name-days, same span"),
      summarize(allf, "CTRL-b all name-days, full history"),
      summarize(locv, "CTRL-c local +/-126td ex-trigger")],
     f"2. pooled excess vs SPY, h={H}")
print("   episode-vs-all-days edge = %+0.3fpp" % (100*(ev.mean()-allf.mean())))
print("   " + cluster_note(ed, ev, k=3))

# ---------- 3. gate attribution ----------
print("\n3. GATE ATTRIBUTION (name-episodes, h=%d, excess vs SPY)" % H)
cells = {
    "JOINT rk63>=95 & dd<=-15%": trig,
    "rk63>=95 ALONE": (rk63 >= RK).fillna(False),
    "rk63>=95 & dd>-15% (shallow)": ((rk63 >= RK) & (dd > DD)).fillna(False),
    "dd<=-15% ALONE": (dd <= DD).fillna(False),
    "dd<=-15% & rk63<95": ((dd <= DD) & (rk63 < RK)).fillna(False),
}
rows = []
for lbl, mk in cells.items():
    _, _, v = episodes_for(mk, H)
    rows.append(summarize(v, f"{lbl}"))
show(rows, "gate cells")

# ---------- 4. cross-name heterogeneity ----------
print("\n4. CROSS-NAME HETEROGENEITY")
per = pd.DataFrame({"name": en, "v": ev}).groupby("name")["v"]
tab = pd.DataFrame({"n": per.count(), "mean_pct": 100*per.mean(),
                    "sd_pct": 100*per.std()})
tab = tab[tab["n"] >= 5].sort_values("mean_pct", ascending=False)
print("  names with >=5 episodes:", len(tab))
print("  positive-mean share: %.1f%%" % (100*(tab["mean_pct"] > 0).mean()))
print(tab.head(8).round(3).to_string())
print("  ...")
print(tab.tail(5).round(3).to_string())
# Cochran Q on name means
w = tab["n"] / (tab["sd_pct"] ** 2).replace(0, np.nan)
mu = (w * tab["mean_pct"]).sum() / w.sum()
Q = float((w * (tab["mean_pct"] - mu) ** 2).sum())
print("  Cochran Q = %.1f on df=%d (pooled mean %.3f%%)" % (Q, len(tab)-1, mu))
# permutation max-of-K
rng = np.random.default_rng(42)
obs_max = tab["mean_pct"].max()
sims = []
for _ in range(2000):
    p = rng.permutation(ev)
    dfp = pd.DataFrame({"name": en, "v": p}).groupby("name")["v"].agg(["count", "mean"])
    dfp = dfp[dfp["count"] >= 5]
    sims.append(100*dfp["mean"].max())
sims = np.array(sims)
print("  permutation P(max name mean >= observed %.3f%%) = %.3f"
      % (obs_max, (sims >= obs_max).mean()))
