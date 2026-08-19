"""C8b: is the round-1 kill horizon-specific, and what does the LIVE shape pay?

Live shape today: 7 qualifiers, SPY above its 200d, slate concentrated in
staples/food. Also re-runs the placebo at every horizon and checks the
staples subset the map pointed at.
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
rk63 = pd.DataFrame({t: pct_rank(px[t], 63) for t in UNIV})
hi252 = pd.DataFrame({t: px[t].rolling(252, min_periods=200).max() for t in UNIV})
dd = pd.DataFrame({t: px[t] / hi252[t] - 1.0 for t in UNIV})
sma200 = pd.DataFrame({t: px[t].rolling(200).mean() for t in UNIV})
trig = ((rk63 >= 95) & (dd <= -0.15)).fillna(False)
spy_above = (spy > spy.rolling(200).mean()).reindex(dates)
n_fire = trig.sum(axis=1)

def excess(h):
    f = pd.DataFrame({t: fwd_lag(px[t], h, 1) for t in UNIV})
    return f.sub(fwd_lag(spy, h, 1).reindex(dates), axis=0)

def epi_vals(mask_df, e):
    ds, ns, vs = [], [], []
    for t in UNIV:
        valid = e[t].dropna().index
        idx = dates[mask_df[t].values & e[t].notna().values]
        if len(idx) == 0:
            continue
        for d in declusters(pd.DatetimeIndex(idx), 21, valid):
            ds.append(d); ns.append(t); vs.append(e.loc[d, t])
    return pd.DatetimeIndex(ds), np.array(ns), np.array(vs)

print("=== horizon sweep: pooled name-episodes, excess vs SPY ===")
rows = []
for h in (1, 2, 3, 5, 10, 21):
    e = excess(h)
    _, _, v = epi_vals(trig, e)
    allv = e.values.ravel(); allv = allv[~np.isnan(allv)]
    r = summarize(v, f"h={h} JOINT")
    r["ctl_all_pct"] = round(100*allv.mean(), 3)
    r["edge_pp"] = round(r["mean_pct"] - 100*allv.mean(), 3)
    rows.append(r)
    # deep-drawdown-alone control at the same horizon
    _, _, v2 = epi_vals((dd <= -0.15).fillna(False), e)
    r2 = summarize(v2, f"h={h} dd<=-15% ALONE")
    r2["ctl_all_pct"] = round(100*allv.mean(), 3)
    r2["edge_pp"] = round(r2["mean_pct"] - 100*allv.mean(), 3)
    rows.append(r2)
show(rows, "joint vs the drawdown gate alone")

print("\n=== alphabetical placebo at every horizon (day baskets, K=4) ===")
K = 4
trig_days = dates[trig.any(axis=1).values]
rows = []
for h in (1, 3, 5, 10):
    e = excess(h)
    deep, alpha, allq = [], [], []
    for d in trig_days:
        q = [t for t in UNIV if trig.loc[d, t] and not np.isnan(e.loc[d, t])]
        if not q:
            continue
        deep.append(np.mean([e.loc[d, t] for t in sorted(q, key=lambda x: dd.loc[d, x])[:K]]))
        alpha.append(np.mean([e.loc[d, t] for t in sorted(q)[:K]]))
        allq.append(np.mean([e.loc[d, t] for t in q]))
    rows += [summarize(np.array(deep), f"h={h} deepest-{K}"),
             summarize(np.array(alpha), f"h={h} ALPHABETICAL-{K}"),
             summarize(np.array(allq), f"h={h} all qualifiers")]
    print(f"  h={h}: deepest minus alphabetical = %+0.3fpp"
          % (100*(np.mean(deep)-np.mean(alpha))))
show(rows, "placebo ladder")

print("\n=== LIVE SHAPE: SPY above 200d, <=10 names firing, name above its own 200d ===")
live_shape = trig.copy()
above_own = (px[UNIV] > sma200).fillna(False)
mask = (spy_above.fillna(False).values[:, None]
        & (n_fire <= 10).values[:, None]
        & above_own.values)
live_shape = pd.DataFrame(trig.values & mask, index=dates, columns=UNIV)
for h in (3, 5, 10):
    e = excess(h)
    _, _, v = epi_vals(live_shape, e)
    allv = e.values.ravel(); allv = allv[~np.isnan(allv)]
    print("  h=%2d live-shape episodes N=%d mean %+0.3f%% hit %.1f%% "
          "edge vs all days %+0.3fpp" % (h, len(v), 100*v.mean(),
                                         100*(v > 0).mean(),
                                         100*(v.mean()-allv.mean())))
print("  today's slate above own 200d:",
      {t: bool(above_own.loc[dates[-1], t]) for t in UNIV if trig.loc[dates[-1], t]})
print("  names firing today: %d (historical median on trigger days %d)"
      % (n_fire.iloc[-1], int(n_fire[trig.any(axis=1).values].median())))

print("\n=== STAPLES/FOOD subset (today's slate class) ===")
STAP = [t for t in ["CPB", "GIS", "CAG", "K", "KHC", "SJM", "HRL", "MKC", "KO",
                    "PEP", "CL", "PG", "KMB", "CHD", "CLX", "MDLZ", "STZ", "TAP",
                    "ADM", "TSN", "SYY", "KR", "COST", "WMT", "MO", "PM"] if t in UNIV]
print("  staples names in universe:", len(STAP))
for h in (3, 5, 10):
    e = excess(h)
    sub = pd.DataFrame(False, index=dates, columns=UNIV)
    sub[STAP] = trig[STAP]
    _, _, v = epi_vals(sub, e)
    print("  h=%2d staples episodes N=%d mean %+0.3f%% hit %.1f%% sign p=%.3f"
          % (h, len(v), 100*v.mean(), 100*(v > 0).mean(),
             sign_test(int((v > 0).sum()), len(v))))
