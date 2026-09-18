"""A11 round 2. Round 1 left exactly one cell with a clean record: TLT h=5,
8-1, sign p 0.0195, mean +0.356%, edge +0.273pp over all days. Every other
cell of the 16 is noise or negative (69% negative, median edge -0.223pp,
which reproduces the registry's own analogue reference class). So TLT h=5 is
the DEFENDED cell and it gets the permutation.

  1. permutation against the DEFENDED cell: 500 random anchor states, same
     k=25 / 21td-decluster machinery, distribution of the TLT h=5 edge and of
     the best-of-16-grid |edge|. This charges the cell for the grid it was
     picked out of, which is the 2026-08-12 registry rule.
  2. gate attribution: what does the ANALOGUE gate buy over its obvious
     parents -- TLT at a 252d level floor, ^TNX at a 252d level ceiling -- and
     what does the DISCARDED COMPLEMENT (parent days that are NOT neighbours)
     pay?
  3. per-episode table and concentration for TLT h=5.
  4. horizon neighbours: h=4 and h=6 either side of the winning h=5.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

TK = ["SPY", "TLT", "GLD", "DBC", "^VIX", "^TNX"]
VEH = ["SPY", "TLT", "GLD", "DBC"]
HS = (1, 3, 5, 10)
K_MAIN, GAP, LAG = 25, 21, 1

px_d = load_prices(TK)
spy_idx = px_d["SPY"]["Close"].index
panel = pd.DataFrame({t: px_d[t]["Close"].reindex(spy_idx) for t in TK})


def lvl_rank(s, lookback=252):
    return rolling_on_valid(s, lambda x: x.rolling(lookback).rank(pct=True) * 100.0)


feat = pd.DataFrame(index=spy_idx)
feat["spy21"] = panel["SPY"] / panel["SPY"].shift(21) - 1.0
feat["vix"] = panel["^VIX"]
feat["tnx_r"] = lvl_rank(panel["^TNX"])
feat["dbc_r"] = lvl_rank(panel["DBC"])
feat["tlt_r"] = lvl_rank(panel["TLT"])
feat = feat.dropna()
z = (feat - feat.mean()) / feat.std(ddof=0)
TODAY = feat.index[-1]

fwd = {(v, h): fwd_lag(panel[v], h, LAG) for v in VEH for h in HS}
pool_ok = np.ones(len(feat), dtype=bool)
for v in VEH:
    pool_ok &= fwd[(v, 10)].reindex(feat.index).notna().values
pool = feat.index[pool_ok]
Zp = z.loc[pool].values
pos_map = pd.Series(range(len(spy_idx)), index=spy_idx)
pool_pos = pos_map.loc[pool].values


def decl_fast(order_positions, gap=GAP):
    keep, last = [], -10 ** 9
    for i in order_positions:
        p = pool_pos[i]
        if p - last >= gap:
            keep.append(i)
            last = p
    return keep


def nn_episodes(target_z, k=K_MAIN):
    d = np.sqrt(((Zp - target_z) ** 2).sum(axis=1))
    top = np.argsort(d)[:k]
    top = top[np.argsort(pool_pos[top])]      # chronological before declustering
    return pool[decl_fast(top)]


base_edge = {}
for v in VEH:
    for h in HS:
        r = fwd[(v, h)]
        base_edge[(v, h)] = r.dropna().mean()


def grid_edges(epi):
    out = {}
    for v in VEH:
        for h in HS:
            r = fwd[(v, h)]
            e = pd.DatetimeIndex(epi).intersection(r.dropna().index)
            out[(v, h)] = 100 * (r.loc[e].mean() - base_edge[(v, h)]) if len(e) else np.nan
    return out


live_epi = nn_episodes(z.loc[TODAY].values)
live_grid = grid_edges(live_epi)
obs_tlt5 = live_grid[("TLT", 5)]
obs_max = max(abs(x) for x in live_grid.values())
print("live episodes (n=%d):" % len(live_epi), ", ".join(str(d.date()) for d in live_epi))
print("DEFENDED cell TLT h=5 edge = %+.3fpp ; best-of-16 |edge| = %.3fpp"
      % (obs_tlt5, obs_max))

print("\n=== 1. PERMUTATION: 500 random anchor states through the same machinery ===")
rng = np.random.default_rng(7)
picks = rng.choice(len(pool), size=500, replace=False)
tlt5, maxes, negshare = [], [], []
for i in picks:
    epi = nn_episodes(Zp[i])
    g = grid_edges(epi)
    tlt5.append(g[("TLT", 5)])
    maxes.append(max(abs(x) for x in g.values()))
    negshare.append(np.mean([x < 0 for x in g.values()]))
tlt5 = np.array(tlt5, dtype=float)
maxes = np.array(maxes, dtype=float)
print("  TLT h=5 edge across random anchors: median %+.3fpp, sd %.3f, "
      "P(edge >= live %+.3f) = %.3f"
      % (np.nanmedian(tlt5), np.nanstd(tlt5), obs_tlt5, float(np.nanmean(tlt5 >= obs_tlt5))))
print("  best-of-16 |edge| across random anchors: median %.3fpp, "
      "P(max |edge| >= live %.3f) = %.3f"
      % (np.nanmedian(maxes), obs_max, float(np.nanmean(maxes >= obs_max))))
print("  share of the 16 cells negative, across random anchors: median %.0f%% "
      "(live %.0f%%) -- the registry reference class"
      % (100 * np.nanmedian(negshare),
         100 * np.mean([x < 0 for x in live_grid.values()])))
# sign-record permutation too: the 8-1 record is the headline, so charge it
r5 = fwd[("TLT", 5)]
recs = []
for i in picks:
    epi = nn_episodes(Zp[i])
    e = pd.DatetimeIndex(epi).intersection(r5.dropna().index)
    if len(e) >= 5:
        recs.append((r5.loc[e] > 0).mean())
recs = np.array(recs)
print("  hit rate of the TLT h=5 cell across random anchors: median %.1f%%, "
      "P(hit >= live 88.9%%) = %.3f" % (100 * np.median(recs), float((recs >= 8 / 9).mean())))

print("\n=== 2. GATE ATTRIBUTION: what does the ANALOGUE buy over its parents? ===")
tlt_r = feat["tlt_r"]
tnx_r = feat["tnx_r"]
dbc_r = feat["dbc_r"]
r5v = r5.dropna()


def cellstat(dates, label, gap=GAP):
    d = pd.DatetimeIndex(dates).intersection(r5v.index)
    if len(d) == 0:
        return {"label": label, "n": 0}
    d = declusters(d, gap, spy_idx)
    v = r5v.loc[d].values
    w = int((v > 0).sum())
    return {"label": label, "n": len(v), "mean_pct": round(100 * v.mean(), 3),
            "edge_pct": round(100 * (v.mean() - r5v.mean()), 3),
            "hit": round(100 * w / len(v), 1), "record": "%d-%d" % (w, len(v) - w),
            "sign_p": round(sign_test(w, len(v)), 4),
            "t": round(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 2),
            "worst_pct": round(100 * v.min(), 2)}


par_tlt = feat.index[tlt_r.values <= 2.0]
par_tnx = feat.index[tnx_r.values >= 98.0]
par_both = feat.index[(tlt_r.values <= 2.0) & (tnx_r.values >= 98.0)]
par_trip = feat.index[(tlt_r.values <= 2.0) & (tnx_r.values >= 98.0) & (dbc_r.values >= 98.0)]
rows = [cellstat(live_epi, "ANALOGUE k=25 (DEFENDED)"),
        cellstat(par_tlt, "parent: TLT 252d level rank <= 2"),
        cellstat(par_tnx, "parent: ^TNX 252d level rank >= 98"),
        cellstat(par_both, "parent: BOTH"),
        cellstat(par_trip, "parent: BOTH + DBC rank >= 98 (today's literal state)"),
        cellstat(pd.DatetimeIndex(par_both).difference(pd.DatetimeIndex(live_epi)),
                 "DISCARDED COMPLEMENT (parent BOTH, not a neighbour)"),
        cellstat(spy_idx, "all days")]
show(rows, "TLT h=5, lag=1, 21td declustered")
print("  >> analogue gate over the 'BOTH' parent: %+.3fpp"
      % (rows[0]["edge_pct"] - rows[3].get("edge_pct", np.nan)))
print("  >> analogue gate over the literal triple state: %+.3fpp"
      % (rows[0]["edge_pct"] - rows[4].get("edge_pct", np.nan)))

print("\n=== 3. PER-EPISODE TABLE, TLT h=5 ===")
e = pd.DatetimeIndex(live_epi).intersection(r5v.index)
for d in e:
    print("   anchor %s  ->  %+.3f%%" % (d.date(), 100 * r5v.loc[d]))
print("  ", cluster_note(e, r5v.loc[e].values))
v = r5v.loc[e].values
order = np.argsort(-v)
print("  drop-best-2: %+.3f%% -> %+.3f%% on n=%d"
      % (100 * v.mean(), 100 * np.delete(v, order[:2]).mean(), len(v) - 2))
print("  cost: %.1f bps vs a ~4 bp TLT round trip -> %.1fx"
      % (10000 * v.mean(), 10000 * v.mean() / 4))
print("  bootstrap P(mean<=0) = %.3f" % bootstrap_p_le0(v))

print("\n=== 4. HORIZON NEIGHBOURS around the winning h=5 ===")
rows = []
for h in (1, 2, 3, 4, 5, 6, 7, 8, 10):
    r = fwd_lag(panel["TLT"], h, LAG)
    rv = r.dropna()
    d = pd.DatetimeIndex(live_epi).intersection(rv.index)
    vv = rv.loc[d].values
    w = int((vv > 0).sum())
    rows.append({"h": h, "n": len(vv), "mean_pct": round(100 * vv.mean(), 3),
                 "edge_pct": round(100 * (vv.mean() - rv.mean()), 3),
                 "record": "%d-%d" % (w, len(vv) - w),
                 "sign_p": round(sign_test(w, len(vv)), 4),
                 "worst_pct": round(100 * vv.min(), 2)})
show(rows, "TLT horizon ladder on the live neighbour episodes")
