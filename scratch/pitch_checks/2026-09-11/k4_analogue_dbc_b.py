"""A11 round 2, second defended cell. The brief says "report which vehicle, if
any, has content, and defend at most one". k4_analogue_b defended TLT h=5
because it has the cleanest record (8-1). The LARGEST |edge| in the grid is
the other candidate -- DBC h=3 at -0.985pp, i.e. a SHORT of the commodity
complex -- and it is the one cell whose sign survives k=10/25/50. It is
defended here so the lane is not killed on the weaker of its two arms.

  1. the DBC short across k and feature drops, sign-stability only
  2. permutation TARGETED at this cell: 500 random anchor states, same
     machinery, distribution of the DBC h=3 edge
  3. gate attribution: what does the plain parent "DBC at a 252-day high" pay
     over the same h=3 short, and what does the DISCARDED COMPLEMENT pay?
  4. concentration, cost, era
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
K_MAIN, GAP, LAG, H = 25, 21, 1, 3

px_d = load_prices(TK)
spy_idx = px_d["SPY"]["Close"].index
panel = pd.DataFrame({t: px_d[t]["Close"].reindex(spy_idx) for t in TK})
lvl_rank = lambda s, n=252: rolling_on_valid(s, lambda x: x.rolling(n).rank(pct=True) * 100.0)

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
ok = np.ones(len(feat), dtype=bool)
for v in VEH:
    ok &= fwd[(v, 10)].reindex(feat.index).notna().values
pool = feat.index[ok]
Zp = z.loc[pool].values
pos_map = pd.Series(range(len(spy_idx)), index=spy_idx)
pool_pos = pos_map.loc[pool].values
r3 = fwd[("DBC", 3)]
r3v = r3.dropna()


def decl_fast(order_positions, gap=GAP):
    keep, last = [], -10 ** 9
    for i in order_positions:
        p = pool_pos[i]
        if p - last >= gap:
            keep.append(i)
            last = p
    return keep


def nn_episodes(target_z, k=K_MAIN, cols=None):
    zz = Zp if cols is None else z.loc[pool, cols].values
    tz = target_z if cols is None else z.loc[TODAY, cols].values
    d = np.sqrt(((zz - tz) ** 2).sum(axis=1))
    top = np.argsort(d)[:k]
    top = top[np.argsort(pool_pos[top])]
    return pool[decl_fast(top)]


def rep(dates, label, short=True):
    d = pd.DatetimeIndex(dates).intersection(r3v.index)
    if len(d) == 0:
        return {"label": label, "n": 0}
    v = -r3v.loc[d].values if short else r3v.loc[d].values
    base = -r3v.mean() if short else r3v.mean()
    w = int((v > 0).sum())
    return {"label": label, "n": len(v), "mean_pct": round(100 * v.mean(), 3),
            "edge_pct": round(100 * (v.mean() - base), 3),
            "hit": round(100 * w / len(v), 1), "record": "%d-%d" % (w, len(v) - w),
            "sign_p": round(sign_test(w, len(v)), 4),
            "t": round(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 2),
            "worst_pct": round(100 * v.min(), 2)}


live = nn_episodes(z.loc[TODAY].values)
print("=== 1. SIGN STABILITY of the DBC h=3 SHORT ===")
rows = [rep(live, "live k=25 (DEFENDED)")]
for k in (10, 15, 25, 40, 50, 75, 100):
    rows.append(rep(nn_episodes(z.loc[TODAY].values, k=k), "k=%d" % k))
show(rows)
FEATS = list(feat.columns)
rows = []
for drop in FEATS:
    cols = [c for c in FEATS if c != drop]
    rows.append(rep(nn_episodes(None, cols=cols), "drop %s" % drop))
show(rows, "leave-one-feature-out (k=25)")

print("\n=== 2. PERMUTATION TARGETED AT THIS CELL ===")
rng = np.random.default_rng(11)
picks = rng.choice(len(pool), size=500, replace=False)
edges, hits = [], []
base = -r3v.mean()
for i in picks:
    e = pd.DatetimeIndex(nn_episodes(Zp[i])).intersection(r3v.index)
    if len(e) < 5:
        continue
    v = -r3v.loc[e].values
    edges.append(100 * (v.mean() - base))
    hits.append((v > 0).mean())
edges = np.array(edges)
obs = rep(live, "x")["edge_pct"]
print("  DBC h=3 SHORT edge across 500 random anchor states: median %+.3fpp, "
      "sd %.3f" % (np.median(edges), edges.std()))
print("  P(random anchor edge >= live %+.3fpp) = %.3f" % (obs, float((edges >= obs).mean())))
print("  P(random anchor |edge| >= |live|)     = %.3f"
      % float((np.abs(edges) >= abs(obs)).mean()))
print("  live hit %.1f%% vs random median %.1f%%, P(hit >= live) = %.3f"
      % (100 * rep(live, "x")["hit"] / 100, 100 * np.median(hits),
         float((np.array(hits) >= rep(live, "x")["hit"] / 100).mean())))

print("\n=== 3. GATE ATTRIBUTION vs the literal parent states ===")
dbc_r, tnx_r, tlt_r = feat["dbc_r"], feat["tnx_r"], feat["tlt_r"]


def dec(ix):
    return declusters(pd.DatetimeIndex(ix).intersection(r3v.index), GAP, spy_idx)


p_hi = dec(feat.index[dbc_r.values >= 99.5])
p_trip = dec(feat.index[(dbc_r.values >= 98) & (tnx_r.values >= 98) & (tlt_r.values <= 2)])
comp = pd.DatetimeIndex(p_hi).difference(pd.DatetimeIndex(live))
show([rep(live, "ANALOGUE k=25 (DEFENDED)"),
      rep(p_hi, "parent: DBC at a 252d level high (rank>=99.5)"),
      rep(p_trip, "parent: today's literal triple state"),
      rep(comp, "DISCARDED COMPLEMENT (DBC high, not a neighbour)"),
      rep(spy_idx, "all days")], "DBC h=3, SHORT, lag=1, 21td declustered")

print("\n=== 4. CONCENTRATION / ERA / COST ===")
d = pd.DatetimeIndex(live).intersection(r3v.index)
v = -r3v.loc[d].values
for dd, vv in zip(d, v):
    print("   anchor %s -> short pays %+.3f%%" % (dd.date(), 100 * vv))
print("  ", cluster_note(d, v))
order = np.argsort(-v)
print("  drop-best-2: %+.3f%% -> %+.3f%% on n=%d"
      % (100 * v.mean(), 100 * np.delete(v, order[:2]).mean(), len(v) - 2))
print("  cost: %.1f bps vs a ~10 bp DBC round trip (thin ETF) -> %.1fx"
      % (10000 * v.mean(), 10000 * v.mean() / 10))
print("  bootstrap P(short mean<=0) = %.3f" % bootstrap_p_le0(v))
show(era_split(d, v), "era split")
print("\n=== 5. horizon ladder for the DBC short on the live episodes ===")
rows = []
for h in (1, 2, 3, 4, 5, 7, 10):
    r = fwd_lag(panel["DBC"], h, LAG)
    rv = r.dropna()
    dd = pd.DatetimeIndex(live).intersection(rv.index)
    vv = -rv.loc[dd].values
    w = int((vv > 0).sum())
    rows.append({"h": h, "n": len(vv), "mean_pct": round(100 * vv.mean(), 3),
                 "edge_pct": round(100 * (vv.mean() + rv.mean()), 3),
                 "record": "%d-%d" % (w, len(vv) - w),
                 "sign_p": round(sign_test(w, len(vv)), 4)})
show(rows)
