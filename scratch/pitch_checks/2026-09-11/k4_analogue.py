"""A11 round 1: nearest-neighbour tapes to the 2026-09-10 state.

PRE-SPECIFIED feature vector (fixed before any forward number is read, and it
is the one written into the candidate brief, so there is no tuning freedom):
  f1  SPY 21-session return
  f2  ^VIX close LEVEL
  f3  ^TNX trailing-252 percentile rank of its LEVEL
  f4  DBC trailing-252 percentile rank of its LEVEL
  f5  TLT trailing-252 percentile rank of its LEVEL

Each standardised over its own full history on the common panel. k=25 nearest
by Euclidean distance, declustered at 21 td (FILTER-THEN-DECLUSTER; the k
nearest are selected first, then thinned, which is the construction the brief
specifies and the only one that keeps "these are the 25 closest tapes"
meaningful. The alternative -- decluster the whole calendar then take the 25
nearest survivors -- is reported as a robustness row because the two are not
commutative).

Caret series (^VIX, ^TNX) are reindexed onto SPY's calendar BEFORE any
differencing, per the standing convention.

Attacks run in the order the brief demands:
  1. the 2026-09-09 trap: year histogram + scheduled-print share vs base rate
  2. k sensitivity (10/25/50) and leave-one-feature-out
  3. neighbour forward vs the ALL-DAYS unconditional, per vehicle
  4. which vehicle, if any, has content
plus the standing registry charge (2026-09-06 reference class): rank the
agreements inside a vehicle x horizon grid rather than quoting the best cell.
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
K_MAIN = 25
GAP = 21
LAG = 1

px_d = load_prices(TK)
spy_idx = px_d["SPY"]["Close"].index          # SPY's calendar is the calendar
panel = pd.DataFrame({t: px_d[t]["Close"].reindex(spy_idx) for t in TK})


def lvl_rank(s, lookback=252):
    """trailing-`lookback` percentile rank of the LEVEL (0-100)."""
    return rolling_on_valid(s, lambda x: x.rolling(lookback).rank(pct=True) * 100.0)


feat = pd.DataFrame(index=spy_idx)
feat["spy21"] = (panel["SPY"] / panel["SPY"].shift(21) - 1.0)
feat["vix"] = panel["^VIX"]
feat["tnx_r"] = lvl_rank(panel["^TNX"])
feat["dbc_r"] = lvl_rank(panel["DBC"])
feat["tlt_r"] = lvl_rank(panel["TLT"])
FEATS = list(feat.columns)

feat = feat.dropna()
print("feature panel: %s .. %s  n=%d" % (feat.index[0].date(), feat.index[-1].date(), len(feat)))

TODAY = feat.index[-1]
print("anchor session:", TODAY.date())
print("live state (raw):", feat.loc[TODAY].round(3).to_dict())

mu, sd = feat.mean(), feat.std(ddof=0)
z = (feat - mu) / sd
print("live state (z)  :", z.loc[TODAY].round(3).to_dict())

# forward returns, lag=1, on SPY's calendar
fwd = {(v, h): fwd_lag(panel[v], h, LAG) for v in VEH for h in HS}
# candidate pool = every session with all four vehicles' h=10 defined
pool_ok = np.ones(len(feat), dtype=bool)
for v in VEH:
    pool_ok &= fwd[(v, 10)].reindex(feat.index).notna().values
pool = feat.index[pool_ok]
print("candidate pool (h=10 defined on all four vehicles): n=%d  %s .. %s"
      % (len(pool), pool[0].date(), pool[-1].date()))


def neighbours(k, drop=None, decluster_first=False):
    cols = [c for c in FEATS if c != drop]
    zz = z[cols]
    d = np.sqrt(((zz.loc[pool] - zz.loc[TODAY]) ** 2).sum(axis=1))
    d = d.sort_values()
    if decluster_first:
        kept = declusters(pd.DatetimeIndex(sorted(d.index)), GAP, spy_idx)
        d = d.loc[d.index.intersection(kept)].sort_values()
        return pd.DatetimeIndex(d.index[:k]), d
    raw = pd.DatetimeIndex(sorted(d.index[:k]))
    return declusters(raw, GAP, spy_idx), d


epi, dist = neighbours(K_MAIN)
raw25 = pd.DatetimeIndex(sorted(dist.index[:K_MAIN]))
print("\n=== 0. the k=25 raw neighbour set ===")
print("distances: nearest %.3f  25th %.3f  |  typical day's distance to today: "
      "median %.3f" % (dist.iloc[0], dist.iloc[K_MAIN - 1], dist.median()))
print("raw 25 dates:", ", ".join(str(d.date()) for d in raw25))
print("after 21td decluster: n=%d ->" % len(epi), ", ".join(str(d.date()) for d in epi))

print("\n=== 1a. YEAR HISTOGRAM of the raw k=25 neighbours (the trap) ===")
yr = pd.Series(1, index=raw25).groupby(raw25.year).sum()
print(yr.to_dict())
print("top-2 years hold %d of %d = %.0f%% of the neighbour set"
      % (yr.sort_values(ascending=False).head(2).sum(), len(raw25),
         100 * yr.sort_values(ascending=False).head(2).sum() / len(raw25)))
epi_yr = pd.Series(1, index=epi).groupby(epi.year).sum()
print("declustered episodes by year:", epi_yr.to_dict())
print("distinct years among episodes: %d of %d episodes" % (len(epi_yr), len(epi)))

print("\n=== 1b. SCHEDULED-PRINT SHARE vs the unconditional base rate ===")
PRINTS = ("cpi", "ppi", "nfp", "fomc_decision")
ev_all = load_events(list(PRINTS))
for kind in PRINTS + ("ALL4",):
    kinds = PRINTS if kind == "ALL4" else (kind,)
    ds = pd.DatetimeIndex(ev_all[ev_all["event"].isin(kinds)]["date"])
    on_n = np.mean([d in set(ds) for d in raw25])
    on_b = np.mean([d in set(ds) for d in pool])
    inw_n = event_in_window(raw25, spy_idx, 3, LAG, kinds).mean()
    inw_b = event_in_window(pool, spy_idx, 3, LAG, kinds).mean()
    print("  %-14s ON the neighbour date %5.1f%% vs base %5.1f%%   |  "
          "inside an h=3 hold %5.1f%% vs base %5.1f%%"
          % (kind, 100 * on_n, 100 * on_b, 100 * inw_n, 100 * inw_b))

print("\n=== 3. NEIGHBOUR FORWARD vs the ALL-DAYS UNCONDITIONAL (episodes, lag=1) ===")
grid = []
for v in VEH:
    for h in HS:
        r = fwd[(v, h)]
        val = r.dropna().index
        e = pd.DatetimeIndex(epi).intersection(val)
        vals = r.loc[e].values
        base = r.loc[val]
        w = int((vals > 0).sum())
        grid.append({
            "veh": v, "h": h, "n": len(vals),
            "mean_pct": round(100 * vals.mean(), 3),
            "base_pct": round(100 * base.mean(), 3),
            "edge_pct": round(100 * (vals.mean() - base.mean()), 3),
            "median_pct": round(100 * np.median(vals), 3),
            "hit": round(100 * w / len(vals), 1),
            "record": "%d-%d" % (w, len(vals) - w),
            "sign_p": round(sign_test(w, len(vals)), 4),
            "sign_p_vs_base": round(sign_test(w, len(vals), float((base > 0).mean())), 4),
            "t": round(vals.mean() / (vals.std(ddof=1) / np.sqrt(len(vals))), 2),
            "worst_pct": round(100 * vals.min(), 2),
            "boot_ple0": round(bootstrap_p_le0(vals), 3),
        })
G = pd.DataFrame(grid)
print(G.to_string(index=False))
neg = (G["edge_pct"] < 0).mean()
print("\n  reference-class charge (registry 2026-09-06): %.0f%% of the %d "
      "vehicle x horizon cells carry NEGATIVE edge; median edge %+.3fpp"
      % (100 * neg, len(G), G["edge_pct"].median()))
print("  best cell: %s" % G.loc[G["edge_pct"].abs().idxmax()].to_dict())

print("\n=== 2a. k SENSITIVITY (10 / 25 / 50), edge_pct per vehicle x horizon ===")
for k in (10, 25, 50):
    e_k, _ = neighbours(k)
    row = {"k": k, "n_epi": len(e_k)}
    for v in VEH:
        for h in HS:
            r = fwd[(v, h)]
            val = r.dropna().index
            ee = pd.DatetimeIndex(e_k).intersection(val)
            row["%s_h%d" % (v, h)] = round(100 * (r.loc[ee].mean() - r.loc[val].mean()), 3)
    print(row)

print("\n=== 2b. LEAVE-ONE-FEATURE-OUT (k=25): set overlap and edge_pct ===")
base_set = set(raw25)
for drop in [None] + FEATS:
    e_d, d_d = neighbours(K_MAIN, drop=drop)
    raw_d = set(pd.DatetimeIndex(sorted(d_d.index[:K_MAIN])))
    ov = len(base_set & raw_d)
    row = {"drop": drop or "(none)", "overlap_of_25": ov, "n_epi": len(e_d)}
    for v in VEH:
        for h in (1, 3, 5, 10):
            r = fwd[(v, h)]
            val = r.dropna().index
            ee = pd.DatetimeIndex(e_d).intersection(val)
            row["%s_h%d" % (v, h)] = round(100 * (r.loc[ee].mean() - r.loc[val].mean()), 3)
    print(row)

print("\n=== 2c. decluster-then-filter (NOT commutative; robustness row) ===")
e_alt, _ = neighbours(K_MAIN, decluster_first=True)
print("n=%d  dates:" % len(e_alt), ", ".join(str(d.date()) for d in e_alt))
print("overlap with the filter-then-decluster episode set: %d of %d"
      % (len(set(e_alt) & set(epi)), len(epi)))
for v in VEH:
    row = {"veh": v}
    for h in HS:
        r = fwd[(v, h)]
        val = r.dropna().index
        ee = pd.DatetimeIndex(e_alt).intersection(val)
        row["h%d_edge" % h] = round(100 * (r.loc[ee].mean() - r.loc[val].mean()), 3)
    print(row)

print("\n=== 4. concentration on the two widest cells ===")
for v in VEH:
    h = 10
    r = fwd[(v, h)]
    val = r.dropna().index
    ee = pd.DatetimeIndex(epi).intersection(val)
    print("  %s h=10: %s" % (v, cluster_note(ee, r.loc[ee].values)))
