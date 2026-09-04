"""C9 round 2 -- the mandatory round-2 set on the one round-1 survivor.

 1. decluster + concentration + per-episode table
 2. definition neighbours: compression threshold, range definition, anchor
    offset, horizon
 3. era / regime split: SVXY leverage break, midterm, fragility-dial regime
 4. gate attribution (done in r1; extended to the range DEFINITION here)
 5. reference class: the identical gate on CPI / PPI / FOMC anchors -- if the
    vol premium collapses into ANY scheduled print out of a dead range, the
    payrolls label is not the finding
 6. the tail: Feb-2018 forensics + loser paths + worst-case sizing
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa: F401,F403,E402
from pitch_lab import (close_panel, fwd_lag, declusters, summarize, sign_test,
                       cluster_note, rolling_on_valid, load_events, show,
                       anchor_positions, bootstrap_p_le0, episode_paths)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)

ROOT = Path(__file__).resolve().parents[3]
px = close_panel(["^VIX", "^VIX3M", "SVXY", "UVXY", "SPY"])
cal = px["SPY"].dropna().index
vix = px["^VIX"]

rng21 = (rolling_on_valid(vix, lambda x: x.rolling(21).max())
         - rolling_on_valid(vix, lambda x: x.rolling(21).min()))
rel = rng21 / rolling_on_valid(vix, lambda x: x.rolling(21).mean())
RNG = rolling_on_valid(rel, lambda x: x.rolling(252).rank(pct=True) * 100)
ABS = rolling_on_valid(rng21, lambda x: x.rolling(252).rank(pct=True) * 100)
# a third, independent definition: realised SPY vol percentile
rv21 = rolling_on_valid(px["SPY"].pct_change(),
                        lambda x: x.rolling(21).std() * np.sqrt(252) * 100)
RV = rolling_on_valid(rv21, lambda x: x.rolling(252).rank(pct=True) * 100)

nfp = load_events(["nfp"])["date"]


def anchors(kind_dates, k=-2, gate=None):
    p, _ = anchor_positions(cal, kind_dates, k)
    a = pd.DatetimeIndex([cal[i] for i in p])
    if gate is not None:
        a = a[gate.reindex(a).fillna(False).values]
    return a


def stat(dates, tkr, h, lag=1, label=""):
    ss = px[tkr].dropna()
    f = fwd_lag(ss, h, lag=lag)
    v = f.reindex(pd.DatetimeIndex(dates)).dropna()
    if len(v) == 0:
        return {"label": label, "n": 0}, v
    drift = 100 * f.dropna().mean()
    st = summarize(v.values, label)
    st["excess_pp"] = round(st["mean_pct"] - drift, 3)
    st["signp"] = round(sign_test(int((v.values > 0).sum()), len(v)), 4)
    return st, v


G15 = RNG <= 15.0
A = anchors(nfp, -2, G15)

print("=" * 110)
print(f"LIVE: rel-range pctile {RNG.iloc[-1]:.1f}  abs-range pctile {ABS.iloc[-1]:.1f}  "
      f"SPY realised-vol pctile {RV.iloc[-1]:.1f}")
print("=" * 110)

# ---------------------------------------------------------------------------
# 1. PER-EPISODE TABLE + declustering
# ---------------------------------------------------------------------------
print("\n1. PER-EPISODE TABLE (gate ON, h=1)")
_, v_svxy = stat(A, "SVXY", 1)
_, v_vix = stat(A, "^VIX", 1)
_, v_uvxy = stat(A, "UVXY", 1)
tbl = pd.DataFrame({"vix_h1_pct": (100 * v_vix).round(2)})
tbl["svxy_h1_pct"] = (100 * v_svxy).round(2)
tbl["uvxy_h1_pct"] = (100 * v_uvxy).round(2)
tbl["rng_pctile"] = RNG.reindex(tbl.index).round(1)
tbl["vix_level"] = vix.reindex(tbl.index).round(2)
print(tbl.to_string())
gap = declusters(A, 21, cal)
print(f"\n  declustered at 21 td: {len(gap)} of {len(A)} anchors survive "
      f"(NFP prints are ~21 td apart by construction, so this is a no-op check)")
print("  " + cluster_note(v_svxy.index, v_svxy.values, k=2))
print("  " + cluster_note(v_svxy.index, v_svxy.values, k=3))
order = np.argsort(-v_svxy.values)
for k in (1, 2, 3):
    keep = np.delete(v_svxy.values, order[:k])
    st = summarize(keep)
    print(f"  SVXY drop top {k}: n={st['n']} mean {st['mean_pct']:+.3f}% "
          f"hit {st['hit']:.1f} signp {sign_test(int((keep>0).sum()), len(keep)):.4f}")

# ---------------------------------------------------------------------------
# 2. DEFINITION NEIGHBOURS
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("2. DEFINITION NEIGHBOURS")
print("=" * 110)
print("\n2a. compression threshold (rel-range percentile), SVXY h=1 and ^VIX h=1")
rows = []
for thr in (5, 10, 15, 20, 25, 30, 40, 50, 100):
    a = anchors(nfp, -2, RNG <= thr)
    s1, _ = stat(a, "SVXY", 1, label=f"SVXY thr<={thr}")
    s2, _ = stat(a, "^VIX", 1, label=f"VIX  thr<={thr}")
    rows.append({"thr": thr, "n_anchors": len(a),
                 "svxy_n": s1.get("n"), "svxy_mean": round(s1.get("mean_pct", np.nan), 3),
                 "svxy_hit": round(s1.get("hit", np.nan), 1),
                 "svxy_signp": s1.get("signp"),
                 "vix_n": s2.get("n"), "vix_mean": round(s2.get("mean_pct", np.nan), 3),
                 "vix_hit": round(s2.get("hit", np.nan), 1)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n2b. range DEFINITION (relative vs absolute vs SPY realised vol), thr<=15")
for lbl, g in (("rel range (pitched)", RNG <= 15), ("abs range", ABS <= 15),
               ("SPY realised vol", RV <= 15)):
    a = anchors(nfp, -2, g)
    s1, _ = stat(a, "SVXY", 1, label=f"SVXY | {lbl}")
    s2, _ = stat(a, "^VIX", 1, label=f"VIX | {lbl}")
    show([s1, s2], f"{lbl} (n_anchors={len(a)})")

print("\n2c. anchor offset -3/-2/-1 and horizon 1/2/3, gate ON")
rows = []
for k in (-3, -2, -1):
    a = anchors(nfp, k, G15)
    for h in (1, 2, 3):
        s1, _ = stat(a, "SVXY", h, label=f"k={k} h={h}")
        rows.append(s1)
show(rows, "SVXY")

# ---------------------------------------------------------------------------
# 3. REGIME SPLITS incl. the fragility dial
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("3. REGIME SPLITS")
print("=" * 110)
frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
ma10 = frag["63d"].rolling(10).mean()
print("  VINTAGE: rd2_fragility rows before 2026-07-02 are the recompute "
      "vintage; 2026-07-02+ are point-in-time appends.")
rd = ma10.reindex(v_svxy.index)
print(f"  dial coverage on SVXY episodes: {int(rd.notna().sum())} of {len(v_svxy)} "
      f"(series starts 2016)")
cov = pd.DataFrame({"dial": rd.round(1), "svxy_h1_pct": (100 * v_svxy).round(2)}).dropna()
print(cov.to_string())
print(f"  MAX historical dial on a gated NFP anchor = {rd.max():.1f}   "
      f"against today's {ma10.iloc[-1]:.1f}")
rd_v = ma10.reindex(v_vix.index)
print(f"  (on the full 45-anchor VIX set: dial coverage {int(rd_v.notna().sum())}, "
      f"max {rd_v.max():.1f})")
hi = cov["dial"] >= 60
if hi.sum():
    print(f"  gated anchors at dial>=60: n={int(hi.sum())}, "
          f"SVXY mean {cov.loc[hi, 'svxy_h1_pct'].mean():+.3f}%")

# ---------------------------------------------------------------------------
# 4. REFERENCE CLASS -- the same gate on other scheduled prints
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("4. REFERENCE CLASS -- is it payrolls, or ANY print out of a dead range?")
print("=" * 110)
rows_v, rows_s = [], []
for kind in ("nfp", "cpi", "ppi", "fomc_decision"):
    ev = load_events([kind])["date"]
    a = anchors(ev, -2, G15)
    s2, _ = stat(a, "^VIX", 1, label=f"^VIX | {kind} (n_anchor={len(a)})")
    s1, _ = stat(a, "SVXY", 1, label=f"SVXY | {kind} (n_anchor={len(a)})")
    rows_v.append(s2)
    rows_s.append(s1)
show(rows_v, "^VIX h=1, gate ON, by event kind")
show(rows_s, "SVXY h=1, gate ON, by event kind")
# permutation: is NFP distinguishable from the other three?
vals = {}
for kind in ("nfp", "cpi", "ppi", "fomc_decision"):
    ev = load_events([kind])["date"]
    a = anchors(ev, -2, G15)
    _, vv = stat(a, "^VIX", 1)
    vals[kind] = vv
pool = np.concatenate([v.values for v in vals.values()])
ns = {k: len(v) for k, v in vals.items()}
rng_ = np.random.default_rng(7)
obs = vals["nfp"].values.mean()
maxk = []
for _ in range(5000):
    perm = rng_.permutation(pool)
    i, mins = 0, []
    for k, n in ns.items():
        mins.append(perm[i:i + n].mean())
        i += n
    maxk.append(min(mins))       # most-negative group mean (best for short vol)
p_perm = float((np.array(maxk) <= obs).mean())
print(f"\n  permutation max-of-4 (most negative group mean): NFP observed "
      f"{100*obs:+.3f}%, family-wise P = {p_perm:.4f}  "
      f"(N per group {ns})")

# ---------------------------------------------------------------------------
# 5. THE TAIL -- Feb 2018, loser paths, worst case
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("5. TAIL RISK")
print("=" * 110)
for d in ("2018-01-31", "2018-02-01", "2018-02-02"):
    ts = pd.Timestamp(d)
    if ts in RNG.index:
        print(f"  {d}: rel-range pctile {RNG.loc[ts]:.1f}  abs {ABS.loc[ts]:.1f}  "
              f"VIX {vix.loc[ts]:.2f}  -> gate {'ON' if RNG.loc[ts] <= 15 else 'OFF'}")
p18, _ = anchor_positions(cal, load_events(["nfp"])["date"], -2)
a18 = pd.DatetimeIndex([cal[i] for i in p18])
a18 = a18[(a18.year == 2018)]
print(f"  2018 NFP anchors and their range percentile:")
for d in a18:
    print(f"    {d.date()}  rel {RNG.get(d, np.nan):.1f}  "
          f"gate {'ON' if RNG.get(d, 99) <= 15 else 'OFF'}")

print("\n  loser paths on the gated SVXY set (h=3 cumulative from entry):")
paths = episode_paths(px, v_svxy.index, [("SVXY", 1.0)], h=3, lag=1)
lose = paths[paths[1] < 0]
print((100 * paths).round(2).to_string())
if len(lose):
    print(f"  losers on day 1: n={len(lose)} of {len(paths)}, "
          f"mean d1 {100*lose[1].mean():+.2f}%, d3 {100*lose[3].mean():+.2f}%, "
          f"worst d3 {100*lose[3].min():+.2f}%")

print("\n  ungated NFP-anchor SVXY tail (what the gate is NOT protecting against):")
a_all = anchors(nfp, -2)
_, vall = stat(a_all, "SVXY", 1)
print(f"    all NFP anchors h=1: n={len(vall)} worst {100*vall.min():.2f}% "
      f"on {vall.idxmin().date()}")
_, vall3 = stat(a_all, "SVXY", 3)
print(f"    all NFP anchors h=3: n={len(vall3)} worst {100*vall3.min():.2f}% "
      f"on {vall3.idxmin().date()}")

# cost
print("\n  cost: SVXY round trip ~8-12 bps (it is not SPY). "
      f"edge h=1 = {100*100*v_svxy.mean():.1f} bps -> "
      f"{100*100*v_svxy.mean()/10:.1f}x at 10 bps")
