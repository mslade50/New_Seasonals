"""C4 round 3b -- the vehicle question, the entry form, the loser paths, and an
honest multiplicity accounting.

Round 3 answered the candidate's actual question: the miner has NO independent
pre-CPI fingerprint.  GDX-1.77*GLD residual on the cell is +0.112% (t=0.26,
49.0% hit) at h=3 and +0.260% (t=0.48) at h=5.  Every basis point of the GDX
number is gold beta.  So this script asks the follow-on: if the live object is
the METAL, does the metal's own cell stand up, and is GDX or GLD the better
vehicle risk-adjusted?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

LAG, K = 1, 3
PX = load_prices(["GDX", "GLD"])
P = close_panel(["GDX", "GLD"]).dropna(subset=["GDX"])
g, gl = P["GDX"], P["GLD"]
idx = g.index
rk5g = pct_rank(g, 5)
THR = (rk5g >= 80.0).fillna(False)


def anchor_k(kind: str, k: int) -> pd.DatetimeIndex:
    ev = load_events([kind])["date"]
    out = []
    for d in ev:
        p = int(np.searchsorted(idx.values, np.datetime64(d)))
        if p - k < 0 or p >= len(idx):
            continue
        out.append(idx[p - k])
    return pd.DatetimeIndex(sorted(set(out)))


A3 = anchor_k("cpi", K)

# --------------------------------------------------- 1. vehicle comparison
print("### 1. VEHICLE: GDX vs GLD on the same cell, risk-adjusted ###")
rows = []
for H in (2, 3, 5, 10):
    for tk, s in (("GDX", g), ("GLD", gl)):
        fw = fwd_lag(s, H, LAG)
        ok = fw.notna()
        t = pd.DatetimeIndex(A3).intersection(idx[THR.values & ok.values])
        e = declusters(t, H, idx[ok.values])
        v = fw.loc[e].values
        r = summarize(v, f"{tk} h={H}")
        r["mean_over_sd"] = round(r["mean_pct"] / r["sd_pct"], 3)
        r["edge"] = round(r["mean_pct"] - 100 * fw[ok].mean(), 3)
        r["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        rows.append(r)
show(rows, "vehicle comparison (trigger = GDX rank5>=80, anchor CPI-3)")

# --------------------------------------------------- 2. h=10, the best cell
print("\n### 2. h=10 was the strongest horizon -- is it still the CPI, or drift? ###")
H = 10
fw = fwd_lag(g, H, LAG)
fwl = fwd_lag(gl, H, LAG)
ok = fw.notna() & fwl.notna()
VALID = idx[ok.values]
T = pd.DatetimeIndex(A3).intersection(idx[THR.values & ok.values])
e = declusters(T, H, VALID)
v = fw.loc[e].values
print(f"  cell N={len(v)} mean={100*v.mean():+.3f}% hit={100*(v>0).mean():.1f}% "
      f"t={v.mean()/(v.std(ddof=1)/np.sqrt(len(v))):.2f} sign_p={sign_test(int((v>0).sum()), len(v)):.4f}")
print(f"  thrust NOT on CPI-3 = "
      f"{100*fw.loc[declusters(idx[THR.values & ok.values].difference(A3), H, VALID)].mean():+.3f}%")
print(f"  ALL CPI-3, no thrust = "
      f"{100*fw.loc[declusters(pd.DatetimeIndex(A3).intersection(VALID), H, VALID)].mean():+.3f}%")
rows = []
for kind in ("ppi", "nfp", "opex"):
    a = pd.DatetimeIndex(anchor_k(kind, K)).intersection(idx[THR.values & ok.values]).difference(A3)
    rows.append(summarize(fw.loc[declusters(a, H, VALID)].values, f"{kind}-3 ex-CPI x thrust"))
rows.append(summarize(v, "CPI-3 x thrust"))
show(rows, "h=10 disjoint anchor placebo")
beta = np.polyfit(fwl[ok].values, fw[ok].values, 1)[0]
show([summarize(v, "GDX"), summarize(fwl.loc[e].values, "GLD same days"),
      summarize((fw - beta * fwl).loc[e].values, f"GDX-{beta:.2f}*GLD residual"),
      summarize((fw - beta * fwl)[ok].values, "residual all days")],
     f"h=10 gold decomposition (beta={beta:.2f})")
mt = np.array([d.year % 4 == 2 for d in e])
show([summarize(v[mt], f"MIDTERM (N={int(mt.sum())})"), summarize(v[~mt], "non-midterm")],
     "h=10 cycle split")
show(era_split(e, v), "h=10 era")
print("  concentration:", cluster_note(e, v, k=2))
o = np.argsort(v)
print(f"  drop-2-best {100*np.delete(v, o[-2:]).mean():+.3f}%  worst 3:",
      [(str(pd.Timestamp(e[i]).date()), round(100*v[i], 2)) for i in o[:3]])

# --------------------------------------------------- 3. entry form
print("\n### 3. entry form: MOC vs close-anchored LIMIT, WHOLE variants ###")
H = 5
d = PX["GDX"]
atr = pd.Series(wilder_atr(d["High"], d["Low"], d["Close"], 14), index=d.index)
fw = fwd_lag(g, H, LAG)
ok = fw.notna()
T = pd.DatetimeIndex(A3).intersection(idx[THR.values & ok.values])
e = declusters(T, H, idx[ok.values])
pos = pd.Series(range(len(idx)), index=idx)
rows = []
for k_atr in (0.0, 0.25, 0.5, 0.75):
    fills, rets = 0, []
    for dt in e:
        p = int(pos[dt])
        if p + 1 + H >= len(idx):
            continue
        a_ = atr.reindex(idx).iloc[p]
        if not np.isfinite(a_):
            continue
        lim = g.iloc[p] - k_atr * a_
        ed = p + 1
        if k_atr == 0.0:
            fill = g.iloc[ed]
        else:
            lo_ = d["Low"].reindex(idx).iloc[ed]
            if not np.isfinite(lo_) or lo_ > lim:
                continue
            fill = min(lim, d["Open"].reindex(idx).iloc[ed])
        fills += 1
        rets.append(g.iloc[ed + H] / fill - 1.0)
    r = summarize(np.array(rets), "MOC at D+1" if k_atr == 0 else f"LIMIT close-{k_atr}ATR, D+1 only")
    r["fill_rate"] = round(100 * fills / len(e), 1)
    r["capture_pp"] = round(r["mean_pct"] * fills / len(e), 3)
    rows.append(r)
show(rows, f"entry variants, h={H} (capture_pp = mean x fill rate, the whole-variant number)")
print(f"  ATR today: {atr.iloc[-1]:.3f} on close {g.iloc[-1]:.2f} "
      f"= {100*atr.iloc[-1]/g.iloc[-1]:.2f}% of price")

# --------------------------------------------------- 4. loser paths
print("\n### 4. loser paths (h=5) -- the concrete invalidation number ###")
paths = episode_paths(P, e, [("GDX", 1.0)], H, LAG)
v = fw.loc[e].values
mask_l = np.array([x < 0 for x in v])
lose = paths.loc[[dt for i, dt in enumerate(e) if mask_l[i]]]
print(f"losing episodes {len(lose)}/{len(e)}")
print((100 * lose).round(2).to_string())
print("\nday-1 across ALL episodes:", {k: round(x, 3) for k, x in summarize(paths[1].values, "d1").items()
                                      if k in ("n", "mean_pct", "hit", "worst_pct")})
d1 = paths[1].values
print(f"  P(episode loses | day1 <= -1%) = "
      f"{100*(v[d1 <= -0.01] < 0).mean():.0f}%  (n={int((d1 <= -0.01).sum())})")
print(f"  P(episode loses | day1 > 0)    = "
      f"{100*(v[d1 > 0] < 0).mean():.0f}%  (n={int((d1 > 0).sum())})")
print(f"  mean | day1 <= -1% : {100*v[d1 <= -0.01].mean():+.2f}%   "
      f"mean | day1 > 0 : {100*v[d1 > 0].mean():+.2f}%")

# --------------------------------------------------- 5. multiplicity ledger
print("\n### 5. multiplicity ledger -- what I actually searched ###")
print("  PRE-SPECIFIED by the candidate + today's calendar: instrument GDX,")
print("    direction long, anchor k=3 (fixed by CPI on 2026-08-12), thrust conditioner.")
print("  SEARCHED BY ME: threshold in {60,70,80,85,90,95} = 6, horizon in")
print("    {1,2,3,4,5,7,10} = 7  -> 42 cells at the live k.")
print("  Cells with sign_p <= 0.05 at k=3 across that grid:")
cnt, tot = 0, 0
for thr in (60.0, 70.0, 80.0, 85.0, 90.0, 95.0):
    m = (rk5g >= thr).fillna(False)
    for h in (1, 2, 3, 4, 5, 7, 10):
        f2 = fwd_lag(g, h, LAG)
        o2 = f2.notna()
        t2 = pd.DatetimeIndex(A3).intersection(idx[m.values & o2.values])
        if len(t2) < 8:
            continue
        e2 = declusters(t2, h, idx[o2.values])
        v2 = f2.loc[e2].values
        sp = sign_test(int((v2 > 0).sum()), len(v2))
        tot += 1
        if sp <= 0.05:
            cnt += 1
            print(f"    thr>={thr:.0f} h={h:2d}: N={len(v2):3d} mean={100*v2.mean():+6.3f}% "
                  f"hit={100*(v2>0).mean():4.1f}% sign_p={sp:.4f}")
print(f"  -> {cnt}/{tot} cells at sign_p<=0.05 (expect ~{0.05*tot:.1f} by chance if all null)")
