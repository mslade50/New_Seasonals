"""b2c — C7 round 2 continued.

Two jobs:

  A. REDO the reference class properly. b2b built each sector complex from the
     WHOLE sector map (52 energy names, 121 financials), so the breadth
     threshold meant something different in every sector and the comparison was
     void. Here every sector is cut to EXACTLY 11 members — the size of the
     declared energy complex — so "5 of 11 at z10 >= 2" is literally the same
     estimator everywhere. 25 random subsamples per sector, plus the noise
     permutation.

  B. MECHANISM. "Five names thrusting at once" presupposes five confirmations.
     Two of the five live names are ETFs (XLE, XOP) whose top holdings are the
     other three (COP, CVX, VLO). Measure the factor structure of the complex
     and the effective number of independent members, and read the count
     trigger's overlap with XLE's own z10 against it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, declusters, load_prices, show, summarize, vehicle_ret,
)
from strategy_config import LIQUID_PLUS_COMMODITIES  # noqa: E402

pd.set_option("display.width", 240)
H = 5
SIZE = 11
K = 5
COMPLEX = ["XLE", "XOP", "USO", "COP", "CVX", "VLO", "OXY", "SLB", "EOG",
           "HAL", "WMB"]


def tape_z10(close: pd.Series, n: int = 10) -> pd.Series:
    r = close.pct_change(n)
    v = close.pct_change().rolling(21).std()
    return r / (v * np.sqrt(n))


# ---------------------------------------------------------------------------
# A. reference class, every complex cut to exactly 11 members
# ---------------------------------------------------------------------------
sm = pd.read_parquet("data/sector_map.parquet")
sm = sm[sm["ticker"].isin(set(LIQUID_PLUS_COMMODITIES))]
mp = pd.read_parquet("data/master_prices.parquet", columns=["ticker", "date"])
nbar = mp.groupby("ticker").size()
sm = sm[sm["ticker"].map(nbar).fillna(0) >= 5000]
SPDR = {"Basic Materials": "XLB", "Consumer Cyclical": "XLY",
        "Consumer Defensive": "XLP", "Energy": "XLE",
        "Financial Services": "XLF", "Healthcare": "XLV",
        "Industrials": "XLI", "Technology": "XLK", "Utilities": "XLU"}

pools = {}
for sec, veh in SPDR.items():
    mem = sorted(set(sm.loc[sm["sector"] == sec, "ticker"]) - {veh})
    if len(mem) >= SIZE:
        pools[sec] = (veh, mem)
print("=== A. reference class, every complex cut to exactly 11 members ===")
for sec, (veh, mem) in pools.items():
    print(f"  {sec:22s} vehicle {veh}  pool {len(mem)}: {mem}")

# preload every ticker once
allt = sorted({t for _, (v, m) in pools.items() for t in m} |
              {v for v, _ in pools.values()} | set(COMPLEX))
raw = load_prices(allt)
Z = {t: tape_z10(raw[t]["Close"]) for t in raw}

rng = np.random.default_rng(7)
res = []
for sec, (veh, mem) in pools.items():
    pan = close_panel([veh])
    idx = pan.index
    ret = vehicle_ret(pan, [(veh, 1.0)], H)
    valid = ret.notna()
    base = float(ret[valid].mean())
    zz = pd.DataFrame({t: Z[t] for t in mem}).reindex(idx)
    draws = []
    for _ in range(25):
        pick = list(rng.choice(mem, size=SIZE, replace=False))
        sub = zz[pick]
        allv = sub.notna().all(axis=1)
        cnt = (sub >= 2.0).sum(axis=1).where(allv)
        m = (cnt >= K).fillna(False)
        d = idx[m.values & valid.values]
        if len(d) < 5:
            continue
        epi = declusters(d, 10, idx)
        draws.append(100 * (float(ret.loc[epi].mean()) - base))
    if draws:
        res.append({"sector": sec, "vehicle": veh, "n_draws": len(draws),
                    "excess_mean_pp": float(np.mean(draws)),
                    "excess_med_pp": float(np.median(draws)),
                    "excess_min_pp": float(np.min(draws)),
                    "excess_max_pp": float(np.max(draws)),
                    "base_pct": 100 * base})
rd = pd.DataFrame(res).sort_values("excess_mean_pp", ascending=False)
print("\n  identical estimator (5 of 11 at z10>=2 -> long the SPDR, h=5), "
      "25 random 11-member subsamples per sector:")
print(rd.round(3).to_string(index=False))
if len(rd):
    print(f"\n  cross-sector: mean {rd['excess_mean_pp'].mean():+.3f}pp  "
          f"sd {rd['excess_mean_pp'].std():.3f}pp  "
          f"positive {int((rd['excess_mean_pp'] > 0).sum())} of {len(rd)}")
    if "Energy" in set(rd["sector"]):
        e = rd[rd["sector"] == "Energy"].iloc[0]
        rank = int((rd["excess_mean_pp"] >= e["excess_mean_pp"]).sum())
        print(f"  ENERGY (11-member subsamples of its liquid pool): "
              f"{e['excess_mean_pp']:+.3f}pp -> rank {rank} of {len(rd)}, "
              f"P(random sector >= energy) = {rank/len(rd):.3f}")
        print(f"  the DECLARED energy complex scored -0.814pp (b2b); its own "
              f"pool's subsample range is [{e['excess_min_pp']:+.3f}, "
              f"{e['excess_max_pp']:+.3f}]pp over 25 draws")

print("\n  --- noise permutation, same estimator, same sectors ---")
pool2 = []
for r in res:
    pan = close_panel([r["vehicle"]])
    ret = vehicle_ret(pan, [(r["vehicle"], 1.0)], H).dropna()
    pool2.append((r["vehicle"], ret.values, 35, r["base_pct"] / 100.0))
rng2 = np.random.default_rng(42)
maxes = []
for _ in range(3000):
    best = -1e9
    for veh, v, k, base in pool2:
        pick = rng2.choice(len(v), size=k, replace=False)
        best = max(best, 100 * (float(v[pick].mean()) - base))
    maxes.append(best)
maxes = np.array(maxes)
print(f"  {len(pool2)} sectors x 35 episodes, 3000 draws: noise max excess "
      f"mean {maxes.mean():+.3f}pp, p95 {np.percentile(maxes, 95):+.3f}pp, "
      f"max {maxes.max():+.3f}pp")
print(f"  energy's DECLARED complex observed -0.814pp -> "
      f"P(noise max >= it) = {float((maxes >= -0.814).mean()):.3f}")

# ---------------------------------------------------------------------------
# B. is "five names" five objects?
# ---------------------------------------------------------------------------
print("\n=== B. mechanism: how many independent objects are in the count? ===")
pan = close_panel(COMPLEX).dropna()
rets = pan.pct_change().dropna()
C = rets.corr()
iu = np.triu_indices(len(COMPLEX), 1)
print(f"  complex daily-return pairwise corr: mean {C.values[iu].mean():.3f}, "
      f"median {np.median(C.values[iu]):.3f}, "
      f"min {C.values[iu].min():.3f}, max {C.values[iu].max():.3f}")
LIVE = ["XLE", "XOP", "COP", "CVX", "VLO"]
sub = rets[LIVE].corr()
iu2 = np.triu_indices(len(LIVE), 1)
print(f"  today's FIVE firing names {LIVE}: mean pairwise corr "
      f"{sub.values[iu2].mean():.3f}")
print("  the pairs that carry it:")
for i, j in zip(*iu2):
    print(f"    {LIVE[i]:4s}-{LIVE[j]:4s} {sub.values[i, j]:.3f}")
ev = np.linalg.eigvalsh(C.values)[::-1]
neff = (ev.sum() ** 2) / (ev ** 2).sum()
print(f"  PC1 explains {100*ev[0]/ev.sum():.1f}% of complex variance; "
      f"participation-ratio effective N = {neff:.2f} of {len(COMPLEX)} members")
ev5 = np.linalg.eigvalsh(sub.values)[::-1]
print(f"  within today's five: PC1 {100*ev5[0]/ev5.sum():.1f}%, "
      f"effective N = {(ev5.sum()**2)/(ev5**2).sum():.2f} of 5")

# z10 correlation, which is what the count actually thresholds
zpan = pd.DataFrame({t: Z[t] for t in COMPLEX}).reindex(pan.index).dropna()
Cz = zpan.corr()
print(f"  z10 pairwise corr across the complex: mean {Cz.values[iu].mean():.3f}"
      f"  (the count thresholds THIS, not returns)")
print(f"  P(count>=5 | XLE z10>=2) = "
      f"{float(((zpan >= 2).sum(axis=1) >= 5)[zpan['XLE'] >= 2].mean()):.3f}; "
      f"P(XLE z10>=2 | count>=5) = "
      f"{float((zpan['XLE'] >= 2)[(zpan >= 2).sum(axis=1) >= 5].mean()):.3f}")

# and the ETF double-count: how often does a firing XOP/XLE come with its own
# top holdings already firing?
zc = (zpan >= 2)
both_etf = zc["XLE"] & zc["XOP"]
print(f"  XLE and XOP fire together on {int(both_etf.sum())} days; XLE fires "
      f"{int(zc['XLE'].sum())}, XOP {int(zc['XOP'].sum())} -> "
      f"P(XOP | XLE) = {float(zc['XOP'][zc['XLE']].mean()):.3f}")
