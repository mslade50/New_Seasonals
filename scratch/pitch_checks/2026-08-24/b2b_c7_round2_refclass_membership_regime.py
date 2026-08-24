"""b2b — C7 round 2: reference class, complex membership, regime, concentration.

Round 1 found the count gradient runs BACKWARDS (count==3 +0.490%, count==5
-0.779% day level) and that conditioning XLE's own thrust on the cluster costs
-0.94pp. This round asks whether any of that is definition-specific:

  A. Complex membership neighbours — drop USO, drop WMB, add XOM/OIH, the
     single-names-only core. The complex was declared before results; these are
     the neighbours, run after.
  B. Reference-class permutation — the IDENTICAL breadth rule (fraction of the
     complex at z10 >= 2 above energy's own 5/11 = 45.5%) applied to every
     sector complex in data/sector_map.parquet, vehicle = that sector's SPDR.
     Where does energy rank, and what does the noise MAX look like?
  C. Regime: midterm (today), SPY 200d, decluster gap, and the de-concentrated
     number (round 1's top-2 episodes are 113% of the total, so the NEGATIVE
     needs the same de-concentration test a positive would get).
  D. The positive band that is NOT today's state, priced against its controls,
     so the near-miss number is exact.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, cluster_note, declusters, load_prices, local_control,
    rolling_on_valid, show, sign_test, summarize, vehicle_ret,
)

pd.set_option("display.width", 240)
H = 5
FRAC = 5 / 11.0          # energy's live breadth, the estimator's one free knob
COMPLEX = ["XLE", "XOP", "USO", "COP", "CVX", "VLO", "OXY", "SLB", "EOG",
           "HAL", "WMB"]


def tape_z10(close: pd.Series, n: int = 10) -> pd.Series:
    r = close.pct_change(n)
    v = close.pct_change().rolling(21).std()
    return r / (v * np.sqrt(n))


def breadth_cell(members: list[str], vehicle: str, frac: float,
                 h: int = H) -> dict:
    """Episode summary of: long `vehicle` when >= frac of `members` carry
    z10 >= 2 on a day where every member has a valid z10."""
    tk = sorted(set(members) | {vehicle})
    raw = load_prices(tk)
    have = [t for t in members if t in raw]
    if vehicle not in raw or len(have) < 6:
        return {"label": f"{vehicle} <{len(have)} members>", "n": 0}
    pan = close_panel(tk).dropna(subset=[vehicle])
    idx = pan.index
    z = pd.DataFrame({t: tape_z10(raw[t]["Close"]) for t in have}).reindex(idx)
    allv = z.notna().all(axis=1)
    cnt = (z >= 2.0).sum(axis=1).where(allv)
    m = ((cnt / len(have)) >= frac).fillna(False)
    ret = vehicle_ret(pan, [(vehicle, 1.0)], h)
    valid = ret.notna()
    d = idx[m.values & valid.values]
    if len(d) < 5:
        return {"label": vehicle, "n": len(d)}
    epi = declusters(d, 10, idx)
    r = summarize(ret.loc[epi].values, vehicle)
    r["n_days"] = len(d)
    r["members"] = len(have)
    r["base_pct"] = round(100 * float(ret[valid].mean()), 3)
    r["excess_pp"] = round(r["mean_pct"] - r["base_pct"], 3)
    r["today"] = bool(m.iloc[-1])
    return r


# ---------------------------------------------------------------------------
print("=== A. complex membership neighbours (vehicle XLE, count>=5-equivalent) ===")
NEIGH = {
    "declared 11 (count>=5)": COMPLEX,
    "drop USO (10, k>=5)": [t for t in COMPLEX if t != "USO"],
    "drop WMB (10, k>=5)": [t for t in COMPLEX if t != "WMB"],
    "drop the two ETFs XOP+USO (9)": [t for t in COMPLEX if t not in ("XOP", "USO")],
    "single names only (8)": ["COP", "CVX", "VLO", "OXY", "SLB", "EOG", "HAL", "WMB"],
    "add XOM + OIH (13)": COMPLEX + ["XOM", "OIH"],
    "sector-map energy ex-XLE (12)": ["COP", "CVX", "EOG", "HAL", "OIH", "OXY",
                                      "SLB", "USO", "VLO", "WMB", "XOM", "XOP"],
}
rows = []
for lbl, mem in NEIGH.items():
    r = breadth_cell(mem, "XLE", FRAC)
    r["label"] = lbl
    rows.append(r)
show(rows, f"membership neighbours, long XLE h={H}, breadth >= {FRAC:.1%}")

print("\n  vehicle neighbours on the DECLARED complex:")
vr = []
for veh in ("XLE", "XOP", "USO", "OIH", "SPY"):
    r = breadth_cell(COMPLEX, veh, FRAC)
    vr.append(r)
show(vr, "vehicle swap")

# ---------------------------------------------------------------------------
print("\n=== B. reference class: the same breadth rule on every sector ===")
sm = pd.read_parquet("data/sector_map.parquet")
mp = pd.read_parquet("data/master_prices.parquet", columns=["ticker", "date"])
nbar = mp.groupby("ticker").size()
sm = sm[sm["ticker"].map(nbar).fillna(0) >= 5000]
SPDR = {"Basic Materials": "XLB", "Consumer Cyclical": "XLY",
        "Consumer Defensive": "XLP", "Energy": "XLE",
        "Financial Services": "XLF", "Healthcare": "XLV",
        "Industrials": "XLI", "Technology": "XLK", "Utilities": "XLU"}
ref = []
for sec, veh in SPDR.items():
    mem = sorted(set(sm.loc[sm["sector"] == sec, "ticker"]) - {veh})
    if len(mem) < 8:
        continue
    r = breadth_cell(mem, veh, FRAC)
    r["label"] = f"{sec} ({veh})"
    ref.append(r)
show(ref, f"reference class: breadth >= {FRAC:.1%} of the sector complex -> long its SPDR")
rd = pd.DataFrame([r for r in ref if r.get("n", 0) >= 5])
if len(rd):
    rd = rd.sort_values("excess_pp", ascending=False)
    print(f"  cross-sector excess: mean {rd['excess_pp'].mean():+.3f}pp  "
          f"sd {rd['excess_pp'].std():.3f}pp  "
          f"positive {int((rd['excess_pp'] > 0).sum())} of {len(rd)}")
    if "Energy (XLE)" in set(rd["label"]):
        e = rd[rd["label"] == "Energy (XLE)"].iloc[0]
        rank = int((rd["excess_pp"] >= e["excess_pp"]).sum())
        print(f"  ENERGY: excess {e['excess_pp']:+.3f}pp -> rank {rank} of "
              f"{len(rd)};  P(a random sector >= energy) = {rank/len(rd):.3f}")
    print(rd[["label", "n", "n_days", "members", "mean_pct", "base_pct",
              "excess_pp", "hit", "today"]].to_string(index=False))

# noise permutation over the same nine sectors
print("\n  --- permutation MAX out of pure noise, same estimator ---")
rng = np.random.default_rng(42)
raws, panels = {}, {}
draw_pool = []
for r in ref:
    if r.get("n", 0) < 5:
        continue
    veh = r["label"].split("(")[1].rstrip(")")
    pan = close_panel([veh])
    ret = vehicle_ret(pan, [(veh, 1.0)], H).dropna()
    draw_pool.append((veh, ret, r["n"], r["base_pct"] / 100.0))
maxes = []
for _ in range(2000):
    best = -1e9
    for veh, ret, k, base in draw_pool:
        pick = rng.choice(len(ret), size=k, replace=False)
        best = max(best, 100 * (float(ret.values[pick].mean()) - base))
    maxes.append(best)
maxes = np.array(maxes)
eobs = float(rd.loc[rd["label"] == "Energy (XLE)", "excess_pp"].iloc[0])
print(f"  {len(draw_pool)} sectors, 2000 draws. noise max excess: mean "
      f"{maxes.mean():+.3f}pp, p95 {np.percentile(maxes, 95):+.3f}pp, "
      f"max {maxes.max():+.3f}pp")
print(f"  energy observed {eobs:+.3f}pp -> P(noise max >= energy) = "
      f"{float((maxes >= eobs).mean()):.3f}")

# ---------------------------------------------------------------------------
print("\n=== C. regime, decluster gap, de-concentration ===")
tk = sorted(set(COMPLEX + ["SPY"]))
raw = load_prices(tk)
pan = close_panel(tk).dropna(subset=["XLE", "SPY"])
IDX = pan.index
z = pd.DataFrame({t: tape_z10(raw[t]["Close"]) for t in COMPLEX}).reindex(IDX)
allv = z.notna().all(axis=1)
cnt = (z >= 2.0).sum(axis=1).where(allv)
TRIG = (cnt >= 5).fillna(False)
ret = vehicle_ret(pan, [("XLE", 1.0)], H)
valid = ret.notna()
d = IDX[TRIG.values & valid.values]
epi = declusters(d, 10, IDX)
v = ret.loc[epi].values

mid = pd.DatetimeIndex(epi).year % 4 == 2
show([summarize(v[mid], f"MIDTERM (today's regime, N={int(mid.sum())})"),
      summarize(v[~mid], "non-midterm")], "midterm split")
spy200 = rolling_on_valid(pan["SPY"], lambda x: x.rolling(200).mean())
bull = (pan["SPY"] > spy200).reindex(epi).values
show([summarize(v[bull], "SPY above its 200d (today)"),
      summarize(v[~bull], "SPY below")], "regime split")

print("  decluster-gap sensitivity:")
for gap in (1, 5, 10, 21, 63):
    e2 = declusters(d, gap, IDX)
    w = ret.loc[e2].values
    print(f"   gap={gap:3d} td: N={len(e2):3d}  mean {100*w.mean():+.3f}%  "
          f"hit {100*(w > 0).mean():.1f}%  median {100*np.median(w):+.3f}%")

order = np.argsort(v)
print(f"\n  de-concentration (the negative gets the same test a positive would):")
for k in (0, 1, 2, 3):
    keep = np.ones(len(v), bool)
    keep[order[:k]] = False
    print(f"   drop the {k} worst episodes: N={int(keep.sum())} mean "
          f"{100*v[keep].mean():+.3f}%  hit {100*(v[keep] > 0).mean():.1f}%")
wins = int((v > 0).sum())
print(f"  record {wins}-{len(v)-wins}, sign p = {sign_test(wins, len(v)):.4f}  "
      f"(vs XLE's own conditional up-rate "
      f"{100*float((ret[valid] > 0).mean()):.1f}%)")
print(f"  sign test against XLE's OWN up-rate: p = "
      f"{sign_test(wins, len(v), float((ret[valid] > 0).mean())):.4f}")

# ---------------------------------------------------------------------------
print("\n=== D. the band that IS positive, and how far today sits from it ===")
for lo, hi in ((2, 4), (2, 3), (3, 3), (2, 11), (5, 11)):
    m = ((cnt >= lo) & (cnt <= hi)).fillna(False)
    dd = IDX[m.values & valid.values]
    if len(dd) < 5:
        continue
    ee = declusters(dd, 10, IDX)
    w = ret.loc[ee].values
    loc = local_control(IDX[valid.values], dd)
    print(f"  count in [{lo},{hi}]: N_days={len(dd):4d} N_epi={len(ee):3d} "
          f"mean {100*w.mean():+.3f}%  hit {100*(w > 0).mean():.1f}%  "
          f"vs all-days {100*float(ret[valid].mean()):+.3f}%  "
          f"vs local {100*float(ret.loc[loc].mean()):+.3f}%  "
          f"excess-local {100*(w.mean() - ret.loc[loc].mean()):+.3f}pp")
print(f"  TODAY count = {int(cnt.iloc[-1])}")
