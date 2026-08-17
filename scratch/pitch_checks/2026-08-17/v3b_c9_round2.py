"""C9 round 2 -- the ONE cell that did not die in round 1.

Round 1 (v3_c9_iwm_high_opex.py) killed the directional leg at every horizon
(long IWM: edge -0.15 / -0.14 / -0.16 pp against the local control) and
found the opex gate to be an INVERTER on it (state & opex week h=10
-0.250% against state-without-opex +0.373%).

The single cell left standing was the RELATIVE leg at h=4:
  IWM long / SPY short, IWM within 0.10% of its 52w high while SPY is more
  than 0.10% off, inside opex week -> +0.217% over 26 episodes, 53.8% hit.
Round 2 tears exactly that down: concentration, threshold neighbours, an
anchor placebo ladder on the opex week itself, and cost.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

H = 4
COST_BPS = 8.0   # two legs x ~4 bps round trip

px = close_panel(["IWM", "SPY"])
ALL = px.index
oh = {t: 1.0 - px[t] / px[t].rolling(252).max() for t in ("IWM", "SPY")}
ret = vehicle_ret(px, [("IWM", 1.0), ("SPY", -1.0)], H, 1)
valid = ret.dropna().index

opex = pd.to_datetime(load_events(["opex"])["date"].unique())


def week_mask(off_lo: int, off_hi: int) -> pd.Series:
    """Sessions from opex+off_lo through opex+off_hi (td offsets)."""
    m = pd.Series(False, index=ALL)
    for d in opex:
        p = ALL.searchsorted(d)
        if p >= len(ALL):
            continue
        m.iloc[max(0, p + off_lo):max(0, p + off_hi) + 1] = True
    return m


state = (100 * oh["IWM"] <= 0.10) & (100 * oh["SPY"] > 0.10)
cell = state & week_mask(-4, 0)


def stat(mask, label):
    d = pd.DatetimeIndex(ALL[mask.values]).intersection(valid)
    if len(d) == 0:
        return {"label": label, "n": 0}
    e = declusters(d, max(H, 5), valid)
    r = summarize(ret.loc[e].values, label)
    r["n_days"] = len(d)
    return r


print("=" * 78)
print("1. CONCENTRATION of the +0.217% cell")
print("=" * 78)
d = pd.DatetimeIndex(ALL[cell.values]).intersection(valid)
epi = declusters(d, max(H, 5), valid)
ep = ret.loc[epi].values
print(f"  N_ep={len(ep)}  mean {100*ep.mean():+.3f}%  record "
      f"{int((ep>0).sum())}-{int((ep<=0).sum())}")
base = float((ret.loc[valid] > 0).mean())
print(f"  sign p vs coin {sign_test(int((ep>0).sum()), len(ep)):.4f};  "
      f"vs the spread's OWN {100*base:.1f}% base rate "
      f"{sign_test(int((ep>0).sum()), len(ep), p=base):.4f}")
print(f"  {cluster_note(epi, ep)}")
srt = np.sort(ep)
print(f"  drop-best {100*srt[:-1].mean():+.3f}%  drop-best-2 "
      f"{100*srt[:-2].mean():+.3f}%  drop-best-3 {100*srt[:-3].mean():+.3f}%")
loyo = {int(y): round(100 * ep[epi.year != y].mean(), 3)
        for y in sorted(set(epi.year))}
print(f"  LOYO means {loyo}\n  LOYO floor {min(loyo.values()):+.3f}%")
print(f"  cost: {100*ep.mean()*100:.1f} bps against a {COST_BPS} bps two-leg "
      f"round trip = {100*ep.mean()*100/COST_BPS:.1f}x (need >=5x)")
print("  episode dates:", ", ".join(str(x.date()) for x in epi))

print("\n" + "=" * 78)
print("2. ANCHOR PLACEBO LADDER -- slide the 5-session window around opex")
print("=" * 78)
rows = []
for k in range(-9, 6):
    rows.append(stat(state & week_mask(k, k + 4), f"opex{k:+d}..{k+4:+d}"))
show(rows, "2. window placebo (episodes)")
real = [r for r in rows if r["label"] == "opex-4..+0"][0]["mean_pct"]
better = sum(1 for r in rows if r.get("mean_pct", -9e9) > real)
print(f"  the TRUE window (opex-4..0) ranks {better+1} of {len(rows)}")

print("\n" + "=" * 78)
print("3. THRESHOLD NEIGHBOURS of the price state (opex week held fixed)")
print("=" * 78)
rows = []
for a in (0.05, 0.10, 0.25, 0.50, 1.00):
    for b in (0.10, 0.25, 0.50, 1.00):
        rows.append(stat((100 * oh["IWM"] <= a) & (100 * oh["SPY"] > b)
                         & week_mask(-4, 0), f"IWM<={a} SPY>{b}"))
show(rows, "3. threshold grid (episodes)")

print("\n" + "=" * 78)
print("4. HORIZON PROFILE of the same cell (sign stability)")
print("=" * 78)
show(horizon_scan(px, pd.DatetimeIndex(ALL[cell.values]),
                  [("IWM", 1.0), ("SPY", -1.0)],
                  hs=(1, 2, 3, 4, 5, 8, 10)), "4. horizon scan")

print("\n" + "=" * 78)
print("5. ERA SPLIT of the cell itself")
print("=" * 78)
show(era_split(epi, ep), "5. pre-2018 / 2018+")
byyr = pd.Series(ep, index=epi.year).groupby(level=0).agg(["mean", "count"])
byyr["mean"] = (100 * byyr["mean"]).round(3)
print(byyr.to_string())
