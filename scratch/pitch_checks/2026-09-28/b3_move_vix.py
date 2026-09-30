"""C3 round 1: ^MOVE 21d return rank >= 95 while equity vol sleeps (^VIX 21d abs-range
percentile <= 15 over 504d, as the site flag, OR ^VIX < 16). Claim: rates vol
transmits to equity vol. Trades: SHORT SVXY residual vs 1.48x SPY (2018-03+), or
SHORT SPY, h=1..10. Pre-2018 object: forward ^VIX % change and ^VIX/^VIX3M change.
Parents: MOVE leg alone, calm-VIX leg alone, all days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

raw = load_prices(["SPY", "SVXY", "^VIX", "^VIX3M", "^MOVE"])
IDX = raw["SPY"]["Close"].index
IDX = IDX[IDX >= "2003-01-01"]
px = pd.DataFrame({t: raw[t]["Close"].reindex(IDX).ffill(limit=2) for t in ["SPY", "SVXY", "^VIX", "^VIX3M"]})
px.loc[px.index < "2018-03-01", "SVXY"] = np.nan
px["VTS"] = px["^VIX"] / px["^VIX3M"]

mv = raw["^MOVE"]["Close"].dropna()
MR21 = pct_rank(mv, 21, 252).reindex(IDX).ffill(limit=2)
vix = raw["^VIX"]["Close"].dropna()
rng = vix.rolling(21).max() - vix.rolling(21).min()
arr = rng.to_numpy()
pc = np.full(len(arr), np.nan)
for i in range(504, len(arr)):
    w = arr[i - 504:i + 1]
    v = w[~np.isnan(w)]
    if len(v) >= 403:
        pc[i] = (v[:-1] < v[-1]).sum() / (len(v) - 1) * 100
RP = pd.Series(pc, index=vix.index).reindex(IDX).ffill(limit=2)
VX = px["^VIX"]

calm = (RP <= 15) | (VX < 16)
mvg = MR21 >= 95
cell = mvg & calm
print(f"LIVE {IDX[-1].date()}: MOVE r21 {MR21.iloc[-1]:.1f}, VIX rangepct {RP.iloc[-1]:.1f}, VIX {VX.iloc[-1]:.2f} "
      f"-> cell {bool(cell.iloc[-1])}")
print(f"days: MOVE leg {int(mvg.sum())}, calm leg {int(calm.sum())}, joint {int(cell.sum())}; "
      f"joint 2018-03+ {int(cell[cell.index >= '2018-03-01'].sum())}")

B = 1.48
VEH = {"shortSVXYres": [("SVXY", -1.0), ("SPY", B)], "shortSPY": [("SPY", -1.0)],
       "longVIX": [("^VIX", 1.0)], "longVTS": [("VTS", 1.0)]}


def epi_stats(mask, legs, h, gap=10):
    ret = vehicle_ret(px, legs, h, 1)
    valid = ret.notna()
    s = px.index[mask.reindex(px.index, fill_value=False).to_numpy() & valid.to_numpy()]
    if len(s) == 0:
        return {"n": 0}, np.array([]), s
    e = declusters(s, gap, px.index)
    v = ret.loc[e].to_numpy()
    return summarize(v), v, e


rows = []
for name, legs in VEH.items():
    for h in (1, 2, 3, 5, 10):
        ret = vehicle_ret(px, legs, h, 1)
        base = ret.dropna()
        if name == "shortSVXYres":
            base = base[base.index >= "2018-03-01"]
        r = {"veh": name, "h": h}
        for lab, m in (("cell", cell), ("MOVE only", mvg & ~calm), ("calm only", calm & ~mvg), ("calm all", calm)):
            s, v, e = epi_stats(m, legs, h)
            w = int((v > 0).sum()) if len(v) else 0
            r[lab] = f"{s.get('mean_pct', np.nan):+.3f} {w}-{len(v)-w}" if s["n"] else "n/a"
        r["alldays"] = f"{100*base.mean():+.3f}"
        rows.append(r)
print("\n=== EPISODES (declustered 10td), % mean + record, pitched sign ===")
print(pd.DataFrame(rows).to_string(index=False))

battery(px, cell, VEH["shortSPY"], 5, "C3 short SPY h=5", 3.0, min_gap=10,
        variants={"MOVE r21>=90 & calm": (MR21 >= 90) & calm, "MOVE r21>=97.5 & calm": (MR21 >= 97.5) & calm,
                  "MOVE>=95 & VIX<16 only": mvg & (VX < 16), "MOVE>=95 & rangepct<=15 only": mvg & (RP <= 15),
                  "MOVE>=95 & VIX<14": mvg & (VX < 14)})
battery(px, cell & (px.index >= "2018-03-01"), VEH["shortSVXYres"], 5, "C3 short SVXY residual h=5 (2018-03+)",
        5.0, min_gap=10)
battery(px, cell, VEH["longVIX"], 5, "C3 long ^VIX (object, not tradeable) h=5", 0.0001, min_gap=10)

s, v, e = epi_stats(cell, VEH["shortSPY"], 5)
print("\nC3 episodes (short SPY h5 / long VIX h5):")
rv = vehicle_ret(px, VEH["longVIX"], 5, 1)
rs = vehicle_ret(px, VEH["shortSVXYres"], 5, 1)
for d in e:
    print(f"  {d.date()}  MOVEr21 {MR21[d]:.1f} VIX {VX[d]:.2f} rp {RP[d]:.1f}  shortSPY {100*vehicle_ret(px, VEH['shortSPY'], 5, 1)[d]:+.2f}  "
          f"VIX {100*rv[d]:+.2f}  shortSVXYres {100*rs[d]:+.2f}")
