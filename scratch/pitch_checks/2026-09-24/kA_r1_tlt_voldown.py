"""kA R1 round 1: long TLT after a <= -1.25% down day on >= 1.5x 63d volume
closing at a fresh 252 low, h=1..5. Pre-specified sign LONG (reversal).
Job: kill it. Must separate from the MOVE-spike sweep (08-18) and show the
at-the-low leg FILTERS (watchlist 5 lesson).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

TK = ["TLT", "IEF", "^TNX", "^MOVE", "SPY"]
raw = load_prices(TK)
tlt = raw["TLT"]
IDX = tlt.index
px = close_panel(TK).reindex(IDX)
c = tlt["Close"]
v = tlt["Volume"].astype(float)

r1 = c.pct_change()
vol_ratio = v / v.rolling(63).mean().shift(1)          # vs trailing 63d, ex-today
vol_ratio_inc = v / v.rolling(63).mean()               # incl. today (contrast)
lo252 = c.rolling(252).min()
at_low = c <= lo252 * (1 + 1e-9)
dist_low = c / lo252 - 1.0
move = px["^MOVE"]
move_chg = rolling_on_valid(move, lambda x: x.pct_change())
move_thr = move_chg.quantile(0.967)

print("LIVE 2026-09-23:")
print(f"  TLT r1 {100*r1.iloc[-1]:+.2f}%  vol ratio (ex-today 63d) {vol_ratio.iloc[-1]:.2f}x"
      f"  incl-today {vol_ratio_inc.iloc[-1]:.2f}x  at 252 low {bool(at_low.iloc[-1])}"
      f"  dist {100*dist_low.iloc[-1]:.3f}%")
print(f"  MOVE chg {100*move_chg.iloc[-1]:+.2f}% (spike thr 96.7 pctile = {100*move_thr:.2f}%)")
a = wilder_atr(tlt["High"].to_numpy(), tlt["Low"].to_numpy(), c.to_numpy())
print(f"  TLT close {c.iloc[-1]:.2f}  Wilder-14 ATR {a[-1]:.3f} ({100*a[-1]/c.iloc[-1]:.2f}%)")

down = r1 <= -0.0125
volx = vol_ratio >= 1.5
cell = (down & volx & at_low).fillna(False)
print(f"\ncell days: {int(cell.sum())}")
print("  dates:", ", ".join(str(d.date()) for d in IDX[cell.values]))

variants = {
    "down<=-1.0 & vol>=1.5 & at low": (r1 <= -0.010) & volx & at_low,
    "down<=-1.5 & vol>=1.5 & at low": (r1 <= -0.015) & volx & at_low,
    "down<=-1.25 & vol>=1.25 & at low": down & (vol_ratio >= 1.25) & at_low,
    "down<=-1.25 & vol>=2.0 & at low": down & (vol_ratio >= 2.0) & at_low,
    "down<=-1.25 & vol>=1.5 & within 0.5% low": down & volx & (dist_low <= 0.005),
    "down<=-1.25 & vol>=1.5 & within 1% low": down & volx & (dist_low <= 0.01),
    "PARENT down<=-1.25 (any)": down,
    "down & vol (any location)": down & volx,
    "down & vol & NOT at low": down & volx & ~at_low,
    "down & at low (no vol gate)": down & at_low,
    "down & at low & vol<1.5 (complement)": down & at_low & (vol_ratio < 1.5),
    "MOVE spike (>=96.7 pctile)": move_chg >= move_thr,
    "cell & MOVE spike": cell & (move_chg >= move_thr),
    "cell & NOT MOVE spike": cell & ~(move_chg >= move_thr),
}

for H in (1, 2, 3, 5):
    battery(px, cell, [("TLT", 1.0)], H, f"R1 LONG TLT: down<=-1.25 & vol>=1.5x & 252 low",
            cost_bps=2.5, variants=variants if H in (2, 5) else None,
            event_kinds=("nfp", "cpi"))

# overlap with the MOVE-spike sweep
ms = (move_chg >= move_thr).reindex(IDX, fill_value=False)
print(f"\nOverlap: cell days {int(cell.sum())}, of which MOVE spike same day "
      f"{int((cell & ms).sum())}")

# midterm split and rising/falling yield regime at h=5 episodes
H = 5
ret = vehicle_ret(px, [("TLT", 1.0)], H, 1)
dd = IDX[cell.values & ret.notna().values]
epi = declusters(dd, H, IDX)
ep = ret.loc[epi].values
mid = np.array([d.year % 4 == 2 for d in epi])
tnx = px["^TNX"]
tnx63 = rolling_on_valid(tnx, lambda x: x - x.shift(63)).reindex(IDX)
rising = (tnx63.loc[epi] > 0).values
show([summarize(ep[mid], "midterm"), summarize(ep[~mid], "non-midterm"),
      summarize(ep[rising], "TNX 63d change > 0"), summarize(ep[~rising], "TNX 63d change <= 0")],
     "h=5 episode splits")
print("episodes h=1..5 table:")
tab = pd.DataFrame({f"h{h}": vehicle_ret(px, [("TLT", 1.0)], h, 1).loc[epi] * 100
                    for h in (1, 2, 3, 5)})
tab["volx"] = vol_ratio.loc[epi].round(2)
tab["r1"] = (100 * r1.loc[epi]).round(2)
print(tab.round(3).to_string())
