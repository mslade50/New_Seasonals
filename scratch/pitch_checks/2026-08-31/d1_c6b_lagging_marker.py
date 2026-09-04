"""C6 round 2 -- is the SKEW-up/VIX-down state a LAGGING marker?

Round 1's offset ladder ran +0.715% at k=-5, +0.968% at k=-2 (t 5.95) and
collapsed to +0.055% at the anchor. That is the registry's lagging-marker
shape. This probe states it directly: what has SPY already DONE by the time
the state prints?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

raw = pd.read_parquet(ROOT / "data" / "master_prices.parquet")
raw["date"] = pd.to_datetime(raw["date"])


def ser(t):
    g = raw[raw.ticker == t].sort_values("date").set_index("date")
    return g[~g.index.duplicated(keep="last")]["Close"]


spy, skew, vix = ser("SPY"), ser("^SKEW"), ser("^VIX")
IDX = spy.index
sk, vx = skew.reindex(IDX), vix.reindex(IDX)
JOINT = (((sk / sk.shift(1) - 1) >= 0.03) & ((vx / vx.shift(1) - 1) < 0)).fillna(False)
trig = IDX[JOINT.values]
epi = declusters(trig, 5, IDX)

print("=" * 78)
print("TRAILING SPY RETURN AT THE MOMENT THE STATE PRINTS (episodes, N=%d)" % len(epi))
print("=" * 78)
rows = []
for k in (1, 2, 3, 5, 10, 21):
    tr = (spy / spy.shift(k) - 1.0)
    rows.append({"trailing_k": k,
                 "cond_mean_pct": round(100 * float(tr.loc[epi].mean()), 3),
                 "all_days_pct": round(100 * float(tr.dropna().mean()), 3),
                 "cond_hit": round(100 * float((tr.loc[epi] > 0).mean()), 1),
                 "all_hit": round(100 * float((tr.dropna() > 0).mean()), 1)})
show(rows, "SPY return in the k sessions ENDING on the trigger day")

print("\nRead: the state prints AFTER the move, not before it. Round 1's ladder")
print("pays +0.968% entered 2 sessions BEFORE the trigger and +0.055% entered")
print("at it; the trailing table above is why.")

print("\n" + "=" * 78)
print("AND: is the CO-MOVEMENT framing even the right sign?")
print("  'fear' form = SKEW up >= 3% with the VIX also UP")
print("=" * 78)
FEAR = (((sk / sk.shift(1) - 1) >= 0.03) & ((vx / vx.shift(1) - 1) > 0)).fillna(False)
px = pd.DataFrame({"SPY": spy}).dropna()
rows = []
for h in (1, 3, 5, 10):
    r = vehicle_ret(px, [("SPY", 1.0)], h, 1)
    v = r.notna()
    for lbl, m in (("POSITIONING (vix down)", JOINT), ("FEAR (vix up)", FEAR)):
        t = IDX[m.values & v.values]
        e = declusters(t, 5, IDX)
        rr = summarize(r.loc[e].values, f"h={h} {lbl}")
        rr["all_days_pct"] = round(100 * float(r.dropna().mean()), 3)
        rows.append(rr)
show(rows, "the two co-movement halves, long SPY, episodes")
