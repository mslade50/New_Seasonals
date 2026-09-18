"""Cross the two conflicting splits from 03.

The anchor-move split says the opex session extends a >=8% VIX drop (-2.71%,
20 of 25). The LEVEL split says that at a 15-17 VIX the opex-session decline
disappears (+0.33%, 23-26). Today is both: -12.82% into a 15.44 close. One of
the two conditions has to give, so price the intersection and report which.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, load_events, summarize, show, sign_test  # noqa: E402

px = close_panel(["^VIX"])
idx = px.index
vix = px["^VIX"].dropna()
r1 = vix / vix.shift(1) - 1.0
fwd1 = vix.shift(-1) / vix - 1.0

opex = pd.DatetimeIndex(load_events(["opex"])["date"])
pos = pd.Series(range(len(idx)), index=idx)
anchors = pd.DatetimeIndex(
    [idx[pos[d] - 1] for d in opex if d in pos.index and pos[d] > 0]
).intersection(fwd1.dropna().index)

move = r1.reindex(anchors)
lvl = vix.reindex(anchors)
hard = anchors[(move <= -0.08).values]

print("levels at the 25 anchors with a >=8% VIX drop into opex:")
h = pd.DataFrame({"date": [d.date() for d in hard],
                  "anchor_move_pct": (100 * move.loc[hard]).round(2).values,
                  "anchor_vix": vix.loc[hard].round(2).values,
                  "opex_chg_pct": (100 * fwd1.loc[hard]).round(2).values})
print(h.to_string(index=False))
print(f"\nmin anchor VIX in that set: {vix.loc[hard].min():.2f}  "
      f"median {vix.loc[hard].median():.2f}  "
      f"(today 15.44)")
below20 = hard[(vix.loc[hard] < 20).values]
print(f"  anchors below a 20 VIX: {len(below20)}")
v = fwd1.loc[below20]
k = int((v > 0).sum())
print(f"  record {k}-{len(v) - k} up, mean {100 * v.mean():+.2f}%, "
      f"median {100 * v.median():+.2f}%, sign p(down) = {sign_test(len(v) - k, len(v)):.4f}")
print(f"  dates {[str(d.date()) for d in below20]}")

below17 = hard[(vix.loc[hard] < 17).values]
print(f"\n  anchors below a 17 VIX: {len(below17)} -> "
      f"{[str(d.date()) for d in below17]}")
if len(below17):
    v = fwd1.loc[below17]
    k = int((v > 0).sum())
    print(f"  record {k}-{len(v) - k} up, mean {100 * v.mean():+.2f}%, "
          f"values {(100 * v).round(2).tolist()}")

print("\n=== regression-free check: is the level or the move doing the work? ===")
rows = []
for lab, m in [(">=8% drop, VIX >= 20", (move <= -0.08).values & (lvl >= 20).values),
               (">=8% drop, VIX < 20", (move <= -0.08).values & (lvl < 20).values),
               ("no big drop, VIX < 20", (move > -0.08).values & (lvl < 20).values),
               ("no big drop, VIX >= 20", (move > -0.08).values & (lvl >= 20).values)]:
    d = anchors[m]
    r = summarize(fwd1.loc[d].values, lab)
    if r["n"]:
        k = int((fwd1.loc[d] > 0).sum())
        r["record"] = f"{k}-{r['n'] - k} up"
        r["sign_p_down"] = round(sign_test(r["n"] - k, r["n"]), 4)
    rows.append(r)
show(rows, "opex-session VIX change, move x level")
