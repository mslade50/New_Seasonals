"""Today: the S&P rose 1.14% the session after a decision day on which it fell.

Yesterday's brief published the h1 cell (the session after a down decision
day). Today is that session and it resolved in the top tail, so the honest
follow-on is what happens AFTER a rebound that size, which is a different
question and a different anchor. h1 from here is tomorrow, the witching
session.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, load_events, summarize, show, sign_test, cluster_note,
)

px = close_panel(["^GSPC", "SPY", "IWM", "^VIX", "QQQ"])
idx = px.index
g = px["^GSPC"].dropna()
r1 = g / g.shift(1) - 1.0
f1 = g.shift(-1) / g - 1.0
f5 = g.shift(-5) / g - 1.0

fomc = pd.DatetimeIndex(load_events(["fomc_decision"])["date"])
pos = pd.Series(range(len(idx)), index=idx)
# anchor = the session AFTER a decision, i.e. today
after = pd.DatetimeIndex(
    [idx[pos[d] + 1] for d in fomc if d in pos.index and pos[d] + 1 < len(idx)])
after = after.intersection(r1.dropna().index)
dec_down = pd.DatetimeIndex(
    [a for a in after if r1.get(idx[pos[a] - 1], np.nan) < 0])
print(f"post-decision sessions: {len(after)}; decision day was down: {len(dec_down)}")
print(f"today's move: {100 * r1.iloc[-1]:+.2f}%; "
      f"decision day {100 * r1.iloc[-2]:+.2f}%")

cells = [
    ("decision down, next session up 1%+", dec_down[(r1.reindex(dec_down) >= 0.01).values]),
    ("decision down, next session up 0-1%", dec_down[((r1.reindex(dec_down) >= 0) & (r1.reindex(dec_down) < 0.01)).values]),
    ("decision down, next session down", dec_down[(r1.reindex(dec_down) < 0).values]),
    ("any post-decision session up 1%+", after[(r1.reindex(after) >= 0.01).values]),
]
for h, f in [(1, f1), (5, f5)]:
    rows = []
    for lab, d in cells:
        e = pd.DatetimeIndex(d).intersection(f.dropna().index)
        r = summarize(f.loc[e].values, lab)
        if r["n"]:
            k = int((f.loc[e] > 0).sum())
            r["record"] = f"{k}-{r['n'] - k} up"
            r["sign_p_down"] = round(sign_test(r["n"] - k, r["n"]), 4)
        rows.append(r)
    rows.append(summarize(f.dropna().values, "CTL all sessions"))
    show(rows, f"^GSPC h={h} from the post-decision session")

d = cells[0][1]
print(f"\ndecision-down + 1%+ rebound, dates ({len(d)}):")
tbl = pd.DataFrame({"date": [x.date() for x in d],
                    "decision_pct": (100 * r1.reindex([idx[pos[x] - 1] for x in d])).round(2).values,
                    "rebound_pct": (100 * r1.loc[d]).round(2).values,
                    "next_pct": (100 * f1.reindex(d)).round(2).values,
                    "next5_pct": (100 * f5.reindex(d)).round(2).values})
print(tbl.to_string(index=False))
v = f1.reindex(d).dropna()
k = int((v > 0).sum())
print(f"  h1 record {k}-{len(v) - k} up, mean {100 * v.mean():+.2f}%, "
      f"median {100 * v.median():+.2f}%, sign p(down) = {sign_test(len(v) - k, len(v)):.4f}")
for lab, m in [("pre-2018", d < pd.Timestamp("2018-01-01")),
               ("2018+", d >= pd.Timestamp("2018-01-01"))]:
    vv = f1.reindex(d[m]).dropna()
    kk = int((vv > 0).sum())
    print(f"    {lab:<10} n={len(vv):<3} mean {100 * vv.mean():+.3f}%  {kk}-{len(vv) - kk} up")
print(f"  concentration h1: {cluster_note(v.index, v.values, k=2)}")

print("\n=== and with the VIX down 8%+ on the rebound session (today: -12.8%) ===")
vx = px["^VIX"].dropna()
rv = vx / vx.shift(1) - 1.0
d2 = pd.DatetimeIndex([x for x in d if rv.get(x, 0) <= -0.08])
print(f"  n={len(d2)}: {[str(x.date()) for x in d2]}")
for h, f in [(1, f1), (5, f5)]:
    v = f.reindex(d2).dropna()
    k = int((v > 0).sum())
    print(f"  h={h}: n={len(v)} mean {100 * v.mean():+.2f}% median {100 * v.median():+.2f}% "
          f"{k}-{len(v) - k} up, sign p(down) = {sign_test(len(v) - k, len(v)):.4f}")
