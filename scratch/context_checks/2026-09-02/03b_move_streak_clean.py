"""03 computed its ranks on a union-calendar panel and got VIX 63d rank 48.0
against the engine's 7.1. Redo every rank on the instrument's OWN index, then
re-run the ^MOVE up-streak cell with era split and concentration.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = load_prices(["^MOVE", "^VIX", "SPY", "TLT", "^TNX", "^VIX3M"])

print("=== live state, each on its own index ===")
for t, n in [("^MOVE", 5), ("^VIX", 63), ("^VIX", 21), ("^TNX", 5)]:
    c = px[t]["Close"].dropna()
    rk = pct_rank(c, n)
    r = c.dropna().pct_change(n, fill_method=None)
    print(f"{t:<7} {n}d return {100*r.iloc[-1]:+7.2f}%  rank {rk.iloc[-1]:5.1f}  "
          f"last bar {c.index[-1].date()}  close {c.iloc[-1]:.2f}")

mv = px["^MOVE"]["Close"].dropna()
vx = px["^VIX"]["Close"].dropna()
print(f"\n^MOVE streak of up closes ending today: ", end="")
up = mv > mv.shift(1)
k = 0
for u in up.values[::-1]:
    if u:
        k += 1
    else:
        break
print(k)

# --- the ^MOVE 5+ up-streak cell, on MOVE's own index ---
streak = up.groupby((~up).cumsum()).cumsum()
trig = mv.index[(streak >= 5) & up]
trig = trig[trig < mv.index[-1]]
print(f"\n=== ^MOVE 5+ up closes: {len(trig)} sessions since {mv.index[0].date()} ===")

for h in (5, 10):
    epi = declusters(trig, h, mv.index)
    r = fwd_ret(mv, h)
    v = r.reindex(epi).dropna()
    base = r.dropna()
    rows = []
    for lab, m in [("all", np.ones(len(v), bool)),
                   ("pre-2018", np.asarray(v.index < pd.Timestamp("2018-01-01"))),
                   ("2018+", np.asarray(v.index >= pd.Timestamp("2018-01-01")))]:
        vv = v[m]
        s = summarize(vv.values, lab)
        u = int((vv > 0).sum())
        s["up"], s["down"] = u, len(vv) - u
        s["sign_p"] = round(sign_test(len(vv) - u, len(vv)), 4) if len(vv) else None
        rows.append(s)
    lc = local_control(mv.index, epi, 126)
    s = summarize(r.reindex(lc).dropna().values, "local ctl +/-126td")
    rows.append(s)
    s = summarize(base.values, "all days")
    rows.append(s)
    show(rows, f"^MOVE forward h={h} after a 5+ up streak ({len(epi)} episodes)")
    print(cluster_note(v.index, v.values, k=2))

# does the streak length matter, and what about the level
print("\n=== conditioned on where MOVE sits (live: %.1f) ===" % mv.iloc[-1])
epi = declusters(trig, 5, mv.index)
r5 = fwd_ret(mv, 5)
lvl = rolling_on_valid(mv, lambda x: x.rolling(252).rank(pct=True) * 100)
print("live MOVE level rank (252d): %.1f" % lvl.iloc[-1])
for lab, m in [("level rank <= 60", lvl.reindex(epi) <= 60),
               ("level rank > 60", lvl.reindex(epi) > 60)]:
    idx = epi[m.fillna(False).values]
    v = r5.reindex(idx).dropna()
    s = summarize(v.values, lab)
    u = int((v > 0).sum())
    s["up"], s["down"] = u, len(v) - u
    s["sign_p"] = round(sign_test(len(v) - u, len(v)), 4) if len(v) else None
    show([s])

# --- SPY into the same window, since a bond-vol pop is the interesting part ---
spy = px["SPY"]["Close"].dropna()
epi = declusters(trig, 21, mv.index)
rows = []
for h in (5, 21):
    r = fwd_ret(spy, h)
    v = r.reindex(epi).dropna()
    s = summarize(v.values, f"SPY h={h}")
    s["ctl_all_pct"] = round(100 * r.dropna().mean(), 3)
    u = int((v > 0).sum())
    s["up"], s["down"] = u, len(v) - u
    s["sign_p"] = round(sign_test(max(u, len(v) - u), len(v)), 4)
    rows.append(s)
show(rows, f"SPY after a MOVE 5+ up streak ({len(epi)} episodes, 21td decluster)")
