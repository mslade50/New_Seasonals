"""Bond vol woke up: ^MOVE +6.13%, the largest move in the 98-name tape.

It came off a low base. MOVE's 21-day return rank sits at 23.4 and its close is
34.5% below its own 52-week high. VIX did something similar in miniature,
+3.40% to 14.92, still 52.0% below its 52-week high.

No trigger fires on this because the sweep has no bond-vol trigger. Cell: MOVE
up 5% or more in a session while its trailing-21-day return rank is in the
bottom third of its own year, declustered to first-in-30-calendar-days.

What follows for TLT, SPY and the yield.

Convention: lag=0 close-to-close, h=1 is the next session.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, cluster_note, era_split, fwd_ret,  # noqa: E402
                       local_control, pct_rank, show, sign_test, summarize)

px = close_panel(["^MOVE", "TLT", "SPY", "^TNX", "^VIX", "HYG"])
px = px[px.index >= "1999-01-01"]
mv = px["^MOVE"].dropna()

r1 = mv.pct_change()
r21 = mv.pct_change(21)
rank21 = pct_rank(r21, 1, 252) if False else r21.rolling(252, min_periods=252).apply(
    lambda w: 100.0 * (w[-1] > w[:-1]).mean(), raw=True)

state = (r1 >= 0.05) & (rank21 <= 33.3)
print(f"today: MOVE {mv.iloc[-1]:.2f}, 1d {100*r1.iloc[-1]:+.2f}%, "
      f"21d rank {rank21.iloc[-1]:.1f}, state={bool(state.iloc[-1])}")
print(f"raw state days: {int(state.sum())} of {int(state.notna().sum())}")


def first_in_calendar_days(mask, days=30):
    out = pd.Series(False, index=mask.index)
    last = None
    for d in mask.index[mask.fillna(False).values]:
        if last is None or (d - last).days > days:
            out.loc[d] = True
        last = d
    return out


epi = first_in_calendar_days(state, 30)
ed = epi.index[epi.values]
hist = ed[ed < pd.Timestamp("2026-08-31")]
print(f"episodes: {len(ed)}, with forward data {len(hist)}")
print("  ", [str(d.date()) for d in hist])

HS = (1, 3, 5, 10, 21)
for tkr in ["TLT", "SPY", "^TNX", "^MOVE"]:
    s = px[tkr].dropna()
    rows = []
    for h in HS:
        f = fwd_ret(s, h).dropna()
        d = pd.DatetimeIndex(hist).intersection(f.index)
        r = summarize(f.loc[d].values, f"h={h}")
        if r.get("n"):
            v = f.loc[d].values
            ctl = local_control(f.index, d, 126)
            r["record"] = f"{int((v>0).sum())}-{int((v<0).sum())}"
            r["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
            r["local_ctl_pct"] = round(100 * f.loc[ctl].mean(), 3)
            r["edge"] = round(r["mean_pct"] - 100 * f.loc[ctl].mean(), 3)
        rows.append(r)
    show(rows, f"{tkr} after a MOVE jump off a low base")

print("\n=== era split + concentration, TLT h=5 and SPY h=5 ===")
for tkr in ["TLT", "SPY"]:
    f = fwd_ret(px[tkr].dropna(), 5).dropna()
    d = pd.DatetimeIndex(hist).intersection(f.index)
    if len(d) >= 4:
        show(era_split(d, f.loc[d].values), f"{tkr} h=5")
        print("  ", cluster_note(d, f.loc[d].values, k=2))

print("\n=== does MOVE keep rising? ===")
for h in HS:
    f = fwd_ret(mv, h).dropna()
    d = pd.DatetimeIndex(hist).intersection(f.index)
    v = f.loc[d].values
    up = int((v > 0).sum())
    print(f"  h={h:2d} n={len(v):2d} {up}-{len(v)-up} higher mean {100*v.mean():+.2f}% "
          f"sign_p {sign_test(up, len(v)):.4f}")

print("\n=== how rare is a 6%+ MOVE day at all? ===")
big = r1[r1 >= 0.06].dropna()
print(f"  n={len(big)} of {int(r1.notna().sum())} sessions ({100*len(big)/r1.notna().sum():.2f}%)")
print("  most recent 6:", [f"{d.date()} {100*v:+.1f}%" for d, v in big.tail(6).items()])
