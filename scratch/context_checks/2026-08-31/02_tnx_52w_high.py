"""Tonight's lead candidate: the 10-year yield closed at a 52-week high.

^TNX closed 4.758 on 2026-08-31, its first 52-week high in 30+ calendar days,
+0.81% on the session and +6.32% over 63 sessions. ^FVX printed one too.

The engine's base cell (n=24, era_stable False) only tells me what the yield
itself did next. The question worth Scott's attention is the cross-asset one
the sweep does not compute: when the long end breaks to a new yield high,
what did EQUITIES and DURATION do over the following sessions, and is that
different from an ordinary day?

Convention: context lane, lag=0 close-to-close from the anchor close, so h=1
is the next session. Trigger definition copied from build_context_state's
_first_in_calendar_days(_high_52w(close), 30) so this reproduces P1 exactly.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, cluster_note, era_split, fwd_ret,  # noqa: E402
                       local_control, sign_test, show, summarize)

TICKERS = ["^TNX", "SPY", "TLT", "HYG", "IEF", "^VIX", "DX-Y.NYB", "GC=F"]
px = close_panel(TICKERS)
px = px[px.index >= "1999-01-01"]

tnx = px["^TNX"].dropna()


def high_52w(close):
    return close >= close.rolling(252, min_periods=252).max()


def first_in_calendar_days(mask, days=30):
    out = pd.Series(False, index=mask.index)
    last = None
    for d in mask.index[mask.fillna(False).values]:
        if last is None or (d - last).days > days:
            out.loc[d] = True
        last = d
    return out


trig = first_in_calendar_days(high_52w(tnx), 30)
dates = trig.index[trig.values]
print(f"^TNX first-52w-high-in-30+-days events since {tnx.index[0].date()}: {len(dates)}")
print("  all:", [str(d.date()) for d in dates])

# today's event is the last one; exclude it from the history (no forward data)
hist = dates[dates < pd.Timestamp("2026-08-31")]
print(f"  with forward data: {len(hist)}\n")

HS = (1, 2, 3, 5, 10, 21)
for tkr in ["SPY", "TLT", "HYG", "^TNX", "^VIX"]:
    s = px[tkr].dropna()
    rows = []
    for h in HS:
        f = fwd_ret(s, h)
        d = pd.DatetimeIndex(hist).intersection(f.dropna().index)
        r = summarize(f.loc[d].values, f"h={h}")
        if r.get("n"):
            base = f.dropna()
            ctl = local_control(f.dropna().index, d, 126)
            r["all_days_pct"] = round(100 * base.mean(), 3)
            r["local_ctl_pct"] = round(100 * f.loc[ctl].mean(), 3)
            r["edge_vs_local"] = round(r["mean_pct"] - 100 * f.loc[ctl].mean(), 3)
        rows.append(r)
    show(rows, f"{tkr} after a ^TNX 52-week yield high")

# The headline arm: pin down whichever horizon is sharpest for SPY.
print("\n=== SPY detail, by horizon: record and exact sign test ===")
s = px["SPY"].dropna()
for h in HS:
    f = fwd_ret(s, h).dropna()
    d = pd.DatetimeIndex(hist).intersection(f.index)
    v = f.loc[d].values
    up = int((v > 0).sum())
    n = len(v)
    print(f"  h={h:2d}  n={n:2d}  {up}-{n-up} up  mean {100*v.mean():+.3f}%  "
          f"med {100*np.median(v):+.3f}%  sign_p {sign_test(up, n):.4f}")

print("\n=== era split, SPY h=5 and h=10 ===")
for h in (5, 10):
    f = fwd_ret(s, h).dropna()
    d = pd.DatetimeIndex(hist).intersection(f.index)
    show(era_split(d, f.loc[d].values), f"SPY h={h}")
    print("  ", cluster_note(d, f.loc[d].values, k=2))

print("\n=== TLT era split, h=5 and h=10 ===")
tl = px["TLT"].dropna()
for h in (5, 10):
    f = fwd_ret(tl, h).dropna()
    d = pd.DatetimeIndex(hist).intersection(f.index)
    show(era_split(d, f.loc[d].values), f"TLT h={h}")

print("\n=== does the yield keep going? ^TNX itself ===")
for h in HS:
    f = fwd_ret(tnx, h).dropna()
    d = pd.DatetimeIndex(hist).intersection(f.index)
    v = f.loc[d].values
    up = int((v > 0).sum())
    print(f"  h={h:2d}  n={len(v):2d}  {up}-{len(v)-up} higher  mean {100*v.mean():+.3f}%  "
          f"sign_p {sign_test(up, len(v)):.4f}")

print("\n=== the events, dated, with SPY h=5 ===")
f5 = fwd_ret(s, 5).dropna()
for d in hist:
    if d in f5.index:
        print(f"  {d.date()}  yield {tnx.loc[d]:.3f}  SPY next 5d {100*f5.loc[d]:+.2f}%")
    else:
        print(f"  {d.date()}  yield {tnx.loc[d]:.3f}  SPY next 5d n/a")
