"""Today SMH fell -4.09% while SPY fell only -0.68%: a semis purge on a
shallow index day. Candidate idea: buy the purge. Cell: SMH 1d return
<= -3.5% while SPY's same-session return >= -1.5%. Entry is next-session
MOO (that is the only thing an evening draft can do), holds 1/2/3/5 td.
Declustered to the first hit in 10 sessions. Controls: all SMH sessions
(same MOO legs) and SMH <= -3.5% days where SPY ALSO fell hard (the
broad-selloff sibling, to see whether the shallow-tape condition matters).
Era note: 2015+ sub-cell reported separately.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_prices, sign_test, summarize  # noqa

px = load_prices(["SMH", "SPY"])
smh, spy = px["SMH"], px["SPY"]
close = smh["Close"].dropna()
opn = smh["Open"].reindex(close.index)
r1 = close.pct_change()
spy_r1 = spy["Close"].pct_change().reindex(close.index)

purge = (r1 <= -0.035)
shallow = purge & (spy_r1 >= -0.015)
broad = purge & (spy_r1 < -0.015)

def decluster(mask: pd.Series, window: int = 10) -> pd.Series:
    out = mask.copy()
    last = -10**9
    vals = mask.values.copy()
    for i, m in enumerate(vals):
        if m:
            if i - last < window:
                vals[i] = False
            else:
                last = i
    out[:] = vals
    return out

shallow_d = decluster(shallow)
broad_d = decluster(broad)

def fwd_moo(hold: int) -> pd.Series:
    # next-session open -> close hold sessions after that open
    entry = opn.shift(-1)
    exit_ = close.shift(-hold)
    return (exit_ / entry - 1)

print("SMH <= -3.5% day: next-session MOO entry, close-of-hold exit")
print("=" * 84)
for hold in (1, 2, 3, 5):
    ser = fwd_moo(hold)
    for label, mask in (("shallow tape (SPY>=-1.5%)", shallow_d),
                        ("broad selloff (SPY<-1.5%)", broad_d),
                        ("all sessions             ", pd.Series(True, index=close.index))):
        v = ser[mask].dropna()
        s = summarize(v.values, label)
        up = int((v > 0).sum())
        print(f"  h{hold}  {label}  n {s['n']:4d}  mean {s['mean_pct']:+.3f}%"
              f"  med {s['median_pct']:+.3f}%  hit {s['hit']:.1f}%"
              f"  {up}-{s['n']-up}  signp {sign_test(up, s['n']):.4f}")
    print("-" * 84)

for label, mask in (("shallow, 2015+", shallow_d & (close.index >= "2015-01-01")),
                    ("shallow, 2022+", shallow_d & (close.index >= "2022-01-01"))):
    v = fwd_moo(3)[mask].dropna()
    s = summarize(v.values, label)
    up = int((v > 0).sum())
    print(f"  h3  {label}  n {s['n']:4d}  mean {s['mean_pct']:+.3f}%"
          f"  med {s['median_pct']:+.3f}%  hit {s['hit']:.1f}%  {up}-{s['n']-up}"
          f"  signp {sign_test(up, s['n']):.4f}")
v = fwd_moo(3)[shallow_d & (close.index >= "2022-01-01")].dropna()
print("  shallow 2022+ h3 per-anchor:", ", ".join(
    f"{d.date()}:{100*x:+.2f}" for d, x in zip(v.index, v.values)))
