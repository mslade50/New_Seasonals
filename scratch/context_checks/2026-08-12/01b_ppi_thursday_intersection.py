"""01 found that the PPI duration bid lives entirely OUTSIDE Thursday prints, and that
the August-Thursday bond cell lives entirely outside PPI Thursdays. Tomorrow is an
August Thursday PPI that lands 1 td after a CPI, i.e. the intersection of all three.
Pin the intersection cells and check the weekday breakdown is not one odd day.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, era_split, fwd_ret, load_events, sign_test, summarize  # noqa: E402

px = close_panel(["TLT", "IEF", "^TNX", "SPY", "^GSPC"])
dates = px.index
ev = load_events()
ppi = set(ev.loc[ev["event"] == "ppi", "date"])
cpi = set(ev.loc[ev["event"] == "cpi", "date"])

NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}


def rows():
    """(anchor, print session) for every PPI that landed on a session we have."""
    out = []
    for d in sorted(ppi):
        pos = dates.searchsorted(d)
        if pos >= len(dates) or dates[pos] != d or pos == 0:
            continue
        out.append((dates[pos - 1], dates[pos]))
    return out


R = rows()


def show(label, anchors, tkr="TLT", h=1):
    s = px[tkr].dropna()
    f = fwd_ret(s, h)
    a = pd.DatetimeIndex(anchors).intersection(f.dropna().index)
    v = f.loc[a].values
    if len(v) == 0:
        print(f"  {label:<46} {tkr:<6} n=0")
        return None
    d = summarize(v)
    up = int((v > 0).sum())
    print(f"  {label:<46} {tkr:<6} n={len(v):<4} mean={d['mean_pct']:+.3f}%  "
          f"hit={d['hit']:.1f}%  t={d['t']:+.2f}  {up}-{len(v) - up} up  "
          f"sign p={sign_test(up, len(v)):.4f}")
    return a, v


print("=== PPI duration bid by the PRINT session's weekday ===")
for wd in range(5):
    anc = [a for a, p in R if p.weekday() == wd]
    for t in ("TLT", "IEF"):
        show(f"PPI prints on {NAMES[wd]}", anc, t)
    print()

print("=== the intersection that describes tomorrow ===")
b2b_thu = [a for a, p in R if p.weekday() == 3 and a in cpi]
b2b_not_thu = [a for a, p in R if p.weekday() != 3 and a in cpi]
thu_not_b2b = [a for a, p in R if p.weekday() == 3 and a not in cpi]
for t in ("TLT", "IEF", "^TNX", "^GSPC"):
    show("Thu PPI, 1 td after a CPI", b2b_thu, t)
print()
for t in ("TLT", "IEF"):
    show("non-Thu PPI, 1 td after a CPI", b2b_not_thu, t)
    show("Thu PPI, not after a CPI", thu_not_b2b, t)

print("\n  dates in the Thu-PPI-after-CPI cell:")
for a in sorted(b2b_thu):
    pos = dates.searchsorted(a)
    r = (px['TLT'].iloc[pos + 1] / px['TLT'].iloc[pos] - 1) * 100
    print(f"    print {dates[pos + 1].date()}   TLT {r:+.2f}%")

print("\n=== is the non-Thursday PPI effect era-stable? ===")
non_thu = [a for a, p in R if p.weekday() != 3]
for t in ("TLT", "IEF"):
    res = show("non-Thu PPI, all", non_thu, t)
    if res:
        for part in era_split(res[0], res[1]):
            print(f"       {part['label']:<10} n={part['n']:<4} "
                  f"mean={part['mean_pct']:+.3f}%  hit={part['hit']:.1f}%  t={part['t']:+.2f}")

print("\n=== August Thursdays, the other side of the same coin ===")
aug_thu_ppi = [a for a, p in R if p.weekday() == 3 and p.month == 8]
aug_thu_all, aug_thu_clean = [], []
for i in range(len(dates) - 1):
    nxt = dates[i + 1]
    if nxt.weekday() == 3 and nxt.month == 8:
        aug_thu_all.append(dates[i])
        if nxt not in ppi:
            aug_thu_clean.append(dates[i])
for t in ("TLT", "IEF"):
    show("every August Thursday", aug_thu_all, t)
    show("August Thursday, no PPI", aug_thu_clean, t)
    show("August Thursday WITH a PPI", aug_thu_ppi, t)
    print()

print("=== equities on the day after a CPI when PPI follows ===")
b2b = [a for a, p in R if a in cpi]
for t in ("SPY", "^GSPC"):
    res = show("PPI 1 td after a CPI", b2b, t)
    if res:
        v = res[1]
        print(f"       median {100 * np.median(v):+.3f}%  worst {100 * v.min():+.2f}%  "
              f"best {100 * v.max():+.2f}%")
