"""Event lane, conditioned on the live tape. Tomorrow opens August's final five
sessions and the S&P enters 1.87% below its 52-week high. Split the August
last-5 cell on how close the index entered to its high, and give IWM the control
its raw 17-9 record was missing.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import load_prices, summarize, sign_test, era_split, cluster_note  # noqa

ASOF = pd.Timestamp("2026-08-24")
px = load_prices(["SPY", "^GSPC", "QQQ", "IWM"])


def windows(ser, month, n_last=5):
    s = ser.dropna()
    last_bar = s.index[-1]
    out = []
    for (y, m), grp in s.groupby([s.index.year, s.index.month]):
        if m != month or len(grp) < n_last + 2:
            continue
        if grp.index[-1] == last_bar:
            continue
        i = len(grp) - n_last - 1
        out.append((y, grp.index[i], grp.iloc[-1] / grp.iloc[i] - 1))
    return out


def line(rows, label):
    v = np.array([r[2] for r in rows])
    if len(v) < 3:
        print(f"{label:<46} n={len(v)} too few")
        return
    st = summarize(v, "")
    up = int((v > 0).sum())
    print(f"{label:<46} n={st['n']:<3} mean {st['mean_pct']:>6.2f}%  med {st['median_pct']:>6.2f}%  "
          f"{up}-{len(v)-up} up  sign p {sign_test(up, len(v)):.4f}  t {st['t']:>5.2f}  "
          f"worst {st['worst_pct']:>6.1f}%")


print("=== IWM: August last-5 against its own non-August control ===")
line(windows(px["IWM"]["Close"], 8), "IWM August last-5")
allr = []
for m in range(1, 13):
    if m != 8:
        allr += windows(px["IWM"]["Close"], m)
line(allr, "IWM non-August last-5 control")

print()
print("=== August last-5 split by how close the index ENTERED to its 52w high ===")
for t in ["^GSPC", "SPY", "QQQ", "IWM"]:
    c = px[t]["Close"].dropna()
    hi = c.rolling(252).max()
    rows = windows(c, 8)
    near, far = [], []
    for y, a, r in rows:
        d = c.loc[a] / hi.loc[a] - 1
        (near if d >= -0.03 else far).append((y, round(100 * d, 2), r))
    print(f"-- {t}  (2026 enters at {100*(c.iloc[-1]/hi.iloc[-1]-1):.2f}% from its 252d high)")
    line(near, "   entered within 3% of the 52w high")
    line(far, "   entered more than 3% below")
    print(f"      near years: {[(r[0], r[1]) for r in near]}")

print()
print("=== the near-high August last-5 cell, S&P detail ===")
c = px["^GSPC"]["Close"].dropna()
hi = c.rolling(252).max()
rows = windows(c, 8)
near = [(y, a, r) for y, a, r in rows if c.loc[a] / hi.loc[a] - 1 >= -0.03]
v = np.array([r[2] for r in near])
print("  year by year:", [(r[0], round(100 * r[2], 2)) for r in near])
print("  era:", [(e['label'], e['n'], round(e.get('mean_pct', float('nan')), 2))
                 for e in era_split(pd.DatetimeIndex([r[1] for r in near]), v)])
print(" ", cluster_note(pd.DatetimeIndex([r[1] for r in near]), v))
mid = [r for r in near if r[0] % 4 == 2]
print(f"  midterm subset: {[(r[0], round(100*r[2], 2)) for r in mid]}")
