"""Event lane. Tomorrow is trading day 17 of 21 in August, so it opens the
month's LAST FIVE sessions. The engine's E:month_end trigger only anchors the
last three, so this window is invisible to the sweep.

Cell: the final five trading sessions of August, one non-overlapping read per
year, measured from the close of td (n-5). Controls: the last five sessions of
every other month, and the same August window split by cycle year.
Also the Tuesday leg, since the next session is a Tuesday.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import load_prices, summarize, sign_test  # noqa

ASOF = pd.Timestamp("2026-08-24")
px = load_prices(["SPY", "QQQ", "^GSPC", "IWM", "^VIX", "TLT", "GC=F"])


def last_n_window(ser, month, n_last=5):
    """Return per-year (anchor_date, forward return over the month's last n_last sessions)."""
    s = ser.dropna()
    out = []
    for (y, m), grp in s.groupby([s.index.year, s.index.month]):
        if m != month or len(grp) < n_last + 2:
            continue
        anchor_i = len(grp) - n_last - 1          # close BEFORE the window opens
        anchor_d = grp.index[anchor_i]
        ret = grp.iloc[-1] / grp.iloc[anchor_i] - 1
        out.append((y, anchor_d, ret, len(grp)))
    return out


def report(rows, label):
    rows = [r for r in rows if r[1] < ASOF]
    v = np.array([r[2] for r in rows])
    st = summarize(v, "")
    up = int((v > 0).sum())
    print(f"{label:<52} n={st['n']:<3} mean {st['mean_pct']:>7.2f}%  med {st['median_pct']:>7.2f}%  "
          f"{up}-{len(v)-up} up  sign p {sign_test(up, len(v)):.4f}  t {st['t']:>5.2f}  "
          f"worst {st['worst_pct']:>6.1f}%  best {st['best_pct']:>5.1f}%")
    return rows, v


print("=== last five sessions of AUGUST, one read per year ===")
for t in ["SPY", "^GSPC", "QQQ", "IWM", "^VIX", "TLT", "GC=F"]:
    rows, v = report(last_n_window(px[t]["Close"], 8), f"{t} August last-5")

print()
print("=== same window, every OTHER month (control) ===")
for t in ["SPY", "^GSPC", "QQQ"]:
    allr = []
    for m in range(1, 13):
        if m == 8:
            continue
        allr += last_n_window(px[t]["Close"], m)
    report(allr, f"{t} non-August last-5 control")

print()
print("=== August last-5 by cycle year (midterm = year %% 4 == 2) ===")
for t in ["SPY", "^GSPC", "QQQ"]:
    rows = [r for r in last_n_window(px[t]["Close"], 8) if r[1] < ASOF]
    for lab, sel in (("midterm", [r for r in rows if r[0] % 4 == 2]),
                     ("non-midterm", [r for r in rows if r[0] % 4 != 2])):
        v = np.array([r[2] for r in sel])
        if len(v) < 3:
            continue
        st = summarize(v, "")
        up = int((v > 0).sum())
        print(f"  {t:<7} {lab:<12} n={st['n']:<3} mean {st['mean_pct']:>7.2f}%  "
              f"{up}-{len(v)-up} up  sign p {sign_test(up, len(v)):.4f}  "
              f"years {[r[0] for r in sel]}")

print()
print("=== the ^GSPC August last-5 year by year ===")
for y, d, r, ln in last_n_window(px["^GSPC"]["Close"], 8):
    if d >= ASOF:
        print(f"  {y}  anchor {d.date()}  (LIVE, window opens next session)")
    else:
        print(f"  {y}  anchor {d.date()}  {100*r:>6.2f}%")
