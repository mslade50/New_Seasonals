"""Rerun of drill 09 with the incomplete-month bug fixed: 2026's August is still
running, so it is an ANCHOR, never a completed observation. Focus on gold, whose
August month-end cell was the only one to clear its control, and cross it against
drill 07's magnitude finding.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import load_prices, summarize, sign_test, era_split, cluster_note  # noqa

ASOF = pd.Timestamp("2026-08-24")
px = load_prices(["SPY", "QQQ", "^GSPC", "IWM", "^VIX", "TLT", "GC=F", "SI=F"])


def last_n_window(ser, month, n_last=5):
    s = ser.dropna()
    last_bar = s.index[-1]
    out = []
    for (y, m), grp in s.groupby([s.index.year, s.index.month]):
        if m != month or len(grp) < n_last + 2:
            continue
        # month must be COMPLETE: its final bar must not be the series' final bar
        if grp.index[-1] == last_bar:
            continue
        i = len(grp) - n_last - 1
        out.append((y, grp.index[i], grp.iloc[-1] / grp.iloc[i] - 1))
    return out


def report(rows, label):
    v = np.array([r[2] for r in rows])
    st = summarize(v, "")
    up = int((v > 0).sum())
    print(f"{label:<44} n={st['n']:<3} mean {st['mean_pct']:>7.2f}%  med {st['median_pct']:>7.2f}%  "
          f"{up}-{len(v)-up} up  sign p {sign_test(up, len(v)):.4f}  t {st['t']:>5.2f}  "
          f"worst {st['worst_pct']:>6.1f}%  best {st['best_pct']:>5.1f}%")
    return rows, v


print("=== last five sessions of AUGUST, completed years only ===")
res = {}
for t in ["SPY", "^GSPC", "QQQ", "IWM", "^VIX", "TLT", "GC=F", "SI=F"]:
    res[t] = report(last_n_window(px[t]["Close"], 8), f"{t} August last-5")

print()
print("=== control: same window in every other month ===")
for t in ["SPY", "^GSPC", "QQQ", "GC=F", "TLT"]:
    allr = []
    for m in range(1, 13):
        if m != 8:
            allr += last_n_window(px[t]["Close"], m)
    report(allr, f"{t} non-August last-5")

print()
print("=== gold: August last-5 detail ===")
rows, v = res["GC=F"]
print("  year by year:", [(r[0], round(100 * r[2], 2)) for r in rows])
print("  era:", [(e['label'], e['n'], round(e.get('mean_pct', float('nan')), 2))
                 for e in era_split(pd.DatetimeIndex([r[1] for r in rows]), v)])
print(" ", cluster_note(pd.DatetimeIndex([r[1] for r in rows]), v))
mid = [r for r in rows if r[0] % 4 == 2]
vm = np.array([r[2] for r in mid])
print(f"  midterm years only: n={len(vm)} mean {100*vm.mean():.2f}% "
      f"{int((vm>0).sum())}-{int((vm<=0).sum())} up sign p "
      f"{sign_test(int((vm>0).sum()), len(vm)):.4f}  years {[r[0] for r in mid]}")

print()
print("=== the cross: gold's August last-5 in years it entered already stretched ===")
g = px["GC=F"]["Close"]
r21 = g.pct_change(21)
hot, cool = [], []
for y, anchor, ret in rows:
    z = r21.reindex([anchor]).iloc[0]
    (hot if z >= 0.05 else cool).append((y, round(100 * z, 1), ret))
for lab, sel in (("entered 21d >= +5%", hot), ("entered 21d < +5%", cool)):
    vv = np.array([r[2] for r in sel])
    st = summarize(vv, "")
    up = int((vv > 0).sum())
    print(f"  {lab:<22} n={st['n']:<3} mean {st['mean_pct']:>6.2f}%  {up}-{len(vv)-up} up  "
          f"sign p {sign_test(up, len(vv)):.4f}   {[(r[0], r[1]) for r in sel]}")
print(f"  2026 enters at 21d = {100*r21.iloc[-1]:.2f}%")
