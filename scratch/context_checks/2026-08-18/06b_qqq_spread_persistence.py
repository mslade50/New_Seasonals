"""Follow-on to drill 06: the index forward returns are noise, the SPREAD is not.

QQQ/SPY over the next 21 sessions after the trigger: -1.100%, 50-75, t -2.01,
against +0.094% for all 21-session windows. The relative leg is the claim, so
it gets the era split, the concentration note, the local control and the
horizon shape that the index legs did not earn.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, summarize, show, sign_test, era_split, cluster_note,
    declusters, local_control,
)

px = close_panel(["SPY", "QQQ"])
idx = px.index
r = px.pct_change(fill_method=None)
spread = r["QQQ"] - r["SPY"]
ratio = px["QQQ"] / px["SPY"]

mask = (spread <= -0.01) & (r["SPY"] < 0) & (r["SPY"] > -0.01)
trig = idx[mask.fillna(False).values]
trig = trig[trig < idx[-1]]
epi = declusters(pd.DatetimeIndex(trig), 5, idx)

H = 21
f = ratio.shift(-H) / ratio - 1.0
s = f.loc[f.index.intersection(epi)].dropna()
print(f"episodes with a full 21td forward window: {len(s)}")

print("\n" + "=" * 74)
print("A. era split and concentration on QQQ/SPY, h=21")
print("=" * 74)
show(era_split(s.index, s.values), "QQQ/SPY h=21 after the trigger")
print(" ", cluster_note(s.index, s.values, k=2))
order = np.argsort(-np.abs(s.values))[:2]
keep = np.ones(len(s), bool)
keep[order] = False
row = summarize(s.values[keep], "ex the 2 largest episodes")
row["rec"] = f"{int((s.values[keep] > 0).sum())}-{int((s.values[keep] <= 0).sum())}"
show([row], "drop-the-biggest robustness")

by_yr = pd.Series(s.values).groupby(s.index.year.values).mean() * 100
print(f"  QQQ lagged SPY on average in {int((by_yr < 0).sum())} of "
      f"{len(by_yr)} calendar years with a trigger")

print("\n" + "=" * 74)
print("B. controls")
print("=" * 74)
valid = f.dropna().index
loc = local_control(valid, pd.DatetimeIndex(epi).intersection(valid), win=126)
out = [
    summarize(s.values, "trigger episodes"),
    summarize(f.loc[valid].values, "CTRL all 21td windows"),
    summarize(f.loc[loc].values, "CTRL local +/-126td ex-trigger"),
]
for row in out:
    pass
show(out, "QQQ/SPY over 21 sessions")

print("\n" + "=" * 74)
print("C. horizon shape of the relative leg")
print("=" * 74)
out = []
for h in (1, 3, 5, 10, 21, 42, 63):
    ff = ratio.shift(-h) / ratio - 1.0
    v = ff.loc[ff.index.intersection(epi)].dropna().values
    row = summarize(v, f"h={h}")
    row["rec"] = f"{int((v > 0).sum())}-{int((v <= 0).sum())}"
    row["sign_p_down"] = round(sign_test(int((v <= 0).sum()), len(v)), 4)
    base = ff.dropna()
    row["ctl_pct"] = round(100 * base.mean(), 3)
    row["edge_pp"] = round(row["mean_pct"] - 100 * base.mean(), 3)
    out.append(row)
show(out, "QQQ/SPY, where the relative drag sits")

print("\n" + "=" * 74)
print("D. is it just 'QQQ is more volatile'? repeat with the sign flipped")
print("=" * 74)
mask_up = (spread >= 0.01) & (r["SPY"] > 0) & (r["SPY"] < 0.01)
trig_up = idx[mask_up.fillna(False).values]
epi_up = declusters(pd.DatetimeIndex(trig_up[trig_up < idx[-1]]), 5, idx)
out = []
for h in (5, 21):
    ff = ratio.shift(-h) / ratio - 1.0
    v = ff.loc[ff.index.intersection(epi_up)].dropna().values
    row = summarize(v, f"QQQ OUTperformed by 1pp, h={h}")
    row["rec"] = f"{int((v > 0).sum())}-{int((v <= 0).sum())}"
    base = ff.dropna()
    row["edge_pp"] = round(row["mean_pct"] - 100 * base.mean(), 3)
    out.append(row)
show(out, f"the mirror cell (n episodes {len(epi_up)})")
print("  if both sides drag, the cell is a volatility artifact, not rotation")

print("\n" + "=" * 74)
print("E. today's context: how often has this fired in 2026?")
print("=" * 74)
cnt = pd.Series(1, index=pd.DatetimeIndex(trig)).groupby(
    pd.DatetimeIndex(trig).year).sum()
print("  raw trigger days per year (last 14):", dict(cnt.tail(14)))
print(f"  2026 to date: {int(cnt.get(2026, 0))}, "
      f"median year since 2000: {cnt.median():.0f}")
print(f"  today's QQQ-SPY spread: {100*spread.iloc[-1]:+.2f}pp, "
      f"{100*(spread.tail(252) < spread.iloc[-1]).mean():.1f}th pctile of the year")
