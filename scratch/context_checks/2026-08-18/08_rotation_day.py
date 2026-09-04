"""A rotation day: wide sector dispersion under a shallow index decline.

Today SPY fell 0.68% while XLK fell 2.47% and XLV rose 1.60%, a 4pp spread
between the best and worst sector on a session the index barely moved. No
engine trigger covers dispersion, so this is a tape-derived cross, recorded as
such in the cell map.

Universe rule: sector ETFs are BREADTH CONTEXT and are never the subject. The
subject here is SPY. The sectors only define the state.
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

SECTORS = ["XLK", "XLV", "XLE", "XLP", "XLI", "XLF", "XLY", "XLB", "XLU"]
px = close_panel(["SPY", "^VIX"] + SECTORS)
have = [s for s in SECTORS if px[s].notna().sum() > 1000]
print("sectors with usable history:", have)
px = px.dropna(subset=have + ["SPY"], how="any")
idx = px.index
print(f"panel: {len(idx)} sessions, {idx[0].date()} to {idx[-1].date()}")

r = px.pct_change(fill_method=None)
sec = r[have]
disp = sec.max(axis=1) - sec.min(axis=1)          # best minus worst sector
rank = disp.rolling(252).rank(pct=True) * 100

print("\ntoday: SPY %+.2f%%, sector spread %.2fpp, %.1fth pctile of the year"
      % (100 * r["SPY"].iloc[-1], 100 * disp.iloc[-1], rank.iloc[-1]))
print("  best  %s %+.2f%%" % (sec.iloc[-1].idxmax(), 100 * sec.iloc[-1].max()))
print("  worst %s %+.2f%%" % (sec.iloc[-1].idxmin(), 100 * sec.iloc[-1].min()))

# the cell: shallow index decline, top-decile sector dispersion
mask = (r["SPY"] < 0) & (r["SPY"] > -0.01) & (rank >= 90)
trig = idx[mask.fillna(False).values]
trig = trig[trig < idx[-1]]
epi = declusters(pd.DatetimeIndex(trig), 5, idx)
print(f"\nraw trigger days {len(trig)}, declustered episodes {len(epi)}")
print("  last ten:", [str(d.date()) for d in epi][-10:])

print("\n" + "=" * 74)
print("A. what SPY does next")
print("=" * 74)
out = []
for h in (1, 3, 5, 10, 21):
    f = px["SPY"].shift(-h) / px["SPY"] - 1.0
    v = f.loc[f.index.intersection(epi)].dropna().values
    row = summarize(v, f"SPY h={h}")
    row["rec"] = f"{int((v > 0).sum())}-{int((v <= 0).sum())}"
    row["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    base = f.dropna()
    row["ctl_all_pct"] = round(100 * base.mean(), 3)
    row["edge_pp"] = round(row["mean_pct"] - 100 * base.mean(), 3)
    out.append(row)
show(out, "SPY after a wide-dispersion shallow decline")

print("\n" + "=" * 74)
print("B. the comparison that matters: shallow decline WITHOUT the dispersion")
print("=" * 74)
mask_lo = (r["SPY"] < 0) & (r["SPY"] > -0.01) & (rank <= 50)
epi_lo = declusters(pd.DatetimeIndex(idx[mask_lo.fillna(False).values]), 5, idx)
out = []
for h in (1, 5, 21):
    f = px["SPY"].shift(-h) / px["SPY"] - 1.0
    for name, e in [("dispersion TOP decile", epi), ("dispersion bottom half", epi_lo)]:
        v = f.loc[f.index.intersection(e)].dropna().values
        row = summarize(v, f"h={h} {name}")
        row["rec"] = f"{int((v > 0).sum())}-{int((v <= 0).sum())}"
        out.append(row)
show(out, "same shallow decline, split by how much the sectors disagreed")

print("\n" + "=" * 74)
print("C. era, concentration, local control on the horizon that pays")
print("=" * 74)
for h in (1, 5):
    f = px["SPY"].shift(-h) / px["SPY"] - 1.0
    s = f.loc[f.index.intersection(epi)].dropna()
    show(era_split(s.index, s.values), f"SPY h={h}: era split")
    print(" ", cluster_note(s.index, s.values, k=2))
    valid = f.dropna().index
    loc = local_control(valid, pd.DatetimeIndex(epi).intersection(valid), win=126)
    print(f"  CTRL local +/-126td ex-trigger: {100*f.loc[loc].mean():+.3f}% (n={len(loc)})")
    order = np.argsort(-np.abs(s.values))[:2]
    keep = np.ones(len(s), bool)
    keep[order] = False
    print(f"  ex the 2 largest episodes: {100*s.values[keep].mean():+.3f}% "
          f"(n={keep.sum()})")

print("\n" + "=" * 74)
print("D. does VIX rising at the same time change it? (today VIX +4.3%)")
print("=" * 74)
vix_up = r["^VIX"] > 0.03
f = px["SPY"].shift(-5) / px["SPY"] - 1.0
for name, m in [("VIX up >3% too (today's shape)", vix_up),
                ("VIX quiet", ~vix_up)]:
    e = pd.DatetimeIndex(epi).intersection(idx[m.fillna(False).values])
    v = f.loc[f.index.intersection(e)].dropna().values
    if len(v) == 0:
        continue
    row = summarize(v, f"SPY h=5 | {name}")
    row["rec"] = f"{int((v > 0).sum())}-{int((v <= 0).sum())}"
    row["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    show([row], name)

print("\n" + "=" * 74)
print("E. how rare is today's state, and when did it last cluster?")
print("=" * 74)
cnt = pd.Series(1, index=pd.DatetimeIndex(trig)).groupby(
    pd.DatetimeIndex(trig).year).sum()
print("  trigger days per year:", dict(cnt.tail(15)))
print(f"  2026 to date: {int(cnt.get(2026, 0))}")
print(f"  today's spread {100*disp.iloc[-1]:.2f}pp vs median "
      f"{100*disp.median():.2f}pp, 252d-pctile {rank.iloc[-1]:.1f}")
