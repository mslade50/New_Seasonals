"""Tonight's context brief scored the 3-session run into monthly opex for TLT
at n=287, mean +0.262%, hit 59.6%, t=2.92 (tag: solid). Anchor = 3 td before
opex, k3 close -> opex close. Before that becomes an idea, three things have
to hold:

1. The tradeable leg. Tonight IS the anchor (opex Fri 8/21), so an idea can
   only enter MOO tomorrow. Measure open(t+1) -> opex close, i.e. what MOO
   forfeits vs the brief's close->close cell.
2. The regime split. Yesterday's pitch stand-down killed a late-August TLT
   seasonal as "a bond-bull fossil" (rising-yield regime pays a fraction of
   the falling-yield cell). Split this cell by TLT above/below its 200d SMA
   at the anchor. TLT closed tonight 4.5% BELOW its 200d and 0.4% off its
   52-week low, so the below-200d cell is the one that has to carry the idea.
3. Recency. Post-2022 (the rate-shock era) sub-cell on its own.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_events, load_prices, sign_test, summarize  # noqa

df = load_prices(["TLT"])["TLT"]
close = df["Close"].dropna()
opn = df["Open"].reindex(close.index)
idx = close.index

ev = load_events(["opex"])
opx = pd.DatetimeIndex(sorted(set(ev.loc[ev["event"] == "opex", "date"])))

rows = []
for e in opx:
    loc = idx.searchsorted(e)
    if loc >= len(idx) or loc - 3 < 0:
        continue
    # opex lands on a holiday-shifted date sometimes; use the session at/before
    if idx[loc] != e:
        loc -= 1
        if loc - 3 < 0:
            continue
    a = loc - 3
    rows.append({
        "anchor": idx[a],
        "cc": close.iloc[loc] / close.iloc[a] - 1,
        "moo": close.iloc[loc] / opn.iloc[a + 1] - 1,
        "gap": opn.iloc[a + 1] / close.iloc[a] - 1,
    })
cells = pd.DataFrame(rows).set_index("anchor")

sma200 = close.rolling(200).mean()
lo252 = close.rolling(252).min()
cells["below_200d"] = (close.reindex(cells.index) < sma200.reindex(cells.index))
cells["near_52wlo"] = (close.reindex(cells.index) / lo252.reindex(cells.index) - 1) < 0.02
cells = cells.dropna(subset=["cc", "moo"])

def report(label: str, sub: pd.DataFrame) -> None:
    for name in ("cc", "moo", "gap"):
        v = sub[name].dropna()
        if not len(v):
            print(f"  {label:28s} {name:3s}  n 0")
            continue
        s = summarize(v.values, label)
        up = int((v > 0).sum())
        print(f"  {label:28s} {name:3s}  n {s['n']:3d}  mean {s['mean_pct']:+.3f}%"
              f"  med {s['median_pct']:+.3f}%  hit {s['hit']:.1f}%"
              f"  {up}-{s['n']-up}  signp {sign_test(up, s['n']):.4f}")

print("TLT 3-session run into monthly opex (anchor = opex-3td)")
print("cc = anchor close->opex close (the brief's cell); moo = next open->opex close")
print("=" * 78)
report("all", cells)
print("-" * 78)
report("above 200d at anchor", cells[~cells["below_200d"].astype(bool)])
report("below 200d at anchor", cells[cells["below_200d"].astype(bool)])
print("-" * 78)
report("within 2% of 52w low", cells[cells["near_52wlo"].astype(bool)])
report("2022+ era", cells[cells.index >= "2022-01-01"])
report("below 200d, 2022+", cells[cells["below_200d"].astype(bool) & (cells.index >= "2022-01-01")])

v = cells.loc[cells["below_200d"].astype(bool), "moo"].dropna().tail(12)
print("  below-200d moo, last 12:", ", ".join(
    f"{d.date()}:{100*x:+.2f}" for d, x in zip(v.index, v.values)))
