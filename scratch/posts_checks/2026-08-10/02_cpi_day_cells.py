"""Wednesday 2026-08-13 is an August CPI print in a midterm year. CPI-day
close-to-close for SPY and TLT: all prints, August, midterm, and the
joint cell. Reply-ammo material; descriptive, not an entry."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import close_panel, load_events, show, sign_test, summarize

px = close_panel(["SPY", "TLT"])
ev = load_events(["cpi"])
cpi = pd.DatetimeIndex(sorted(ev["date"].unique()))

for tkr in ["SPY", "TLT"]:
    s = px[tkr].dropna()
    ret = s.pct_change()
    idx = s.index
    cells = {"all": [], "august": [], "midterm": [], "aug+midterm": []}
    for d in cpi:
        pos = idx.searchsorted(d)
        if pos >= len(idx) or idx[pos] != d:
            continue
        r = ret.iloc[pos]
        if np.isnan(r):
            continue
        cells["all"].append(r)
        if d.month == 8:
            cells["august"].append(r)
        if d.year % 4 == 2:
            cells["midterm"].append(r)
        if d.month == 8 and d.year % 4 == 2:
            cells["aug+midterm"].append(r)
    rows = []
    for label, vals in cells.items():
        v = np.array(vals)
        if not len(v):
            continue
        r = summarize(v, f"{tkr} CPI day, {label}")
        wins = int((v > 0).sum())
        r["sign_p"] = round(sign_test(wins, len(v)), 4)
        r["record"] = f"{wins}/{len(v)}"
        rows.append(r)
    rows.append(summarize(ret.dropna().to_numpy(), f"{tkr} all days"))
    show(rows, f"{tkr}: CPI-day cells")
