"""Thursday 2026-08-13 is a PPI print landing the trading day AFTER CPI
(Wednesday 2026-08-12). PPI-day close-to-close for SPY: all PPI prints,
the subset where the prior trading day was a CPI print (the doubleheader
ordering), and the day-after-CPI baseline for contrast. Descriptive, not
an entry."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import close_panel, load_events, show, sign_test, summarize

px = close_panel(["SPY"])
ev = load_events(["cpi", "ppi"])
cpi = set(pd.DatetimeIndex(ev.loc[ev["event"] == "cpi", "date"].unique()))
ppi = pd.DatetimeIndex(sorted(ev.loc[ev["event"] == "ppi", "date"].unique()))

s = px["SPY"].dropna()
ret = s.pct_change()
idx = s.index

cells = {"ppi all": [], "ppi day-after-cpi": [], "day-after-cpi (any)": []}
for d in ppi:
    pos = idx.searchsorted(d)
    if pos >= len(idx) or idx[pos] != d or pos == 0:
        continue
    r = ret.iloc[pos]
    if np.isnan(r):
        continue
    cells["ppi all"].append(r)
    if idx[pos - 1] in cpi:
        cells["ppi day-after-cpi"].append(r)
for d in sorted(cpi):
    pos = idx.searchsorted(d)
    if pos + 1 >= len(idx) or idx[pos] != d:
        continue
    r = ret.iloc[pos + 1]
    if not np.isnan(r):
        cells["day-after-cpi (any)"].append(r)

rows = []
for label, vals in cells.items():
    v = np.array(vals)
    if not len(v):
        continue
    r = summarize(v, f"SPY {label}")
    wins = int((v > 0).sum())
    r["sign_p"] = round(sign_test(wins, len(v)), 4)
    r["record"] = f"{wins}/{len(v)}"
    rows.append(r)
rows.append(summarize(ret.dropna().to_numpy(), "SPY all days"))
show(rows, "SPY: PPI-day cells (doubleheader ordering)")
