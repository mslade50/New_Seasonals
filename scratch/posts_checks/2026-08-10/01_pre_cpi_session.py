"""Tomorrow (Tue 2026-08-11) is the session before Wednesday's CPI print.
What does the pre-CPI session itself do, and does August / the midterm
cycle sharpen it? Descriptive stat for a post, not an entry."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import (close_panel, load_events, show, sign_test, summarize,
                       era_split)

px = close_panel(["SPY"])
spy = px["SPY"].dropna()
ret = spy.pct_change()

ev = load_events(["cpi"])
cpi = pd.DatetimeIndex(sorted(ev["date"].unique()))
idx = spy.index

rows, aug_rows, mid_rows = [], [], []
for d in cpi:
    pos = idx.searchsorted(d)
    if pos <= 1 or pos >= len(idx):
        continue
    if idx[pos] != d:            # CPI on a non-session day: skip
        continue
    pre = ret.iloc[pos - 1]      # the pre-CPI session's own close-to-close
    if np.isnan(pre):
        continue
    rows.append((idx[pos - 1], pre))
    if d.month == 8:
        aug_rows.append((idx[pos - 1], pre))
    if d.year % 4 == 2:
        mid_rows.append((idx[pos - 1], pre))

def rep(label, rr):
    dates = pd.DatetimeIndex([a for a, _ in rr])
    vals = np.array([b for _, b in rr])
    s = summarize(vals, label)
    wins = int((vals > 0).sum())
    s["sign_p"] = round(sign_test(wins, len(vals)), 4)
    s["record"] = f"{wins}/{len(vals)}"
    return s, dates, vals

all_s, all_d, all_v = rep("pre-CPI session, all", rows)
aug_s, _, _ = rep("pre-CPI, August", aug_rows)
mid_s, _, _ = rep("pre-CPI, midterm yrs", mid_rows)
base = summarize(ret.dropna().to_numpy(), "all sessions control")
show([all_s, aug_s, mid_s, base], "SPY: the session before a CPI print")
print("era:", era_split(all_d, all_v))
