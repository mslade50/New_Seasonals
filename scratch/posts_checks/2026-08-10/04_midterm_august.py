"""August in a midterm year (2026 qualifies). Full-month August returns
for SPY and ^GSPC: all Augusts vs midterm Augusts vs the all-months
baseline. Small N by construction (one midterm August per four years);
sign-test doctrine applies. Descriptive, not an entry."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
from pitch_lab import close_panel, show, sign_test, summarize

px = close_panel(["SPY", "^GSPC"])

for tkr in ["SPY", "^GSPC"]:
    s = px[tkr].dropna()
    monthly = s.resample("ME").last().pct_change().dropna()
    aug = monthly[monthly.index.month == 8]
    mid = aug[aug.index.year % 4 == 2]
    rows = []
    for label, v in [("all months", monthly.to_numpy()),
                     ("all augusts", aug.to_numpy()),
                     ("midterm augusts", mid.to_numpy())]:
        if not len(v):
            continue
        r = summarize(v, f"{tkr} {label}")
        wins = int((v > 0).sum())
        r["sign_p"] = round(sign_test(wins, len(v)), 4)
        r["record"] = f"{wins}/{len(v)}"
        rows.append(r)
    yrs = ", ".join(f"{y}:{r:+.1%}" for y, r in zip(mid.index.year, mid))
    show(rows, f"{tkr}: August cells (midterm years: {yrs})")
