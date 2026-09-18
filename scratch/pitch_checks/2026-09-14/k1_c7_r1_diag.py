import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import pandas as pd

px = close_panel(["^TNX"])
tnx = px["^TNX"].dropna()
r5 = pct_rank(tnx, 5)
hi = tnx.rolling(252).max()
at_max = tnx >= hi - 1e-9
print("n", len(tnx), "r5>=95 days", int((r5 >= 95).sum()), "at_max days", int(at_max.sum()),
      "joint", int(((r5 >= 95) & at_max).sum()))
print(tnx.resample("YE").agg(["min", "max", "count"]).to_string())
print("at_max by year:", at_max.groupby(tnx.index.year).sum().to_dict())
print("joint by year:", ((r5 >= 95) & at_max).groupby(tnx.index.year).sum().to_dict())
# near-max forms
for tol in (0.0, 0.01, 0.02):
    nm = tnx >= hi * (1 - tol)
    print(f"within {tol:.0%} of 252 max & r5>=95:", int(((r5 >= 95) & nm).sum()))
