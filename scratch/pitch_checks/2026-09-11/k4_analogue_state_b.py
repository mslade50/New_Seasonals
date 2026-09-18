"""A11 round 2, closing number: DO THE NEIGHBOURS ACTUALLY CARRY TODAY'S STATE?

Gate attribution in k4_analogue_dbc_b showed the parent "DBC at a 252-day high"
and the DISCARDED COMPLEMENT of that parent have IDENTICAL statistics (n=50,
-0.216%), which can only happen if none of the nine analogue episodes is in the
parent at all. Print the raw feature values of the nine against today's.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

TK = ["SPY", "TLT", "GLD", "DBC", "^VIX", "^TNX"]
px_d = load_prices(TK)
idx = px_d["SPY"]["Close"].index
panel = pd.DataFrame({t: px_d[t]["Close"].reindex(idx) for t in TK})
lr = lambda s: rolling_on_valid(s, lambda x: x.rolling(252).rank(pct=True) * 100.0)

f = pd.DataFrame(index=idx)
f["spy21_pct"] = 100 * (panel["SPY"] / panel["SPY"].shift(21) - 1.0)
f["vix"] = panel["^VIX"]
f["tnx_rank"] = lr(panel["^TNX"])
f["dbc_rank"] = lr(panel["DBC"])
f["tlt_rank"] = lr(panel["TLT"])
f = f.dropna()

EPI = ["2018-02-14", "2018-04-20", "2018-10-09", "2021-03-25", "2021-05-14",
       "2022-04-20", "2023-09-21", "2025-01-13", "2026-07-30"]
print("TODAY 2026-09-10:")
print(f.iloc[-1].round(2).to_string())
print("\nthe nine analogue episodes:")
print(f.loc[pd.DatetimeIndex(EPI)].round(2).to_string())
sub = f.loc[pd.DatetimeIndex(EPI)]
for c in ["tnx_rank", "dbc_rank", "tlt_rank"]:
    print("\n%s: today %.1f | neighbours min %.1f med %.1f max %.1f"
          % (c, f.iloc[-1][c], sub[c].min(), sub[c].median(), sub[c].max()))
print("\nneighbours with dbc_rank >= 99.5 : %d of 9" % int((sub["dbc_rank"] >= 99.5).sum()))
print("neighbours with tnx_rank >= 99.5 : %d of 9" % int((sub["tnx_rank"] >= 99.5).sum()))
print("neighbours with tlt_rank <= 0.5  : %d of 9" % int((sub["tlt_rank"] <= 0.5).sum()))
print("neighbours matching ALL THREE extremes: %d of 9"
      % int(((sub["dbc_rank"] >= 99.5) & (sub["tnx_rank"] >= 99.5)
             & (sub["tlt_rank"] <= 0.5)).sum()))
print("\nhow many sessions in the whole history match all three extremes? %d"
      % int(((f["dbc_rank"] >= 99.5) & (f["tnx_rank"] >= 99.5)
             & (f["tlt_rank"] <= 0.5)).sum()))
