"""C1 addendum: era split of the PITCHED cell at the CORRECT horizon (h=9,
entry 2026-08-18 close -> exit 2026-08-31 close), both vehicle shapes.
Closes required item (3): 2018+ and 2021+ called out separately."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np, pandas as pd

LAG, H = 1, 9
raw = load_prices(["SPY", "TLT"])
spy, tlt = raw["SPY"]["Close"], raw["TLT"]["Close"]
idx = spy.index.intersection(tlt.index)
px = pd.DataFrame({"SPY": spy.reindex(idx), "TLT": tlt.reindex(idx)}).dropna(); idx = px.index
div = spy.pct_change(21).reindex(idx) - tlt.pct_change(21).reindex(idx)
ymv = pd.Series(idx.year*100+idx.month, index=idx)
is_last = ymv.ne(ymv.shift(-1)); is_last.iloc[-1] = False
pos = pd.Series(range(len(idx)), index=idx); t9 = pos+LAG+H
A = pd.Series(False, index=idx); ok = t9 < len(idx)
A.loc[idx[ok.values]] = is_last.values[t9[ok.values].values]
rT, rS = fwd_lag(px["TLT"], H, LAG), fwd_lag(px["SPY"], H, LAG)
for name, r in (("LONG TLT (primary pitched shape)", rT), ("SPREAD TLT-SPY", rT-rS)):
    v = r.notna()
    for glab, g in (("gate OFF", pd.Series(True, index=idx)),
                    ("div >= +5pp", (div >= 0.05).fillna(False)),
                    ("div >= +7.32pp (LIVE)", (div >= 0.0732).fillna(False))):
        d = idx[(A & g).values & v.values]; x = r.loc[d]
        rows = [summarize(x.values, "all"),
                summarize(x[d < pd.Timestamp("2018-01-01")].values, "pre-2018"),
                summarize(x[d >= pd.Timestamp("2018-01-01")].values, "2018+"),
                summarize(x[d >= pd.Timestamp("2021-01-01")].values, "2021+")]
        show(rows, f"{name} | {glab} | h=9")
