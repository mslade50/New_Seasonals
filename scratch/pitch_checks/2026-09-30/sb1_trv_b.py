"""sb1 TRV round-2 follow-up: P&C peer mechanism check (post-hurricane-season relief) and TRV-vs-peer residual."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sb1_engine as E

import numpy as np
import pandas as pd

PEERS = ["ALL", "CB", "PGR", "HIG", "KIE", "AIG", "CINF", "WRB"]
E.PX = load_prices(["TRV", "XLF", "SPY"] + PEERS)
have = [p for p in PEERS if p in E.PX]
base = pd.DataFrame({t: E.PX[t]["Close"] for t in ["TRV", "XLF", "SPY"]}).dropna()
w = E.windows("TRV", base, 2, 21)
print("TRV T+2/21d windows: mean %+.2f%%" % (100 * w.ret.mean()))
rows = []
for p in have:
    s = E.PX[p]["Close"].reindex(base.index)
    r = pd.Series([s.values[int(x.e1)] / s.values[int(x.e0)] - 1 for _, x in w.iterrows()], index=w.index).dropna()
    u = fwd_lag(s.dropna(), 21, 2).mean()
    m = r[r.index.isin(E.MID)]
    rows.append({"peer": p, "n": len(r), "mean%": round(100 * r.mean(), 2), "hit": f"{int((r>0).sum())}/{len(r)}",
                 "uncond%": round(100 * u, 2), "2018+%": round(100 * r[r.index >= 2018].mean(), 2),
                 "mid%": round(100 * m.mean(), 2), "mid_hit": f"{int((m>0).sum())}/{len(m)}",
                 "trv_minus_peer%": round(100 * (w.ret.reindex(r.index) - r).mean(), 2)})
print(pd.DataFrame(rows).to_string(index=False))

# does a big-Sept-cat year matter? TRV return over the prior 21d (Sep) vs Oct window
pre = pd.Series([base["TRV"].values[int(x.e0) - 2] / base["TRV"].values[int(x.e0) - 23] - 1 for _, x in w.iterrows()], index=w.index)
print(f"\ncorr(TRV Sep 21d ret, Oct window) = {np.corrcoef(pre.values, w.ret.values)[0,1]:+.2f}")
lo = pre <= pre.median()
print(f"  weak-Sep half: {100*w.ret[lo].mean():+.2f}% ({int((w.ret[lo]>0).sum())}/{lo.sum()})   "
      f"strong-Sep half: {100*w.ret[~lo].mean():+.2f}% ({int((w.ret[~lo]>0).sum())}/{(~lo).sum()})")
print(f"  this year: TRV 21d to 2026-09-29 = {100*(base['TRV'].iloc[-1]/base['TRV'].iloc[-22]-1):+.2f}%")
