"""C9 family check + book overlap.

(1) Is 'long crude after a drop inside an uptrend' (C9, and the c6 long flip) a 2026
    regime effect? c6 def long and C9 by year-bucket, ex-2026.
(2) Book overlap: what the ledger (data/backtest_trades_full.parquet) holds or held
    recently in energy / gold / natgas names, OVS in particular.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

s = close_panel(["USO"])["USO"].dropna()
s = s[s > 0]
r1 = s.pct_change()
r21 = pct_rank(s, 21)
r63 = pct_rank(s, 63)
crack = r1 <= -0.03
prior = crack.shift(1).rolling(10).sum() > 0
c6 = ((r21 >= 75) & crack & ~prior).fillna(False)
c9 = ((r1 <= -0.04) & (r63 >= 70)).fillna(False)
for lbl, m in (("c6 def (long)", c6), ("C9", c9)):
    for h in (3, 5):
        ret = fwd_lag(s, h, 1)
        t = s.index[m.values & ret.notna().values]
        e = declusters(t, h, s.index)
        v = ret.loc[e]
        ex = v[v.index.year < 2026]
        w, we = int((v > 0).sum()), int((ex > 0).sum())
        print(f"{lbl:14s} h={h}: all {100*v.mean():+.3f}% ({w}-{len(v)-w})  ex-2026 {100*ex.mean():+.3f}% "
              f"({we}-{len(ex)-we}, sign p {sign_test(we, len(ex)):.3f})  2026 n={len(v)-len(ex)} "
              f"share {100*v[v.index.year==2026].sum()/v.sum():.0f}%")

bt = pd.read_parquet(Path(__file__).resolve().parents[3] / "data" / "backtest_trades_full.parquet")
print("\nledger columns:", list(bt.columns)[:30])
tcol = next(c for c in bt.columns if c.lower() in ("ticker", "symbol"))
scol = next((c for c in bt.columns if "strat" in c.lower()), None)
ecol = next((c for c in bt.columns if "entry" in c.lower() and "date" in c.lower()), None)
xcol = next((c for c in bt.columns if "exit" in c.lower() and "date" in c.lower()), None)
print("using", tcol, scol, ecol, xcol)
bt[ecol] = pd.to_datetime(bt[ecol])
names = ["USO", "XLE", "XOP", "OXY", "HAL", "UNG", "GLD", "GDX", "SLV", "CVX", "XOM", "COP", "SLB", "DVN", "EOG",
         "FANG", "MPC", "VLO", "PSX", "APA", "OIH", "BNO", "UCO", "SCO", "BOIL", "KOLD", "NUGT", "DUST", "MRO", "HES"]
rec = bt[(bt[ecol] >= "2026-08-15") & bt[tcol].isin(names)]
cols = [c for c in (tcol, scol, ecol, xcol, "direction", "Direction", "side") if c and c in bt.columns]
print(f"\nledger rows in energy/gold names entered since 2026-08-15: {len(rec)}")
print(rec[cols].tail(20).to_string(index=False))
if scol:
    ovs = bt[bt[scol].astype(str).str.contains("Overbot|OVS", case=False)]
    ovs_e = ovs[ovs[tcol].isin(names)]
    print(f"\nOVS rows all-time in these names: {len(ovs_e)} of {len(ovs)}; last 5:")
    print(ovs_e.sort_values(ecol)[cols].tail(5).to_string(index=False))
