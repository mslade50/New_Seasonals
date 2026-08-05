"""Within the dial>=50 regime, does the P/C fear state separate family trades?

2x2: dial (10d MA of rd2_fragility_ts 63d, research recompute, 2017+) x
P/C fear state (pct252 of 10d-MA equity P/C > 85). Same trade set as
family_dipbuy_putcall_study.py. NOTE: research-recompute dial, not the PIT
vintage the frag band was validated on — indicative only.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from scratch.putcall_dial_study import load_spy, rolling_pct_rank  # noqa: E402
from scratch.family_dipbuy_putcall_study import FAMILY  # noqa: E402

pc = pd.read_parquet(os.path.join(ROOT, "data", "cboe_putcall.parquet"))
equity = pc["equity"].dropna().sort_index()
spy = load_spy()
cal = spy.index[(spy.index >= equity.index.min()) & (spy.index <= equity.index.max())]
eq = equity.reindex(cal).ffill(limit=3)
pct = rolling_pct_rank(eq.rolling(10, min_periods=10).mean(), 252).dropna()

led = pd.read_parquet(os.path.join(ROOT, "data", "backtest_trades_full.parquet"))
fam = led[led["Strategy"].isin(FAMILY)].copy()
fam["sig_date"] = pd.to_datetime(fam["Signal Date"])
fam["pc_pct"] = fam["sig_date"].map(pct)

frag = pd.read_parquet(os.path.join(ROOT, "data", "rd2_fragility_ts.parquet"))
frag.index = pd.to_datetime(frag.index)
dial = frag["63d"].rolling(10, min_periods=1).mean()
fam["dial"] = fam["sig_date"].map(dial)
fam = fam.dropna(subset=["pc_pct", "dial", "R_Multiple"])
print(f"{len(fam)} family trades with both states "
      f"({fam['sig_date'].min().date()} -> {fam['sig_date'].max().date()})\n")

fam["dial_hi"] = fam["dial"] >= 50
fam["fear"] = fam["pc_pct"] > 85
g = fam.groupby(["dial_hi", "fear"])["R_Multiple"].agg(
    n="size", avgR="mean", medR="median",
    win=lambda s: 100 * (s > 0).mean()).round(3)
print(g.to_string())

hi = fam[fam["dial_hi"]]
print(f"\ndial>=50 total: n={len(hi)} avgR={hi['R_Multiple'].mean():.3f} "
      f"({hi.groupby('sig_date').ngroups} signal dates)")
for fear, gg in hi.groupby("fear"):
    d = gg.groupby("sig_date")["R_Multiple"].mean()
    print(f"  fear={fear}: {len(gg)} trades / {len(d)} dates, "
          f"date-avgR {d.mean():.3f}, years: "
          f"{sorted(gg['sig_date'].dt.year.unique())}")
