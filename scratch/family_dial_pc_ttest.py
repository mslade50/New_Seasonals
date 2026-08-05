"""Date-clustered Welch t for the fear split WITHIN dial>=50 family trades."""
from __future__ import annotations

import os
import sys

import pandas as pd
from scipy import stats as sps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from scratch.putcall_dial_study import load_spy, rolling_pct_rank  # noqa: E402
from scratch.family_dipbuy_putcall_study import FAMILY  # noqa: E402

pc = pd.read_parquet(os.path.join(ROOT, "data", "cboe_putcall.parquet"))["equity"].dropna().sort_index()
spy = load_spy()
cal = spy.index[(spy.index >= pc.index.min()) & (spy.index <= pc.index.max())]
pct = rolling_pct_rank(pc.reindex(cal).ffill(limit=3).rolling(10).mean(), 252).dropna()

led = pd.read_parquet(os.path.join(ROOT, "data", "backtest_trades_full.parquet"))
fam = led[led["Strategy"].isin(FAMILY)].copy()
fam["sig_date"] = pd.to_datetime(fam["Signal Date"])
fam["pc"] = fam["sig_date"].map(pct)
frag = pd.read_parquet(os.path.join(ROOT, "data", "rd2_fragility_ts.parquet"))
frag.index = pd.to_datetime(frag.index)
fam["dial"] = fam["sig_date"].map(frag["63d"].rolling(10, min_periods=1).mean())
fam = fam.dropna(subset=["pc", "dial", "R_Multiple"])

hi = fam[fam["dial"] >= 50]
on = hi[hi["pc"] > 85].groupby("sig_date")["R_Multiple"].mean()
off = hi[hi["pc"] <= 85].groupby("sig_date")["R_Multiple"].mean()
t, p = sps.ttest_ind(on, off, equal_var=False)
print(f"dial>=50: fear dates n={len(on)} avg {on.mean():.3f} vs "
      f"no-fear dates n={len(off)} avg {off.mean():.3f} -> t={t:.2f} p={p:.4f}")
mw = sps.mannwhitneyu(on, off, alternative="two-sided")
print(f"Mann-Whitney (rank-based, small-n robust): U={mw.statistic:.0f} p={mw.pvalue:.4f}")
