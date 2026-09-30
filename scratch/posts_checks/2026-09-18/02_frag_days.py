"""Posts check (2026-09-18): how long the sizing statistic (10d MA of the 63d
fragility column, data/rd2_fragility.parquet, point-in-time append-only) has
sat above the live sizing threshold. Feeds the journal draft only; the post
blurs the threshold to "our sizing line" per the playbook.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

f = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
ma = f["63d"].rolling(10).mean()
above = ma > 50
run = 0
for v in above.values[::-1]:
    if not v:
        break
    run += 1
print(f"last row {ma.index[-1].date()}  ma10_63d {ma.iloc[-1]:.1f}")
print(f"consecutive sessions above 50 through {ma.index[-1].date()}: {run} "
      f"(started {ma.index[-run].date()})")
since = above[above.index >= "2016-01-01"]
print(f"sessions above 50 since 2016: {int(since.sum())} of {len(since)} "
      f"({100 * since.mean():.1f}%)")
