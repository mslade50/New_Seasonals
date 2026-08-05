"""One-shot: merge the Wayback-archived legacy CBOE equity P/C CSV
(2006-11-01 -> 2019-10-04) into data/cboe_putcall.parquet.

Legacy file has only the equity series; other columns stay NaN for those rows.
Existing (newer) rows always win on any date collision.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cboe_putcall as cp

LEGACY = Path(sys.argv[1]) if len(sys.argv) > 1 else None
assert LEGACY and LEGACY.exists(), "pass path to equitypc_legacy.csv"

leg = pd.read_csv(LEGACY, skiprows=2)
leg.columns = [c.strip() for c in leg.columns]
leg["DATE"] = pd.to_datetime(leg["DATE"].str.strip(), format="%m/%d/%Y")
leg = leg.set_index("DATE").sort_index()
equity = pd.to_numeric(leg["P/C Ratio"], errors="coerce").dropna()
equity.index.name = "date"
print(f"legacy: {len(equity)} rows {equity.index.min().date()} -> {equity.index.max().date()}")

cur = cp._load()
print(f"cache before: {len(cur)} rows {cur.index.min().date()} -> {cur.index.max().date()}")

add = pd.DataFrame({"equity": equity})
add = add.loc[~add.index.isin(cur.index)]
merged = pd.concat([cur, add]).sort_index()
assert not merged.index.duplicated().any()
cp._save(merged)
print(f"cache after: {len(merged)} rows {merged.index.min().date()} -> {merged.index.max().date()}")
print(f"equity non-null: {merged['equity'].notna().sum()}")
