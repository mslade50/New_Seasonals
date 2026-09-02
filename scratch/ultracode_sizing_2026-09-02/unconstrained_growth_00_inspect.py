"""Inspect inputs for the unconstrained-growth study (schema only)."""
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
pd.set_option("display.width", 250, "display.max_columns", 60)

led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
print("ledger cols:", list(led.columns))
print(led.dtypes.to_string())
print("rows", len(led), "with pnl", led["PnL_flat_750k"].notna().sum())
print("dates", led["Entry Date"].min(), led["Exit Date"].max())
print("Direction:", led["Direction"].value_counts().to_dict())
print("Tier:", led["Tier"].value_counts().to_dict())
print(led.groupby("Strategy").size().to_string())
print(led[["Strategy", "Ticker", "Direction", "Entry Date", "Exit Date", "Shares_flat", "Entry Price", "Risk_flat_750k", "PnL_flat_750k", "R_Multiple", "Size_Mult"]].tail(5).to_string())
meta = pq.read_schema(ROOT / "data/backtest_trades_full.parquet").metadata
print({k.decode(): v.decode()[:60] for k, v in meta.items() if k.startswith(b"ledger")})

dp = pd.read_parquet(ROOT / "data/backtest_daily_pnl.parquet")
print("daily pnl cols:", list(dp.columns), len(dp), dp["date"].min(), dp["date"].max())
print(dp.tail(3).to_string())

sch = pq.read_schema(ROOT / "data/master_prices.parquet")
print("master_prices schema:", sch.names)
md = pq.ParquetFile(ROOT / "data/master_prices.parquet").metadata
print("row groups", md.num_row_groups, "rows", md.num_rows)

fr = pd.read_parquet(ROOT / "data/rd2_fragility.parquet")
print("frag cols", list(fr.columns), fr.index.min(), fr.index.max(), len(fr))

import sys
sys.path.insert(0, str(ROOT))
import strategy_config as sc
print("GRM", sc.GLOBAL_RISK_MULTIPLIER, "ACCOUNT_VALUE", sc.ACCOUNT_VALUE)
print("LEV3X_ALL n", len(getattr(sc, "LEV3X_ALL", [])), getattr(sc, "LEV3X_ALL", [])[:50])
for s in sc.STRATEGY_BOOK:
    ex = s.get("execution", {})
    print(s.get("name"), "| risk_bps", ex.get("risk_bps"), "| stop_atr", ex.get("stop_atr"), "| hold", ex.get("hold_days"), "| dir", s.get("direction") or ex.get("direction"))
