"""Inspect the ledger + supporting data for the estimation-haircut study."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
print("rows", len(led))
print("cols", list(led.columns))
print(led.dtypes.to_string())
import pyarrow.parquet as pq  # noqa: E402
print("meta", {k.decode(): v.decode()[:80] for k, v in (pq.read_schema(ROOT / "data/backtest_trades_full.parquet").metadata or {}).items() if k.startswith(b"ledger")})
led["Signal Date"] = pd.to_datetime(led["Signal Date"])
print("date range", led["Signal Date"].min(), led["Signal Date"].max())
g = led.groupby(["Strategy", "Tier"]).agg(N=("R_Multiple", "size"), avgR=("R_Multiple", "mean"), first=("Signal Date", "min"), last=("Signal Date", "max"), pnl=("PnL_flat_750k", "sum"))
print(g.to_string())
for c in ["Exit Type", "Direction", "Tranche", "Entry Type", "Exit_Type", "ExitType"]:
    if c in led.columns:
        print(c, led[c].value_counts().to_dict())
print(led.head(3).T.to_string())

sys.path.insert(0, str(ROOT))
import strategy_config as sc  # noqa: E402

print("STRATEGY_BOOK names:")
for s in sc.STRATEGY_BOOK:
    ex = s.get("execution", {})
    print(" -", s["name"], "| risk_bps", ex.get("risk_bps"), "| entry", ex.get("entry_type"), "| hold", ex.get("hold_days"), "| universe len", len(s.get("universe", []) or []) if not isinstance(s.get("universe"), str) else s.get("universe"))
