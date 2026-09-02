"""Inspect ledger schema + strategy_config execution fields relevant to flow-conditional sizing."""
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
sys.path.insert(0, str(ROOT))
pd.set_option("display.width", 250, "display.max_columns", 60)

led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
print(led.dtypes.to_string())
print("rows", len(led), "date range", led["Signal Date"].min(), led["Signal Date"].max())
print(led.groupby("Strategy").agg(N=("R_Multiple", "size"), first=("Signal Date", "min"), avgR=("R_Multiple", "mean"),
                                   risk_bps=("Risk bps", "mean"), size_mult=("Size_Mult", "mean")).to_string())
print(led.head(3).T.to_string())
for c in led.columns:
    if led[c].dtype == object and led[c].nunique() < 30:
        print(c, led[c].unique()[:30])
print(led.schema_metadata if hasattr(led, "schema_metadata") else "")

import strategy_config as sc
for s in sc.STRATEGY_BOOK:
    ex = s.get("execution", {})
    keys = {k: ex.get(k) for k in ["risk_bps", "hold_days", "entry_type", "frag_risk_bands", "same_day_signal_derate", "same_day_derate_floor",
                                    "signal_recency_ladder", "max_one_pos", "fill_window_days", "pc_fear_bands", "cycle_risk_mults", "stop_atr", "target_atr"]}
    print(s.get("name"), "| dir", s.get("direction"), "| universe n", len(s.get("universe", []) or []), "|", {k: v for k, v in keys.items() if v is not None})
