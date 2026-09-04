"""Independent data-forensics check of the EWZ phantom-fill theory.

Verifies, from the LOCAL parquet cache, whether the master_prices.parquet
(written Jun 11 06:47) shows EWZ pre-ex or post-ex bars, and what the
6/8 Close + ATR(14) produce for the -0.25 ATR limit.

Then attempts a LIVE yfinance pull to recover the raw vs adjusted lows and
the actual ex-div date/factor.
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(r"C:\Users\McKinley Slade\dev\New_Seasonals")
MP = ROOT / "data" / "master_prices.parquet"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

df = pd.read_parquet(MP)
print("=== master_prices.parquet shape/cols ===")
print(df.shape)
print(list(df.columns)[:20])
print("index name:", df.index.name, "| index type:", type(df.index))

# How is EWZ stored? Could be MultiIndex columns or a Ticker column.
print("\n=== locate EWZ ===")
if isinstance(df.columns, pd.MultiIndex):
    print("MultiIndex columns. levels:", df.columns.names)
    tickers = df.columns.get_level_values(-1).unique()
    print("EWZ present:", "EWZ" in set(tickers))
elif "Ticker" in df.columns:
    print("Ticker column present. EWZ rows:", (df["Ticker"] == "EWZ").sum())
else:
    print("Other layout. sample index:", df.index[:3])
