import pandas as pd
from pathlib import Path

root = Path(r"C:\Users\McKinley Slade\dev\New_Seasonals")

mp = pd.read_parquet(root / "data/master_prices.parquet")
print("master_prices shape:", mp.shape)
print("columns type:", type(mp.columns))
print("first 10 cols:", list(mp.columns[:10]))
print("index:", mp.index[:3], mp.index[-3:])

frag = pd.read_parquet(root / "data/rd2_fragility.parquet")
print("\nfrag shape:", frag.shape, "cols:", list(frag.columns))
print(frag.tail(3))

tr = pd.read_parquet(root / "data/backtest_trades_full.parquet")
print("\ntrades shape:", tr.shape)
print(tr[["Strategy", "Signal Date", "Exit Date", "PnL_flat_750k"]].tail(3))
