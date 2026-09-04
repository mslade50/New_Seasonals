from pathlib import Path
import pandas as pd

root = Path(r"C:\Users\McKinley Slade\dev\New_Seasonals")
led = pd.read_parquet(root / "data/backtest_trades_full.parquet")
print("ledger shape", led.shape)
print(led.dtypes.to_string())
print(led.head(3).to_string())
frag = pd.read_parquet(root / "data/rd2_fragility.parquet")
print("\nfrag", frag.shape, frag.columns.tolist(), frag.index.min(), frag.index.max())
mp = pd.read_parquet(root / "data/master_prices.parquet")
print("\nmaster_prices cols type:", type(mp.columns), mp.shape)
print(mp.columns[:10].tolist())
print(mp.index[:3], mp.index[-3:])
