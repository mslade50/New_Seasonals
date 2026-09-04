from pathlib import Path
import pandas as pd

root = Path(r"C:\Users\McKinley Slade\dev\New_Seasonals")

frag = pd.read_parquet(root / "data" / "rd2_fragility.parquet")
print("frag:", frag.shape, frag.columns.tolist(), frag.index[:3], frag.index[-1])

led = pd.read_parquet(root / "data" / "backtest_trades_full.parquet")
print("ledger:", led.shape)
print(led[["Strategy", "Exit Date", "PnL_flat_750k"]].head(3))

mp = pd.read_parquet(root / "data" / "master_prices.parquet")
print("mp cols:", mp.columns.tolist()[:12], "index:", type(mp.index), mp.shape)
print(mp.head(3))

fep = pd.read_parquet(root / "scratch" / "ultracode_research" / "factor_etf_prices.parquet")
print("factor_etf_prices:", fep.shape, fep.columns.tolist())
print(fep.head(3))
print(fep.tail(3))
print(fep.apply(lambda s: s.first_valid_index()))
