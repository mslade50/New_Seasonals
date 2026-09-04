from pathlib import Path
import pandas as pd

root = Path(r"C:\Users\McKinley Slade\dev\New_Seasonals")

fr = pd.read_parquet(root / "data" / "rd2_fragility.parquet")
print("frag shape", fr.shape, "cols", list(fr.columns))
print(fr.head(3))
print(fr.tail(3))

mp = pd.read_parquet(root / "data" / "master_prices.parquet")
print("mp type cols:", type(mp.columns), mp.shape)
print(mp.columns[:10])
print(mp.index[:3], mp.index[-3:])

tr = pd.read_parquet(root / "data" / "backtest_trades_full.parquet")
print("trades", tr.shape)
print(tr.columns.tolist())
print(tr[['Strategy','Signal Date','Entry Date','Exit Date','R_Multiple','PnL_flat_750k']].head(3))
