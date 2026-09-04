import pandas as pd
import pyarrow.parquet as pq

root = r"C:\Users\McKinley Slade\dev\New_Seasonals"

f = pq.ParquetFile(root + r"\data\master_prices.parquet")
print("master_prices schema (first 20):")
print([f.schema_arrow.names[i] for i in range(min(20, len(f.schema_arrow.names)))])
print("num cols:", len(f.schema_arrow.names), "num rows:", f.metadata.num_rows)

frag = pd.read_parquet(root + r"\data\rd2_fragility.parquet")
print("\nfrag:", frag.shape, frag.columns.tolist(), frag.index[:3], frag.index[-3:])

tr = pd.read_parquet(root + r"\data\backtest_trades_full.parquet")
print("\ntrades:", tr.shape)
print(tr.columns.tolist())
print(tr[['Strategy','Signal Date','R_Multiple']].head(3))
print("Signal Date dtype:", tr['Signal Date'].dtype)
print("strategies:", tr['Strategy'].value_counts().to_dict())
