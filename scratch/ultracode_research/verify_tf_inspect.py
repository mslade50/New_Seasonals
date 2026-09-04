import pandas as pd

mp = pd.read_parquet("data/master_prices.parquet")
print(type(mp.index), mp.index.names, mp.shape)
print(mp.columns.tolist()[:20])
print(mp.head(3))
print(mp.tail(3))

frag = pd.read_parquet("data/rd2_fragility.parquet")
print(frag.shape, frag.columns.tolist(), frag.index.min(), frag.index.max())

tr = pd.read_parquet("data/backtest_trades_full.parquet")
print(tr.shape, tr.columns.tolist())
print(tr[['Strategy','Exit Date','PnL_flat_750k']].head(3))
