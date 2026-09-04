import pandas as pd

tr = pd.read_parquet('data/backtest_trades_full.parquet')
print(tr.columns.tolist())
ovs = tr[tr['Strategy'].str.contains('Overbot Vol', na=False)]
print('OVS full history N=', len(ovs))
print(ovs[['Signal Date','Entry Date','Exit Date']].dtypes)
if 'Size_Mult' in ovs.columns:
    print(ovs['Size_Mult'].value_counts(dropna=False))
fr = pd.read_parquet('data/rd2_fragility.parquet')
print(fr.head())
print(fr.tail())
print(fr.index.min(), fr.index.max())
