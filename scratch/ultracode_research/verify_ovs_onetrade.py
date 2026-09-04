import pandas as pd

tr = pd.read_parquet('data/backtest_trades_full.parquet')
ovs = tr[tr['Strategy'].str.contains('Overbot Vol', na=False)].copy()
ovs['Signal Date'] = pd.to_datetime(ovs['Signal Date'])
fr = pd.read_parquet('data/rd2_fragility.parquet')
live = fr['63d'].rolling(10, min_periods=1).mean().rename('score').reset_index()
live.columns = ['Date', 'score']
j = pd.merge_asof(ovs.sort_values('Signal Date'), live.sort_values('Date'),
                  left_on='Signal Date', right_on='Date',
                  direction='backward', tolerance=pd.Timedelta(days=5))
j = j.dropna(subset=['score'])
band = j[(j['score'] >= 3) & (j['score'] < 21)]
cand = band[band['R_Multiple'] < -2]
print(cand[['trade_id', 'Ticker', 'Signal Date', 'Date', 'score', 'R_Multiple',
            'Size_Mult', 'Exit Type']].to_string())
# how many trades have score joined from >0 days back (weekend/gap)?
gap_days = (band['Signal Date'] - band['Date']).dt.days
print('\njoin gap-day distribution in 3-21 band:', gap_days.value_counts().to_dict())
# earliest joined trades
print('\nfirst 3 joined trades:')
print(j[['Ticker', 'Signal Date', 'score']].head(3).to_string())
