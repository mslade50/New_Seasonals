import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\McKinley Slade\dev\New_Seasonals")

frag = pd.read_parquet(ROOT / 'data' / 'rd2_fragility.parquet')
live = frag['63d'].rolling(10, min_periods=1).mean()
mmean = live.groupby(pd.Grouper(freq='ME')).mean().loc['2016-07-01':'2026-06-30']
hi = mmean[mmean >= 50].index

tr = pd.read_parquet(ROOT / 'data' / 'backtest_trades_full.parquet')
tr['Exit Date'] = pd.to_datetime(tr['Exit Date'])
tr['m'] = tr['Exit Date'] + pd.offsets.MonthEnd(0)

for label, sub in [('BOOK all', tr),
                   ('BOOK ex-OVS', tr[tr['Strategy'] != 'Overbot Vol Spike']),
                   ('OVS only', tr[tr['Strategy'] == 'Overbot Vol Spike'])]:
    m = sub.groupby('m')['PnL_flat_750k'].sum() / 750_000.0
    m = m.reindex(mmean.index, fill_value=0.0)
    print(f"{label:12s} hi-frag avg/mo {m.reindex(hi).mean()*100:+.2f}%  "
          f"rest {m.drop(hi).mean()*100:+.2f}%  "
          f"(hi months w/ trades: {(m.reindex(hi) != 0).sum()}/{len(hi)})")

# per-episode breakdown of A1 sector sleeve in the 16 hi months
import numpy as np
sys_path = str(ROOT)
mp = pd.read_parquet(ROOT / 'data' / 'master_prices.parquet',
                     columns=['ticker', 'date', 'Close'])
SECTORS = ['XLB','XLC','XLE','XLF','XLI','XLK','XLP','XLRE','XLU','XLV','XLY']
w = mp[mp['ticker'].isin(SECTORS)].pivot_table(index='date', columns='ticker',
                                               values='Close').sort_index()
w.index = pd.to_datetime(w.index)
pxm = w.ffill(limit=5).resample('ME').last()
px = pxm[SECTORS]
rets = px.pct_change(fill_method=None)
mom = 0.5*(px/px.shift(6)-1) + 0.5*(px/px.shift(12)-1)
elig = px.notna().cumsum() >= 13
r = {}
for i in range(len(px.index)-1):
    t, t1 = px.index[i], px.index[i+1]
    m_ = mom.loc[t].where(elig.loc[t]).dropna()
    if len(m_) < 3:
        continue
    top = m_.nlargest(3)
    r[t1] = rets.loc[t1].reindex(top.index).fillna(0).mean()
a1 = pd.Series(r)
print("\nA1 gross return by hi-frag month:")
for d in hi:
    print(f"  {d.strftime('%Y-%m')}: {a1.get(d, float('nan'))*100:+.2f}%")
