import pandas as pd
import numpy as np
from pathlib import Path

df = pd.read_parquet("data/master_prices.parquet")
ewz = df[df['ticker'] == 'EWZ'].copy()
ewz['date'] = pd.to_datetime(ewz['date'])
ewz = ewz.sort_values('date').set_index('date')
window = ewz.loc['2026-05-15':'2026-06-12', ['Open','High','Low','Close']]
print("EWZ tail of cache:")
print(window.tail(8).to_string())

# Compute ATR(14) Wilder as the backtester likely does. Check daily_scan / indicators for exact method.
high = ewz['High']; low = ewz['Low']; close = ewz['Close']
pc = close.shift(1)
tr = pd.concat([high-low, (high-pc).abs(), (low-pc).abs()], axis=1).max(axis=1)
# Wilder RMA
atr_wilder = tr.ewm(alpha=1/14, adjust=False).mean()
# Simple rolling mean
atr_sma = tr.rolling(14).mean()
print("\n6/8 ATR Wilder:", round(float(atr_wilder.loc['2026-06-08']),6))
print("6/8 ATR SMA14 :", round(float(atr_sma.loc['2026-06-08']),6))
print("6/8 Close:", float(ewz.loc['2026-06-08','Close']))
c = float(ewz.loc['2026-06-08','Close'])
for name, a in [('Wilder', float(atr_wilder.loc['2026-06-08'])), ('SMA', float(atr_sma.loc['2026-06-08']))]:
    lim = c - 0.25*a
    print(f"  limit via {name}: {lim:.4f} round2={round(lim,2)}")
