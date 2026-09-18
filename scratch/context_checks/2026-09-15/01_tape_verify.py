"""Verify Tuesday 2026-09-15 bars. Tonight's commodity moves (KC -7.81, ZC +4.64, SB +4.07) sit within
a few bp of last night's (KC -7.78, ZC +4.65, SB +5.01), so check for duplicated bars and roll seams
(gap vs intraday, volume vs 20d median). HE=F -12.59% is new and hogs have documented roll gaps.
Also print rates in bp and the ^VIX phantom-bar list."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pandas as pd
from pitch_lab import load_prices

TK = ['SPY', '^GSPC', 'QQQ', 'IWM', 'TLT', 'IEF', 'LQD', 'HYG', '^TNX', '^FVX', '^IRX',
      '^VIX', '^VIX3M', '^MOVE', 'CL=F', 'GC=F', 'HE=F', 'KC=F', 'SB=F', 'CT=F',
      'ZC=F', 'ZS=F', 'ZW=F', 'JPY=X', 'DX-Y.NYB', 'EEM', 'USDSEK=X']
px = load_prices(TK)

print(f"{'tk':9} {'date':11} {'close':>10} {'ret1d%':>8} {'gap%':>8} {'intra%':>8} {'volx20':>8}")
for t in TK:
    d = px.get(t)
    if d is None or d.empty:
        print(f"{t:9} MISSING")
        continue
    d = d.dropna(subset=['Close'])
    c = d['Close']
    for i in (-2, -1):
        ret = (c.iloc[i] / c.iloc[i - 1] - 1) * 100
        gap = (d['Open'].iloc[i] / c.iloc[i - 1] - 1) * 100
        intra = (c.iloc[i] / d['Open'].iloc[i] - 1) * 100
        vm = d['Volume'].iloc[i - 20:i].median()
        vr = d['Volume'].iloc[i] / vm if vm and vm > 0 else float('nan')
        print(f"{t:9} {str(d.index[i].date()):11} {c.iloc[i]:10.3f} {ret:8.2f} {gap:8.2f} {intra:8.2f} {vr:8.2f}")

for t in ['^TNX', '^FVX', '^IRX']:
    c = px[t]['Close'].dropna()
    print(f"{t}: {c.iloc[-1]:.3f}  1d {100*(c.iloc[-1]-c.iloc[-2]):+.1f}bp  5d {100*(c.iloc[-1]-c.iloc[-6]):+.1f}bp"
          f"  10d {100*(c.iloc[-1]-c.iloc[-11]):+.1f}bp  21d {100*(c.iloc[-1]-c.iloc[-22]):+.1f}bp  252d max {c.iloc[-252:].max():.3f}")
tnx = px['^TNX']['Close'].dropna()
above5 = tnx[tnx >= 5.0]
print("last ^TNX close >= 5.00:", above5.index[-1].date() if len(above5) else None, "max since 2008:", tnx['2008':].max(),
      tnx['2008':].idxmax().date())

spy = px['SPY']['Close'].dropna().index
vix = px['^VIX']['Close'].dropna().index
print("\n^VIX bars on non-SPY dates since 2020:", [str(x.date()) for x in vix.difference(spy) if x.year >= 2020])
print("last 4 ^VIX bars:", [(str(i.date()), round(v, 2)) for i, v in px['^VIX']['Close'].dropna().iloc[-4:].items()])
