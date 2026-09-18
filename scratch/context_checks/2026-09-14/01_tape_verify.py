"""Verify Monday 2026-09-14 headline bars, screen commodity moves for roll seams,
and list ^VIX bars that sit on non-NYSE dates (phantom-bar fault)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pandas as pd
from pitch_lab import load_prices

TK = ['SPY', '^GSPC', 'QQQ', 'IWM', 'TLT', 'IEF', 'LQD', 'HYG', '^TNX', '^FVX', '^IRX',
      '^VIX', '^VIX3M', '^VVIX', '^MOVE', 'CL=F', 'GC=F', 'SI=F', 'HG=F', 'KC=F', 'SB=F',
      'ZC=F', 'ZS=F', 'LE=F', 'JPY=X', 'EEM', 'EWJ', 'EWZ', 'FXI', 'EWT', 'EWY', 'INDA']
px = load_prices(TK)

print(f"{'tk':8} {'date':11} {'close':>10} {'ret1d%':>8} {'gap%':>8} {'intraday%':>9} {'volratio':>8} {'52wHi%':>8}")
for t in TK:
    d = px.get(t)
    if d is None or d.empty:
        print(f"{t:8} MISSING")
        continue
    d = d.dropna(subset=['Close'])
    c = d['Close']
    ret = (c.iloc[-1] / c.iloc[-2] - 1) * 100
    gap = (d['Open'].iloc[-1] / c.iloc[-2] - 1) * 100
    intra = (c.iloc[-1] / d['Open'].iloc[-1] - 1) * 100
    v = d['Volume'].iloc[-1]
    vm = d['Volume'].iloc[-21:-1].median()
    vr = v / vm if vm and vm == vm and vm > 0 else float('nan')
    hi = (c.iloc[-1] / c.iloc[-252:].max() - 1) * 100
    print(f"{t:8} {str(d.index[-1].date()):11} {c.iloc[-1]:10.3f} {ret:8.2f} {gap:8.2f} {intra:9.2f} {vr:8.2f} {hi:8.2f}")

spy = px['SPY']['Close'].dropna().index
vix = px['^VIX']['Close'].dropna().index
extra = vix.difference(spy)
print("\n^VIX bars on non-SPY dates since 2020:", [str(x.date()) for x in extra if x.year >= 2020])
print("last 6 ^VIX bars:", [(str(i.date()), round(v, 2)) for i, v in px['^VIX']['Close'].dropna().iloc[-6:].items()])
print("last 6 EEM bars:", [(str(i.date()), round(v, 2)) for i, v in px['EEM']['Close'].dropna().iloc[-6:].items()])
