"""Verify Wednesday 2026-09-16 bars before any "today printed" claim. Commodity moves (HE -11.6, KC -6.64,
SB +5.46, CT +4.2) sit near last night's documented roll seams, so check gap vs intraday and volume vs the
20d median. ES=F +0.43 / NQ=F +1.07 disagree with cash (^GSPC -0.45, ^NDX +0.02), so check the futures'
prior bar. Rates in bp, the 5% 10-year threshold, and the sector tape behind a -1.21% Dow on a flat NDX."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pandas as pd
from pitch_lab import load_prices

TK = ['SPY', '^GSPC', 'QQQ', '^NDX', 'IWM', 'DIA', '^DJI', 'ES=F', 'NQ=F', 'TLT', 'IEF', 'HYG', '^TNX', '^FVX',
      '^IRX', '^VIX', '^VIX3M', '^MOVE', 'CL=F', 'NG=F', 'GC=F', 'HE=F', 'KC=F', 'SB=F', 'CT=F', 'ZC=F', 'ZS=F',
      'JPY=X', 'CHF=X', 'USDSEK=X', 'DX-Y.NYB', 'EEM',
      'XLK', 'XLF', 'XLV', 'XLI', 'XLE', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']
px = load_prices(TK)

print(f"{'tk':9} {'date':11} {'close':>10} {'ret1d%':>8} {'gap%':>8} {'intra%':>8} {'volx20':>8}")
for t in TK:
    d = px.get(t)
    if d is None or d.empty:
        print(f"{t:9} MISSING")
        continue
    d = d.dropna(subset=['Close'])
    c = d['Close']
    for i in (-3, -2, -1):
        ret = (c.iloc[i] / c.iloc[i - 1] - 1) * 100
        gap = (d['Open'].iloc[i] / c.iloc[i - 1] - 1) * 100
        intra = (c.iloc[i] / d['Open'].iloc[i] - 1) * 100
        vm = d['Volume'].iloc[i - 20:i].median()
        vr = d['Volume'].iloc[i] / vm if vm and vm > 0 else float('nan')
        print(f"{t:9} {str(d.index[i].date()):11} {c.iloc[i]:10.3f} {ret:8.2f} {gap:8.2f} {intra:8.2f} {vr:8.2f}")

for t in ['^TNX', '^FVX', '^IRX']:
    c = px[t]['Close'].dropna()
    print(f"{t}: {c.iloc[-1]:.3f}  1d {100*(c.iloc[-1]-c.iloc[-2]):+.1f}bp  5d {100*(c.iloc[-1]-c.iloc[-6]):+.1f}bp"
          f"  10d {100*(c.iloc[-1]-c.iloc[-11]):+.1f}bp  21d {100*(c.iloc[-1]-c.iloc[-22]):+.1f}bp  252d max {c.iloc[-253:-1].max():.3f}")
tnx = px['^TNX']['Close'].dropna()
prior = tnx.iloc[:-1]
above5 = prior[prior >= 5.0]
print("last ^TNX close >= 5.00 before today:", above5.index[-1].date() if len(above5) else None,
      " max close 2008..yesterday:", round(prior['2008':].max(), 3), prior['2008':].idxmax().date())
irx = px['^IRX']['Close'].dropna()
pi = irx.iloc[:-1]
print("last ^IRX close >= today's", irx.iloc[-1], ":", pi[pi >= irx.iloc[-1]].index[-1].date() if (pi >= irx.iloc[-1]).any() else None)
fvx = px['^FVX']['Close'].dropna()
pf = fvx.iloc[:-1]
print("last ^FVX close >= today's", fvx.iloc[-1], ":", pf[pf >= fvx.iloc[-1]].index[-1].date() if (pf >= fvx.iloc[-1]).any() else None)
chf = px['CHF=X']['Close'].dropna()
pc = chf.iloc[:-1]
print("CHF=X today", round(chf.iloc[-1], 4), "last close >= today:", pc[pc >= chf.iloc[-1]].index[-1].date() if (pc >= chf.iloc[-1]).any() else None)
dx = px['DX-Y.NYB']['Close'].dropna()
r = dx.pct_change()
streak = 0
for x in r.iloc[::-1]:
    if x > 0:
        streak += 1
    else:
        break
print("DXY up streak:", streak, " last 7 closes:", [round(v, 3) for v in dx.iloc[-7:]])

spy = px['SPY']['Close'].dropna().index
vix = px['^VIX']['Close'].dropna().index
print("\n^VIX bars on non-SPY dates since 2026:", [str(x.date()) for x in vix.difference(spy) if x.year >= 2026])
