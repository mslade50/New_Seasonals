"""Verify today's headline bars and screen the commodity moves for continuous-contract roll seams.
Yesterday's brief found coffee/corn/cotton/sugar/soybeans were roll artifacts; re-run the same test."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_prices

TK = ['SPY','^GSPC','QQQ','IWM','TLT','IEF','LQD','HYG','^TNX','^FVX','^IRX',
      '^VIX','^VIX3M','^VVIX','^MOVE','CL=F','GC=F','SI=F','HG=F','PL=F','PA=F',
      'KC=F','SB=F','CT=F','ZC=F','NG=F','JPY=X','DX-Y.NYB','EEM','EURUSD=X']
px = load_prices(TK)

print(f"{'tk':10} {'date':11} {'close':>10} {'ret1d%':>8} {'gap%':>8} {'vol':>12} {'vol20med':>12} {'volratio':>8} {'52wHi%':>8} {'52wLo%':>8}")
for t in TK:
    d = px.get(t)
    if d is None or d.empty:
        print(f"{t:10} MISSING"); continue
    d = d.dropna(subset=['Close'])
    last = d.index[-1]
    c = d['Close']
    ret = (c.iloc[-1]/c.iloc[-2]-1)*100
    gap = (d['Open'].iloc[-1]/c.iloc[-2]-1)*100 if 'Open' in d else float('nan')
    v = d['Volume'].iloc[-1] if 'Volume' in d else float('nan')
    vm = d['Volume'].iloc[-21:-1].median() if 'Volume' in d else float('nan')
    vr = v/vm if vm and vm == vm and vm > 0 else float('nan')
    w = c.iloc[-252:]
    hi = (c.iloc[-1]/w.max()-1)*100
    lo = (c.iloc[-1]/w.min()-1)*100
    print(f"{t:10} {str(last.date()):11} {c.iloc[-1]:10.3f} {ret:8.2f} {gap:8.2f} {v:12.0f} {vm:12.0f} {vr:8.2f} {hi:8.2f} {lo:8.2f}")
