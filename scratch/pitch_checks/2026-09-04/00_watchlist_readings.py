"""Today's readings for the watchlist legs that are not directly in the tape file."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import pandas as pd, numpy as np

TK=['^TNX','DX-Y.NYB','^SKEW','GDX','GLD','FXI','IHI','SMH','XLI','XLU','TLT','USO','^VIX','^MOVE','IEF','LQD','HYG','SPY','XLE','IWM','XLV','XLK','XOP','COP','CVX','VLO','OXY','SLB','EOG','HAL','WMB','XLF','XLY','XLP','XLB','XLRE','XLC']
px = close_panel(TK)
t = json.load(open('data/pitch_tape.json'))['tickers']
ASOF = pd.Timestamp('2026-09-03')

def cl(tk):
    s = px[tk].dropna(); return s[s.index <= ASOF]

def pit_rank(tk, win, look=252):
    s = cl(tk); r = s.pct_change(win)
    return r.rolling(look).apply(lambda a: (a[:-1] < a[-1]).mean()*100, raw=True).iloc[-1]

def lvl_pctile(tk, look=252):
    s = cl(tk); return (s.tail(look) < s.iloc[-1]).mean()*100

out = {}
for tk, win in [('^TNX',21),('DX-Y.NYB',21),('^SKEW',5),('GDX',5),('GLD',5),('FXI',5),('IHI',21),
                ('SMH',63),('SMH',5),('XLI',5),('XLU',21),('TLT',21),('USO',63),('USO',5),('^VIX',21)]:
    try: out[f'{tk}_r{win}'] = round(float(pit_rank(tk,win)),1)
    except Exception as e: out[f'{tk}_r{win}'] = f'ERR {e}'
for tk in ['^MOVE','^SKEW','^VIX']:
    try: out[f'{tk}_lvl_pctile252'] = round(float(lvl_pctile(tk)),1)
    except Exception as e: out[f'{tk}_lvl_pctile252'] = f'ERR {e}'

# 52w distances the watchlist legs name
for tk in ['TLT','IEF','LQD','HYG','SPY','XLE','GLD','SMH','IWM']:
    s = cl(tk); hi = s.tail(252).max(); lo = s.tail(252).min()
    out[f'{tk}_pct_above_252low'] = round(float(s.iloc[-1]/lo-1)*100, 2)
    out[f'{tk}_pct_below_252high'] = round(float(s.iloc[-1]/hi-1)*100, 2)

# VIX 21d relative-range percentile (watchlist 33/35 arm)
v = cl('^VIX')
rel = (v - v.rolling(21).min()) / (v.rolling(21).max() - v.rolling(21).min()) * 100
out['VIX_rel_range_pct_21d'] = round(float(rel.iloc[-1]),2)
rng = (v.rolling(21).max()/v.rolling(21).min() - 1)
out['VIX_21d_range_pctile_252'] = round(float((rng.tail(252) < rng.iloc[-1]).mean()*100),1)

# 252-session yield change (watchlist 18 arm, in bp)
y = cl('^TNX')
out['TNX_252d_change_bp'] = round(float((y.iloc[-1] - y.iloc[-253])*100),1)

# one-day XLV minus XLK gap (watchlist 14)
out['XLV_minus_XLK_1d_pp'] = round(float(cl('XLV').pct_change().iloc[-1]*100 - cl('XLK').pct_change().iloc[-1]*100),2)

# energy complex z10 >= 2 count (watchlist 19)
COMPLEX = ['XLE','XOP','USO','COP','CVX','VLO','OXY','SLB','EOG','HAL','WMB']
z = {}
for tk in COMPLEX:
    s = cl(tk); r = s.pct_change()
    z[tk] = round(float((s.iloc[-1]/s.iloc[-11]-1) / (r.rolling(21).std().iloc[-1]*np.sqrt(10))),2)
out['energy_z10'] = z
out['energy_z10_ge2_count'] = sum(1 for v_ in z.values() if v_ >= 2.0)

# sector triple-floor holders (watchlist 34)
SEC = ['XLK','XLF','XLV','XLY','XLP','XLI','XLB','XLU','XLRE','XLC','XLE']
tri = []
for tk in SEC:
    try:
        if pit_rank(tk,5) <= 15 and pit_rank(tk,21) <= 15 and pit_rank(tk,63) <= 15: tri.append(tk)
    except Exception: pass
out['sector_triple_floor'] = tri
print(json.dumps(out, indent=1))
