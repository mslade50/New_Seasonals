"""Surface-map reconnaissance: today's live values for every cell verdict."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change, rolling_on_valid
import numpy as np, pandas as pd

def level_rank(s, lookback=252):
    return rolling_on_valid(s, lambda x: x.rolling(lookback).rank(pct=True) * 100.0)

TK = ['SPY','QQQ','IWM','TLT','IEF','^TNX','HYG','LQD','GLD','GDX','SLV',
      'USO','UNG','DBC','XLE','XOP','OIH','UUP','DX-Y.NYB','EFA','EEM','FXI',
      '^VIX','^VIX3M','^MOVE','SVXY','UVXY','^SKEW','^GSPC','^NDX','EWZ','EWJ',
      'XLK','XLV','XLU','XLP','XLI','XLF','XLY','XLB','VLO','CVX','COP']
px = load_prices(TK)
spy = px['SPY']['Close']; idx = spy.index
vix = px['^VIX']['Close'].reindex(idx).ffill()

relrange = (vix.rolling(21).max() - vix.rolling(21).min()) / vix.rolling(21).mean()
rr_pct = level_rank(relrange, 252)
print('VIX 21d rel-range = %.4f, trailing-252 LEVEL pctile = %.2f' % (relrange.iloc[-1], rr_pct.iloc[-1]))
print(rr_pct.tail(6).round(2).to_string())

uso = px['USO']['Close']; dbc = px['DBC']['Close']
for t,s in [('USO',uso),('DBC',dbc),('XLE',px['XLE']['Close']),('XOP',px['XOP']['Close'])]:
    r21=_valid_pct_change(s,21)
    print('%-4s 21d ret %+7.2f%%  PIT rank(252) %5.1f' % (t, 100*r21.iloc[-1], pct_rank(s,21,252).iloc[-1]))

SEC=['XLK','XLV','XLU','XLP','XLI','XLF','XLY','XLB','XLE']
sp = pd.DataFrame({t: px[t]['Close'] for t in SEC}).dropna()
r = sp.pct_change(21)
sprd = (r.max(axis=1)-r.min(axis=1))*100
print('\nsector 21d max-min spread %.2fpp | PIT rank252 %.1f | all-history pctile %.1f' % (
    sprd.iloc[-1], level_rank(sprd,252).iloc[-1], 100*(sprd.iloc[-1] > sprd.dropna()).mean()))
print('  leader %s %+.2f%%  laggard %s %+.2f%%' % (r.iloc[-1].idxmax(),100*r.iloc[-1].max(),r.iloc[-1].idxmin(),100*r.iloc[-1].min()))

rv21 = spy.pct_change().rolling(21).std()*np.sqrt(252)*100
print('\nSPY 21d realized vol %.2f%% | PIT rank252 %.1f' % (rv21.iloc[-1], level_rank(rv21,252).iloc[-1]))
print('VIX %.2f VIX3M %.2f ratio %.3f | MOVE %.2f rank252 %.1f | SKEW %.2f rank252 %.1f' % (
    vix.iloc[-1], px['^VIX3M']['Close'].iloc[-1], vix.iloc[-1]/px['^VIX3M']['Close'].iloc[-1],
    px['^MOVE']['Close'].iloc[-1], level_rank(px['^MOVE']['Close'],252).iloc[-1],
    px['^SKEW']['Close'].iloc[-1], level_rank(px['^SKEW']['Close'],252).iloc[-1]))

# dispersion: cross-sectional 21d realized vol of the 9 SPDRs vs SPY's
dsp = sp.pct_change()
xs = dsp.std(axis=1)*np.sqrt(252)*100
print('\nsector cross-sectional daily dispersion (ann) %.2f%% rank252 %.1f' % (xs.rolling(21).mean().iloc[-1], level_rank(xs.rolling(21).mean(),252).iloc[-1]))

# TNX/DBC joint again with episode context
p = pd.DataFrame({'tnx':px['^TNX']['Close'],'dbc':dbc}).dropna()
tnx_hi = p['tnx'] >= p['tnx'].rolling(252).max()-1e-9
dbc_hi = p['dbc'] >= p['dbc'].rolling(252).max()-1e-9
j = tnx_hi & dbc_hi
print('\nTNX&DBC both at 252d high: %d days, episodes(gap21): %s' % (int(j.sum()), [str(d.date()) for d in declusters(p.index[j],21,p.index)]))
print('  last 12 joint days:', [str(d.date()) for d in p.index[j][-12:]])
print('  today joint?', bool(j.iloc[-1]))
