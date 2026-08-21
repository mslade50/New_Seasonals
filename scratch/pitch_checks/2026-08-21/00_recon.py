"""Recon for the 2026-08-21 surface map: pin today's exact states."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

ASOF = pd.Timestamp('2026-08-20')  # freshest bar

SEMIS = ['NVDA','AVGO','AMD','INTC','MU','TXN','QCOM','AMAT','LRCX','KLAC','ADI','NXPI','MRVL','ON','SWKS','MCHP','TER','GLW']
RETAIL = ['WMT','TGT','COST','HD','LOW','TJX','ROST','KR','DG','DLTR','BBY','M','JWN','GPS','ORLY','AZO','XRT']
BANKS = ['JPM','BAC','C','WFC','GS','MS','USB','PNC','TFC','SCHW','BK','STT','KRE','XLF']

def breadth(names, label):
    px = close_panel(names)
    px = px.loc[:ASOF]
    r5 = px.pct_change(5)
    r63 = px.pct_change(63)
    rk5 = pd.DataFrame({c: pct_rank(px[c], 5) for c in px.columns})
    rk63 = pd.DataFrame({c: pct_rank(px[c], 63) for c in px.columns})
    row5 = rk5.loc[ASOF].dropna()
    row63 = rk63.loc[ASOF].dropna()
    frac_wash = float((row5 <= 20).mean())
    print(f"\n### {label}: n={len(row5)}  frac 5d-rank<=20 = {frac_wash:.1%}  median 63d rank = {row63.median():.1f}")
    print('  5d ranks:', ' '.join(f"{c}:{row5[c]:.0f}" for c in sorted(row5.index, key=lambda c: row5[c])))
    print('  63d ranks:', ' '.join(f"{c}:{row63[c]:.0f}" for c in sorted(row63.index, key=lambda c: row63[c])))

for names, label in [(SEMIS,'SEMIS'),(RETAIL,'RETAIL'),(BANKS,'BANKS')]:
    breadth(names, label)

# key single-name / index states
px = close_panel(['SPY','QQQ','IWM','^SKEW','^VIX','^VIX3M','HYG','LQD','IEF','TLT','GLD','GDX','SLV','^TNX','UUP','DX-Y.NYB','XLE','USO','EFA','EEM','EWJ','FXI','SVXY','^MOVE','XLV','XLP','SMH','XRT'])
px = px.loc[:ASOF]
print('\n### today state table')
for c in px.columns:
    s = px[c].dropna()
    if s.empty or s.index[-1] != ASOF: 
        print(f"{c:10s} STALE last={s.index[-1].date() if len(s) else 'NA'}")
        continue
    hi = s.rolling(252).max().iloc[-1]; lo = s.rolling(252).min().iloc[-1]
    print(f"{c:10s} px={s.iloc[-1]:9.2f} r5d={s.pct_change(5).iloc[-1]*100:7.2f}% rk5={pct_rank(s,5).iloc[-1]:5.1f} rk21={pct_rank(s,21).iloc[-1]:5.1f} rk63={pct_rank(s,63).iloc[-1]:5.1f} d52h={(s.iloc[-1]/hi-1)*100:7.2f}% d52l={(s.iloc[-1]/lo-1)*100:7.2f}%")

# trading day of month
all_d = px.index
aug = all_d[(all_d.year==2026)&(all_d.month==8)]
print('\nAug 2026 sessions so far:', len(aug), 'last', aug[-1].date())
