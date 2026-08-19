"""Sort the whole tape several ways for the B1 surface map. No stats, just the picture."""
import json
from pathlib import Path

t = json.load(open('data/pitch_tape.json'))
tk = t['tickers']
print(f"asof {t['asof']} freshest {t['freshest_bar']} n={len(tk)}")

CLASSES = {
    'us_large': ['SPY', 'QQQ', '^GSPC', '^NDX', 'DIA'],
    'us_small': ['IWM', 'MDY'],
    'rates': ['TLT', 'IEF', '^TNX', 'SHY', 'TIP'],
    'credit': ['HYG', 'LQD'],
    'gold_miners': ['GLD', 'GDX', 'GDXJ'],
    'metals': ['SLV', 'PPLT', 'COPX', 'CPER'],
    'energy': ['USO', 'UNG', 'DBC', 'XLE', 'XOP', 'OIH'],
    'dollar_fx': ['UUP', 'DX-Y.NYB', 'FXE', 'FXY', 'FXB'],
    'intl': ['EFA', 'EEM', 'FXI', 'EWZ', 'EWJ', 'EWG', 'INDA', 'EWW'],
    'vol': ['^VIX', '^VIX3M', '^MOVE', 'SVXY', '^SKEW', 'VXX'],
    'sectors': ['XLK','XLF','XLV','XLY','XLP','XLI','XLE','XLU','XLB','XLRE','XLC'],
}

def row(s, d):
    return (f"{s:10s} 1d{d['ret_1d']:7.2f} 5d{d['ret_5d']:7.2f} 21d{d['ret_21d']:7.2f} "
            f"63d{d['ret_63d']:8.2f} 252d{d['ret_252d']:8.2f} | r5 {d['rank_5d']:5.1f} r21 {d['rank_21d']:5.1f} "
            f"r63 {d['rank_63d']:5.1f} | z10{d['z10']:6.2f} atr%{d['atr_pct']:5.2f} "
            f"| 52wh{d['dist_52w_high_pct']:7.2f} 52wl{d['dist_52w_low_pct']:8.2f} 200d{d['dist_sma200_pct']:7.2f}")

print("\n===== BY CLASS =====")
for cls, names in CLASSES.items():
    print(f"\n-- {cls}")
    for s in names:
        if s in tk:
            print(' ', row(s, tk[s]))
        else:
            print(f"  {s:10s} NOT IN TAPE")

def top(key, n=14, rev=True, label=''):
    items = sorted(tk.items(), key=lambda kv: (kv[1][key] is None, kv[1][key] if kv[1][key] is not None else 0), reverse=rev)[:n]
    print(f"\n-- {label or key} ({'high' if rev else 'low'})")
    for s, d in items:
        print(' ', row(s, d))

print("\n===== EXTREMES (whole tape) =====")
top('dist_52w_high_pct', rev=True, label='closest to 52w high')
top('dist_52w_low_pct', rev=False, label='closest to 52w low')
top('dist_sma200_pct', rev=True, label='most extended above 200d')
top('dist_sma200_pct', rev=False, label='most below 200d')
top('z10', rev=True, label='highest z10')
top('z10', rev=False, label='lowest z10')
top('rank_5d', rev=True, label='rank5 high')
top('rank_5d', rev=False, label='rank5 low')
top('rank_21d', rev=True, label='rank21 high')
top('rank_21d', rev=False, label='rank21 low')
top('rank_63d', rev=True, label='rank63 high')
top('rank_63d', rev=False, label='rank63 low')
top('ret_1d', rev=True, label='biggest 1d up')
top('ret_1d', rev=False, label='biggest 1d down')
top('vol_vs_63d', rev=True, label='volume vs 63d')
