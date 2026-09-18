import json
t = json.load(open('data/pitch_tape.json'))
ts = t['tickers']
CLASSES = {
 'us_large': ['SPY','QQQ','^GSPC','^NDX'],
 'us_small': ['IWM'],
 'rates': ['TLT','IEF','^TNX','^IRX','SHY'],
 'credit': ['HYG','LQD'],
 'gold_miners': ['GLD','GDX','GDXJ'],
 'other_metals': ['SLV','COPX','PPLT'],
 'energy': ['USO','UNG','DBC','XLE','XOP'],
 'dollar_fx': ['UUP','DX-Y.NYB','FXE','FXY'],
 'intl': ['EFA','EEM','FXI','EWZ','EWJ','EWG'],
 'vol': ['^VIX','^VIX3M','^VVIX','^MOVE','SVXY','UVXY','VXX'],
 'sectors': ['XLK','XLF','XLV','XLY','XLP','XLI','XLE','XLU','XLB','XLRE','XLC'],
}
print('freshest', t['freshest_bar'], 'asof', t['asof'])
for c, tks in CLASSES.items():
    print('###', c)
    for tk in tks:
        r = ts.get(tk)
        if r is None:
            print(f'  {tk:<10} MISSING')
            continue
        print(f'  {tk:<10} close={r["close"]:>9.2f} r1={r["ret_1d"]:>6.2f} r5={r["ret_5d"]:>7.2f} r21={r["ret_21d"]:>7.2f} r63={r["ret_63d"]:>7.2f} r252={r["ret_252d"]:>7.2f} '
              f'rk5={r["rank_5d"]:>5.1f} rk21={r["rank_21d"]:>5.1f} rk63={r["rank_63d"]:>5.1f} z10={r["z10"]:>5.2f} d52h={r["dist_52w_high_pct"]:>7.2f} d52l={r["dist_52w_low_pct"]:>7.2f} d200={r["dist_sma200_pct"]:>7.2f} atr%={r["atr_pct"]:>5.2f} rv21={r["rvol21_ann"]:>5.1f}')
print()
print('ALL TICKERS:', ' '.join(sorted(ts)))
