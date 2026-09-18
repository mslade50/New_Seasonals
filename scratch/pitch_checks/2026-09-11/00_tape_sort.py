import json
from pathlib import Path

t = json.load(open('data/pitch_tape.json'))
ts = t['tickers']
rows = [dict(tk=k, **v) for k, v in ts.items()]

def show(key, n=14, rev=True, label=''):
    rs = [r for r in rows if r.get(key) is not None]
    rs.sort(key=lambda r: r[key], reverse=rev)
    print(f'--- {label or key} {"top" if rev else "bottom"} ---')
    for r in rs[:n]:
        print(f'{r["tk"]:<10} {key}={r[key]:>8.2f}  r5={r["ret_5d"]:>6.2f} r21={r["ret_21d"]:>7.2f} r63={r["ret_63d"]:>7.2f} '
              f'rk5={r["rank_5d"]:>5.1f} rk21={r["rank_21d"]:>5.1f} rk63={r["rank_63d"]:>5.1f} z10={r["z10"]:>5.2f} '
              f'd52h={r["dist_52w_high_pct"]:>7.2f} d200={r["dist_sma200_pct"]:>7.2f} atr%={r["atr_pct"]:>5.2f}')
    print()

for key, rev in [('ret_5d', True), ('ret_5d', False), ('ret_21d', True), ('ret_21d', False),
                 ('ret_63d', True), ('ret_63d', False), ('z10', True), ('z10', False),
                 ('dist_52w_high_pct', True), ('dist_52w_high_pct', False),
                 ('dist_sma200_pct', True), ('dist_sma200_pct', False),
                 ('rank_63d', True), ('rank_63d', False), ('vol_vs_63d', True)]:
    show(key, rev=rev)
