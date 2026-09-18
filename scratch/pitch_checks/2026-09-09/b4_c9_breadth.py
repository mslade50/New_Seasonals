"""C9 round 1 -- breadth of 21-day rank floors as an INDEX conditioner.

FIRST question, per the 2026-08-19 registry lesson: is today's count actually
extreme on a POINT-IN-TIME trailing-252 percentile? Built on both universes,
survivorship stated.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change, rolling_on_valid
import numpy as np, pandas as pd

TAPE = json.load(open(Path(__file__).resolve().parents[3] / 'data' / 'pitch_tape.json'))['tickers']
SPDR = ['XLK', 'XLV', 'XLU', 'XLP', 'XLI', 'XLF', 'XLY', 'XLB', 'XLE', 'XLRE', 'XLC']

idxpx = close_panel(['SPY', 'IWM'])
spy, iwm = idxpx['SPY'].dropna(), idxpx['IWM'].dropna()
spy_dd = spy / rolling_on_valid(spy, lambda x: x.rolling(252).max()) - 1.0
print('SPY dd from 252d high today: %+.2f%%' % (100 * spy_dd.iloc[-1]))


def breadth(tickers, thr=2.5, min_hist=273):
    px = close_panel(tickers)
    px = px.reindex(spy.index)
    ranks = {}
    for t in px.columns:
        s = px[t]
        if s.notna().sum() < min_hist:
            continue
        r = pct_rank(s, 21, 252)
        # eligible only once the name has a full window of its own
        ranks[t] = r
    R = pd.DataFrame(ranks)
    elig = R.notna().sum(axis=1)
    cnt = (R <= thr).sum(axis=1)
    frac = (cnt / elig.replace(0, np.nan))
    return cnt, elig, frac


for uname, tickers in [('SPDR sectors (%d)' % len(SPDR), SPDR),
                       ('218-name tape (SURVIVORSHIP-BIASED)', TAPE)]:
    cnt, elig, frac = breadth(tickers)
    pit = rolling_on_valid(frac, lambda x: x.rolling(252).rank(pct=True) * 100.0)
    print('\n===== %s =====' % uname)
    print('today: count=%d of %d eligible = %.2f%%  |  PIT trailing-252 pctile = %.1f'
          % (cnt.iloc[-1], elig.iloc[-1], 100 * frac.iloc[-1], pit.iloc[-1]))
    print('  all-history pctile of today\'s fraction: %.1f'
          % (100 * (frac.iloc[-1] > frac.dropna()).mean()))
    print('  last 8 sessions PIT pctile:', pit.tail(8).round(1).tolist())
    print('  eligible-name count over time: 2006 %s / 2016 %s / today %d'
          % (elig.asof(pd.Timestamp('2006-06-30')), elig.asof(pd.Timestamp('2016-06-30')),
             elig.iloc[-1]))

    # top decile of the PIT percentile, with SPY near its high
    for spy_gate, glbl in [(spy_dd > -0.02, 'SPY within 2% of high'),
                           (pd.Series(True, index=spy.index), 'no SPY gate')]:
        m = (pit >= 90) & spy_gate.reindex(pit.index, fill_value=False)
        d = pit.index[m.fillna(False).values]
        if len(d) == 0:
            print('  [%s] no days' % glbl); continue
        e = declusters(d, 10, spy.index)
        print('  [%-24s] days=%4d episodes(gap10)=%3d' % (glbl, len(d), len(e)))
        for vn, legs in [('SPY', [('SPY', 1.0)]), ('IWM', [('IWM', 1.0)])]:
            row = {'   %s' % vn: ''}
            for h in (1, 2, 3, 5, 10):
                r = horizon_scan(idxpx.dropna(), d, legs, hs=(h,), lag=1, min_gap=10)[0]
                if r['n']:
                    row['h%d' % h] = round(r['mean_pct'], 3)
                    row['h%dedge' % h] = round(r['edge_pct'], 3)
                    row['h%dN' % h] = r['n']
            print('   ', row)
