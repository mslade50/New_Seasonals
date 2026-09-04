"""The run into Jackson Hole.

Tomorrow (Wed 2026-08-26) is 3 td before the Fri 2026-08-28 symposium.
Engine base cell: ^GSPC h1 +0.31% on 17-9, t 1.48. Soft. The tellable cell
is the whole three-session run-in and the symposium session itself, plus
whether entering the window near a high changes anything (2026 enters
1.56% below the 52w high with the S&P's 21d return in its 81st percentile).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
from pitch_lab import (load_prices, load_events, fwd_ret, local_control,
                       summarize, era_split, sign_test, cluster_note, show)

TK = ['^GSPC', 'QQQ', 'IWM', 'TLT', '^VIX', 'GC=F', 'DX-Y.NYB']
px = load_prices(TK)
c = {t: px[t]['Close'].astype(float) for t in TK}
spx = c['^GSPC']

ev = load_events(['jackson_hole'])
print('jackson_hole events:', len(ev))
print(ev.tail(4).to_string())

dates = pd.DatetimeIndex(sorted(spx.index))
pos = pd.Series(range(len(dates)), index=dates)


def anchor_k(ev_dates, k):
    """Session k td BEFORE each event (the anchor convention)."""
    out = []
    for e in ev_dates:
        nxt = dates[dates >= e]
        if len(nxt) == 0:
            continue
        p = pos[nxt[0]] - k
        if p >= 0:
            out.append((dates[p], nxt[0]))
    return out


ev_dates = pd.DatetimeIndex(ev['date'])
pairs = [(a, e) for a, e in anchor_k(ev_dates, 3) if a < dates[-1]]
anch = pd.DatetimeIndex([a for a, _ in pairs])
print('\nanchors (3 td before a symposium): %d, %s .. %s'
      % (len(anch), anch[0].date(), anch[-1].date()))
print('tomorrow is the anchor for 2026-08-28; current anchor would be',
      dates[-1].date())

# --- h1, h2, h3 (h3 = the symposium session close) and h5
for t in ['^GSPC', 'QQQ', 'IWM', '^VIX', 'TLT', 'GC=F', 'DX-Y.NYB']:
    rows = []
    for h in (1, 2, 3, 5):
        f = fwd_ret(c[t], h)
        v = f.reindex(anch).dropna()
        r = summarize(v.values, '%s h%d' % (t, h))
        r['sign_p'] = round(sign_test(int((v.values > 0).sum()), len(v)), 4)
        rows.append(r)
    f3 = fwd_ret(c[t], 3)
    rows.append(summarize(f3.dropna().values, '  CTRL all days h3'))
    rows.append(summarize(f3.reindex(local_control(c[t].index, anch, 126)).dropna().values,
                          '  CTRL local h3'))
    show(rows, '%s: run into the symposium' % t)

# --- the symposium session's OWN move (anchor = day before, h1)
pairs1 = [(a, e) for a, e in anchor_k(ev_dates, 1) if a < dates[-1]]
anch1 = pd.DatetimeIndex([a for a, _ in pairs1])
rows = []
for t in ['^GSPC', 'IWM', 'TLT', 'GC=F']:
    f = fwd_ret(c[t], 1)
    v = f.reindex(anch1).dropna()
    r = summarize(v.values, '%s symposium session' % t)
    r['sign_p'] = round(sign_test(int((v.values > 0).sum()), len(v)), 4)
    rows.append(r)
show(rows, 'the symposium session itself (h1 from the day before)')

# --- the three-session run-in, split by how the index enters
print('\n--- ^GSPC h3 run-in, per year, with the entry state ---')
f3 = fwd_ret(spx, 3)
hi252 = spx.rolling(252).max()
r21 = spx.pct_change(21, fill_method=None)
rank21 = r21.rolling(252).rank(pct=True) * 100
recs = []
for a, e in pairs:
    if a not in f3.index or np.isnan(f3.get(a, np.nan)):
        continue
    dist = 100 * (spx.get(a) / hi252.get(a) - 1)
    recs.append((a, e, 100 * f3.get(a), dist, rank21.get(a),
                 a.year % 4 == 2))
for a, e, r, dist, rk, mid in recs:
    print('  %s -> symposium %s  run-in %+6.2f%%  entered %+5.2f%% from 52w high, 21d rank %5.1f %s'
          % (a.date(), e.date(), r, dist, rk if rk == rk else float('nan'),
             'MIDTERM' if mid else ''))

vals = np.array([r for _, _, r, _, _, _ in recs]) / 100
idx = pd.DatetimeIndex([a for a, _, _, _, _, _ in recs])
print('\nrun-in all years: %s' % summarize(vals, 'h3'))
print('sign p %.4f' % sign_test(int((vals > 0).sum()), len(vals)))
print('concentration: %s' % cluster_note(idx, vals))
print('era: %s' % era_split(idx, vals)[-1])

near = np.array([d > -3 for _, _, _, d, _, _ in recs])
print('\nentered within 3%% of the 52w high (2026 enters -1.56%%): n=%d %s'
      % (near.sum(), summarize(vals[near], 'h3 near-high')))
print('  sign p %.4f' % sign_test(int((vals[near] > 0).sum()), int(near.sum())))
print('entered further below: n=%d %s' % ((~near).sum(), summarize(vals[~near], 'h3 far')))

mid = np.array([m for _, _, _, _, _, m in recs])
print('\nmidterm years: n=%d %s' % (mid.sum(), summarize(vals[mid], 'h3 midterm')))
print('  sign p %.4f' % sign_test(int((vals[mid] > 0).sum()), int(mid.sum())))
print('non-midterm: n=%d %s' % ((~mid).sum(), summarize(vals[~mid], 'h3 non-midterm')))
