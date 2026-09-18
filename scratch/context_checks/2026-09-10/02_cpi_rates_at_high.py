"""CPI conditioned on the rate state it walks into.

Tonight: ^TNX closed AT a 252-day high (+2.21% on the PPI), TLT/IEF/LQD all at 252-day lows.
Anchor = the session 1 td before a CPI (engine convention), h1 = the CPI session itself.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np, pandas as pd
from pitch_lab import load_prices, load_events, summarize, era_split, sign_test, cluster_note, local_control

TK = ['^GSPC','SPY','TLT','^VIX','^TNX','IEF','QQQ','CL=F']
px = load_prices(TK)
ev = load_events()
cpi = pd.DatetimeIndex(sorted(pd.to_datetime(ev.loc[ev['event'] == 'cpi', 'date'].unique())))

tnx = px['^TNX']['Close'].dropna()
dates = tnx.index

# anchor = session strictly before each CPI print
anch = []
for d in cpi:
    prior = dates[dates < d]
    if len(prior) == 0: continue
    a = prior[-1]
    # only keep if the CPI date itself is a trading session in the panel
    if d in dates:
        anch.append((a, d))
anch = [(a, d) for a, d in anch if a >= pd.Timestamp('1999-01-01')]
print(f"CPI prints with a tradeable anchor+print pair: {len(anch)}")

# state at the anchor: 10y yield vs its own trailing-252 max
roll_max = tnx.rolling(252).max()
roll_min = tnx.rolling(252).min()

def cell(name, mask_fn, subj, label):
    rows = []
    for a, d in anch:
        if a not in tnx.index or pd.isna(roll_max.loc[a]): continue
        if not mask_fn(a): continue
        s = px[subj]['Close'].dropna()
        if a not in s.index or d not in s.index: continue
        i0, i1 = s.index.get_loc(a), s.index.get_loc(d)
        rows.append((a, d, (s.iloc[i1] / s.iloc[i0] - 1) * 100))
    if not rows:
        print(f"  {name:52} {subj:7} n=0"); return None
    dd = pd.DatetimeIndex([r[0] for r in rows]); v = np.array([r[2] for r in rows])
    w = int((v > 0).sum()); n = len(v)
    print(f"  {name:52} {subj:7} n={n:4} mean={v.mean():+7.3f}% hit={100*w/n:5.1f}% "
          f"t={v.mean()/(v.std(ddof=1)/np.sqrt(n)):+6.2f} rec={w}-{n-w} signp={sign_test(w,n):.4f}")
    return dd, v

print("\n=== BASELINE: every CPI anchor, h1 = the print session ===")
for s in ['^GSPC','TLT','^VIX','^TNX','QQQ','CL=F']:
    cell("all CPI prints", lambda a: True, s, "")

print("\n=== 10-year yield AT its own 252d high on the anchor (tonight: exactly at it) ===")
at_high = lambda a: tnx.loc[a] >= roll_max.loc[a] - 1e-12
for s in ['^GSPC','TLT','^VIX','^TNX','QQQ','CL=F']:
    cell("anchor closed at a 252d high in the 10y", at_high, s, "")

print("\n=== 10-year within 1% of its 252d high ===")
near_high = lambda a: tnx.loc[a] >= 0.99 * roll_max.loc[a]
for s in ['^GSPC','TLT','^VIX','^TNX','QQQ','CL=F']:
    cell("anchor within 1% of a 252d high in the 10y", near_high, s, "")

print("\n=== control: 10y in the BOTTOM half of its 252d range ===")
lowhalf = lambda a: tnx.loc[a] <= (roll_min.loc[a] + roll_max.loc[a]) / 2
for s in ['^GSPC','TLT','^VIX']:
    cell("anchor with the 10y in the lower half of its range", lowhalf, s, "")
