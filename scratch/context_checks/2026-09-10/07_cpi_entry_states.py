"""Two more CPI-eve conditionings tonight actually satisfies:
  (a) crude closed +8.20%, an inflation input spiking into the print
  (b) the S&P closed lower and the 10-year yield closed higher on the eve
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np, pandas as pd
from pitch_lab import load_prices, load_events, summarize, era_split, sign_test, cluster_note

px = load_prices(['CL=F','^GSPC','SPY','QQQ','^VIX','^TNX','TLT'])
ev = load_events()
cpi = pd.DatetimeIndex(sorted(pd.to_datetime(ev.loc[ev['event']=='cpi','date'].unique())))
cl = {t: px[t]['Close'].dropna() for t in px}
d = cl['^GSPC'].index
pairs = [(d[d < c][-1], c) for c in cpi if c in d and len(d[d < c])]
pairs = [(a, c) for a, c in pairs if a >= pd.Timestamp('1999-01-01')]
allA = pd.DatetimeIndex([a for a, _ in pairs])

def leg(sel, sym):
    s = cl[sym]
    rows = [(a, s.loc[c]/s.loc[a]-1) for a, c in pairs if a in sel and a in s.index and c in s.index]
    return pd.DatetimeIndex([r[0] for r in rows]), np.array([r[1] for r in rows])

def show(lab, sel, sym, verbose=False):
    dd, v = leg(sel, sym)
    if len(v) < 4: print(f"  {lab:48} {sym:6} n={len(v)}"); return None
    st = summarize(v); w = int((v > 0).sum())
    print(f"  {lab:48} {sym:6} n={len(v):3} mean={st['mean_pct']:+7.3f}% up={w:3}/{len(v):3} "
          f"t={st['t']:+6.2f} signp={sign_test(w,len(v)):.4f}")
    if verbose:
        print("      episodes:", ", ".join(f"{x.date()} {100*y:+.1f}%" for x, y in zip(dd, v)))
        print("      era:", [(e['label'], e['n'], round(e['mean_pct'],3), round(e['hit'],1)) for e in era_split(dd, v)])
        print("      concentration:", cluster_note(dd, v))
    return dd, v

# ---- (a) crude spiking into the print -----------------------------------
crude_r = cl['CL=F'].pct_change()
print("=== CPI eves by the eve's own crude move ===")
for lo, lab in [(0.03, 'crude eve +3% or more'), (0.05, 'crude eve +5% or more'), (0.08, 'crude eve +8% or more')]:
    sel = set(allA[(crude_r.reindex(allA) >= lo).fillna(False)])
    print(f"  [{lab}] n_eves={len(sel)}")
    for sym in ('^GSPC','^VIX','CL=F','TLT'):
        show(lab, sel, sym)
    print()

sel5 = set(allA[(crude_r.reindex(allA) >= 0.05).fillna(False)])
show('crude eve +5% or more', sel5, '^GSPC', verbose=True)
print()
show('crude eve +5% or more', sel5, 'CL=F', verbose=True)

# ---- (b) the eve's own risk shape ---------------------------------------
print("\n=== CPI eves where the S&P fell AND the 10-year yield rose (tonight) ===")
spx_r = cl['^GSPC'].pct_change(); tnx_r = cl['^TNX'].pct_change()
sel = set(allA[((spx_r.reindex(allA) < 0) & (tnx_r.reindex(allA) > 0)).fillna(False)])
print(f"  n_eves={len(sel)} of {len(allA)}")
for sym in ('^GSPC','QQQ','^VIX','TLT','^TNX'):
    show('eve: S&P down, 10-year yield up', sel, sym)

print("\n  tighten: S&P down AND 10y yield up 2% or more (tonight +2.21%)")
sel2 = set(allA[((spx_r.reindex(allA) < 0) & (tnx_r.reindex(allA) >= 0.02)).fillna(False)])
print(f"  n_eves={len(sel2)}")
for sym in ('^GSPC','^VIX','TLT','^TNX'):
    show('eve: S&P down, 10y yield up 2%+', sel2, sym, verbose=(sym in ('^GSPC','^TNX')))

print("\n=== control: all CPI eves ===")
for sym in ('^GSPC','QQQ','^VIX','TLT','^TNX','CL=F'):
    show('all CPI eves', set(allA), sym)
