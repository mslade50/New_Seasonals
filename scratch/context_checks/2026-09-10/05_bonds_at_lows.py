"""TLT, IEF and LQD all closed AT a 252-day low on the same session, with ^TNX at a 252-day high.
Two questions: what follows the state, and what does a CPI print do when it walks into it.
09-01 published IEF+LQD at lows with SPY near its high; this adds TLT, the new-low framing and the CPI leg.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np, pandas as pd
from pitch_lab import load_prices, load_events, summarize, era_split, sign_test, cluster_note, declusters, local_control

px = load_prices(['TLT','IEF','LQD','^TNX','SPY','^GSPC','HYG','^VIX'])
ev = load_events()
cpi = pd.DatetimeIndex(sorted(pd.to_datetime(ev.loc[ev['event']=='cpi','date'].unique())))

cl = {t: px[t]['Close'].dropna() for t in px}
d = cl['TLT'].index.intersection(cl['IEF'].index).intersection(cl['LQD'].index).intersection(cl['^TNX'].index)
d = d[d >= pd.Timestamp('2003-01-01')]   # LQD starts 2002, TLT 2002

def at_low(s, k=252, tol=0.0):
    r = s.rolling(k).min()
    return (s <= r * (1 + tol))

low3 = (at_low(cl['TLT']) & at_low(cl['IEF']) & at_low(cl['LQD'])).reindex(d).fillna(False)
tnx_hi = (cl['^TNX'] >= cl['^TNX'].rolling(252).max()).reindex(d).fillna(False)

print(f"sessions 2003+: {len(d)}")
print(f"TLT+IEF+LQD all at a 252d low, same session: {int(low3.sum())}")
print(f"  ... and ^TNX also at a 252d high:          {int((low3 & tnx_hi).sum())}")

def fwd(sel, sym, h):
    s = cl[sym]; out = []
    for a in sel:
        if a not in s.index: continue
        i = s.index.get_loc(a)
        if i + h < len(s): out.append((a, s.iloc[i+h] / s.iloc[i] - 1))
    return pd.DatetimeIndex([o[0] for o in out]), np.array([o[1] for o in out])

def show(lab, sel, sym, h):
    dd, v = fwd(sel, sym, h)
    if len(v) < 4: print(f"  {lab:44} {sym:5} h{h:<2} n={len(v)}"); return None
    st = summarize(v); w = int((v > 0).sum())
    print(f"  {lab:44} {sym:5} h{h:<2} n={len(v):4} mean={st['mean_pct']:+7.3f}% up={w:3}/{len(v):3} "
          f"t={st['t']:+6.2f} signp={sign_test(w,len(v)):.4f}")
    return dd, v

trig = d[low3]
trig_hi = d[low3 & tnx_hi]
dec = declusters(trig_hi, 5, d)
print(f"  declustered at 5 td: {len(dec)} episodes\n")

print("=== forward from the all-three-at-a-low session ===")
for h in (1, 5, 21):
    for sym in ('TLT','IEF','SPY','^VIX'):
        show("all three at a 252d low", trig, sym, h)
    print()

print("=== and with ^TNX at a 252d high the same session (tonight's exact shape) ===")
for h in (1, 5, 21):
    for sym in ('TLT','IEF','SPY'):
        show("all three low + 10y at a 252d high", dec, sym, h)
    print()

out = show("all three low + 10y high", dec, 'TLT', 5)
if out:
    dd, v = out
    print("  episodes:", ", ".join(f"{x.date()} {100*y:+.1f}%" for x, y in zip(dd, v)))
    print("  era:", [(e['label'], e['n'], round(e['mean_pct'],3)) for e in era_split(dd, v)])
    print("  concentration:", cluster_note(dd, v))

print("\n=== CONTROL: all sessions 2003+ ===")
for sym in ('TLT','IEF','SPY'):
    for h in (1, 5, 21):
        dd, v = fwd(d, sym, h)
        print(f"  baseline {sym:5} h{h:<2} n={len(v):5} mean={summarize(v)['mean_pct']:+7.3f}% "
              f"up={100*(v>0).mean():5.1f}%")

print("\n=== local +/-126td control for the all-three-low state, TLT h5 ===")
lc = local_control(d, trig, 126)
dd, v = fwd(lc, 'TLT', 5)
print(f"  local control n={len(v)} mean={summarize(v)['mean_pct']:+.3f}% up={100*(v>0).mean():.1f}%")

print("\n=== CPI leg: eve enters with TLT at a 252d low ===")
pairs = [(d[d < c][-1], c) for c in cpi if c in d and len(d[d < c])]
tlt_low = at_low(cl['TLT'])
for sym in ('TLT','^GSPC','^VIX'):
    s = cl[sym]
    rows = [(a, s.loc[c]/s.loc[a]-1) for a, c in pairs
            if a in tlt_low.index and tlt_low.loc[a] and a in s.index and c in s.index]
    if len(rows) < 4: print(f"  {sym} n={len(rows)}"); continue
    dd = pd.DatetimeIndex([r[0] for r in rows]); v = np.array([r[1] for r in rows])
    w = int((v > 0).sum())
    print(f"  CPI eve with TLT at a 252d low  {sym:6} n={len(v):3} mean={summarize(v)['mean_pct']:+7.3f}% "
          f"up={w}/{len(v)} signp={sign_test(w,len(v)):.4f}  years={sorted(set(dd.year))}")
