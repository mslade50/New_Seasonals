"""Two loose ends before writing:
  (a) the carried ^VIX calendar fault (phantom bars on 2026 market closures) -- does it touch
      tonight's streak count or the CPI-eve pairs?
  (b) era + concentration for the QQQ risk-off CPI-eve cell.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np, pandas as pd
from pitch_lab import load_prices, load_events, summarize, era_split, sign_test, cluster_note

px = load_prices(['^VIX','SPY','^GSPC','QQQ','^TNX'])
cl = {t: px[t]['Close'].dropna() for t in px}
vix, spy = cl['^VIX'], cl['SPY']
extra = vix.index.difference(spy.index)
extra = extra[extra >= pd.Timestamp('2026-01-01')]
print("(a) ^VIX bars in 2026 with no SPY bar:", [str(x.date()) for x in extra])

tail = vix.loc['2026-08-28':]
print("\n    recent ^VIX closes (SPY bar present?):")
for dt, v in tail.items():
    print(f"      {dt.date()}  {v:7.2f}  {'SPY' if dt in spy.index else 'NO SPY BAR'}")

# streak counted on the real trading calendar only
v2 = vix.reindex(spy.index).dropna()
r = v2.pct_change()
run = 0
for u in (r > 0).values:
    run = run + 1 if u else 0
print(f"\n    consecutive ^VIX up closes on the SPY calendar, through {v2.index[-1].date()}: {run}")
run_raw = 0
for u in (vix.pct_change() > 0).values:
    run_raw = run_raw + 1 if u else 0
print(f"    same count on the raw ^VIX index: {run_raw}")

# does the fault touch any CPI eve pair?
ev = load_events()
cpi = pd.DatetimeIndex(sorted(pd.to_datetime(ev.loc[ev['event']=='cpi','date'].unique())))
d = cl['^GSPC'].index
pairs = [(d[d < c][-1], c) for c in cpi if c in d and len(d[d < c])]
bad = [(a, c) for a, c in pairs if a in extra or c in extra]
print(f"    CPI eve/print pairs touching a phantom bar: {len(bad)}  (pairs are built on the ^GSPC calendar)")

# ---------- (b) QQQ risk-off eve cell ----------
pairs = [(a, c) for a, c in pairs if a >= pd.Timestamp('1999-01-01')]
A = pd.DatetimeIndex([a for a, _ in pairs])
sel = set(A[((cl['^GSPC'].pct_change().reindex(A) < 0) & (cl['^TNX'].pct_change().reindex(A) > 0)).fillna(False)])
print("\n(b) CPI eves with the S&P down and the 10-year yield up:")
for sym in ('QQQ','^GSPC','SPY'):
    s = cl[sym]
    rr = [(a, s.loc[c]/s.loc[a]-1) for a, c in pairs if a in sel and a in s.index and c in s.index]
    dd = pd.DatetimeIndex([x[0] for x in rr]); v = np.array([x[1] for x in rr])
    w = int((v > 0).sum()); st = summarize(v)
    print(f"    {sym:6} n={len(v):3} mean={st['mean_pct']:+7.3f}% median={st['median_pct']:+7.3f}% "
          f"up={w}/{len(v)} t={st['t']:+5.2f} signp={sign_test(w,len(v)):.4f}")
    if sym == 'QQQ':
        print("      era:", [(e['label'], e['n'], round(e['mean_pct'],3), round(e['hit'],1)) for e in era_split(dd, v)])
        print("      concentration:", cluster_note(dd, v))
        print("      Friday prints inside the cell:",
              int((pd.DatetimeIndex([c for a, c in pairs if a in sel]).weekday == 4).sum()))
        # control: QQQ next session after ANY down-S&P / up-yield session, no CPI
        m = (cl['^GSPC'].pct_change() < 0) & (cl['^TNX'].pct_change() > 0)
        m = m.reindex(cl['QQQ'].index).fillna(False)
        q = cl['QQQ']; base = []
        for i in range(len(q)-1):
            if m.iloc[i] and q.index[i] not in sel and q.index[i] >= pd.Timestamp('1999-01-01'):
                base.append(q.iloc[i+1]/q.iloc[i]-1)
        base = np.array(base)
        print(f"      control, same state on a NON-CPI eve: n={len(base)} "
              f"mean={summarize(base)['mean_pct']:+.3f}% up={100*(base>0).mean():.1f}%")
