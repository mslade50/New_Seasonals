"""Close out the three cells that still need era, concentration and date evidence."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np, pandas as pd
from pitch_lab import load_prices, load_events, summarize, era_split, sign_test, cluster_note, declusters

px = load_prices(['^VVIX','^GSPC','TLT','IEF','LQD','^TNX','^VIX','QQQ','SPY'])
cl = {t: px[t]['Close'].dropna() for t in px}

# ---------- (1) VVIX self-reversion, declustered ----------
vv = cl['^VVIX']; d = vv.index
rank5 = vv.pct_change(5).rolling(252).rank(pct=True) * 100
trig = d[(rank5 >= 95).reindex(d).fillna(False)]
dec = declusters(trig[trig >= pd.Timestamp('2007-01-01')], 5, d)
rows = []
for a in dec:
    i = d.get_loc(a)
    if i + 5 < len(d): rows.append((a, vv.iloc[i+5]/vv.iloc[i]-1))
dd = pd.DatetimeIndex([r[0] for r in rows]); v = np.array([r[1] for r in rows])
st = summarize(v); w = int((v > 0).sum())
print(f"(1) ^VVIX h5 after a 5d-rank>=95 episode: n={len(v)} mean={st['mean_pct']:+.3f}% "
      f"median={st['median_pct']:+.3f}% up={w}/{len(v)} t={st['t']:+.2f} signp(down)={sign_test(len(v)-w,len(v)):.4f}")
print("    era:", [(e['label'], e['n'], round(e['mean_pct'],2), round(e['hit'],1)) for e in era_split(dd, v)])
print("    concentration:", cluster_note(dd, v))
base = []
for i in range(len(d)-5):
    if d[i] >= pd.Timestamp('2007-01-01'): base.append(vv.iloc[i+5]/vv.iloc[i]-1)
base = np.array(base)
print(f"    control, every session 2007+: n={len(base)} mean={summarize(base)['mean_pct']:+.3f}% up={100*(base>0).mean():.1f}%")

# ---------- (2) TLT+IEF+LQD all at a 252d low ----------
print()
di = cl['TLT'].index.intersection(cl['IEF'].index).intersection(cl['LQD'].index)
di = di[di >= pd.Timestamp('2003-01-01')]
low = lambda s: (s <= s.rolling(252).min()).reindex(di).fillna(False)
all3 = low(cl['TLT']) & low(cl['IEF']) & low(cl['LQD'])
hits = di[all3]
print(f"(2) TLT+IEF+LQD all at a 252d low, same session, 2003+: n={len(hits)}")
print("    dates:", ", ".join(str(x.date()) for x in hits))
tnxhi = (cl['^TNX'] >= cl['^TNX'].rolling(252).max()).reindex(hits).fillna(False)
print(f"    of those, ^TNX also at a 252d high: {int(tnxhi.sum())}/{len(hits)}")
prior = hits[hits < pd.Timestamp('2026-09-10')]
print(f"    most recent before tonight: {prior[-1].date() if len(prior) else 'none'}"
      f"   ({len(di[(di > prior[-1]) & (di <= pd.Timestamp('2026-09-10'))])} sessions ago)")
print(f"    years: {sorted(set(hits.year))}")

# ---------- (3) CPI eve with the 10y at a 252d high ----------
print()
ev = load_events()
cpi = pd.DatetimeIndex(sorted(pd.to_datetime(ev.loc[ev['event']=='cpi','date'].unique())))
tnx = cl['^TNX']; dt = tnx.index
pairs = [(dt[dt < c][-1], c) for c in cpi if c in dt and len(dt[dt < c])]
pairs = [(a, c) for a, c in pairs if a >= pd.Timestamp('1999-01-01')]
near = tnx >= 0.99 * tnx.rolling(252).max()
sel = [(a, c) for a, c in pairs if near.get(a, False)]
print(f"(3) CPI eves with the 10y within 1% of a 252d high: n={len(sel)}")
for sym in ('^TNX','TLT','^GSPC','^VIX'):
    s = cl[sym]
    rr = [(a, s.loc[c]/s.loc[a]-1) for a, c in sel if a in s.index and c in s.index]
    dd2 = pd.DatetimeIndex([r[0] for r in rr]); vv2 = np.array([r[1] for r in rr])
    w = int((vv2 > 0).sum())
    print(f"    {sym:6} n={len(vv2):3} mean={summarize(vv2)['mean_pct']:+7.3f}% up={w}/{len(vv2)} "
          f"t={summarize(vv2)['t']:+5.2f} signp(down)={sign_test(len(vv2)-w,len(vv2)):.4f}")
    if sym == '^TNX':
        print("      episodes:", ", ".join(f"{x.date()} {100*y:+.1f}%" for x, y in zip(dd2, vv2)))
        print("      era:", [(e['label'], e['n'], round(e['mean_pct'],2)) for e in era_split(dd2, vv2)])
        print("      concentration:", cluster_note(dd2, vv2))
# overlap with the VIX-up conditioning
vixup = set(a for a, _ in pairs if cl['^VIX'].pct_change().get(a, 0) > 0)
print(f"    overlap with the VIX-up eve set: {len([a for a,_ in sel if a in vixup])}/{len(sel)}")
# strictly AT the high, tonight's shape
atk = [(a, c) for a, c in pairs if tnx.get(a, np.nan) >= tnx.rolling(252).max().get(a, np.inf) - 1e-12]
print(f"    strictly AT a 252d high: n={len(atk)} -> {[str(a.date()) for a,_ in atk]}")
