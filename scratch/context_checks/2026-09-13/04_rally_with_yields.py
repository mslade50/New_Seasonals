"""Friday's CPI session: ^GSPC +0.86%, ^VIX -11.2%, ^FVX +5.8bp to a 252-day high, ^TNX +3.1bp to a 252-day high.
Split S&P rallies of +0.75% or better by what the 5-year did the same session.
Forward: ^GSPC h1/h5/h21, ^VIX h5. Declustered 5td within each bucket.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import load_prices, load_events, summarize, era_split, sign_test, cluster_note, declusters

px = load_prices(['SPY', '^GSPC', '^FVX', '^TNX', '^VIX', '^MOVE'])
nyse = px['SPY']['Close'].dropna().index
nyse = nyse[nyse >= '1999-01-01']
d = nyse
spx = px['^GSPC']['Close'].reindex(d)
vix = px['^VIX']['Close'].reindex(d)
fvx = px['^FVX']['Close'].reindex(d).ffill(limit=2)
tnx = px['^TNX']['Close'].reindex(d).ffill(limit=2)
move = px['^MOVE']['Close'].reindex(d)
pos = pd.Series(range(len(d)), index=d)

r1 = spx.pct_change()
fbp = 100 * (fvx - fvx.shift(1))
vr1 = vix.pct_change()
tnx_hi = tnx >= tnx.rolling(252, min_periods=200).max() - 1e-9
print("tonight:", d[-1].date(), f"spx {100*r1.iloc[-1]:+.2f}% fvx {fbp.iloc[-1]:+.1f}bp vix {100*vr1.iloc[-1]:+.1f}% tnx_hi {bool(tnx_hi.iloc[-1])}")

ev = load_events(['cpi'])
cpi = set(pd.to_datetime(ev['date']))


def fwd(s, a, h):
    return np.array([s.iloc[pos[x] + h] / s[x] - 1 for x in a if pos[x] + h < len(d)])


def report(label, a, dc=5):
    a = pd.DatetimeIndex([x for x in a if pos[x] + 21 < len(d)])
    a = declusters(a, dc, d) if dc else a
    print(f"\n--- {label}: n={len(a)}")
    out = {}
    for h in (1, 5, 21):
        v = fwd(spx, a, h)
        st = summarize(v)
        w = int((v > 0).sum())
        out[h] = (a, v)
        print(f"  ^GSPC h{h:2}: mean {st['mean_pct']:+.3f}% med {st['median_pct']:+.3f}% up {w}/{len(v)} ({st['hit']:.1f}%) t {st['t']:+.2f}")
    v = fwd(vix, a, 5)
    print(f"  ^VIX  h 5: mean {100*v.mean():+.2f}% med {100*np.median(v):+.2f}% up {int((v>0).sum())}/{len(v)}")
    return out


rally = d[(r1 >= 0.0075).values]
report("all ^GSPC +0.75% sessions (control)", rally)
up = report("rally with 5y +5bp or more", rally[(fbp.reindex(rally) >= 5).values])
dn = report("rally with 5y -5bp or more lower", rally[(fbp.reindex(rally) <= -5).values])
report("rally, 5y +5bp, VIX -8% or more", rally[((fbp.reindex(rally) >= 5) & (vr1.reindex(rally) <= -0.08)).values])
tn = report("rally with 10y closing at a 252-day high", rally[tnx_hi.reindex(rally).fillna(False).values])
a, v = tn[21]
print("  h21 era:", [(e['label'], e['n'], round(e['mean_pct'], 2), round(e['hit'], 1)) for e in era_split(a, v)])
print("  h21 concentration:", cluster_note(a, v))
print("  episodes:", [str(x.date()) for x in a])
cp = report("CPI-day rally with 5y +5bp or more", rally[((fbp.reindex(rally) >= 5) & pd.Series([x in cpi for x in rally], index=rally)).values], dc=0)
a, v = cp[5]
print("  episodes h5:", [(str(x.date()), round(100 * vv, 2)) for x, vv in zip(a, v)])

# 5y up vs 5y down rallies, h21 difference
a1, v1 = up[21]
a2, v2 = dn[21]
diff = 100 * (v1.mean() - v2.mean())
se = 100 * np.sqrt(v1.var(ddof=1) / len(v1) + v2.var(ddof=1) / len(v2))
print(f"\nh21: yields-up rallies minus yields-down rallies = {diff:+.3f}pp se {se:.3f} t {diff/se:+.2f}")
print("yields-up h21 era:", [(e['label'], e['n'], round(e['mean_pct'], 2), round(e['hit'], 1)) for e in era_split(a1, v1)])
print("yields-down h21 era:", [(e['label'], e['n'], round(e['mean_pct'], 2), round(e['hit'], 1)) for e in era_split(a2, v2)])

# VIX collapse while MOVE did not fall
print("\n=== VIX -10% session with MOVE flat or higher ===")
m = (vr1 <= -0.10) & (move.pct_change() >= 0)
sel = declusters(d[m.fillna(False).values], 5, d)
sel = pd.DatetimeIndex([x for x in sel if x >= pd.Timestamp('2003-01-01') and pos[x] + 5 < len(d)])
v = fwd(vix, sel, 5)
print(f"  n={len(v)} VIX h5 mean {100*v.mean():+.2f}% up {int((v>0).sum())}/{len(v)}")
m2 = (vr1 <= -0.10) & (move.pct_change() < 0)
sel2 = declusters(d[m2.fillna(False).values], 5, d)
sel2 = pd.DatetimeIndex([x for x in sel2 if x >= pd.Timestamp('2003-01-01') and pos[x] + 5 < len(d)])
v2 = fwd(vix, sel2, 5)
print(f"  control MOVE lower: n={len(v2)} VIX h5 mean {100*v2.mean():+.2f}% up {int((v2>0).sum())}/{len(v2)}")
