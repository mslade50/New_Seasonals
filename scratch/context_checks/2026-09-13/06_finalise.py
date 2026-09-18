"""Final numbers for the brief plus the kill checks named in the footnote.

1. FOMC-week Monday VIX: sign test against the ordinary-Monday base rate, concentration, tonight's
   intersection with a -10% Friday.
2. Yield surge into the decision: the rank>=90 anchors vs the other anchors, with tests.
3. Decision on VIX expiry: is it the expiry or the quarterly (SEP) meeting?
4. Kills: NG=F September Mondays vs all Mondays; CL=F k3 FOMC concentration.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import load_prices, load_events, summarize, era_split, sign_test, cluster_note

px = load_prices(['SPY', '^GSPC', '^VIX', '^FVX', '^TNX', 'IEF', 'TLT', 'NG=F', 'CL=F'])
nyse = px['SPY']['Close'].dropna().index
d = nyse[nyse >= '1999-01-01']
pos = pd.Series(range(len(d)), index=d)
C = {t: px[t]['Close'].reindex(d) for t in px}
for t in ['^FVX', '^TNX']:
    C[t] = C[t].ffill(limit=2)
vix = C['^VIX']
ev = load_events()
fomc = pd.DatetimeIndex(sorted(ev.loc[ev['event'] == 'fomc_decision', 'date'].unique())).intersection(d)
vxe = pd.DatetimeIndex(sorted(ev.loc[ev['event'] == 'vix_expiry', 'date'].unique())).intersection(d)
wk_next = pd.Series(d.weekday, index=d).shift(-1)
fri_mon = d[(d.weekday == 4) & (wk_next == 0).values]
k3 = pd.DatetimeIndex([d[pos[f] - 3] for f in fomc if pos[f] >= 3])


def fwd(s, a, h):
    return np.array([s.iloc[pos[x] + h] / s[x] - 1 for x in a if pos[x] + h < len(d)])


print("=== 1. FOMC-week Monday ===")
k3fm = k3.intersection(fri_mon)
other = fri_mon.difference(k3)
v = fwd(vix, k3fm, 1)
b = fwd(vix, other, 1)
p0 = float((b > 0).mean())
w = int((v > 0).sum())
print(f"FOMC Mondays up {w}/{len(v)} mean {100*v.mean():+.2f}% med {100*np.median(v):+.2f}%; other Mondays up {100*p0:.1f}% mean {100*b.mean():+.2f}% med {100*np.median(b):+.2f}%")
print(f"  sign p vs other-Monday base: {sign_test(w, len(v), p0):.4f}")
print("  concentration:", cluster_note(k3fm[:len(v)], v))
print("  era:", [(e['label'], e['n'], round(e['mean_pct'], 2), round(e['hit'], 1)) for e in era_split(k3fm[:len(v)], v)])
vh3 = fwd(vix, k3fm, 3)
print(f"  Friday close -> decision-day close (h3): up {(vh3>0).sum()}/{len(vh3)} mean {100*vh3.mean():+.2f}% med {100*np.median(vh3):+.2f}%")
vs = fwd(C['^GSPC'], k3fm, 1)
print(f"  ^GSPC on FOMC Mondays: up {(vs>0).sum()}/{len(vs)} mean {100*vs.mean():+.3f}%")
# last 10 years
recent = k3fm[k3fm >= '2016-01-01']
vr = fwd(vix, recent, 1)
print(f"  since 2016: up {(vr>0).sum()}/{len(vr)} mean {100*vr.mean():+.2f}%")

print("\n=== 2. Yield surge into the decision ===")
fvx = C['^FVX']
ch5 = fvx - fvx.shift(5)
rank5 = ch5.rolling(252, min_periods=200).apply(lambda x: (x[:-1] < x[-1]).mean() * 100, raw=True)
k3v = pd.DatetimeIndex([x for x in k3 if pos[x] + 5 < len(d)])
surge = k3v[(rank5.reindex(k3v) >= 90).values]
rest = k3v.difference(surge)
for t, lab in [('IEF', 'IEF h3'), ('TLT', 'TLT h3'), ('^GSPC', '^GSPC h3')]:
    v1, v2 = fwd(C[t], surge, 3), fwd(C[t], rest, 3)
    diff = 100 * (v1.mean() - v2.mean())
    se = 100 * np.sqrt(v1.var(ddof=1) / len(v1) + v2.var(ddof=1) / len(v2))
    p0 = float((v2 > 0).mean())
    w = int((v1 > 0).sum())
    print(f"{lab}: surge up {w}/{len(v1)} mean {100*v1.mean():+.3f}% | rest up {int((v2>0).sum())}/{len(v2)} ({100*p0:.1f}%) mean {100*v2.mean():+.3f}% "
          f"| diff {diff:+.3f}pp t {diff/se:+.2f} | P(<= {w} up | base) {1 - sign_test(w + 1, len(v1), p0):.4f}")
h5s = np.array([100 * (fvx.iloc[pos[x] + 5] - fvx[x]) for x in surge])
h5r = np.array([100 * (fvx.iloc[pos[x] + 5] - fvx[x]) for x in rest])
w = int((h5s > 0).sum())
p0 = float((h5r > 0).mean())
print(f"5y h5 bp: surge higher {w}/{len(h5s)} mean {h5s.mean():+.2f} med {np.median(h5s):+.2f} | rest higher {int((h5r>0).sum())}/{len(h5r)} mean {h5r.mean():+.2f} "
      f"| sign p vs rest base {sign_test(w, len(h5s), p0):.4f}")
print("  h5 concentration:", cluster_note(surge, h5s / 100))
print("  h5 era (bp):", [(e['label'], e['n'], round(e['mean_pct'], 2), round(e['hit'], 1)) for e in era_split(surge, h5s / 100)])
spx3 = fwd(C['^GSPC'], surge, 3)
print("  ^GSPC h3 episodes:", [(str(x.date()), round(100 * y, 2)) for x, y in zip(surge, spx3)])
print("  ^GSPC h3 concentration:", cluster_note(surge, spx3))
print("  ^GSPC h3 era:", [(e['label'], e['n'], round(e['mean_pct'], 2), round(e['hit'], 1)) for e in era_split(surge, spx3)])
ief3 = fwd(C['IEF'], surge, 3)
print("  IEF h3 era:", [(e['label'], e['n'], round(e['mean_pct'], 3), round(e['hit'], 1)) for e in era_split(surge, ief3)])
print("  IEF h3 concentration:", cluster_note(surge, ief3))
print("  midterm anchors in surge set:", [str(x.date()) for x in surge if x.year % 4 == 2])

print("\n=== 3. Decision on VIX expiry: expiry or quarterly meeting? ===")
k1 = {f: d[pos[f] - 1] for f in fomc if pos[f] >= 1}
groups = {
    'quarterly month, on expiry': [f for f in fomc if f.month in (3, 6, 9, 12) and f in vxe],
    'quarterly month, not expiry': [f for f in fomc if f.month in (3, 6, 9, 12) and f not in vxe],
    'other month, on expiry': [f for f in fomc if f.month not in (3, 6, 9, 12) and f in vxe],
    'other month, not expiry': [f for f in fomc if f.month not in (3, 6, 9, 12) and f not in vxe],
}
for lab, fs in groups.items():
    a = pd.DatetimeIndex([k1[f] for f in fs])
    v = fwd(vix, a, 1)
    if len(v) == 0:
        print(f"{lab:30} n=0")
        continue
    a3 = pd.DatetimeIndex([d[pos[f] - 3] for f in fs])
    v3 = fwd(vix, a3, 3)
    print(f"{lab:30} n={len(v):3} decision-day VIX up {(v>0).sum():3}/{len(v):3} ({100*(v>0).mean():4.1f}%) mean {100*v.mean():+6.2f}% med {100*np.median(v):+6.2f}%"
          f" | Fri->Wed h3 up {(v3>0).sum()}/{len(v3)} med {100*np.median(v3):+.2f}%")
sept = [f for f in fomc if f.month == 9]
a = pd.DatetimeIndex([k1[f] for f in sept])
v = fwd(vix, a, 1)
print(f"September decisions: n={len(v)} decision-day VIX up {(v>0).sum()} mean {100*v.mean():+.2f}% med {100*np.median(v):+.2f}%")
post12 = [f for f in fomc if f.month in (3, 6, 9, 12) and f >= pd.Timestamp('2012-01-01')]
a = pd.DatetimeIndex([k1[f] for f in post12])
v = fwd(vix, a, 1)
print(f"dot-plot meetings 2012+: n={len(v)} decision-day VIX up {(v>0).sum()} mean {100*v.mean():+.2f}% med {100*np.median(v):+.2f}%")

print("\n=== 4a. NG=F September Mondays vs all Mondays ===")
ng = C['NG=F']
for lab, sel in [('all Fri->Mon', fri_mon), ('September Fri->Mon', fri_mon[fri_mon.month == 9]), ('other months Fri->Mon', fri_mon[fri_mon.month != 9])]:
    v = fwd(ng, sel, 1)
    v = v[~np.isnan(v)]
    print(f"  {lab:24} n={len(v)} mean {100*v.mean():+.3f}% up {100*(v>0).mean():.1f}%")
allses = fwd(ng, d[:-1], 1)
allses = allses[~np.isnan(allses)]
print(f"  all sessions            n={len(allses)} mean {100*allses.mean():+.3f}% up {100*(allses>0).mean():.1f}%")
sepm = fri_mon[fri_mon.month == 9]
v = fwd(ng, sepm, 1)
m = ~np.isnan(v)
print("  Sept NG concentration:", cluster_note(sepm[:len(v)][m], v[m]))
print("  Sept NG era:", [(e['label'], e['n'], round(e['mean_pct'], 2), round(e['hit'], 1)) for e in era_split(sepm[:len(v)][m], v[m])])

print("\n=== 4b. CL=F k3 FOMC h1 ===")
cl = C['CL=F']
v = fwd(cl, k3, 1)
m = ~np.isnan(v)
print(f"  n={m.sum()} mean {100*np.nanmean(v):+.3f}%")
print("  concentration:", cluster_note(k3[:len(v)][m], v[m], k=3))
k3fm_cl = fwd(cl, k3fm, 1)
oth_cl = fwd(cl, other, 1)
print(f"  CL on FOMC Mondays mean {100*np.nanmean(k3fm_cl):+.3f}% vs other Mondays {100*np.nanmean(oth_cl):+.3f}%")
v2 = v[m]
trim = np.sort(v2)[5:-5]
print(f"  trimmed (drop 5 each tail) mean {100*trim.mean():+.3f}%, median {100*np.median(v2):+.3f}%")
