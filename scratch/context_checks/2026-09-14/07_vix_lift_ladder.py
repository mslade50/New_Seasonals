"""Sensitivity check for 04: the k2 VIX-lift cell looked clean at >=7% (VIX lower by the decision close
31 of 37) but the >=5% & S&P-down-<1% cell had a positive mean. Threshold ladder so the published cut is
not a scanned one, restricted to S&P falling less than 1% (tonight -0.48%), with the same-state
non-FOMC control (no decision within 5 sessions, declustered 5td) at each rung.
All on the NYSE calendar.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import load_prices, load_events, summarize, era_split, sign_test, cluster_note, declusters

px = load_prices(['SPY', '^GSPC', '^VIX'])
d = px['SPY']['Close'].dropna().index
d = d[d >= '1999-01-01']
pos = pd.Series(range(len(d)), index=d)
V = px['^VIX']['Close'].reindex(d)
S = px['^GSPC']['Close'].reindex(d)
vr, sr = V.pct_change(), S.pct_change()

ev = load_events(['fomc_decision'])
fo = pd.DatetimeIndex(sorted(ev['date'].unique()))
k2 = pd.DatetimeIndex([d[pos[f] - 2] for f in fo if f in pos.index and pos[f] >= 2])
k2 = k2[k2 < d[-2]]
near = set()
for f in fo:
    if f in pos.index:
        for k in range(0, 6):
            if pos[f] - k >= 0:
                near.add(d[pos[f] - k])


def fwd(s, a, h):
    return np.array([s.iloc[pos[x] + h] / s[x] - 1 for x in a if pos[x] + h < len(d)])


print(f"{'cell':52} {'n':>4} {'VIXh2 mean':>10} {'med':>7} {'lower':>9} {'signp':>7} | {'ctrl n':>6} {'ctrl mean':>9} {'ctrl lower%':>11}")
for lo in (0.0, 0.03, 0.05, 0.06, 0.07, 0.08, 0.10):
    for hi in (9.9,):
        m = (vr >= lo) & (vr < hi) & (sr > -0.01)
        a = k2[m.reindex(k2).fillna(False).values]
        v = fwd(V, a, 2)
        lower = int((v < 0).sum())
        ctrl = pd.DatetimeIndex([x for x in d[m.fillna(False).values] if x not in near and pos[x] + 2 < len(d)])
        ctrl = declusters(ctrl, 5, d)
        cv = fwd(V, ctrl, 2)
        print(f"k2, VIX lift >= {lo*100:4.1f}%, S&P > -1%{'':22} {len(v):4d} {100*v.mean():+9.2f}% {100*np.median(v):+6.2f}% "
              f"{lower:3d}/{len(v):<4d} {sign_test(lower, len(v)):7.4f} | {len(cv):6d} {100*cv.mean():+8.2f}% {100*(cv<0).mean():10.1f}%")

print("\nbands (non-cumulative), S&P > -1%:")
for lo, hi in [(-9, 0.0), (0.0, 0.03), (0.03, 0.05), (0.05, 0.07), (0.07, 0.10), (0.10, 9.9)]:
    m = (vr >= lo) & (vr < hi) & (sr > -0.01)
    a = k2[m.reindex(k2).fillna(False).values]
    v = fwd(V, a, 2)
    if len(v) == 0:
        continue
    s2 = fwd(S, a, 2)
    print(f"  lift [{lo*100:5.1f}%, {hi*100:5.1f}%): n={len(v):3d} VIX h2 mean {100*v.mean():+6.2f}% med {100*np.median(v):+6.2f}% "
          f"lower {int((v<0).sum())}/{len(v)} | S&P h2 mean {100*s2.mean():+.2f}% up {int((s2>0).sum())}/{len(s2)}")

sel = k2[((vr >= 0.07) & (sr > -0.01)).reindex(k2).fillna(False).values]
v2 = fwd(V, sel, 2)
v1 = fwd(V, sel, 1)
s1 = fwd(S, sel, 1)
s2 = fwd(S, sel, 2)
print(f"\nchosen cut >=7% & S&P > -1%: n={len(sel)}")
for nm, v in [("VIX h1 (eve)", v1), ("VIX h2 (decision close)", v2), ("S&P h1", s1), ("S&P h2", s2)]:
    st = summarize(v)
    up = int((v > 0).sum())
    print(f"  {nm}: mean {st['mean_pct']:+.2f}% med {st['median_pct']:+.2f}% up {up}/{st['n']} t {st['t']:+.2f} "
          f"signp_dn {sign_test(st['n'] - up, st['n']):.5f} signp_up {sign_test(up, st['n']):.4f}")
print("  era VIX h2:", [(e['label'], e['n'], round(e['mean_pct'], 2), round(100 - e['hit'], 1)) for e in era_split(sel, v2)], "(mean, pct lower)")
print("  conc VIX h2:", cluster_note(sel, v2))
print("  higher by decision close:", [(str(x.date()), round(100 * vr[x], 1), round(100 * (V.iloc[pos[x] + 2] / V[x] - 1), 1)) for x, vv in zip(sel, v2) if vv >= 0])
print("  VIX level at anchor: median", round(float(V.reindex(sel).median()), 2), "tonight", V.iloc[-1])
lv = V.reindex(sel).values
for lo, hi in [(0, 17), (17, 22), (22, 99)]:
    mm = (lv >= lo) & (lv < hi)
    vv = v2[mm]
    if len(vv):
        print(f"  anchor VIX in [{lo},{hi}): n={len(vv)} h2 mean {100*vv.mean():+.2f}% lower {int((vv<0).sum())}")
# all k2 control and all non-FOMC days with same state
allk2 = fwd(V, k2, 2)
print(f"\nall k2 VIX h2: n={len(allk2)} mean {100*allk2.mean():+.2f}% lower {int((allk2<0).sum())}/{len(allk2)}")
ctrl = pd.DatetimeIndex([x for x in d[((vr >= 0.07) & (sr > -0.01)).fillna(False).values] if x not in near and pos[x] + 2 < len(d)])
for dc in (0, 5):
    c = declusters(ctrl, dc, d) if dc else ctrl
    cv = fwd(V, c, 2)
    print(f"non-FOMC VIX >=7% & S&P > -1%, decluster {dc}: n={len(cv)} VIX h2 mean {100*cv.mean():+.2f}% med {100*np.median(cv):+.2f}% lower {int((cv<0).sum())}/{len(cv)}")
