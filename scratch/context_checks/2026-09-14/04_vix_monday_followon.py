"""Last night's headline cell printed: the VIX rose 7.95% to 17.10 on the Monday two sessions before
the decision, gapping +10.5% and fading -2.3% from the open, on an S&P decline of only 0.48%.

(a) Pre-decision Mondays (k2) with a VIX lift of 5%+ / 7%+: what did the eve (Tuesday) and the
    decision day do, for the VIX and the S&P? Against pre-decision Mondays with a smaller lift.
(b) General cell: VIX +7% or more while the S&P falls less than 0.5% (a vol bid without a selloff),
    declustered, VIX h1/h5 and S&P h1/h5, against VIX +7% days with a real S&P selloff.
All on the NYSE calendar (SPY index) because ^VIX carries phantom bars on 2026 closures.
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
vr = V.pct_change()
sr = S.pct_change()


def fwd(s, a, h):
    return np.array([s.iloc[pos[x] + h] / s[x] - 1 if pos[x] + h < len(d) else np.nan for x in a])


def line(name, v):
    s = summarize(v)
    if s['n'] == 0:
        return f"{name}: n=0"
    w = int((v[~np.isnan(v)] > 0).sum())
    return (f"{name}: mean {s['mean_pct']:+.2f}% med {s['median_pct']:+.2f}% up {w}/{s['n']} t {s['t']:+.2f} "
            f"signp_up {sign_test(w, s['n']):.4f} signp_dn {sign_test(s['n'] - w, s['n']):.4f}")


print("tonight:", d[-1].date(), "VIX", V.iloc[-1], "vix 1d", round(100 * vr.iloc[-1], 2), "spx 1d", round(100 * sr.iloc[-1], 2))

ev = load_events(['fomc_decision'])
fo = pd.DatetimeIndex(sorted(ev['date'].unique()))
k2 = pd.DatetimeIndex([d[pos[f] - 2] for f in fo if f in pos.index and pos[f] >= 2])
k2 = k2[k2 < d[-1]]
k2m = k2[k2.weekday == 0]
print(f"\n(a) pre-decision k2 sessions: {len(k2)}, of which Mondays {len(k2m)}")
for lab, sel in [("k2 all, VIX lift >= 5%", k2[(vr.reindex(k2) >= 0.05).values]),
                 ("k2 all, VIX lift >= 7%", k2[(vr.reindex(k2) >= 0.07).values]),
                 ("k2 all, VIX lift < 5%", k2[(vr.reindex(k2) < 0.05).values]),
                 ("k2 Mondays, VIX lift >= 5%", k2m[(vr.reindex(k2m) >= 0.05).values]),
                 ("k2 all, VIX lift >= 5% and S&P down < 1%", k2[((vr.reindex(k2) >= 0.05) & (sr.reindex(k2) > -0.01)).values])]:
    print(f"\n--- {lab}: n={len(sel)}")
    print("  " + line("VIX h1 (eve)", fwd(V, sel, 1)))
    print("  " + line("VIX h2 (decision close)", fwd(V, sel, 2)))
    print("  " + line("S&P h1 (eve)", fwd(S, sel, 1)))
    print("  " + line("S&P h2 (decision close)", fwd(S, sel, 2)))
    print("  " + line("S&P h5", fwd(S, sel, 5)))
    if 0 < len(sel) <= 40:
        print("  episodes:", [(str(x.date()), round(100 * vr[x], 1), round(100 * sr[x], 2),
                               round(100 * (V.iloc[pos[x] + 2] / V[x] - 1), 1), round(100 * (S.iloc[pos[x] + 2] / S[x] - 1), 2))
                              for x in sel if pos[x] + 2 < len(d)])
        v2 = fwd(V, sel, 2)
        m = ~np.isnan(v2)
        if m.sum() >= 5:
            print("  era VIX h2:", [(e['label'], e['n'], round(e.get('mean_pct', np.nan), 2), round(e.get('hit', np.nan), 1)) for e in era_split(sel[m], v2[m])])
            print("  conc VIX h2:", cluster_note(sel[m], v2[m]))

print("\n(b) general cells, declustered 5td")
alld = d[20:-1]
base = alld
print("  " + line("all sessions VIX h1", fwd(V, base[:-5], 1)))
print("  " + line("all sessions VIX h5", fwd(V, base[:-5], 5)))
print("  " + line("all sessions S&P h5", fwd(S, base[:-5], 5)))
calm = alld[((vr.reindex(alld) >= 0.07) & (sr.reindex(alld) > -0.005)).values]
sell = alld[((vr.reindex(alld) >= 0.07) & (sr.reindex(alld) <= -0.01)).values]
for lab, sel in [("VIX >= +7% with S&P down less than 0.5% (or up)", calm),
                 ("VIX >= +7% with S&P down 1% or more", sell)]:
    for dc in (0, 5):
        a = declusters(sel, dc, d) if dc else sel
        a = pd.DatetimeIndex([x for x in a if pos[x] + 5 < len(d)])
        print(f"\n--- {lab}, decluster {dc}: n={len(a)}")
        for h in (1, 2, 5):
            print("  " + line(f"VIX h{h}", fwd(V, a, h)))
        for h in (1, 5):
            print("  " + line(f"S&P h{h}", fwd(S, a, h)))
        if dc == 5:
            v5 = fwd(V, a, 5)
            print("  era VIX h5:", [(e['label'], e['n'], round(e['mean_pct'], 2), round(e['hit'], 1), round(e['t'], 2)) for e in era_split(a, v5)])
            print("  conc VIX h5:", cluster_note(a, v5))
            # VIX level split
            lv = V.reindex(a).values
            for lo, hi in [(0, 18), (18, 25), (25, 99)]:
                m = (lv >= lo) & (lv < hi)
                print(f"  VIX close in [{lo},{hi}): " + line("VIX h5", v5[m]))
