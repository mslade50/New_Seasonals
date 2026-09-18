"""Two follow-ons.

A. Friday's CPI print took ^VIX -11.2%. Mechanism to test: a vol collapse on a SCHEDULED print is
   event variance leaving the index, so it should stick better than a collapse on an ordinary
   session, which tends to mean-revert. Compare CPI-day VIX <= -X% vs non-event sessions with the
   same drop, h1/h3/h5, declustered 5td. Also split on whether an FOMC decision sits inside h5.

B. Wednesday's FOMC decision coincides with the monthly VIX expiry. How often, and does the
   decision-day VIX move differ when they share a date?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import load_prices, load_events, summarize, era_split, sign_test, cluster_note, declusters

px = load_prices(['SPY', '^GSPC', '^VIX'])
nyse = px['SPY']['Close'].dropna().index
d = nyse[nyse >= '1999-01-01']
vix = px['^VIX']['Close'].reindex(d)
spx = px['^GSPC']['Close'].reindex(d)
pos = pd.Series(range(len(d)), index=d)
vr1 = vix.pct_change(fill_method=None)
ev = load_events()
dates_of = {k: pd.DatetimeIndex(sorted(ev.loc[ev['event'] == k, 'date'].unique())) for k in ['cpi', 'fomc_decision', 'vix_expiry', 'nfp', 'ppi']}
cpi = dates_of['cpi'].intersection(d)
fomc = dates_of['fomc_decision'].intersection(d)
scheduled = set(dates_of['cpi']) | set(dates_of['fomc_decision']) | set(dates_of['nfp']) | set(dates_of['ppi'])


def fwd(s, a, h):
    return np.array([s.iloc[pos[x] + h] / s[x] - 1 for x in a])


def line(label, a, s=vix, hs=(1, 3, 5)):
    a = pd.DatetimeIndex([x for x in a if pos[x] + max(hs) < len(d)])
    print(f"  {label}: n={len(a)}")
    res = {}
    for h in hs:
        v = fwd(s, a, h)
        if len(v) == 0:
            continue
        w = int((v > 0).sum())
        st = summarize(v)
        res[h] = (a, v)
        print(f"     h{h}: mean {st['mean_pct']:+6.2f}% med {st['median_pct']:+6.2f}% up {w}/{len(v)} ({st['hit']:.1f}%)")
    return res


print("=== A. VIX collapse on a CPI print vs an ordinary session ===")
for thr in (-0.05, -0.08, -0.10):
    print(f"\nVIX <= {int(100*thr)}%")
    c = cpi[(vr1.reindex(cpi) <= thr).values]
    rc = line("CPI day, ^VIX", c)
    line("CPI day, ^GSPC", c, spx)
    o = d[(vr1 <= thr).fillna(False).values]
    o = pd.DatetimeIndex([x for x in o if x not in scheduled])
    o = declusters(o, 5, d)
    ro = line("non-event session, declustered 5td, ^VIX", o)
    if thr == -0.08:
        a, v = rc[5]
        print("     CPI episodes (date, VIX day %, level, h5 %):",
              [(str(x.date()), round(100 * vr1[x], 1), round(vix[x], 1), round(100 * vv, 1)) for x, vv in zip(a, v)])
        print("     era h5:", [(e['label'], e['n'], round(e['mean_pct'], 2), round(e['hit'], 1)) for e in era_split(a, v)])
        print("     concentration h5:", cluster_note(a, v))
        dn = int((v < 0).sum())
        base_dn = float((ro[5][1] < 0).mean())
        print(f"     h5 lower {dn}/{len(v)}; sign p vs 0.5 {sign_test(dn, len(v)):.4f}; vs non-event base {base_dn:.3f}: {sign_test(dn, len(v), base_dn):.4f}")
        v1, v2 = v, ro[5][1]
        diff = 100 * (v1.mean() - v2.mean())
        se = 100 * np.sqrt(v1.var(ddof=1) / len(v1) + v2.var(ddof=1) / len(v2))
        print(f"     h5 CPI minus non-event = {diff:+.2f}pp se {se:.2f} t {diff/se:+.2f}")
        # FOMC inside the next 5 sessions?
        has_f = [any((pos[x] < pos.get(f, -1) <= pos[x] + 5) for f in fomc) for x in a]
        print("     FOMC inside h5 for:", [str(x.date()) for x, hf in zip(a, has_f) if hf])

print("\n=== B. FOMC decision sharing a date with VIX expiry ===")
vx = dates_of['vix_expiry'].intersection(d)
both = fomc.intersection(vx)
only = fomc.difference(vx)
print(f"FOMC decisions {len(fomc)}; on a VIX expiry date {len(both)}: {[str(x.date()) for x in both]}")
for label, sel in [("decision day shares VIX expiry", both), ("decision day, no expiry", only)]:
    k1 = pd.DatetimeIndex([d[pos[f] - 1] for f in sel if pos[f] >= 1])
    k3 = pd.DatetimeIndex([d[pos[f] - 3] for f in sel if pos[f] >= 3])
    print(f"\n{label}")
    line("decision-day ^VIX (k1 anchor, h1)", k1, hs=(1, 2))
    line("^VIX from k3 anchor to decision close (h3)", k3, hs=(3,))
    line("decision-day ^GSPC (k1, h1)", k1, spx, hs=(1,))
exp_only = vx.difference(fomc)
k1 = pd.DatetimeIndex([d[pos[f] - 1] for f in exp_only if f in pos.index and pos[f] >= 1])
print("\nVIX expiry, no FOMC")
line("expiry-day ^VIX (k1, h1)", k1, hs=(1, 2))
