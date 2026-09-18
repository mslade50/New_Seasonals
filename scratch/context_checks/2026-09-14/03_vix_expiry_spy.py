"""E:vix_expiry k2: SPY +0.187% t 2.84, QQQ 188-130 up (bh_pass), IWM t 2.49 on the session two before a
VIX expiry (h1 = the session before expiry, usually a Tuesday). Published 08-17 at 0.19 on SPY.

New specificity needed: (a) is it just Tuesday? (b) does it live in FOMC weeks, where the eve of the
decision is also the eve of expiry (tonight's configuration), or outside them? (c) era.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import load_prices, load_events, summarize, era_split, sign_test, cluster_note

px = load_prices(['SPY', 'QQQ', 'IWM', '^GSPC'])
d = px['SPY']['Close'].dropna().index
d = d[d >= '1999-01-01']
pos = pd.Series(range(len(d)), index=d)
R1 = {t: (px[t]['Close'].reindex(d).shift(-1) / px[t]['Close'].reindex(d) - 1) for t in ['SPY', 'QQQ', 'IWM']}

ev = load_events(None)
print("event kinds:", sorted(ev['event'].unique()) if 'event' in ev.columns else ev.columns.tolist())
kcol = 'event' if 'event' in ev.columns else ev.columns[1]
vx = pd.DatetimeIndex(sorted(ev.loc[ev[kcol] == 'vix_expiry', 'date'].unique()))
fo = pd.DatetimeIndex(sorted(ev.loc[ev[kcol] == 'fomc_decision', 'date'].unique()))
vx = vx[(vx > d[3]) & (vx <= d[-1])]


def anchor(evdates, k):
    out = []
    for e in evdates:
        if e in pos.index and pos[e] - k >= 0:
            out.append(d[pos[e] - k])
    return pd.DatetimeIndex(out)


k2 = anchor(vx, 2)
k2 = k2[k2 < d[-1]]
fomc_eve_set = set(anchor(fo, 1))
# h1 session of the k2 anchor is d[pos+1]; FOMC-overlap = that h1 session is the eve of a decision,
# or the decision lands within the expiry week (decision within +/-2 sessions of expiry)
fo_pos = set(pos[f] for f in fo if f in pos.index)


def overlap(a):
    p = pos[a]
    return any((p + 2 + j) in fo_pos for j in (-2, -1, 0, 1, 2))


def rep(label, a):
    a = pd.DatetimeIndex([x for x in a if pos[x] + 1 < len(d)])
    print(f"\n--- {label}: n={len(a)}")
    for t in ['SPY', 'QQQ', 'IWM']:
        v = R1[t].reindex(a).values
        s = summarize(v)
        if s['n'] == 0:
            continue
        w = int((v[~np.isnan(v)] > 0).sum())
        print(f"  {t} h1: mean {s['mean_pct']:+.3f}% med {s['median_pct']:+.3f}% up {w}/{s['n']} ({s['hit']:.1f}%) "
              f"t {s['t']:+.2f} signp {sign_test(w, s['n']):.4f}")
    return a


alld = d[:-1]
rep("all sessions (h1 = next session)", alld)
tue_anchor = alld[(pd.DatetimeIndex([d[pos[x] + 1] for x in alld]).weekday == 1)]
rep("all anchors whose h1 is a Tuesday", tue_anchor)
a = rep("VIX expiry k2, all", k2)
print("  h1 weekday mix:", pd.Series([d[pos[x] + 1].weekday() for x in a]).value_counts().to_dict())
v = R1['SPY'].reindex(a).values
print("  era SPY:", [(e['label'], e['n'], round(e['mean_pct'], 3), round(e['hit'], 1), round(e['t'], 2)) for e in era_split(a, v)])
print("  conc SPY:", cluster_note(a, v))

ov = pd.DatetimeIndex([x for x in a if overlap(x)])
nov = pd.DatetimeIndex([x for x in a if not overlap(x)])
b = rep("VIX expiry k2, FOMC decision within 2 sessions of expiry", ov)
vb = R1['SPY'].reindex(b).values
print("  episodes (anchor, SPY h1%):", [(str(x.date()), round(100 * R1['SPY'][x], 2)) for x in b])
if len(b) >= 5:
    print("  era:", [(e['label'], e['n'], round(e.get('mean_pct', np.nan), 3), round(e.get('hit', np.nan), 1)) for e in era_split(b, vb)])
    print("  conc:", cluster_note(b, vb))
c = rep("VIX expiry k2, no FOMC near", nov)
vc = R1['SPY'].reindex(c).values
print("  era:", [(e['label'], e['n'], round(e['mean_pct'], 3), round(e['hit'], 1), round(e['t'], 2)) for e in era_split(c, vc)])

eve_same = pd.DatetimeIndex([x for x in a if d[pos[x] + 1] in fomc_eve_set])
rep("VIX expiry k2 whose h1 is exactly the FOMC eve (tonight's configuration)", eve_same)
print("  dates:", [(str(x.date()), round(100 * R1['SPY'][x], 2), round(100 * R1['QQQ'][x], 2)) for x in eve_same])

fk2 = anchor(fo, 2)
fk2 = fk2[fk2 < d[-1]]
rep("FOMC k2 with no VIX expiry within 2 sessions", pd.DatetimeIndex([x for x in fk2 if x not in set(ov) and x not in set(eve_same)]))
