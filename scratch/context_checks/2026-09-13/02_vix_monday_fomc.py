"""The engine's two loudest tomorrow cells are both ^VIX on a Monday:
  E:fomc_decision|^VIX|k3  (+1.76%, t 3.57)   -- a Wed decision puts the k3 anchor on a Friday
  E:weekday_month|^VIX     (+2.99%, t 3.36)   -- Mondays in September
Both are measured against ALL days. The weekend lift in VIX is a known artifact, so the
right control is all Mondays. Then condition on Friday's -11.2% VIX session.

Calendar: VIX restricted to SPY (NYSE) dates, dropping the 2026 phantom bars.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import load_prices, load_events, summarize, era_split, sign_test, cluster_note, declusters

px = load_prices(['^VIX', 'SPY', '^GSPC'])
nyse = px['SPY']['Close'].dropna().index
nyse = nyse[nyse >= '1999-01-01']
vix = px['^VIX']['Close'].reindex(nyse).dropna()
spx = px['^GSPC']['Close'].reindex(nyse).dropna()
d = vix.index
ev = load_events(['fomc_decision'])
fomc = pd.DatetimeIndex(sorted(ev['date'].unique()))

r1 = vix.pct_change()
nxt = vix.shift(-1) / vix - 1
nxt_spx = (spx.shift(-1) / spx - 1).reindex(d)
wk_next = pd.Series(d.weekday, index=d).shift(-1)


def line(label, idx, s=nxt):
    idx = pd.DatetimeIndex([x for x in idx if x in s.index and not np.isnan(s.loc[x])])
    v = s.reindex(idx).values
    if len(v) == 0:
        print(f"{label:58} n=0")
        return idx, v
    st = summarize(v)
    w = int((v > 0).sum())
    print(f"{label:58} n={st['n']:4} mean={st['mean_pct']:+7.3f}% med={st['median_pct']:+7.3f}% "
          f"up={w}/{len(v)} ({st['hit']:.1f}%) t={st['t']:+.2f} signp_up={sign_test(w, len(v)):.4f}")
    return idx, v


fri_mon = d[(d.weekday == 4) & (wk_next == 0).values]
other = d.difference(fri_mon)
print("=== VIX next-session change, calendar controls ===")
line("all sessions", d[:-1])
line("Friday close -> Monday close (all)", fri_mon)
line("every other session", other[:-1])

# k3 anchors before FOMC decisions
pos = pd.Series(range(len(d)), index=d)
k3 = []
for f in fomc:
    if f in pos.index and pos[f] >= 3:
        k3.append(d[pos[f] - 3])
k3 = pd.DatetimeIndex(k3)
k3 = k3[k3 >= '1999-01-01']
k3_fm = k3.intersection(fri_mon)
print("\n=== FOMC k3 anchors ===")
line("all k3 anchors", k3)
line("k3 anchor on a Friday, h1 a Monday", k3_fm)
line("k3 anchor NOT Fri->Mon", k3.difference(fri_mon))
line("Fri->Mon, NOT an FOMC week", fri_mon.difference(k3))
a, v = line("k3 Fri->Mon", k3_fm)
b = nxt.reindex(fri_mon.difference(k3)).dropna().values
diff = 100 * (v.mean() - b.mean())
se = 100 * np.sqrt(v.var(ddof=1) / len(v) + b.var(ddof=1) / len(b))
print(f"  FOMC-week Monday minus other Mondays = {diff:+.3f}pp, se {se:.3f}, t {diff/se:+.2f}")
print("  era:", [(e['label'], e['n'], round(e['mean_pct'], 3), round(e['hit'], 1)) for e in era_split(a, v)])

print("\n=== September Mondays ===")
sep_fm = fri_mon[fri_mon.month == 9]
a, v = line("September Fri->Mon", sep_fm)
b = nxt.reindex(fri_mon[fri_mon.month != 9]).dropna().values
diff = 100 * (v.mean() - b.mean())
se = 100 * np.sqrt(v.var(ddof=1) / len(v) + b.var(ddof=1) / len(b))
print(f"  Sept Monday minus other Mondays = {diff:+.3f}pp, se {se:.3f}, t {diff/se:+.2f}")
print("  era:", [(e['label'], e['n'], round(e['mean_pct'], 3), round(e['hit'], 1)) for e in era_split(a, v)])
print("  concentration:", cluster_note(a, v))
for m in range(1, 13):
    vv = nxt.reindex(fri_mon[fri_mon.month == m]).dropna().values
    print(f"   month {m:2}: n={len(vv):3} mean={100*vv.mean():+6.2f}% up={100*(vv>0).mean():5.1f}%")

print("\n=== Friday VIX collapse -> Monday ===")
for thr in [-0.05, -0.08, -0.10]:
    sel = fri_mon[(r1.reindex(fri_mon) <= thr).values]
    a, v = line(f"Friday VIX <= {int(thr*100)}% -> Monday VIX", sel)
    line(f"   same Mondays, ^GSPC", sel, nxt_spx)
    anyday = d[:-1][(r1.reindex(d[:-1]) <= thr).values]
    line(f"   any session VIX <= {int(thr*100)}% -> next VIX", anyday)
    if thr == -0.10:
        print("   concentration:", cluster_note(a, v))
        print("   era:", [(e['label'], e['n'], round(e['mean_pct'], 3), round(e['hit'], 1)) for e in era_split(a, v)])
        print("   episodes:", [(str(x.date()), round(100 * r1[x], 1), round(100 * nxt[x], 1), round(vix[x], 2)) for x in a])

print("\n=== Friday VIX collapse inside FOMC weeks (k3) ===")
for thr in [-0.03, -0.05, -0.08]:
    sel = k3_fm[(r1.reindex(k3_fm) <= thr).values]
    a, v = line(f"k3 Fri VIX <= {int(thr*100)}% -> Monday VIX", sel)
    if len(v):
        # through decision day close (h3) and ^GSPC
        h3 = np.array([vix.iloc[pos[x] + 3] / vix[x] - 1 for x in a if pos[x] + 3 < len(d)])
        w = int((h3 > 0).sum())
        print(f"      -> decision-day close VIX h3: n={len(h3)} mean={100*h3.mean():+.2f}% up={w}/{len(h3)}")
        print("      episodes:", [(str(x.date()), round(100 * r1[x], 1), round(100 * nxt[x], 1)) for x in a])
