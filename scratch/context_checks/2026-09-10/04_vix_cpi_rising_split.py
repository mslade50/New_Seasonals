"""Nail down: the CPI-day VIX decline belongs to CPI eves where VIX CLOSED HIGHER.

Note the pitch_lab convention: summarize/era_split take FRACTIONS and return PERCENT.
Drill 03 fed them percent and its era numbers were 100x. Fixed here.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np, pandas as pd
from pitch_lab import load_prices, load_events, summarize, era_split, sign_test, cluster_note

px = load_prices(['^VIX', '^GSPC', 'SPY', '^VIX3M', 'TLT'])
ev = load_events()
cpi = pd.DatetimeIndex(sorted(pd.to_datetime(ev.loc[ev['event'] == 'cpi', 'date'].unique())))

vix = px['^VIX']['Close'].dropna()
spx = px['^GSPC']['Close'].dropna()
d = vix.index
pairs = [(d[d < c][-1], c) for c in cpi if c in d and len(d[d < c])]
pairs = [(a, c) for a, c in pairs if a >= pd.Timestamp('1999-01-01')]
r1 = vix.pct_change()

def leg(sel, s=vix):
    rows = [(a, s.loc[c] / s.loc[a] - 1) for a, c in pairs if a in sel and a in s.index and c in s.index]
    return pd.DatetimeIndex([r[0] for r in rows]), np.array([r[1] for r in rows])

allA = pd.DatetimeIndex([a for a, _ in pairs])
rising = set(allA[r1.reindex(allA) > 0])
falling = set(allA[r1.reindex(allA) <= 0])

for name, sel in [("CPI eve closed HIGHER in VIX", rising), ("CPI eve closed lower/flat", falling)]:
    dd, v = leg(sel)
    st = summarize(v, name)
    w = int((v > 0).sum()); n = len(v)
    print(f"\n{name}: n={n} mean={st['mean_pct']:+.3f}% median={st['median_pct']:+.3f}% "
          f"up={w}/{n} t={st['t']:+.2f} signp(down)={sign_test(n-w, n):.4f}")
    print("  era:", [(e['label'], e['n'], round(e['mean_pct'], 3), round(e['hit'], 1)) for e in era_split(dd, v)])
    print("  concentration:", cluster_note(dd, v))
    sd, sv = leg(sel, spx)
    sw = int((sv > 0).sum())
    print(f"  ^GSPC same days: n={len(sv)} mean={summarize(sv)['mean_pct']:+.3f}% up={sw}/{len(sv)} "
          f"signp={sign_test(sw, len(sv)):.4f}")

# --- controls -------------------------------------------------------------
print("\n=== CONTROLS ===")
nxt_all = (vix.shift(-1) / vix - 1).dropna()
nxt_all = nxt_all[nxt_all.index >= pd.Timestamp('1999-01-01')]
w = int((nxt_all > 0).sum())
print(f"every session 1999+          n={len(nxt_all):5} mean={100*nxt_all.mean():+.3f}% up={100*w/len(nxt_all):.1f}%")

up_all = nxt_all[r1.reindex(nxt_all.index) > 0]
w = int((up_all > 0).sum())
print(f"any session VIX closed higher n={len(up_all):5} mean={100*up_all.mean():+.3f}% up={100*w/len(up_all):.1f}%"
      f"   <- the state without the CPI")

dn_all = nxt_all[r1.reindex(nxt_all.index) <= 0]
w = int((dn_all > 0).sum())
print(f"any session VIX closed lower  n={len(dn_all):5} mean={100*dn_all.mean():+.3f}% up={100*w/len(dn_all):.1f}%")

# the edge that matters: CPI-rising vs non-CPI-rising
dd, v = leg(rising)
non = up_all.drop([x for x in dd if x in up_all.index], errors='ignore')
diff = 100 * v.mean() - 100 * non.mean()
se = np.sqrt(v.var(ddof=1) / len(v) + non.var(ddof=1) / len(non)) * 100
print(f"\nCPI eve rising ({100*v.mean():+.3f}%) minus any rising session ({100*non.mean():+.3f}%) "
      f"= {diff:+.3f}pp, se {se:.3f}, t {diff/se:+.2f}")

# does the size of the eve move matter inside the rising set?
print("\n=== inside the rising set, by eve move size ===")
for lo, hi, lab in [(0, .02, '0-2%'), (.02, .05, '2-5%'), (.05, .10, '5-10%'), (.10, 9, '10%+')]:
    sel = set(allA[(r1.reindex(allA) > lo) & (r1.reindex(allA) <= hi)])
    dd2, v2 = leg(sel)
    if len(v2) < 5: print(f"  {lab:6} n={len(v2)}"); continue
    w2 = int((v2 > 0).sum())
    print(f"  {lab:6} n={len(v2):3} mean={100*v2.mean():+7.3f}% up={w2:3}/{len(v2):3} "
          f"signp(down)={sign_test(len(v2)-w2, len(v2)):.4f}")

# tonight is also a Friday CPI in a midterm September; check the rising cell survives those
print("\n=== rising-eve cell, extra conditions ===")
dd, v = leg(rising)
for lab, m in [("September prints", dd.month == 9),
               ("midterm years", dd.year % 4 == 2),
               ("print lands on a Friday", pd.DatetimeIndex([c for a, c in pairs if a in rising]).weekday == 4)]:
    vv = v[m]
    if len(vv) < 4: print(f"  {lab:26} n={len(vv)}"); continue
    w = int((vv > 0).sum())
    print(f"  {lab:26} n={len(vv):3} mean={100*vv.mean():+7.3f}% up={w}/{len(vv)} signp(down)={sign_test(len(vv)-w, len(vv)):.4f}")
