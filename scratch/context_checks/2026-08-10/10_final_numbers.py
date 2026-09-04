"""Final pass: every number that goes in the brief, on native indices.

Recent-era records for the two headline cells, plus the natgas PPI cell redone
off NG=F's own session index (drill 04 built it on a two-ticker panel), plus
the precious-complex thrust cell from drill 03.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    load_prices, load_events, fwd_ret, declusters, summarize, era_split,
    sign_test, cluster_note,
)

px = load_prices(["SPY", "^VIX", "TLT", "CL=F", "NG=F", "GC=F", "SI=F", "PL=F", "PA=F"])
C = {t: px[t]["Close"].dropna().sort_index() for t in px}


def rec(v, label, indent="   "):
    r = summarize(v.values, "")
    up = int((v.values > 0).sum())
    print(f"{indent}{label:54s} n={r['n']:4d} mean {r['mean_pct']:+.3f}% "
          f"med {r['median_pct']:+.3f}% hit {r['hit']:.1f}% t={r['t']:+.2f} | "
          f"{up}-{len(v)-up} up-p {sign_test(up, len(v)):.4f} "
          f"dn-p {sign_test(len(v)-up, len(v)):.4f}")


def anchors_k(sess, kinds, k):
    ev = load_events(kinds)
    out = []
    for d in ev["date"]:
        d = pd.Timestamp(d)
        pos = sess.searchsorted(d)
        if pos < len(sess) and sess[pos] == d and pos - k >= 0:
            out.append(sess[pos - k])
    return pd.DatetimeIndex(out)


print("=" * 78)
print("1. VIX on the CPI eve, SPY at a 52w high -- by period")
print("=" * 78)
spy, vix = C["SPY"], C["^VIX"]
dh = spy / spy.rolling(252).max() - 1.0
a = anchors_k(spy.index, ["cpi"], 2)
a_hi = a[(dh.reindex(a).fillna(-1) > -0.005).values]
v = fwd_ret(vix, 1).reindex(a_hi).dropna()
rec(v, "full sample")
for lo, hi_ in ((2000, 2009), (2010, 2019), (2020, 2026)):
    sub = v[(v.index.year >= lo) & (v.index.year <= hi_)]
    if len(sub) >= 3:
        rec(sub, f"{lo}-{hi_}")
print(f"   last 10 observations: "
      f"{[(str(d.date()), round(100*x, 1)) for d, x in v.tail(10).items()]}")
print(f"   ^VIX today closed 15.46, +3.76% on the session, "
      f"{100*(vix.iloc[-1]/vix.rolling(252).max().iloc[-1]-1):.1f}% below its 52w high")

print("\n" + "=" * 78)
print("2. TLT over the CPI session, crude hot -- by period")
print("=" * 78)
tlt, cl = C["TLT"], C["CL=F"]
cl21 = cl / cl.shift(21) - 1.0
at = anchors_k(tlt.index, ["cpi"], 2)
hot = (cl21.reindex(at).fillna(-1) >= 0.10).values
v2 = fwd_ret(tlt, 2).reindex(at[hot]).dropna()
rec(v2, "full sample, crude 21d >= +10%")
for lo, hi_ in ((2002, 2009), (2010, 2019), (2020, 2026)):
    sub = v2[(v2.index.year >= lo) & (v2.index.year <= hi_)]
    if len(sub) >= 3:
        rec(sub, f"{lo}-{hi_}")
print(f"   last 8 observations: "
      f"{[(str(d.date()), round(100*x, 2)) for d, x in v2.tail(8).items()]}")
neg = v2[v2 < 0]
print(f"   the {len(neg)} losses: {[(str(d.date()), round(100*x, 2)) for d, x in neg.items()]}")

print("\n" + "=" * 78)
print("3. NG=F, 3 td before a PPI, on NG's own session index")
print("=" * 78)
ng = C["NG=F"]
ap = anchors_k(ng.index, ["ppi"], 3)
v3 = fwd_ret(ng, 1).reindex(ap).dropna()
rec(v3, "the cell")
print(f"        {cluster_note(v3.index, v3.values)}")
for e in era_split(v3.index, v3.values):
    if e.get("n", 0):
        print(f"        era n={e['n']:4d} mean {e['mean_pct']:+.3f}% hit {e['hit']:.1f}% "
              f"t={e['t']:+.2f}")
rec(fwd_ret(ng, 1).dropna(), "control: all NG sessions")
tdom = pd.Series(1, index=ng.index).groupby([ng.index.year, ng.index.month]).cumsum()
band = ng.index[(tdom.values >= 6) & (tdom.values <= 8)]
rec(fwd_ret(ng, 1).reindex(band).dropna(), "control: trading-day-of-month 6-8")
rec(fwd_ret(ng, 1).reindex(band.difference(ap)).dropna(),
    "control: tdom 6-8 that are NOT PPI k3 anchors")
for lo, hi_ in ((2000, 2009), (2010, 2017), (2018, 2026)):
    sub = v3[(v3.index.year >= lo) & (v3.index.year <= hi_)]
    if len(sub) >= 3:
        rec(sub, f"   {lo}-{hi_}")
print(f"   today's NG=F session: {100*(ng.iloc[-1]/ng.iloc[-2]-1):+.2f}%, "
      f"and today IS the 3-td-before-PPI anchor (PPI 2026-08-13)")

print("\n" + "=" * 78)
print("4. the precious complex thrust")
print("=" * 78)
r5 = {t: C[t] / C[t].shift(5) - 1.0 for t in ("GC=F", "SI=F", "PL=F", "PA=F")}
idx = C["GC=F"].index
m = pd.Series(True, index=idx)
for t, thr in (("GC=F", 0.05), ("SI=F", 0.08), ("PL=F", 0.05), ("PA=F", 0.05)):
    m &= (r5[t].reindex(idx).fillna(-9) >= thr)
trig = idx[m.values]
trig = trig[trig <= idx[-2]]
dc = declusters(trig, 10, idx)
print(f"   raw {len(trig)} sessions -> {len(dc)} episodes; "
      f"years {sorted(set(pd.DatetimeIndex(dc).year))}")
print("   live 5d: " + "  ".join(f"{t} {100*r5[t].iloc[-1]:+.2f}%" for t in r5))
for t in ("SI=F", "GC=F"):
    for h in (1, 5):
        vv = fwd_ret(C[t], h).reindex(dc).dropna()
        if len(vv) >= 3:
            rec(vv, f"{t} h{h}")
    vv = fwd_ret(C[t], 5).reindex(dc).dropna()
    print(f"        {cluster_note(vv.index, vv.values)}")
    for e in era_split(vv.index, vv.values):
        if e.get("n", 0):
            print(f"        h5 era n={e['n']:3d} mean {e['mean_pct']:+.3f}% "
                  f"hit {e['hit']:.1f}% t={e['t']:+.2f}")
