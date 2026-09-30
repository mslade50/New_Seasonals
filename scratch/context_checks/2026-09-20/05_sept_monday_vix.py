"""September Mondays and the VIX.

Engine cell: anchor the session before a September Monday, h1 = that Monday.
n 87, VIX h1 +3.05%, 56-31 up, t 3.46, sign p 0.0048, BH pass, era-stable,
edge +2.79pp over ALL days.

That control cannot separate two confounded things: Mondays in general (the VIX
carries a well known weekend/decay seam) and September in general. Run the 2x2.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, summarize, show, sign_test, cluster_note  # noqa

px = close_panel(["^VIX", "^GSPC", "SPY", "^VIX3M"]).dropna(subset=["^VIX", "^GSPC"])
px = px[px.index >= "1999-01-01"]
vix, spx = px["^VIX"], px["^GSPC"]
ret = vix / vix.shift(1) - 1.0          # the session's own VIX change
sret = spx / spx.shift(1) - 1.0
print("coverage:", px.index.min().date(), "->", px.index.max().date(), "n", len(px))

dow = px.index.dayofweek
mon = dow == 0
sep = px.index.month == 9

cells = {
    "September Mondays": mon & sep,
    "Mondays, other months": mon & ~sep,
    "September, non-Monday": ~mon & sep,
    "all other sessions": ~mon & ~sep,
    "ALL sessions": np.ones(len(px), dtype=bool),
}
rows = []
for name, m in cells.items():
    v = ret[m].dropna().values
    r = summarize(v, name)
    up = int((v > 0).sum())
    r["rec"] = f"{up}-{len(v)-up}"
    r["sign_p"] = round(sign_test(max(up, len(v) - up), len(v)), 4)
    rows.append(r)
show(rows, "VIX same-session change, the 2x2")

print("\n--- the two marginals, which one carries it? ---")
m_sep_mon = ret[mon & sep].dropna()
m_mon_oth = ret[mon & ~sep].dropna()
m_sep_non = ret[~mon & sep].dropna()
print(f"September Monday minus other-month Monday: "
      f"{100*(m_sep_mon.mean()-m_mon_oth.mean()):+.2f}pp")
print(f"September Monday minus September non-Monday: "
      f"{100*(m_sep_mon.mean()-m_sep_non.mean()):+.2f}pp")

# Welch t on September Mondays vs all other Mondays
a, b = m_sep_mon.values, m_mon_oth.values
se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
print(f"Welch t (Sep Mon vs other Mon): {(a.mean()-b.mean())/se:+.2f}  n {len(a)} vs {len(b)}")

print("\n--- month-by-month Monday VIX change, to see if September is special ---")
out = []
for mth in range(1, 13):
    v = ret[mon & (px.index.month == mth)].dropna().values
    up = int((v > 0).sum())
    out.append({"month": mth, "n": len(v), "mean_pct": round(100 * v.mean(), 3),
                "median_pct": round(100 * float(np.median(v)), 3),
                "hit_up": round(100 * up / len(v), 1), "rec": f"{up}-{len(v)-up}"})
print(pd.DataFrame(out).to_string(index=False))

print("\n--- concentration and era for September Mondays ---")
d = ret[mon & sep].dropna()
print(cluster_note(d.index, d.values))
for cut in ("2011-01-01", "2018-01-01"):
    m = d.index < pd.Timestamp(cut)
    for lbl, vv in ((f"pre-{cut[:4]}", d.values[m]), (f"{cut[:4]}+", d.values[~m])):
        up = int((vv > 0).sum())
        print(f"  {lbl:9} n {len(vv):3} mean {100*vv.mean():+6.2f}% median {100*np.median(vv):+6.2f}% "
              f"rec {up}-{len(vv)-up} signp {sign_test(max(up,len(vv)-up), len(vv)):.4f}")

print("\n--- what is the S&P doing on those same September Mondays? ---")
rows = []
for name, m in cells.items():
    v = sret[m].dropna().values
    r = summarize(v, name)
    up = int((v > 0).sum())
    r["rec"] = f"{up}-{len(v)-up}"
    rows.append(r)
show(rows, "S&P same-session change, the 2x2")

print("\n--- drop the crisis years, does the September-Monday VIX cell survive? ---")
for drop in ([2008], [2008, 2020], [2008, 2011, 2020]):
    m = mon & sep & ~px.index.year.isin(drop)
    v = ret[m].dropna().values
    up = int((v > 0).sum())
    print(f"  ex {drop}: n {len(v)} mean {100*v.mean():+.2f}% median {100*np.median(v):+.2f}% "
          f"rec {up}-{len(v)-up} signp {sign_test(max(up,len(v)-up), len(v)):.4f} "
          f"t {v.mean()/(v.std(ddof=1)/np.sqrt(len(v))):+.2f}")

print("\n--- the specific shape tomorrow: a September Monday that follows quad witching ---")
# third-Friday Septembers: the Monday after the September expiry
post = []
for y in sorted(set(px.index.year)):
    sept = px.index[(px.index.year == y) & (px.index.month == 9)]
    fri = [d for d in sept if d.dayofweek == 4]
    if len(fri) < 3:
        continue
    exp = fri[2]
    nxt = px.index[px.index > exp]
    if len(nxt) and nxt[0].month == 9 and nxt[0].dayofweek == 0:
        post.append(nxt[0])
post = pd.DatetimeIndex(post)
v = ret.reindex(post).dropna()
up = int((v > 0).sum())
print(f"n {len(v)} Mondays after the September expiry, VIX mean {100*v.mean():+.2f}%, "
      f"median {100*np.median(v.values):+.2f}%, rec {up}-{len(v)-up}, "
      f"signp {sign_test(max(up,len(v)-up), len(v)):.4f}")
sv = sret.reindex(post).dropna()
sup = int((sv > 0).sum())
print(f"   S&P those days: mean {100*sv.mean():+.2f}%, rec {sup}-{len(sv)-sup}, "
      f"signp {sign_test(max(sup,len(sv)-sup), len(sv)):.4f}")
print("   years:", [str(d.date()) for d in v.index])
print("   vix moves:", [round(100 * x, 1) for x in v.values])
