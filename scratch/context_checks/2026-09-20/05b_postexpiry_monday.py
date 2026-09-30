"""The Monday after September quad witching, against the right controls.

05 showed the engine's "September Mondays" VIX cell is mostly a generic Monday seam
(+1.87% on other-month Mondays, +0.11% on September non-Mondays). So the question for
tomorrow is whether the post-expiry September Monday is more than just a Monday.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, sign_test, cluster_note  # noqa

px = close_panel(["^VIX", "^GSPC", "IWM", "^RUT", "TLT"]).dropna(subset=["^VIX", "^GSPC"])
px = px[px.index >= "1999-01-01"]
vix, spx = px["^VIX"], px["^GSPC"]
vret = vix / vix.shift(1) - 1.0
sret = spx / spx.shift(1) - 1.0
mon = px.index.dayofweek == 0
sep = px.index.month == 9


def third_friday_monday(month: int):
    out = []
    for y in sorted(set(px.index.year)):
        mm = px.index[(px.index.year == y) & (px.index.month == month)]
        fri = [d for d in mm if d.dayofweek == 4]
        if len(fri) < 3:
            continue
        nxt = px.index[px.index > fri[2]]
        if len(nxt) and nxt[0].dayofweek == 0:
            out.append(nxt[0])
    return pd.DatetimeIndex(out)


def rep(v, label):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    up = int((v > 0).sum())
    t = v.mean() / (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan
    print(f"  {label:34} n {len(v):4}  mean {100*v.mean():+6.2f}%  median {100*np.median(v):+6.2f}%  "
          f"rec {up}-{len(v)-up}  signp {sign_test(max(up, len(v)-up), len(v)):.4f}  t {t:+.2f}")
    return v


print("=== VIX on the Monday after the September expiry, vs controls ===")
sep_mon_post = third_friday_monday(9)
a = rep(vret.reindex(sep_mon_post).dropna().values, "Mon after SEPT expiry")
b = rep(vret[mon & sep].dropna().values, "all September Mondays")
c = rep(vret[mon & ~sep].dropna().values, "Mondays, other months")
d = rep(vret[~mon].dropna().values, "every non-Monday session")


def welch(x, y, lbl):
    se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
    print(f"  Welch t {lbl}: {(x.mean()-y.mean())/se:+.2f}")


welch(a, c, "post-Sept-expiry Mon vs other-month Mon")
welch(a, b, "post-Sept-expiry Mon vs all Sept Mon")
va = vret.reindex(sep_mon_post).dropna()
print(" ", cluster_note(va.index, va.values))

print("\n=== the same Monday for EVERY expiry month (is September special?) ===")
rows = []
for m in range(1, 13):
    idx = third_friday_monday(m)
    v = vret.reindex(idx).dropna().values
    s = sret.reindex(idx).dropna().values
    up, sup = int((v > 0).sum()), int((s > 0).sum())
    rows.append({"month": m, "n": len(v), "vix_mean": round(100 * v.mean(), 2),
                 "vix_med": round(100 * float(np.median(v)), 2), "vix_rec": f"{up}-{len(v)-up}",
                 "spx_mean": round(100 * s.mean(), 3), "spx_rec": f"{sup}-{len(s)-sup}",
                 "spx_signp": round(sign_test(max(sup, len(s) - sup), len(s)), 4)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== the S&P leg: post-September-expiry Monday vs controls ===")
sa = rep(sret.reindex(sep_mon_post).dropna().values, "Mon after SEPT expiry")
sb = rep(sret[mon & sep].dropna().values, "all September Mondays")
sc = rep(sret[mon & ~sep].dropna().values, "Mondays, other months")
sd = rep(sret[~mon & ~sep].dropna().values, "non-Monday, non-September")
welch(sa, sc, "S&P post-Sept-expiry Mon vs other-month Mon")
ssa = sret.reindex(sep_mon_post).dropna()
print(" ", cluster_note(ssa.index, ssa.values))
for cut in ("2011-01-01", "2018-01-01"):
    m = ssa.index < pd.Timestamp(cut)
    for lbl, vv in ((f"pre-{cut[:4]}", ssa.values[m]), (f"{cut[:4]}+", ssa.values[~m])):
        up = int((vv > 0).sum())
        print(f"    S&P {lbl:9} n {len(vv):3} mean {100*vv.mean():+6.2f}% rec {up}-{len(vv)-up} "
              f"signp {sign_test(max(up, len(vv)-up), len(vv)):.4f}")

print("\n=== per-year table for the S&P and VIX on that Monday ===")
tb = pd.DataFrame({"date": [str(x.date()) for x in sep_mon_post],
                   "spx_pct": (100 * sret.reindex(sep_mon_post).values).round(2),
                   "vix_pct": (100 * vret.reindex(sep_mon_post).values).round(1)})
print(tb.to_string(index=False))

print("\n=== does an S&P down Monday need the VIX up? joint record ===")
j = pd.DataFrame({"s": sret.reindex(sep_mon_post), "v": vret.reindex(sep_mon_post)}).dropna()
print(f"  both (S&P down, VIX up): {int(((j.s < 0) & (j.v > 0)).sum())} of {len(j)}")
print(f"  S&P down at all:         {int((j.s < 0).sum())} of {len(j)}")
print(f"  S&P down 1%+:            {int((j.s < -0.01).sum())} of {len(j)}")
