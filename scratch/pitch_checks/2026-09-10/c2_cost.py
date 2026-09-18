"""c2 (part 1) — UNG ROLL/EXPENSE DRAG, priced against NG=F front. RUN FIRST.

Repo rule 7: roll and expense drag must be priced on any commodity ETF, and
UNG's is materially worse than USO's. NG=F IS in the cache (2000-08-30+), so
the drag is measurable directly rather than inferred.

Three numbers this produces:
  A. UNG CAGR vs NG=F CAGR over the common span -> bp per session of drag
  B. the SEPTEMBER-CONDITIONAL drag (contango is steepest in injection season,
     which is exactly the pitched month -- the drag is not uniform)
  C. the pitched cell measured on BOTH vehicles: if NG=F pays and UNG does not,
     the kill is vehicle drag; if UNG itself pays, the drag is already inside
     the realised UNG series and the cost kill FAILS.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = close_panel(["UNG", "NG=F", "USO", "CL=F", "SPY"])
ung = px["UNG"].dropna()
ng = px["NG=F"].dropna()
uso = px["USO"].dropna()
cl = px["CL=F"].dropna()

# ------------------------------------------------------------------ A. drag
def cagr(s):
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1.0, yrs

common = ung.index.intersection(ng.index)
u, n_ = ung.reindex(common).dropna(), ng.reindex(common).dropna()
common = u.index.intersection(n_.index)
u, n_ = u.reindex(common), n_.reindex(common)

cu, yu = cagr(u)
cn, yn = cagr(n_)
print("=" * 78)
print("A. UNG vs NG=F front, common span "
      f"{common[0].date()} .. {common[-1].date()}  ({yu:.1f} yrs, {len(common)} sessions)")
print(f"   UNG  CAGR {100*cu:+.2f}%/yr   NG=F CAGR {100*cn:+.2f}%/yr")
gap_yr = cu - cn
sess_per_yr = len(common) / yu
print(f"   annual drag = {100*gap_yr:+.2f}%/yr  over {sess_per_yr:.1f} sessions/yr")
per_sess = (1 + gap_yr) ** (1 / sess_per_yr) - 1
print(f"   ** per-session drag = {1e4*per_sess:+.2f} bp **")
for h in (3, 5, 10, 21):
    print(f"      h={h:2d} sessions -> {1e4*((1+per_sess)**h - 1):+.1f} bp")

# the USO benchmark the repo already measured, reproduced for calibration
cmn2 = uso.index.intersection(cl.index)
uu, cc = uso.reindex(cmn2).dropna(), cl.reindex(cmn2).dropna()
cmn2 = uu.index.intersection(cc.index)
uu, cc = uu.reindex(cmn2), cc.reindex(cmn2)
cuu, yuu = cagr(uu)
ccc, _ = cagr(cc)
ps_uso = (1 + (cuu - ccc)) ** (1 / (len(cmn2) / yuu)) - 1
print(f"\n   calibration (repo's own USO number): USO {100*cuu:+.2f}%/yr vs "
      f"CL=F {100*ccc:+.2f}%/yr over {yuu:.1f}y "
      f"-> {1e4*((1+ps_uso)**3 - 1):+.1f} bp per 3-session hold "
      f"(repo recorded -8.8 bp)")
print(f"   UNG is {per_sess/ps_uso:.2f}x USO's per-session drag")

# ------------------------------- B. drag by calendar month (SEPTEMBER matters)
ru = u.pct_change()
rn = n_.pct_change()
d = (ru - rn).dropna()
by_m = d.groupby(d.index.month).agg(["mean", "count"])
by_m["bp_per_session"] = 1e4 * by_m["mean"]
print("\nB. tracking difference (UNG daily ret - NG=F daily ret) by month")
print(by_m[["bp_per_session", "count"]].round(2).to_string())
sep = d[d.index.month == 9]
print(f"   ** September mean drag {1e4*sep.mean():+.2f} bp/session, "
      f"n={len(sep)} sessions **")
for h in (3, 5, 10, 21):
    print(f"      Sept h={h:2d} -> {1e4*h*sep.mean():+.1f} bp cumulative")

# --------------------------------------- C. the pitched cell on BOTH vehicles
# trigger: month == September AND close within 6% of trailing-252 low
def make_mask(s, month=9, pct=0.06):
    lo = rolling_on_valid(s, lambda x: x.rolling(252).min())
    near = (s / lo - 1.0) <= pct
    return (near & (s.index.month == month)).fillna(False)

print("\n" + "=" * 78)
print("C. the pitched cell, gross, on each vehicle "
      "(Sept AND within 6% of trailing-252 low)")
for name, s in (("UNG", ung), ("NG=F", ng)):
    m = make_mask(s)
    sig = s.index[m.values]
    print(f"\n  --- {name}: {int(m.sum())} trigger days, "
          f"{len(set(sig.year))} distinct Septembers: {sorted(set(sig.year))}")
    for h in (3, 5, 10, 21):
        r = fwd_lag(s, h, 1)
        t = pd.DatetimeIndex(sig).intersection(r.dropna().index)
        epi = declusters(t, max(h, 5), s.index)
        if len(epi) == 0:
            print(f"     h={h}: no episodes")
            continue
        v = r.loc[epi].values
        base = r.dropna()
        sm = summarize(v, f"h={h}")
        gross_bp = 100 * sm["mean_pct"]
        print(f"     h={h:2d}  N_epi={sm['n']:2d}  mean {sm['mean_pct']:+.2f}% "
              f"({gross_bp:+.0f} bp)  med {sm['median_pct']:+.2f}%  "
              f"hit {sm['hit']:.0f}%  t {sm['t']:+.2f}  "
              f"worst {sm['worst_pct']:+.1f}%  | all-days base "
              f"{100*base.mean():+.3f}%  sign p "
              f"{sign_test(int((v>0).sum()), len(v)):.4f}")

print("\n" + "=" * 78)
print("VERDICT INPUT: UNG cell is measured on the REALISED UNG series, which")
print("already contains the roll+expense drag above. The kill is 'vehicle")
print("drag' only if NG=F pays and UNG does not.")
