"""r3 - RED TEAM attack 3: does this post-hoc sign flip carry the two failure
modes that killed the prior two inversions?

  (i) an ERA FENCE around one macro episode
 (ii) DEFINITION FRAGILITY

Plus: the "positive in 9 of 9 years" claim is the strongest within-IHI number
in the case, so it is priced against the cross-section too -- how many of the
27 sector ETFs are ALSO positive in every year their identical trigger fires?
If several are, year-consistency at N=16 is a noise artefact, not evidence.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

H = 5
TK = ["XLV", "XLK", "XLE", "XLF", "XLI", "XLB", "XLP", "XLU", "XLY",
      "SMH", "XBI", "KRE", "IHI", "VNQ", "XOP", "OIH", "GDX", "XME",
      "ITA", "ITB", "IYR", "IYT", "XRT", "XHB", "IBB", "GDXJ", "COPX"]
px_map = load_prices(TK)
c = px_map["IHI"]["Close"].dropna()
r21 = pct_rank(c, 21)
dd = c / c.rolling(252).max() - 1.0
m = ((r21 >= 99) & (dd <= -0.10)).fillna(False)
ret = fwd_lag(c, H)
trig = c.index[m.values & ret.notna().values]
epi = declusters(trig, 5, c.index)
epi = epi[ret.reindex(epi).notna().values]
v = ret.loc[epi].values
span = (c.index >= trig[0]) & (c.index <= trig[-1]) & ret.notna().values
ctrl = ret[span].values
base = float((ctrl > 0).mean())
n = len(v)

print("=== (i) ERA FENCE: is this a GFC-recovery-in-medtech artefact? ===")
yr = pd.DatetimeIndex(epi).year
tot = v.sum()
for lbl, sel in [("2008-2012 (GFC recovery)", (yr >= 2008) & (yr <= 2012)),
                 ("2013-2019 (nothing fires)", (yr >= 2013) & (yr <= 2019)),
                 ("2020-2026 (modern)", yr >= 2020),
                 ("pre-2018", yr < 2018), ("2018+", yr >= 2018)]:
    if sel.sum() == 0:
        print(f"  {lbl:28s} N=0")
        continue
    vv = v[sel]
    w = int((vv > 0).sum())
    print(f"  {lbl:28s} N={len(vv):2d} mean {100*vv.mean():+.3f}% "
          f"hit {100*(vv>0).mean():5.1f}% record {w}-{len(vv)-w} "
          f"share_of_total_R {100*vv.sum()/tot:5.1f}%  "
          f"sign_p_base {sign_test(w, len(vv), base):.4f}")
print(f"  -> 2008-2012 holds {int(((yr>=2008)&(yr<=2012)).sum())}/{n} episodes "
      f"and {100*v[(yr>=2008)&(yr<=2012)].sum()/tot:.0f}% of total R. "
      f"A 12-YEAR HOLE 2013-2019 with ZERO firings.")
print(f"  IHI's own 2008-2012 drift over the same h={H} span: "
      f"{100*ret[(c.index.year>=2008)&(c.index.year<=2012)&ret.notna()].mean():+.3f}% "
      f"vs full-span {100*ctrl.mean():+.3f}%")

print("\n=== LEAVE-ONE-YEAR-OUT on the 9 firing years ===")
ys = sorted(set(yr))
rows = []
for y in ys:
    keep = yr != y
    vv = v[keep]
    w = int((vv > 0).sum())
    rows.append({"drop_year": y, "n_dropped": int((yr == y).sum()), "n": len(vv),
                 "mean_pct": round(100*vv.mean(), 3),
                 "excess_pp": round(100*(vv.mean()-ctrl.mean()), 3),
                 "hit": round(100*(vv > 0).mean(), 1),
                 "sign_p_base": round(sign_test(w, len(vv), base), 4),
                 "boot_P_le0": round(bootstrap_p_le0(vv), 4)})
ld = pd.DataFrame(rows)
print(ld.to_string(index=False))
print(f"  LOYO floor: min mean {ld.mean_pct.min():+.3f}% (drop "
      f"{int(ld.loc[ld.mean_pct.idxmin(),'drop_year'])}), "
      f"max sign_p_base {ld.sign_p_base.max():.4f}")

print("\n=== drop-best-episode / drop-best-year ladder ===")
o = np.argsort(-v)
lad = [100*v.mean()]
for k in (1, 2, 3, 4):
    lad.append(100*np.delete(v, o[:k]).mean())
print(f"  drop-k-best episodes: " + " -> ".join(f"{x:+.3f}%" for x in lad))
bys = pd.Series(v, index=yr).groupby(level=0).mean().sort_values(ascending=False)
print(f"  best years by mean: {dict((int(k), round(100*x,2)) for k,x in bys.head(3).items())}")
for k in (1, 2):
    dropy = set(bys.head(k).index)
    vv = v[~np.isin(yr, list(dropy))]
    print(f"  drop-{k}-best-year(s) {sorted(dropy)}: N={len(vv)} "
          f"mean {100*vv.mean():+.3f}% excess {100*(vv.mean()-ctrl.mean()):+.3f}pp "
          f"hit {100*(vv>0).mean():.1f}%")

print("\n=== (ii) DEFINITION FRAGILITY ===")
rows = []


def cellstat(mask, ret_):
    t = c.index[mask.values & ret_.notna().values]
    if len(t) == 0:
        return None
    e = declusters(t, 5, c.index)
    e = e[ret_.reindex(e).notna().values]
    if len(e) == 0:
        return None
    vv = ret_.loc[e].values
    sp = (c.index >= t[0]) & (c.index <= t[-1]) & ret_.notna().values
    return {"n": len(vv), "mean_pct": round(100*vv.mean(), 3),
            "hit": round(100*(vv > 0).mean(), 1),
            "excess_pp": round(100*(vv.mean()-ret_[sp].mean()), 3),
            "yrs_pos": f"{int((pd.Series(vv, index=pd.DatetimeIndex(e).year).groupby(level=0).mean()>0).sum())}/"
                       f"{pd.DatetimeIndex(e).year.nunique()}"}


# rank lookback for the percentile itself
for lb in (126, 252, 504):
    rr = pct_rank(c, 21, lookback=lb)
    s = cellstat(((rr >= 99) & (dd <= -0.10)).fillna(False), ret)
    rows.append({"knob": f"rank lookback {lb}d", **(s or {})})
# return window inside the rank
for w in (10, 15, 21, 26, 42):
    rr = pct_rank(c, w)
    s = cellstat(((rr >= 99) & (dd <= -0.10)).fillna(False), ret)
    rows.append({"knob": f"thrust window {w}d", **(s or {})})
# drawdown reference window
for dw in (200, 252, 504):
    d2 = c / c.rolling(dw).max() - 1.0
    s = cellstat(((r21 >= 99) & (d2 <= -0.10)).fillna(False), ret)
    rows.append({"knob": f"52wh window {dw}d", **(s or {})})
# rank threshold in raw-count terms rather than percentile
for q in (98, 98.5, 99, 99.5, 100):
    s = cellstat(((r21 >= q) & (dd <= -0.10)).fillna(False), ret)
    rows.append({"knob": f"r21 >= {q}", **(s or {})})
# MAGNITUDE gate instead of a rank gate (registry trap)
r21raw = c.pct_change(21)
for g in (0.06, 0.08, 0.10, 0.1394):
    s = cellstat(((r21raw >= g) & (dd <= -0.10)).fillna(False), ret)
    rows.append({"knob": f"MAGNITUDE ret21 >= {100*g:.1f}%", **(s or {})})
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== 'positive in 9 of 9 years' priced against the cross-section ===")
out = []
for t in TK:
    cc = px_map[t]["Close"].dropna()
    rr = pct_rank(cc, 21)
    d2 = cc / cc.rolling(252).max() - 1.0
    mm = ((rr >= 99) & (d2 <= -0.10)).fillna(False)
    rt = fwd_lag(cc, H)
    tt = cc.index[mm.values & rt.notna().values]
    if len(tt) == 0:
        continue
    e = declusters(tt, 5, cc.index)
    e = e[rt.reindex(e).notna().values]
    vv = rt.loc[e].values
    ybm = pd.Series(vv, index=pd.DatetimeIndex(e).year).groupby(level=0).mean()
    out.append({"ticker": t, "n_epi": len(vv), "n_yrs": len(ybm),
                "yrs_pos": int((ybm > 0).sum()),
                "all_yrs_pos": bool((ybm > 0).all()),
                "mean_pct": round(100*vv.mean(), 3)})
od = pd.DataFrame(out).sort_values(["all_yrs_pos", "yrs_pos"], ascending=False)
print(od.to_string(index=False))
nall = int(od.all_yrs_pos.sum())
print(f"  tickers positive in EVERY firing year: {nall} of {len(od)} "
      f"({100*nall/len(od):.0f}%)  -> IHI is {'NOT ' if nall > 1 else ''}unique on this")
