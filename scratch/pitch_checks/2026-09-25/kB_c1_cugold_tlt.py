"""kB C1 round 1: copper/gold ratio 21d rank >= 90 with ^TNX at its 252 high ->
SHORT TLT continuation, h=3..10. Pre-specified SHORT. Job: kill it.

Order of attack (registry-driven):
  0. premise print (is the ratio doing what the name says? roll seams?)
  1. ungated parents FIRST: TNX 252 high alone, TLT 252 low alone, the complement
     (TNX high with ratio rank < 50 / ratio falling). The ratio gate must FILTER.
  2. battery on the cell, 2021-22 concentration check first.
  3. secondary: the ratio itself as a trade (long HG=F / short beta-GC=F).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

TK = ["HG=F", "GC=F", "^TNX", "TLT", "GLD", "IEF"]
raw = load_prices(TK)
cl = {t: raw[t]["Close"].dropna() for t in TK}

# ---- ratio on the joint futures calendar
fut = pd.concat([cl["HG=F"], cl["GC=F"]], axis=1, keys=["HG", "GC"]).dropna()
ratio = fut["HG"] / fut["GC"]
rk21 = pct_rank(ratio, 21)
chg21 = ratio / ratio.shift(21) - 1
lvl = ratio.rolling(252).rank(pct=True) * 100
rmax = ratio.rolling(252).max()
tnx = cl["^TNX"]
tnx_hi = tnx >= tnx.rolling(252).max() - 1e-12
tnx_off = tnx - tnx.rolling(252).max()
tlt = cl["TLT"]
tlt_lo = tlt <= tlt.rolling(252).min() + 1e-12
tlt_off = tlt / tlt.rolling(252).min() - 1

print("=== 0. premise / live values ===")
print(f"ratio {ratio.iloc[-1]:.5f} on {ratio.index[-1].date()}  rk21 {rk21.iloc[-1]:.1f}  "
      f"lvl252 {lvl.iloc[-1]:.1f}  21d chg {100*chg21.iloc[-1]:+.2f}%  off 252 max "
      f"{100*(ratio.iloc[-1]/rmax.iloc[-1]-1):+.2f}%")
print(f"TNX {tnx.iloc[-1]:.3f} at-high={bool(tnx_hi.iloc[-1])}  TLT {tlt.iloc[-1]:.2f} at-low={bool(tlt_lo.iloc[-1])}")
print(f"ratio history starts {ratio.index[0].date()}  TLT starts {tlt.index[0].date()}")
# decompose the ratio's 21d move: copper vs gold
hg21 = fut["HG"].iloc[-1] / fut["HG"].iloc[-22] - 1
gc21 = fut["GC"].iloc[-1] / fut["GC"].iloc[-22] - 1
print(f"21d: HG {100*hg21:+.2f}%  GC {100*gc21:+.2f}%  -> ratio move is {100*abs(np.log(1+gc21))/(abs(np.log(1+hg21))+abs(np.log(1+gc21))):.0f}% gold (log share)")
# roll seam check: GC=F vs GLD daily
g = pd.concat([cl["GC=F"].pct_change(), cl["GLD"].pct_change()], axis=1, keys=["GC", "GLD"]).dropna()
bad = g[(g["GC"] - g["GLD"]).abs() > 0.03]
print(f"GC=F vs GLD daily gaps > 3pp: {len(bad)} days; e.g. {[str(d.date()) for d in bad.index[:8]]}")
r1 = ratio.pct_change()
print("largest |ratio 1d| moves:", [(str(d.date()), round(100*v, 2)) for d, v in r1.abs().nlargest(8).items()])

# ---- vehicle panel on TLT calendar
px = pd.DataFrame({"TLT": tlt})
idx = px.index
R = lambda s: s.reindex(idx)
th = R(tnx_hi).fillna(False).astype(bool)
rk = R(rk21)
lv = R(lvl)
ch = R(chg21)
tl = R(tlt_lo).fillna(False).astype(bool)

cell = th & (rk >= 90).fillna(False)
print(f"\ncell days total {int(cell.sum())}; last 15:", [str(d.date()) for d in idx[cell.values][-15:]])
print("TNX-high days in last 30 sessions:", [str(d.date()) for d in idx[-30:][th.values[-30:]]])

# cluster depth of today under declusters(10)
cd = declusters(idx[cell.values], 10, idx)
print("cell episode anchors (gap 10) last 5:", [str(d.date()) for d in cd[-5:]])
cp = declusters(idx[th.values], 10, idx)
print("TNX-high episode anchors (gap 10) last 5:", [str(d.date()) for d in cp[-5:]])

variants = {
    "PARENT: TNX 252 high alone": th,
    "PARENT: TLT 252 low alone": tl,
    "PARENT: TLT low & TNX high": tl & th,
    "COMPLEMENT: TNX high & rk21 < 50": th & (rk < 50).fillna(False),
    "COMPLEMENT: TNX high & ratio 21d chg < 0": th & (ch < 0).fillna(False),
    "COMPLEMENT: TNX high & rk21 < 90": th & (rk < 90).fillna(False),
    "NB rk21 >= 80": th & (rk >= 80).fillna(False),
    "NB rk21 >= 95": th & (rk >= 95).fillna(False),
    "NB lvl252 >= 90 (level form)": th & (lv >= 90).fillna(False),
    "NB rk21>=90 & TLT at low": th & (rk >= 90).fillna(False) & tl,
    "NO-TNX: rk21 >= 90 alone": (rk >= 90).fillna(False),
    "NO-TNX: rk21>=90 & TLT low": (rk >= 90).fillna(False) & tl,
}
for h in (3, 5, 10):
    battery(px, cell, [("TLT", -1.0)], h, "C1 short TLT | TNX 252 hi & Cu/Au rk21>=90",
            6.0, variants=variants, min_gap=10, event_kinds=("nfp", "cpi"))

# ---- 2021-22 concentration check up front (h=5, gap 10)
ret5 = vehicle_ret(px, [("TLT", -1.0)], 5)
e = declusters(idx[(cell & ret5.notna()).values], 10, idx)
v = ret5.loc[e]
yrs = pd.Series(v.values, index=e.year).groupby(level=0).agg(["count", "sum", "mean"])
print("\n=== h=5 cell episodes by year (pct) ===")
print((yrs.assign(sum=100*yrs["sum"], mean=100*yrs["mean"])).round(3).to_string())
x = v[~e.year.isin([2021, 2022])]
print("ex-2021-22:", summarize(x.values, "ex 21-22"))
x = v[e.year != 2022]
print("ex-2022:", summarize(x.values, "ex 22"))
print("record:", int((v > 0).sum()), "-", int((v <= 0).sum()), " sign p", round(sign_test(int((v > 0).sum()), len(v)), 4))

# ---- secondary: the ratio itself as a trade, ex-ante beta
print("\n\n=== secondary: long HG=F / short beta*GC=F continuation after rk21>=90 (& TNX high) ===")
fr = fut.pct_change()
cov = fr["HG"].rolling(252).cov(fr["GC"])
var = fr["GC"].rolling(252).var()
beta = (cov / var)
print(f"ex-ante beta HG on GC today {beta.iloc[-1]:.3f}  (full-sample {fr['HG'].cov(fr['GC'])/fr['GC'].var():.3f})")
fpx = fut.copy()
fidx = fpx.index
th_f = tnx_hi.reindex(fidx).ffill().fillna(False).astype(bool)
for h in (3, 5, 10):
    rh = fwd_lag(fpx["HG"], h) - beta * fwd_lag(fpx["GC"], h)
    rows = []
    for lbl, m in {"rk21>=90 & TNX hi": (rk21 >= 90) & th_f,
                   "rk21>=90 any": (rk21 >= 90),
                   "TNX hi any": th_f,
                   "all days": pd.Series(True, index=fidx)}.items():
        s = fidx[(m.fillna(False) & rh.notna()).values]
        ee = declusters(s, 10, fidx)
        rr = summarize(rh.loc[ee].values, f"h={h} {lbl} eps")
        vv = rh.loc[ee].values
        rr["sign_p"] = sign_test(int((vv > 0).sum()), len(vv)) if len(vv) else np.nan
        rows.append(rr)
    show(rows, f"ratio trade h={h}")
    ee = declusters(fidx[(((rk21 >= 90) & th_f).fillna(False) & rh.notna()).values], 10, fidx)
    show(era_split(ee, rh.loc[ee].values), f"ratio trade h={h} cell era split")
