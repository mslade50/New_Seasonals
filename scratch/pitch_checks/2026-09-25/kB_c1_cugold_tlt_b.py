"""kB C1 round 2 (h=10 is the only horizon with a positive episode mean in round 1;
h=5 was 3-10). Attack the h=10 cell: concentration, neighbour grid, gate
attribution (filter vs re-anchor against the TNX-high parent and the TLT-low
parent), era/regime (hiking proxy, midterm), and the LIVE reading: is today's
ratio move copper-led (the growth read the mechanism names) or gold-led?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("future.no_silent_downcasting", True)
TK = ["HG=F", "GC=F", "^TNX", "^IRX", "TLT"]
raw = load_prices(TK)
cl = {t: raw[t]["Close"].dropna() for t in TK}
fut = pd.concat([cl["HG=F"], cl["GC=F"]], axis=1, keys=["HG", "GC"]).dropna()
ratio = fut["HG"] / fut["GC"]
tnx = cl["^TNX"]
tnx_hi = tnx >= tnx.rolling(252).max() - 1e-12
tlt = cl["TLT"]
tlt_lo = tlt <= tlt.rolling(252).min() + 1e-12
px = pd.DataFrame({"TLT": tlt})
idx = px.index
R = lambda s: s.reindex(idx)
th = R(tnx_hi).fillna(False).astype(bool)
tl = R(tlt_lo).fillna(False).astype(bool)
rk = {lb: R(pct_rank(ratio, lb)) for lb in (10, 21, 42, 63)}
cell = th & (rk[21] >= 90).fillna(False)
H = 10
ret = vehicle_ret(px, [("TLT", -1.0)], H)
valid = ret.notna()


def eps(mask: pd.Series) -> pd.DatetimeIndex:
    return declusters(idx[(mask & valid).values], 10, idx)


e = eps(cell)
v = ret.loc[e]
print(f"=== (a) concentration, h={H}, N={len(e)} ===")
print(cluster_note(e, v.values))
srt = v.sort_values(ascending=False)
print("episodes sorted:", [(str(d.date()), round(100 * x, 2)) for d, x in srt.items()])
print(f"drop-best-2 mean {100*srt.iloc[2:].mean():+.3f}% (N={len(srt)-2});  drop-best-1 {100*srt.iloc[1:].mean():+.3f}%")
loyo = {y: 100 * v[e.year != y].mean() for y in sorted(set(e.year))}
print("LOYO means:", {k: round(x, 3) for k, x in loyo.items()})
own = ret[valid].mean()
print(f"own drift (short TLT, h={H}) all days {100*own:+.3f}%")
w = int((v > 0).sum())
print(f"record {w}-{len(v)-w} sign p {sign_test(w, len(v)):.4f}; vs own-drift hit rate p "
      f"{sign_test(w, len(v), float((ret[valid] > 0).mean())):.4f}")

print("\n=== (b) neighbour grid: lookback x rank threshold (episode means, h=10 / h=5 / h=3) ===")
rows = []
for lb in (10, 21, 42, 63):
    for thr in (80, 85, 90, 95):
        m = th & (rk[lb] >= thr).fillna(False)
        out = {"lb": lb, "thr": thr}
        for hh in (3, 5, 10):
            rr = vehicle_ret(px, [("TLT", -1.0)], hh)
            ee = declusters(idx[(m & rr.notna()).values], 10, idx)
            out[f"n"] = len(ee)
            out[f"h{hh}"] = round(100 * rr.loc[ee].mean(), 3)
        rows.append(out)
g = pd.DataFrame(rows)
print(g.to_string(index=False))
print(f"h10 positive cells {int((g.h10 > 0).sum())}/16; h5 positive {int((g.h5 > 0).sum())}/16; h3 positive {int((g.h3 > 0).sum())}/16")
# TNX neighbours: within 1% / 2% of 252 high, 126 high
for lbl, tm in {"TNX 126 high": tnx >= tnx.rolling(126).max() - 1e-12,
                "TNX within 1% of 252 hi": tnx >= 0.99 * tnx.rolling(252).max(),
                "TNX within 2% of 252 hi": tnx >= 0.98 * tnx.rolling(252).max()}.items():
    m = R(tm).fillna(False).astype(bool) & (rk[21] >= 90).fillna(False)
    ee = eps(m)
    print(f"  {lbl} & rk21>=90: N={len(ee)} h10 {100*ret.loc[ee].mean():+.3f}%  hit {100*(ret.loc[ee]>0).mean():.0f}%")

print("\n=== (d) gate attribution at h=10 ===")
rows = []
for lbl, m in {"CELL": cell, "TNX high (parent)": th, "TNX high & rk21<90": th & (rk[21] < 90).fillna(False),
               "TNX high & rk21<50": th & (rk[21] < 50).fillna(False),
               "TLT low (parent)": tl, "TLT low & TNX high": tl & th,
               "TLT low & TNX high & rk21>=90": tl & cell,
               "TLT low & TNX high & rk21<90": tl & th & (rk[21] < 90).fillna(False)}.items():
    ee = eps(m)
    rr = summarize(ret.loc[ee].values, lbl)
    rows.append(rr)
show(rows, "gate table h=10 (episodes, gap 10)")
pe = eps(th)
ce = eps(cell)
pm = pd.Series(False, index=idx); pm.loc[pe] = True
cm = pd.Series(False, index=idx); cm.loc[ce] = True
fv = filter_vs_reanchor(ret, pm, cm, idx, window_td=21, label="TNX-high parent -> +rk21>=90")
if fv["n_matched"]:
    rn = reanchor_null(ret, fv["deleted_dates"] + [a for a, _, _ in fv["pairs"]], fv["shifts"], idx,
                       float(ret.loc[ce].mean()))
    print("  reanchor_null:", {k: (round(x, 4) if isinstance(x, float) else x) for k, x in rn.items()})

print("\n=== (c) era / regime split at h=10 ===")
show(era_split(e, v.values), "pre/post 2018")
ex = v[~e.year.isin([2021, 2022])]
print("ex-2021-22:", {k: (round(x, 3) if isinstance(x, float) else x) for k, x in summarize(ex.values, "ex21-22").items()})
irx = cl["^IRX"]
hike = R((irx - irx.shift(63)) > 0.25).fillna(False).astype(bool)
mid = pd.Series(idx.year % 4 == 2, index=idx)
show([summarize(v[hike.loc[e].values].values, "hiking proxy (^IRX +25bp/63d)"),
      summarize(v[~hike.loc[e].values].values, "not hiking"),
      summarize(v[mid.loc[e].values].values, "midterm year"),
      summarize(v[~mid.loc[e].values].values, "other years")], "regime split, cell episodes")
# regime split of the PARENT too (does the mechanism's regime carry the parent anyway?)
pv = ret.loc[pe]
show([summarize(pv[hike.loc[pe].values].values, "PARENT TNX-hi, hiking"),
      summarize(pv[~hike.loc[pe].values].values, "PARENT TNX-hi, not hiking")], "parent regime split")

print("\n=== live-reading check: what drove the ratio's 21d move in each cell episode? ===")
hg21 = np.log(fut["HG"] / fut["HG"].shift(21))
gc21 = np.log(fut["GC"] / fut["GC"].shift(21))
rows = []
for d in list(e) + [idx[-1]]:
    dd = fut.index[fut.index <= d][-1]
    a, b = hg21.loc[dd], gc21.loc[dd]
    rows.append({"date": str(d.date()), "HG21_pct": round(100 * a, 2), "GC21_pct": round(100 * b, 2),
                 "gold_share_pct": round(100 * max(-b, 0) / (max(a, 0) + max(-b, 0) + 1e-12), 0),
                 "ret_h10_pct": round(100 * ret.get(d, np.nan), 3)})
print(pd.DataFrame(rows).to_string(index=False))
# gold-led cells across the whole day-level history: gold falling >= 5% in 21d with TNX high
gl = th & R(gc21 <= np.log(0.95)).fillna(False).astype(bool)
cu = th & R(hg21 >= np.log(1.05)).fillna(False).astype(bool)
for lbl, m in {"TNX hi & gold 21d <= -5% (gold-led)": gl, "TNX hi & copper 21d >= +5% (copper-led)": cu,
               "TNX hi & gold<=-5% & rk21>=90": gl & (rk[21] >= 90).fillna(False)}.items():
    ee = eps(m)
    print(f"  {lbl}: N={len(ee)} h10 {100*ret.loc[ee].mean():+.3f}% hit {100*(ret.loc[ee]>0).mean():.0f}%  "
          f"h5 {100*vehicle_ret(px, [('TLT', -1.0)], 5).loc[ee].mean():+.3f}%")
