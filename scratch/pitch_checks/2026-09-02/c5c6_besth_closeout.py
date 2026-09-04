"""Close-out: re-run the DECIDING tests for C5 and C6 at each candidate's OWN
best horizon, so a "but the edge lives at h=3 / h=7" rescue has nowhere to go.

C5's horizon scan peaks at h=3 (edge +0.786pp, t 1.924 on 11 episodes) and
C6 (ITA) at h=7 (edge +1.483pp on 6). Both were graded at h=10 / h=5 in the
round-1 scripts. If the reference class, the gate attribution and the beta
decomposition still kill them at their own best horizon, the kill is horizon
independent and no multiplicity argument is even needed.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_lab import *  # noqa
from _rc import cochran, per_name, perm_max_of_n, pooled, welch  # noqa

pd.set_option("display.width", 250)

SPDRS = ["XLK", "XLV", "XLP", "XLU", "XLI", "XLF", "XLY", "XLE", "XLB"]
DEF5 = ["ITA", "RTX", "GD", "LMT", "NOC"]
DEF_EQ = ["RTX", "GD", "LMT", "NOC", "LHX"]
COMPLEXES = {
    "defense    ": DEF_EQ,
    "big banks  ": ["JPM", "BAC", "WFC", "C", "GS"],
    "semis      ": ["INTC", "MU", "AMD", "TXN", "QCOM"],
    "oil majors ": ["XOM", "CVX", "COP", "SLB", "OXY"],
    "staples    ": ["PG", "KO", "PEP", "CL", "WMT"],
    "pharma     ": ["JNJ", "PFE", "MRK", "ABT", "BMY"],
    "utilities  ": ["DUK", "SO", "D", "NEE", "AEP"],
    "rails/machy": ["UNP", "CSX", "NSC", "CAT", "DE"],
    "multi-inds ": ["HON", "GE", "MMM", "EMR", "ITW"],
    "semicap    ": ["AMAT", "KLAC", "LRCX", "ADI", "TSM"],
}
ALL = sorted({t for v in COMPLEXES.values() for t in v} | set(DEF5)
             | set(SPDRS) | {"SPY"})
px = close_panel(ALL)
pxd = {t: px[t] for t in px.columns}
spy_hi = rolling_on_valid(px["SPY"], lambda x: x.rolling(252).max())
SPY_DD = px["SPY"] / spy_hi - 1.0
SPY_NEAR = SPY_DD >= -0.03


def triple(s, k=10):
    return ((pct_rank(s, 5) <= k) & (pct_rank(s, 21) <= k)
            & (pct_rank(s, 63) <= k))


def epi_of(mask, ret, gap):
    v = ret.dropna().index
    return declusters(px.index[mask.fillna(False).values].intersection(v), gap, v)


print("=" * 78)
print("C5 AT ITS OWN BEST HORIZON, h=3")
print("=" * 78)
H, GAP = 3, 5
MAIN5 = (triple(px["XLI"]) & SPY_NEAR).fillna(False)
ret = vehicle_ret(px, [("XLI", 1.0)], H)
valid = ret.dropna().index
r5, r21, r63 = (pct_rank(px["XLI"], n) for n in (5, 21, 63))
rows = []
for lbl, m in [("FULL (r5,r21,r63 <=10 & SPY within 3%)", MAIN5),
               ("drop r5", (r21 <= 10) & (r63 <= 10) & SPY_NEAR),
               ("drop r21", (r5 <= 10) & (r63 <= 10) & SPY_NEAR),
               ("drop r63", (r5 <= 10) & (r21 <= 10) & SPY_NEAR),
               ("drop SPY near-high (bare triple floor)", triple(px["XLI"])),
               ("SPY near-high alone", SPY_NEAR)]:
    e = epi_of(m, ret, GAP)
    r = summarize(ret.loc[e].values, lbl) if len(e) else {"label": lbl, "n": 0}
    if r.get("n"):
        r["edge_pct"] = round(r["mean_pct"] - 100 * ret.loc[valid].mean(), 3)
    rows.append(r)
show(rows, f"C5 gate attribution at h={H}")
full = rows[0]["mean_pct"]
print("  dose of each leg (FULL minus drop-one):")
for r in rows[1:5]:
    print(f"    {r['label']:40s} {full - r['mean_pct']:+7.3f} pp")

e = epi_of(MAIN5, ret, GAP)
v = ret.loc[e].values
ctrl = ret.loc[valid].values
print(f"\n  C5 h=3 episodes N={len(v)} mean {100*v.mean():+.3f}% vs XLI drift "
      f"{100*ctrl.mean():+.3f}% -> edge {100*(v.mean()-ctrl.mean()):+.3f}pp "
      f"welch t {welch(v, ctrl):+.2f}, record {int((v>0).sum())}-"
      f"{int((v<=0).sum())}, sign p {sign_test(int((v>0).sum()), len(v)):.4f}")
print("  " + cluster_note(e, v, k=2))


def mk5(_t, s):
    return (triple(s) & SPY_NEAR.reindex(s.index, fill_value=False)).fillna(False)


pn = per_name(pxd, SPDRS, mk5, H, GAP)
show(pn.sort_values("t_excess", ascending=False), f"C5 reference class, nine SPDRs, h={H}")
co = cochran(pn)
print(f"  Cochran Q {co['Q']:.2f} on {co['df']} df, p {co['p']:.4f}, "
      f"I-squared {co['I2_pct']:.1f}%; fixed-effect COMMON excess "
      f"{co['fe_common_pct']:+.3f}pp (t {co['fe_t']:+.2f})")
ok = pn.dropna(subset=["t_excess"]).sort_values("t_excess", ascending=False)
print(f"  XLI ranks {list(ok['tkr']).index('XLI')+1} of {len(ok)} by excess-t; "
      f"leader {list(ok['tkr'])[0]}")
p = pooled(pxd, SPDRS, mk5, H, GAP, "POOLED nine SPDRs")
print(f"  POOLED N={p['n']} mean {p['mean_pct']:+.3f}% hit {p['hit']:.1f}% t {p['t']:+.2f}")
pm = perm_max_of_n(pxd, SPDRS, mk5, H, GAP, n_perm=1000)
xe, xt = pm["obs"].get("XLI", (np.nan, np.nan))
best = max(pm["obs"].items(), key=lambda kv: kv[1][0])
print(f"  permutation max-of-{pm['n_names']}: best {best[0]} {100*best[1][0]:+.3f}pp, "
      f"XLI {100*xe:+.3f}pp |t| {abs(xt):.2f}; "
      f"P(max excess >= XLI's) {(pm['null_exc'] >= xe).mean():.4f}, "
      f"P(max|t| >= XLI's) {(pm['null_t'] >= abs(xt)).mean():.4f}")

print("\n" + "=" * 78)
print("C6 AT ITS OWN BEST HORIZON, h=7")
print("=" * 78)
H, GAP = 7, 7
Z = {t: zscore(px[t], 10) for t in ALL}


def washout(names, k=4, thr=-1.5):
    cnt = sum((Z[t] <= thr).astype(float).where(Z[t].notna(), np.nan) for t in names)
    return (cnt >= k).fillna(False)


BARE = washout(DEF5)
MAIN6 = (BARE & SPY_NEAR).fillna(False)
ret = vehicle_ret(px, [("ITA", 1.0)], H)
valid = ret.dropna().index
rows = []
for lbl, m in [("FULL 4of5 & SPY within 3%", MAIN6),
               ("BARE 4-of-5 washout", BARE),
               ("5-singles complex (LHX for ITA)", (washout(DEF_EQ) & SPY_NEAR)),
               ("SPY near-high alone", SPY_NEAR)]:
    e = epi_of(m, ret, GAP)
    r = summarize(ret.loc[e].values, lbl) if len(e) else {"label": lbl, "n": 0}
    if r.get("n"):
        r["edge_pct"] = round(r["mean_pct"] - 100 * ret.loc[valid].mean(), 3)
    rows.append(r)
show(rows, f"C6 gate attribution (ITA vehicle) at h={H}")

e = epi_of(MAIN6, ret, GAP)
y = vehicle_ret(px, [("ITA", 1.0)], H)
x = vehicle_ret(px, [("SPY", 1.0)], H)
both = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
b, a = np.polyfit(both["x"], both["y"], 1)
ee = pd.DatetimeIndex(e).intersection(both.index)
resid = both.loc[ee, "y"] - (a + b * both.loc[ee, "x"])
print(f"\n  ITA beta on SPY at h={H} = {b:.3f}; episode ITA "
      f"{100*both.loc[ee,'y'].mean():+.3f}%, SPY over the same windows "
      f"{100*both.loc[ee,'x'].mean():+.3f}%")
print(f"  ALPHA = {100*resid.mean():+.3f}% on N={len(resid)}, t "
      f"{resid.mean()/(resid.std(ddof=1)/np.sqrt(len(resid))):+.2f}, record "
      f"{int((resid>0).sum())}-{int((resid<=0).sum())}")
vv = ret.loc[e].values
print("  " + cluster_note(e, vv, k=2))

rows, keep = [], []
for name, names in COMPLEXES.items():
    m = (washout(names) & SPY_NEAR).fillna(False)
    legs = [(t, 0.2) for t in names]
    r_ = vehicle_ret(px, legs, H)
    v_ = r_.dropna().index
    t_ = px.index[m.values].intersection(v_)
    if len(t_) < 3:
        rows.append({"complex": name, "n_days": len(t_), "n_epi": 0})
        continue
    e_ = declusters(t_, GAP, v_)
    vals = r_.loc[e_].dropna().values
    if len(vals) < 2:
        rows.append({"complex": name, "n_days": len(t_), "n_epi": len(vals)})
        continue
    span = (v_ >= t_[0]) & (v_ <= t_[-1])
    ctrl = r_.loc[v_[span]].dropna().values
    exc = vals.mean() - ctrl.mean()
    se_d = np.sqrt(vals.var(ddof=1) / len(vals) + ctrl.var(ddof=1) / len(ctrl))
    rows.append({"complex": name, "n_days": len(t_), "n_epi": len(vals),
                 "mean_pct": 100 * vals.mean(), "hit": 100 * (vals > 0).mean(),
                 "drift_pct": 100 * ctrl.mean(), "excess_pct": 100 * exc,
                 "t_excess": exc / se_d, "se_d_pct": 100 * se_d})
df = pd.DataFrame(rows).sort_values("t_excess", ascending=False)
show(df.to_dict("records"), f"C6 reference class, 10 complexes, h={H}")
co = cochran(df)
if co:
    print(f"  Cochran Q {co['Q']:.2f} on {co['df']} df, p {co['p']:.4f}, "
          f"I-squared {co['I2_pct']:.1f}%; fixed-effect COMMON excess "
          f"{co['fe_common_pct']:+.3f}pp (t {co['fe_t']:+.2f})")
ok = df.dropna(subset=["t_excess"])
nm = list(ok["complex"])
if any("defense" in n for n in nm):
    print(f"  DEFENSE ranks {[j for j,n in enumerate(nm) if 'defense' in n][0]+1} "
          f"of {len(nm)} by excess-t; leader {nm[0].strip()}")
