"""C6 round 2b -- the two things that decide it.

(1) WHY the 126d-lookback neighbour collapses: are its extra episodes actually
    parabolic thrusts, or does a short reference window rank a modest 21d move
    at 99? If the extras are not thrusts, the collapse is explained, not
    fragility.
(2) INDEPENDENT-INSTRUMENT replication: the identical rule (own 21d return at a
    252d PIT rank >= 99 AND own 1d <= -2%) run on gold-miner single names with
    longer or separate history (NEM/AEM/AU/KGC 2000+, GDXJ 2009+, plus SLV as
    the non-miner metal). Six GDX episodes is the whole sample; if the rule is
    real it should show up on instruments GDX did not select.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np, pandas as pd

NAMES = ["GDX", "GDXJ", "NEM", "AEM", "AU", "KGC", "GOLD_PROXY_SKIP", "SLV", "GLD"]
NAMES = [n for n in NAMES if n != "GOLD_PROXY_SKIP"]
px = close_panel(NAMES)

def cell(s, lookback=252, rank_thr=99.0, dn=-0.02):
    r21 = rolling_on_valid(s, lambda x: x / x.shift(21) - 1.0)
    rk = rolling_on_valid(r21, lambda x: x.rolling(lookback).rank(pct=True) * 100.0)
    r1 = rolling_on_valid(s, lambda x: x / x.shift(1) - 1.0)
    return (rk >= rank_thr) & (r1 <= dn), r21, r1

# ---- (1) the 126d diagnostic
g = px["GDX"]
m252, r21, r1 = cell(g, 252)
m126, _, _ = cell(g, 126)
e252 = declusters(px.index[m252.fillna(False)], 10, px.index)
e126 = declusters(px.index[m126.fillna(False)], 10, px.index)
print("=== 126d vs 252d lookback: what are the EXTRA episodes? ===")
print("252d episodes:", [str(d.date()) for d in e252])
print("126d episodes:", [str(d.date()) for d in e126])
extra = pd.DatetimeIndex(sorted(set(e126) - set(e252)))
h = 5
ret = fwd_lag(g, h, 1)
print("\nEXTRA-only episodes (in 126d, not 252d):")
for d in extra:
    print(f"  {d.date()}  r21 {100*r21.loc[d]:+6.1f}%  1d {100*r1.loc[d]:+6.2f}%  "
          f"fwd5 {100*ret.loc[d]:+6.2f}%")
print(f"\nr21 magnitude: 252d cell median {100*r21.loc[e252].median():.1f}% "
      f"(min {100*r21.loc[e252].min():.1f}%)  vs extras median "
      f"{100*r21.loc[extra].median():.1f}% (min {100*r21.loc[extra].min():.1f}%)")
print("TODAY r21 = %.1f%%" % (100 * r21.iloc[-1]))

# ---- (2) independent-instrument replication
print("\n\n=== independent replication: same rule on each instrument's OWN series ===")
rows = []
for t in NAMES:
    s = px[t].dropna()
    if len(s) < 600:
        continue
    m, rr21, rr1 = cell(s, 252)
    m = m.fillna(False)
    for hh in (5, 10):
        r = fwd_lag(s, hh, 1)
        ok = r.notna()
        sig = s.index[m.reindex(s.index, fill_value=False).values & ok.values]
        if len(sig) == 0:
            rows.append({"tkr": t, "h": hh, "n_days": 0}); continue
        e = declusters(sig, 10, s.index)
        v = r.loc[e].values
        base = r.dropna()
        w = int((v > 0).sum())
        rows.append({
            "tkr": t, "h": hh, "n_days": len(sig), "n_epi": len(e),
            "mean_pct": round(100 * v.mean(), 3),
            "ctl_all_pct": round(100 * base.mean(), 3),
            "edge_pct": round(100 * (v.mean() - base.mean()), 3),
            "hit": round(100 * (v > 0).mean(), 1),
            "record": f"{w}-{len(v)-w}",
            "sign_p": round(sign_test(w, len(v)), 4),
            "worst_pct": round(100 * v.min(), 2),
        })
print(pd.DataFrame(rows).to_string(index=False))

# ---- pooled across the 5 single-name miners (GDX excluded: it selected the cell)
print("\n=== POOLED single-name miners (GDX/GDXJ/GLD/SLV EXCLUDED) ===")
for hh in (5, 10):
    allv, dts = [], []
    for t in ("NEM", "AEM", "AU", "KGC"):
        s = px[t].dropna()
        m, _, _ = cell(s, 252)
        r = fwd_lag(s, hh, 1)
        sig = s.index[m.fillna(False).reindex(s.index, fill_value=False).values
                      & r.notna().values]
        e = declusters(sig, 10, s.index)
        allv += list(r.loc[e].values); dts += list(e)
    v = np.array(allv)
    w = int((v > 0).sum())
    base = []
    for t in ("NEM", "AEM", "AU", "KGC"):
        base += list(fwd_lag(px[t].dropna(), hh, 1).dropna().values)
    b = np.array(base)
    print(f"h={hh}: N={len(v)} mean {100*v.mean():+.3f}%  vs all-days "
          f"{100*b.mean():+.3f}%  edge {100*(v.mean()-b.mean()):+.3f}%  "
          f"hit {100*(v>0).mean():.1f}%  record {w}-{len(v)-w}  "
          f"sign p (vs base rate {100*(b>0).mean():.1f}%) = "
          f"{sign_test(w, len(v), float((b>0).mean())):.4f}  "
          f"worst {100*v.min():+.2f}%  bootstrap P(mean<=0) {bootstrap_p_le0(v):.3f}")
    yrs = pd.DatetimeIndex(dts).year
    print("   years:", dict(pd.Series(1, index=yrs).groupby(level=0).sum()))
