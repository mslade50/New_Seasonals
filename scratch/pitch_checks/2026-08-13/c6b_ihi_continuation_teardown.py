"""c6b - round 2 on the C6 SIGN FLIP.

C6 was pitched as a FADE. The fade is wrong-signed (short IHI -0.95% h=3,
-1.50% h=5, negative in 9 of 9 years). The CONTINUATION direction is what
prices. This is a post-hoc sign flip on a pitched candidate, which the
registry names as a trap, so it gets held to a higher bar than a
pre-specified idea: full round 2 plus a search-cost accounting.

Round 2: decluster + concentration + TODAY'S depth, definition neighbours,
era / midterm / SPY-near-high, gate attribution, and the correlation against
today's LIVE pitch legs (DX to 08-14, GDX to 08-17).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

TK = ["IHI", "XLV", "SPY", "GDX", "GLD"]
px = close_panel(TK).dropna()
raw = load_prices(TK)
ihi, xlv, spy, gdx = (raw[t]["Close"] for t in ["IHI", "XLV", "SPY", "GDX"])

al = pd.concat([ihi.pct_change(), xlv.pct_change(), spy.pct_change(),
                gdx.pct_change()], axis=1).dropna()
al.columns = ["ihi", "xlv", "spy", "gdx"]
b_xlv = np.polyfit(al["xlv"], al["ihi"], 1)[0]

r21 = pct_rank(ihi, 21)
dd = ihi / ihi.rolling(252).max() - 1.0
m = ((r21 >= 99) & (dd <= -0.10)).reindex(px.index).fillna(False)
trig = px.index[m.values]

print("=== 1. episodes + concentration + TODAY'S DEPTH ===")
for h in (3, 5):
    r = fwd_lag(ihi, h).reindex(px.index)
    epi = declusters(trig, 5, px.index)
    epi = epi[r.reindex(epi).notna().values]
    v = r.loc[epi].values
    print(f"\nh={h}: N={len(epi)} mean {100*v.mean():+.3f}% hit {100*(v>0).mean():.1f}% "
          f"record {int((v>0).sum())}-{int((v<=0).sum())} sign p {sign_test(int((v>0).sum()), len(v)):.4f} "
          f"(vs IHI own hit {100*(r.dropna()>0).mean():.1f}% -> "
          f"p {sign_test(int((v>0).sum()), len(v), (r.dropna()>0).mean()):.4f})")
    print("  ", cluster_note(epi, v))
    o = np.argsort(-v)
    print(f"   drop-best ladder: {100*v.mean():+.3f}% -> {100*np.delete(v, o[0]).mean():+.3f}% "
          f"-> {100*np.delete(v, o[:2]).mean():+.3f}% -> {100*np.delete(v, o[:3]).mean():+.3f}%")
    yr = pd.Series(v, index=epi.year).groupby(level=0).agg(["mean", "count"])
    print("   by year:", {int(k): (round(100 * a, 2), int(b))
                          for k, (a, b) in yr.iterrows()})
    print(f"   bootstrap P(mean<=0) = {bootstrap_p_le0(v):.4f}")

# depth
mv = m.values
dep = np.zeros(len(mv), int)
for i in range(len(mv)):
    dep[i] = dep[i - 1] + 1 if mv[i] and i > 0 else (1 if mv[i] else 0)
dser = pd.Series(dep, index=px.index)
print(f"\n=== depth: today = {int(dser.iloc[-1])}, population median "
      f"{float(dser.loc[trig].median()):.1f} ===")
for h in (3, 5, 10):
    r = fwd_lag(ihi, h).reindex(px.index)
    rows = []
    for lo, hi in [(1, 1), (2, 2), (3, 3), (4, 99)]:
        sel = trig[(dser.loc[trig] >= lo) & (dser.loc[trig] <= hi)]
        rows.append(summarize(r.loc[sel].values, f"h={h} depth {lo}-{hi} (N={len(sel)})"))
    show(rows)

print("\n=== 2. definition neighbours (long IHI, episodes) ===")
rows = []
for rq in (95, 97, 99, 100):
    for dm in (-0.05, -0.08, -0.10, -0.15):
        mm = ((r21 >= rq) & (dd <= dm)).reindex(px.index).fillna(False)
        t = px.index[mm.values]
        for h in (3, 5):
            r = fwd_lag(ihi, h).reindex(px.index)
            e = declusters(t, 5, px.index)
            e = e[r.reindex(e).notna().values]
            if len(e):
                v = r.loc[e].values
                rows.append({"h": h, "r21>=": rq, "dd<=": dm, "n_epi": len(e),
                             "mean_pct": round(100 * v.mean(), 3),
                             "hit": round(100 * (v > 0).mean(), 1)})
print(pd.DataFrame(rows).pivot_table(index=["r21>=", "dd<="],
                                     columns="h",
                                     values=["mean_pct", "hit", "n_epi"]).to_string())
# rank lookback neighbours
print("\n  lookback neighbours (r21 window 10/21/42, 252d rank), h=5:")
for w in (10, 21, 42):
    rr = pct_rank(ihi, w)
    mm = ((rr >= 99) & (dd <= -0.10)).reindex(px.index).fillna(False)
    r = fwd_lag(ihi, 5).reindex(px.index)
    e = declusters(px.index[mm.values], 5, px.index)
    e = e[r.reindex(e).notna().values]
    if len(e):
        v = r.loc[e].values
        print(f"   rank window {w}d: N={len(e)} mean {100*v.mean():+.3f}% "
              f"hit {100*(v>0).mean():.1f}%  (today's value {rr.iloc[-1]:.1f})")

print("\n=== 3. era / midterm / SPY-near-high, long IHI + XLV residual, h=5 ===")
h = 5
r_i, r_v, r_s = (fwd_lag(x, h).reindex(px.index) for x in (ihi, xlv, spy))
res = r_i - b_xlv * r_v
epi = declusters(trig, 5, px.index)
epi = epi[res.reindex(epi).notna().values]
mt = np.array([d.year % 4 == 2 for d in epi])
spyhi = ((spy / spy.rolling(252).max() - 1) > -0.01).reindex(px.index).fillna(False)
g = spyhi.loc[epi].values
show([summarize(r_i.loc[epi].values, "long IHI, all episodes"),
      summarize(res.loc[epi].values, f"IHI resid vs XLV (b={b_xlv:.2f})"),
      summarize(res[res.notna()].values, "  that residual, all days"),
      summarize(r_i.loc[epi[mt]].values, f"long IHI midterm (N={int(mt.sum())})"),
      summarize(res.loc[epi[mt]].values, "resid midterm"),
      summarize(r_i.loc[epi[g]].values, f"long IHI SPY near high (N={int(g.sum())})"),
      summarize(res.loc[epi[g]].values, "resid SPY near high")])
show(era_split(epi, r_i.loc[epi].values), "long IHI era")
show(era_split(epi, res.loc[epi].values), "XLV residual era")

print("\n=== 4. gate attribution: each gate alone (long IHI, h=5) ===")
for lbl, mm in [("r21>=99 ONLY", (r21 >= 99)),
                ("dd<=-10% ONLY", (dd <= -0.10)),
                ("BOTH", (r21 >= 99) & (dd <= -0.10))]:
    mm = mm.reindex(px.index).fillna(False)
    e = declusters(px.index[mm.values], 5, px.index)
    e = e[r_i.reindex(e).notna().values]
    v = r_i.loc[e].values
    vr = res.loc[e].values
    print(f"  {lbl:16s} N={len(e):4d} long {100*v.mean():+.3f}% hit {100*(v>0).mean():.1f}%"
          f"   resid {100*vr.mean():+.3f}% hit {100*(vr>0).mean():.1f}%")

print("\n=== 5. overlap with LIVE pitch exposure (GDX long to 08-17) ===")
for h in (3, 5):
    ri = fwd_lag(ihi, h)
    rg = fwd_lag(gdx, h)
    j = pd.concat([ri, rg], axis=1).dropna()
    print(f"  h={h}: corr(IHI fwd, GDX fwd) all days = {j.iloc[:, 0].corr(j.iloc[:, 1]):+.3f}"
          f"   daily-return corr = {al['ihi'].corr(al['gdx']):+.3f}")
    e = declusters(trig, 5, px.index)
    e = e[rg.reindex(e).notna().values]
    print(f"       GDX return on IHI trigger episodes: {100*rg.loc[e].mean():+.3f}% (N={len(e)})")

print("\n=== 6. search cost: how many directions/horizons/thresholds were scanned ===")
print("  directions 2 x horizons {3,5,10} x (r21 4 x dd 4) = 96 cells swept in c6+c6b.")
print("  The pitched hypothesis was the FADE only; the continuation is a")
print("  checker-found cell and owes the search charge (registry 2026-08-07/12).")
