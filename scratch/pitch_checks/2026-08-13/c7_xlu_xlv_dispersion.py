"""C7 - long XLU / short XLV at opposite 21d rank extremes.

Cell: XLU rank21 <= 10 AND XLV rank21 >= 85 (trailing 252d).
Registry: long XLU is dead in FIVE expressions (outright washout incl. the
21d-rank form killed 2026-08-12, the XLP pair, the SPY spread, the rates
channel) and must not be reopened without a new mechanism. So the ONLY thing
that can save this is the XLV leg. That leg therefore gets measured ALONE:
excess over XLV's own drift, and the residual against SPY at the measured beta.
Gate attribution runs each leg's trigger separately.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

TK = ["XLU", "XLV", "SPY", "XLP"]
px = close_panel(TK).dropna()
raw = load_prices(TK)
xlu, xlv, spy = raw["XLU"]["Close"], raw["XLV"]["Close"], raw["SPY"]["Close"]

u21, v21 = pct_rank(xlu, 21), pct_rank(xlv, 21)


def mask(u=10, v=85):
    return ((u21 <= u) & (v21 >= v)).reindex(px.index).fillna(False)


print("=== occurrence counts ===")
rows = []
for u in (5, 10, 15, 20):
    for v in (80, 85, 90, 95):
        m = mask(u, v)
        d = px.index[m.values]
        rows.append({"xlu_r21<=": u, "xlv_r21>=": v, "n_days": len(d),
                     "n_epi": len(declusters(d, 5, px.index)),
                     "first": str(d[0].date()) if len(d) else "-",
                     "last": str(d[-1].date()) if len(d) else "-"})
print(pd.DataFrame(rows).to_string(index=False))

al = pd.concat([xlu.pct_change(), xlv.pct_change(), spy.pct_change()], axis=1).dropna()
al.columns = ["xlu", "xlv", "spy"]
b_uv = np.polyfit(al["xlv"], al["xlu"], 1)[0]      # XLU on XLV
b_vs = np.polyfit(al["spy"], al["xlv"], 1)[0]      # XLV on SPY
b_us = np.polyfit(al["spy"], al["xlu"], 1)[0]      # XLU on SPY
print(f"\nbeta XLU~XLV = {b_uv:.3f} (corr {al['xlu'].corr(al['xlv']):.3f})")
print(f"beta XLV~SPY = {b_vs:.3f} (corr {al['xlv'].corr(al['spy']):.3f})")
print(f"beta XLU~SPY = {b_us:.3f} (corr {al['xlu'].corr(al['spy']):.3f})")

m = mask()
trig = px.index[m.values]
epi = declusters(trig, 5, px.index)
print(f"\nAS SPECIFIED n_days={len(trig)} epi={len(epi)}")
print("episode dates:", ", ".join(str(d.date()) for d in epi))

# ---------------------------------------------------------------- WHICH LEG
print("\n\n########## WHICH LEG CARRIES IT (the decisive question) ##########")
for h in (3, 5, 10):
    r_u = fwd_lag(xlu, h).reindex(px.index)
    r_v = fwd_lag(xlv, h).reindex(px.index)
    r_s = fwd_lag(spy, h).reindex(px.index)
    valid = r_u.notna() & r_v.notna() & r_s.notna()
    e = epi[valid.loc[epi].values]
    span = (px.index >= trig[0]) & (px.index <= trig[-1]) & valid.values
    rows = [
        summarize(r_u.loc[e].values, f"h={h} XLU long leg (N={len(e)})"),
        summarize(r_u[span].values, f"h={h} XLU own drift, same span"),
        summarize(-r_v.loc[e].values, f"h={h} XLV SHORT leg"),
        summarize(-r_v[span].values, f"h={h} XLV short own drift, same span"),
        summarize((r_u - b_uv * r_v).loc[e].values,
                  f"h={h} PAIR beta-neutral (b={b_uv:.2f})"),
        summarize((r_u - r_v).loc[e].values, f"h={h} PAIR equal-dollar"),
        summarize((-r_v + b_vs * r_s).loc[e].values,
                  f"h={h} XLV short residual vs SPY (b={b_vs:.2f})"),
        summarize((-r_v + b_vs * r_s)[span].values,
                  f"h={h} that residual, same-span drift"),
        summarize((r_u - b_us * r_s).loc[e].values,
                  f"h={h} XLU long residual vs SPY (b={b_us:.2f})"),
        summarize(r_s.loc[e].values, f"h={h} SPY on the same windows"),
    ]
    show(rows, f"leg decomposition h={h}, episodes")

# ---------------------------------------------------------------- gate attribution
print("\n\n########## GATE ATTRIBUTION: each leg's trigger alone ##########")
mu = (u21 <= 10).reindex(px.index).fillna(False)
mv = (v21 >= 85).reindex(px.index).fillna(False)
for h in (3, 5):
    r_u = fwd_lag(xlu, h).reindex(px.index)
    r_v = fwd_lag(xlv, h).reindex(px.index)
    r_s = fwd_lag(spy, h).reindex(px.index)
    pair = r_u - b_uv * r_v
    vres = -r_v + b_vs * r_s
    rows = []
    for lbl, mm in [("XLU r21<=10 ONLY", mu), ("XLV r21>=85 ONLY", mv),
                    ("BOTH (the cell)", m)]:
        e = declusters(px.index[mm.values], 5, px.index)
        e = e[pair.reindex(e).notna().values]
        rows.append(summarize(pair.loc[e].values, f"h={h} pair | {lbl} (N={len(e)})"))
        rows.append(summarize(vres.loc[e].values, f"h={h} XLVshort-resid | {lbl}"))
    show(rows, f"gate attribution h={h}")

# ---------------------------------------------------------------- batteries
variants = {"xlu<=5": mask(5, 85), "xlu<=15": mask(15, 85),
            "xlv>=80": mask(10, 80), "xlv>=90": mask(10, 90),
            "xlv>=95": mask(10, 95)}
for h in (3, 5):
    battery(px, m, [("XLU", 1.0), ("XLV", -b_uv)], h,
            f"C7 PAIR beta-neutral (b={b_uv:.2f})", 9.0, variants)
    battery(px, m, [("XLV", -1.0)], h, "C7 XLV SHORT alone", 8.0, variants)
    battery(px, m, [("XLV", -1.0), ("SPY", b_vs)], h,
            f"C7 XLV short vs long SPY (b={b_vs:.2f})", 9.0, variants)
    battery(px, m, [("XLU", 1.0)], h, "C7 XLU LONG alone (registry: dead 5x)", 8.0)

# ---------------------------------------------------------------- live gate + cuts
print("\n\n########## SPY-near-52w-high state (LIVE today, killed the XLU cell) ##########")
spyhi = ((spy / spy.rolling(252).max() - 1) > -0.01).reindex(px.index).fillna(False)
print(f"  SPY dd today = {100*(spy.iloc[-1]/spy.rolling(252).max().iloc[-1]-1):+.3f}% -> gate ON")
for h in (3, 5):
    r_u, r_v, r_s = (fwd_lag(x, h).reindex(px.index) for x in (xlu, xlv, spy))
    pair, vres = r_u - b_uv * r_v, -r_v + b_vs * r_s
    e = epi[pair.reindex(epi).notna().values]
    g = spyhi.loc[e].values
    show([summarize(pair.loc[e[g]].values, f"h={h} pair, SPY near high (N={int(g.sum())})"),
          summarize(pair.loc[e[~g]].values, f"h={h} pair, SPY not near high"),
          summarize(vres.loc[e[g]].values, f"h={h} XLVshort-resid, SPY near high"),
          summarize(vres.loc[e[~g]].values, f"h={h} XLVshort-resid, SPY not near high"),
          summarize((-r_v).loc[e[g]].values, f"h={h} XLV short raw, SPY near high"),
          summarize((-r_v).loc[e[~g]].values, f"h={h} XLV short raw, SPY not near high")])

print("\n########## midterm + era + depth, h=5 ##########")
h = 5
r_u, r_v, r_s = (fwd_lag(x, h).reindex(px.index) for x in (xlu, xlv, spy))
pair, vres = r_u - b_uv * r_v, -r_v + b_vs * r_s
e = epi[pair.reindex(epi).notna().values]
mt = np.array([d.year % 4 == 2 for d in e])
show([summarize(pair.loc[e[mt]].values, f"pair midterm (N={int(mt.sum())})"),
      summarize(pair.loc[e[~mt]].values, "pair non-midterm"),
      summarize(vres.loc[e[mt]].values, "XLVshort-resid midterm"),
      summarize(vres.loc[e[~mt]].values, "XLVshort-resid non-midterm")])
show(era_split(e, pair.loc[e].values), "pair era split")
show(era_split(e, vres.loc[e].values), "XLV-short-residual era split")
print("  pair concentration:", cluster_note(e, pair.loc[e].values))
print("  XLVresid concentration:", cluster_note(e, vres.loc[e].values))
print("  by year (pair):", (pd.Series(pair.loc[e].values, index=e.year)
                            .groupby(level=0).sum().mul(100).round(2).to_dict()))
print("  by year (XLVresid):", (pd.Series(vres.loc[e].values, index=e.year)
                                .groupby(level=0).sum().mul(100).round(2).to_dict()))

mvv = m.values
dep = np.zeros(len(mvv), int)
for i in range(len(mvv)):
    dep[i] = dep[i - 1] + 1 if mvv[i] and i > 0 else (1 if mvv[i] else 0)
dser = pd.Series(dep, index=px.index)
print(f"\n  today's cluster depth = {int(dser.iloc[-1])}, population median "
      f"{float(dser.loc[trig].median()):.1f}")
