"""c6 round 1: long EFA against beta-SPY after EFA lags SPY by >= 1.0pp (beta
residual) on an expiry session with the dollar flat. Decisive control: the SAME
gap on ordinary non-expiry sessions. Rows: EEM / EWJ as the lagging leg, dollar
gate on/off, quads vs monthly opex, September quads, lag 0 vs lag 1."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kC_common import *  # noqa

px = nyse_panel(["EFA", "SPY", "DX-Y.NYB", "EEM", "EWJ"], ffill=("DX-Y.NYB",))
idx = px.index
rdx = dret(px["DX-Y.NYB"])
for leg in ["EFA", "EEM", "EWJ"]:
    ri, rr, b = resid_index(px, leg)
    px[f"R_{leg}"] = ri
    px[f"gap_{leg}"] = rr
    px[f"beta_{leg}"] = b
    px[f"raw_{leg}"] = dret(px[leg]) - dret(px["SPY"])

opex = to_sessions(load_events(["opex"])["date"], idx)
quad = to_sessions(load_events(["quad_witching"])["date"], idx)
is_opex = pd.Series(idx.isin(opex), index=idx)
is_quad = pd.Series(idx.isin(quad), index=idx)
dxflat = rdx.abs() <= 0.002

live = idx[-1]
print(f"LIVE {live.date()}: " + ", ".join(
    f"{l} ret {100*dret(px[l]).iloc[-1]:+.2f}% resid {100*px[f'gap_{l}'].iloc[-1]:+.2f}pp "
    f"beta {px[f'beta_{l}'].iloc[-1]:.2f}" for l in ["EFA", "EEM", "EWJ"]))
print(f"  SPY {100*dret(px['SPY']).iloc[-1]:+.2f}%  DX {100*rdx.iloc[-1]:+.2f}%  quad={is_quad.iloc[-1]}")

COST = 4  # bps round trip per leg (EFA/SPY liquid)
for leg in ["EFA", "EEM", "EWJ"]:
    g = px[f"gap_{leg}"]
    lagm = g <= -0.01
    rows = []
    for h in (1, 2, 3, 5):
        for lag in (0, 1):
            for lbl, m in [("quad & gap & dxflat", is_quad & lagm & dxflat),
                           ("quad & gap", is_quad & lagm),
                           ("opex & gap & dxflat", is_opex & lagm & dxflat),
                           ("opex & gap", is_opex & lagm),
                           ("NON-expiry & gap & dxflat", ~is_opex & lagm & dxflat),
                           ("NON-expiry & gap", ~is_opex & lagm),
                           ("all quads (no gap)", is_quad)]:
                s, e, v = cellstats(px, m, [(f"R_{leg}", 1.0)], h, f"{lbl} h={h} lag={lag}",
                                    lag=lag, cost_bps=COST * 2)
                rows.append(s)
    show(rows, f"{leg}: long {leg} vs beta-SPY after a <= -1.0pp residual session")

# decisive diff: expiry vs non-expiry at the tradeable lag, EFA
g = px["gap_EFA"]
lagm = g <= -0.01
print("\n=== EFA decisive control: expiry vs non-expiry, lag=1 (Welch on episodes) ===")
for h in (1, 3, 5):
    _, _, a = cellstats(px, is_opex & lagm & dxflat, [("R_EFA", 1.0)], h, "x")
    _, _, q = cellstats(px, is_quad & lagm & dxflat, [("R_EFA", 1.0)], h, "x")
    _, _, c = cellstats(px, ~is_opex & lagm & dxflat, [("R_EFA", 1.0)], h, "x")
    d1, t1 = welch(a, c)
    d2, t2 = welch(q, c)
    print(f"h={h}: opex-minus-nonexp {d1:+.3f}pp t {t1:+.2f} | quad-minus-nonexp {d2:+.3f}pp t {t2:+.2f}")

# magnitude bins on non-expiry days (does ANY >=1pp lag reverse?)
rows = []
for lo, hi in [(-0.10, -0.02), (-0.02, -0.015), (-0.015, -0.01), (-0.01, -0.005), (0.005, 0.01), (0.01, 0.10)]:
    for lag in (0, 1):
        m = (~is_opex) & (g > lo) & (g <= hi)
        s, _, _ = cellstats(px, m, [("R_EFA", 1.0)], 1, f"nonexp resid in ({lo},{hi}] h=1 lag={lag}", lag=lag)
        rows.append(s)
show(rows, "EFA residual next-session by gap bin, NON-expiry (non-synchronous reversal check)")

# September quads, every one, EFA, with gap
rows = []
for d in quad:
    if d.month != 9 or d not in idx:
        continue
    p = idx.get_loc(d)
    if p + 6 >= len(idx):
        continue
    R = px["R_EFA"].values
    rows.append({"quad": d.date(), "resid_pp": round(100 * g.loc[d], 2),
                 "dx_pct": round(100 * rdx.loc[d], 2),
                 "h1_lag1": round(100 * (R[p + 2] / R[p + 1] - 1), 3),
                 "h3_lag1": round(100 * (R[p + 4] / R[p + 1] - 1), 3),
                 "h5_lag1": round(100 * (R[p + 6] / R[p + 1] - 1), 3),
                 "h1_lag0": round(100 * (R[p + 1] / R[p] - 1), 3)})
print("\n=== every September quad, EFA residual ===")
print(pd.DataFrame(rows).to_string(index=False))

# battery on the headline cell, EFA quad & gap & dxflat (min_gap 5)
battery(px, is_opex & lagm & dxflat, [("R_EFA", 1.0)], 3,
        "c6 EFA resid after <=-1pp on OPEX with dollar flat", cost_bps=COST * 2,
        variants={"quad only": is_quad & lagm & dxflat,
                  "opex no dx gate": is_opex & lagm,
                  "opex gap<=-0.75": is_opex & (g <= -0.0075) & dxflat,
                  "opex gap<=-1.5": is_opex & (g <= -0.015) & dxflat,
                  "NONEXP same": ~is_opex & lagm & dxflat},
        min_gap=5, event_kinds=("fomc_decision",))
