"""c6 round 2 (charged even though round 1 read as a kill, so the kill names the
right thing): the deep-drawdown row, the GLD-beta residual, definition
neighbours, and the reference class (complex-confirmed UP day -> highest-beta
member continues) on energy and rates complexes.

A. deep-drawdown row (SLV >= 20% / 35% under its 52w high) vs a STATE-MATCHED
   control (every SLV day in the same drawdown bucket), lag profile, entry-day
   split, signed concentration, era.
B. GLD-beta residual: SLV fwd - beta_pit(SLV on GLD, trailing 252 through D-1)
   x GLD fwd.
C. definition neighbours: 1.25/1.5/1.75/2.0/2.5% thresholds; rank form (each
   member's 1d return >= 90th pct of its trailing 252); dollar gate on/off; with
   the LAG PROFILE on every one.
D. reference class: energy (USO, XLE, XOP -> long XOP and USO), rates (TLT, IEF,
   LQD -> long TLT), equity (SPY, QQQ, IWM -> long IWM) under the rank form.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kC_common import *  # noqa

GAP = 5
T = ["GLD", "SLV", "GDX", "DX-Y.NYB", "USO", "XLE", "XOP", "TLT", "IEF", "LQD",
     "SPY", "QQQ", "IWM"]
px = panel(T, "GLD", ffill=("DX-Y.NYB",))
px = px.loc["2006-05-22":]
r1 = {t: dret(px[t]) for t in T}
rk = {t: pct_rank(px[t], 1) for t in T}
slv = px["SLV"]
dd52 = slv / slv.rolling(252, min_periods=200).max() - 1.0
dx_ok = r1["DX-Y.NYB"] <= 0
L = [("SLV", 1.0)]


def trig_thr(thr, dollar=True):
    m = (r1["GLD"] >= thr) & (r1["SLV"] >= thr) & (r1["GDX"] >= thr)
    return (m & dx_ok if dollar else m).fillna(False)


base = trig_thr(0.015)

print("=" * 100)
print("A. DEEP-DRAWDOWN ROW vs STATE-MATCHED control (all SLV days in the same bucket)")
for lbl, st in [("dd52 <= -20%", dd52 <= -0.20), ("dd52 <= -35%", dd52 <= -0.35)]:
    st = st.fillna(False)
    rows = []
    for h in (1, 2, 3, 5):
        for lag in (0, 1, 2):
            r = vehicle_ret(px, L, h, lag)
            s, epi, vals = cellstats(px, base & st, L, h, f"{lbl} h={h} lag={lag}", GAP, lag)
            ctl = r[st & r.notna()]
            s["state_ctl_pct"] = 100 * ctl.mean()
            s["edge_vs_state_pp"] = s["mean_pct"] - 100 * ctl.mean()
            s["welch_t_state"] = welch(vals, ctl.values)
            s["p_state_base"] = round(sign_test(int((vals > 0).sum()), len(vals), float((ctl > 0).mean())), 4)
            rows.append(s)
    show(rows, f"row {lbl}: lag profile vs state-matched control")
    for h in (3, 5):
        s, epi, vals = cellstats(px, base & st, L, h, "", GAP)
        print("  ", signed_conc(epi, vals, f"{lbl} h={h}"))
        show(era_split(epi, vals), f"  era {lbl} h={h}")
        yrs = pd.Series(vals, index=pd.DatetimeIndex(epi).year)
        print("   by year (n, sum pp):", {int(y): (int(g.size), round(100 * g.sum(), 2)) for y, g in yrs.groupby(level=0)})
    e1 = r1["SLV"].shift(-1)
    rows = []
    for h in (3, 5):
        for el, em in [("entry-day up", e1 > 0), ("entry-day down", e1 <= 0)]:
            s, _, _ = cellstats(px, (base & st & em).fillna(False), L, h, f"{lbl} h={h} {el}", GAP)
            rows.append(s)
    show(rows, f"  entry-day split inside {lbl}")

print("\n" + "=" * 100)
print("B. GLD-BETA RESIDUAL of SLV (beta trailing 252 through D-1)")
beta = pit_beta(r1["SLV"], r1["GLD"])
print(f"  live beta SLV on GLD: {beta.iloc[-1]:.2f}   sd ratio {r1['SLV'].tail(252).std()/r1['GLD'].tail(252).std():.2f}")
rows = []
for h in (1, 3, 5):
    rs = vehicle_ret(px, [("SLV", 1.0)], h)
    rg = vehicle_ret(px, [("GLD", 1.0)], h)
    res = rs - beta * rg
    for lbl, m in [("all", base), ("dd52<=-20%", base & (dd52 <= -0.20).fillna(False))]:
        days = px.index[m.values & res.notna().values]
        epi = declusters(days, GAP, px.index)
        v = res.loc[epi].values
        s = summarize(v, f"resid h={h} {lbl}")
        s["ctl_pct"] = 100 * res.dropna().mean()
        s["rec"] = f"{int((v>0).sum())}-{int((v<=0).sum())}"
        s["p_coin"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        rows.append(s)
show(rows)

print("\n" + "=" * 100)
print("C. DEFINITION NEIGHBOURS with lag profile (episode mean %, long SLV; own drift h1 .052 h3 .156 h5 .260)")
defs = {f"thr {t*100:.2f}%": trig_thr(t) for t in (0.0125, 0.015, 0.0175, 0.02, 0.025)}
defs["thr 1.5% no dollar gate"] = trig_thr(0.015, False)
defs["thr 2.0% no dollar gate"] = trig_thr(0.02, False)
defs["rank>=90 all three, dx<=0"] = ((rk["GLD"] >= 90) & (rk["SLV"] >= 90) & (rk["GDX"] >= 90) & dx_ok).fillna(False)
defs["rank>=95 all three, dx<=0"] = ((rk["GLD"] >= 95) & (rk["SLV"] >= 95) & (rk["GDX"] >= 95) & dx_ok).fillna(False)
defs["SLV alone >= 1.5% (parent)"] = (r1["SLV"] >= 0.015).fillna(False)
defs["SLV>=1.5%, GLD or GDX < 1.5% (anti)"] = ((r1["SLV"] >= 0.015) & ~((r1["GLD"] >= 0.015) & (r1["GDX"] >= 0.015))).fillna(False)
rows = []
for lbl, m in defs.items():
    row = {"def": lbl}
    for h in (1, 3, 5):
        for lag in (0, 1):
            s, _, vals = cellstats(px, m, L, h, "", GAP, lag)
            row[f"h{h}L{lag}"] = round(s.get("mean_pct", np.nan), 3)
        row[f"n"] = s.get("n", 0)
        row[f"rec_h5L1"] = s.get("rec", "")
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 100)
print("D. REFERENCE CLASS: complex-confirmed UP day (every member 1d rank >= 90, dollar not up)")
print("   -> long the highest-beta member; edge = episode mean minus own all-days drift, lag 0 vs 1")
fams = {
    "metals (GLD,SLV,GDX)->SLV": (["GLD", "SLV", "GDX"], "SLV"),
    "metals ->GDX": (["GLD", "SLV", "GDX"], "GDX"),
    "energy (USO,XLE,XOP)->XOP": (["USO", "XLE", "XOP"], "XOP"),
    "energy ->USO": (["USO", "XLE", "XOP"], "USO"),
    "rates (TLT,IEF,LQD)->TLT": (["TLT", "IEF", "LQD"], "TLT"),
    "equity (SPY,QQQ,IWM)->IWM": (["SPY", "QQQ", "IWM"], "IWM"),
    "equity ->QQQ": (["SPY", "QQQ", "IWM"], "QQQ"),
}
rows = []
for lbl, (mem, veh) in fams.items():
    m = dx_ok.copy()
    for t in mem:
        m = m & (rk[t] >= 90)
    m = m.fillna(False)
    for h in (1, 3, 5):
        for lag in (0, 1):
            s, _, _ = cellstats(px, m, [(veh, 1.0)], h, f"{lbl} h={h} lag={lag}", GAP, lag)
            rows.append({k: s.get(k) for k in ("label", "n", "mean_pct", "ctl_pct", "edge_pp", "rec", "p_base")})
show(rows)
