"""GAP 13: basis mismatches.
(i)  Weak Close Decent Sznls carries a LEGACY seasonal size multiplier hard-coded in
     pages/strat_backtester.py (process_signals_fast sizing step 3: sznl >= 65 -> 1.5x,
     33 <= sznl < 50 -> 0.66x, else 1.0x; mirrored in daily_scan.py ~L2882). It enters the
     ledger through Risk_flat_750k / Shares_flat, hence Size_Mult. R = PnL/Risk is invariant
     to a pure per-trade scalar, so avgR tables cannot see it; PnL-per-unit-risk, Sharpe and
     anything fit on the SIZED daily series (the tilt) can. Here: is the seasonal multiplier
     correlated with the adds/solo state, and does normalising to unit size change the gap?
(ii) OLV pre-earnings override (10 bps nominal, -10..0 TD): ship date, basis at ship, and the
     earnings-offset cells on today's ledger + data/earnings_calendar.parquet.
Writes gap13_basis.json.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SIZ = HERE.parent
ROOT = SIZ.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(SIZ))
from earnings_filter import load_earnings_dates_map, signed_offset  # noqa: E402
from flow_conditional_lib import build_trade_mtm, dd_stats  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40)
OUT: dict = {}
NAV = 750_000.0


def cl_se(x: np.ndarray, c: np.ndarray):
    """cluster-robust SE of the mean (clusters = episodes)."""
    x = np.asarray(x, float); c = np.asarray(c)
    n = len(x); mu = x.mean()
    s = 0.0
    for g in np.unique(c):
        s += (x[c == g] - mu).sum() ** 2
    return float(np.sqrt(s) / n), float(mu)


# ================================================================= (i) WCDS
f = pd.read_parquet(SIZ / "within_strategy_adds_features.parquet")
w = f[f.Strategy == "Weak Close Decent Sznls"].copy().sort_values("Entry Date").reset_index(drop=True)
print(f"WCDS legs {len(w)}; Size_Mult values: {w.Size_Mult.round(3).value_counts().to_dict()}")
KNOWN_RESID = [1.0, 1.25, 0.25, 0.5, 0.3125, 0.125, 0.0625]
def sznl_mult(m: float):
    for s in (1.5, 1.0, 0.66):
        r = m / s
        if any(abs(r - k) / k < 0.03 for k in KNOWN_RESID):
            return s
    return np.nan
w["sznl_mult"] = w.Size_Mult.map(sznl_mult)
print("unresolved Size_Mult:", w.loc[w.sznl_mult.isna(), "Size_Mult"].round(3).tolist())
w["sznl_bucket"] = w.sznl_mult.map({1.5: "sznl>=65 (1.5x)", 1.0: "sznl 50-65 (1.0x)", 0.66: "sznl 33-50 (0.66x)"}).fillna("unresolved")
w["adds"] = w.n_open > 0
w["depth"] = np.where(w.n_open == 0, "0 solo", np.where(w.n_open == 1, "1", np.where(w.n_open == 2, "2", "3+")))
w["R"] = w.R_Multiple
# unit basis: divide out the WHOLE Size_Mult (seasonal x frag/PC x cap), and a seasonal-only normalisation
w["pnl_unit"] = w.PnL_flat_750k / w.Size_Mult; w["risk_unit"] = w.Risk_flat_750k / w.Size_Mult
w["pnl_nosznl"] = w.PnL_flat_750k / w.sznl_mult.fillna(1.0); w["risk_nosznl"] = w.Risk_flat_750k / w.sznl_mult.fillna(1.0)


def cell(g: pd.DataFrame) -> dict:
    se, mu = cl_se(g.R.values, g.episode.values) if len(g) > 1 else (np.nan, float(g.R.mean()) if len(g) else np.nan)
    return dict(N=int(len(g)), episodes=int(g.episode.nunique()), avgR=mu, se_cl=se, win=float((g.R > 0).mean()) if len(g) else np.nan,
                rpr_sized=float(g.PnL_flat_750k.sum() / g.Risk_flat_750k.sum()) if g.Risk_flat_750k.sum() else np.nan,
                rpr_unit=float(g.pnl_unit.sum() / g.risk_unit.sum()) if g.risk_unit.sum() else np.nan,
                rpr_nosznl=float(g.pnl_nosznl.sum() / g.risk_nosznl.sum()) if g.risk_nosznl.sum() else np.nan,
                share_1p5=float((g.sznl_mult == 1.5).mean()), share_0p66=float((g.sznl_mult == 0.66).mean()), mean_size_mult=float(g.Size_Mult.mean()),
                pnl=float(g.PnL_flat_750k.sum()), risk=float(g.Risk_flat_750k.sum()))


print("\n=== (i.a) WCDS solo vs adds, and by depth: sized vs unit-size PnL per unit risk, seasonal-mult mix ===")
OUT["wcds"] = {"by_depth": {}, "solo_vs_adds": {}}
for lab, g in [("solo", w[~w.adds]), ("adds", w[w.adds])] + [(d, w[w.depth == d]) for d in ["0 solo", "1", "2", "3+"]]:
    c = cell(g); OUT["wcds"]["by_depth"][lab] = c
    print(f"  {lab:7s} N {c['N']:3d} ep {c['episodes']:3d} avgR {c['avgR']:+.3f} (se {c['se_cl']:.3f}) win {c['win']:.0%} | rpr sized {c['rpr_sized']:+.3f} unit {c['rpr_unit']:+.3f} no-sznl {c['rpr_nosznl']:+.3f} | share 1.5x {c['share_1p5']:.0%} 0.66x {c['share_0p66']:.0%} mean mult {c['mean_size_mult']:.2f}")
S, A = w[~w.adds], w[w.adds]
seS, muS = cl_se(S.R.values, S.episode.values); seA, muA = cl_se(A.R.values, A.episode.values)
t_gap = (muA - muS) / np.sqrt(seS ** 2 + seA ** 2)
OUT["wcds"]["solo_vs_adds"] = dict(gap_avgR=float(muA - muS), t_cluster=float(t_gap), gap_rpr_sized=float(A.PnL_flat_750k.sum() / A.Risk_flat_750k.sum() - S.PnL_flat_750k.sum() / S.Risk_flat_750k.sum()),
                                   gap_rpr_unit=float(A.pnl_unit.sum() / A.risk_unit.sum() - S.pnl_unit.sum() / S.risk_unit.sum()))
print(f"  adds - solo: avgR gap {muA-muS:+.3f} (cluster t {t_gap:.2f}); rpr gap sized {OUT['wcds']['solo_vs_adds']['gap_rpr_sized']:+.3f} unit {OUT['wcds']['solo_vs_adds']['gap_rpr_unit']:+.3f}")

print("\n=== (i.b) seasonal-mult bucket x solo/adds (avgR is invariant to the multiplier; the mix is what can differ) ===")
OUT["wcds"]["by_sznl"] = {}
rows = []
for sb in ["sznl>=65 (1.5x)", "sznl 50-65 (1.0x)", "sznl 33-50 (0.66x)"]:
    for lab, g in [("solo", w[(w.sznl_bucket == sb) & ~w.adds]), ("adds", w[(w.sznl_bucket == sb) & w.adds])]:
        c = cell(g); OUT["wcds"]["by_sznl"][f"{sb}|{lab}"] = c
        rows.append(dict(sznl=sb, state=lab, **{k: c[k] for k in ["N", "avgR", "se_cl", "win", "rpr_sized", "pnl"]}))
D = pd.DataFrame(rows); print(D.round(3).to_string(index=False))
# stratified adds-solo gap (weight by the solo+adds N in each sznl stratum) -> removes the mix effect
gaps = []
for sb in ["sznl>=65 (1.5x)", "sznl 50-65 (1.0x)", "sznl 33-50 (0.66x)"]:
    a = OUT["wcds"]["by_sznl"][f"{sb}|adds"]; s = OUT["wcds"]["by_sznl"][f"{sb}|solo"]
    if a["N"] >= 5 and s["N"] >= 5:
        gaps.append((a["avgR"] - s["avgR"], a["N"] + s["N"], sb))
strat_gap = sum(g * n for g, n, _ in gaps) / sum(n for _, n, _ in gaps)
OUT["wcds"]["stratified_gap_avgR"] = float(strat_gap); OUT["wcds"]["within_stratum_gaps"] = {sb: float(g) for g, _, sb in gaps}
print(f"  within-stratum adds-solo avgR gaps: {[(sb, round(g, 3)) for g, _, sb in gaps]}; N-weighted stratified gap {strat_gap:+.3f} vs raw {muA-muS:+.3f}")
# does the seasonal multiplier itself earn? R by sznl bucket
print("\n=== (i.c) does the legacy seasonal multiplier size the right trades? avgR by bucket ===")
OUT["wcds"]["sznl_edge"] = {}
for sb in ["sznl>=65 (1.5x)", "sznl 50-65 (1.0x)", "sznl 33-50 (0.66x)"]:
    c = cell(w[w.sznl_bucket == sb]); OUT["wcds"]["sznl_edge"][sb] = c
    print(f"  {sb:20s} N {c['N']:3d} avgR {c['avgR']:+.3f} (se {c['se_cl']:.3f}) win {c['win']:.0%} pnl {c['pnl']:,.0f}")
# sleeve-level basis for the tilt: sized vs unit daily series
tr = pd.DataFrame(dict(Ticker=w.Ticker, **{"Entry Date": w["Entry Date"]}, ExitDate=w["Exit Date"], Shares=w.Shares_flat, EntryPrice=w["Entry Price"], PnL=w.PnL_flat_750k, Direction=w.Direction))
days, MTM = build_trade_mtm(tr)
sized = pd.Series(MTM.sum(0), index=days)
unit = pd.Series((MTM / w.Size_Mult.values[:, None]).sum(0), index=days)
nosz = pd.Series((MTM / w.sznl_mult.fillna(1.0).values[:, None]).sum(0), index=days)
OUT["wcds"]["sleeve_daily"] = {}
print("\n=== (i.d) WCDS sleeve daily series: sized (what the tilt was fit on) vs unit-size vs seasonal-mult removed ===")
for lab, s in [("sized", sized), ("unit (all mults out)", unit), ("seasonal mult out", nosz)]:
    st = dd_stats(s); mu, var = s.mean(), s.var()
    st.update(mean_over_var_x1e4=float(mu / var * 1e4), pnl_per_risk=float(s.sum() / (w.Risk_flat_750k.sum() if lab == "sized" else w.risk_unit.sum() if lab.startswith("unit") else w.risk_nosznl.sum())))
    OUT["wcds"]["sleeve_daily"][lab] = st
    print(f"  {lab:22s} total {st['total']:>10,.0f} sharpe {st['sharpe']:.2f} maxDD {st['maxdd']:>9,.0f} mean/var x1e4 {st['mean_over_var_x1e4']:.3f} pnl/risk {st['pnl_per_risk']:.3f}")
# 0.8 / 1.2 replay on the sized and the seasonal-mult-removed basis (adds = n_open > 0; no cap re-application, WCDS rarely cap-bound)
print("\n=== (i.e) 0.8x solo / 1.2x adds replay, sized vs seasonal-mult-removed basis ===")
OUT["wcds"]["adds_rule"] = {}
mult = np.where(w.adds, 1.2, 0.8)
for lab, base_pnl, base_risk in [("sized", w.PnL_flat_750k.values, w.Risk_flat_750k.values), ("seasonal mult out", w.pnl_nosznl.values, w.risk_nosznl.values)]:
    b = base_pnl.sum() / base_risk.sum(); r = (base_pnl * mult).sum() / (base_risk * mult).sum()
    OUT["wcds"]["adds_rule"][lab] = dict(base_rpr=float(b), rule_rpr=float(r), d_pct=float(r / b - 1), base_pnl=float(base_pnl.sum()), rule_pnl=float((base_pnl * mult).sum()), risk_ratio=float((base_risk * mult).sum() / base_risk.sum()))
    print(f"  {lab:20s} pnl/risk {b:.3f} -> {r:.3f} ({r/b-1:+.1%}); pnl {base_pnl.sum():,.0f} -> {(base_pnl*mult).sum():,.0f}; risk x{(base_risk*mult).sum()/base_risk.sum():.3f}")

# ================================================================= (ii) OLV earnings override
print("\n=== (ii) OLV pre-earnings override: ledger cells by signed earnings offset ===")
led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
olv = led[(led.Strategy == "Oversold Low Volume") & led.PnL_flat_750k.notna()].copy()
emap = load_earnings_dates_map(str(ROOT / "data/earnings_calendar.parquet"))
ovp = ROOT / "data/earnings_calendar_overflow.parquet"
if ovp.exists():
    extra = load_earnings_dates_map(str(ovp))
    for k, v in extra.items():
        emap[k] = np.unique(np.concatenate([emap.get(k, np.array([], dtype="datetime64[D]")), v]))
    print("overflow earnings file merged:", len(extra), "tickers")
cal = pd.read_parquet(ROOT / "data/earnings_calendar.parquet")
print(f"earnings calendar: {cal.ticker.nunique()} tickers, dates {cal.date.min().date()}..{cal.date.max().date()}; rows before 2010: {(cal.date < '2010-01-01').mean():.1%}, before 2015: {(cal.date < '2015-01-01').mean():.1%}")
per_tk_min = cal.groupby("ticker").date.min()
olv["off"] = [signed_offset(d, emap.get(t.upper())) for d, t in zip(olv["Signal Date"], olv.Ticker)]
olv["has_cal"] = olv.Ticker.str.upper().isin(emap.keys())
olv["cal_min"] = olv.Ticker.str.upper().map(per_tk_min)
olv["before_cal_start"] = olv["Signal Date"] < olv["cal_min"]
olv["R"] = olv.R_Multiple; olv["year"] = olv["Signal Date"].dt.year
bins = [(-999, -22, "< -21"), (-21, -11, "-21..-11"), (-10, 0, "-10..0 (override)"), (1, 5, "+1..+5"), (6, 10, "+6..+10"), (11, 21, "+11..+21"), (22, 999, "> +21")]
def obin(o):
    if pd.isna(o):
        return "NaN (no calendar)"
    for lo, hi, lab in bins:
        if lo <= o <= hi:
            return lab
    return "?"
olv["obin"] = olv.off.map(obin)
order = ["NaN (no calendar)"] + [b[2] for b in bins]
rows = []
for b in order:
    g = olv[olv.obin == b]
    if len(g) == 0:
        continue
    rows.append(dict(cell=b, N=len(g), avgR=g.R.mean(), se=g.R.std() / np.sqrt(len(g)), win=(g.R > 0).mean(), rpr=g.PnL_flat_750k.sum() / g.Risk_flat_750k.sum(), pnl=g.PnL_flat_750k.sum(),
                     mean_risk_bps=g["Risk bps"].mean(), mean_size_mult=g.Size_Mult.mean(), first_year=int(g.year.min()), n_2015plus=int((g.year >= 2015).sum()), worst=g.R.min()))
D = pd.DataFrame(rows); print(D.round(3).to_string(index=False))
OUT["olv_cells"] = D.round(4).to_dict("records")
print(f"OLV trades {len(olv)}: with calendar {olv.has_cal.mean():.0%}; signal before the ticker's first calendar date {olv.before_cal_start.fillna(False).mean():.0%} (offsets there point at a far-future first print -> land in '< -21' or NaN)")
OUT["olv_coverage"] = dict(N=int(len(olv)), share_with_calendar=float(olv.has_cal.mean()), share_signal_before_calendar_start=float(olv.before_cal_start.fillna(False).mean()),
                           cal_min=str(cal.date.min().date()), cal_max=str(cal.date.max().date()))
# the ledger's sizing inside the window (should be 15 bps effective x recency)
win = olv[olv.obin == "-10..0 (override)"]
print("override-window rows: Risk bps values", win["Risk bps"].round(1).value_counts().to_dict(), "| Size_Mult", win.Size_Mult.round(3).value_counts().head(6).to_dict())
# post-print deficit, finer and by era
print("\npost-print cells, finer + by era (2015+ where the calendar is dense):")
for lab, lo, hi in [("+1..+3", 1, 3), ("+4..+10", 4, 10), ("+1..+10", 1, 10), ("-10..0", -10, 0), ("-10..-1", -10, -1), ("0", 0, 0), ("outside +-10", None, None)]:
    if lo is None:
        g = olv[olv.off.notna() & ((olv.off < -10) | (olv.off > 10))]
    else:
        g = olv[(olv.off >= lo) & (olv.off <= hi)]
    g15 = g[g.year >= 2015]
    OUT.setdefault("olv_fine", {})[lab] = dict(N=int(len(g)), avgR=float(g.R.mean()) if len(g) else None, win=float((g.R > 0).mean()) if len(g) else None, rpr=float(g.PnL_flat_750k.sum() / g.Risk_flat_750k.sum()) if len(g) else None,
                                              N_2015=int(len(g15)), avgR_2015=float(g15.R.mean()) if len(g15) else None)
    print(f"  {lab:12s} N {len(g):3d} avgR {g.R.mean() if len(g) else float('nan'):+.3f} win {(g.R>0).mean() if len(g) else float('nan'):.0%} rpr {g.PnL_flat_750k.sum()/g.Risk_flat_750k.sum() if len(g) else float('nan'):+.3f} | 2015+: N {len(g15)} avgR {g15.R.mean() if len(g15) else float('nan'):+.3f}")
# what the override costs / saves on today's ledger: window trades at full size (35/25 nominal) would have earned R x (full risk)
full_bps = np.where(win.Tier == "Overflow", 25.0, 35.0) * 1.5
win_full_risk = full_bps / 1e4 * NAV * (win.Size_Mult / (win["Risk bps"] / full_bps)).clip(lower=0)  # not exact; report the simple version below
simple = float((win.R * (full_bps / 1e4 * NAV)).sum()) - float(win.PnL_flat_750k.sum())
OUT["olv_override_cost"] = dict(window_pnl_as_booked=float(win.PnL_flat_750k.sum()), window_pnl_if_full_size_no_ladder=float((win.R * (full_bps / 1e4 * NAV)).sum()), foregone=simple, N=int(len(win)))
print(f"override window: booked PnL {win.PnL_flat_750k.sum():,.0f} vs at full base bps (no ladder) {(win.R * (full_bps/1e4*NAV)).sum():,.0f} -> the override forgoes ~{simple:,.0f} on the ledger")
# by year in the window (is the 'no deficit' one era?)
yb = win.groupby("year").agg(N=("R", "size"), avgR=("R", "mean"), pnl=("PnL_flat_750k", "sum"))
print("window by year:\n", yb.round(2).to_string())
OUT["olv_window_by_year"] = yb.round(4).reset_index().to_dict("records")
json.dump(OUT, open(HERE / "gap13_basis.json", "w"), indent=1, default=float)
print("wrote gap13_basis.json")
