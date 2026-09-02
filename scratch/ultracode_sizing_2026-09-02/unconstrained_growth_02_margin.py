"""Unconstrained growth, part 2: the margin boundary.  Open gross notional by
day from the ledger (Entry..Exit inclusive, Shares_flat x Entry Price, flat
$750k), an IBKR portfolio-margin style requirement (single stock / broad ETF
base rate, 3x ETFs at 3x the base, concentration add-on), the multiple m of
current sizing at which the requirement crosses NAV, and a joint block
bootstrap (return + margin ratio drawn as one row) in which requirement >
equity is an absorbing ruin state.  Writes unconstrained_growth_02_margin.json.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import strategy_config as sc  # noqa: E402

NAV = 750_000.0
GRM_NOW = 1.5
RNG = np.random.default_rng(20260902)
M_GRID = np.array([0.5, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0])
OUT: dict = {"m_grid": M_GRID.tolist()}

led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
led = led[led["PnL_flat_750k"].notna()].copy()
led["notional"] = led["Entry Price"] * led["Shares_flat"]
LEV3X = set(sc.LEV3X_ALL)
BROAD = {"SPY", "QQQ", "IWM", "DIA", "^GSPC", "^NDX", "VOO", "IVV", "MDY", "IJR", "IJH", "RSP", "EFA", "EEM", "VEA", "VWO"}
SECTOR = {"XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC", "GLD", "SLV", "TLT", "IEF", "LQD", "HYG", "USO", "UNG", "GDX", "GDXJ", "XOP", "XBI", "SMH", "KRE", "XME", "XRT", "XHB", "ITB", "IYR", "VNQ", "FXI", "EWZ", "EWJ", "EWY", "EWT", "EWG", "EWU", "EWC", "EWW", "EWA", "EWH", "INDA", "KWEB", "ARKK", "IBB", "OIH", "TAN", "URA", "LIT", "DBC", "DBA", "UUP", "FXE", "FXY", "SLX", "COPX", "PPLT", "PALL", "CORN", "WEAT", "SOYB", "WOOD", "REMX", "SIL", "JETS", "PHO", "MOO", "BOTZ", "SOXX", "HACK", "IGV", "VGT", "XLG", "SPHB", "SPLV", "VXX", "UVXY", "SVXY", "TLH", "TBT", "TMF", "BND", "AGG"}
def klass(t: str) -> str:
    if t in LEV3X: return "lev3x"
    if t in BROAD: return "broad_etf"
    if t in SECTOR: return "sector_etf"
    return "single"
led["klass"] = led["Ticker"].map(klass)
print("notional by class (share of trade-notional):", (led.groupby("klass")["notional"].sum() / led["notional"].sum()).round(3).to_dict())

# ------------------------------------------------------------ 1. open notional by day, by class / direction / strategy / ticker
idx = pd.bdate_range("2003-01-01", "2026-09-01")
def open_by(group_col):
    keys = led[group_col].unique()
    mat = pd.DataFrame(0.0, index=idx, columns=keys)
    pos = {k: i for i, k in enumerate(keys)}
    arr = mat.values
    d0 = idx.searchsorted(led["Entry Date"].values); d1 = idx.searchsorted(led["Exit Date"].values)
    for a, b, k, n in zip(d0, d1, led[group_col].values, led["notional"].values):
        arr[a:b + 1, pos[k]] += n
    return mat
by_class = open_by("klass"); by_dir = open_by("Direction"); by_strat = open_by("Strategy"); by_tkr = open_by("Ticker")
gross = by_class.sum(axis=1)
W = gross[gross.index >= "2016-01-01"]
OUT["gross_notional_pct_nav_at_1x"] = dict(
    full=dict(p50=float(gross.quantile(.5) / NAV), p95=float(gross.quantile(.95) / NAV), p99=float(gross.quantile(.99) / NAV), max=float(gross.max() / NAV), max_date=str(gross.idxmax().date())),
    since2016=dict(p50=float(W.quantile(.5) / NAV), p95=float(W.quantile(.95) / NAV), p99=float(W.quantile(.99) / NAV), max=float(W.max() / NAV), max_date=str(W.idxmax().date())))
print("\n=== 1. gross open notional / NAV at current sizing (m=1) ===")
print(json.dumps(OUT["gross_notional_pct_nav_at_1x"], indent=1))
top = gross.sort_values(ascending=False).head(10)
print("top-10 notional days and who owns them:")
owners = []
for d, v in top.items():
    sh = (by_strat.loc[d] / v).sort_values(ascending=False)
    cl = (by_class.loc[d] / v)
    owners.append(dict(date=str(d.date()), gross_pct_nav=float(v / NAV), long_pct=float(by_dir.loc[d].get("Long", 0) / NAV), short_pct=float(by_dir.loc[d].get("Short", 0) / NAV),
                       lev3x_share=float(cl.get("lev3x", 0)), top_strats={k: round(float(x), 2) for k, x in sh.head(3).items()}))
    print(f"  {d.date()} gross {v/NAV:.0%} NAV  L {by_dir.loc[d].get('Long',0)/NAV:.0%} S {by_dir.loc[d].get('Short',0)/NAV:.0%}  3x {cl.get('lev3x',0):.0%}  {dict(sh.head(3).round(2))}")
OUT["top_notional_days"] = owners
# who owns the top decile of notional days (2016+)
hi = W[W >= W.quantile(.9)].index
share_hi = (by_strat.loc[hi].sum() / by_strat.loc[hi].sum().sum()).sort_values(ascending=False)
OUT["top_decile_notional_owner_share_2016"] = {k: float(v) for k, v in share_hi.head(8).items()}
print("owners of top-decile notional days 2016+:", share_hi.head(6).round(3).to_dict())
# by-strategy p95 open notional
OUT["strategy_open_notional_p95_pct_nav"] = {k: float(v) for k, v in (by_strat[by_strat.index >= "2016-01-01"].quantile(.95) / NAV).sort_values(ascending=False).items()}

# ------------------------------------------------------------ 2. margin requirement scenarios at m=1
def requirement(base_rate: float, conc_threshold: float | None, conc_rate: float, lev_mult: float = 3.0) -> pd.Series:
    """Requirement in $ at m=1 (flat basis).  Concentration: any ticker whose open notional > conc_threshold*NAV is charged conc_rate (x lev_mult if 3x)."""
    rate_cls = {"single": base_rate, "broad_etf": min(base_rate, 0.15), "sector_etf": base_rate, "lev3x": base_rate * lev_mult}
    req = sum(by_class[c] * rate_cls[c] for c in by_class.columns)
    if conc_threshold is not None:
        conc = by_tkr.where(by_tkr > conc_threshold * NAV, 0.0)
        # add-on = (conc_rate - base) on the concentrated notional; 3x names scale by lev_mult
        is3 = np.array([t in LEV3X for t in by_tkr.columns])
        addon = (conc * np.where(is3, (conc_rate - base_rate) * lev_mult, conc_rate - base_rate)).sum(axis=1)
        req = req + addon
    return req
SCEN = {
    "pm_15": dict(base_rate=0.15, conc_threshold=None, conc_rate=0.15),
    "pm_15_conc30": dict(base_rate=0.15, conc_threshold=0.25, conc_rate=0.30),
    "pm_25_conc30": dict(base_rate=0.25, conc_threshold=0.25, conc_rate=0.30),
    "pm_30_stress": dict(base_rate=0.30, conc_threshold=0.15, conc_rate=0.40),
    "regT_50": dict(base_rate=0.50, conc_threshold=None, conc_rate=0.50, lev_mult=1.8),
}
reqs = {k: requirement(**v) for k, v in SCEN.items()}
print("\n=== 2. margin requirement / NAV at m=1 and the m at which it crosses 100% NAV (flat $750k) ===")
OUT["margin_scenarios"] = {}
for k, q in reqs.items():
    qq = q[q.index >= "2016-01-01"] / NAV
    rows = dict(p50=float(qq.quantile(.5)), p95=float(qq.quantile(.95)), p99=float(qq.quantile(.99)), max=float(qq.max()), max_date=str(qq.idxmax().date()),
                m_ruin_at_max=float(1 / qq.max()), m_ruin_at_p99=float(1 / qq.quantile(.99)), m_ruin_at_p95=float(1 / qq.quantile(.95)),
                days_per_year_over_50pct_at_m={f"{m:g}": float((qq * m > 0.5).mean() * 252) for m in M_GRID},
                full_max=float((q / NAV).max()), full_max_date=str(q.idxmax().date()), m_ruin_full_max=float(NAV / q.max()))
    OUT["margin_scenarios"][k] = rows
    print(f"{k:14s} req/NAV p50 {rows['p50']:.1%} p95 {rows['p95']:.1%} p99 {rows['p99']:.1%} max {rows['max']:.1%} ({rows['max_date']}) "
          f"-> m at which max day = NAV: {rows['m_ruin_at_max']:.2f} (GRM {rows['m_ruin_at_max']*GRM_NOW:.2f}); p99 day: {rows['m_ruin_at_p99']:.2f}; 2003+ max {rows['full_max']:.1%} -> m {rows['m_ruin_full_max']:.2f}")

# ------------------------------------------------------------ 3. joint bootstrap: return + requirement ratio; ruin = requirement > equity
sd = json.load(open(ROOT / "dist/data/strategy_daily.json"))
tot = pd.Series(sd["total_flat"], index=pd.to_datetime(sd["dates"]), dtype=float)
tot = tot[tot.index <= "2026-08-07"]
def stationary_bootstrap_idx(n, paths, T, mean_block, rng):
    p = 1.0 / mean_block
    idx = np.empty((paths, T), dtype=np.int64); idx[:, 0] = rng.integers(0, n, paths)
    for t in range(1, T):
        new = rng.random(paths) < p
        idx[:, t] = np.where(new, rng.integers(0, n, paths), (idx[:, t - 1] + 1) % n)
    return idx
PATHS = 3000
OUT["ruin_bootstrap"] = {}
print("\n=== 3. joint block bootstrap: P(requirement > equity within 1y / 3y) by m, flat (additive) and compounding bases ===")
for wname, start in [("2016+", "2016-01-01"), ("2003+", "2003-01-21")]:
    J = pd.DataFrame({"r": tot / NAV}).join(pd.DataFrame({k: v / NAV for k, v in reqs.items()})).dropna()
    J = J[J.index >= start]
    n = len(J); rv = J["r"].values
    OUT["ruin_bootstrap"][wname] = {}
    for T, tag in [(252, "1y"), (756, "3y")]:
        idx = stationary_bootstrap_idx(n, PATHS, T, 10.0, RNG)
        R = rv[idx]
        OUT["ruin_bootstrap"][wname][tag] = {}
        for k in ["pm_15", "pm_15_conc30", "pm_25_conc30", "pm_30_stress"]:
            Q = J[k].values[idx]
            row = {}
            for m in M_GRID:
                eq_flat = 1.0 + np.cumsum(m * R, axis=1)                 # flat basis: additive PnL, fixed notional
                ruin_flat = ((m * Q > eq_flat) | (eq_flat <= 0)).any(axis=1).mean()
                # compounding basis: notional proportional to equity -> requirement ratio is m*Q regardless of equity;
                # but a day's loss shrinks equity before the next day's requirement re-scales, so check m*Q > 1 - m*r (loss on the day)
                ruin_comp = ((m * Q > 1.0) | (1 + m * R <= 0)).any(axis=1).mean()
                # 'stressed' variant: requirement evaluated against equity after a same-day -10% SPY-style shock is not modelled; add 15% cushion instead
                ruin_flat_cushion = ((m * Q > 0.85 * eq_flat) | (eq_flat <= 0)).any(axis=1).mean()
                row[f"{m:g}"] = dict(p_ruin_flat=float(ruin_flat), p_ruin_comp=float(ruin_comp), p_ruin_flat_15pct_cushion=float(ruin_flat_cushion))
            OUT["ruin_bootstrap"][wname][tag][k] = row
            if tag == "3y":
                print(f"{wname} {tag} {k:14s}: " + "  ".join(f"m{m:g}:{row[f'{m:g}']['p_ruin_flat']:.0%}/{row[f'{m:g}']['p_ruin_comp']:.0%}" for m in [1, 2, 3, 4, 5, 6, 8, 10]))

# ------------------------------------------------------------ 4. what a hard gross cap would have cost (how binding is the tail?)
print("\n=== 4. share of trade-days / PnL on days where gross exceeds x% NAV at m=1 (2016+) ===")
sd_tot = tot[tot.index >= "2016-01-01"]
g16 = gross.reindex(sd_tot.index)
OUT["gross_tail_pnl_share"] = {}
for x in [1.0, 1.5, 2.0, 2.5, 3.0]:
    mask = g16 > x * NAV
    OUT["gross_tail_pnl_share"][f"{x:g}"] = dict(days_share=float(mask.mean()), pnl_share=float(sd_tot[mask].sum() / sd_tot.sum()), mean_bps_on_days=float(sd_tot[mask].mean() / NAV * 1e4) if mask.any() else None,
                                                mean_bps_other=float(sd_tot[~mask].mean() / NAV * 1e4))
    print(f"  gross > {x:.0%}: {mask.mean():.1%} of days, {sd_tot[mask].sum()/sd_tot.sum():.1%} of PnL, mean {sd_tot[mask].mean()/NAV*1e4:.1f} bps/day vs {sd_tot[~mask].mean()/NAV*1e4:.1f} other")

json.dump(OUT, open(HERE / "unconstrained_growth_02_margin.json", "w"), indent=1, default=float)
print("\nwrote", HERE / "unconstrained_growth_02_margin.json")
