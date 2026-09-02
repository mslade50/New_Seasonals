"""Growth-maximizer lens, part 1: the margin boundary with TIMS-TIERED rates.

The unconstrained_growth study charged every non-3x leg 15%.  IBKR portfolio
margin (TIMS) stresses broad-based indices +6/-8% (~8%), small-cap indices
+/-10%, single stocks / sector ETFs +/-15%, and leveraged ETFs at leverage x
the base (45% for 3x) -- with a rules-based fallback of 75% long / 90% short
on 3x ETFs and per-share minimums on cheap short stocks ($5/sh under $16.67,
100% under $5).  Because the tail-notional days are broad-index dip-buy
clusters, the tier matters for the binding GRM.  This script rebuilds the
open-notional book per trade (Entry..Exit inclusive, Shares_flat x Entry
Price, flat $750k) and computes:
  * requirement/NAV by day under six scenarios, and the feasible multiple m
    (GRM = 1.5 m) at the max / p99 / p95 day on $750k and on the live ~$632k
  * the same with broad-index legs routed to index futures (5.5% initial)
  * long / short / net / gross exposure distribution and a -30% one-day
    equity-shock loss (IBKR exposure-fee proxy) as a fraction of NAV
  * margin dollars consumed per dollar of ATR risk, by strategy and class
Writes growthmax_1_margin_tiered.json beside it.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import strategy_config as sc  # noqa: E402

NAV = 750_000.0
LIVE_NLV = 632_000.0          # primary account, memory note 2026-08-18
GRM_NOW = 1.5
OUT: dict = {"basis": "flat $750k, ledger data/backtest_trades_full.parquet as on disk 2026-09-02", "live_nlv_assumed": LIVE_NLV}

led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
led = led[led["PnL_flat_750k"].notna()].copy()
led["notional"] = led["Entry Price"] * led["Shares_flat"]
led["is_short"] = led["Direction"].eq("Short")

LEV3X = set(sc.LEV3X_ALL)
BEAR_EQ = set(sc.LEV3X_BEAR_EQ)
BULL_EQ = {"SPXL", "TQQQ", "UDOW", "TNA", "MIDU", "SOXL", "TECL", "FAS", "LABU", "WEBL", "CURE", "RETL", "NAIL", "DPST", "DFEN", "EDC", "YINN", "BRZU", "MEXX", "DRN"}
BROAD = {"SPY", "QQQ", "DIA", "^GSPC", "^NDX", "VOO", "IVV", "RSP", "EFA", "EEM", "VEA", "VWO"}
SMALLCAP_IDX = {"IWM", "MDY", "IJR", "IJH"}
SECTOR = {"XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC", "GLD", "SLV", "TLT", "IEF", "LQD", "HYG", "USO", "UNG", "GDX", "GDXJ", "XOP", "XBI", "SMH", "KRE", "XME", "XRT", "XHB", "ITB", "IYR", "VNQ", "FXI", "EWZ", "EWJ", "EWY", "EWT", "EWG", "EWU", "EWC", "EWW", "EWA", "EWH", "INDA", "KWEB", "ARKK", "IBB", "OIH", "TAN", "URA", "LIT", "DBC", "DBA", "UUP", "FXE", "FXY", "SLX", "COPX", "PPLT", "PALL", "CORN", "WEAT", "SOYB", "WOOD", "REMX", "SIL", "JETS", "PHO", "MOO", "BOTZ", "SOXX", "HACK", "IGV", "VGT", "XLG", "SPHB", "SPLV", "VXX", "UVXY", "SVXY", "TLH", "TBT", "BND", "AGG", "CEF", "IHI", "ITA"}


def klass(t: str) -> str:
    if t in LEV3X:
        return "lev3x"
    if t in BROAD:
        return "broad_idx"
    if t in SMALLCAP_IDX:
        return "smallcap_idx"
    if t in SECTOR:
        return "sector_etf"
    return "single"


led["klass"] = led["Ticker"].map(klass)
# equity beta for the -30% shock (per $ notional, signed by direction)
def beta_of(t: str) -> float:
    if t in BULL_EQ:
        return 3.0
    if t in BEAR_EQ:
        return -3.0
    if t in LEV3X:
        return 0.0          # commodity / bond 3x: no equity beta in the shock
    if t in {"GLD", "SLV", "TLT", "IEF", "LQD", "BND", "AGG", "UUP", "FXE", "FXY", "USO", "UNG", "DBC", "DBA", "CORN", "WEAT", "SOYB", "PPLT", "PALL", "CEF", "TLH", "TBT"}:
        return 0.0
    if t in {"HYG", "VXX", "UVXY", "SVXY"}:
        return 0.5 if t == "HYG" else (-4.0 if t in {"VXX", "UVXY"} else 3.0)
    return 1.0
led["beta"] = led["Ticker"].map(beta_of) * np.where(led["is_short"], -1.0, 1.0)

# ---------------------------------------------------------------- scenario rates per trade
TIMS = {"broad_idx": 0.08, "smallcap_idx": 0.10, "sector_etf": 0.15, "single": 0.15, "lev3x": 0.45}
FLAT15 = {"broad_idx": 0.15, "smallcap_idx": 0.15, "sector_etf": 0.15, "single": 0.15, "lev3x": 0.45}
FUT = 0.055   # ES/NQ/RTY/MES initial margin as a share of notional (~5-6%)


def rate_series(rates: dict, lev_rules: bool = False, cheap_short: bool = False, futures_broad: bool = False) -> np.ndarray:
    r = led["klass"].map(rates).values.astype(float)
    if lev_rules:  # rules-based 3x: 75% long / 90% short, and IBKR takes the higher of rules and PM
        is3 = led["klass"].eq("lev3x").values
        r = np.where(is3, np.where(led["is_short"].values, 0.90, 0.75), r)
    if futures_broad:
        r = np.where(led["klass"].isin(["broad_idx", "smallcap_idx"]).values, FUT, r)
    req = r * led["notional"].values
    if cheap_short:  # Reg-T short minimums that carry into PM for low-priced names
        px = led["Entry Price"].values; sh = led["Shares_flat"].values
        short_single = (led["is_short"] & led["klass"].eq("single")).values
        floor = np.where(px < 5.0, led["notional"].values, np.where(px < 16.67, 5.0 * sh, 0.0))
        req = np.where(short_single, np.maximum(req, floor), req)
    return req


SCEN = {
    "flat15_pm": dict(rates=FLAT15),                                            # the unconstrained study's base case
    "tims_pm": dict(rates=TIMS),                                                # tiered TIMS, 3x at 45%
    "tims_pm_cheapshort": dict(rates=TIMS, cheap_short=True),
    "tims_lev_rules": dict(rates=TIMS, lev_rules=True, cheap_short=True),       # 3x at rules-based 75/90 (IBKR 'higher of')
    "tims_pm_futures_broad": dict(rates=TIMS, cheap_short=True, futures_broad=True),
    "tims_lev_rules_futures_broad": dict(rates=TIMS, lev_rules=True, cheap_short=True, futures_broad=True),
}
per_trade_req = {k: rate_series(**v) for k, v in SCEN.items()}

# ---------------------------------------------------------------- accumulate by day
idx = pd.bdate_range("2003-01-01", "2026-09-01")
d0 = idx.searchsorted(led["Entry Date"].values); d1 = idx.searchsorted(led["Exit Date"].values)
n_days = len(idx)


def accumulate(vals: np.ndarray) -> pd.Series:
    out = np.zeros(n_days)
    for a, b, v in zip(d0, d1, vals):
        out[a:b + 1] += v
    return pd.Series(out, index=idx)


req_day = {k: accumulate(v) for k, v in per_trade_req.items()}
long_not = accumulate(np.where(~led["is_short"], led["notional"], 0.0))
short_not = accumulate(np.where(led["is_short"], led["notional"], 0.0))
beta_not = accumulate(led["beta"].values * led["notional"].values)
gross = long_not + short_not
net = long_not - short_not

# concentration add-on: per-ticker open notional, any ticker above thr*NAV charged +15pp (30% total) on the excess-eligible notional
tick = led["Ticker"].values
uniq = pd.Index(sorted(set(tick)))
tk_pos = {t: i for i, t in enumerate(uniq)}
tk_mat = np.zeros((n_days, len(uniq)))
for a, b, t, v in zip(d0, d1, tick, led["notional"].values):
    tk_mat[a:b + 1, tk_pos[t]] += v
tk_is3 = np.array([t in LEV3X for t in uniq])


def conc_addon(thr: float) -> pd.Series:
    conc = np.where(tk_mat > thr * NAV, tk_mat, 0.0)
    return pd.Series((conc * np.where(tk_is3, 0.45, 0.15)).sum(axis=1), index=idx)


req_day["tims_pm_cheapshort_conc25"] = req_day["tims_pm_cheapshort"] + conc_addon(0.25)
req_day["tims_pm_cheapshort_conc50"] = req_day["tims_pm_cheapshort"] + conc_addon(0.50)
req_day["tims_lev_rules_conc25"] = req_day["tims_lev_rules"] + conc_addon(0.25)

# ---------------------------------------------------------------- exposure distribution
def dist(s: pd.Series, start: str = "2016-01-01") -> dict:
    x = s[s.index >= start] / NAV
    return dict(p50=float(x.quantile(.5)), p90=float(x.quantile(.9)), p95=float(x.quantile(.95)), p99=float(x.quantile(.99)), max=float(x.max()), max_date=str(x.idxmax().date()))


OUT["exposure_pct_nav_2016plus"] = dict(gross=dist(gross), long=dist(long_not), short=dist(short_not), net=dist(net), beta_notional=dist(beta_not))
OUT["exposure_pct_nav_2003plus"] = dict(gross=dist(gross, "2003-01-01"), net=dist(net, "2003-01-01"), beta_notional=dist(beta_not, "2003-01-01"))
shock = 0.30 * beta_not.clip(lower=0)            # loss in a -30% one-day equity shock (only net-long beta loses)
OUT["equity_shock_30pct_loss_pct_nav"] = dict(dist2016=dist(shock), m_at_which_max_loss_equals_nav=float(NAV / shock[shock.index >= "2016-01-01"].max()),
                                              days_per_year_loss_over_30pct_nav_at_m1=float((shock[shock.index >= "2016-01-01"] > 0.30 * NAV).mean() * 252))
print("=== exposure / NAV (2016+) ===")
for k, v in OUT["exposure_pct_nav_2016plus"].items():
    print(f"  {k:14s} p50 {v['p50']:.0%} p95 {v['p95']:.0%} p99 {v['p99']:.0%} max {v['max']:.0%} ({v['max_date']})")
_e = OUT["equity_shock_30pct_loss_pct_nav"]
print(f"  -30% equity shock loss / NAV: p95 {_e['dist2016']['p95']*100:.1f}% p99 {_e['dist2016']['p99']*100:.1f}% max {_e['dist2016']['max']*100:.1f}% -> equals NAV at m {_e['m_at_which_max_loss_equals_nav']:.2f}")

# ---------------------------------------------------------------- feasibility table
print("\n=== margin requirement / NAV and feasible multiple m (GRM = 1.5 m) ===")
OUT["scenarios"] = {}
for k, q in req_day.items():
    x16 = q[q.index >= "2016-01-01"] / NAV
    xall = q / NAV
    row = dict(p50=float(x16.quantile(.5)), p95=float(x16.quantile(.95)), p99=float(x16.quantile(.99)), max=float(x16.max()), max_date=str(x16.idxmax().date()),
               m_max_750=float(1 / x16.max()), m_p99_750=float(1 / x16.quantile(.99)), m_p95_750=float(1 / x16.quantile(.95)),
               m_max_live=float(LIVE_NLV / NAV / x16.max()), m_p99_live=float(LIVE_NLV / NAV / x16.quantile(.99)),
               m_max_750_2003plus=float(1 / xall.max()), max_date_2003plus=str(xall.idxmax().date()),
               # with a 15% equity cushion (survive a same-day 15% loss without a forced liquidation)
               m_max_750_cushion15=float(0.85 / x16.max()), m_p99_live_cushion15=float(0.85 * LIVE_NLV / NAV / x16.quantile(.99)))
    row["grm_max_750"] = row["m_max_750"] * GRM_NOW; row["grm_p99_live"] = row["m_p99_live"] * GRM_NOW; row["grm_max_live"] = row["m_max_live"] * GRM_NOW
    OUT["scenarios"][k] = row
    print(f"  {k:30s} req/NAV p95 {row['p95']:.0%} p99 {row['p99']:.0%} max {row['max']:.0%} ({row['max_date']}) | m: max@750k {row['m_max_750']:.2f} (GRM {row['grm_max_750']:.2f}), "
          f"p99@live {row['m_p99_live']:.2f} (GRM {row['grm_p99_live']:.2f}), max@live {row['m_max_live']:.2f}, max@750k+15%cushion {row['m_max_750_cushion15']:.2f}; 2003+ max {row['m_max_750_2003plus']:.2f}")

# days per year the requirement exceeds 70% of NAV at various m (the guard's trigger frequency)
print("\n=== days/yr with requirement > 70% NAV (guard frequency), tims_pm_cheapshort, 2016+ ===")
q = req_day["tims_pm_cheapshort"]; x = q[q.index >= "2016-01-01"] / NAV
OUT["guard_days_per_year_over_70pct"] = {f"{m:g}": float((x * m > 0.70).mean() * 252) for m in [1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]}
OUT["guard_days_per_year_over_100pct"] = {f"{m:g}": float((x * m > 1.00).mean() * 252) for m in [1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]}
print("  >70%:", {k: round(v, 1) for k, v in OUT["guard_days_per_year_over_70pct"].items()})
print("  >100%:", {k: round(v, 1) for k, v in OUT["guard_days_per_year_over_100pct"].items()})

# top requirement days and their composition (tims_pm_cheapshort)
top = x.sort_values(ascending=False).head(8)
comp = []
for d, v in top.items():
    i = idx.get_loc(d)
    on = (d0 <= i) & (d1 >= i)
    sub = led[on]
    cl = (sub.groupby("klass")["notional"].sum() / sub["notional"].sum()).round(2).to_dict()
    st = (sub.groupby("Strategy")["notional"].sum() / sub["notional"].sum()).sort_values(ascending=False).head(3).round(2).to_dict()
    comp.append(dict(date=str(d.date()), req_pct_nav=float(v), gross_pct_nav=float(gross[d] / NAV), net_pct_nav=float(net[d] / NAV), classes=cl, top_strats=st))
OUT["top_requirement_days"] = comp
print("\n=== top requirement days (tims_pm_cheapshort) ===")
for c in comp:
    print(f"  {c['date']} req {c['req_pct_nav']:.0%} gross {c['gross_pct_nav']:.0%} net {c['net_pct_nav']:.0%} {c['classes']} {c['top_strats']}")

# ---------------------------------------------------------------- margin per $ of ATR risk, by strategy and class
print("\n=== margin $ per $ of ATR risk (tims_pm_cheapshort), by strategy ===")
led["req_tims"] = per_trade_req["tims_pm_cheapshort"]
led["req_flat15"] = per_trade_req["flat15_pm"]
led["req_lev_rules"] = per_trade_req["tims_lev_rules"]
g = led.groupby("Strategy").agg(N=("notional", "size"), notional=("notional", "sum"), risk=("Risk_flat_750k", "sum"), req_tims=("req_tims", "sum"), req_flat15=("req_flat15", "sum"),
                                req_rules=("req_lev_rules", "sum"), pnl=("PnL_flat_750k", "sum"), hold=("hold_days_target", "mean"))
g["notional_per_risk"] = g["notional"] / g["risk"]
g["margin_per_risk_tims"] = g["req_tims"] / g["risk"]
g["margin_per_risk_rules"] = g["req_rules"] / g["risk"]
g["margin_per_risk_flat15"] = g["req_flat15"] / g["risk"]
g["pnl_per_margin_day"] = g["pnl"] / (g["req_tims"] * g["hold"].clip(lower=1))     # $ PnL per $ of margin-day consumed
g = g.sort_values("margin_per_risk_tims")
print(g[["N", "notional_per_risk", "margin_per_risk_tims", "margin_per_risk_rules", "margin_per_risk_flat15", "hold", "pnl_per_margin_day"]].round(3).to_string())
OUT["margin_per_risk_by_strategy"] = g.round(4).reset_index().to_dict("records")
gc = led.groupby("klass").agg(N=("notional", "size"), notional=("notional", "sum"), risk=("Risk_flat_750k", "sum"), req_tims=("req_tims", "sum"))
gc["margin_per_risk_tims"] = gc["req_tims"] / gc["risk"]; gc["notional_per_risk"] = gc["notional"] / gc["risk"]
print(gc.round(3).to_string()); OUT["margin_per_risk_by_class"] = gc.round(4).reset_index().to_dict("records")
book_mpr = float(led["req_tims"].sum() / led["Risk_flat_750k"].sum())
OUT["book_margin_per_risk_tims"] = book_mpr
print(f"book margin per $ risk (TIMS tiered): {book_mpr:.2f}; flat15: {led['req_flat15'].sum() / led['Risk_flat_750k'].sum():.2f}; rules-3x: {led['req_lev_rules'].sum() / led['Risk_flat_750k'].sum():.2f}")

json.dump(OUT, open(HERE / "growthmax_1_margin_tiered.json", "w"), indent=1, default=float)
print("\nwrote growthmax_1_margin_tiered.json")
