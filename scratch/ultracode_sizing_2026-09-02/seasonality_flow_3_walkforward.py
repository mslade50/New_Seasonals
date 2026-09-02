"""seasonality_flow_3_walkforward.py (2026-09-02): does a calendar-conditional sizing
multiplier improve the book OUT OF SAMPLE?

Mechanism: every trade's daily MTM vector (dist/data/trade_mtm.json, flat $750k) is
scaled by a multiplier keyed on (strategy, calendar cell of the SIGNAL date), the
book is re-summed day by day, and the result is compared with the unscaled book.
Two out-of-sample designs: WALK-FORWARD (fit on signal years < Y, apply to Y,
Y = 2010..2026) and LOYO (fit on all years != Y). Fitted rules are shrunk toward
1.0 with a pseudo-count N0 and clipped to [lo, hi]. Also: out-of-repo PRIORS that
are not fitted (Sep 0.5x, May-Oct 0.75x, earnings-season 0.75x on single-stock
strats, opex-week 0.75x / TOM 1.25x on dip-buys), evaluated on the full sample with
year-clustered stats. Every overlay is also reported vol-matched to the baseline.
Cap interactions (per-strategy 250 bps/day) are NOT replayed; a multiplier > 1
would bind them more often, so raises are upper bounds.
Writes seasonality_flow_walkforward.json.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats as sps
from seasonality_flow_common import (HERE, ROOT, NAV, MONTHS, load_ledger, load_trade_mtm, load_spy, trading_calendar,
                                     jdump, perf, maxdd)

pd.set_option("display.width", 260, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)
led = load_ledger()
dates, mtm = load_trade_mtm()
earn = pd.read_parquet(ROOT / "data/earnings_calendar.parquet", columns=["date"])
cal = trading_calendar(load_spy().index, earn)
led = led[led["sig"].isin(cal.index)].copy()
F = cal.loc[led["sig"]]
led["month"] = F["month"].values
led["quarter"] = F["quarter"].values
led["half"] = F["half"].values
led["tom"] = F["tom"].values
led["opex"] = F["opex_week"].values
led["eseason"] = F["eseason_data"].values
led["dow"] = F["dow"].values
led = led[led["trade_id"].isin(mtm.keys())].copy()
N_DAYS = len(dates)
DIP_BUYS = ["Weak Close Decent Sznls", "SPY QQQ MonFri Reversion", "Monday Dip", "Indices Oversold Bounce", "Monthly Weak Close"]
SINGLE_STOCK = ["Oversold Low Volume", "Overbot Vol Spike", "LT Trend ST OS", "52wh Breakout", "St OS Sznl", "ATR Extended Gap Up", "Sector BO"]

# trade -> (start, vector) ; book series builder
tid = led["trade_id"].values
starts = np.array([mtm[t][0] for t in tid])
vecs = [mtm[t][1] for t in tid]


def book(mults: np.ndarray) -> pd.Series:
    out = np.zeros(N_DAYS)
    for s, v, m in zip(starts, vecs, mults):
        if m == 1.0:
            out[s:s + len(v)] += v
        elif m != 0.0:
            out[s:s + len(v)] += v * m
    return pd.Series(out, index=dates)


base = book(np.ones(len(led)))
print("baseline check: sum trade pnl", led["pnl"].sum(), " book sum", base.sum())


def stats_vs(base: pd.Series, alt: pd.Series, lo: str, hi: str, label: str) -> dict:
    b = base[(base.index >= lo) & (base.index <= hi)]
    a = alt[(alt.index >= lo) & (alt.index <= hi)]
    pb, pa = perf(b), perf(a)
    scale = pb["ann_vol_pct"] / pa["ann_vol_pct"] if pa["ann_vol_pct"] > 0 else 1.0
    pv = perf(a * scale)
    yb, ya = b.groupby(b.index.year).sum(), a.groupby(a.index.year).sum()
    d = ya - yb
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 2 and d.std(ddof=1) > 0 else np.nan
    return dict(label=label, window=[lo, hi], base=pb, alt=pa, alt_volmatched=pv,
                d_total_pnl=pa["total_pnl"] - pb["total_pnl"], d_pnl_pct_of_base=(pa["total_pnl"] - pb["total_pnl"]) / pb["total_pnl"] * 100,
                d_sharpe=pa["sharpe"] - pb["sharpe"], d_sharpe_volmatched=pv["sharpe"] - pb["sharpe"],
                d_maxdd_pts=pa["maxdd_pct"] - pb["maxdd_pct"], d_pnl_over_maxdd_pct=(pa["pnl_over_maxdd"] / pb["pnl_over_maxdd"] - 1) * 100,
                years_better=int((d > 0).sum()), years=int(len(d)), worst_year_ratio=float((ya / yb).replace([np.inf, -np.inf], np.nan).min()),
                t_yearly_diff=float(t) if np.isfinite(t) else None)


# ---------------------------------------------------------------- fitted rules
def cell_mults(fit: pd.DataFrame, key_cols: list[str], by_strategy: bool, N0: float, lo: float, hi: float,
               kelly: bool = False) -> dict:
    """multiplier per (strategy?, cell) = shrunk ratio of cell expected R (or mu/sigma^2) to overall."""
    out = {}
    grp = ["Strategy"] + key_cols if by_strategy else key_cols
    if by_strategy:
        base_stats = fit.groupby("Strategy")["R"].agg(["mean", "var"])
    else:
        base_stats = pd.DataFrame({"mean": [fit["R"].mean()], "var": [fit["R"].var()]}, index=["ALL"])
    for k, g in fit.groupby(grp):
        s = k[0] if by_strategy else "ALL"
        mu0, v0 = base_stats.loc[s, "mean"], base_stats.loc[s, "var"]
        n = len(g)
        w = n / (n + N0)
        if mu0 <= 0 or not np.isfinite(mu0):
            out[k] = 1.0
            continue
        if kelly:
            v1 = g["R"].var() if n > 2 else v0
            stat1, stat0 = g["R"].mean() / max(v1, 1e-6), mu0 / max(v0, 1e-6)
        else:
            stat1, stat0 = g["R"].mean(), mu0
        ratio = (stat0 + w * (stat1 - stat0)) / stat0
        out[k] = float(np.clip(ratio, lo, hi))
    return out


RULES = {
    "strat_month": (["month"], True), "book_month": (["month"], False),
    "strat_quarter": (["quarter"], True), "strat_half": (["half"], False if False else True), "book_half": (["half"], False),
    "strat_eseason": (["eseason"], True), "strat_opex": (["opex"], True), "strat_tom": (["tom"], True), "strat_dow": (["dow"], True),
}
YEARS = list(range(2010, 2027))


def apply_rule(name, N0, lo, hi, kelly, design) -> pd.Series:
    key_cols, by_s = RULES[name]
    mults = np.ones(len(led))
    for Y in YEARS:
        fit = led[led["yr"] < Y] if design == "wf" else led[led["yr"] != Y]
        cm = cell_mults(fit, key_cols, by_s, N0, lo, hi, kelly)
        m = (led["yr"] == Y).values
        for i in np.where(m)[0]:
            k = tuple([led["Strategy"].iat[i]] + [led[c].iat[i] for c in key_cols]) if by_s else tuple(led[c].iat[i] for c in key_cols)
            mults[i] = cm.get(k, 1.0)
    return book(mults), mults


results = {"fitted": [], "priors": [], "in_sample_reference": []}
LO, HI = "2010-01-01", "2026-08-07"
print("\n=== FITTED calendar multipliers, out of sample 2010-2026 ===")
hdr = f"{'rule':16s}{'design':5s}{'N0':>5s}{'clip':>10s}{'kelly':>6s}{'dPnL%':>8s}{'dSharpe':>8s}{'dSh_vm':>8s}{'dMaxDD':>8s}{'dPnL/DD%':>9s}{'yrs+':>6s}{'worstYr':>8s}{'t_yr':>6s}{'meanMult':>9s}"
print(hdr)
for name in RULES:
    for design in ["wf", "loyo"]:
        for N0, lo, hi, kelly in [(30, 0.5, 1.5, False), (60, 0.7, 1.3, False), (30, 0.5, 1.5, True)]:
            if name != "strat_month" and (N0, kelly) != (30, False):
                continue
            alt, mults = apply_rule(name, N0, lo, hi, kelly, design)
            st = stats_vs(base, alt, LO, HI, f"{name}|{design}|N0={N0}|clip={lo}-{hi}|kelly={kelly}")
            st.update(rule=name, design=design, N0=N0, clip=[lo, hi], kelly=kelly,
                      mean_mult_riskw=float(np.average(mults[(led['yr'] >= 2010).values], weights=led.loc[led['yr'] >= 2010, 'risk'])))
            results["fitted"].append(st)
            print(f"{name:16s}{design:5s}{N0:5d}{f'{lo}-{hi}':>10s}{str(kelly):>6s}{st['d_pnl_pct_of_base']:8.1f}{st['d_sharpe']:8.3f}{st['d_sharpe_volmatched']:8.3f}{st['d_maxdd_pts']:8.2f}{st['d_pnl_over_maxdd_pct']:9.1f}{st['years_better']:4d}/{st['years']:<2d}{st['worst_year_ratio']:8.2f}{(st['t_yearly_diff'] or 0):6.2f}{st['mean_mult_riskw']:9.3f}")

# in-sample reference (fit on everything, apply everywhere) -- the number a naive study would report
for name in ["strat_month", "book_month"]:
    key_cols, by_s = RULES[name]
    cm = cell_mults(led, key_cols, by_s, 30, 0.5, 1.5, False)
    mults = np.array([cm.get(tuple([led["Strategy"].iat[i]] + [led[c].iat[i] for c in key_cols]) if by_s else tuple(led[c].iat[i] for c in key_cols), 1.0) for i in range(len(led))])
    st = stats_vs(base, book(mults), LO, HI, f"{name}|IN-SAMPLE")
    st.update(rule=name, design="in_sample")
    results["in_sample_reference"].append(st)
    print(f"{name:16s}IS   {'':5s}{'':10s}{'':6s}{st['d_pnl_pct_of_base']:8.1f}{st['d_sharpe']:8.3f}{st['d_sharpe_volmatched']:8.3f}{st['d_maxdd_pts']:8.2f}{st['d_pnl_over_maxdd_pct']:9.1f}{st['years_better']:4d}/{st['years']:<2d}{st['worst_year_ratio']:8.2f}")

# ---------------------------------------------------------------- priors (no fit)
def prior_mults(fn) -> np.ndarray:
    return np.array([fn(r) for r in led.itertuples(index=False)], dtype=float)


PRIORS = {
    "book_sep_0.5x": lambda r: 0.5 if r.month == 9 else 1.0,
    "book_augsep_0.75x": lambda r: 0.75 if r.month in (8, 9) else 1.0,
    "book_mayoct_0.75x": lambda r: 0.75 if r.half == "MayOct" else 1.0,
    "book_novapr_1.25x": lambda r: 1.25 if r.half == "NovApr" else 1.0,
    "book_dec_1.25x": lambda r: 1.25 if r.month == 12 else 1.0,
    "book_q4_1.25x": lambda r: 1.25 if r.quarter == 4 else 1.0,
    "singlestock_eseason_0.75x": lambda r: 0.75 if (r.eseason and r.Strategy in SINGLE_STOCK) else 1.0,
    "singlestock_eseason_1.25x": lambda r: 1.25 if (r.eseason and r.Strategy in SINGLE_STOCK) else 1.0,
    "dipbuy_opex_0.75x": lambda r: 0.75 if (r.opex and r.Strategy in DIP_BUYS) else 1.0,
    "dipbuy_tom_1.25x": lambda r: 1.25 if (r.tom and r.Strategy in DIP_BUYS) else 1.0,
    "book_tom_0.75x": lambda r: 0.75 if r.tom else 1.0,
    "olv_aug_0.5x": lambda r: 0.5 if (r.month == 8 and r.Strategy == "Oversold Low Volume") else 1.0,
    "olv_sepdec_0.5x": lambda r: 0.5 if (r.month in (9, 11, 12) and r.Strategy == "Oversold Low Volume") else 1.0,
    "lttrend_jun_0.5x": lambda r: 0.5 if (r.month == 6 and r.Strategy == "LT Trend ST OS") else 1.0,
    "b52wh_oct_1.5x": lambda r: 1.5 if (r.month == 10 and r.Strategy == "52wh Breakout") else 1.0,
    "ovs_dec_1.25x": lambda r: 1.25 if (r.month == 12 and r.Strategy == "Overbot Vol Spike") else 1.0,
}
print("\n=== PRIORS (unfitted) on full sample 2003-2026 and on 2010-2026 ===")
print(f"{'prior':28s}{'win':11s}{'dPnL%':>8s}{'dSharpe':>8s}{'dSh_vm':>8s}{'dMaxDD':>8s}{'dPnL/DD%':>9s}{'yrs+':>6s}{'worstYr':>8s}{'t_yr':>6s}")
for name, fn in PRIORS.items():
    mults = prior_mults(fn)
    alt = book(mults)
    for lo, hi in [("2003-01-01", "2026-08-07"), ("2010-01-01", "2026-08-07")]:
        st = stats_vs(base, alt, lo, hi, name)
        st.update(prior=name, n_trades_touched=int((mults != 1).sum()))
        results["priors"].append(st)
        print(f"{name:28s}{lo[:4]}-{hi[:4]}  {st['d_pnl_pct_of_base']:8.1f}{st['d_sharpe']:8.3f}{st['d_sharpe_volmatched']:8.3f}{st['d_maxdd_pts']:8.2f}{st['d_pnl_over_maxdd_pct']:9.1f}{st['years_better']:4d}/{st['years']:<2d}{st['worst_year_ratio']:8.2f}{(st['t_yearly_diff'] or 0):6.2f}")

# ---------------------------------------------------------------- the fitted month table (through 2025) for reference
cm = cell_mults(led[led["yr"] <= 2025], ["month"], True, 30, 0.5, 1.5, False)
tab = pd.Series(cm).unstack()
tab.columns = [MONTHS[c - 1] for c in tab.columns]
print("\n=== strat x month multiplier table fit through 2025 (N0=30, clip 0.5-1.5) -- what would be live in 2026 ===")
print(tab.round(2).to_string())
results["mult_table_through_2025"] = {k: {MONTHS[m - 1]: v for (kk, m), v in cm.items() if kk == k} for k in tab.index}
cmb = cell_mults(led[led["yr"] <= 2025], ["month"], False, 30, 0.5, 1.5, False)
results["book_month_mult_through_2025"] = {MONTHS[k[0] - 1]: v for k, v in cmb.items()}
print("book-level:", {MONTHS[k[0] - 1]: round(v, 2) for k, v in cmb.items()})
jdump(results, HERE / "seasonality_flow_walkforward.json")
print("wrote", HERE / "seasonality_flow_walkforward.json")
