"""Unconstrained growth, part 2b: refine the margin boundary.  (i) single-ticker
concentration at m=1 (the max-notional days are SPY/QQQ dip-buy clusters, and a
broad-index ETF is what portfolio margin is built for), (ii) a scenario where
the concentration add-on applies to NON-broad tickers only, (iii) the live
primary NLV (~$632k on 2026-08-18) instead of the $750k sizing constant, and
(iv) the m at which the requirement crosses NAV on the max / p99 / p95 day for
every scenario, as the feasibility table.  Writes unconstrained_growth_02b_margin_refine.json.
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
NAV = 750_000.0; LIVE_NLV = 632_000.0; GRM_NOW = 1.5
OUT: dict = {"live_nlv_assumed": LIVE_NLV, "note": "live NLV from memory note dated 2026-08-18; flat sizing base is 750k"}

led = pd.read_parquet(ROOT / "data/backtest_trades_full.parquet")
led = led[led["PnL_flat_750k"].notna()].copy()
led["notional"] = led["Entry Price"] * led["Shares_flat"]
LEV3X = set(sc.LEV3X_ALL)
BROAD = {"SPY", "QQQ", "IWM", "DIA", "^GSPC", "^NDX", "VOO", "IVV", "MDY", "IJR", "IJH", "RSP", "EFA", "EEM", "VEA", "VWO"}
alias = {"^GSPC": "SPY", "^NDX": "QQQ"}
led["tk"] = led["Ticker"].map(lambda t: alias.get(t, t))
led["is3x"] = led["tk"].isin(LEV3X); led["broad"] = led["tk"].isin(BROAD)
idx = pd.bdate_range("2003-01-01", "2026-09-01")
tks = led["tk"].unique(); pos = {k: i for i, k in enumerate(tks)}
long_m = np.zeros((len(idx), len(tks))); short_m = np.zeros_like(long_m)
d0 = idx.searchsorted(led["Entry Date"].values); d1 = idx.searchsorted(led["Exit Date"].values)
for a, b, k, n, dr in zip(d0, d1, led["tk"].values, led["notional"].values, led["Direction"].values):
    (long_m if dr == "Long" else short_m)[a:b + 1, pos[k]] += n
gross_t = long_m + short_m
net_t = long_m - short_m
by_tkr = pd.DataFrame(gross_t, index=idx, columns=tks)
W = by_tkr[by_tkr.index >= "2016-01-01"]

# ------------------------------------------------------------ 1. single-ticker concentration at m=1
mx = W.max(axis=1) / NAV
top_name = W.idxmax(axis=1)
print("=== 1. largest single-ticker open notional / NAV at m=1 (2016+) ===")
print(f"p50 {mx.quantile(.5):.0%} p95 {mx.quantile(.95):.0%} p99 {mx.quantile(.99):.0%} max {mx.max():.0%} on {mx.idxmax().date()} ({top_name[mx.idxmax()]})")
days_over = {f"{x:g}": float((mx > x).mean() * 252) for x in [0.25, 0.5, 1.0, 1.5, 2.0]}
print("days/yr with a single ticker over 25/50/100/150/200% NAV:", {k: round(v, 1) for k, v in days_over.items()})
big = (W.max(axis=1) > 0.5 * NAV)
print("tickers on days with a >50% NAV single position:", W[big].idxmax(axis=1).value_counts().head(8).to_dict())
OUT["single_ticker_concentration_at_1x"] = dict(p50=float(mx.quantile(.5)), p95=float(mx.quantile(.95)), p99=float(mx.quantile(.99)), max=float(mx.max()), max_date=str(mx.idxmax().date()),
                                                max_ticker=str(top_name[mx.idxmax()]), days_per_year_over=days_over,
                                                tickers_over_50pct=W[big].idxmax(axis=1).value_counts().head(8).to_dict())
# non-broad concentration
nb = [t for t in tks if t not in BROAD]
mx_nb = W[nb].max(axis=1) / NAV
print(f"largest NON-broad single position: p95 {mx_nb.quantile(.95):.0%} p99 {mx_nb.quantile(.99):.0%} max {mx_nb.max():.0%} ({W[nb].idxmax(axis=1)[mx_nb.idxmax()]} on {mx_nb.idxmax().date()})")
OUT["single_ticker_concentration_at_1x"]["nonbroad"] = dict(p95=float(mx_nb.quantile(.95)), p99=float(mx_nb.quantile(.99)), max=float(mx_nb.max()), max_ticker=str(W[nb].idxmax(axis=1)[mx_nb.idxmax()]), max_date=str(mx_nb.idxmax().date()))

# ------------------------------------------------------------ 2. requirement scenarios (per-ticker rates), incl. non-broad-only concentration add-on
is3 = np.array([t in LEV3X for t in tks]); isb = np.array([t in BROAD for t in tks])
def req(base, broad_rate, lev_mult, conc_thr, conc_rate, conc_broad: bool):
    rate = np.where(isb, broad_rate, base) * np.where(is3, lev_mult, 1.0)
    R = gross_t * rate[None, :]
    if conc_thr is not None:
        conc_mask = gross_t > conc_thr * NAV
        if not conc_broad:
            conc_mask &= ~isb[None, :]
        addon = np.where(conc_mask, gross_t * (conc_rate - np.where(isb, broad_rate, base)) * np.where(is3, lev_mult, 1.0)[None, :], 0.0)
        R = R + addon
    return pd.Series(R.sum(axis=1), index=idx)
SCEN = {
    "pm_15_broad10": dict(base=0.15, broad_rate=0.10, lev_mult=3, conc_thr=None, conc_rate=0.15, conc_broad=False),
    "pm_15": dict(base=0.15, broad_rate=0.15, lev_mult=3, conc_thr=None, conc_rate=0.15, conc_broad=False),
    "pm_15_conc30_nonbroad": dict(base=0.15, broad_rate=0.15, lev_mult=3, conc_thr=0.25, conc_rate=0.30, conc_broad=False),
    "pm_15_conc30_all": dict(base=0.15, broad_rate=0.15, lev_mult=3, conc_thr=0.25, conc_rate=0.30, conc_broad=True),
    "pm_25_conc30_nonbroad": dict(base=0.25, broad_rate=0.15, lev_mult=3, conc_thr=0.25, conc_rate=0.30, conc_broad=False),
    "pm_30_stress_all": dict(base=0.30, broad_rate=0.20, lev_mult=3, conc_thr=0.15, conc_rate=0.40, conc_broad=True),
    "regT_50": dict(base=0.50, broad_rate=0.50, lev_mult=1.8, conc_thr=None, conc_rate=0.5, conc_broad=False),
}
print("\n=== 2. feasibility table: requirement/NAV at m=1 and the m (GRM) at which it reaches 100% of equity ===")
OUT["feasibility"] = {}
for k, v in SCEN.items():
    q = req(**v); qq = q[q.index >= "2016-01-01"] / NAV; qf = q / NAV
    row = dict(p50=float(qq.quantile(.5)), p95=float(qq.quantile(.95)), p99=float(qq.quantile(.99)), max=float(qq.max()), max_date=str(qq.idxmax().date()),
               m_at_max=float(1 / qq.max()), m_at_p99=float(1 / qq.quantile(.99)), m_at_p95=float(1 / qq.quantile(.95)),
               m_at_max_2003=float(1 / qf.max()), max_date_2003=str(qf.idxmax().date()),
               m_at_max_live_nlv=float(LIVE_NLV / NAV / qq.max()), m_at_p99_live_nlv=float(LIVE_NLV / NAV / qq.quantile(.99)),
               days_per_year_req_over_50pct_at_m1=float((qq > 0.5).mean() * 252), days_per_year_req_over_100pct_at_m1=float((qq > 1.0).mean() * 252))
    row.update({f"grm_at_{s}": row[f"m_at_{s}"] * GRM_NOW for s in ["max", "p99", "p95", "max_live_nlv"]})
    OUT["feasibility"][k] = row
    print(f"{k:24s} req/NAV p95 {row['p95']:5.1%} p99 {row['p99']:5.1%} max {row['max']:6.1%} ({row['max_date']}) | m at max {row['m_at_max']:.2f} (GRM {row['grm_at_max']:.2f}), "
          f"p99 {row['m_at_p99']:.2f}, p95 {row['m_at_p95']:.2f} | live NLV: m at max {row['m_at_max_live_nlv']:.2f} | 2003+ max-day m {row['m_at_max_2003']:.2f} ({row['max_date_2003']})")

# ------------------------------------------------------------ 3. what fraction of the requirement tail is the long dip-buy cluster on SPY/QQQ?
q = req(**SCEN["pm_15"]); qq = q[q.index >= "2016-01-01"]
hi = qq[qq >= qq.quantile(.99)].index
bshare = float((gross_t[idx.isin(hi)][:, isb].sum()) / gross_t[idx.isin(hi)].sum())
net_share = float(np.abs(net_t[idx.isin(hi)].sum(axis=1)).mean() / gross_t[idx.isin(hi)].sum(axis=1).mean())
print(f"\n=== 3. on the top-1% requirement days (pm_15), broad-index ETFs are {bshare:.0%} of gross and |net|/gross = {net_share:.0%} (long-side dominated) ===")
OUT["tail_composition_top1pct_days"] = dict(broad_share=bshare, abs_net_over_gross=net_share, n_days=int(len(hi)))
# index-hedged variant: if the SPY/QQQ long cluster were beta-hedged with futures (futures margin ~5-6% of notional both legs)
# requirement ~ 15% on the stock leg is unchanged; hedging does not free margin. Note instead the MES margin add-on.
OUT["hedge_note"] = "A beta hedge via index futures does NOT reduce the equity leg's requirement (futures margin is additive, ~5-6% of notional); only a gross cap, a lower-rate instrument (index futures for the SPY/QQQ legs at ~5% margin) or NAV growth relaxes the boundary."

json.dump(OUT, open(HERE / "unconstrained_growth_02b_margin_refine.json", "w"), indent=1, default=float)
print("\nwrote", HERE / "unconstrained_growth_02b_margin_refine.json")
