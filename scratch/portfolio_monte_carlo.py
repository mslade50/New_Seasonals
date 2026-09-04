"""Portfolio Monte Carlo (2026-07-28): distribution of day/month/year outcomes
for the CURRENT book construction (post-stacking ledger, flat $750k sizing).

Basis: daily mark-to-market PnL rebuilt from the deployed R2 ledger vintage
(gha:30349268135, 4,740 trades incl. 3x-fade stacking) via the site's own
get_daily_mtm_series convention, so every day reconciles to booked trade PnL.

Method: empirical stats straight from the 23y daily series, plus a stationary
block bootstrap (Politis-Romano, mean block 10 trading days, circular) to get
horizon distributions that keep vol clustering. Blocks scramble the calendar,
so month/year seasonality is NOT preserved - this is a "basic" MC by design.

Dollar thresholds are on the $750k sizing basis (live sizes off ACCOUNT_VALUE
regardless of actual NAV): 1.5% = $11,250. Same dollars against the actual
~$620k live NAV are ~1.8%, so the % lines are also shown at the live-NAV
equivalent threshold ($9,300).
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import data_provider
from pages.strat_backtester import get_daily_mtm_series

NAV = 750_000.0
LEDGER = os.path.join(_ROOT, "data", "_r2_ledger_check.parquet")
N_SIMS = 10_000
MEAN_BLOCK = 10
SEED = 42
BANDS = [5, 25, 50, 75, 95]


def load_daily():
    df = pd.read_parquet(LEDGER)
    sig = (df.drop(columns=["Shares"])  # compounded-basis shares; use flat
             .rename(columns={"Entry Price": "Price", "PnL_flat_750k": "PnL",
                              "Shares_flat": "Shares"}))
    tickers = sorted(set(sig["Ticker"].astype(str).str.replace(".", "-", regex=False)))
    md = data_provider.get_history(tickers, start="2002-01-01")
    daily = get_daily_mtm_series(sig, md)
    # trim to the ledger's actual data range (the series pads to wall-clock today)
    last = pd.to_datetime(sig["Exit Date"]).max()
    daily = daily[(daily.index >= pd.Timestamp("2003-01-01")) & (daily.index <= last)]
    # active-day mask from the union of trade intervals
    active = pd.Series(False, index=daily.index)
    en = pd.to_datetime(sig["Entry Date"])
    ex = pd.to_datetime(sig["Exit Date"])
    for a, b in zip(en.values, ex.values):
        active.loc[a:b] = True
    return daily, active, df


def empirical(daily, active, label):
    pct = daily / NAV * 100
    up_all = (daily > 0).mean()
    act = daily[active]
    print(f"\n=== EMPIRICAL DAILY ({label}) ===")
    print(f"days={len(daily)}  active={active.mean() * 100:.0f}%  "
          f"P(up, all days)={up_all * 100:.1f}%  "
          f"P(up | active)={(act > 0).mean() * 100:.1f}%  "
          f"P(flat)={(daily == 0).mean() * 100:.1f}%")
    print(f"mean=${daily.mean():,.0f}/d  ann=${daily.mean() * 252:,.0f}  "
          f"std=${daily.std():,.0f}  skew={pct.skew():.2f}  "
          f"annSharpe={daily.mean() / daily.std() * np.sqrt(252):.2f}")
    print(f"VaR95=${-np.percentile(daily, 5):,.0f}  "
          f"VaR99=${-np.percentile(daily, 1):,.0f}  "
          f"CVaR99=${-daily[daily <= np.percentile(daily, 1)].mean():,.0f}")
    for thr_pct, thr in ((1.0, 7_500), (1.5, 11_250), (1.5, 9_300), (2.0, 15_000), (3.0, 22_500)):
        n = (daily < -thr).sum()
        per_yr = n / (len(daily) / 252)
        nav_note = "live-NAV eq" if thr == 9_300 else "of 750k"
        print(f"  daily loss > ${thr:,} ({thr_pct}% {nav_note}): {n}x "
              f"({per_yr:.1f}/yr, ~1 per {12 / per_yr:.1f} months)" if n else
              f"  daily loss > ${thr:,}: never in sample")
    worst = daily.nsmallest(5)
    print("worst days: " + "; ".join(f"{d.date()} ${v:,.0f}" for d, v in worst.items()))


def stationary_bootstrap_paths(vals, horizon, n_sims, rng):
    n = len(vals)
    p = 1.0 / MEAN_BLOCK
    out = np.empty((n_sims, horizon))
    for s in range(n_sims):
        idx = np.empty(horizon, dtype=int)
        i = rng.integers(n)
        for t in range(horizon):
            idx[t] = i
            if rng.random() < p:
                i = rng.integers(n)
            else:
                i = (i + 1) % n
        out[s] = vals[idx]
    return out


def horizon_report(paths, name):
    tot = paths.sum(axis=1)
    bands = np.percentile(tot, BANDS)
    eq = paths.cumsum(axis=1)
    dd = (eq - np.maximum.accumulate(eq, axis=1)).min(axis=1)
    print(f"\n=== {name} (block bootstrap, {len(tot):,} sims) ===")
    print("PnL bands ($):   " + "  ".join(f"p{p}={v:,.0f}" for p, v in zip(BANDS, bands)))
    print("PnL bands (%):   " + "  ".join(f"p{p}={v / NAV * 100:+.1f}%" for p, v in zip(BANDS, bands)))
    print(f"P(negative)={100 * (tot < 0).mean():.1f}%   "
          f"P(< -2% NAV)={100 * (tot < -0.02 * NAV).mean():.1f}%   "
          f"P(< -5% NAV)={100 * (tot < -0.05 * NAV).mean():.1f}%")
    print(f"P(>= 1 day < -1.5%)={100 * (paths < -11_250).any(axis=1).mean():.1f}%   "
          f"within-path maxDD: p50={np.percentile(dd, 50) / NAV * 100:.1f}%  "
          f"p95={np.percentile(dd, 5) / NAV * 100:.1f}%  "
          f"worst={dd.min() / NAV * 100:.1f}%")
    return tot


def main():
    daily, active, df = load_daily()
    booked = df["PnL_flat_750k"].sum()
    print(f"daily series {daily.index[0].date()} -> {daily.index[-1].date()}, "
          f"sum=${daily.sum():,.0f} vs booked ${booked:,.0f} "
          f"(diff ${daily.sum() - booked:,.0f})")

    empirical(daily, active, "full 2003+")
    modern = daily[daily.index >= "2020-01-01"]
    empirical(modern, active[active.index >= "2020-01-01"], "modern 2020+")

    rng = np.random.default_rng(SEED)
    vals = daily.values
    month = stationary_bootstrap_paths(vals, 21, N_SIMS, rng)
    year = stationary_bootstrap_paths(vals, 252, N_SIMS, rng)
    tot_m = horizon_report(month, "1 MONTH (21td)")
    tot_y = horizon_report(year, "1 YEAR (252td)")

    # calendar-anchored actuals for reference (real months/years, seasonality intact)
    cal_m = daily.resample("ME").sum()
    cal_y = daily.resample("YE").sum()
    print("\n=== CALENDAR ACTUALS (for reference) ===")
    print(f"months: n={len(cal_m)}  P(neg)={100 * (cal_m < 0).mean():.1f}%  "
          f"median=${cal_m.median():,.0f}  worst=${cal_m.min():,.0f} "
          f"({cal_m.idxmin().strftime('%Y-%m')})")
    print(f"years:  n={len(cal_y)}  P(neg)={100 * (cal_y < 0).mean():.1f}%  "
          f"median=${cal_y.median():,.0f}  worst=${cal_y.min():,.0f} "
          f"({cal_y.idxmin().year})")

    out = pd.DataFrame({"month_pnl": np.sort(tot_m), "year_pnl": np.sort(tot_y)})
    out.to_csv(os.path.join(_HERE, "portfolio_mc_distributions.csv"), index=False)
    daily.to_frame("pnl_flat").to_csv(os.path.join(_HERE, "portfolio_mc_daily_basis.csv"))
    print("\nwrote scratch/portfolio_mc_distributions.csv + portfolio_mc_daily_basis.csv")


if __name__ == "__main__":
    main()
