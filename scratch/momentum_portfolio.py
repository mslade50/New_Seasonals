"""Monthly-rebalanced momentum portfolio backtest.

Realizes the validated edge from momentum_longhold.py as an actual tradeable
basket: each month-end, rank tradable CSV_UNIVERSE names by 12-1 month momentum,
hold the top-N equal-weight for the next month, rebalance. Reports an equity
curve with CAGR / vol / Sharpe / max drawdown vs SPY, by-year returns, turnover,
and an after-cost line.

Run:  python scratch/momentum_portfolio.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from strategy_config import CSV_UNIVERSE  # noqa: E402

PRICES_PATH = ROOT / "data" / "master_prices.parquet"
MIN_PRICE = 5.0
MIN_DOLLAR_VOL = 1_000_000
START_DATE = "2005-01-01"
TOP_N = 30           # equal-weight holdings
COST_BPS = 10.0      # one-way cost per name traded (slippage+commission)
MOM_SKIP = 1         # months skipped (skip most recent month)
MOM_LOOK = 12        # months lookback


def perf_stats(monthly_ret: pd.Series) -> dict:
    monthly_ret = monthly_ret.dropna()
    if monthly_ret.empty:
        return {}
    eq = (1 + monthly_ret).cumprod()
    yrs = len(monthly_ret) / 12.0
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = monthly_ret.std() * np.sqrt(12)
    sharpe = (monthly_ret.mean() * 12) / vol if vol else np.nan
    dd = (eq / eq.cummax() - 1).min()
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MaxDD": dd,
            "HitMo": (monthly_ret > 0).mean(), "FinalX": eq.iloc[-1]}


def main() -> None:
    print(f"Loading {PRICES_PATH.name} ...", flush=True)
    raw = pd.read_parquet(PRICES_PATH)
    keep = set(CSV_UNIVERSE) | {"SPY"}
    raw = raw[raw["ticker"].isin(keep)].copy()
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw[raw["date"] >= pd.Timestamp(START_DATE)]

    raw["dollar_vol"] = raw["Close"] * raw["Volume"]

    # month-end panels: close, and 20d-avg dollar volume as of month end
    raw = raw.sort_values(["ticker", "date"])
    raw["dv20"] = raw.groupby("ticker")["dollar_vol"].transform(lambda s: s.rolling(20).mean())

    mclose = raw.pivot_table(index="date", columns="ticker", values="Close").resample("ME").last()
    mdv = raw.pivot_table(index="date", columns="ticker", values="dv20").resample("ME").last()

    spy = mclose["SPY"].pct_change()
    universe_cols = [c for c in mclose.columns if c in set(CSV_UNIVERSE)]
    mclose_u = mclose[universe_cols]
    mdv_u = mdv[universe_cols]

    mret = mclose_u.pct_change()
    # 12-1 momentum: return from t-13 to t-1 (skip most recent month)
    mom = mclose_u.shift(MOM_SKIP) / mclose_u.shift(MOM_SKIP + MOM_LOOK) - 1.0

    tradable = (mclose_u >= MIN_PRICE) & (mdv_u >= MIN_DOLLAR_VOL)
    mom_ok = mom.where(tradable)

    dates = mclose_u.index
    port_ret, port_ret_net, turnovers = [], [], []
    prev_holdings: set[str] = set()
    idx_used = []

    for i in range(len(dates) - 1):
        row = mom_ok.iloc[i].dropna()
        if len(row) < TOP_N:
            continue
        holdings = list(row.nlargest(TOP_N).index)
        nxt = mret.iloc[i + 1][holdings]
        gross = nxt.mean(skipna=True)

        turn = len(set(holdings) ^ prev_holdings) / (2 * TOP_N)  # one-way fraction
        cost = turn * 2 * (COST_BPS / 1e4)  # buy+sell legs
        port_ret.append(gross)
        port_ret_net.append(gross - cost)
        turnovers.append(turn)
        idx_used.append(dates[i + 1])
        prev_holdings = set(holdings)

    pr = pd.Series(port_ret, index=idx_used)
    prn = pd.Series(port_ret_net, index=idx_used)
    spy_aligned = spy.reindex(pr.index)

    g, n, b = perf_stats(pr), perf_stats(prn), perf_stats(spy_aligned)

    print(f"\nMomentum portfolio: top-{TOP_N} by {MOM_LOOK}-{MOM_SKIP} mom, monthly rebal, "
          f"equal-weight, {COST_BPS:.0f}bps/side")
    print(f"  period          {pr.index[0]:%Y-%m} -> {pr.index[-1]:%Y-%m}  ({len(pr)} months)")
    print(f"  avg turnover    {np.mean(turnovers)*100:.0f}% / month\n")
    print(f"  {'':<14}{'CAGR':>8}{'Vol':>8}{'Sharpe':>8}{'MaxDD':>9}{'Hit/mo':>8}{'Final x':>9}")
    for label, st in (("GROSS", g), ("NET (cost)", n), ("SPY", b)):
        print(f"  {label:<14}{st['CAGR']*100:>7.1f}%{st['Vol']*100:>7.1f}%{st['Sharpe']:>8.2f}"
              f"{st['MaxDD']*100:>8.1f}%{st['HitMo']*100:>7.0f}%{st['FinalX']:>8.1f}x")

    print("\n=== net return by year vs SPY ===")
    by = pd.DataFrame({"mom_net": prn, "spy": spy_aligned})
    yr = by.groupby(by.index.year).apply(lambda d: (1 + d).prod() - 1) * 100
    yr["excess"] = yr["mom_net"] - yr["spy"]
    print(yr.round(1).to_string())


if __name__ == "__main__":
    main()
