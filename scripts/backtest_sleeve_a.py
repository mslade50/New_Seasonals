"""
backtest_sleeve_a.py — Sleeve A: Sector Rotation Trend (SRT)

Faber-style sector rotation translated to the signal-based engine:
  - Cross-sectional: top 20% of 126d return rank within SECTOR_INDEX_ETFS
  - Trend: each ticker must be above its rising 200 SMA
  - Momentum confirmation: trailing 252d return rank > 60 (positive 12mo)
  - Cadence: Monday-only signal firing; hold_days=21 (~1mo) gives a
    rolling monthly-rebalance behavior via max_one_pos=True
  - No stops/targets — time exit only (clean trend signal)

Engine calls mirror daily_portfolio_report.run_12month_backtest.
Outputs a one-pager with Sharpe / CAGR / max DD / win rate / PF / trade count.
"""
import copy
import datetime
import os
import sys

import numpy as np
import pandas as pd

# Path setup
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import data_provider
from strategy_config import SECTOR_INDEX_ETFS, ACCOUNT_VALUE, STRATEGY_BOOK
from pages.strat_backtester import (
    download_historical_data,
    load_seasonal_map,
    load_atr_seasonal_map,
    precompute_all_indicators,
    generate_candidates_fast,
    process_signals_fast,
    get_daily_mtm_series,
)


# ---------------------------------------------------------------------------
# Build Sleeve A strategy dict by patching the 52wh Breakout template
# ---------------------------------------------------------------------------
def build_sleeve_a():
    template = next(s for s in STRATEGY_BOOK if s["name"] == "52wh Breakout")
    strat = copy.deepcopy(template)

    strat["id"] = "SRT — xsec 126d>=80, perf 252d>60, Rising 200 SMA, Mon entry, 21d hold"
    strat["name"] = "Sector Rotation Trend"
    strat["description"] = "Sleeve A — Sector Rotation Trend backtest"
    strat["universe_tickers"] = SECTOR_INDEX_ETFS

    s = strat["settings"]

    # Clear the breakout-specific filters from the template
    s["perf_filters"] = [
        {"window": 252, "logic": ">", "thresh": 60.0, "thresh_max": 100.0, "consecutive": 1},
    ]
    s["use_52w"] = False
    s["use_ath"] = False
    s["use_vol"] = False
    s["use_vol_rank"] = False
    s["atr_sznl_filters"] = []
    s["dial_filters"] = []

    # Sleeve A v3 — discrete first-instance breakout: fire when ticker FIRST
    # enters top quintile of 126d xsec rank, hold 63d, no stop, small bps
    s["use_xsec_filter"] = True
    s["xsec_filters"] = [
        {"window": 126, "logic": ">", "thresh": 80.0, "thresh_max": 100.0, "consecutive": 1},
    ]
    s["trend_filter"] = "Market > 200 SMA"
    s["ma_consec_filters"] = [
        {"length": 200, "logic": "Above", "consec": 5},
    ]
    # Don't gate to Monday — let breakout fire on its actual day
    s["use_dow_filter"] = False
    s["allowed_days"] = [0, 1, 2, 3, 4]
    # First-instance into top quintile: avoids continuous re-firing
    s["perf_first_instance"] = True
    s["perf_lookback"] = 63   # debounce window — no re-fire for 63d

    # Entry / hold
    s["entry_type"] = "T+1 Open"
    s["max_one_pos"] = True
    s["allow_same_day_reentry"] = False
    s["max_daily_entries"] = 4
    s["max_total_positions"] = 6

    # Execution: no stop (let trend ride), small risk so 6 positions don't lever
    strat["execution"] = {
        "risk_bps": 20,
        "slippage_bps": 2,
        "stop_atr": 2.0,
        "tgt_atr": 8.0,
        "hold_days": 63,
        "use_stop_loss": False,
        "use_take_profit": False,
    }
    return strat


# ---------------------------------------------------------------------------
# Load OHLCV
# ---------------------------------------------------------------------------
def load_data(tickers, data_start):
    if data_provider.has_master():
        print(f"  Loading {len(tickers)} tickers from master_prices.parquet ...")
        md = data_provider.get_history(list(tickers), start=data_start.strftime("%Y-%m-%d"))
        missing = [t for t in tickers if t not in md or md[t] is None or md[t].empty]
        if missing:
            print(f"  Backfilling {len(missing)} missing tickers via yfinance: {missing}")
            md.update(download_historical_data(missing, start_date=data_start.strftime("%Y-%m-%d")))
        return md
    print("  No master_prices.parquet — falling back to yfinance ...")
    return download_historical_data(list(tickers), start_date=data_start.strftime("%Y-%m-%d"))


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------
def compute_stats(daily_pnl, starting_equity, sig_df):
    if daily_pnl.empty:
        return {}

    daily_pnl = daily_pnl.copy()
    daily_pnl.index = pd.to_datetime(daily_pnl.index)
    equity = starting_equity + daily_pnl.cumsum()
    # Daily return on equity (use prior-day equity as the denominator).
    eq_prev = equity.shift(1).fillna(starting_equity)
    daily_ret = daily_pnl / eq_prev
    daily_ret = daily_ret.replace([np.inf, -np.inf], 0).fillna(0)

    n_days = len(daily_pnl)
    years = n_days / 252.0
    total_return = equity.iloc[-1] / starting_equity - 1.0
    cagr = (equity.iloc[-1] / starting_equity) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = (daily_ret.mean() * 252) / ann_vol if ann_vol > 0 else 0.0

    # Sortino (downside deviation)
    downside = daily_ret[daily_ret < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0.0
    sortino = (daily_ret.mean() * 252) / downside_vol if downside_vol > 0 else 0.0

    # Max drawdown
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_dd = drawdown.min()

    # Trade stats
    n_trades = len(sig_df)
    if n_trades > 0:
        wins = sig_df[sig_df["PnL"] > 0]
        losses = sig_df[sig_df["PnL"] < 0]
        win_rate = len(wins) / n_trades
        total_won = wins["PnL"].sum()
        total_lost = abs(losses["PnL"].sum())
        pf = total_won / total_lost if total_lost > 0 else float("inf")
        avg_win = wins["PnL"].mean() if len(wins) > 0 else 0.0
        avg_loss = losses["PnL"].mean() if len(losses) > 0 else 0.0
        expectancy = sig_df["PnL"].mean()
    else:
        win_rate = pf = avg_win = avg_loss = expectancy = 0.0

    return {
        "Start": daily_pnl.index.min().date(),
        "End": daily_pnl.index.max().date(),
        "Years": round(years, 2),
        "Trades": n_trades,
        "Total return": f"{total_return * 100:.1f}%",
        "CAGR": f"{cagr * 100:.2f}%",
        "Ann vol": f"{ann_vol * 100:.2f}%",
        "Sharpe": round(sharpe, 3),
        "Sortino": round(sortino, 3),
        "Max DD": f"{max_dd * 100:.2f}%",
        "Win rate": f"{win_rate * 100:.1f}%",
        "Profit factor": round(pf, 2),
        "Avg win $": round(avg_win, 0),
        "Avg loss $": round(avg_loss, 0),
        "Expectancy $/trade": round(expectancy, 0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    starting_equity = ACCOUNT_VALUE
    data_start = datetime.date(2000, 1, 1)
    bt_start = datetime.date(2003, 1, 1)  # let 200d SMA + 252d ranks warm up
    bt_end = datetime.date.today()

    print("=" * 70)
    print("SLEEVE A — Sector Rotation Trend backtest")
    print(f"Universe: {len(SECTOR_INDEX_ETFS)} sector + index ETFs")
    print(f"Data range: {data_start} - {bt_end}")
    print(f"Backtest range: {bt_start} - {bt_end}")
    print(f"Starting equity: ${starting_equity:,.0f}")
    print("=" * 70)

    strat = build_sleeve_a()
    book = [strat]

    sznl_map = load_seasonal_map()
    atr_sznl_map = load_atr_seasonal_map()

    tickers = set(strat["universe_tickers"])
    tickers.update(["SPY", "^VIX"])
    md = load_data(tickers, data_start)
    if not md:
        print("FAILED to load data")
        return

    vix_df = md.get("^VIX")
    vix_series = None
    if vix_df is not None and not vix_df.empty:
        vd = vix_df.copy()
        if isinstance(vd.columns, pd.MultiIndex):
            vd.columns = vd.columns.get_level_values(0)
        vd.columns = [c.capitalize() for c in vd.columns]
        vix_series = vd["Close"]

    print("\n  Precomputing indicators (cold start ~2-5 min) ...")
    processed = precompute_all_indicators(md, book, sznl_map, vix_series, atr_sznl_map)

    print(f"\n  Generating candidates from {bt_start} ...")
    candidates, signal_data = generate_candidates_fast(processed, book, sznl_map, bt_start)
    print(f"  Found {len(candidates)} candidate signal-dates")

    if not candidates:
        print("No signals fired — check filter logic")
        return

    print("\n  Processing signals into trades ...")
    sig_df = process_signals_fast(
        candidates, signal_data, processed, book, starting_equity,
        cap_bps=250, overflow_active=False,
    )
    print(f"  {len(sig_df)} trades executed")

    if sig_df.empty:
        print("No trades executed")
        return

    # Filter to backtest window for stats
    sig_df["Entry Date"] = pd.to_datetime(sig_df["Entry Date"])
    sig_df = sig_df[sig_df["Entry Date"] >= pd.Timestamp(bt_end - datetime.timedelta(days=int(365 * 23)))]

    daily_pnl = get_daily_mtm_series(sig_df, md, start_date=bt_start)

    stats = compute_stats(daily_pnl, starting_equity, sig_df)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for k, v in stats.items():
        print(f"  {k:.<25} {v}")
    print("=" * 70)

    # Per-year breakdown
    daily_pnl.index = pd.to_datetime(daily_pnl.index)
    yearly = daily_pnl.groupby(daily_pnl.index.year).sum()
    eq_yearly = (starting_equity + daily_pnl.cumsum()).groupby(daily_pnl.index.year).last()
    print("\nYearly P&L (vs starting equity $%d):" % starting_equity)
    for yr in yearly.index:
        ret_pct = yearly[yr] / starting_equity * 100
        print(f"  {yr}: ${yearly[yr]:>+14,.0f}  ({ret_pct:+6.2f}% of start)   equity end-of-year: ${eq_yearly[yr]:>14,.0f}")

    # Trades by ticker
    print("\nTrades by ticker (top 15):")
    by_tk = sig_df.groupby("Ticker").agg(trades=("PnL", "size"), pnl=("PnL", "sum"))
    by_tk = by_tk.sort_values("pnl", ascending=False)
    for tk, row in by_tk.head(15).iterrows():
        print(f"  {tk:<8} trades={int(row['trades']):>4}  pnl=${row['pnl']:>+14,.0f}")


if __name__ == "__main__":
    main()
