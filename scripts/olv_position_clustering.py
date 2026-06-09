"""
olv_position_clustering.py — find OLV same-ticker entry clustering.

Runs the "Oversold Low Volume" (OLV) strategy over the FULL universe it can
trade live (liquid + overflow = CSV_UNIVERSE) using the same backtest engine as
the strat_backtester page, then scans the executed entries for any single ticker
that opened 3+ positions inside a rolling 5-trading-day window.

"Positions opened" = executed trades (sig_df), i.e. signals whose persistent
limit actually filled. OLV has no earnings blackout (only a pre-earnings size
override) so nothing is dropped on the earnings axis; the per-strategy daily cap
only scales size, never trade count, so entry clustering is cap-independent.

Window: rolling 5 TRADING days. Three entries on trading-day ordinals whose
(max - min) <= 4 fall inside one 5-trading-day span.
"""
import copy
import datetime
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import data_provider
from strategy_config import CSV_UNIVERSE, ACCOUNT_VALUE, STRATEGY_BOOK
from pages.strat_backtester import (
    download_historical_data,
    load_seasonal_map,
    load_atr_seasonal_map,
    precompute_all_indicators,
    generate_candidates_fast,
    process_signals_fast,
)

WINDOW_TD = 5          # rolling window length in trading days
MIN_ENTRIES = 3        # "more than 2 out of 5" -> 3 or more


def load_data(tickers, data_start):
    if data_provider.has_master():
        print(f"  Loading {len(tickers)} tickers from master_prices.parquet ...")
        md = data_provider.get_history(list(tickers), start=data_start.strftime("%Y-%m-%d"))
        missing = [t for t in tickers if t not in md or md[t] is None or md[t].empty]
        if missing:
            print(f"  {len(missing)} tickers missing from master (skipping yfinance backfill): "
                  f"{missing[:20]}{'...' if len(missing) > 20 else ''}")
        return md
    print("  No master_prices.parquet — falling back to yfinance ...")
    return download_historical_data(list(tickers), start_date=data_start.strftime("%Y-%m-%d"))


def build_olv():
    template = next(s for s in STRATEGY_BOOK if s["name"] == "Oversold Low Volume")
    strat = copy.deepcopy(template)
    # Run against the full live universe (liquid + overflow). Strategy filters
    # (min_price/min_vol/min_age) prune internally; we just feed everything.
    strat["universe_tickers"] = list(CSV_UNIVERSE)
    return strat


def find_clusters(sig_df, trading_days):
    """For each ticker, find rolling 5-TD windows holding >= MIN_ENTRIES entries.

    trading_days: sorted DatetimeIndex serving as the trading-day calendar.
    Returns a list of episode dicts (non-overlapping, anchored at first entry).
    """
    ord_map = {d.normalize(): i for i, d in enumerate(trading_days)}

    def to_ord(ts):
        ts = pd.Timestamp(ts).normalize()
        i = ord_map.get(ts)
        if i is not None:
            return i
        # entry date not on the reference calendar -> nearest prior trading day
        pos = trading_days.searchsorted(ts, side="right") - 1
        return int(pos) if pos >= 0 else 0

    episodes = []
    for tk, grp in sig_df.groupby("Ticker"):
        # one entry per ticker per day (engine forbids same-day re-entry); dedupe defensively
        dates = sorted(pd.to_datetime(grp["Entry Date"]).dt.normalize().unique())
        ords = [to_ord(d) for d in dates]
        n = len(dates)
        i = 0
        while i < n:
            # all entries within [ord_i, ord_i + (WINDOW_TD-1)]
            hi = ords[i] + (WINDOW_TD - 1)
            j = i
            while j < n and ords[j] <= hi:
                j += 1
            cnt = j - i
            if cnt >= MIN_ENTRIES:
                win_dates = dates[i:j]
                episodes.append({
                    "Ticker": tk,
                    "n_entries": cnt,
                    "first": pd.Timestamp(win_dates[0]).date(),
                    "last": pd.Timestamp(win_dates[-1]).date(),
                    "span_td": ords[j - 1] - ords[i] + 1,
                    "dates": [pd.Timestamp(d).date() for d in win_dates],
                })
                i = j  # non-overlapping: jump past this episode
            else:
                i += 1
    episodes.sort(key=lambda e: (e["first"], e["Ticker"]))
    return episodes


def main():
    starting_equity = ACCOUNT_VALUE
    data_start = datetime.date(2000, 1, 1)
    bt_start = datetime.date(2003, 1, 1)  # warm up 252d ranks / 200 SMA

    print("=" * 72)
    print("OLV position clustering — 3+ entries in a rolling 5-trading-day window")
    print(f"Universe: full CSV_UNIVERSE ({len(CSV_UNIVERSE)} tickers, liquid + overflow)")
    print(f"Backtest range: {bt_start} - today | start equity ${starting_equity:,.0f}")
    print("=" * 72)

    strat = build_olv()
    book = [strat]

    sznl_map = load_seasonal_map()
    atr_sznl_map = load_atr_seasonal_map()

    tickers = set(strat["universe_tickers"])
    tickers.update(["SPY", "^VIX"])
    md = load_data(tickers, data_start)
    if not md:
        print("FAILED to load data")
        return

    spy_df = md.get("SPY")
    if spy_df is None or spy_df.empty:
        print("FAILED — no SPY for trading calendar")
        return
    _spy = spy_df.copy()
    if isinstance(_spy.columns, pd.MultiIndex):
        _spy.columns = _spy.columns.get_level_values(0)
    trading_days = pd.DatetimeIndex(sorted(pd.to_datetime(_spy.index).normalize().unique()))

    vix_df = md.get("^VIX")
    vix_series = None
    if vix_df is not None and not vix_df.empty:
        vd = vix_df.copy()
        if isinstance(vd.columns, pd.MultiIndex):
            vd.columns = vd.columns.get_level_values(0)
        vd.columns = [c.capitalize() for c in vd.columns]
        vix_series = vd["Close"]

    print("\n  Precomputing indicators (full universe — this is the slow part) ...")
    processed = precompute_all_indicators(md, book, sznl_map, vix_series, atr_sznl_map)

    print(f"\n  Generating candidates from {bt_start} ...")
    candidates, signal_data = generate_candidates_fast(processed, book, sznl_map, bt_start)
    print(f"  Found {len(candidates)} candidate signal-dates")
    if not candidates:
        print("No signals fired.")
        return

    print("\n  Processing signals into executed trades ...")
    sig_df = process_signals_fast(
        candidates, signal_data, processed, book, starting_equity,
        cap_bps=100000, overflow_active=False,  # huge cap so nothing is size-pruned
    )
    print(f"  {len(sig_df)} positions opened (filled)")
    if sig_df.empty:
        print("No trades executed.")
        return

    sig_df["Entry Date"] = pd.to_datetime(sig_df["Entry Date"])

    # cache the full trade table so follow-up questions need no re-run
    _trades_out = os.path.join(_HERE, "olv_trades.csv")
    sig_df.to_csv(_trades_out, index=False)
    print(f"  Wrote full trade table -> {_trades_out}")

    # ---- overall entry stats ----
    n_tickers = sig_df["Ticker"].nunique()
    print(f"\n  Total positions: {len(sig_df)} across {n_tickers} tickers")
    print(f"  Date range: {sig_df['Entry Date'].min().date()} -> {sig_df['Entry Date'].max().date()}")

    # ---- clustering ----
    episodes = find_clusters(sig_df, trading_days)

    print("\n" + "=" * 72)
    print(f"CLUSTERS: ticker opened {MIN_ENTRIES}+ positions within a rolling {WINDOW_TD}-trading-day window")
    print("=" * 72)
    if not episodes:
        print("  None found.")
    else:
        print(f"  {len(episodes)} episodes found.\n")
        print(f"  {'Ticker':<8} {'n':>2}  {'span':>4}  {'first':<11} {'last':<11}  entry dates")
        print("  " + "-" * 86)
        for e in episodes:
            ds = ", ".join(str(d) for d in e["dates"])
            print(f"  {e['Ticker']:<8} {e['n_entries']:>2}  {e['span_td']:>3}d  "
                  f"{str(e['first']):<11} {str(e['last']):<11}  {ds}")

        # ---- summaries ----
        by_n = pd.Series([e["n_entries"] for e in episodes]).value_counts().sort_index()
        print("\n  Episodes by entry-count:")
        for n, c in by_n.items():
            print(f"    {n} entries in 5 TD : {c} episode(s)")

        by_tk = pd.Series([e["Ticker"] for e in episodes]).value_counts()
        print(f"\n  Tickers with clustering ({by_tk.size}); top offenders:")
        for tk, c in by_tk.head(20).items():
            print(f"    {tk:<8} {c} episode(s)")

        by_yr = pd.Series([e["first"].year for e in episodes]).value_counts().sort_index()
        print("\n  Episodes by year (of window start):")
        for yr, c in by_yr.items():
            print(f"    {yr}: {c}")

    # ---- per-episode returns ----
    if episodes:
        print("\n" + "=" * 72)
        print("EPISODE RETURNS (combined PnL of the stacked legs in each window)")
        print("  $ = sum PnL (compounding-scaled, era-dependent) | R = sum PnL/Risk$")
        print("  ret% = each leg's entry->exit price move")
        print("=" * 72)
        ep_rows = []
        grand_pnl = grand_r = 0.0
        n_legs_total = n_legs_win = 0
        for e in episodes:
            tk = e["Ticker"]
            wanted = set(pd.Timestamp(d) for d in e["dates"])
            legs = sig_df[(sig_df["Ticker"] == tk) &
                          (sig_df["Entry Date"].dt.normalize().isin(wanted))].copy()
            legs = legs.sort_values("Entry Date")
            sign = np.where(legs["Action"].astype(str).str.upper().str.contains("SHORT"), -1.0, 1.0)
            legs["ret_pct"] = sign * (legs["Exit Price"] - legs["Price"]) / legs["Price"] * 100.0
            legs["R"] = legs["PnL"] / legs["Risk $"].replace(0, np.nan)
            tot_pnl = float(legs["PnL"].sum())
            tot_r = float(legs["R"].sum())
            grand_pnl += tot_pnl
            grand_r += tot_r
            n_legs_total += len(legs)
            n_legs_win += int((legs["PnL"] > 0).sum())
            ep_rows.append({**{k: v for k, v in e.items() if k != "dates"},
                            "tot_pnl": round(tot_pnl), "tot_R": round(tot_r, 2),
                            "legs": "|".join(f"{pd.Timestamp(r['Entry Date']).date()}"
                                             f":{r['ret_pct']:+.1f}%/{r['R']:+.2f}R"
                                             f"({str(r['Exit Type'])[:4]})"
                                             for _, r in legs.iterrows())})
            print(f"\n  {tk}  {e['first']} -> {e['last']}  ({e['n_entries']} legs)"
                  f"   TOTAL: ${tot_pnl:>+,.0f}   {tot_r:>+.2f}R")
            for _, r in legs.iterrows():
                print(f"      {pd.Timestamp(r['Entry Date']).date()}  "
                      f"entry {r['Price']:>9.2f} -> exit {r['Exit Price']:>9.2f}  "
                      f"{r['ret_pct']:>+6.1f}%  {r['R']:>+5.2f}R  "
                      f"${r['PnL']:>+10,.0f}  [{str(r['Exit Type'])}]")
        wl = f"{n_legs_win}/{n_legs_total} legs green ({n_legs_win/n_legs_total*100:.0f}%)" if n_legs_total else "n/a"
        print("\n  " + "-" * 70)
        print(f"  GRAND TOTAL across {len(episodes)} episodes: "
              f"${grand_pnl:>+,.0f}   {grand_r:>+.2f}R   {wl}")
        pd.DataFrame(ep_rows).to_csv(os.path.join(_HERE, "olv_episode_returns.csv"), index=False)
        print(f"  Wrote {os.path.join(_HERE, 'olv_episode_returns.csv')}")

    # persist for follow-up
    out = os.path.join(_HERE, "olv_clusters.csv")
    if episodes:
        pd.DataFrame([{**{k: v for k, v in e.items() if k != "dates"},
                       "dates": "|".join(str(d) for d in e["dates"])}
                      for e in episodes]).to_csv(out, index=False)
        print(f"\n  Wrote {out}")
    print("\nDone.")


if __name__ == "__main__":
    main()
