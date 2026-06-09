"""
atr_gap_overflow_test.py — what would 'ATR Extended Gap Up' do on the overflow
tier? It is NOT currently overflow-eligible; this is an exploratory run.

Builds a 2-pass book (liquid universe + overflow universe) for that one
strategy, runs the production engine (cap_bps=250), and prints tier-split stats
comparable to the full ledger. No overflow bps override (the strategy isn't in
OVERFLOW_RISK_OVERRIDES), so the overflow pass keeps its native 60 bps — same
convention OVS uses. R_Multiple / Return_Pct are sizing-invariant, so the edge
read is robust regardless of bps.
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
from strategy_config import STRATEGY_BOOK, ACCOUNT_VALUE, LIQUID_PLUS_COMMODITIES
from pages.strat_backtester import (
    download_historical_data, load_seasonal_map, load_atr_seasonal_map,
    precompute_all_indicators, generate_candidates_fast, process_signals_fast,
)
from daily_portfolio_report import OVERFLOW_TICKERS

STRAT_NAME = "ATR Extended Gap Up"
DATA_START = datetime.date(2000, 1, 1)
BT_START = datetime.date(2003, 1, 1)


def load_data(tickers):
    md = data_provider.get_history(list(tickers), start=DATA_START.strftime("%Y-%m-%d"))
    missing = [t for t in tickers if t not in md or md[t] is None or md[t].empty]
    if missing:
        print(f"  {len(missing)} missing from master (skipped): "
              f"{missing[:15]}{'...' if len(missing) > 15 else ''}")
    return md


def stats_block(g):
    n = len(g)
    if n == 0:
        return None
    r = g["R_Multiple"]
    ret = g["Return_Pct"]
    gross_win = r[r > 0].sum()
    gross_loss = -r[r < 0].sum()
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    return {
        "Trades": n,
        "Win%": round((g["PnL"] > 0).mean() * 100, 1),
        "Tot_R": round(r.sum(), 1),
        "Avg_R": round(r.mean(), 3),
        "Med_R": round(r.median(), 3),
        "PF_R": round(pf, 2),
        "AvgRet%": round(ret.mean(), 2),
        "MedRet%": round(ret.median(), 2),
        "BestR": round(r.max(), 2),
        "WorstR": round(r.min(), 2),
        "PnL_flat750k": round(g["PnL_flat"].sum()),
    }


def main():
    eq = ACCOUNT_VALUE
    print("=" * 74)
    print(f"OVERFLOW EXPLORATION — {STRAT_NAME}")
    print(f"  {BT_START} -> today | start equity ${eq:,.0f} | cap_bps=250")
    print("=" * 74)

    base = next(s for s in STRATEGY_BOOK if s["name"] == STRAT_NAME)
    liq = copy.deepcopy(base)
    liq["universe_tickers"] = list(LIQUID_PLUS_COMMODITIES)
    of = copy.deepcopy(base)
    of["universe_tickers"] = list(OVERFLOW_TICKERS)
    book = [liq, of]
    print(f"  Liquid universe: {len(liq['universe_tickers'])} | "
          f"Overflow universe: {len(of['universe_tickers'])} | native {base['execution']['risk_bps']} bps (no override)")

    sznl_map = load_seasonal_map()
    atr_sznl_map = load_atr_seasonal_map()

    tickers = set(liq["universe_tickers"]) | set(of["universe_tickers"]) | {"SPY", "^VIX"}
    md = load_data(tickers)
    vix_df = md.get("^VIX")
    vix_series = None
    if vix_df is not None and not vix_df.empty:
        vd = vix_df.copy()
        if isinstance(vd.columns, pd.MultiIndex):
            vd.columns = vd.columns.get_level_values(0)
        vd.columns = [c.capitalize() for c in vd.columns]
        vix_series = vd["Close"]

    print("\n  Precomputing indicators ...")
    processed = precompute_all_indicators(md, book, sznl_map, vix_series, atr_sznl_map)
    print(f"  Generating candidates from {BT_START} ...")
    candidates, signal_data = generate_candidates_fast(processed, book, sznl_map, BT_START)
    print(f"  {len(candidates)} candidate signal-dates")
    if not candidates:
        print("No signals.")
        return

    sig = process_signals_fast(candidates, signal_data, processed, book, eq,
                               cap_bps=250, overflow_active=True)
    sig_flat = process_signals_fast(candidates, signal_data, processed, book, eq,
                                    cap_bps=250, overflow_active=True, flat_sizing=True)
    print(f"  {len(sig)} trades (compounded), {len(sig_flat)} (flat)")
    if sig.empty:
        print("No trades.")
        return

    sig = sig.reset_index(drop=True)
    # attach flat PnL by key
    key = ["Strategy", "Ticker", "Date", "Entry Date", "Price"]
    fl = sig_flat[key + ["PnL"]].copy()
    fl["_k"] = fl[key].round({"Price": 4}).astype(str).agg("|".join, axis=1)
    sig["_k"] = sig[key].round({"Price": 4}).astype(str).agg("|".join, axis=1)
    sig["PnL_flat"] = sig["_k"].map(fl.drop_duplicates("_k").set_index("_k")["PnL"]).values

    _of = set(OVERFLOW_TICKERS)
    sig["Tier"] = np.where(sig["Ticker"].isin(_of), "Overflow", "Liquid")
    sig["R_Multiple"] = sig["PnL"] / sig["Risk $"].replace(0, np.nan)
    _sign = np.where(sig["Action"].astype(str).str.upper().str.contains("SHORT"), -1.0, 1.0)
    sig["Return_Pct"] = _sign * (sig["Exit Price"] - sig["Price"]) / sig["Price"] * 100.0

    print("\n" + "=" * 74)
    print("STATS BY TIER")
    print("=" * 74)
    rows = []
    for tier in ["Liquid", "Overflow"]:
        s = stats_block(sig[sig["Tier"] == tier])
        if s:
            rows.append({"Tier": tier, **s})
    alls = stats_block(sig)
    rows.append({"Tier": "BOTH", **alls})
    summ = pd.DataFrame(rows)
    print(summ.to_string(index=False))

    of_df = sig[sig["Tier"] == "Overflow"]
    if not of_df.empty:
        print("\n  OVERFLOW exit-type breakdown:")
        for et, c in of_df["Exit Type"].value_counts().items():
            sub = of_df[of_df["Exit Type"] == et]
            print(f"    {str(et):<8} {c:>4}  avgR={sub['R_Multiple'].mean():+.2f}")
        print("\n  OVERFLOW R by year:")
        by_yr = of_df.groupby(of_df["Date"].dt.year)["R_Multiple"].agg(["count", "sum"])
        for yr, row in by_yr.iterrows():
            print(f"    {yr}: n={int(row['count']):>3}  R={row['sum']:+.1f}")
        print("\n  OVERFLOW top/bottom tickers by total R:")
        by_tk = of_df.groupby("Ticker")["R_Multiple"].agg(["count", "sum"]).sort_values("sum")
        for tk, row in pd.concat([by_tk.head(8), by_tk.tail(8)]).iterrows():
            print(f"    {tk:<8} n={int(row['count']):>2}  R={row['sum']:+.2f}")

    out = os.path.join(_HERE, "atr_gap_overflow_trades.csv")
    sig.drop(columns="_k").to_parquet(out.replace(".csv", ".parquet"), index=False)
    sig.drop(columns="_k").to_csv(out, index=False)
    print(f"\n  Wrote trades -> {out} (+ .parquet)")
    print("Done.")


if __name__ == "__main__":
    main()
