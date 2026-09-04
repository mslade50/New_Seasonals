"""stop_gap_slippage_impact.py — before/after impact of the gap-aware stop fill
+ slippage change in process_signals_fast, plus a no-stop @ 0.75x counterfactual.

Reuses the exact full-book ledger pipeline (same data, same candidates, same
prod knobs: cap_bps=250, overflow_active=True, flat $750k sizing so per-trade R
and dollars are era-comparable and the trade SET is identical across variants).

Four runs on ONE precompute (process is the cheap part):
  old      : stop_gap_fill=False                 -> legacy site behavior (fill AT stop, no slip)
  gaponly  : gap fill, 0 slippage                -> isolates the gap-through effect / counts gaps
  new      : gap fill + 3 bps / +10 bps gap slip -> the new prod behavior
  nostop   : use_stop_loss=False on stop strats, risk x0.75 -> "no stop, 25% less risk"

R is sizing-basis invariant (R = PnL / Risk$), so the 0.75x only shows up in the
nostop dollar column. Writes two CSVs to scratch/ and prints both tables.
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

# Stub streamlit before importing the engine (avoids the local protobuf import
# crash; the headless backtest path never touches the real streamlit runtime).
class _NoOp:
    def __getattr__(self, name):
        def f(*a, **k):
            return self
        return f
    def __call__(self, *a, **k): return self
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def cache_data(self, *a, **k):
        # Handle both @st.cache_resource (bare: a=(fn,)) and @st.cache_data(ttl=..)
        if len(a) == 1 and callable(a[0]) and not k:
            return a[0]
        def deco(fn): return fn
        return deco
    cache_resource = cache_data
sys.modules['streamlit'] = _NoOp()

import data_provider
from strategy_config import STRATEGY_BOOK, ACCOUNT_VALUE
from pages.strat_backtester import (
    download_historical_data,
    load_seasonal_map,
    load_atr_seasonal_map,
    precompute_all_indicators,
    generate_candidates_fast,
    process_signals_fast,
)
from daily_portfolio_report import (
    build_full_strategy_book,
    OVERFLOW_TICKERS,
    OVERFLOW_ELIGIBLE,
)

DATA_START = datetime.date(2000, 1, 1)
BT_START = datetime.date(2003, 1, 1)
OUT_IMPACT = os.path.join(_HERE, "stop_gap_slippage_impact.csv")
OUT_NOSTOP = os.path.join(_HERE, "stop_nostop_counterfactual.csv")


def load_data(tickers):
    if data_provider.has_master():
        print(f"  Loading {len(tickers)} tickers from master_prices.parquet ...")
        return data_provider.get_history(list(tickers), start=DATA_START.strftime("%Y-%m-%d"))
    print("  No master_prices.parquet — falling back to yfinance ...")
    return download_historical_data(list(tickers), start_date=DATA_START.strftime("%Y-%m-%d"))


def _enrich(sig):
    """Add Direction / R / stop_price; keep only the columns we need."""
    df = sig.copy().reset_index(drop=True)
    df["Direction"] = np.where(
        df["Action"].astype(str).str.upper().str.contains("SHORT"), "Short", "Long")
    df["R"] = df["PnL"] / df["Risk $"].replace(0, np.nan)
    # stop level the trade was minted with (relative, same basis as bars)
    df["stop_price"] = np.where(
        df["Direction"] == "Long",
        df["Price"] - df["ATR"] * df["stop_atr"],
        df["Price"] + df["ATR"] * df["stop_atr"])
    df["_key"] = (df["Strategy"].astype(str) + "|" + df["Ticker"].astype(str) + "|"
                  + pd.to_datetime(df["Date"]).astype(str) + "|"
                  + pd.to_datetime(df["Entry Date"]).astype(str) + "|"
                  + df["Price"].round(4).astype(str))
    return df


def main():
    print("=" * 78)
    print("STOP GAP-FILL + SLIPPAGE IMPACT — full book, full history")
    print("=" * 78)

    full_book = build_full_strategy_book()
    sznl_map = load_seasonal_map()
    atr_sznl_map = load_atr_seasonal_map()
    if not atr_sznl_map:
        print("  WARNING: atr_seasonal_ranks.parquet missing — OLV/seasonal strats under-fire.")

    all_tickers = set()
    for s in full_book:
        all_tickers.update(s["universe_tickers"])
    all_tickers.update(["SPY", "^VIX"])
    md = load_data(all_tickers)
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

    print("\n  Precomputing indicators (slow part) ...")
    processed = precompute_all_indicators(md, full_book, sznl_map, vix_series, atr_sznl_map)
    print("  Generating candidates ...")
    candidates, signal_data = generate_candidates_fast(processed, full_book, sznl_map, BT_START)
    print(f"  {len(candidates)} candidate signal-dates")
    if not candidates:
        print("No signals.")
        return

    common = dict(cap_bps=250, overflow_active=True, flat_sizing=True)

    print("\n  [1/4] old (fill AT stop, no slippage) ...")
    old = _enrich(process_signals_fast(candidates, signal_data, processed, full_book,
                                       ACCOUNT_VALUE, stop_gap_fill=False, **common))
    print(f"        {len(old)} trades")

    print("  [2/4] gaponly (gap fill, 0 slippage) ...")
    gaponly = _enrich(process_signals_fast(candidates, signal_data, processed, full_book,
                                           ACCOUNT_VALUE, stop_gap_fill=True,
                                           stop_slip_bps=0.0, stop_gap_slip_bps=0.0, **common))

    print("  [3/4] new (gap fill + 3 bps / +10 bps gap slippage) ...")
    new = _enrich(process_signals_fast(candidates, signal_data, processed, full_book,
                                       ACCOUNT_VALUE, **common))

    # no-stop book: drop the stop on every strat that uses one, size x0.75
    nostop_book = copy.deepcopy(full_book)
    stop_strats = set()
    for s in nostop_book:
        if s["execution"].get("use_stop_loss", True):
            stop_strats.add(s["name"])
            s["execution"]["use_stop_loss"] = False
    risk_mults = {n: 0.75 for n in stop_strats}
    print(f"  [4/4] nostop (use_stop_loss=False, risk x0.75) on {len(stop_strats)} strats ...")
    nostop = _enrich(process_signals_fast(candidates, signal_data, processed, nostop_book,
                                          ACCOUNT_VALUE, risk_multipliers=risk_mults, **common))

    # ---- gap detection: in gaponly a Stop exit differs from the stop price iff it gapped ----
    g_stop = gaponly[gaponly["Exit Type"] == "Stop"].copy()
    g_stop["gapped"] = (g_stop["Exit Price"] - g_stop["stop_price"]).abs() > 1e-6
    gap_by_strat = g_stop.groupby("Strategy")["gapped"].sum()

    # ---- per-strategy impact table (only strats that use stops) ----
    rows = []
    for name in sorted(stop_strats):
        o = old[old["Strategy"] == name]
        n = new[new["Strategy"] == name]
        if o.empty:
            continue
        o_stops = o[o["Exit Type"] == "Stop"]
        rows.append({
            "Strategy": name,
            "Trades": len(o),
            "StopOuts": len(o_stops),
            "Gapped": int(gap_by_strat.get(name, 0)),
            "old_AvgR": round(o["R"].mean(), 3),
            "new_AvgR": round(n["R"].mean(), 3),
            "dAvgR": round(n["R"].mean() - o["R"].mean(), 3),
            "old_TotR": round(o["R"].sum(), 1),
            "new_TotR": round(n["R"].sum(), 1),
            "dTotR": round(n["R"].sum() - o["R"].sum(), 1),
            "old_WorstR": round(o["R"].min(), 2),
            "new_WorstR": round(n["R"].min(), 2),
            "old_PnL$": round(o["PnL"].sum()),
            "new_PnL$": round(n["PnL"].sum()),
            "dPnL$": round(n["PnL"].sum() - o["PnL"].sum()),
        })
    impact = pd.DataFrame(rows)
    # book total
    o_all = old[old["Strategy"].isin(stop_strats)]
    n_all = new[new["Strategy"].isin(stop_strats)]
    impact.loc[len(impact)] = {
        "Strategy": "** BOOK (stop strats) **",
        "Trades": len(o_all),
        "StopOuts": int((o_all["Exit Type"] == "Stop").sum()),
        "Gapped": int(g_stop["gapped"].sum()),
        "old_AvgR": round(o_all["R"].mean(), 3),
        "new_AvgR": round(n_all["R"].mean(), 3),
        "dAvgR": round(n_all["R"].mean() - o_all["R"].mean(), 3),
        "old_TotR": round(o_all["R"].sum(), 1),
        "new_TotR": round(n_all["R"].sum(), 1),
        "dTotR": round(n_all["R"].sum() - o_all["R"].sum(), 1),
        "old_WorstR": round(o_all["R"].min(), 2),
        "new_WorstR": round(n_all["R"].min(), 2),
        "old_PnL$": round(o_all["PnL"].sum()),
        "new_PnL$": round(n_all["PnL"].sum()),
        "dPnL$": round(n_all["PnL"].sum() - o_all["PnL"].sum()),
    }
    impact.to_csv(OUT_IMPACT, index=False)

    # ---- no-stop @ 0.75x counterfactual vs new ----
    rows2 = []
    for name in sorted(stop_strats):
        n = new[new["Strategy"] == name]
        z = nostop[nostop["Strategy"] == name]
        if n.empty:
            continue
        rows2.append({
            "Strategy": name,
            "Trades": len(z),
            "new_Win%": round((n["PnL"] > 0).mean() * 100, 1),
            "nostop_Win%": round((z["PnL"] > 0).mean() * 100, 1),
            "new_AvgR": round(n["R"].mean(), 3),
            "nostop_AvgR": round(z["R"].mean(), 3),
            "new_TotR": round(n["R"].sum(), 1),
            "nostop_TotR": round(z["R"].sum(), 1),
            "new_WorstR": round(n["R"].min(), 2),
            "nostop_WorstR": round(z["R"].min(), 2),
            "new_PnL$": round(n["PnL"].sum()),
            "nostop_PnL$@0.75x": round(z["PnL"].sum()),
        })
    nostop_tbl = pd.DataFrame(rows2)
    n_all = new[new["Strategy"].isin(stop_strats)]
    z_all = nostop[nostop["Strategy"].isin(stop_strats)]
    nostop_tbl.loc[len(nostop_tbl)] = {
        "Strategy": "** BOOK (stop strats) **",
        "Trades": len(z_all),
        "new_Win%": round((n_all["PnL"] > 0).mean() * 100, 1),
        "nostop_Win%": round((z_all["PnL"] > 0).mean() * 100, 1),
        "new_AvgR": round(n_all["R"].mean(), 3),
        "nostop_AvgR": round(z_all["R"].mean(), 3),
        "new_TotR": round(n_all["R"].sum(), 1),
        "nostop_TotR": round(z_all["R"].sum(), 1),
        "new_WorstR": round(n_all["R"].min(), 2),
        "nostop_WorstR": round(z_all["R"].min(), 2),
        "new_PnL$": round(n_all["PnL"].sum()),
        "nostop_PnL$@0.75x": round(z_all["PnL"].sum()),
    }
    nostop_tbl.to_csv(OUT_NOSTOP, index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print("\n" + "=" * 78)
    print("TABLE 1 — OLD (site today) vs NEW (gap fill + slippage)")
    print("=" * 78)
    print(impact.to_string(index=False))
    print(f"\n  -> {OUT_IMPACT}")
    print("\n" + "=" * 78)
    print("TABLE 2 — NEW (with stop) vs NO-STOP sized at 0.75x risk")
    print("=" * 78)
    print(nostop_tbl.to_string(index=False))
    print(f"\n  -> {OUT_NOSTOP}")
    print("\nDone.")


if __name__ == "__main__":
    main()
