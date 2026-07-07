"""
futures_lab_build.py — RESEARCH-ONLY one-off: replay the strategy book over the
non-equity history still in data/master_prices.parquet (futures =F, FX =X,
BTC-USD / ETH-USD) and emit a static payload for the private site's Futures Lab
page (site/research/futures_lab.json).

NOT part of the nightly pipeline. NOT tradeable output:
  - yfinance =F tickers are front-month CONTINUOUS contracts — roll gaps show
    up as price moves the strategies could never have traded.
  - Futures were removed from the tradeable universe 2026-07-06 (equity share
    order path only).
  - min_price / min_vol liquidity floors are zeroed for this replay (FX quotes
    ~1.0 with volume=0 would otherwise be excluded wholesale); volume-GATED
    strategies still cannot fire on FX because yfinance FX volume is 0.

Run:  python scratch/futures_lab_build.py
"""
import copy
import datetime
import json
import os
import sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("OVERFLOW_UNIVERSE_ACTIVE", "0")

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import data_provider
from strategy_config import STRATEGY_BOOK, ACCOUNT_VALUE
from pages.strat_backtester import (
    load_seasonal_map,
    load_atr_seasonal_map,
    precompute_all_indicators,
    generate_candidates_fast,
    process_signals_fast,
)

OUT_JSON = os.path.join(_ROOT, "site", "research", "futures_lab.json")
DATA_START = "2000-01-01"
BT_START = datetime.date(2003, 1, 1)


def _non_equity_universes():
    """Pull ticker groups straight from master_prices so 'any others ending =F
    present' are included automatically."""
    mp = pd.read_parquet(os.path.join(_ROOT, "data", "master_prices.parquet"),
                         columns=["ticker"])
    ticks = sorted(mp["ticker"].astype(str).unique())
    futures = [t for t in ticks if t.endswith("=F")]
    fx = [t for t in ticks if t.endswith("=X")]
    crypto = [t for t in ticks if t in ("BTC-USD", "ETH-USD")]
    return futures, fx + crypto


def _research_book(universe):
    """Deep-copied STRATEGY_BOOK with every strategy pointed at the non-equity
    universe. Liquidity floors zeroed (research replay — see module docstring);
    everything else (filters, sizing, exits, overlays) stays prod."""
    book = copy.deepcopy(STRATEGY_BOOK)
    for s in book:
        s["universe_tickers"] = list(universe)
        s["settings"]["min_price"] = 0.0
        s["settings"]["min_vol"] = 0
    return book


def _pf(r):
    pos = r[r > 0].sum()
    neg = r[r < 0].sum()
    if neg == 0:
        return None if pos == 0 else float("inf")
    return float(pos / abs(neg))


def _round(v, d=4):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if np.isnan(f) or np.isinf(f):
        return None
    return round(f, d)


def run_universe(label, universe, sznl_map, atr_sznl_map):
    print(f"\n=== Universe: {label} ({len(universe)} tickers) ===")
    book = _research_book(universe)
    tickers = sorted(set(universe) | {"SPY", "^VIX"})
    md = data_provider.get_history(tickers, start=DATA_START)
    missing = [t for t in universe if t not in md or md[t] is None or md[t].empty]
    if missing:
        print(f"  missing from master: {missing}")

    vix_df = md.get("^VIX")
    vix_series = None
    if vix_df is not None and not vix_df.empty:
        vd = vix_df.copy()
        if isinstance(vd.columns, pd.MultiIndex):
            vd.columns = vd.columns.get_level_values(0)
        vd.columns = [c.capitalize() for c in vd.columns]
        vix_series = vd["Close"]

    processed = precompute_all_indicators(md, book, sznl_map, vix_series, atr_sznl_map)
    candidates, signal_data = generate_candidates_fast(processed, book, sznl_map, BT_START)
    print(f"  {len(candidates)} candidate signal-dates")

    sig = process_signals_fast(
        candidates, signal_data, processed, book, ACCOUNT_VALUE,
        cap_bps=250, overflow_active=False, flat_sizing=True,
    ) if candidates else pd.DataFrame()
    print(f"  {len(sig)} trades executed")

    # candidate signals per strategy (fill-rate denominator / context)
    sig_counts = {}
    for c in candidates:
        name = book[c[3]]["name"]
        sig_counts[name] = sig_counts.get(name, 0) + 1

    n_zero = 0
    if not sig.empty:
        n_zero = int((sig["Shares"] <= 0).sum())
        sig = sig[sig["Shares"] > 0].copy()
        sig["R"] = sig["PnL"] / sig["Risk $"].replace(0, np.nan)
        sig["Direction"] = np.where(
            sig["Action"].astype(str).str.upper().str.contains("SHORT"), "Short", "Long")

    summary = []
    for s in book:
        name = s["name"]
        g = sig[sig["Strategy"] == name] if not sig.empty else pd.DataFrame()
        row = {
            "strategy": name,
            "direction": s["settings"]["trade_direction"],
            "entry_type": s["settings"]["entry_type"],
            "n_signals": sig_counts.get(name, 0),
            "n": int(len(g)),
        }
        if len(g):
            r = g["R"].dropna()
            row.update({
                "tot_r": _round(r.sum(), 2),
                "avg_r": _round(r.mean(), 3),
                "win_pct": _round((g["PnL"] > 0).mean() * 100, 1),
                "pf": _round(_pf(r), 2) if _pf(r) is not None else None,
                "pnl_flat": _round(g["PnL"].sum(), 0),
                "first_trade": str(pd.Timestamp(g["Date"].min()).date()),
                "last_trade": str(pd.Timestamp(g["Date"].max()).date()),
            })
        else:
            row.update({"tot_r": 0.0, "avg_r": None, "win_pct": None, "pf": None,
                        "pnl_flat": 0.0, "first_trade": None, "last_trade": None})
        summary.append(row)
    summary.sort(key=lambda x: -(x["tot_r"] or 0))

    trades_cols = {k: [] for k in
                   ["strategy", "ticker", "direction", "signal_date", "entry_date",
                    "exit_date", "exit_type", "entry_price", "exit_price", "r", "pnl_flat"]}
    if not sig.empty:
        sig = sig.sort_values("Exit Date")
        for _, t in sig.iterrows():
            trades_cols["strategy"].append(t["Strategy"])
            trades_cols["ticker"].append(t["Ticker"])
            trades_cols["direction"].append(t["Direction"])
            trades_cols["signal_date"].append(str(pd.Timestamp(t["Date"]).date()))
            trades_cols["entry_date"].append(str(pd.Timestamp(t["Entry Date"]).date()))
            trades_cols["exit_date"].append(str(pd.Timestamp(t["Exit Date"]).date()))
            trades_cols["exit_type"].append(t["Exit Type"])
            trades_cols["entry_price"].append(_round(t["Price"], 4))
            trades_cols["exit_price"].append(_round(t["Exit Price"], 4))
            trades_cols["r"].append(_round(t["R"], 3))
            trades_cols["pnl_flat"].append(_round(t["PnL"], 0))

    return {
        "label": label,
        "tickers": universe,
        "missing_from_master": missing,
        "n_signals": len(candidates),
        "n_trades": int(len(sig)) if not sig.empty else 0,
        "n_zero_share_dropped": n_zero,
        "summary_by_strategy": summary,
        "trades": {"n": len(trades_cols["strategy"]), "columns": trades_cols},
    }


def main():
    futures, fx_crypto = _non_equity_universes()
    print(f"Futures universe ({len(futures)}): {futures}")
    print(f"FX+crypto universe ({len(fx_crypto)}): {fx_crypto}")

    sznl_map = load_seasonal_map()
    atr_sznl_map = load_atr_seasonal_map()
    if not atr_sznl_map:
        print("WARNING: atr_seasonal_ranks.parquet missing — ATR-seasonal strategies under-fire.")

    out = {
        "computed_asof": str(datetime.date.today()),
        "bt_start": str(BT_START),
        "basis": "flat_750k",
        "engine": "process_signals_fast, prod defaults (cap_bps=250, gap-aware stop fills)",
        "universes": {
            "futures": run_universe("Futures (=F continuous)", futures, sznl_map, atr_sznl_map),
            "fx_crypto": run_universe("FX majors + crypto", fx_crypto, sznl_map, atr_sznl_map),
        },
        "caveats": [
            "RESEARCH ONLY — futures left the tradeable universe 2026-07-06; nothing here can be staged or executed through the equity order path.",
            "yfinance =F series are front-month CONTINUOUS contracts: roll gaps appear as tradeable price moves that never existed. Entry/exit/R arithmetic is distorted around every roll.",
            "min_price / min_vol liquidity floors were zeroed for this replay (prod settings would exclude nearly all FX at ~1.0 quotes with volume=0). All other filters, sizing and exits are prod config.",
            "yfinance FX volume is 0, so volume-gated strategies (e.g. OLV's vol-rank filter, 52wh's volume spike) structurally cannot fire on FX — their n=0 there is a data artifact, not evidence.",
            "Crypto trades 7 days/week: hold_days count bars, so a 2-bar hold is ~2 calendar days incl. weekends; equity-calendar conventions (BDay gates) do not map cleanly.",
            "Futures holiday/overnight sessions violate equity-calendar assumptions the engine was built on (one reason they were removed from the universe).",
            "Integer-share sizing zeroes out very high-priced instruments (e.g. BTC-USD when stop distance > risk budget); those signals are dropped and counted in n_zero_share_dropped.",
            "Tiny sample sizes for most strategy x universe cells — treat every cell with n < 30 as an anecdote, not an estimate.",
            "Earnings-based gates (OVS blackout, OLV pre-earnings sizing) pass through on these tickers (no earnings data = NaN-as-True), matching prod convention.",
            "Computed once on the date stamped above; NOT rebuilt nightly.",
        ],
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"), allow_nan=False)
    size_kb = os.path.getsize(OUT_JSON) / 1024
    print(f"\nWrote {OUT_JSON} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
