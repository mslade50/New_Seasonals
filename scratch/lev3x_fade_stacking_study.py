"""Stacking study (2026-07-27): what if the two 3x Overbot Fades allowed
multiple concurrent positions per ticker (max_one_pos=False)?

Context: live has no max_one_pos guard (daily_scan / order_staging never check
open positions), so consecutive-day re-fires stack live while the engine books
only the first leg (SQQQ 7/23 + 7/24 + 7/27 signals -> live short 2,159 sh vs
one modeled 1,246-sh leg). This measures what the STACKED book would have done
2003->present so we can decide which side to align: block live re-entries, or
adopt stacking in the model.

Method: identical candidates (filters unchanged), two process_signals_fast
passes on flat $750k sizing with prod caps - baseline book vs a deep copy with
max_one_pos=False on both strategies. Whole-variant comparison first (totR,
PnL, maxDD, worst windows); the marginal stacked legs are also profiled since
they are a deterministic signal subset, not a price-path-selected fill stream.
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
from strategy_config import STRATEGY_BOOK, ACCOUNT_VALUE
from pages.strat_backtester import (
    load_seasonal_map,
    load_atr_seasonal_map,
    precompute_all_indicators,
    generate_candidates_fast,
    process_signals_fast,
    get_daily_mtm_series,
)

BT_START = datetime.date(2003, 1, 1)
STRATS = ["3x ETF Overbot Fade", "3x Bear ETF Overbot Fade"]


def build_books():
    base = [copy.deepcopy(s) for s in STRATEGY_BOOK if s["name"] in STRATS]
    stack = copy.deepcopy(base)
    for s in stack:
        s["settings"]["max_one_pos"] = False
    return base, stack


def max_concurrency(g):
    """Max simultaneous open legs in one ticker (entry..exit inclusive)."""
    best = 1
    for _, tg in g.groupby("Ticker"):
        events = []
        for _, r in tg.iterrows():
            events.append((r["Entry Date"], 1))
            events.append((r["Exit Date"] + pd.Timedelta(days=1), -1))
        depth = 0
        for _, d in sorted(events, key=lambda e: (e[0], -e[1])):
            depth += d
            best = max(best, depth)
    return best


def summarize(sig, md, label):
    rows = []
    for name in STRATS:
        g = sig[sig["Strategy"] == name].copy()
        r = g["R_Multiple"].astype(float)
        pos, neg = r[r > 0].sum(), r[r <= 0].sum()
        mtm = get_daily_mtm_series(g, md)
        eq = mtm.cumsum()
        dd = eq - eq.cummax()
        w5 = mtm.rolling(5).sum()
        rows.append({
            "variant": label, "strategy": name, "n": len(g),
            "totR": r.sum(), "avgR": r.mean(), "win%": (r > 0).mean() * 100,
            "PF": pos / abs(neg) if neg else np.inf,
            "worstR": r.min(),
            "PnL_flat": g["PnL"].sum(),
            "maxDD_flat": dd.min(),
            "worst5d_flat": w5.min(),
            "maxstack": max_concurrency(g),
        })
    return rows


def main():
    base_book, stack_book = build_books()
    tickers = set()
    for s in base_book:
        tickers.update(s["universe_tickers"])
    tickers.update(["SPY", "^VIX"])

    md = data_provider.get_history(sorted(tickers), start="2000-01-01")
    vix_df = md.get("^VIX")
    vix_series = None
    if vix_df is not None and not vix_df.empty:
        vd = vix_df.copy()
        vd.columns = [c.capitalize() for c in vd.columns]
        vix_series = vd["Close"]

    sznl_map = load_seasonal_map()
    atr_sznl_map = load_atr_seasonal_map()

    processed = precompute_all_indicators(md, base_book, sznl_map, vix_series,
                                          atr_sznl_map)
    candidates, signal_data = generate_candidates_fast(processed, base_book,
                                                       sznl_map, BT_START)
    print(f"{len(candidates)} candidates (shared across variants)")

    sig_base = process_signals_fast(candidates, signal_data, processed,
                                    base_book, ACCOUNT_VALUE, cap_bps=250,
                                    flat_sizing=True)
    sig_stack = process_signals_fast(candidates, signal_data, processed,
                                     stack_book, ACCOUNT_VALUE, cap_bps=250,
                                     flat_sizing=True)
    for sig in (sig_base, sig_stack):
        sig["Date"] = pd.to_datetime(sig["Date"])
        sig["Entry Date"] = pd.to_datetime(sig["Entry Date"])
        sig["Exit Date"] = pd.to_datetime(sig["Exit Date"])
        sig["R_Multiple"] = sig["PnL"] / sig["Risk $"].replace(0, np.nan)

    summary = pd.DataFrame(summarize(sig_base, md, "baseline (max_one_pos)")
                           + summarize(sig_stack, md, "stacked (multi-pos)"))
    pd.set_option("display.width", 200)
    print("\n=== WHOLE-VARIANT COMPARISON (flat $750k) ===")
    print(summary.round(2).to_string(index=False))

    # marginal legs = stacked-only trades (keyed by strategy+ticker+signal date)
    key = ["Strategy", "Ticker", "Date"]
    kb = set(map(tuple, sig_base[key].astype(str).values))
    marg = sig_stack[~sig_stack[key].astype(str).apply(tuple, axis=1).isin(kb)]
    print("\n=== MARGINAL STACKED LEGS (blocked today by max_one_pos) ===")
    for name in STRATS:
        m = marg[marg["Strategy"] == name]
        r = m["R_Multiple"].astype(float)
        if not len(m):
            print(f"{name}: none")
            continue
        pos, neg = r[r > 0].sum(), r[r <= 0].sum()
        print(f"{name}: n={len(m)}  totR={r.sum():+.1f}  avgR={r.mean():+.3f}  "
              f"win={100 * (r > 0).mean():.0f}%  PF={pos / abs(neg) if neg else np.inf:.2f}  "
              f"worst={r.min():+.2f}  PnL=${m['PnL'].sum():,.0f}")

    print("\n=== PER-YEAR totR ===")
    yr = []
    for label, sig in (("base", sig_base), ("stack", sig_stack)):
        for name in STRATS:
            g = sig[sig["Strategy"] == name]
            s = g.groupby(g["Date"].dt.year)["R_Multiple"].sum()
            s.name = f"{name[:12]}|{label}"
            yr.append(s)
    print(pd.concat(yr, axis=1).fillna(0).round(1).to_string())

    print("\n=== STACKED VARIANT: 2026-07 SQQQ legs (ties to live) ===")
    sq = sig_stack[(sig_stack["Ticker"] == "SQQQ")
                   & (sig_stack["Date"] >= "2026-07-01")]
    cols = [c for c in ("Strategy", "Date", "Entry Date", "Exit Date",
                        "Exit Type", "Price", "R_Multiple", "Shares", "PnL")
            if c in sq.columns]
    print(sq[cols].to_string(index=False))

    out = os.path.join(_HERE, "lev3x_fade_stacking_results.csv")
    summary.to_csv(out, index=False)
    marg.to_csv(os.path.join(_HERE, "lev3x_fade_stacking_marginal_legs.csv"),
                index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
