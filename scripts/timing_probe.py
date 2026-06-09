"""
timing_probe.py — measure where the full-book ledger rebuild spends its time,
so we can right-size a daily-incremental shortcut. Prints per-phase wall time.
Run with the indicator cache WARM (data unchanged) to isolate non-precompute cost.
"""
import os, sys, time
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import data_provider
from strategy_config import ACCOUNT_VALUE
from pages.strat_backtester import (
    load_seasonal_map, load_atr_seasonal_map, precompute_all_indicators,
    generate_candidates_fast, process_signals_fast, get_daily_mtm_series,
)
from daily_portfolio_report import build_full_strategy_book

def t(msg, t0):
    dt = time.time() - t0
    print(f"  [{dt:7.1f}s]  {msg}", flush=True)
    return time.time()

def main():
    import datetime
    BT_START = datetime.date(2003, 1, 1)
    print("TIMING PROBE — full-book ledger phases (cache warm)", flush=True)
    t0 = time.time(); s = t0
    book = build_full_strategy_book()
    sznl = load_seasonal_map(); atr = load_atr_seasonal_map()
    tickers = set()
    for st_ in book: tickers.update(st_["universe_tickers"])
    tickers.update(["SPY", "^VIX"])
    s = t(f"setup + maps ({len(tickers)} tickers)", s)

    md = data_provider.get_history(list(tickers), start="2000-01-01")
    s = t(f"load_data ({len(md)} loaded)", s)

    vix = md.get("^VIX"); vix_s = None
    if vix is not None and not vix.empty:
        v = vix.copy()
        if isinstance(v.columns, pd.MultiIndex): v.columns = v.columns.get_level_values(0)
        v.columns = [c.capitalize() for c in v.columns]; vix_s = v["Close"]

    proc = precompute_all_indicators(md, book, sznl, vix_s, atr)
    s = t(f"precompute_all_indicators ({len(proc)} frames)  <-- the big one", s)

    cand, sigdata = generate_candidates_fast(proc, book, sznl, BT_START)
    s = t(f"generate_candidates_fast ({len(cand)} candidates)", s)

    sig = process_signals_fast(cand, sigdata, proc, book, ACCOUNT_VALUE, cap_bps=250, overflow_active=True)
    s = t(f"process_signals_fast x1 ({len(sig)} trades)", s)

    sigf = process_signals_fast(cand, sigdata, proc, book, ACCOUNT_VALUE, cap_bps=250, overflow_active=True, flat_sizing=True)
    s = t("process_signals_fast x1 (flat)", s)

    sig["Entry Date"] = pd.to_datetime(sig["Entry Date"])
    _ = get_daily_mtm_series(sig, md, start_date=BT_START)
    s = t("get_daily_mtm_series", s)

    print(f"  TOTAL: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
