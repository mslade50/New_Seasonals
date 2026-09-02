"""Dump the engine's RAW candidate signal-dates (the same generate_candidates_fast
call scripts/build_trade_ledger.py makes) to a parquet in this scratch folder.

Why: data/backtest_trades_full.parquet holds FILLS only. Signal-flow state must
be measured on STAGED signals (what the scanner sees at the close), including
the ones whose limits never fill. Read-only against repo code; writes only to
scratch/ultracode_sizing_2026-09-02/flow_candidates.parquet.
"""
from __future__ import annotations
import datetime
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:/Users/McKinley Slade/dev/New_Seasonals")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
OUT_DIR = ROOT / "scratch/ultracode_sizing_2026-09-02"

import data_provider  # noqa: E402
from pages.strat_backtester import (  # noqa: E402
    load_seasonal_map, load_atr_seasonal_map, precompute_all_indicators, generate_candidates_fast,
)
from daily_portfolio_report import build_full_strategy_book, OVERFLOW_TICKERS  # noqa: E402
from strategy_config import LIQUID_PLUS_COMMODITIES  # noqa: E402

DATA_START = datetime.date(2000, 1, 1)
BT_START = datetime.date(2003, 1, 1)

t0 = time.time()
full_book = build_full_strategy_book()
sznl_map = load_seasonal_map()
atr_sznl_map = load_atr_seasonal_map()
all_tickers = set()
for s in full_book:
    all_tickers.update(s["universe_tickers"])
all_tickers.update(["SPY", "^VIX"])
print(f"loading {len(all_tickers)} tickers ...", flush=True)
md = data_provider.get_history(list(all_tickers), start=DATA_START.strftime("%Y-%m-%d"))
vix_df = md.get("^VIX")
vix_series = None
if vix_df is not None and not vix_df.empty:
    vd = vix_df.copy()
    if isinstance(vd.columns, pd.MultiIndex):
        vd.columns = vd.columns.get_level_values(0)
    vd.columns = [c.capitalize() for c in vd.columns]
    vix_series = vd["Close"]
print(f"loaded in {time.time()-t0:.0f}s; precomputing indicators ...", flush=True)
processed = precompute_all_indicators(md, full_book, sznl_map, vix_series, atr_sznl_map)
print(f"indicators in {time.time()-t0:.0f}s; generating candidates ...", flush=True)
candidates, signal_data = generate_candidates_fast(processed, full_book, sznl_map, BT_START)
print(f"{len(candidates)} candidates in {time.time()-t0:.0f}s", flush=True)

liquid = set(LIQUID_PLUS_COMMODITIES)
rows = []
for sig_val, ticker, t_clean, strat_idx, signal_idx in candidates:
    strat = full_book[strat_idx]
    sd = signal_data.get((t_clean, signal_idx), {})
    rows.append(dict(
        signal_date=pd.Timestamp(sig_val), ticker=ticker, strategy=strat["name"],
        strat_idx=strat_idx, risk_bps=float(strat["execution"].get("risk_bps", float("nan"))),
        atr=sd.get("atr"), close=sd.get("close"), open=sd.get("open"),
    ))
cand = pd.DataFrame(rows)
# tier: overflow variants are appended AFTER the liquid book, so index >= len(STRATEGY_BOOK) is overflow
from strategy_config import STRATEGY_BOOK  # noqa: E402
cand["tier"] = ["Overflow" if i >= len(STRATEGY_BOOK) else "Liquid" for i in cand["strat_idx"]]
cand = cand.sort_values(["signal_date", "strategy", "ticker"]).reset_index(drop=True)
cand.to_parquet(OUT_DIR / "flow_candidates.parquet", index=False)
print(cand.groupby(["strategy", "tier"]).size().to_string())
print(f"wrote {OUT_DIR / 'flow_candidates.parquet'} rows={len(cand)} in {time.time()-t0:.0f}s")
