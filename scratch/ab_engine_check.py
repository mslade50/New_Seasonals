"""A/B parity harness for the 2026-08-12 engine fixes.

Runs the full-book flat-sizing engine pass (the ledger's basis: cap_bps=250,
overflow_active=True, flat_sizing=True) in whichever tree it is invoked from
and dumps sig_df to --out. Run once from the HEAD worktree (old engine) and
once from the working tree (new engine) over the SAME data dir, then diff
with ab_engine_diff.py. R2 refresh must be disabled by the caller (env) so
the shared parquet cannot change between the two runs.
"""
import argparse
import datetime
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)


class _NoOpStreamlit:
    def __getattr__(self, name):
        def f(*a, **k):
            return self  # chainable: st.empty().text(...) etc.
        return f

    def __call__(self, *a, **k): return self
    def __enter__(self): return self
    def __exit__(self, *a): return False

    def cache_data(self, *a, **k):
        def deco(fn):
            return fn
        if a and callable(a[0]):
            return a[0]
        return deco

    cache_resource = cache_data


sys.modules["streamlit"] = _NoOpStreamlit()

import data_provider  # noqa: E402
from strategy_config import ACCOUNT_VALUE  # noqa: E402
from pages.strat_backtester import (  # noqa: E402
    load_seasonal_map, load_atr_seasonal_map, precompute_all_indicators,
    generate_candidates_fast, process_signals_fast,
)
from daily_portfolio_report import build_full_strategy_book  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    BT_START = datetime.date(2003, 1, 1)

    book = build_full_strategy_book()
    sznl = load_seasonal_map()
    atr = load_atr_seasonal_map()
    tickers = set()
    for st_ in book:
        tickers.update(st_["universe_tickers"])
    tickers.update(["SPY", "^VIX"])
    md = data_provider.get_history(list(tickers), start="2000-01-01")
    print(f"loaded {len(md)} tickers", flush=True)

    vix = md.get("^VIX")
    vix_s = None
    if vix is not None and not vix.empty:
        v = vix.copy()
        if isinstance(v.columns, pd.MultiIndex):
            v.columns = v.columns.get_level_values(0)
        v.columns = [c.capitalize() for c in v.columns]
        vix_s = v["Close"]

    proc = precompute_all_indicators(md, book, sznl, vix_s, atr)
    cand, sigdata = generate_candidates_fast(proc, book, sznl, BT_START)
    print(f"{len(cand)} candidates", flush=True)

    sig = process_signals_fast(cand, sigdata, proc, book, ACCOUNT_VALUE,
                               cap_bps=250, overflow_active=True,
                               flat_sizing=True)
    print(f"{len(sig)} trades", flush=True)
    sig.to_parquet(args.out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
