"""
Build the per-ticker indicator cache for the active strategy book and
mirror the results to R2 so Streamlit Cloud (and any fresh checkout)
can pull precomputed parquets instead of doing the multi-minute
expanding-rank build on every cold start.

Runs in two passes so both backtester modes are covered:
    1. Default per-strategy universes (typically LIQUID_PLUS_COMMODITIES).
    2. Overflow swap — OVERFLOW_ELIGIBLE_STRATEGIES re-pointed at
       CSV_UNIVERSE | seasonal_map keys, mirroring the
       `use_overflow_universe` toggle in pages/strat_backtester.py.

Each pass writes per-ticker parquets to data/bt_indicator_cache/ with a
filename hash derived from (ticker, row count, first/last date,
indicator params, indicator version). The script tracks every file
created or refreshed during the run and uploads only those to R2 under
the `bt_indicator_cache/` key prefix.

Usage (locally, mainly for dry-runs):
    python scripts/build_indicator_cache.py            # both passes, upload to R2
    python scripts/build_indicator_cache.py --no-upload  # build only
    python scripts/build_indicator_cache.py --liquid-only

In the weekly GHA workflow the script runs with the R2 secrets in the
environment; uploads are gated on those being present.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

# Stub streamlit before any project import — strat_backtester decorates
# functions with @st.cache_data/@st.cache_resource and calls st.progress
# inside the precompute loop. The decorator stubs handle both
# `@st.cache_data` (bare) and `@st.cache_data(ttl=...)` (parens) forms.
class _NoOp:
    def __getattr__(self, name):
        return self
    def __call__(self, *a, **k):
        return self
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _cache_passthrough(*a, **k):
    # Used as `@st.cache_data` with no parens: arg is the function itself.
    if len(a) == 1 and callable(a[0]) and not k:
        return a[0]
    # Used as `@st.cache_data(ttl=...)`: return a decorator that returns fn.
    def deco(fn): return fn
    return deco


_st_stub = _NoOp()
_st_stub.cache_data = _cache_passthrough
_st_stub.cache_resource = _cache_passthrough
sys.modules['streamlit'] = _st_stub

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'pages'))

import data_provider  # noqa: E402
from cache_io import is_configured, upload_from_local, download_to_local  # noqa: E402
from strategy_config import (  # noqa: E402
    _STRATEGY_BOOK_RAW,
    CSV_UNIVERSE,
)

# These live in pages/strat_backtester.py — we reuse the production
# functions to keep the cache key formula in lockstep with the page.
from strat_backtester import (  # noqa: E402
    load_seasonal_map,
    load_atr_seasonal_map,
    precompute_all_indicators,
    OVERFLOW_ELIGIBLE_STRATEGIES,
)

CACHE_DIR = os.path.join(ROOT, 'data', 'bt_indicator_cache')
R2_PREFIX = 'bt_indicator_cache'
EARNINGS_LOCAL = os.path.join(ROOT, 'data', 'earnings_calendar.parquet')


def _pull_caches_from_r2():
    """Best-effort pull of master_prices + earnings before building."""
    if not is_configured():
        print('[build_indicator_cache] R2 not configured - using whatever is on disk')
        return
    if not data_provider.has_master():
        raise SystemExit('master_prices.parquet not available locally or via R2 - aborting')
    if not os.path.exists(EARNINGS_LOCAL):
        download_to_local('earnings_calendar.parquet', EARNINGS_LOCAL)


def _load_vix_series():
    """Daily VIX close series from master_prices (^VIX)."""
    hist = data_provider.get_history(['^VIX'])
    vix = hist.get('^VIX')
    if vix is None or vix.empty:
        return None
    return vix['Close']


def _snapshot_cache_mtimes() -> dict:
    """Map of cache filename -> mtime before the build, so we can detect
    which files were created/refreshed during this run."""
    if not os.path.isdir(CACHE_DIR):
        return {}
    out = {}
    for fname in os.listdir(CACHE_DIR):
        if not fname.endswith('.parquet'):
            continue
        full = os.path.join(CACHE_DIR, fname)
        try:
            out[fname] = os.path.getmtime(full)
        except OSError:
            pass
    return out


def _detect_changed(before: dict) -> list:
    """Filenames whose mtime changed (or that are new) since `before`."""
    changed = []
    for fname in os.listdir(CACHE_DIR):
        if not fname.endswith('.parquet'):
            continue
        full = os.path.join(CACHE_DIR, fname)
        try:
            m = os.path.getmtime(full)
        except OSError:
            continue
        if before.get(fname) != m:
            changed.append(fname)
    return changed


def _upload_changed(changed: list, dry: bool) -> int:
    """Upload each new/refreshed file to R2 under R2_PREFIX/. Returns count."""
    if dry:
        print(f'[build_indicator_cache] --no-upload set; skipping {len(changed)} file uploads')
        return 0
    if not is_configured():
        print('[build_indicator_cache] R2 not configured; skipping upload')
        return 0
    ok = 0
    t0 = time.time()
    # Concurrency = 8 mirrors the production ThreadPoolExecutor in
    # precompute_all_indicators. R2 hasn't shown trouble at this rate.
    def _one(fname):
        local = os.path.join(CACHE_DIR, fname)
        key = f'{R2_PREFIX}/{fname}'
        return upload_from_local(local, key)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_one, f): f for f in changed}
        for i, fut in enumerate(as_completed(futures), 1):
            if fut.result():
                ok += 1
            if i % 100 == 0 or i == len(changed):
                rate = i / max(time.time() - t0, 1e-3)
                print(f'  uploaded {i:>5}/{len(changed)} ({rate:.1f}/s, ok={ok})')
    print(f'[build_indicator_cache] uploaded {ok}/{len(changed)} files in {time.time()-t0:.1f}s')
    return ok


def _build_strategies(label: str, sznl_map, overflow: bool) -> list:
    """Return a fresh deep-copy of the strategy book, with overflow tickers
    swapped in for OVERFLOW_ELIGIBLE_STRATEGIES when overflow=True."""
    strategies = [copy.deepcopy(s) for s in _STRATEGY_BOOK_RAW]
    if overflow:
        overflow_tickers = sorted(set(CSV_UNIVERSE) | set(sznl_map.keys()))
        for s in strategies:
            if s['name'] in OVERFLOW_ELIGIBLE_STRATEGIES:
                s['universe_tickers'] = overflow_tickers
        print(f'[{label}] overflow universe = {len(overflow_tickers):,} tickers')
    return strategies


def _build_master_dict(strategies, sznl_map) -> dict:
    """Same long_term_list construction the page does, then pull from master_prices."""
    long_term_tickers = set()
    for strat in strategies:
        long_term_tickers.update(strat['universe_tickers'])
        s = strat['settings']
        if s.get('use_market_sznl'):
            long_term_tickers.add(s.get('market_ticker', '^GSPC'))
        if 'Market' in s.get('trend_filter', ''):
            long_term_tickers.add(s.get('market_ticker', 'SPY'))
        if s.get('use_vix_filter'):
            long_term_tickers.add('^VIX')
    long_term_tickers.add('SPY')
    long_term_list = [t.replace('.', '-') for t in long_term_tickers]
    md = data_provider.get_history(long_term_list)
    n_missing = len(set(long_term_list) - set(md.keys()))
    print(f'  loaded {len(md):,}/{len(long_term_list)} tickers from master_prices '
          f'({n_missing} missing)')
    return md


def _run_pass(label: str, overflow: bool, sznl_map, atr_sznl_map, vix_series, dry_upload: bool):
    print(f'\n=== {label} pass (overflow={overflow}) ===')
    strategies = _build_strategies(label, sznl_map, overflow)
    md = _build_master_dict(strategies, sznl_map)
    before = _snapshot_cache_mtimes()
    t0 = time.time()
    precompute_all_indicators(md, strategies, sznl_map, vix_series, atr_sznl_map)
    elapsed = time.time() - t0
    changed = _detect_changed(before)
    print(f'[{label}] precompute took {elapsed:.1f}s; {len(changed):,} cache files touched')
    if changed:
        _upload_changed(changed, dry_upload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-upload', action='store_true', help='Skip the R2 upload step.')
    ap.add_argument('--liquid-only', action='store_true', help='Skip the overflow pass.')
    ap.add_argument('--overflow-only', action='store_true', help='Skip the liquid pass.')
    args = ap.parse_args()

    _pull_caches_from_r2()
    os.makedirs(CACHE_DIR, exist_ok=True)

    print('Loading seasonal maps...')
    sznl_map = load_seasonal_map()
    atr_sznl_map = load_atr_seasonal_map()
    vix_series = _load_vix_series()
    print(f'  sznl_map={len(sznl_map):,} tickers, atr_sznl_map={len(atr_sznl_map):,} tickers, '
          f'vix_series={"yes" if vix_series is not None else "no"}')

    if not args.overflow_only:
        _run_pass('liquid', False, sznl_map, atr_sznl_map, vix_series, args.no_upload)
    if not args.liquid_only:
        _run_pass('overflow', True, sznl_map, atr_sznl_map, vix_series, args.no_upload)

    print('\nDone.')


if __name__ == '__main__':
    main()
