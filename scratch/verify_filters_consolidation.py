"""Parity harness for the filters.py consolidation (2026-07-16).

Compares the RETIRED implementations (extracted from git HEAD before the
splice) against the consolidated filters.py, on REAL master prices across
the FULL strategy book:

1. SCAN side: old daily_scan.check_signal vs filters.check_signal_live —
   must be 100% identical booleans (every unification adopted scan
   semantics, so zero diffs are expected).
2. ENGINE side: old strat_backtester.get_historical_mask vs
   filters.evaluate_filter_mask(mode='backtest') — full series compared.
   The ONLY acceptable diffs are the deliberate scan-truth corrections:
     a. ETF_ATR_EXEMPT tickers (SPY/QQQ/IWM/DIA) newly passing rows the old
        mask rejected purely on the min ATR%-floor (live always took these).
     b. use_range_atr_filter rows with non-positive/NaN ATR newly passing
        (scan passes them through; old mask NaN-rejected).
   Any diff outside those classes fails the harness.

Exit 0 only on full parity. Usage:
    git show <pre-splice-sha>:daily_scan.py > scratch/_old_daily_scan.py
    git show <pre-splice-sha>:pages/strat_backtester.py > scratch/_old_strat_backtester.py
    python scratch/verify_filters_consolidation.py [--per-strategy N]
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class _NoOp:
    def __getattr__(self, name):
        def f(*a, **k):
            return self
        return f
    def __call__(self, *a, **k): return self
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def cache_data(self, *a, **k):
        def deco(fn): return fn
        return deco
    cache_resource = cache_data


sys.modules['streamlit'] = _NoOp()


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-strategy", type=int, default=9999)
    args = ap.parse_args()

    import daily_scan as ds
    import filters
    from strategy_config import STRATEGY_BOOK

    old_ds = _load_module("_old_daily_scan",
                          os.path.join(ROOT, "scratch", "_old_daily_scan.py"))
    old_bt = _load_module("_old_strat_backtester",
                          os.path.join(ROOT, "scratch", "_old_strat_backtester.py"))

    tickers = set()
    for s in STRATEGY_BOOK:
        tickers.update(t.replace('.', '-') for t in s['universe_tickers'][:args.per_strategy])
        tickers.add(str(s['settings'].get('market_ticker', 'SPY')).replace('.', '-'))
        if s['settings'].get('use_ref_ticker_filter') and s['settings'].get('ref_filters'):
            tickers.add(str(s['settings'].get('ref_ticker', 'IWM')).replace('.', '-'))
    tickers.update(['SPY', '^VIX'])

    print(f"Loading {len(tickers)} tickers...")
    master = ds.load_master_prices_dict(sorted(tickers))
    master = {t: d for t, d in master.items() if d is not None and len(d) >= 250}
    print(f"  {len(master)} usable")

    sznl_map = ds.load_seasonal_map()
    atr_sznl_map = ds.load_atr_seasonal_map()

    # Build cross-sectional rank matrices like the prod scan does (daily_scan
    # step 4b) — without them, frames lack xsec_rank_ret_* and the old engine
    # silently passed while the unified code evaluates the scan's neutral 50,
    # producing harness-only diffs on Sector BO (the book's one xsec user).
    xsec_windows = set()
    for s in STRATEGY_BOOK:
        if s['settings'].get('use_xsec_filter') and s['settings'].get('xsec_filters'):
            xsec_windows.update(xf['window'] for xf in s['settings']['xsec_filters'])
    xsec_rank_matrices = None
    if xsec_windows:
        RANK_MIN_PERIODS = 252
        rank_dict = {}
        for t, d in master.items():
            if d is None or 'Close' not in d.columns or len(d) < 50:
                continue
            for w in xsec_windows:
                ret = d['Close'].pct_change(w)
                tp = ret.expanding(min_periods=RANK_MIN_PERIODS).rank(pct=True) * 100.0
                rank_dict.setdefault(w, {})[t] = tp
        xsec_rank_matrices = {}
        for w in xsec_windows:
            if rank_dict.get(w):
                mat = pd.DataFrame(rank_dict[w])
                xsec_rank_matrices[w] = mat.rank(axis=1, pct=True) * 100.0
        print(f"  xsec matrices built for windows {sorted(xsec_windows)}")

    vix_df = master.get('^VIX')
    vix_series = None
    if vix_df is not None and not vix_df.empty:
        tv = vix_df.copy()
        tv.columns = [c.capitalize() for c in tv.columns]
        if tv.index.tz is not None:
            tv.index = tv.index.tz_localize(None)
        vix_series = tv['Close']

    memo = {}
    scan_pairs = scan_ok = 0
    eng_pairs = eng_rows = eng_diff_expected = 0
    failures = []

    for strat in STRATEGY_BOOK:
        st = strat['settings']
        mkt_ticker = st.get('market_ticker', 'SPY')
        mkt_df = master.get(mkt_ticker)
        if mkt_df is None:
            mkt_df = master.get('SPY')
        market_series = None
        if mkt_df is not None:
            tm = mkt_df.copy()
            tm['SMA200'] = tm['Close'].rolling(200).mean()
            market_series = tm['Close'] > tm['SMA200']
        mkt_key = mkt_ticker if master.get(mkt_ticker) is not None else (
            'SPY' if master.get('SPY') is not None else None)

        ref_ranks, ref_key = None, None
        if st.get('use_ref_ticker_filter') and st.get('ref_filters'):
            rk = str(st.get('ref_ticker', 'IWM')).replace('.', '-')
            rdf = master.get(rk)
            if rdf is not None and len(rdf) > 250:
                rcalc = ds.calculate_indicators(rdf.copy(), sznl_map, rk, market_series, vix_series)
                ref_ranks = {}
                for rf in st['ref_filters']:
                    col = f"rank_ret_{rf['window']}d"
                    if col in rcalc.columns:
                        ref_ranks[rf['window']] = rcalc[col]
                if ref_ranks:
                    ref_key = (rk, tuple(sorted(ref_ranks)))

        min_atr = float(st.get('min_atr_pct', 0) or 0)

        for ticker in strat['universe_tickers'][:args.per_strategy]:
            t_clean = ticker.replace('.', '-')
            df = master.get(t_clean)
            if df is None or len(df) < 250:
                continue
            frame = ds.memoized_indicators(
                memo, (t_clean, mkt_key, ref_key), df, sznl_map, t_clean,
                market_series, vix_series, ref_ranks, xsec_rank_matrices,
                atr_sznl_map)

            # ---- scan side: strict boolean parity ----
            scan_pairs += 1
            s_old = bool(old_ds.check_signal(frame, st, sznl_map, ticker=t_clean))
            s_new = bool(filters.check_signal_live(frame, st, sznl_map=sznl_map, ticker=t_clean))
            if s_old == s_new:
                scan_ok += 1
            else:
                failures.append(f"SCAN {strat['name']}/{t_clean}: old={s_old} new={s_new}")

            # ---- engine side: series parity with whitelisted corrections ----
            eng_pairs += 1
            m_old = old_bt.get_historical_mask(frame, st, sznl_map, ticker_name=t_clean).values
            m_new = filters.evaluate_filter_mask(
                frame, st, sznl_map=sznl_map, ticker_name=t_clean, mode='backtest').values
            eng_rows += len(m_new)
            diff = m_old != m_new
            if not diff.any():
                continue
            atr_pct = frame['ATR_Pct'].values if 'ATR_Pct' in frame.columns else np.full(len(frame), np.nan)
            atr_v = frame['ATR'].values.astype(float)
            exempt = t_clean.upper() in filters.ETF_ATR_EXEMPT
            for i in np.where(diff)[0]:
                new_passes = bool(m_new[i]) and not bool(m_old[i])
                floor_case = (exempt and min_atr > 0 and new_passes
                              and (np.isnan(atr_pct[i]) or atr_pct[i] < min_atr))
                range_atr_case = (st.get('use_range_atr_filter', False) and new_passes
                                  and (np.isnan(atr_v[i]) or atr_v[i] <= 0))
                if floor_case or range_atr_case:
                    eng_diff_expected += 1
                else:
                    failures.append(
                        f"ENGINE {strat['name']}/{t_clean} @ {frame.index[i].date()}: "
                        f"old={bool(m_old[i])} new={bool(m_new[i])} (unexplained)")
                    if len(failures) > 40:
                        break

    print(f"\nscan pairs:            {scan_pairs} | identical: {scan_ok}")
    print(f"engine pairs:          {eng_pairs} | rows compared: {eng_rows:,}")
    print(f"engine diffs, all in whitelisted correction classes: {eng_diff_expected}")
    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for f in failures[:25]:
            print(f"  {f}")
        return 1
    print("\nPARITY: scan 100% identical; every engine diff is a documented "
          "scan-truth correction (ATR-floor exemption / range-ATR warmup).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
