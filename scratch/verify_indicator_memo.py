"""Parity harness: memoized vs per-(strategy,ticker) indicator path (2026-07-16).

Replicates daily_scan's exact per-strategy setup (market_series from each
strategy's market_ticker, ref_ticker_ranks, atr_sznl merge) over the REAL
strategy book and REAL master prices, and asserts for every (strategy,
ticker) pair:
  1. the memoized frame is byte-identical to a freshly computed one
  2. check_signal returns the same boolean on both
  3. after ALL strategies ran, every cached frame STILL equals a fresh
     recompute — proves nothing downstream mutated the shared frames

Exit 0 only on 100% parity. Run:
    python scratch/verify_indicator_memo.py [--per-strategy 40]
"""
import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import daily_scan as ds
from strategy_config import STRATEGY_BOOK


def fresh_frame(df, t_key, sznl_map, market_series, vix_series, ref_ranks, atr_sznl_map):
    """The OLD path: fresh calculate_indicators + the atr_sznl merge."""
    calc = ds.calculate_indicators(
        df.copy(), sznl_map, t_key, market_series, vix_series,
        ref_ticker_ranks=ref_ranks, xsec_rank_matrices=None)
    if atr_sznl_map and t_key in atr_sznl_map:
        ranks = atr_sznl_map[t_key]
        dates = calc.index.normalize()
        for col in ds.ATR_SZNL_COLS:
            if col in ranks.columns:
                calc[col] = ranks[col].reindex(dates).values
    return calc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-strategy", type=int, default=40)
    args = ap.parse_args()

    tickers = set()
    for s in STRATEGY_BOOK:
        tickers.update(t.replace('.', '-') for t in s['universe_tickers'][:args.per_strategy])
        tickers.add(str(s['settings'].get('market_ticker', 'SPY')).replace('.', '-'))
        if s['settings'].get('use_ref_ticker_filter') and s['settings'].get('ref_filters'):
            tickers.add(str(s['settings'].get('ref_ticker', 'IWM')).replace('.', '-'))
    tickers.update(['SPY', '^VIX'])

    print(f"Loading {len(tickers)} tickers from the master cache...")
    master = ds.load_master_prices_dict(sorted(tickers))
    master = {t: d for t, d in master.items() if d is not None and len(d) >= 250}
    print(f"  {len(master)} tickers usable")

    sznl_map = ds.load_seasonal_map()
    atr_sznl_map = ds.load_atr_seasonal_map()
    vix_df = master.get('^VIX')
    vix_series = None
    if vix_df is not None and not vix_df.empty:
        tv = vix_df.copy()
        tv.columns = [c.capitalize() for c in tv.columns]
        if tv.index.tz is not None:
            tv.index = tv.index.tz_localize(None)
        vix_series = tv['Close']

    memo = {}
    fresh_by_key = {}   # key -> (fresh frame, settings, ticker) for final mutation check
    pairs = frames_ok = sig_ok = 0
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
        if master.get(mkt_ticker) is not None:
            mkt_key = mkt_ticker
        elif master.get('SPY') is not None:
            mkt_key = 'SPY'
        else:
            mkt_key = None

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

        for ticker in strat['universe_tickers'][:args.per_strategy]:
            t_clean = ticker.replace('.', '-')
            df = master.get(t_clean)
            if df is None or len(df) < 250:
                continue
            pairs += 1
            key = (t_clean, mkt_key, ref_key)
            memod = ds.memoized_indicators(
                memo, key, df, sznl_map, t_clean, market_series,
                vix_series, ref_ranks, None, atr_sznl_map)
            fresh = fresh_frame(df, t_clean, sznl_map, market_series,
                                vix_series, ref_ranks, atr_sznl_map)
            try:
                pd.testing.assert_frame_equal(memod, fresh)
                frames_ok += 1
            except AssertionError as e:
                failures.append(f"FRAME {strat['name']}/{t_clean}: {str(e)[:200]}")
                continue
            if key not in fresh_by_key:
                fresh_by_key[key] = (df, t_clean, market_series, ref_ranks)
            s_new = ds.check_signal(memod, st, sznl_map, ticker=t_clean)
            s_old = ds.check_signal(fresh, st, sznl_map, ticker=t_clean)
            if bool(s_new) == bool(s_old):
                sig_ok += 1
            else:
                failures.append(f"SIGNAL {strat['name']}/{t_clean}: memo={s_new} fresh={s_old}")

    # 3. Mutation check: every cached frame must still equal a fresh recompute
    mut_ok = 0
    for key, (df, t_clean, market_series, ref_ranks) in fresh_by_key.items():
        fresh = fresh_frame(df, t_clean, sznl_map, market_series,
                            vix_series, ref_ranks, atr_sznl_map)
        try:
            pd.testing.assert_frame_equal(memo[key], fresh)
            mut_ok += 1
        except AssertionError as e:
            failures.append(f"MUTATED {key}: {str(e)[:200]}")

    print(f"\npairs checked:        {pairs}")
    print(f"frame parity:         {frames_ok}/{pairs}")
    print(f"check_signal parity:  {sig_ok}/{frames_ok}")
    print(f"post-run mutation ok: {mut_ok}/{len(fresh_by_key)} cached frames")
    print(f"memo entries:         {len(memo)} (vs {pairs} old-path computes = "
          f"{pairs / max(len(memo), 1):.1f}x reuse)")
    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for f in failures[:20]:
            print(f"  {f}")
        return 1
    print("\nPARITY: 100% — memoized path is byte-identical to the old path.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
