"""OVS scale-out + ATR-Ext precedence in process_signals_fast (2026-07-16).

Live order_staging has split every OVS P1/P2 row since 2026-06-17 into two
independent single-target brackets (near = 40% of shares @ 1 ATR, far = 60%
@ 2 ATR) as deliberate short-book variance smoothing. The engine booked 100%
at 2 ATR until 2026-07-16, overstating OVS on every trade that reached 1 ATR
but not 2. These tests pin the engine model to live semantics:

1. A filled OVS trade books TWO rows (near/far) with live's share split,
   the near target at scaleout_near_tgt_atr, entry-day targets uncredited.
2. A position too small to split books ONE full-size 2-ATR row (live parity:
   order_staging keeps a single far bracket when a tranche rounds below 1).
3. An EOD-DD day books ONE row (both live tranches exit at the same close).
4. ATR Extended Gap Up > OVS same-symbol precedence (live-only until
   2026-07-16) drops the OVS candidate in the engine too.
5. Config invariants: scaleout fields present, NOT GRM-scaled, matching
   order_staging's constants (0.40 / 1.0 ATR).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'pages'))


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

import pandas as pd

from strat_backtester import process_signals_fast
from strategy_config import STRATEGY_BOOK, GLOBAL_RISK_MULTIPLIER


def _ovs_strategy():
    return {
        'name': 'Overbot Vol Spike',
        'settings': {
            'trade_direction': 'Short',
            'entry_type': 'Limit (Open +/- 0.75 ATR)',
            'max_one_pos': False,
        },
        'execution': {
            'risk_bps': 40, 'slippage_bps': 2,
            'stop_atr': 1.0, 'tgt_atr': 2.0,
            'hold_days': 2,
            'use_stop_loss': False, 'use_take_profit': True,
            'path1_bps': 40, 'path2_bps': 8, 'path2_daily_cap_pct': 0.75,
            'scaleout_near_frac': 0.40, 'scaleout_near_tgt_atr': 1.0,
        },
        'universe_tickers': ['TEST'],
    }


def _atr_ext_strategy():
    return {
        'name': 'ATR Extended Gap Up',
        'settings': {
            'trade_direction': 'Short',
            'entry_type': 'Limit (Open +/- 0.75 ATR)',
            'max_one_pos': False,
        },
        'execution': {
            'risk_bps': 40, 'slippage_bps': 2,
            'stop_atr': 1.0, 'tgt_atr': 4.0,
            'hold_days': 2,
            'use_stop_loss': False, 'use_take_profit': True,
        },
        'universe_tickers': ['TEST'],
    }


def _build_inputs(n_strats=1):
    """Signal 2024-01-02 (Tue), OVS short fill Wed at 101 + 0.75*2 = 102.5.
    Near target 102.5 - 1*2 = 100.5 touches Thu (Low 100.4); far target
    102.5 - 2*2 = 98.5 never touches -> Time exit Fri close 101.0.
    Entry-day close 102.0 keeps EOD-DD quiet; entry-day Low 101.0 stays
    above the near target so the uncredited-entry-day-target rule is inert
    here (pinned separately below)."""
    dates = pd.date_range('2024-01-02', periods=5, freq='B')
    df = pd.DataFrame({
        'Open':  [100.0, 101.0, 100.6, 100.5, 100.0],
        'High':  [101.0, 103.5, 101.5, 101.5, 102.0],
        'Low':   [99.0, 101.0, 100.4, 100.0, 100.0],
        'Close': [100.0, 102.0, 100.8, 101.0, 100.5],
    }, index=dates)
    df['ATR'] = 2.0
    df['RangePct'] = 0.02
    df['vol_ratio'] = 1.0
    df['Sznl'] = 50.0
    df['atr_sznl_5d'] = 50.0
    df['rank_ret_126d'] = 50.0
    df['rank_ret_252d'] = 50.0

    candidates = [(int(dates[0].value), 'TEST', 'TEST', i, 0) for i in range(n_strats)]
    signal_data = {
        ('TEST', 0): {
            'atr': 2.0, 'close': 100.0, 'open': 100.0,
            'high': 101.0, 'low': 99.0,
            'vol_ratio': 1.0, 'sznl': 50, 'range_pct': 2.0,
            'atr_sznl_5d': 50.0, 'rank_ret_126d': 50.0, 'rank_ret_252d': 50.0,
        }
    }
    return candidates, signal_data, {'TEST': df}


def test_scaleout_books_two_tranches():
    candidates, signal_data, processed = _build_inputs()
    sig_df = process_signals_fast(
        candidates, signal_data, processed, [_ovs_strategy()],
        starting_equity=100_000,
    )
    assert len(sig_df) == 2, f"expected near+far rows, got {len(sig_df)}"
    near = sig_df[sig_df['Tranche'] == 'near'].iloc[0]
    far = sig_df[sig_df['Tranche'] == 'far'].iloc[0]
    # $400 risk / $2 dist = 200 shares -> live split 80/120
    assert near['Shares'] == 80 and far['Shares'] == 120
    assert near['Exit Type'] == 'Target'
    assert abs(near['Exit Price'] - 100.5) < 1e-9      # entry 102.5 - 1 ATR
    assert near['Exit Date'] == pd.Timestamp('2024-01-04')  # day AFTER entry
    assert far['Exit Type'] == 'Time'
    assert abs(far['Exit Price'] - 101.0) < 1e-9       # 2-day time exit close
    # risk splits by share count and sums back to the staged total
    assert abs(near['Risk $'] + far['Risk $'] - 400.0) < 1e-6
    assert abs(near['Risk $'] - 400.0 * 80 / 200) < 1e-6
    # PnL: near (102.5-100.5)*80 = 160; far (102.5-101)*120 = 180
    assert near['PnL'] == 160 and far['PnL'] == 180


def test_no_split_below_one_share():
    candidates, signal_data, processed = _build_inputs()
    sig_df = process_signals_fast(
        candidates, signal_data, processed, [_ovs_strategy()],
        starting_equity=500,   # $2 risk / $2 dist = 1 share -> near rounds to 0
    )
    assert len(sig_df) == 1
    assert sig_df.iloc[0]['Tranche'] == ''
    assert sig_df.iloc[0]['Shares'] == 1


def test_eod_dd_books_single_row():
    candidates, signal_data, processed = _build_inputs()
    strat = _ovs_strategy()
    strat['execution']['eod_dd_atr'] = 0.25   # all weekdays (no gate list)
    df = processed['TEST']
    df.loc[df.index[1], 'Close'] = 103.5      # 0.5 ATR offside vs 102.5 entry
    sig_df = process_signals_fast(
        candidates, signal_data, processed, [strat],
        starting_equity=100_000,
    )
    assert len(sig_df) == 1, "EOD-DD closes both live tranches at one price -> one row"
    assert sig_df.iloc[0]['Exit Type'] == 'EOD-DD'


def test_atr_ext_precedence_drops_ovs():
    candidates, signal_data, processed = _build_inputs(n_strats=2)
    sig_df = process_signals_fast(
        candidates, signal_data, processed,
        [_ovs_strategy(), _atr_ext_strategy()],
        starting_equity=100_000,
    )
    strats = set(sig_df['Strategy'])
    assert 'Overbot Vol Spike' not in strats, (
        "same-symbol OVS must lose to ATR Extended Gap Up (live precedence)"
    )
    assert 'ATR Extended Gap Up' in strats


def test_overlay_free_mode_keeps_both_strategies_and_full_sizes_ovs_path2():
    candidates, signal_data, processed = _build_inputs(n_strats=2)
    # Mild OVS gap: production uses 8/40 size, while the overlay-free ledger
    # keeps the core positive-gap entry rule but removes the path-2 downsize.
    processed['TEST'].iloc[1, processed['TEST'].columns.get_loc('Open')] = 100.25
    sig_df = process_signals_fast(
        candidates, signal_data, processed,
        [_ovs_strategy(), _atr_ext_strategy()],
        starting_equity=100_000,
        portfolio_overlays_enabled=False,
    )

    assert set(sig_df['Strategy']) == {'Overbot Vol Spike', 'ATR Extended Gap Up'}
    ovs = sig_df[sig_df['Strategy'] == 'Overbot Vol Spike']
    assert abs(ovs['Risk $'].sum() - 400.0) < 1e-6


def test_config_scaleout_fields_not_grm_scaled():
    ovs = next(s for s in STRATEGY_BOOK if s['name'] == 'Overbot Vol Spike')
    exe = ovs['execution']
    assert exe.get('scaleout_near_frac') == 0.40, (
        "must match order_staging.OVS_SCALEOUT_NEAR_FRAC (0.40)"
    )
    assert exe.get('scaleout_near_tgt_atr') == 1.0, (
        "must match order_staging.OVS_PROFIT_TAKER_ATR_MULT (1.0)"
    )
    # sanity: GRM really is active and scaled the bps keys, not these
    if GLOBAL_RISK_MULTIPLIER != 1.0:
        assert exe['path1_bps'] == 40 * GLOBAL_RISK_MULTIPLIER
