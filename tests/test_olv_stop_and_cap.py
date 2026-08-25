"""OLV vol-confirmed stop + per-ticker notional cap (2026-07-20).

Pins the engine model of the package that replaced the resting 1.25 ATR STP,
the ladder, and the sector loss gate:

1. A close through the stop level on QUIET volume (< stop_vol_mult x trailing
   20d median) is HELD — the trade runs to its time exit.
2. A close through the level on ELEVATED volume exits at the NEXT session's
   open with stop slippage (Exit Type 'Stop').
3. An intraday touch through the level that closes back above it never exits
   (the old engine stopped here — the churn this package removes).
4. Target and confirm on the same day -> Target (the resting LMT fills
   intraday, before any close evaluation).
5. A confirm on the final hold day is moot — the TIME exit at that close wins.
6. ticker_notional_cap: stacked same-ticker legs are clipped/skipped to keep
   entry notional under pct_nav x sizing equity; exempt tickers pass through.
7. Live config invariants: stop_mode present, no open-count ladder / sector
   gate, cap NOT GRM-scaled, exempt list covers the traded ETFs.
8. Signal-recency ladder (2026-07-30): rung = prior signal days in the
   trailing window_td sessions (fill-independent), mults [0.5, 0.7, 1.0];
   window expiry resets to rung 1; the earnings size override composes with
   the recency mult instead of clobbering it.
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

import numpy as np
import pandas as pd

from strat_backtester import process_signals_fast, STOP_SLIP_BPS
from strategy_config import STRATEGY_BOOK, GLOBAL_RISK_MULTIPLIER


def _olv_strategy(**exec_over):
    ex = {
        'risk_bps': 35, 'slippage_bps': 2, 'stop_atr': 1.25, 'tgt_atr': 2.5,
        'hold_days': 10, 'use_stop_loss': True, 'use_take_profit': True,
        'fill_window_days': 3,
        'stop_mode': 'vol_confirm_close', 'stop_vol_mult': 1.5,
    }
    ex.update(exec_over)
    return {
        'name': 'Oversold Low Volume',
        'settings': {
            'trade_direction': 'Long',
            'entry_type': 'Limit Order -0.25 ATR (Persistent)',
            'max_one_pos': False,
        },
        'execution': ex,
        'universe_tickers': ['TEST'],
    }


def _frame(closes=None, lows=None, highs=None, vols=None, n=16):
    """Signal on dates[0] (close 100, ATR 2 -> limit 99.5). Day 1 Low 99.0
    fills the persistent limit at 99.5. Stop level 97.0, target 104.5.
    Time exit = entry_idx(1) + 10 = dates[11] close."""
    dates = pd.date_range('2024-01-02', periods=n, freq='B')
    df = pd.DataFrame({
        'Open':  [100.0] * n,
        'High':  [100.5] * n,
        'Low':   [100.0] * n,
        'Close': [100.0] * n,
        'Volume': [1_000_000.0] * n,
    }, index=dates)
    df.loc[dates[1], 'Low'] = 99.0          # limit fill day
    for d, v in (closes or {}).items():
        df.loc[dates[d], 'Close'] = v
    for d, v in (lows or {}).items():
        df.loc[dates[d], 'Low'] = v
    for d, v in (highs or {}).items():
        df.loc[dates[d], 'High'] = v
    for d, v in (vols or {}).items():
        df.loc[dates[d], 'Volume'] = v
    df['ATR'] = 2.0
    return dates, df


def _run(df, strat, equity=100_000, extra_candidates=None):
    dates = df.index
    def _sd(close):
        # np.float64 like prod (dict values are pulled from DataFrames there;
        # the engine's pnl .round(0) relies on numpy scalars propagating)
        close = np.float64(close)
        return {'atr': np.float64(2.0), 'close': close, 'open': 100.0,
                'high': 100.5, 'low': 100.0, 'vol_ratio': 1.0, 'sznl': 50,
                'range_pct': 2.0, 'atr_sznl_5d': 50.0,
                'rank_ret_126d': 50.0, 'rank_ret_252d': 50.0}

    candidates = [(int(dates[0].value), 'TEST', 'TEST', 0, 0)]
    signal_data = {('TEST', 0): _sd(100.0)}
    if extra_candidates:
        for sig_idx in extra_candidates:
            candidates.append((int(dates[sig_idx].value), 'TEST', 'TEST', 0, sig_idx))
            signal_data[('TEST', sig_idx)] = _sd(float(df.iloc[sig_idx]['Close']))
    return process_signals_fast(candidates, signal_data, {'TEST': df},
                                [strat], starting_equity=equity)


def test_quiet_breach_is_held_to_time_exit():
    dates, df = _frame(closes={2: 96.5, 3: 96.8}, lows={2: 96.4, 3: 96.5})
    sig_df = _run(df, _olv_strategy())
    assert len(sig_df) == 1
    row = sig_df.iloc[0]
    assert row['Exit Type'] == 'Time'
    assert row['Exit Date'] == dates[11]
    assert abs(row['Exit Price'] - 100.0) < 1e-9


def test_volume_confirmed_breach_exits_next_open():
    dates, df = _frame(closes={2: 96.5}, lows={2: 96.4},
                       vols={2: 2_000_000.0})
    df.loc[dates[3], 'Open'] = 96.0
    sig_df = _run(df, _olv_strategy())
    row = sig_df.iloc[0]
    assert row['Exit Type'] == 'Stop'
    assert row['Exit Date'] == dates[3]
    assert abs(row['Exit Price'] - 96.0 * (1 - STOP_SLIP_BPS / 1e4)) < 1e-9


def test_intraday_touch_with_recovered_close_never_stops():
    # Low pierces far through the level on huge volume, but the close
    # recovers above it — the old intraday STP booked -1R here.
    dates, df = _frame(lows={2: 95.0}, closes={2: 98.0},
                       vols={2: 5_000_000.0})
    sig_df = _run(df, _olv_strategy())
    assert sig_df.iloc[0]['Exit Type'] == 'Time'


def test_target_beats_same_day_confirm():
    dates, df = _frame(highs={2: 104.6}, closes={2: 96.5}, lows={2: 96.4},
                       vols={2: 5_000_000.0})
    row = _run(df, _olv_strategy()).iloc[0]
    assert row['Exit Type'] == 'Target'
    assert abs(row['Exit Price'] - 104.5) < 1e-9


def test_final_day_confirm_is_moot_time_exit_wins():
    dates, df = _frame(closes={11: 92.0}, lows={11: 91.9},
                       vols={11: 5_000_000.0})
    row = _run(df, _olv_strategy()).iloc[0]
    assert row['Exit Type'] == 'Time'
    assert row['Exit Date'] == dates[11]
    assert abs(row['Exit Price'] - 92.0) < 1e-9


def test_vol_mult_boundary_confirms_at_exactly_1_5x():
    dates, df = _frame(closes={2: 96.5}, lows={2: 96.4},
                       vols={2: 1_500_000.0})   # exactly 1.5x the 1M median
    row = _run(df, _olv_strategy()).iloc[0]
    assert row['Exit Type'] == 'Stop'


def test_notional_cap_clips_and_skips_stacked_legs():
    # equity 100k, cap 20% -> $20k. Each leg: 35 bps = $350 risk / $2.5 dist
    # = 140 shares @ 99.5 = $13,930. Leg 1 fits; leg 2 clipped to the $6,070
    # room (61 shares); leg 3 finds no room -> skipped entirely.
    dates, df = _frame(n=20)
    df.loc[dates[2], 'Low'] = 99.0   # leg-2 fill (signal day 1)
    df.loc[dates[3], 'Low'] = 99.0   # leg-3 fill (signal day 2)
    strat = _olv_strategy(
        ticker_notional_cap={'pct_nav': 0.20, 'exempt': []})
    sig_df = _run(df, strat, extra_candidates=[1, 2])
    assert len(sig_df) == 2, f"third leg should be skipped, got {len(sig_df)}"
    sh = sorted(sig_df['Shares'].tolist(), reverse=True)
    assert sh[0] == 140
    assert sh[1] == 61


def _recency_strategy(**exec_over):
    # risk_bps 40 (not the prod 35) keeps 0.7x off a float floor boundary:
    # 0.7 x $350 = 244.999… floors to 97/98 depending on equity drift, while
    # 0.7 x $400 = 280.000…6 floors stably (the prod scan rounds the same
    # way — the test just avoids asserting on the knife edge).
    exec_over.setdefault('risk_bps', 40)
    return _olv_strategy(
        signal_recency_ladder={'window_td': 21, 'mults': [0.5, 0.7, 1.0]},
        **exec_over)


def test_recency_ladder_grades_consecutive_signals():
    # Signals at idx 0/1/2 -> priors 0/1/2 -> mults 0.5/0.7/1.0.
    # Base: 40 bps of 100k = $400 risk / 2.5 dist = 160 shares.
    dates, df = _frame(n=20)
    df.loc[dates[2], 'Low'] = 99.0
    df.loc[dates[3], 'Low'] = 99.0
    sig_df = _run(df, _recency_strategy(), extra_candidates=[1, 2])
    assert sorted(sig_df['Shares'].tolist()) == [80, 112, 160]


def test_recency_ladder_counts_signals_not_fills():
    # The idx-0 signal NEVER fills (no low touches 99.5 in its T+1..T+3
    # window, days 1-3), but it still counts: the idx-1 signal — filled on
    # day 4, inside ITS window — grades to 0.7x. The open-count ladder
    # would have sized it at 0.5x (nothing open).
    dates, df = _frame(n=20)
    df.loc[dates[1], 'Low'] = 100.0    # undo the default fill day
    df.loc[dates[4], 'Low'] = 99.0     # fills idx-1 only (idx-0 expired T+3)
    sig_df = _run(df, _recency_strategy(), extra_candidates=[1])
    assert len(sig_df) == 1
    assert sig_df.iloc[0]['Shares'] == 112   # 0.7x, not 0.5x


def test_recency_ladder_window_expiry_resets_to_first_rung():
    # Second signal 30 sessions later: outside the 21td window -> 0.5x again.
    # Third signal 20 sessions after the second: inside -> 0.7x.
    dates, df = _frame(n=60)
    df.loc[dates[31], 'Low'] = 99.0
    df.loc[dates[51], 'Low'] = 99.0
    sig_df = _run(df, _recency_strategy(), extra_candidates=[30, 50])
    by_sig = sig_df.sort_values('Date')['Shares'].tolist()
    assert by_sig == [80, 80, 112]


def test_earnings_override_composes_with_recency_mult(monkeypatch):
    # Pre-earnings override (10 bps) x first-iteration recency (0.5x):
    # $100 x 0.5 = $50 risk -> 20 shares. The old flat replace booked 40.
    import strat_backtester as sb
    dates, df = _frame()
    monkeypatch.setattr(
        sb, 'load_earnings_dates_map',
        lambda: {'TEST': np.array(['2024-01-05'], dtype='datetime64[D]')})
    strat = _recency_strategy(
        earnings_size_override={'min_td': -10, 'max_td': 0, 'risk_bps': 10})
    sig_df = _run(df, strat)
    assert len(sig_df) == 1
    assert sig_df.iloc[0]['Shares'] == 20


def test_earnings_override_alone_still_flat_replaces(monkeypatch):
    # No recency ladder on the strategy -> override behaves exactly as before
    # (St OS Sznl path): 10 bps flat = $100 -> 40 shares.
    import strat_backtester as sb
    dates, df = _frame()
    monkeypatch.setattr(
        sb, 'load_earnings_dates_map',
        lambda: {'TEST': np.array(['2024-01-05'], dtype='datetime64[D]')})
    strat = _olv_strategy(
        earnings_size_override={'min_td': -10, 'max_td': 0, 'risk_bps': 10})
    sig_df = _run(df, strat)
    assert sig_df.iloc[0]['Shares'] == 40


def _frag_series(monkeypatch, sb, score):
    grid = pd.date_range('2023-12-01', '2024-04-01', freq='D')
    monkeypatch.setitem(sb._FRAG_SCORE_CACHE, 'series',
                        pd.Series(float(score), index=grid))


def test_frag_band_halves_olv_at_dial_65(monkeypatch):
    # OLV appetite band (2026-08-24): 0.5x at dial >= 65. Base 35 bps on
    # $100k = $350 risk / $2.50 stop = 140 shares -> 70 at dial 89.5,
    # untouched at 64.99.
    import strat_backtester as sb
    dates, df = _frame()
    strat = _olv_strategy(frag_risk_bands=[[65, 999, 0.5]])
    _frag_series(monkeypatch, sb, 89.5)
    assert _run(df, strat).iloc[0]['Shares'] == 70
    _frag_series(monkeypatch, sb, 64.99)
    assert _run(df, strat).iloc[0]['Shares'] == 140


def test_earnings_override_composes_with_frag_band(monkeypatch):
    # Gate-5 composition (2026-08-24): the override replaces the BASE but the
    # band survives — 10 bps x 0.5 recency x 0.5 band = $25 -> 10 shares.
    # Before this the band silently vanished inside the earnings window.
    import strat_backtester as sb
    dates, df = _frame()
    monkeypatch.setattr(
        sb, 'load_earnings_dates_map',
        lambda: {'TEST': np.array(['2024-01-05'], dtype='datetime64[D]')})
    strat = _olv_strategy(
        signal_recency_ladder={'window_td': 21, 'mults': [0.5, 0.7, 1.0]},
        earnings_size_override={'min_td': -10, 'max_td': 0, 'risk_bps': 10},
        frag_risk_bands=[[65, 999, 0.5]])
    _frag_series(monkeypatch, sb, 89.5)
    assert _run(df, strat).iloc[0]['Shares'] == 10
    # below the band the override path is unchanged (10 x 0.5 recency = 20)
    _frag_series(monkeypatch, sb, 30.0)
    assert _run(df, strat).iloc[0]['Shares'] == 20


def test_recency_prior_from_mask_window_semantics():
    # daily_scan's counting helper: trailing window EXCLUDES the last bar
    # (today's own signal) and counts exactly window_td sessions before it.
    from filters import recency_prior_from_mask
    idx = pd.date_range('2024-01-02', periods=30, freq='B')
    mask = pd.Series(False, index=idx)
    mask.iloc[-1] = True                 # today — never counted
    assert recency_prior_from_mask(mask, 21) == 0
    mask.iloc[-2] = True                 # yesterday — in window
    assert recency_prior_from_mask(mask, 21) == 1
    mask.iloc[-22] = True                # exactly window_td back — in window
    assert recency_prior_from_mask(mask, 21) == 2
    assert recency_prior_from_mask(mask, 20) == 1   # -22 falls out at 20td
    short = pd.Series([True, True], index=idx[:2])  # mask shorter than window
    assert recency_prior_from_mask(short, 21) == 1


def test_notional_cap_exempt_ticker_passes_through():
    dates, df = _frame(n=20)
    df.loc[dates[2], 'Low'] = 99.0
    df.loc[dates[3], 'Low'] = 99.0
    strat = _olv_strategy(
        ticker_notional_cap={'pct_nav': 0.20, 'exempt': ['TEST']})
    sig_df = _run(df, strat, extra_candidates=[1, 2])
    assert len(sig_df) == 3
    assert (sig_df['Shares'] == 140).all()


# ---------------------------------------------------------------------------
# daily_scan side: OLV_Exits staging + Portfolio-notional loader (mocked Sheets)
# ---------------------------------------------------------------------------

import gspread


class _FakeWS:
    def __init__(self, rows=None):
        self._rows = rows or []
        self.cleared = False
        self.written = None

    def get_all_values(self):
        return self._rows

    def clear(self):
        self.cleared = True

    def update(self, values=None, **kw):
        self.written = values


class _FakeSheet:
    def __init__(self, tabs):
        self._tabs = tabs

    def worksheet(self, name):
        if name not in self._tabs:
            raise gspread.WorksheetNotFound(name)
        return self._tabs[name]

    def add_worksheet(self, title, rows=None, cols=None):
        ws = _FakeWS([])
        self._tabs[title] = ws
        return ws


class _FakeGC:
    def __init__(self, sheet):
        self._sheet = sheet

    def open(self, name):
        return self._sheet


def _px_frame(close_last, vol_last, n=30):
    dates = pd.date_range('2026-06-01', periods=n, freq='B')
    df = pd.DataFrame({
        'Open': 100.0, 'High': 101.0, 'Low': 99.0,
        'Close': [100.0] * (n - 1) + [close_last],
        'Volume': [1_000_000.0] * (n - 1) + [vol_last],
    }, index=dates)
    return df


def _portfolio_rows(entry_date="2026-06-20"):
    return [
        ["Strategy", "Ticker", "Shares", "Price", "ATR", "Time Stop", "Entry Date"],
        ["Oversold Low Volume", "AAA", "100", "100.0", "2.0", "2026-07-28", entry_date],  # stop 97.5
        ["Oversold Low Volume", "BBB", "50", "100.0", "2.0", "2026-07-29", entry_date],
        ["Overbot Vol Spike", "CCC", "10", "50.0", "1.0", "2026-07-22", entry_date],       # ignored
    ]


def test_stage_olv_exits_confirms_loud_holds_quiet(monkeypatch):
    import daily_scan

    exits_ws = _FakeWS([])
    sheet = _FakeSheet({"Portfolio": _FakeWS(_portfolio_rows()),
                        "OLV_Exits": exits_ws})
    monkeypatch.setattr(daily_scan, "get_google_client",
                        lambda: _FakeGC(sheet))
    master = {
        "AAA": _px_frame(close_last=97.0, vol_last=2_000_000.0),  # confirmed
        "BBB": _px_frame(close_last=97.0, vol_last=1_000_000.0),  # quiet -> held
    }
    daily_scan.stage_olv_vol_confirm_exits(master)

    assert exits_ws.cleared
    assert exits_ws.written is not None
    header, *rows = exits_ws.written
    assert len(rows) == 1, f"only AAA should confirm, got {rows}"
    row = dict(zip(header, rows[0]))
    assert row["Symbol"] == "AAA"
    assert row["Action"] == "SELL"
    assert row["Quantity"] == "100"
    assert row["Strategy_Ref"] == "Oversold Low Volume"
    confirm = pd.Timestamp(row["Confirm_Date"])
    from trading_calendar import TRADING_DAY
    assert pd.Timestamp(row["Execute_On"]) == (confirm + TRADING_DAY).normalize()
    assert row["Time_Exit_Date"] == "2026-07-28"   # per-leg bracket key


def test_stage_olv_exits_always_rewrites_tab_even_when_empty(monkeypatch):
    import daily_scan

    exits_ws = _FakeWS([["Symbol"], ["STALE"]])
    sheet = _FakeSheet({"Portfolio": _FakeWS(_portfolio_rows()),
                        "OLV_Exits": exits_ws})
    monkeypatch.setattr(daily_scan, "get_google_client",
                        lambda: _FakeGC(sheet))
    master = {
        "AAA": _px_frame(close_last=100.0, vol_last=1_000_000.0),  # no breach
        "BBB": _px_frame(close_last=100.0, vol_last=1_000_000.0),
    }
    daily_scan.stage_olv_vol_confirm_exits(master)
    assert exits_ws.cleared, "stale exit rows must be cleared on every run"
    assert len(exits_ws.written) == 1, "header-only write expected"


def test_stage_olv_exits_fails_open_without_client(monkeypatch):
    import daily_scan
    monkeypatch.setattr(daily_scan, "get_google_client", lambda: None)
    warnings = daily_scan.stage_olv_vol_confirm_exits({})   # must not raise
    assert warnings, "silent failure — the outage must surface in the email"


def test_stage_olv_exits_skips_entry_day_close(monkeypatch):
    # Day-2 arming convention: a confirmed breach on the ENTRY day's close
    # must NOT stage an exit (engine's loop starts at entry_idx+1).
    import daily_scan
    last_bar = str(_px_frame(97.0, 2_000_000.0).index[-1].date())
    exits_ws = _FakeWS([])
    sheet = _FakeSheet({"Portfolio": _FakeWS(_portfolio_rows(entry_date=last_bar)),
                        "OLV_Exits": exits_ws})
    monkeypatch.setattr(daily_scan, "get_google_client", lambda: _FakeGC(sheet))
    master = {"AAA": _px_frame(97.0, 2_000_000.0),
              "BBB": _px_frame(97.0, 2_000_000.0)}
    daily_scan.stage_olv_vol_confirm_exits(master)
    assert len(exits_ws.written) == 1, "entry-day confirms must be held (day-2 arming)"


def test_stage_olv_exits_stacked_legs_evaluated_independently(monkeypatch):
    # Two open legs in ONE ticker, different entries -> different stop
    # levels. The same session close confirms the higher-entry leg only;
    # the staged exit must carry THAT leg's Time_Exit_Date and quantity.
    import daily_scan

    rows = [
        ["Strategy", "Ticker", "Shares", "Price", "ATR", "Time Stop", "Entry Date"],
        ["Oversold Low Volume", "AAA", "100", "100.0", "2.0", "2026-07-28", "2026-06-20"],  # stop 97.5
        ["Oversold Low Volume", "AAA", "80", "95.0", "2.0", "2026-07-30", "2026-06-23"],    # stop 92.5
    ]
    exits_ws = _FakeWS([])
    sheet = _FakeSheet({"Portfolio": _FakeWS(rows), "OLV_Exits": exits_ws})
    monkeypatch.setattr(daily_scan, "get_google_client", lambda: _FakeGC(sheet))
    master = {"AAA": _px_frame(close_last=97.0, vol_last=2_000_000.0)}
    daily_scan.stage_olv_vol_confirm_exits(master)

    header, *out = exits_ws.written
    assert len(out) == 1, "only the breached leg may stage an exit"
    row = dict(zip(header, out[0]))
    assert row["Symbol"] == "AAA"
    assert row["Quantity"] == "100"
    assert row["Time_Exit_Date"] == "2026-07-28", \
        "the staged exit must key the CONFIRMED leg's bracket, not the other leg's"


def test_stage_olv_exits_stale_ticker_carries_all_legs(monkeypatch):
    # Carry-forward is PER LEG: a stale symbol with TWO previously staged
    # exits (stacked legs) must re-stage both, not collapse to one.
    import daily_scan

    future = str((pd.Timestamp.now() + pd.Timedelta(days=1)).date())

    def _prior(texit, qty):
        return {"Symbol": "BBB", "Action": "SELL", "Quantity": qty,
                "Strategy_Ref": "Oversold Low Volume", "Confirm_Date": "2026-07-17",
                "Execute_On": future, "Time_Exit_Date": texit,
                "Stop_Level": 97.5, "Confirm_Close": 97.0, "Vol_X_Med20": 2.0,
                "Staged_At": "x"}

    class _PriorWS(_FakeWS):
        def get_all_records(self):
            return [_prior("2026-07-29", 50), _prior("2026-08-03", 30)]

    rows = [
        ["Strategy", "Ticker", "Shares", "Price", "ATR", "Time Stop", "Entry Date"],
        ["Oversold Low Volume", "AAA", "100", "100.0", "2.0", "2026-07-28", "2026-06-20"],
        ["Oversold Low Volume", "BBB", "50", "100.0", "2.0", "2026-07-29", "2026-06-20"],
        ["Oversold Low Volume", "BBB", "30", "100.0", "2.0", "2026-08-03", "2026-06-24"],
    ]
    exits_ws = _PriorWS([])
    sheet = _FakeSheet({"Portfolio": _FakeWS(rows), "OLV_Exits": exits_ws})
    monkeypatch.setattr(daily_scan, "get_google_client", lambda: _FakeGC(sheet))
    fresh = _px_frame(100.0, 1_000_000.0)              # AAA: fresh, no breach
    stale = _px_frame(97.0, 2_000_000.0).iloc[:-1]     # BBB: one bar behind
    daily_scan.stage_olv_vol_confirm_exits({"AAA": fresh, "BBB": stale})
    header, *out = exits_ws.written
    assert len(out) == 2, "both stacked legs' staged exits must carry forward"
    carried = {dict(zip(header, r))["Time_Exit_Date"] for r in out}
    assert carried == {"2026-07-29", "2026-08-03"}


def test_stage_olv_exits_warns_on_ambiguous_leg_keys(monkeypatch):
    # Two open legs sharing (Symbol, Time_Exit_Date) cannot be told apart by
    # downstream bracket matching — the scan must warn (email surface).
    import daily_scan

    rows = [
        ["Strategy", "Ticker", "Shares", "Price", "ATR", "Time Stop", "Entry Date"],
        ["Oversold Low Volume", "AAA", "100", "100.0", "2.0", "2026-07-28", "2026-06-20"],
        ["Oversold Low Volume", "AAA", "80", "99.0", "2.0", "2026-07-28", "2026-06-20"],
    ]
    exits_ws = _FakeWS([])
    sheet = _FakeSheet({"Portfolio": _FakeWS(rows), "OLV_Exits": exits_ws})
    monkeypatch.setattr(daily_scan, "get_google_client", lambda: _FakeGC(sheet))
    warnings = daily_scan.stage_olv_vol_confirm_exits(
        {"AAA": _px_frame(100.0, 1_000_000.0)})
    assert any("share Time_Exit_Date" in w for w in warnings)


def test_stage_olv_exits_stale_ticker_carries_prior_row(monkeypatch):
    # A ticker whose last bar lags the freshest session is NOT re-evaluated,
    # and its previously staged exit (Execute_On >= today) survives the
    # clear+rewrite instead of being wiped by stale data.
    import daily_scan
    future = str((pd.Timestamp.now() + pd.Timedelta(days=1)).date())
    prior = {"Symbol": "BBB", "Action": "SELL", "Quantity": 50,
             "Strategy_Ref": "Oversold Low Volume", "Confirm_Date": "2026-07-17",
             "Execute_On": future, "Time_Exit_Date": "2026-07-29",
             "Stop_Level": 97.5, "Confirm_Close": 97.0, "Vol_X_Med20": 2.0,
             "Staged_At": "x"}

    class _PriorWS(_FakeWS):
        def get_all_records(self):
            return [prior]

    exits_ws = _PriorWS([])
    sheet = _FakeSheet({"Portfolio": _FakeWS(_portfolio_rows()),
                        "OLV_Exits": exits_ws})
    monkeypatch.setattr(daily_scan, "get_google_client", lambda: _FakeGC(sheet))
    fresh = _px_frame(100.0, 1_000_000.0)              # AAA: fresh, no breach
    stale = _px_frame(97.0, 2_000_000.0).iloc[:-1]     # BBB: one bar behind
    warnings = daily_scan.stage_olv_vol_confirm_exits({"AAA": fresh, "BBB": stale})
    header, *rows = exits_ws.written
    assert len(rows) == 1
    assert dict(zip(header, rows[0]))["Symbol"] == "BBB", \
        "stale ticker's prior staged exit must be carried forward"
    assert any("stale" in w for w in warnings)


def test_load_open_position_notionals(monkeypatch):
    import daily_scan

    rows = [
        ["Strategy", "Ticker", "Shares", "Price", "ATR"],
        ["Oversold Low Volume", "AAA", "100", "100.0", "2.0"],
        ["Oversold Low Volume", "AAA", "50", "98.0", "2.0"],
        ["Oversold Low Volume", "BBB", "10", "10.0", "1.0"],
        ["Overbot Vol Spike", "AAA", "999", "100.0", "1.0"],
    ]
    sheet = _FakeSheet({"Portfolio": _FakeWS(rows)})
    monkeypatch.setattr(daily_scan, "get_google_client",
                        lambda: _FakeGC(sheet))
    out = daily_scan.load_open_position_notionals({"Oversold Low Volume"})
    assert out[("AAA", "Oversold Low Volume")] == 100 * 100.0 + 50 * 98.0
    assert out[("BBB", "Oversold Low Volume")] == 100.0
    assert ("AAA", "Overbot Vol Spike") not in out
    assert daily_scan.load_open_position_notionals(set()) == {}


def test_scan_stamps_use_stop_false_for_vol_confirm():
    # The staging row must NOT carry a resting STP leg for stop_mode
    # strategies: Use_Stop=False while stop_atr still defines sizing.
    olv = next(s for s in STRATEGY_BOOK if s['name'] == 'Oversold Low Volume')
    ex = olv['execution']
    use_stop = ex.get('use_stop_loss', False)
    if ex.get('stop_mode') == 'vol_confirm_close':
        use_stop = False
    assert use_stop is False


def test_live_config_invariants():
    olv = next(s for s in STRATEGY_BOOK if s['name'] == 'Oversold Low Volume')
    ex = olv['execution']
    assert ex.get('stop_mode') == 'vol_confirm_close'
    assert ex.get('stop_vol_mult') == 1.5
    assert ex.get('use_stop_loss') is True          # stop_atr still sizes risk
    assert ex.get('stop_atr') == 1.25
    # Signal-recency ladder (2026-07-30): rung = prior signal days in the
    # trailing 21 sessions — 0 prior 0.5x, 1 prior 0.7x, 2+ full. Replaced
    # the one-day-old open-count ladder [0.5, 1, 1]. Mults are NOT
    # GRM-scaled (pure multiplier on the scaled base).
    assert ex.get('signal_recency_ladder') == \
        {'window_td': 21, 'mults': [0.5, 0.7, 1.0]}, \
        "OLV signal-recency ladder (2026-07-30)"
    assert 'ladder_multipliers' not in ex, \
        "open-count ladder replaced by signal_recency_ladder 2026-07-30"
    assert 'sector_loss_gate' not in ex, "sector gate removed 2026-07-20"
    cap = ex.get('ticker_notional_cap')
    assert cap and cap['pct_nav'] == 0.50, "cap must NOT be GRM-scaled"
    for etf in ('USO', 'GDX', 'SLV', 'DBC', 'EWZ', 'KRE', 'ITA', 'OIH', 'CEF', 'GLD'):
        assert etf in cap['exempt'], f"{etf} missing from cap exemption"
    # risk_bps IS GRM-scaled at import
    assert ex['risk_bps'] == 35 * GLOBAL_RISK_MULTIPLIER
    # OLV is the recency ladder's ONLY carrier; the open-count ladder is
    # carrier-less (dormant machinery); nobody carries a sector gate
    for s in STRATEGY_BOOK:
        if s['name'] != 'Oversold Low Volume':
            assert not s['execution'].get('signal_recency_ladder')
        assert not s['execution'].get('ladder_multipliers')
        assert not s['execution'].get('sector_loss_gate')
