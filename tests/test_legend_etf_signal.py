from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from legend_etf.core import (
    build_futures_rth15,
    evaluate_futures_setup,
    latest_observed_futures_rth_session,
    normalize_minutes,
)

ET = "America/New_York"


def _session(day: str, *, setup: bool, instrument_id: int = 11) -> pd.DataFrame:
    index = pd.date_range(f"{day} 09:30", f"{day} 15:59", freq="1min", tz=ET)
    if not setup:
        close = np.full(len(index), 100.0)
        high = close + 0.2
        low = close - 0.2
    else:
        # Every 15-minute low remains above the seeded EMA.  A final intrabar
        # high makes abs(close-open)/(high-low) exactly 9/12 == 0.75.
        close = np.linspace(110.0, 119.0, len(index))
        high = close.copy()
        low = close.copy()
        high[-1] = 122.0
    frame = pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1.0,
            "instrument_id": instrument_id,
        },
        index=index.tz_convert("UTC"),
    )
    return frame


def _valid_minutes() -> pd.DataFrame:
    prior = _session("2026-08-28", setup=False)
    setup = _session("2026-08-31", setup=True)
    probe_index = pd.date_range(
        "2026-09-01 08:40", periods=6, freq="1min", tz=ET
    ).tz_convert("UTC")
    probe = pd.DataFrame(
        {
            "open": 119.0,
            "high": 119.1,
            "low": 118.9,
            "close": 119.0,
            "volume": 1.0,
            "instrument_id": 11,
        },
        index=probe_index,
    )
    return pd.concat([prior, setup, probe])


def _evaluate(frame: pd.DataFrame):
    return evaluate_futures_setup(
        frame,
        setup_date="2026-08-31",
        entry_date="2026-09-01",
        as_of=pd.Timestamp("2026-09-01 08:45", tz=ET),
    )


def test_exact_trend_boundary_qualifies_and_is_roll_safe():
    result = _evaluate(_valid_minutes())
    assert result.qualifies
    assert result.reason == "qualified"
    assert result.trend_ratio == pytest.approx(0.75)
    assert result.instrument_id == result.entry_instrument_id == 11
    assert result.rth_bar_count == 26


def test_below_trend_boundary_fails():
    frame = _valid_minutes()
    final = pd.Timestamp("2026-08-31 15:59", tz=ET).tz_convert("UTC")
    frame.loc[final, "close"] = 118.999
    frame.loc[final, "open"] = 118.999
    frame.loc[final, "low"] = 118.999
    result = _evaluate(frame)
    assert not result.qualifies
    assert result.reason == "trend_ratio_below_threshold"


def test_equality_with_same_bar_ema_is_a_touch():
    frame = _valid_minutes()
    bars = build_futures_rth15(frame)
    timestamp = pd.Timestamp("2026-08-31 15:00", tz=ET)
    ema = float(bars.loc[timestamp, "ema20"])
    minute = timestamp.tz_convert("UTC")
    frame.loc[minute, "low"] = ema
    result = _evaluate(frame)
    assert not result.qualifies
    assert result.reason == "ema_touch_or_wrong_side"


def test_candle_direction_must_agree_with_ema_side():
    frame = _valid_minutes()
    setup_mask = frame.index.tz_convert(ET).date == pd.Timestamp("2026-08-31").date()
    setup_rows = frame.loc[setup_mask].copy()
    reversed_close = np.linspace(119.0, 110.0, len(setup_rows))
    frame.loc[setup_mask, "open"] = reversed_close
    frame.loc[setup_mask, "close"] = reversed_close
    frame.loc[setup_mask, "low"] = reversed_close
    frame.loc[setup_mask, "high"] = reversed_close
    frame.loc[pd.Timestamp("2026-08-31 09:30", tz=ET).tz_convert("UTC"), "high"] = 122
    result = _evaluate(frame)
    assert not result.qualifies
    assert result.reason == "ema_touch_or_wrong_side"


def test_missing_one_rth_bin_fails():
    frame = _valid_minutes()
    local = frame.index.tz_convert(ET)
    missing = (
        (local.date == pd.Timestamp("2026-08-31").date())
        & (local.hour == 12)
        & (local.minute < 15)
    )
    result = _evaluate(frame.loc[~missing])
    assert not result.qualifies
    assert result.reason == "incomplete_setup_bars"


def test_missing_no_trade_minute_inside_setup_bin_preserves_signal():
    frame = _valid_minutes()
    missing = pd.Timestamp("2026-08-31 12:07", tz=ET).tz_convert("UTC")
    result = _evaluate(frame.drop(index=missing))
    assert result.qualifies


def test_missing_no_trade_minute_inside_prior_bin_preserves_ema_seed():
    frame = _valid_minutes()
    missing = pd.Timestamp("2026-08-28 12:07", tz=ET).tz_convert("UTC")
    result = _evaluate(frame.drop(index=missing))
    assert result.qualifies


def test_insufficient_observed_bar_history_fails_ema_seed():
    frame = _valid_minutes()
    local = frame.index.tz_convert(ET)
    without_prior = frame.loc[
        local.date != pd.Timestamp("2026-08-28").date()
    ]
    result = _evaluate(without_prior)
    assert not result.qualifies
    assert result.reason == "ema_not_ready"


def test_fewer_than_twenty_causal_observed_bars_fails_ema_seed():
    frame = _valid_minutes()
    local = frame.index.tz_convert(ET)
    prior_day = local.date == pd.Timestamp("2026-08-28").date()
    # Keep exactly 18 observed 15-minute bins before the setup session.  The
    # first setup bar is therefore only observation 19 and must not receive an
    # EMA value from future bars.
    keep_prior = prior_day & (local < pd.Timestamp("2026-08-28 14:00", tz=ET))
    frame = frame.loc[~prior_day | keep_prior]
    bars = build_futures_rth15(frame)
    setup_first = pd.Timestamp("2026-08-31 09:30", tz=ET)
    assert bars.loc[setup_first, "ema20"] != bars.loc[setup_first, "ema20"]
    result = _evaluate(frame)
    assert not result.qualifies
    assert result.reason == "ema_not_ready"


def test_missing_older_prior_session_does_not_change_observed_bar_ema():
    frame = pd.concat(
        [
            _session("2026-08-26", setup=False),
            _session("2026-08-27", setup=False),
            _session("2026-08-28", setup=False),
            _session("2026-08-31", setup=True),
            _valid_minutes().loc[
                _valid_minutes().index.tz_convert(ET).date
                == pd.Timestamp("2026-09-01").date()
            ],
        ]
    )
    local = frame.index.tz_convert(ET)
    without_thursday = frame.loc[
        local.date != pd.Timestamp("2026-08-27").date()
    ]
    result = _evaluate(without_thursday)
    assert result.qualifies


def test_missing_immediately_prior_session_does_not_change_observed_bar_ema():
    probe = _valid_minutes().loc[
        _valid_minutes().index.tz_convert(ET).date
        == pd.Timestamp("2026-09-01").date()
    ]
    frame = pd.concat(
        [
            _session("2026-08-26", setup=False),
            _session("2026-08-27", setup=False),
            _session("2026-08-31", setup=True),
            probe,
        ]
    )
    result = _evaluate(frame)
    assert result.qualifies


def test_nonempty_early_close_bins_seed_ema_without_xnys_grid_assumption():
    early_index = pd.date_range(
        "2025-11-28 09:30", "2025-11-28 13:15", freq="1min", tz=ET
    ).tz_convert("UTC")
    early = pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.2,
            "low": 99.8,
            "close": 100.0,
            "volume": 1.0,
            "instrument_id": 11,
        },
        index=early_index,
    )
    probe = _valid_minutes().loc[
        _valid_minutes().index.tz_convert(ET).date
        == pd.Timestamp("2026-09-01").date()
    ].copy()
    probe.index = pd.date_range(
        "2025-12-02 08:40", periods=len(probe), freq="1min", tz=ET
    ).tz_convert("UTC")
    frame = pd.concat(
        [
            _session("2025-11-26", setup=False),
            early,
            _session("2025-12-01", setup=True),
            probe,
        ]
    )
    result = evaluate_futures_setup(
        frame,
        setup_date="2025-12-01",
        entry_date="2025-12-02",
        as_of=pd.Timestamp("2025-12-02 08:45", tz=ET),
    )
    assert result.qualifies


def test_contract_change_between_setup_and_entry_fails():
    frame = _valid_minutes()
    local = frame.index.tz_convert(ET)
    frame.loc[local.date == pd.Timestamp("2026-09-01").date(), "instrument_id"] = 12
    result = _evaluate(frame)
    assert not result.qualifies
    assert result.reason == "roll_crossing"


def test_equity_holiday_intervening_abbreviated_futures_session_blocks_friday():
    friday = _session("2026-01-16", setup=True)
    monday_index = pd.date_range(
        "2026-01-19 09:30", "2026-01-19 12:59", freq="1min", tz=ET
    ).tz_convert("UTC")
    monday = pd.DataFrame(
        {
            "open": 119.0,
            "high": 119.1,
            "low": 118.9,
            "close": 119.0,
            "volume": 1.0,
            "instrument_id": 11,
        },
        index=monday_index,
    )
    probe_index = pd.date_range(
        "2026-01-20 08:40", periods=6, freq="1min", tz=ET
    ).tz_convert("UTC")
    probe = monday.iloc[:6].copy()
    probe.index = probe_index
    frame = pd.concat([friday, monday, probe])
    as_of = pd.Timestamp("2026-01-20 08:45", tz=ET)
    assert latest_observed_futures_rth_session(
        frame, entry_date="2026-01-20", as_of=as_of
    ).isoformat() == "2026-01-19"
    stale_friday = evaluate_futures_setup(
        frame,
        setup_date="2026-01-16",
        entry_date="2026-01-20",
        as_of=as_of,
    )
    assert not stale_friday.qualifies
    assert stale_friday.reason == "intervening_futures_session"
    abbreviated_monday = evaluate_futures_setup(
        frame,
        setup_date="2026-01-19",
        entry_date="2026-01-20",
        as_of=as_of,
    )
    assert not abbreviated_monday.qualifies
    assert abbreviated_monday.reason == "incomplete_setup_session"


def test_current_day_post_asof_data_cannot_change_signal():
    frame = _valid_minutes()
    future = _session("2026-09-01", setup=False, instrument_id=99)
    combined = pd.concat([frame, future]).sort_index()
    result = _evaluate(combined)
    assert result.qualifies
    assert result.entry_instrument_id == 11


def test_normalizer_rejects_duplicates_naive_and_malformed_ohlc():
    frame = _valid_minutes().iloc[:2]
    with pytest.raises(ValueError, match="duplicate"):
        normalize_minutes(pd.concat([frame, frame.iloc[[0]]]), require_instrument=True)
    naive = frame.copy()
    naive.index = naive.index.tz_localize(None)
    with pytest.raises(ValueError, match="naive"):
        normalize_minutes(naive)
    malformed = frame.copy()
    malformed.iloc[0, malformed.columns.get_loc("high")] = 99.9
    with pytest.raises(ValueError, match="malformed"):
        normalize_minutes(malformed)
    nonfinite = frame.copy()
    nonfinite.iloc[0, nonfinite.columns.get_loc("close")] = np.inf
    with pytest.raises(ValueError, match="positive and finite"):
        normalize_minutes(nonfinite)
    nonpositive = frame.copy()
    nonpositive.iloc[0, nonpositive.columns.get_loc("open")] = -1
    with pytest.raises(ValueError, match="positive and finite"):
        normalize_minutes(nonpositive)
    bad_volume = frame.copy()
    bad_volume.iloc[0, bad_volume.columns.get_loc("volume")] = -1
    with pytest.raises(ValueError, match="non-negative and finite"):
        normalize_minutes(bad_volume)
