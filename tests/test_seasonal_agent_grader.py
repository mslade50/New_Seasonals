"""Trail replay in scripts/grade_pitch_journal.py (Daily Seasonal exit.trail).

Rows carrying Trail_Arm_ATR / Trail_ATR arm once a bar after the fill reaches
the arm MFE, then trail a stop Trail_ATR behind the best close from the NEXT
bar on. Rows without them (every pitch row) replay exactly as before.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.grade_pitch_journal import STOP_GAP_SLIP_BPS, STOP_SLIP_BPS, replay_leg  # noqa: E402

DATES = ["2026-08-06", "2026-08-07", "2026-08-10", "2026-08-11", "2026-08-12",
         "2026-08-13"]


def bars(rows):
    frame = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"])
    frame.index = pd.DatetimeIndex(DATES[:len(rows)])
    return frame


def row(**kw):
    base = {
        "Ticker": "TEST", "Action": "BUY", "Entry_Type": "MOC",
        "Entry_Offset_ATR": "", "Limit_Price": "", "Quantity": 100,
        "Stop_ATR": "", "Target_ATR": "", "Stop_Price": "", "Target_Price": "",
        "Time_Exit_Date": "2026-08-13", "Time_Exit_Order": "MOC",
        "Entry_Expire_Date": "2026-08-06", "ATR": 1.0, "Multiplier": 1.0,
        "Execute_On": "2026-08-06", "Risk_Amt": 100.0,
        "Trail_Arm_ATR": 1.5, "Trail_ATR": 1.0,
    }
    base.update(kw)
    return base


def test_trail_arms_on_mfe_then_stops_behind_the_best_close():
    out = replay_leg(bars([
        (100, 101, 99, 100),        # fill 100 at the close
        (100, 101.8, 100, 101.5),   # MFE 1.8 ATR: armed, best close 101.5
        (101.4, 101.6, 100.4, 101),  # trail 100.5 touched
        (100, 100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100),
    ]), row())
    assert out["exit_kind"] == "trail"
    assert out["exit_date"] == DATES[2]
    assert out["exit_price"] == pytest.approx(100.5 * (1 - STOP_SLIP_BPS / 1e4), abs=1e-4)


def test_trail_is_not_live_before_it_arms():
    out = replay_leg(bars([
        (100, 101, 99, 100),
        (100, 101.2, 98.5, 100),    # MFE 1.2 < 1.5, deep dip, no exit
        (100, 100.5, 99, 100),
        (100, 100.5, 99, 100), (100, 100.5, 99, 100), (100, 100.5, 99, 100.7),
    ]), row())
    assert out["exit_kind"] == "time_moc"
    assert out["exit_price"] == 100.7


def test_trail_level_uses_only_the_prior_close():
    # Arms and dips on the same bar: the level set by this bar's close cannot
    # stop it out on this bar.
    out = replay_leg(bars([
        (100, 101, 99, 100),
        (100, 102, 100.2, 101.9),   # armed, best 101.9 -> level 100.9 from next bar
        (101.9, 103, 101.5, 102.8),  # best 102.8 -> level 101.8
        (102.5, 102.6, 101.7, 102),  # 101.8 touched
        (100, 100, 100, 100), (100, 100, 100, 100),
    ]), row())
    assert out["exit_kind"] == "trail" and out["exit_date"] == DATES[3]
    assert out["exit_price"] == pytest.approx(101.8 * (1 - STOP_SLIP_BPS / 1e4), abs=1e-4)


def test_trail_gap_fills_at_the_open_with_extra_slippage():
    out = replay_leg(bars([
        (100, 101, 99, 100),
        (100, 102, 100, 102),       # level 101
        (100.2, 100.5, 99.8, 100),  # opens through 101
        (100, 100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100),
    ]), row())
    assert out["exit_kind"] == "trail_gap"
    slip = (STOP_SLIP_BPS + STOP_GAP_SLIP_BPS) / 1e4
    assert out["exit_price"] == pytest.approx(100.2 * (1 - slip), abs=1e-4)


def test_short_trail_mirrors():
    out = replay_leg(bars([
        (100, 101, 99, 100),
        (100, 100, 98.2, 98.5),     # MFE 1.8: armed, best 98.5 -> level 99.5
        (98.6, 99.6, 98.4, 99),     # 99.5 touched from below
        (100, 100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100),
    ]), row(Action="SELL_SHORT"))
    assert out["exit_kind"] == "trail"
    assert out["exit_price"] == pytest.approx(99.5 * (1 + STOP_SLIP_BPS / 1e4), abs=1e-4)
    assert out["pnl"] > 0


def test_fixed_stop_rules_until_the_trail_is_tighter():
    # Fixed stop at 98 (2 ATR). Before arming it is the only stop.
    out = replay_leg(bars([
        (100, 101, 99, 100),
        (100, 100.5, 97.9, 99),
        (100, 100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100),
        (100, 100, 100, 100),
    ]), row(Stop_ATR=2.0))
    assert out["exit_kind"] == "stop"
    # After arming, the trail (100.5) is tighter than the fixed 98 and rules.
    out = replay_leg(bars([
        (100, 101, 99, 100),
        (100, 101.8, 100, 101.5),
        (101.4, 101.6, 100.4, 101),
        (100, 100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100),
    ]), row(Stop_ATR=2.0))
    assert out["exit_kind"] == "trail"


def test_rows_without_trail_columns_replay_as_before():
    plain = row()
    del plain["Trail_Arm_ATR"], plain["Trail_ATR"]
    out = replay_leg(bars([
        (100, 101, 99, 100),
        (100, 101.8, 100, 101.5),
        (101.4, 101.6, 100.4, 101),
        (100, 100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100.3),
    ]), plain)
    assert out["exit_kind"] == "time_moc" and out["exit_price"] == 100.3
