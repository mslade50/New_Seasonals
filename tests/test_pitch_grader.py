"""Guards for scripts/grade_pitch_journal.py — the replay conventions.

The scoreboard is only worth reading if the replay is fixed and pessimistic in
the same way every time. These tests pin each convention with hand-built bars:
entry fills, day-2 stop arming, stop-before-target on a bar that touches both,
gap-through stop fills with slippage, MOO versus MOC time exits, and the
no-fill case.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pitch_journal as pj  # noqa: E402
from scripts.grade_pitch_journal import (  # noqa: E402
    STOP_GAP_SLIP_BPS,
    STOP_SLIP_BPS,
    build_scoreboard,
    replay_idea,
    replay_leg,
)

DATES = ["2026-08-06", "2026-08-07", "2026-08-10", "2026-08-11", "2026-08-12",
         "2026-08-13"]


def bars(rows):
    """rows = list of (open, high, low, close), one per DATES entry."""
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
    }
    base.update(kw)
    return base


FLAT = [(100, 101, 99, 100)] * 6


# ---------------------------------------------------------------------------
# entries
# ---------------------------------------------------------------------------
def test_moo_fills_at_the_open_and_moc_at_the_close():
    px = bars([(98, 101, 97, 100)] + FLAT[1:])
    assert replay_leg(px, row(Entry_Type="MOO"))["entry_price"] == 98
    assert replay_leg(px, row(Entry_Type="MOC"))["entry_price"] == 100


def test_limit_fills_the_first_touching_session():
    px = bars([(100, 101, 99.6, 100), (100, 101, 99.0, 100)] + FLAT[2:])
    result = replay_leg(px, row(Entry_Type="LIMIT", Limit_Price=99.5,
                                Entry_Expire_Date="2026-08-07"))
    assert result["entry_date"] == "2026-08-07"
    assert result["entry_price"] == 99.5


def test_limit_gapping_through_fills_at_the_open():
    px = bars([(98.0, 99.0, 97.0, 98.5)] + FLAT[1:])
    result = replay_leg(px, row(Entry_Type="LIMIT", Limit_Price=99.5))
    assert result["entry_price"] == 98.0


def test_untouched_limit_books_no_fill():
    px = bars(FLAT)
    result = replay_leg(px, row(Entry_Type="LIMIT", Limit_Price=95.0,
                                Entry_Expire_Date="2026-08-07"))
    assert result["status"] == "no_fill"
    assert result["pnl"] == 0.0


def test_open_anchored_limit_is_priced_off_the_session_open():
    px = bars([(100, 101, 99.0, 100)] + FLAT[1:])
    result = replay_leg(px, row(Entry_Type="LIMIT", Entry_Offset_ATR=-0.5,
                                Limit_Price="", ATR=1.0))
    assert result["entry_price"] == 99.5


# ---------------------------------------------------------------------------
# exits
# ---------------------------------------------------------------------------
def test_stop_is_not_armed_on_the_entry_day():
    # Entry day dives through the stop; day 2 onward never does.
    px = bars([(100, 101, 90, 100)] + FLAT[1:])
    result = replay_leg(px, row(Stop_Price=95.0))
    assert result["exit_kind"] == "time_moc"


def test_stop_fires_from_the_second_session():
    px = bars([FLAT[0], (100, 101, 94, 96)] + FLAT[2:])
    result = replay_leg(px, row(Stop_Price=95.0))
    assert result["exit_kind"] == "stop"
    assert result["exit_date"] == "2026-08-07"
    assert result["exit_price"] == pytest.approx(95.0 * (1 - STOP_SLIP_BPS / 1e4))


def test_a_bar_touching_both_books_the_stop():
    px = bars([FLAT[0], (100, 106, 94, 100)] + FLAT[2:])
    result = replay_leg(px, row(Stop_Price=95.0, Target_Price=105.0))
    assert result["exit_kind"] == "stop"


def test_gap_through_stop_fills_at_the_open_with_extra_slippage():
    px = bars([FLAT[0], (90, 92, 88, 91)] + FLAT[2:])
    result = replay_leg(px, row(Stop_Price=95.0))
    expected = 90.0 * (1 - (STOP_SLIP_BPS + STOP_GAP_SLIP_BPS) / 1e4)
    assert result["exit_kind"] == "stop_gap"
    assert result["exit_price"] == pytest.approx(expected)


def test_target_fills_at_the_target_with_no_slippage():
    px = bars([FLAT[0], (100, 106, 99, 105)] + FLAT[2:])
    result = replay_leg(px, row(Target_Price=105.0))
    assert result["exit_kind"] == "target"
    assert result["exit_price"] == 105.0


def test_short_stop_and_target_flip():
    px = bars([FLAT[0], (100, 106, 99, 105)] + FLAT[2:])
    stopped = replay_leg(px, row(Action="SELL_SHORT", Stop_Price=105.0))
    assert stopped["exit_kind"] == "stop"
    hit = replay_leg(px, row(Action="SELL_SHORT", Target_Price=99.5))
    assert hit["exit_kind"] == "target"
    assert hit["pnl"] > 0


def test_time_exit_moc_uses_the_close_and_moo_the_open():
    px = bars(FLAT[:5] + [(107, 110, 106, 109)])
    assert replay_leg(px, row())["exit_price"] == 109        # exit-day close
    moo = replay_leg(px, row(Time_Exit_Order="MOO"))
    assert moo["exit_price"] == 107                          # exit-day open
    assert moo["exit_kind"] == "time_moo"


def test_moo_time_exit_never_sees_the_final_bars_range():
    # The exit-day range would have hit the stop, but an MOO exit is already
    # out at that session's open.
    px = bars(FLAT[:5] + [(107, 110, 90, 109)])
    assert replay_leg(px, row(Time_Exit_Order="MOO",
                              Stop_Price=95.0))["exit_kind"] == "time_moo"


def test_position_still_open_when_the_exit_date_has_no_bar_yet():
    px = bars(FLAT[:3])
    assert replay_leg(px, row())["status"] == "open"


def test_pnl_and_excursions():
    px = bars([FLAT[0], (100, 104, 98, 103)] + FLAT[2:])
    result = replay_leg(px, row(Target_Price=104.0))
    assert result["pnl"] == pytest.approx((104 - 100) * 100)
    assert result["mfe_atr"] == pytest.approx(4.0)
    assert result["mae_atr"] == pytest.approx(2.0)


def test_multiplier_scales_pnl():
    px = bars([FLAT[0], (100, 104, 99, 103)] + FLAT[2:])
    result = replay_leg(px, row(Target_Price=104.0, Multiplier=1000.0,
                                Quantity=2))
    assert result["pnl"] == pytest.approx((104 - 100) * 2 * 1000)


# ---------------------------------------------------------------------------
# idea level
# ---------------------------------------------------------------------------
def test_idea_r_is_pnl_over_staged_risk():
    px = bars([FLAT[0], (100, 104, 99, 103)] + FLAT[2:])
    idea = {"orders": [row(Target_Price=104.0, Risk_Amt=200.0)]}
    result = replay_idea(idea, {"TEST": px})
    assert result["pnl"] == pytest.approx(400.0)
    assert result["r_multiple"] == pytest.approx(2.0)


def test_idea_nets_its_legs():
    up = bars([FLAT[0], (100, 102, 99, 102)] + FLAT[2:])
    idea = {"orders": [row(Ticker="A", Risk_Amt=100.0),
                       row(Ticker="B", Action="SELL_SHORT", Risk_Amt=100.0)]}
    result = replay_idea(idea, {"A": up, "B": up})
    assert result["pnl"] == pytest.approx(0.0)   # long and short cancel
    assert result["status"] == "closed"


def test_idea_with_one_unfilled_leg_reports_partial():
    px = bars(FLAT)
    idea = {"orders": [
        row(Ticker="A", Risk_Amt=100.0),
        row(Ticker="B", Entry_Type="LIMIT", Limit_Price=50.0,
            Entry_Expire_Date="2026-08-07", Risk_Amt=100.0)]}
    assert replay_idea(idea, {"A": px, "B": px})["status"] == "partial_fill"


def test_idea_is_ungradeable_without_prices():
    assert replay_idea({"orders": [row()]}, {})["status"] == "ungradeable"


# ---------------------------------------------------------------------------
# scoreboard
# ---------------------------------------------------------------------------
def _idea(date, grade, r, approve, axis="relative_value"):
    return {"idea_id": f"{date}-1", "date": date, "grade": grade,
            "novelty_axis": axis, "approve": approve,
            "outcome": {"status": "closed", "r_multiple": r, "pnl": r * 100}}


def test_scoreboard_splits_by_grade_and_by_approval():
    ideas = [_idea("2026-08-03", "A", 1.0, "Y"),
             _idea("2026-08-04", "B", -0.5, ""),
             _idea("2026-08-05", "B", 0.5, "Y"),
             _idea("2026-08-06", "C", -1.0, "N")]
    board = build_scoreboard(ideas, pd.Timestamp("2026-08-06"))
    roll = board["rolling_60d"]
    assert roll["n"] == 4 and roll["graded"] == 4
    assert roll["avg_r"] == pytest.approx(0.0)
    assert roll["hit_rate"] == pytest.approx(50.0)
    assert roll["by_grade"]["B"]["n"] == 2
    gap = roll["approved_vs_declined"]
    assert gap["approved_n"] == 2 and gap["declined_n"] == 2
    assert gap["edge_of_the_filter"] == pytest.approx(1.5)


def test_scoreboard_window_drops_old_ideas():
    ideas = [_idea("2026-01-02", "A", 5.0, "Y"),
             _idea("2026-08-06", "B", 1.0, "Y")]
    roll = build_scoreboard(ideas, pd.Timestamp("2026-08-06"))["rolling_60d"]
    assert roll["n"] == 1
    assert roll["avg_r"] == pytest.approx(1.0)
    assert build_scoreboard(ideas, pd.Timestamp("2026-08-06"))["lifetime"]["n"] == 2


def test_declined_ideas_are_graded_too():
    ideas = [_idea("2026-08-06", "B", 2.0, "")]
    roll = build_scoreboard(ideas, pd.Timestamp("2026-08-06"))["rolling_60d"]
    assert roll["graded"] == 1
    assert not pj.approved(ideas[0])
