"""Regression tests for the shared seasonal-idea simulator
(scripts/seasonal_ticket_sim.py), used by BOTH the forward scorer and the
walk-forward backtest. Locks the conventions: T+1-open entry, planned-risk R
denominator, stop-before-target tie-break, time-stop at the window's last close,
and the maturity guard that defers an unfinished window.
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.seasonal_ticket_sim import parse_ticket, simulate_ticket, _stop_fill


def _bars(rows, start="2020-01-02"):
    idx = pd.bdate_range(start, periods=len(rows))
    return pd.DataFrame(rows, index=idx, columns=["Open", "High", "Low", "Close"])


def test_parse_ticket_long_and_short():
    long_c = {"ticker": "aapl", "direction": "long", "horizon": "5d", "channel": "X",
              "evidence": {"TICKET": "BUY ~100.00 | stop 98.00 (1.0 ATR) | target 104.00 | time-stop 5td | R/R 2.0"}}
    p = parse_ticket(long_c)
    assert p["direction"] == "long" and p["ticker"] == "AAPL"
    assert (p["entry"], p["stop"], p["target"], p["time_stop_days"]) == (100.0, 98.0, 104.0, 5)

    short_c = {"ticker": "NG=F", "direction": "short", "channel": "X",
               "evidence": {"TICKET": "SELL ~3.15 | stop 3.29 (1.0 ATR) | target 2.86 | time-stop 21td | R/R 2.0"}}
    s = parse_ticket(short_c)
    assert s["direction"] == "short" and s["time_stop_days"] == 21

    assert parse_ticket({"ticker": "X", "evidence": {}}) is None  # context row, no ticket


def test_long_hits_target():
    # asof = bar0; enter T+1 open (101). target 104 touched on the 3rd fwd bar.
    px = _bars([
        [100, 100, 100, 100],   # asof bar (close 100)
        [101, 102, 100, 101],   # T+1 entry @101
        [101, 103, 100, 102],
        [102, 105, 101, 104],   # high 105 >= target 104
        [104, 106, 103, 105],
        [105, 106, 104, 105],
    ])
    tk = {"direction": "long", "entry": 100.0, "stop": 98.0, "target": 104.0, "time_stop_days": 5}
    out = simulate_ticket(tk, px, px.index[0])
    assert out["exit_type"] == "Target"
    # R vs PLANNED risk |100-98|=2: (104-101)/2 = 1.5
    assert abs(out["R"] - 1.5) < 1e-6
    assert out["entry_price"] == 101.0


def test_long_hits_stop_before_target_same_bar():
    # bar touches BOTH stop and target; stop must win (conservative)
    px = _bars([
        [100, 100, 100, 100],
        [101, 104, 97, 100],    # T+1: high>=target104 AND low<=stop98 -> Stop
        [100, 101, 99, 100],
        [100, 101, 99, 100],
        [100, 101, 99, 100],
        [100, 101, 99, 100],
    ])
    tk = {"direction": "long", "entry": 100.0, "stop": 98.0, "target": 104.0, "time_stop_days": 5}
    out = simulate_ticket(tk, px, px.index[0])
    assert out["exit_type"] == "Stop"
    # open (101) didn't gap through the stop -> fill at the stop minus 3 bps slippage
    fill = _stop_fill("long", 98.0, 101.0)
    assert abs(out["exit_price"] - fill) < 1e-6
    assert abs(out["R"] - ((fill - 101.0) / 2.0)) < 1e-6


def test_long_stop_gap_through():
    # T+2 bar OPENS below the stop (gap-down through it) -> fill at the open
    # (+ 13 bps), worse than the stop, so the loss exceeds the planned -1.5R.
    px = _bars([
        [100, 100, 100, 100],   # asof
        [101, 102, 100, 101],   # T+1 entry @101
        [95, 96, 94, 95],       # T+2: open 95 < stop 98 -> gap-through
        [95, 96, 94, 95],
        [95, 96, 94, 95],
        [95, 96, 94, 95],
    ])
    tk = {"direction": "long", "entry": 100.0, "stop": 98.0, "target": 104.0, "time_stop_days": 5}
    out = simulate_ticket(tk, px, px.index[0])
    assert out["exit_type"] == "Stop"
    fill = _stop_fill("long", 98.0, 95.0)  # gapped: 95 * (1 - 13/1e4)
    assert abs(out["exit_price"] - fill) < 1e-6
    assert out["exit_price"] < 98.0                      # worse than the stop
    assert out["R"] < (98.0 - 101.0) / 2.0               # loss bigger than -1.5R


def test_time_exit_at_window_last_close():
    px = _bars([[100, 100, 100, 100]] + [[100, 101, 99, 100.5]] * 6)
    tk = {"direction": "long", "entry": 100.0, "stop": 90.0, "target": 110.0, "time_stop_days": 3}
    out = simulate_ticket(tk, px, px.index[0])
    assert out["exit_type"] == "Time"
    # window = 3 bars after asof; exit at the 3rd bar's close (100.5)
    assert out["exit_price"] == 100.5
    # entry = T+1 open = 100 here; R vs planned risk |100-90|=10
    assert abs(out["R"] - ((100.5 - 100.0) / 10.0)) < 1e-6


def test_maturity_guard_defers_unfinished_window():
    # only 2 bars after asof but window needs 5 and no stop/target hit -> None (live)
    px = _bars([[100, 100, 100, 100], [100, 101, 99, 100], [100, 101, 99, 100]])
    tk = {"direction": "long", "entry": 100.0, "stop": 90.0, "target": 110.0, "time_stop_days": 5}
    assert simulate_ticket(tk, px, px.index[0]) is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All seasonal-sim tests passed.")
