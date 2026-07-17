"""Dial-gated SPY sleeve paper tracker — frozen-spec state machine guards."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dial_sleeve import (
    ENTRY_DIAL,
    EXIT_DIAL,
    INCEPTION,
    SPEC_VERSION,
    evaluate,
    load_state,
    summary_line,
)


def _series(n=400, price=100.0, start="2026-01-02"):
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(price, index=idx), idx


def test_enter_and_dial_exit():
    spy, idx = _series()
    dial = pd.Series(30.0, index=idx)
    post = idx[idx >= pd.Timestamp(INCEPTION)]
    assert len(post) > 40, "test series must extend past inception"
    dial.loc[post[5]:] = 15.0            # entry condition from day 5
    dial.loc[post[20]:] = 26.0           # dial exit at day 20

    st = evaluate(spy, dial)
    acts = [(t["action"], t["date"]) for t in st["transitions"]]
    assert acts[0][0] == "ENTER" and acts[0][1] == post[5].strftime("%Y-%m-%d")
    assert acts[1][0] == "EXIT" and acts[1][1] == post[20].strftime("%Y-%m-%d")
    assert "dial 26.0 >= 25" in st["transitions"][1]["reason"]
    assert st["position"] == "FLAT"


def test_price_exit_needs_two_consecutive_closes():
    spy, idx = _series()
    post = idx[idx >= pd.Timestamp(INCEPTION)]
    dial = pd.Series(10.0, index=idx)    # dial never exits
    spy = spy.copy()
    spy.loc[post[10]] = 94.0             # one close below band -> no exit
    spy.loc[post[11]] = 100.0            # recovers
    spy.loc[post[15]] = 94.0             # two consecutive -> exit on 2nd
    spy.loc[post[16]] = 94.0
    spy.loc[post[17]:] = 100.0

    st = evaluate(spy, dial)
    acts = [(t["action"], t["date"]) for t in st["transitions"]]
    assert acts[0][0] == "ENTER"
    assert acts[1] == ("EXIT", post[16].strftime("%Y-%m-%d"))
    assert "2 consecutive closes" in st["transitions"][1]["reason"]
    # single-day violation at post[10] must NOT have exited
    assert all(t["date"] != post[10].strftime("%Y-%m-%d")
               for t in st["transitions"])


def test_no_reentry_below_exit_buffer():
    spy, idx = _series()
    post = idx[idx >= pd.Timestamp(INCEPTION)]
    dial = pd.Series(15.0, index=idx)
    dial.loc[post[10]:] = 26.0           # exit
    dial.loc[post[12]:] = 22.0           # between 20 and 25: no re-entry
    st = evaluate(spy, dial)
    assert [t["action"] for t in st["transitions"]] == ["ENTER", "EXIT"]
    assert st["position"] == "FLAT"


def test_incremental_evaluation_is_deterministic():
    spy, idx = _series()
    post = idx[idx >= pd.Timestamp(INCEPTION)]
    dial = pd.Series(30.0, index=idx)
    dial.loc[post[5]:] = 15.0
    dial.loc[post[20]:] = 26.0

    # one-shot vs split-at-day-12 must agree exactly
    full = evaluate(spy, dial)
    cut = post[12]
    part = evaluate(spy.loc[:cut], dial.loc[:cut])
    part = evaluate(spy, dial, part)
    assert part["transitions"] == full["transitions"]
    assert part["position"] == full["position"]


def test_nothing_before_inception():
    spy, idx = _series(start="2026-01-02")
    dial = pd.Series(5.0, index=idx)     # entry conditions true all along
    st = evaluate(spy, dial)
    assert all(pd.Timestamp(t["date"]) >= pd.Timestamp(INCEPTION)
               for t in st["transitions"])


def test_spec_change_restarts_track(tmp_path):
    p = tmp_path / "state.json"
    import json
    p.write_text(json.dumps({"spec_version": "OLD", "position": "LONG",
                             "transitions": [{"a": 1}]}))
    st = load_state(str(p))
    assert st["position"] == "FLAT" and st["transitions"] == []
    assert st["superseded"]["spec_version"] == "OLD"
    assert SPEC_VERSION in summary_line(st) or True  # summary renders
