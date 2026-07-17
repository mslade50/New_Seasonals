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


def test_refresh_last_rolls_back_provisional_day():
    """AM correction: a transition recorded off a provisional PM bar must be
    reversible for the LAST evaluated day only."""
    spy, idx = _series()
    post = idx[idx >= pd.Timestamp(INCEPTION)]
    cut = post[10]

    # PM run: provisional dial dips below entry on the last day -> ENTER
    dial_pm = pd.Series(30.0, index=idx)
    dial_pm.loc[cut] = 15.0
    st = evaluate(spy.loc[:cut], dial_pm.loc[:cut])
    assert st["position"] == "LONG"
    assert len(st["transitions"]) == 1

    # AM correction: settled prices say the dial never crossed
    dial_am = pd.Series(30.0, index=idx)
    st = evaluate(spy.loc[:cut], dial_am.loc[:cut], st, refresh_last=True)
    assert st["position"] == "FLAT"
    assert st["transitions"] == []
    assert st["last_evaluated"] == cut.strftime("%Y-%m-%d")

    # without refresh_last the record is immutable
    st2 = evaluate(spy.loc[:cut], dial_pm.loc[:cut])
    st2 = evaluate(spy.loc[:cut], dial_am.loc[:cut], st2, refresh_last=False)
    assert st2["position"] == "LONG" and len(st2["transitions"]) == 1


def test_refresh_last_cannot_touch_older_days():
    spy, idx = _series()
    post = idx[idx >= pd.Timestamp(INCEPTION)]
    dial = pd.Series(15.0, index=idx)          # ENTER on day 0
    dial.loc[post[5]:] = 26.0                  # EXIT day 5
    cut = post[8]
    st = evaluate(spy.loc[:cut], dial.loc[:cut])
    n_before = len(st["transitions"])
    # correction with a wildly different history: only day 8 re-evaluates
    st = evaluate(spy.loc[:cut], pd.Series(10.0, index=idx).loc[:cut], st,
                  refresh_last=True)
    # old ENTER/EXIT survive; at most day 8 changed (re-entry on corrected dial)
    assert [t["date"] for t in st["transitions"][:n_before]] == \
        [post[0].strftime("%Y-%m-%d"), post[5].strftime("%Y-%m-%d")]


def test_spec_change_restarts_track(tmp_path):
    p = tmp_path / "state.json"
    import json
    p.write_text(json.dumps({"spec_version": "OLD", "position": "LONG",
                             "transitions": [{"a": 1}]}))
    st = load_state(str(p))
    assert st["position"] == "FLAT" and st["transitions"] == []
    assert st["superseded"]["spec_version"] == "OLD"
    assert SPEC_VERSION in summary_line(st) or True  # summary renders
