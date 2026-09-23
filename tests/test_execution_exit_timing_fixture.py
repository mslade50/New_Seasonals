"""Reviewed-fixture guard for the entry_bracket exit-timing fields (2026-09-23).

stop_arm "fill"|"next_session" and time_stop_at "close"|"open"; absent fields
must build the same orders as before. Mirrors test_exit_timing_fields.py in
OneDrive trading_ibkr against the committed executor fixture.
"""
import ast
from pathlib import Path

import pytest
from ib_insync import LimitOrder, MarketOrder, StopLimitOrder, StopOrder

CORE = Path(__file__).parent / "fixtures" / "execution_runtime" / "execute_order_core.py"
NAMES = {"build_bracket", "next_session_gat", "_exit_timing"}


@pytest.fixture(scope="module")
def ns() -> dict:
    tree = ast.parse(CORE.read_text(encoding="utf-8"))
    body = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in NAMES)
            or (isinstance(n, ast.Assign) and any(
                getattr(t, "id", "") in {"STOP_ARM_VALUES", "TIME_STOP_AT_CLOCK"} for t in n.targets))]
    env = dict(LimitOrder=LimitOrder, MarketOrder=MarketOrder,
               StopLimitOrder=StopLimitOrder, StopOrder=StopOrder)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(CORE), "exec"), env)
    assert NAMES <= set(env)
    return env


def _stop(children):
    return [c for c in children if c.orderType == "STP"][0]


def test_default_stop_has_no_gat_and_time_leg_keeps_its_gat(ns):
    ids = iter(range(1, 99)).__next__
    _, children = ns["build_bracket"]("BUY", 10, 100.0, 95.0, 110.0, None, ids,
                                      "20990105 15:59:00 US/Eastern")
    assert _stop(children).goodAfterTime == ""
    assert children[-1].goodAfterTime == "20990105 15:59:00 US/Eastern"


def test_stop_gat_is_set_only_on_the_stop(ns):
    ids = iter(range(1, 99)).__next__
    _, children = ns["build_bracket"]("BUY", 10, 100.0, 95.0, 110.0, None, ids,
                                      stop_gat="20990102 09:30:00")
    assert _stop(children).goodAfterTime == "20990102 09:30:00"
    assert all(c.goodAfterTime == "" for c in children if c.orderType != "STP")


def test_exit_timing_defaults_and_values(ns):
    timing = ns["_exit_timing"]
    assert timing({}) == (None, "15:59:00")
    assert timing({"stop_arm": "fill", "time_stop_at": "close"}) == (None, "15:59:00")
    assert timing({"time_stop_at": "open"}) == (None, "09:30:00")
    gat, _ = timing({"stop_arm": "next_session"})
    assert gat.endswith(" 09:30:00") and len(gat.split(" ")[0]) == 8


@pytest.mark.parametrize("payload", [{"stop_arm": "tomorrow"}, {"time_stop_at": "moc"},
                                     {"stop_arm": "Fill"}])
def test_exit_timing_rejects_unknown_values(ns, payload):
    with pytest.raises(ValueError):
        ns["_exit_timing"](payload)
