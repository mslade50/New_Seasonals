"""Synthetic cycle-attribution regressions for all Event strategies."""
from copy import deepcopy

import pandas as pd
import pytest

from event_sleeve import EVENT_SLEEVE
from sleeve_fills import reconcile_event_fills


def position(trade):
    return {"positions": {trade: {"shares": 100, "entry_date": "2026-09-10",
                                 "exit_on": "2026-09-16", "exit_order_type": "MOO",
                                 "status": "exit_pending"}}}


def execution(trade, action, qty, date, ref_date=None, **extra):
    return {"strategy": trade, "symbol": EVENT_SLEEVE[trade]["ticker"],
            "account": "TEST_PRIMARY", "account_key": "primary", "con_id": 42,
            "ref_action": action, "ref_date": ref_date or date, "session_date": date,
            "side": "BOT" if action in {"BUY", "BUY_TO_COVER"} else "SLD",
            "qty": qty, **extra}


def fills(trade, exit_qty=100, stable=False):
    short = EVENT_SLEEVE[trade]["side"] == "SHORT"
    return pd.DataFrame([
        execution(trade, "SELL_SHORT" if short else "BUY", 100, "2026-09-10"),
        execution(trade, "BUY_TO_COVER" if short else "SELL", exit_qty, "2026-09-16",
                  "2026-09-10" if stable else None),
    ])


@pytest.mark.parametrize("trade", EVENT_SLEEVE)
@pytest.mark.parametrize("stable", [False, True])
def test_completed_exit_matches_legacy_and_stable_dates_for_every_strategy(trade, stable):
    state = position(trade)
    reconcile_event_fills(state, fills(trade, stable=stable), EVENT_SLEEVE)
    assert state["positions"] == {}
    assert f"{trade}|2026-09-10" in state["completed"]


@pytest.mark.parametrize("trade", EVENT_SLEEVE)
def test_partial_exit_preserves_only_remaining_obligation(trade):
    state = position(trade)
    reconcile_event_fills(state, fills(trade, exit_qty=40), EVENT_SLEEVE)
    assert state["positions"][trade]["shares"] == 60


def test_seven_partial_fills_complete_synthetic_cover():
    trade = "T2_FOMC_MIDTERM_SHORT"
    rows = [execution(trade, "SELL_SHORT", 100, "2026-09-10")]
    rows += [execution(trade, "BUY_TO_COVER", qty, "2026-09-16")
             for qty in [5, 40, 28, 8, 1, 16, 2]]
    state = position(trade)
    reconcile_event_fills(state, pd.DataFrame(rows), EVENT_SLEEVE)
    assert state["positions"] == {}


def test_overcover_is_an_error_not_another_exit():
    trade = "T2_FOMC_MIDTERM_SHORT"
    state = position(trade)
    original = deepcopy(state)
    with pytest.raises(RuntimeError, match="reversed"):
        reconcile_event_fills(state, fills(trade, exit_qty=200), EVENT_SLEEVE)
    assert state == original


def test_old_cycles_other_strategies_and_other_accounts_do_not_close_current_trade():
    trade = "V4_POSTOPEX_VOL"
    rows = fills(trade, exit_qty=40).to_dict("records")
    rows += [execution(trade, "BUY", 200, "2026-08-21"),
             execution(trade, "SELL", 200, "2026-08-26"),
             execution("V2_NOVDEC_VOL", "SELL", 100, "2026-09-16"),
             execution(trade, "SELL", 100, "2026-09-16", account_key="pa", account="TEST_PA")]
    state = position(trade)
    reconcile_event_fills(state, pd.DataFrame(rows), EVENT_SLEEVE)
    assert state["positions"][trade]["shares"] == 60


def test_later_entry_makes_legacy_attribution_ambiguous():
    trade = "V4_POSTOPEX_VOL"
    rows = fills(trade).to_dict("records")
    rows.append(execution(trade, "BUY", 100, "2026-10-16"))
    with pytest.raises(RuntimeError, match="later entry"):
        reconcile_event_fills(position(trade), pd.DataFrame(rows), EVENT_SLEEVE)
