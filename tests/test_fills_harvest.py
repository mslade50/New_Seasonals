"""harvest_fills: schema, upsert, enrichment-preservation and gap detection.

This store is the ONLY durable copy of actual IBKR executions we control --
the broker ring drops rows after 14 days and IBKR's API serves the current
session only. Three failure modes would be silent without these guards: a
merge that shrinks the store, a re-fetch that nulls out a commission we
already had, and a retention hole that reads as a green run.
"""
import datetime as dt
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import harvest_fills as hf


def _row(exec_id="e1", time="2026-09-02T19:59:00+00:00", **kw):
    base = {
        "exec_id": exec_id,
        "time": time,
        "account": "U16584234",
        "account_key": "primary",
        "account_label": "Primary (TWS)",
        "symbol": "LUV",
        "sec_type": "STK",
        "currency": "USD",
        "exchange": "BATS",
        "side": "SLD",
        "qty": 200,
        "price": 38.56,
        "avg_price": 38.561909,
        "cum_qty": 1548,
        "order_id": 2804,
        "perm_id": 131755018,
        "client_id": 99,
        "order_ref": "LUV|BUY|Oversold Low Volume|2026-08-19",
        "commission": 1.198467,
        "realized_pnl": -923.35,
        "expiry": "",
        "expiry_full": "",
        "con_id": 9282,
        "ingested_at": 1788379144001,
    }
    base.update(kw)
    return base


# --- orderRef contract ------------------------------------------------------

def test_parse_order_ref_splits_the_book_contract():
    assert hf.parse_order_ref("LUV|BUY|Oversold Low Volume|2026-08-19") == (
        "LUV", "BUY", "Oversold Low Volume", "2026-08-19")


def test_parse_order_ref_returns_empties_for_untagged_orders():
    # Discretionary fills and pre-2026-07 legs carry no ref; never guess.
    for ref in (None, "", "   ", 42):
        assert hf.parse_order_ref(ref) == ("", "", "", "")


def test_parse_order_ref_pads_short_refs_without_shifting_fields():
    assert hf.parse_order_ref("SPY|BUY") == ("SPY", "BUY", "", "")


# --- normalisation ----------------------------------------------------------

def test_session_date_is_the_eastern_session_not_the_utc_date():
    # 00:30 UTC on the 3rd is 20:30 ET on the 2nd: it belongs to the 2nd.
    df = hf.normalize([_row(time="2026-09-03T00:30:00+00:00")])
    assert df.loc[0, "session_date"] == "2026-09-02"
    # 13:30 UTC is 09:30 ET the same day.
    df = hf.normalize([_row(time="2026-09-02T13:30:00+00:00")])
    assert df.loc[0, "session_date"] == "2026-09-02"


def test_normalize_freezes_the_schema_and_drops_unknown_broker_columns():
    df = hf.normalize([_row(some_new_broker_field="surprise")])
    assert tuple(df.columns) == hf.COLUMNS
    assert "some_new_broker_field" not in df.columns


def test_normalize_parses_the_ref_into_its_own_columns():
    df = hf.normalize([_row()])
    assert df.loc[0, "strategy"] == "Oversold Low Volume"
    assert df.loc[0, "ref_date"] == "2026-08-19"
    assert df.loc[0, "ref_action"] == "BUY"


def test_normalize_requires_exec_id_and_time():
    with pytest.raises(ValueError):
        hf.normalize([{"symbol": "LUV", "time": "2026-09-02T19:59:00+00:00"}])


def test_empty_frame_matches_the_frozen_schema():
    assert tuple(hf.empty_frame().columns) == hf.COLUMNS


# --- merge ------------------------------------------------------------------

def test_merge_upserts_by_exec_id_rather_than_appending():
    first = hf.normalize([_row("a"), _row("b", time="2026-09-02T20:00:00+00:00")])
    again = hf.normalize([_row("b", time="2026-09-02T20:00:00+00:00"), _row("c")])
    merged, stats = hf.merge_fills(first, again)
    assert stats == {"rows_before": 2, "rows_after": 3, "rows_new": 1, "rows_updated": 1}
    assert sorted(merged["exec_id"]) == ["a", "b", "c"]


def test_merge_lets_the_broker_correct_a_stored_row():
    stored = hf.normalize([_row("a", price=38.56)])
    fresh = hf.normalize([_row("a", price=38.99)])
    merged, _ = hf.merge_fills(stored, fresh)
    assert merged.loc[0, "price"] == pytest.approx(38.99)


def test_merge_keeps_a_commission_a_later_fetch_arrives_without():
    # The lag runs both ways: the DO can serve the execution again before the
    # commission report is re-attached. Enrichment we already hold must survive.
    stored = hf.normalize([_row("a", commission=1.19, realized_pnl=-923.35)])
    fresh = hf.normalize([_row("a", commission=None, realized_pnl=None)])
    merged, _ = hf.merge_fills(stored, fresh)
    assert merged.loc[0, "commission"] == pytest.approx(1.19)
    assert merged.loc[0, "realized_pnl"] == pytest.approx(-923.35)


def test_merge_takes_a_commission_that_arrives_late():
    stored = hf.normalize([_row("a", commission=None, realized_pnl=None)])
    fresh = hf.normalize([_row("a", commission=2.5, realized_pnl=10.0)])
    merged, _ = hf.merge_fills(stored, fresh)
    assert merged.loc[0, "commission"] == pytest.approx(2.5)


def test_an_empty_ring_never_shrinks_the_store():
    # A no-trade fortnight, or a broker that returns nothing, is not a reason
    # to lose history.
    stored = hf.normalize([_row("a"), _row("b"), _row("c")])
    merged, stats = hf.merge_fills(stored, hf.empty_frame())
    assert stats["rows_after"] == 3 and len(merged) == 3


def test_merge_never_drops_an_execution_it_already_held():
    # The invariant is set containment: whatever ids we stored must survive,
    # whether or not new rows arrive alongside them.
    stored = hf.normalize([_row("a"), _row("b"), _row("c")])
    merged, _ = hf.merge_fills(stored, hf.normalize([_row("d")]))
    assert set(merged["exec_id"]) == {"a", "b", "c", "d"}


def test_merge_raises_if_the_combine_step_ever_loses_a_row(monkeypatch):
    # Set containment is unreachable by construction today (concat is a
    # union), so this proves the assertion is live and would catch a future
    # rewrite of the merge that quietly dropped history.
    stored = hf.normalize([_row("a"), _row("b")])
    real_concat = hf.pd.concat
    monkeypatch.setattr(hf.pd, "concat",
                        lambda parts, **kw: real_concat(parts, **kw).iloc[1:])
    with pytest.raises(ValueError, match="drop"):
        hf.merge_fills(stored, hf.normalize([_row("c")]))


def test_a_store_holding_a_duplicate_exec_id_is_healed_not_duplicated():
    # Two rows for one execution is a corrupt store; the merge keys it rather
    # than carrying the duplicate forward.
    stored = hf.normalize([_row("a"), _row("a"), _row("b")])
    merged, _ = hf.merge_fills(stored, hf.empty_frame())
    assert len(merged) == 2 and set(merged["exec_id"]) == {"a", "b"}


def test_merge_from_nothing_is_a_clean_first_run():
    fresh = hf.normalize([_row("a")])
    merged, stats = hf.merge_fills(hf.empty_frame(), fresh)
    assert stats["rows_before"] == 0 and stats["rows_new"] == 1
    assert len(merged) == 1


# --- gap detection ----------------------------------------------------------

def test_no_gap_when_the_ring_overlaps_what_we_hold():
    stored = hf.normalize([_row("a", time="2026-09-02T19:59:00+00:00")])
    ring = hf.normalize([_row("b", time="2026-08-20T19:59:00+00:00")])
    assert hf.detect_gap(stored, ring)["gap"] is False


def test_no_gap_when_the_windows_merely_touch():
    stored = hf.normalize([_row("a", time="2026-09-01T19:59:00+00:00")])
    ring = hf.normalize([_row("b", time="2026-09-02T19:59:00+00:00")])
    assert hf.detect_gap(stored, ring)["gap"] is False


def test_gap_when_rows_aged_out_between_harvests():
    stored = hf.normalize([_row("a", time="2026-08-03T19:59:00+00:00")])
    ring = hf.normalize([_row("b", time="2026-08-20T19:59:00+00:00")])
    info = hf.detect_gap(stored, ring)
    assert info["gap"] is True
    assert info["missing_business_days"] == len(
        pd.bdate_range(dt.date(2026, 8, 4), dt.date(2026, 8, 19)))


def test_first_run_is_never_a_gap():
    ring = hf.normalize([_row("b", time="2026-08-20T19:59:00+00:00")])
    assert hf.detect_gap(hf.empty_frame(), ring)["gap"] is False
