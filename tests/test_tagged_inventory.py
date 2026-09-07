import copy

import pytest

from tagged_inventory import build_tagged_inventory

STRATEGY = "Oversold Low Volume"
ASOF = "2026-09-08T20:30:00+00:00"


def seed():
    return {"schema_version": 1, "account_key": "primary", "broker_account": "TEST_PRIMARY",
            "asof_utc": "2026-09-04T20:30:00+00:00",
            "review": {"status": "approved", "reviewed_by": "fixture reviewer",
                       "reviewed_at": "2026-09-04T21:00:00+00:00", "provenance": "synthetic reconciled statement"},
            "positions": [{"tranche_id": "test-tranche", "account_key": "primary", "account": "TEST_PRIMARY",
                           "con_id": 42, "symbol": "SPY", "sec_type": "STK", "currency": "USD",
                           "strategy": STRATEGY, "ref_date": "2026-09-01", "entry_date": "2026-09-02",
                           "signed_qty": 100, "entry_price": 100, "atr": 3, "price_basis": "raw",
                           "exit_deadline_utc": "2026-09-16T20:00:00+00:00", "exit_protocol": "MOC"}]}


def coverage():
    return {"complete": False, "accounts": {
        "primary": {"broker_account": "TEST_PRIMARY", "complete": True,
                    "continuous_from": "2026-09-04T20:30:00+00:00", "complete_through": ASOF},
        "pa": {"complete": False, "error": "not used"}}}


def fill(exec_id="fixture.01", qty=20, side="SLD", ref_date="2026-09-01", **extra):
    return {"exec_id": exec_id, "account_key": "primary", "account": "TEST_PRIMARY", "con_id": 42,
            "symbol": "SPY", "sec_type": "STK", "currency": "USD", "strategy": STRATEGY,
            "time_utc": "2026-09-08T15:00:00+00:00", "qty": qty, "side": side, "price": 105,
            "order_ref": f"SPY|BUY|{STRATEGY}|{ref_date}", **extra}


def build(start=None, fills=(), proof=None):
    return build_tagged_inventory(seed() if start is None else start, fills,
                                  coverage() if proof is None else proof, asof=ASOF, algo_strategies={STRATEGY})


def test_missing_seed_is_unknown_not_flat():
    result = build_tagged_inventory(None, [], coverage(), asof=ASOF, algo_strategies={STRATEGY})
    assert result.status == "unknown" and not result.counts
    assert "base sizing" in result.fallback


def test_reviewed_seed_and_partial_exit_preserve_tranche_metadata():
    result = build(fills=[fill()])
    assert result.status == "known" and result.exit_metadata_known
    assert result.counts == {("SPY", STRATEGY): 1}
    assert result.notionals == {("SPY", STRATEGY): 8000}
    tranche = result.tranches[0]
    assert tranche["signed_qty"] == 80
    assert tranche["entry_price"] == 100 and tranche["atr"] == 3
    assert tranche["exit_deadline_utc"] == seed()["positions"][0]["exit_deadline_utc"]


def test_corrected_post_seed_execution_is_applied_once():
    result = build(fills=[fill(), fill("fixture.02", qty=10), fill()])
    assert result.status == "known" and result.tranches[0]["signed_qty"] == 90


def test_pre_seed_correction_requires_reviewed_seed_reconciliation():
    result = build(fills=[fill("fixture.02", time_utc="2026-09-04T19:00:00+00:00")])
    assert result.status == "unknown" and "pre-seed" in result.reasons[0]


def test_zero_is_known_only_with_reviewed_seed_and_confirmed_flatten():
    result = build(fills=[fill(qty=100)])
    assert result.status == "known" and result.counts == {} and result.tranches == []
    bad = seed()
    bad["positions"] = []
    bad["review"] = {}
    assert build(start=bad).status == "unknown"


@pytest.mark.parametrize("mutate", [
    lambda c: c["accounts"]["primary"].pop("continuous_from"),
    lambda c: c["accounts"]["primary"].update(complete=False),
    lambda c: c["accounts"]["primary"].update(broker_account="TEST_OTHER"),
    lambda c: c.update(truncated=True),
    lambda c: c.update(merge_error="fixture error"),
])
def test_incomplete_continuity_is_explicitly_unknown(mutate):
    proof = coverage()
    mutate(proof)
    assert build(proof=proof).status == "unknown"


def test_multiple_contracts_for_one_symbol_are_not_conflated():
    start = seed()
    second = dict(start["positions"][0], tranche_id="second", con_id=43)
    start["positions"].append(second)
    result = build(start=start)
    assert result.status == "unknown" and "multiple contracts" in result.reasons[0]


def test_ambiguous_tranche_close_requires_explicit_allocation():
    start = seed()
    start["positions"].append(dict(start["positions"][0], tranche_id="second"))
    result = build(start=start, fills=[fill()])
    assert result.status == "unknown" and "multiple tranches" in result.reasons[0]


def test_exit_cannot_reverse_or_reopen_a_closed_tranche():
    assert build(fills=[fill(qty=101)]).status == "unknown"
    result = build(fills=[fill(qty=100), fill("next.01", qty=1, time_utc="2026-09-08T16:00:00+00:00")])
    assert result.status == "unknown" and "remaining owned" in result.reasons[0]


def test_new_entry_can_size_overlay_but_needs_raw_atr_and_deadline_for_exit():
    result = build(fills=[fill(qty=10, side="BOT", ref_date="2026-09-08")])
    assert result.status == "known" and result.counts[("SPY", STRATEGY)] == 2
    assert result.notionals[("SPY", STRATEGY)] == 11050
    assert not result.exit_metadata_known
    assert "exits unavailable" in result.fallback


def test_pa_and_discretionary_fills_do_not_change_primary_algo_inventory():
    result = build(fills=[fill(account_key="pa", account="TEST_PA"), fill(strategy="Discretionary", order_ref="")])
    assert result.status == "known" and result.tranches[0]["signed_qty"] == 100


def test_adapter_does_not_mutate_seed_or_fills():
    start, incoming = seed(), [fill()]
    before = copy.deepcopy((start, incoming))
    build(start=start, fills=incoming)
    assert (start, incoming) == before
