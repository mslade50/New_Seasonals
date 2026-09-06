"""Adversarial coverage for the offline expected-flat control boundary."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from expected_flat_control import (
    ControlState,
    Manifest,
    ManifestError,
    ReasonCode,
    evaluate,
    render_html,
    render_json,
    render_markdown,
)

AS_OF = "2026-09-05T21:30:00Z"
SESSION = "2026-09-05"


def _contract(con_id: int | None = 1001, symbol: str = "AAPL") -> dict:
    row = {
        "symbol": symbol,
        "sec_type": "STK",
        "currency": "USD",
        "exchange": "SMART",
        "expiry": "",
    }
    if con_id is not None:
        row["con_id"] = con_id
    return row


def _receipt(source_id: str) -> dict:
    return {
        "source_id": source_id,
        "receipt_id": f"receipt-{source_id}",
        "session_date": SESSION,
        "complete": True,
        "complete_through": AS_OF,
        "zero_events": True,
        "event_ids": [],
        "event_count": 0,
        "source_sequence_first": 1,
        "source_sequence_last": 1,
        "source_sequences": [1],
    }


def _obligation(
    *,
    event_id: str = "olv-aapl-t1-flat",
    revision: int = 1,
    source_sequence: int = 1,
    account: str = "U111",
    signal_id: str = "OLV:AAPL:2026-09-05",
    tranche_id: str = "t1",
    contract: dict | None = None,
    expected_flat_by: str = "2026-09-05T21:15:00Z",
    active: bool = True,
) -> dict:
    return {
        "event_id": event_id,
        "revision": revision,
        "producer_id": "time-stop-producer",
        "source_sequence": source_sequence,
        "recorded_at": f"2026-09-05T21:{revision:02d}:00Z",
        "session_date": SESSION,
        "account": account,
        "signal_id": signal_id,
        "tranche_id": tranche_id,
        "strategy": "OLV",
        "contract": contract or _contract(),
        "expected_flat_by": expected_flat_by,
        "active": active,
        "note": "expected same-session exit",
    }


def _baseline(
    *,
    baseline_id: str = "baseline-1",
    source_sequence: int = 1,
    qty: str = "100",
    effective_at: str = "2026-09-05T20:00:00Z",
    account: str = "U111",
    signal_id: str = "OLV:AAPL:2026-09-05",
    tranche_id: str = "t1",
    contract: dict | None = None,
) -> dict:
    return {
        "baseline_id": baseline_id,
        "source_sequence": source_sequence,
        "effective_at": effective_at,
        "account": account,
        "signal_id": signal_id,
        "tranche_id": tranche_id,
        "contract": contract or _contract(),
        "signed_qty": qty,
    }


def _execution(
    *,
    execution_id: str = "exec-1",
    family: str = "family-1",
    revision: int = 1,
    action: str = "APPLY",
    source_sequence: int = 1,
    qty: str = "-100",
    occurred_at: str = "2026-09-05T21:20:00Z",
    account: str = "U111",
    signal_id: str = "OLV:AAPL:2026-09-05",
    tranche_id: str = "t1",
    contract: dict | None = None,
) -> dict:
    return {
        "execution_id": execution_id,
        "correction_family_id": family,
        "correction_revision": revision,
        "correction_action": action,
        "source_sequence": source_sequence,
        "occurred_at": occurred_at,
        "account": account,
        "signal_id": signal_id,
        "tranche_id": tranche_id,
        "contract": contract or _contract(),
        "signed_qty_delta": qty,
    }


def _aggregate(
    *,
    snapshot_id: str = "snap-1",
    source_sequence: int = 1,
    qty: str = "0",
    observed_at: str = "2026-09-05T21:29:00Z",
    account: str = "U111",
    contract: dict | None = None,
) -> dict:
    return {
        "snapshot_id": snapshot_id,
        "source_sequence": source_sequence,
        "observed_at": observed_at,
        "account": account,
        "contract": contract or _contract(),
        "signed_qty": qty,
    }


def _refresh_receipts(doc: dict) -> dict:
    producer = doc["producer_receipts"][0]
    producer_rows = [
        row
        for row in doc["obligation_revisions"]
        if row["producer_id"] == producer["source_id"]
    ]
    event_ids = sorted({row["event_id"] for row in producer_rows})
    sequences = sorted({row["source_sequence"] for row in producer_rows}) or [1]
    producer.update(
        zero_events=not event_ids,
        event_ids=event_ids,
        event_count=len(event_ids),
        source_sequence_first=min(sequences),
        source_sequence_last=max(sequences),
        source_sequences=list(range(min(sequences), max(sequences) + 1)),
    )

    executions = doc["execution_revisions"]
    execution_ids = sorted({row["execution_id"] for row in executions})
    sequences = sorted({row["source_sequence"] for row in executions}) or [1]
    doc["execution_receipt"].update(
        zero_events=not execution_ids,
        event_ids=execution_ids,
        event_count=len(execution_ids),
        source_sequence_first=min(sequences),
        source_sequence_last=max(sequences),
        source_sequences=list(range(min(sequences), max(sequences) + 1)),
    )

    baselines = doc["position_baselines"]
    baseline_ids = sorted({row["baseline_id"] for row in baselines})
    sequences = sorted({row["source_sequence"] for row in baselines}) or [1]
    doc["baseline_receipt"].update(
        zero_events=not baseline_ids,
        event_ids=baseline_ids,
        event_count=len(baseline_ids),
        source_sequence_first=min(sequences),
        source_sequence_last=max(sequences),
        source_sequences=list(range(min(sequences), max(sequences) + 1)),
    )

    snapshots = doc["aggregate_positions"]
    snapshot_ids = sorted({row["snapshot_id"] for row in snapshots})
    sequences = sorted({row["source_sequence"] for row in snapshots}) or [1]
    doc["position_receipt"].update(
        zero_events=not snapshot_ids,
        event_ids=snapshot_ids,
        event_count=len(snapshot_ids),
        source_sequence_first=min(sequences),
        source_sequence_last=max(sequences),
        source_sequences=list(range(min(sequences), max(sequences) + 1)),
    )
    return doc


def manifest_dict(*, mode: str = "LIVE", authoritative: bool = True) -> dict:
    doc = {
        "schema_version": 1,
        "run": {
            "run_id": "postclose-20260905",
            "run_mode": mode,
            "operationally_authoritative": authoritative,
            "as_of": AS_OF,
            "session_date": SESSION,
            "required_producers": ["time-stop-producer"],
            "max_position_age_seconds": 300,
        },
        "producer_receipts": [_receipt("time-stop-producer")],
        "execution_receipt": _receipt("execution-ledger"),
        "baseline_receipt": _receipt("allocation-baselines"),
        "position_receipt": _receipt("position-snapshot"),
        "obligation_revisions": [_obligation()],
        "position_baselines": [_baseline()],
        "execution_revisions": [_execution()],
        "aggregate_positions": [_aggregate()],
    }
    return _refresh_receipts(doc)


def _evaluate(doc: dict):
    return evaluate(Manifest.from_dict(doc))


def _reason_set(report) -> set[ReasonCode]:
    return {issue.reason for issue in report.issues} | {
        issue.reason for item in report.obligations for issue in item.issues
    }


def test_authoritative_live_clear_requires_full_tranche_and_aggregate_reconciliation():
    report = _evaluate(manifest_dict())
    assert report.state is ControlState.CLEAR
    assert report.operationally_authoritative is True
    assert report.obligations[0].state is ControlState.CLEAR
    assert report.obligations[0].signed_residual_qty == 0
    assert "OPERATIONALLY AUTHORITATIVE — CLEAR" in report.headline


@pytest.mark.parametrize("mode", ["FIXTURE", "SHADOW"])
def test_offline_clean_evidence_can_never_look_operationally_clear(mode: str):
    report = _evaluate(manifest_dict(mode=mode, authoritative=False))
    assert report.state is ControlState.UNKNOWN
    assert ReasonCode.NON_AUTHORITATIVE_RUN in _reason_set(report)
    assert "NON-AUTHORITATIVE" in report.headline
    assert "DO NOT TREAT AS OPERATIONAL CLEARANCE" in render_markdown(report)
    assert "DO NOT TREAT AS OPERATIONAL CLEARANCE" in render_html(report)


def test_non_live_authoritative_claim_is_rejected_by_strict_schema():
    doc = manifest_dict(mode="SHADOW", authoritative=False)
    doc["run"]["operationally_authoritative"] = True
    with pytest.raises(ManifestError, match="only LIVE"):
        Manifest.from_dict(doc)


def test_non_disabled_run_requires_at_least_one_declared_obligation_producer():
    doc = manifest_dict()
    doc["run"]["required_producers"] = []
    doc["producer_receipts"] = []
    with pytest.raises(ManifestError, match="must not be empty"):
        Manifest.from_dict(doc)


def test_disabled_run_is_not_scheduled_even_when_fixture_contains_a_residual():
    doc = manifest_dict(mode="DISABLED", authoritative=False)
    doc["execution_revisions"] = []
    doc["aggregate_positions"][0]["signed_qty"] = "100"
    report = _evaluate(_refresh_receipts(doc))
    assert report.state is ControlState.NOT_SCHEDULED
    assert ReasonCode.RUN_DISABLED in _reason_set(report)


def test_complete_zero_obligation_receipt_proves_not_scheduled_not_clear():
    doc = manifest_dict()
    doc["obligation_revisions"] = []
    doc["position_baselines"] = []
    doc["execution_revisions"] = []
    doc["aggregate_positions"] = []
    report = _evaluate(_refresh_receipts(doc))
    assert report.state is ControlState.NOT_SCHEDULED
    assert ReasonCode.NO_DUE_OBLIGATIONS in _reason_set(report)


def test_missing_zero_obligation_receipt_fails_closed():
    doc = manifest_dict()
    doc["obligation_revisions"] = []
    doc["position_baselines"] = []
    doc["execution_revisions"] = []
    doc["aggregate_positions"] = []
    doc = _refresh_receipts(doc)
    doc["producer_receipts"] = []
    report = _evaluate(doc)
    assert report.state is ControlState.UNKNOWN
    assert ReasonCode.MISSING_REQUIRED_PRODUCER_RECEIPT in _reason_set(report)


def test_long_short_partial_and_both_direction_reversals_are_signed():
    long_doc = manifest_dict()
    long_doc["execution_revisions"][0]["signed_qty_delta"] = "-40"
    long_doc["aggregate_positions"][0]["signed_qty"] = "60"
    long_report = _evaluate(long_doc)
    assert long_report.state is ControlState.ALERT
    assert long_report.obligations[0].signed_residual_qty == 60
    assert ReasonCode.RESIDUAL_POSITION_LONG in _reason_set(long_report)

    short_doc = manifest_dict()
    short_doc["position_baselines"][0]["signed_qty"] = "-100"
    short_doc["execution_revisions"][0]["signed_qty_delta"] = "40"
    short_doc["aggregate_positions"][0]["signed_qty"] = "-60"
    short_report = _evaluate(short_doc)
    assert short_report.obligations[0].signed_residual_qty == -60
    assert ReasonCode.RESIDUAL_POSITION_SHORT in _reason_set(short_report)

    reverse_short = manifest_dict()
    reverse_short["execution_revisions"][0]["signed_qty_delta"] = "-110"
    reverse_short["aggregate_positions"][0]["signed_qty"] = "-10"
    report = _evaluate(reverse_short)
    assert report.obligations[0].signed_residual_qty == -10
    assert ReasonCode.POSITION_REVERSED in _reason_set(report)

    reverse_long = manifest_dict()
    reverse_long["position_baselines"][0]["signed_qty"] = "-100"
    reverse_long["execution_revisions"][0]["signed_qty_delta"] = "110"
    reverse_long["aggregate_positions"][0]["signed_qty"] = "10"
    report = _evaluate(reverse_long)
    assert report.obligations[0].signed_residual_qty == 10
    assert ReasonCode.POSITION_REVERSED in _reason_set(report)


def test_decimal_quantities_are_exact_and_json_floats_are_forbidden():
    doc = manifest_dict()
    doc["position_baselines"][0]["signed_qty"] = "0.3"
    doc["execution_revisions"][0]["signed_qty_delta"] = "-0.1"
    doc["aggregate_positions"][0]["signed_qty"] = "0.2"
    report = _evaluate(doc)
    assert str(report.obligations[0].signed_residual_qty) == "0.2"
    doc["position_baselines"][0]["signed_qty"] = 0.3
    with pytest.raises(ManifestError, match="floats are forbidden"):
        Manifest.from_dict(doc)


def test_exact_duplicate_execution_is_deduped_not_double_counted():
    doc = manifest_dict()
    doc["execution_revisions"].append(copy.deepcopy(doc["execution_revisions"][0]))
    report = _evaluate(_refresh_receipts(doc))
    assert report.state is ControlState.CLEAR
    assert report.obligations[0].signed_residual_qty == 0


def test_conflicting_duplicate_execution_fails_closed():
    doc = manifest_dict()
    duplicate = copy.deepcopy(doc["execution_revisions"][0])
    duplicate["signed_qty_delta"] = "-90"
    doc["execution_revisions"].append(duplicate)
    doc["aggregate_positions"][0]["signed_qty"] = "10"
    report = _evaluate(_refresh_receipts(doc))
    assert report.state is ControlState.UNKNOWN
    assert ReasonCode.CONFLICTING_DUPLICATE_EXECUTION in _reason_set(report)


def test_correction_family_latest_revision_replaces_original_and_void_removes_family():
    amended = manifest_dict()
    amended["execution_revisions"].append(
        _execution(
            execution_id="exec-1-corrected", revision=2, source_sequence=2, qty="-90"
        )
    )
    amended["aggregate_positions"][0]["signed_qty"] = "10"
    report = _evaluate(_refresh_receipts(amended))
    assert report.state is ControlState.ALERT
    assert report.obligations[0].signed_residual_qty == 10

    voided = manifest_dict()
    voided["execution_revisions"].append(
        _execution(
            execution_id="exec-1-void",
            revision=2,
            action="VOID",
            source_sequence=2,
            qty="0",
        )
    )
    voided["aggregate_positions"][0]["signed_qty"] = "100"
    report = _evaluate(_refresh_receipts(voided))
    assert report.state is ControlState.ALERT
    assert report.obligations[0].signed_residual_qty == 100


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("gap", ReasonCode.CORRECTION_REVISION_GAP),
        ("duplicate-revision", ReasonCode.AMBIGUOUS_CORRECTION_FAMILY),
        ("changed-allocation", ReasonCode.AMBIGUOUS_CORRECTION_FAMILY),
    ],
)
def test_ambiguous_correction_families_fail_closed(mutation: str, reason: ReasonCode):
    doc = manifest_dict()
    row = _execution(
        execution_id="exec-correction", revision=2, source_sequence=2, qty="-100"
    )
    if mutation == "gap":
        row["correction_revision"] = 3
    elif mutation == "duplicate-revision":
        row["correction_revision"] = 1
    else:
        row["tranche_id"] = "t2"
    doc["execution_revisions"].append(row)
    report = _evaluate(_refresh_receipts(doc))
    assert report.state is ControlState.UNKNOWN
    assert reason in _reason_set(report)


def test_void_correction_must_be_explicit_zero_and_family_must_begin_with_apply():
    nonzero_void = manifest_dict()
    nonzero_void["execution_revisions"].append(
        _execution(
            execution_id="exec-void",
            revision=2,
            action="VOID",
            source_sequence=2,
            qty="-100",
        )
    )
    report = _evaluate(_refresh_receipts(nonzero_void))
    assert report.state is ControlState.UNKNOWN
    assert ReasonCode.AMBIGUOUS_CORRECTION_FAMILY in _reason_set(report)

    starts_void = manifest_dict()
    starts_void["execution_revisions"][0]["correction_action"] = "VOID"
    starts_void["execution_revisions"][0]["signed_qty_delta"] = "0"
    starts_void["aggregate_positions"][0]["signed_qty"] = "100"
    report = _evaluate(starts_void)
    assert report.state is ControlState.UNKNOWN
    assert ReasonCode.AMBIGUOUS_CORRECTION_FAMILY in _reason_set(report)


def test_baseline_effective_timestamp_prevents_double_counting_earlier_execution():
    doc = manifest_dict()
    doc["position_baselines"][0]["effective_at"] = "2026-09-05T21:25:00Z"
    # The baseline was taken after this fill and already includes its effect.
    doc["position_baselines"][0]["signed_qty"] = "0"
    report = _evaluate(doc)
    assert report.state is ControlState.CLEAR
    assert report.obligations[0].signed_residual_qty == 0


def test_future_or_missing_or_duplicate_baseline_is_unknown():
    future = manifest_dict()
    future["position_baselines"][0]["effective_at"] = "2026-09-05T21:31:00Z"
    assert ReasonCode.FUTURE_BASELINE in _reason_set(_evaluate(future))

    missing = manifest_dict()
    missing["position_baselines"] = []
    report = _evaluate(missing)
    assert report.state is ControlState.UNKNOWN
    assert ReasonCode.MISSING_BASELINE in _reason_set(report)

    duplicate = manifest_dict()
    duplicate["position_baselines"].append(
        copy.deepcopy(duplicate["position_baselines"][0])
    )
    report = _evaluate(duplicate)
    assert report.state is ControlState.UNKNOWN
    assert ReasonCode.DUPLICATE_BASELINE in _reason_set(report)


def test_signal_tranche_account_allocations_stay_separate_while_contract_total_reconciles():
    doc = manifest_dict()
    # A second, not-yet-due tranche shares the account+contract and remains open.
    doc["obligation_revisions"].append(
        _obligation(
            event_id="olv-aapl-t2-flat",
            source_sequence=2,
            signal_id="OLV:AAPL:2026-09-04",
            tranche_id="t2",
            expected_flat_by="2026-09-06T21:15:00Z",
        )
    )
    doc["position_baselines"].append(
        _baseline(
            baseline_id="baseline-2",
            source_sequence=2,
            qty="50",
            signal_id="OLV:AAPL:2026-09-04",
            tranche_id="t2",
        )
    )
    doc["aggregate_positions"][0]["signed_qty"] = "50"
    report = _evaluate(_refresh_receipts(doc))
    assert report.state is ControlState.CLEAR
    assert [item.state for item in report.obligations] == [
        ControlState.CLEAR,
        ControlState.NOT_SCHEDULED,
    ]

    # The same conId in another account is a different control domain.
    other_contract = _contract()
    doc["aggregate_positions"].append(
        _aggregate(
            snapshot_id="snap-other-account",
            source_sequence=2,
            qty="999",
            account="U222",
            contract=other_contract,
        )
    )
    report = _evaluate(_refresh_receipts(doc))
    assert report.state is ControlState.CLEAR


def test_aggregate_invariant_prevents_false_clear_from_omitted_overlap_allocation():
    doc = manifest_dict()
    doc["aggregate_positions"][0]["signed_qty"] = "25"
    report = _evaluate(doc)
    assert report.state is ControlState.UNKNOWN
    assert ReasonCode.AGGREGATE_POSITION_MISMATCH in _reason_set(report)


def test_missing_duplicate_and_stale_aggregate_snapshots_are_unknown():
    missing = manifest_dict()
    missing["aggregate_positions"] = []
    report = _evaluate(_refresh_receipts(missing))
    assert ReasonCode.MISSING_POSITION_SNAPSHOT in _reason_set(report)

    duplicate = manifest_dict()
    duplicate["aggregate_positions"].append(
        _aggregate(snapshot_id="snap-2", source_sequence=2, qty="0")
    )
    report = _evaluate(_refresh_receipts(duplicate))
    assert ReasonCode.DUPLICATE_POSITION_SNAPSHOT in _reason_set(report)

    stale = manifest_dict()
    stale["aggregate_positions"][0]["observed_at"] = "2026-09-05T21:00:00Z"
    report = _evaluate(stale)
    assert ReasonCode.STALE_POSITION_SNAPSHOT in _reason_set(report)


def test_exact_conid_matching_and_unique_normalized_fallback():
    fallback = manifest_dict()
    fallback["obligation_revisions"][0]["contract"] = _contract(None)
    fallback["position_baselines"][0]["contract"] = _contract(None)
    fallback["execution_revisions"][0]["contract"] = _contract(None)
    # The one aggregate conId makes the normalized fallback unique.
    report = _evaluate(fallback)
    assert report.state is ControlState.CLEAR

    ambiguous = copy.deepcopy(fallback)
    ambiguous["aggregate_positions"].append(
        _aggregate(
            snapshot_id="snap-2", source_sequence=2, qty="0", contract=_contract(2002)
        )
    )
    report = _evaluate(_refresh_receipts(ambiguous))
    assert report.state is ControlState.UNKNOWN
    assert ReasonCode.AMBIGUOUS_CONTRACT_FALLBACK in _reason_set(report)

    exact = manifest_dict()
    # Same normalized symbol metadata, different conId, but exact obligation
    # matching remains pinned to conId=1001.
    exact["aggregate_positions"].append(
        _aggregate(
            snapshot_id="snap-2", source_sequence=2, qty="77", contract=_contract(2002)
        )
    )
    report = _evaluate(_refresh_receipts(exact))
    assert report.state is ControlState.CLEAR


def test_obligation_event_identity_is_append_only_and_qty_independent():
    doc = manifest_dict()
    revised = _obligation(
        event_id="olv-aapl-t1-flat",
        revision=2,
        source_sequence=2,
        expected_flat_by="2026-09-05T21:10:00Z",
    )
    doc["obligation_revisions"].append(revised)
    # Quantity changes live in baseline/execution evidence, never in event_id.
    doc["position_baselines"][0]["signed_qty"] = "150"
    doc["execution_revisions"][0]["signed_qty_delta"] = "-150"
    report = _evaluate(_refresh_receipts(doc))
    assert report.state is ControlState.CLEAR
    assert report.obligations[0].event_id == "olv-aapl-t1-flat"
    assert (
        report.obligations[0]
        .expected_flat_by.isoformat()
        .startswith("2026-09-05T21:10")
    )


def test_two_active_event_ids_cannot_claim_one_due_allocation():
    doc = manifest_dict()
    doc["obligation_revisions"].append(
        _obligation(event_id="duplicate-logical-obligation", source_sequence=2)
    )
    report = _evaluate(_refresh_receipts(doc))
    assert report.state is ControlState.UNKNOWN
    assert ReasonCode.DUPLICATE_ACTIVE_OBLIGATION in _reason_set(report)


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("gap", ReasonCode.OBLIGATION_REVISION_GAP),
        ("conflict", ReasonCode.CONFLICTING_DUPLICATE_REVISION),
        ("identity", ReasonCode.IMMUTABLE_IDENTITY_CHANGED),
        ("time", ReasonCode.NON_MONOTONIC_REVISION),
    ],
)
def test_bad_obligation_revision_history_fails_closed(
    mutation: str, reason: ReasonCode
):
    doc = manifest_dict()
    revised = _obligation(event_id="olv-aapl-t1-flat", revision=2, source_sequence=2)
    if mutation == "gap":
        revised["revision"] = 3
    elif mutation == "conflict":
        revised["revision"] = 1
        revised["expected_flat_by"] = "2026-09-05T21:00:00Z"
    elif mutation == "identity":
        revised["signal_id"] = "OLV:AAPL:changed"
    else:
        revised["recorded_at"] = doc["obligation_revisions"][0]["recorded_at"]
    doc["obligation_revisions"].append(revised)
    report = _evaluate(_refresh_receipts(doc))
    assert report.state is ControlState.UNKNOWN
    assert reason in _reason_set(report)


@pytest.mark.parametrize(
    "target,mutation,reason",
    [
        (
            "producer_receipts",
            lambda row: row.update(complete=False),
            ReasonCode.INCOMPLETE_PRODUCER_RECEIPT,
        ),
        (
            "producer_receipts",
            lambda row: row.update(event_count=9),
            ReasonCode.RECEIPT_EVENT_SET_MISMATCH,
        ),
        (
            "producer_receipts",
            lambda row: row.update(source_sequences=[2]),
            ReasonCode.SOURCE_SEQUENCE_INCOMPLETE,
        ),
        (
            "execution_receipt",
            lambda row: row.update(complete=False),
            ReasonCode.INCOMPLETE_EXECUTION_SOURCE,
        ),
        (
            "execution_receipt",
            lambda row: row.update(event_ids=["wrong"]),
            ReasonCode.EXECUTION_EVENT_SET_MISMATCH,
        ),
        (
            "baseline_receipt",
            lambda row: row.update(complete=False),
            ReasonCode.INCOMPLETE_BASELINE_SOURCE,
        ),
        (
            "baseline_receipt",
            lambda row: row.update(event_ids=["wrong"]),
            ReasonCode.BASELINE_EVENT_SET_MISMATCH,
        ),
        (
            "position_receipt",
            lambda row: row.update(complete=False),
            ReasonCode.INCOMPLETE_POSITION_SOURCE,
        ),
        (
            "position_receipt",
            lambda row: row.update(event_ids=["wrong"]),
            ReasonCode.POSITION_EVENT_SET_MISMATCH,
        ),
    ],
)
def test_source_completeness_contracts_fail_closed(target, mutation, reason):
    doc = manifest_dict()
    row = doc[target][0] if target == "producer_receipts" else doc[target]
    mutation(row)
    report = _evaluate(doc)
    assert report.state is ControlState.UNKNOWN
    assert reason in _reason_set(report)


def test_receipt_session_and_freshness_are_enforced():
    doc = manifest_dict()
    doc["producer_receipts"][0]["session_date"] = "2026-09-04"
    report = _evaluate(doc)
    assert ReasonCode.SOURCE_SESSION_MISMATCH in _reason_set(report)

    doc = manifest_dict()
    doc["execution_receipt"]["complete_through"] = "2026-09-05T21:29:59Z"
    report = _evaluate(doc)
    assert ReasonCode.STALE_EXECUTION_SOURCE in _reason_set(report)


def test_position_snapshot_must_follow_every_piece_of_applied_evidence():
    doc = manifest_dict()
    doc["aggregate_positions"][0]["observed_at"] = "2026-09-05T21:19:00Z"
    # It is within the max-age window, but predates the 21:20 execution.
    doc["run"]["max_position_age_seconds"] = 900
    report = _evaluate(doc)
    assert report.state is ControlState.UNKNOWN
    assert ReasonCode.POSITION_SNAPSHOT_PRECEDES_EVIDENCE in _reason_set(report)


def test_future_obligation_or_execution_event_fails_closed():
    obligation = manifest_dict()
    obligation["obligation_revisions"][0]["recorded_at"] = "2026-09-05T21:31:00Z"
    report = _evaluate(obligation)
    assert report.state is ControlState.UNKNOWN
    assert ReasonCode.FUTURE_SOURCE_EVENT in _reason_set(report)

    execution = manifest_dict()
    execution["execution_revisions"][0]["occurred_at"] = "2026-09-05T21:31:00Z"
    execution["aggregate_positions"][0]["signed_qty"] = "100"
    report = _evaluate(execution)
    assert report.state is ControlState.UNKNOWN
    assert ReasonCode.FUTURE_SOURCE_EVENT in _reason_set(report)


def test_distinct_events_cannot_share_one_source_sequence():
    doc = manifest_dict()
    doc["position_baselines"].append(
        _baseline(
            baseline_id="baseline-collision",
            source_sequence=1,
            qty="0",
            signal_id="OTHER:AAPL:2026-09-05",
            tranche_id="t2",
        )
    )
    report = _evaluate(_refresh_receipts(doc))
    assert report.state is ControlState.UNKNOWN
    assert ReasonCode.SOURCE_SEQUENCE_INCOMPLETE in _reason_set(report)


def test_strict_schema_rejects_unknown_fields_and_naive_timestamps():
    doc = manifest_dict()
    doc["run"]["surprise"] = True
    with pytest.raises(ManifestError, match="unknown field"):
        Manifest.from_dict(doc)

    doc = manifest_dict()
    doc["run"]["as_of"] = "2026-09-05T21:30:00"
    with pytest.raises(ManifestError, match="UTC offset"):
        Manifest.from_dict(doc)


def test_json_markdown_and_html_are_deterministic_human_first_and_escaped():
    doc = manifest_dict(mode="FIXTURE", authoritative=False)
    doc["obligation_revisions"][0]["strategy"] = "OLV <review>"
    report = _evaluate(doc)
    assert render_json(report) == render_json(report)
    markdown = render_markdown(report)
    html = render_html(report)
    assert (
        markdown.index("## What changed")
        < markdown.index("## Required action")
        < markdown.index("## Run summary")
    )
    assert html.index("<h2>What changed</h2>") < html.index("<h2>Required action</h2>")
    assert "OLV &lt;review&gt;" in html
    parsed = json.loads(render_json(report))
    assert parsed["state"] == "UNKNOWN"
    assert parsed["operationally_authoritative"] is False


def test_shadow_markdown_matches_reviewed_golden_artifact():
    fixture_dir = ROOT / "tests" / "fixtures" / "expected_flat"
    raw = json.loads((fixture_dir / "shadow_clean.json").read_text(encoding="utf-8"))
    report = _evaluate(raw)
    expected = (fixture_dir / "shadow_clean.golden.md").read_text(encoding="utf-8")
    assert render_markdown(report) == expected


def test_cli_fixture_writes_three_atomic_artifacts_and_returns_unknown(tmp_path: Path):
    manifest_path = tmp_path / "fixture.json"
    manifest_path.write_text(
        json.dumps(manifest_dict(mode="FIXTURE", authoritative=False)), encoding="utf-8"
    )
    output_dir = tmp_path / "out"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_expected_flat_control.py"),
        "--input",
        str(manifest_path),
        "--output-dir",
        str(output_dir),
        "--run-mode",
        "FIXTURE",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 20
    files = sorted(path.suffix for path in output_dir.iterdir())
    assert files == [".html", ".json", ".md"]
    assert not list(output_dir.glob("*.tmp"))
    assert "NON-AUTHORITATIVE" in next(output_dir.glob("*.md")).read_text(
        encoding="utf-8"
    )


def test_cli_requires_mode_match_and_second_live_authority_gate(tmp_path: Path):
    manifest_path = tmp_path / "live.json"
    manifest_path.write_text(json.dumps(manifest_dict()), encoding="utf-8")
    script = str(ROOT / "scripts" / "run_expected_flat_control.py")
    base = [
        sys.executable,
        script,
        "--input",
        str(manifest_path),
        "--output-dir",
        str(tmp_path / "out"),
    ]

    mismatch = subprocess.run(
        base + ["--run-mode", "SHADOW"], capture_output=True, text=True, check=False
    )
    assert mismatch.returncode == 2
    assert "does not match" in mismatch.stderr

    gated = subprocess.run(
        base + ["--run-mode", "LIVE"], capture_output=True, text=True, check=False
    )
    assert gated.returncode == 2
    assert "requires --acknowledge-authoritative-live" in gated.stderr

    allowed = subprocess.run(
        base + ["--run-mode", "LIVE", "--acknowledge-authoritative-live"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert allowed.returncode == 0
    assert "state=CLEAR" in allowed.stdout
