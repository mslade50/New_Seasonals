"""Adversarial coverage for the offline expected-flat control boundary."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import expected_flat_control as control
from expected_flat_control import (
    ControlState,
    Manifest,
    ManifestError,
    ReasonCode,
    RunMode,
    evaluate,
    render_html,
    render_json,
    render_markdown,
    write_artifacts,
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
        "source_sequence_first": None,
        "source_sequence_last": None,
        "source_sequences": [],
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
    revision: int = 1,
    source_sequence: int = 1,
    qty: str = "100",
    recorded_at: str = "2026-09-05T20:00:00Z",
    effective_at: str = "2026-09-05T20:00:00Z",
    reason: str = "INITIAL",
    account: str = "U111",
    signal_id: str = "OLV:AAPL:2026-09-05",
    tranche_id: str = "t1",
    contract: dict | None = None,
) -> dict:
    return {
        "baseline_id": baseline_id,
        "revision": revision,
        "source_sequence": source_sequence,
        "recorded_at": recorded_at,
        "effective_at": effective_at,
        "account": account,
        "signal_id": signal_id,
        "tranche_id": tranche_id,
        "contract": contract or _contract(),
        "signed_qty": qty,
        "reason": reason,
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
    event_ids = sorted(
        {f"{row['event_id']}@{row['revision']}" for row in producer_rows}
    )
    sequences = sorted({row["source_sequence"] for row in producer_rows})
    producer.update(
        zero_events=not event_ids,
        event_ids=event_ids,
        event_count=len(event_ids),
        source_sequence_first=min(sequences) if sequences else None,
        source_sequence_last=max(sequences) if sequences else None,
        source_sequences=sequences,
    )

    executions = doc["execution_revisions"]
    execution_ids = sorted(
        {f"{row['execution_id']}@{row['correction_revision']}" for row in executions}
    )
    sequences = sorted({row["source_sequence"] for row in executions})
    doc["execution_receipt"].update(
        zero_events=not execution_ids,
        event_ids=execution_ids,
        event_count=len(execution_ids),
        source_sequence_first=min(sequences) if sequences else None,
        source_sequence_last=max(sequences) if sequences else None,
        source_sequences=sequences,
    )

    baselines = doc["position_baselines"]
    baseline_ids = sorted(
        {f"{row['baseline_id']}@{row['revision']}" for row in baselines}
    )
    sequences = sorted({row["source_sequence"] for row in baselines})
    doc["baseline_receipt"].update(
        zero_events=not baseline_ids,
        event_ids=baseline_ids,
        event_count=len(baseline_ids),
        source_sequence_first=min(sequences) if sequences else None,
        source_sequence_last=max(sequences) if sequences else None,
        source_sequences=sequences,
    )

    snapshots = doc["aggregate_positions"]
    snapshot_ids = sorted({row["snapshot_id"] for row in snapshots})
    sequences = sorted({row["source_sequence"] for row in snapshots})
    doc["position_receipt"].update(
        zero_events=not snapshot_ids,
        event_ids=snapshot_ids,
        event_count=len(snapshot_ids),
        source_sequence_first=min(sequences) if sequences else None,
        source_sequence_last=max(sequences) if sequences else None,
        source_sequences=sequences,
    )
    return doc


def manifest_dict(*, mode: str = "LIVE", authoritative: bool = False) -> dict:
    doc = {
        "schema_version": 2,
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


def _evaluate(doc: dict, *, trusted_live: bool = True):
    manifest = Manifest.from_dict(doc)
    if trusted_live:
        manifest = replace(
            manifest,
            run=replace(
                manifest.run,
                run_mode=RunMode.LIVE,
                operationally_authoritative=True,
            ),
        )
    return evaluate(manifest, input_sha256="1" * 64, code_sha256="2" * 64)


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
    report = _evaluate(
        manifest_dict(mode=mode, authoritative=False), trusted_live=False
    )
    assert report.state is ControlState.UNKNOWN
    assert ReasonCode.NON_AUTHORITATIVE_RUN in _reason_set(report)
    assert "NON-AUTHORITATIVE" in report.headline
    assert "DO NOT TREAT AS OPERATIONAL CLEARANCE" in render_markdown(report)
    assert "DO NOT TREAT AS OPERATIONAL CLEARANCE" in render_html(report)


@pytest.mark.parametrize("mode", ["SHADOW", "LIVE"])
def test_file_manifest_authoritative_claim_is_rejected_by_strict_schema(mode: str):
    doc = manifest_dict(mode=mode, authoritative=False)
    doc["run"]["operationally_authoritative"] = True
    with pytest.raises(ManifestError, match="future trusted adapter"):
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
    report = _evaluate(_refresh_receipts(doc), trusted_live=False)
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


def test_future_or_missing_or_conflicting_duplicate_baseline_is_unknown():
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
    assert report.state is ControlState.CLEAR

    duplicate["position_baselines"][-1]["signed_qty"] = "99"
    report = _evaluate(duplicate)
    assert report.state is ControlState.UNKNOWN
    assert ReasonCode.CONFLICTING_DUPLICATE_BASELINE_REVISION in _reason_set(report)


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

    # The same conId in another account is a different control domain, but an
    # unattributed position in that domain must still block the whole run.
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
    assert report.state is ControlState.UNKNOWN
    assert ReasonCode.AGGREGATE_POSITION_MISMATCH in _reason_set(report)


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
    # Same normalized metadata but a different conId is a separate contract.
    # An unattributed quantity there prevents a whole-run CLEAR.
    exact["aggregate_positions"].append(
        _aggregate(
            snapshot_id="snap-2", source_sequence=2, qty="77", contract=_contract(2002)
        )
    )
    report = _evaluate(_refresh_receipts(exact))
    assert report.state is ControlState.UNKNOWN
    assert ReasonCode.AGGREGATE_POSITION_MISMATCH in _reason_set(report)


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


@pytest.mark.parametrize(
    "target",
    [
        "producer_receipts",
        "execution_receipt",
        "baseline_receipt",
        "position_receipt",
    ],
)
def test_future_source_receipts_fail_closed_and_are_visible(target: str):
    doc = manifest_dict()
    receipt = doc[target][0] if target == "producer_receipts" else doc[target]
    receipt["complete_through"] = "2026-09-05T21:30:01Z"
    report = _evaluate(doc)
    assert report.state is ControlState.UNKNOWN
    assert ReasonCode.FUTURE_SOURCE_RECEIPT in _reason_set(report)
    assert any(
        row.state is ControlState.UNKNOWN
        and any(
            issue.reason is ReasonCode.FUTURE_SOURCE_RECEIPT for issue in row.issues
        )
        for row in report.source_completeness
    )


def test_receipt_sequences_cannot_claim_phantom_records():
    doc = manifest_dict()
    doc["execution_revisions"][0]["source_sequence"] = 3
    doc["execution_receipt"].update(
        source_sequence_first=1,
        source_sequence_last=3,
        source_sequences=[1, 2, 3],
    )
    report = _evaluate(doc)
    assert report.state is ControlState.UNKNOWN
    assert ReasonCode.SOURCE_SEQUENCE_INCOMPLETE in _reason_set(report)


def test_zero_event_receipts_use_explicit_null_empty_sequence_form():
    doc = manifest_dict()
    doc["obligation_revisions"] = []
    doc["position_baselines"] = []
    doc["execution_revisions"] = []
    doc["aggregate_positions"] = []
    report = _evaluate(_refresh_receipts(doc))
    assert report.state is ControlState.NOT_SCHEDULED
    assert all(row.event_count == 0 for row in report.source_completeness)
    assert all(row.source_sequence_first is None for row in report.source_completeness)
    assert all(row.source_sequence_last is None for row in report.source_completeness)

    invalid = _refresh_receipts(copy.deepcopy(doc))
    invalid["execution_receipt"].update(
        source_sequence_first=1,
        source_sequence_last=1,
        source_sequences=[1],
    )
    report = _evaluate(invalid)
    assert report.state is ControlState.UNKNOWN
    assert ReasonCode.SOURCE_SEQUENCE_INCOMPLETE in _reason_set(report)


def test_execution_receipt_identity_includes_every_correction_revision():
    doc = manifest_dict()
    doc["execution_revisions"].append(
        _execution(
            execution_id="exec-1",
            revision=2,
            source_sequence=2,
            qty="-90",
            occurred_at="2026-09-05T21:21:00Z",
        )
    )
    doc["aggregate_positions"][0]["signed_qty"] = "10"
    report = _evaluate(_refresh_receipts(doc))
    assert report.state is ControlState.ALERT
    assert report.obligations[0].signed_residual_qty == 10
    execution_source = next(
        row for row in report.source_completeness if row.source_kind == "EXECUTION"
    )
    assert execution_source.state is ControlState.CLEAR
    assert execution_source.event_count == 2
    assert ReasonCode.EXECUTION_EVENT_SET_MISMATCH not in _reason_set(report)


@pytest.mark.parametrize(
    "reason,effective_at",
    [
        ("CORPORATE_ACTION", "2026-09-05T21:05:00Z"),
        ("CORRECTION", "2026-09-05T20:00:00Z"),
    ],
)
def test_revisioned_baseline_adjustments_are_append_only_and_point_in_time(
    reason: str, effective_at: str
):
    doc = manifest_dict()
    doc["position_baselines"].append(
        _baseline(
            revision=2,
            source_sequence=2,
            qty="120",
            recorded_at="2026-09-05T21:10:00Z",
            effective_at=effective_at,
            reason=reason,
        )
    )
    doc["execution_revisions"][0]["signed_qty_delta"] = "-120"
    report = _evaluate(_refresh_receipts(doc))
    assert report.state is ControlState.CLEAR
    reconciliation = report.contract_reconciliations[0]
    assert reconciliation.attributed_signed_qty == 0
    assert reconciliation.latest_baseline_recorded_at.isoformat().startswith(
        "2026-09-05T21:10"
    )


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (
            lambda row: row.update(revision=3),
            ReasonCode.BASELINE_REVISION_GAP,
        ),
        (
            lambda row: row.update(signal_id="DIFFERENT:SIGNAL"),
            ReasonCode.BASELINE_IDENTITY_CHANGED,
        ),
        (
            lambda row: row.update(recorded_at="2026-09-05T19:59:00Z"),
            ReasonCode.NON_MONOTONIC_BASELINE_REVISION,
        ),
        (
            lambda row: row.update(effective_at="2026-09-05T19:59:00Z"),
            ReasonCode.NON_MONOTONIC_BASELINE_REVISION,
        ),
        (
            lambda row: row.update(reason="INITIAL"),
            ReasonCode.NON_MONOTONIC_BASELINE_REVISION,
        ),
        (
            lambda row: row.update(recorded_at="2026-09-05T21:31:00Z"),
            ReasonCode.FUTURE_SOURCE_EVENT,
        ),
    ],
)
def test_invalid_baseline_histories_fail_closed(mutation, reason: ReasonCode):
    doc = manifest_dict()
    revision = _baseline(
        revision=2,
        source_sequence=2,
        qty="100",
        recorded_at="2026-09-05T21:10:00Z",
        effective_at="2026-09-05T20:00:00Z",
        reason="CORRECTION",
    )
    mutation(revision)
    doc["position_baselines"].append(revision)
    report = _evaluate(_refresh_receipts(doc))
    assert report.state is ControlState.UNKNOWN
    assert reason in _reason_set(report)


def test_conflicting_snapshot_id_reuse_fails_closed():
    doc = manifest_dict()
    duplicate = copy.deepcopy(doc["aggregate_positions"][0])
    duplicate["signed_qty"] = "1"
    doc["aggregate_positions"].append(duplicate)
    report = _evaluate(_refresh_receipts(doc))
    assert report.state is ControlState.UNKNOWN
    assert ReasonCode.CONFLICTING_DUPLICATE_SNAPSHOT_ID in _reason_set(report)


@pytest.mark.parametrize("include_snapshot", [False, True])
def test_unrelated_non_due_contract_is_part_of_whole_account_reconciliation(
    include_snapshot: bool,
):
    doc = manifest_dict()
    msft = _contract(2002, "MSFT")
    doc["position_baselines"].append(
        _baseline(
            baseline_id="baseline-msft",
            source_sequence=2,
            qty="5",
            signal_id="OTHER:MSFT:2026-09-04",
            tranche_id="t9",
            contract=msft,
        )
    )
    if include_snapshot:
        doc["aggregate_positions"].append(
            _aggregate(
                snapshot_id="snap-msft",
                source_sequence=2,
                qty="0",
                contract=msft,
            )
        )
    report = _evaluate(_refresh_receipts(doc))
    assert report.state is ControlState.UNKNOWN
    expected = (
        ReasonCode.AGGREGATE_POSITION_MISMATCH
        if include_snapshot
        else ReasonCode.MISSING_POSITION_SNAPSHOT
    )
    assert expected in _reason_set(report)
    assert any(
        row.contract.startswith("MSFT/") for row in report.contract_reconciliations
    )


def test_reconciled_unrelated_non_due_contract_allows_core_clear():
    doc = manifest_dict()
    msft = _contract(2002, "MSFT")
    doc["position_baselines"].append(
        _baseline(
            baseline_id="baseline-msft",
            source_sequence=2,
            qty="5",
            signal_id="OTHER:MSFT:2026-09-04",
            tranche_id="t9",
            contract=msft,
        )
    )
    doc["aggregate_positions"].append(
        _aggregate(
            snapshot_id="snap-msft",
            source_sequence=2,
            qty="5",
            contract=msft,
        )
    )
    report = _evaluate(_refresh_receipts(doc))
    assert report.state is ControlState.CLEAR
    assert len(report.contract_reconciliations) == 2
    assert all(
        row.state is ControlState.CLEAR for row in report.contract_reconciliations
    )


def test_unrelated_active_execution_without_baseline_blocks_clear():
    doc = manifest_dict()
    msft = _contract(2002, "MSFT")
    doc["execution_revisions"].append(
        _execution(
            execution_id="exec-msft",
            family="family-msft",
            source_sequence=2,
            qty="5",
            signal_id="OTHER:MSFT:2026-09-04",
            tranche_id="t9",
            contract=msft,
        )
    )
    doc["aggregate_positions"].append(
        _aggregate(
            snapshot_id="snap-msft",
            source_sequence=2,
            qty="5",
            contract=msft,
        )
    )
    report = _evaluate(_refresh_receipts(doc))
    assert report.state is ControlState.UNKNOWN
    assert ReasonCode.MISSING_BASELINE in _reason_set(report)


@pytest.mark.parametrize("target", ["baseline", "position"])
def test_future_baseline_or_position_record_fails_closed(target: str):
    doc = manifest_dict()
    if target == "baseline":
        doc["position_baselines"][0]["recorded_at"] = "2026-09-05T21:31:00Z"
    else:
        doc["aggregate_positions"][0]["observed_at"] = "2026-09-05T21:31:00Z"
    report = _evaluate(_refresh_receipts(doc))
    assert report.state is ControlState.UNKNOWN
    assert ReasonCode.FUTURE_SOURCE_EVENT in _reason_set(report)


def test_strict_schema_rejects_unknown_fields_and_naive_timestamps():
    doc = manifest_dict()
    doc["run"]["surprise"] = True
    with pytest.raises(ManifestError, match="unknown field"):
        Manifest.from_dict(doc)

    doc = manifest_dict()
    doc["run"]["as_of"] = "2026-09-05T21:30:00"
    with pytest.raises(ManifestError, match="UTC offset"):
        Manifest.from_dict(doc)


def test_control_boundary_has_no_broker_network_email_or_executor_imports():
    forbidden = {
        "boto3",
        "cache_io",
        "daily_scan",
        "google",
        "ib_insync",
        "order_staging",
        "requests",
        "smtplib",
        "socket",
        "strat_backtester",
        "urllib",
    }
    imported: set[str] = set()
    for relative in (
        Path("expected_flat_control.py"),
        Path("scripts/run_expected_flat_control.py"),
    ):
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
    assert imported.isdisjoint(forbidden)


def test_json_markdown_and_html_are_deterministic_human_first_and_escaped():
    doc = manifest_dict(mode="FIXTURE", authoritative=False)
    doc["obligation_revisions"][0]["strategy"] = "OLV <review>"
    report = _evaluate(doc, trusted_live=False)
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
    for heading in (
        "Source completeness",
        "Account-contract reconciliation",
        "Provenance and digests",
    ):
        assert heading in markdown
        assert f"<h2>{heading}</h2>" in html
    for digest in report.digests.to_dict().values():
        assert digest in markdown
        assert digest in html
    parsed = json.loads(render_json(report))
    assert parsed["state"] == "UNKNOWN"
    assert parsed["operationally_authoritative"] is False
    assert parsed["schema_version"] == 2
    assert parsed["algorithm_version"] == "expected-flat/2.0.0"
    assert parsed["source_completeness"][0]["state"] == "CLEAR"
    reconciliation = parsed["contract_reconciliations"][0]
    assert reconciliation["attributed_signed_qty"] == "0"
    assert reconciliation["aggregate_signed_qty"] == "0"
    assert reconciliation["latest_baseline_effective_at"] == "2026-09-05T20:00:00Z"
    assert reconciliation["latest_baseline_recorded_at"] == "2026-09-05T20:00:00Z"
    assert reconciliation["latest_execution_at"] == "2026-09-05T21:20:00Z"
    assert reconciliation["snapshot_observed_at"] == "2026-09-05T21:29:00Z"


def test_markdown_and_html_neutralize_tracking_pixels_and_embedded_newlines():
    attack = "bad\n![pixel](https://attacker.invalid/p.gif)\n<img src=x>"
    doc = manifest_dict(mode="SHADOW")
    doc["obligation_revisions"][0]["strategy"] = attack
    doc["producer_receipts"][0]["receipt_id"] = attack
    doc["producer_receipts"][0]["complete"] = False
    report = _evaluate(doc, trusted_live=False)
    markdown = render_markdown(report)
    html = render_html(report)
    assert "\n![pixel](" not in markdown
    assert "\n<img" not in markdown
    assert "\\!\\[pixel\\]\\(https://attacker.invalid/p.gif\\)" in markdown
    assert "<img src=x>" not in html
    assert "&lt;img src=x&gt;" in html


def test_shadow_markdown_matches_reviewed_golden_artifact():
    fixture_dir = ROOT / "tests" / "fixtures" / "expected_flat"
    raw = json.loads((fixture_dir / "shadow_clean.json").read_text(encoding="utf-8"))
    report = _evaluate(raw, trusted_live=False)
    expected = (fixture_dir / "shadow_clean.golden.md").read_text(encoding="utf-8")
    assert render_markdown(report) == expected


def test_cli_fixture_publishes_complete_immutable_bundle_and_returns_unknown(
    tmp_path: Path,
):
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
    generations = list(output_dir.iterdir())
    assert len(generations) == 1 and generations[0].is_dir()
    generation = generations[0]
    assert sorted(path.name for path in generation.iterdir()) == [
        "completion.json",
        "report.html",
        "report.json",
        "report.md",
    ]
    completion = json.loads((generation / "completion.json").read_text("utf-8"))
    assert completion["complete"] is True
    assert set(completion["files"]) == {"html", "json", "markdown"}
    assert (
        completion["input_sha256"]
        == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )
    for metadata in completion["files"].values():
        artifact = generation / metadata["path"]
        assert artifact.stat().st_size == metadata["bytes"]
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == metadata["sha256"]
    assert not list(output_dir.rglob("*.tmp"))
    assert "NON-AUTHORITATIVE" in (generation / "report.md").read_text(encoding="utf-8")
    retried = subprocess.run(command, capture_output=True, text=True, check=False)
    assert retried.returncode == 20
    assert retried.stdout == completed.stdout
    assert list(output_dir.iterdir()) == [generation]


def test_cli_categorically_rejects_live_authority_and_removed_override(
    tmp_path: Path,
):
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

    live_manifest = subprocess.run(
        base + ["--run-mode", "SHADOW"], capture_output=True, text=True, check=False
    )
    assert live_manifest.returncode == 2
    assert "rejects LIVE manifests" in live_manifest.stderr
    assert not (tmp_path / "out").exists()

    live_option = subprocess.run(
        base + ["--run-mode", "LIVE"], capture_output=True, text=True, check=False
    )
    assert live_option.returncode == 2
    assert "invalid choice" in live_option.stderr
    assert not (tmp_path / "out").exists()

    removed_override = subprocess.run(
        base + ["--run-mode", "SHADOW", "--acknowledge-authoritative-live"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert removed_override.returncode == 2
    assert "unrecognized arguments" in removed_override.stderr
    assert not (tmp_path / "out").exists()

    authoritative = manifest_dict(mode="SHADOW", authoritative=False)
    authoritative["run"]["operationally_authoritative"] = True
    manifest_path.write_text(json.dumps(authoritative), encoding="utf-8")
    rejected_authority = subprocess.run(
        base + ["--run-mode", "SHADOW"], capture_output=True, text=True, check=False
    )
    assert rejected_authority.returncode == 2
    assert "future trusted adapter" in rejected_authority.stderr
    assert not (tmp_path / "out").exists()


def test_cli_rejects_resolved_input_output_collision_without_mutation(tmp_path: Path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    manifest_path = output_dir / "manifest.json"
    original = json.dumps(manifest_dict(mode="FIXTURE"), sort_keys=True)
    manifest_path.write_text(original, encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_expected_flat_control.py"),
            "--input",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--run-mode",
            "FIXTURE",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "outside the resolved output directory" in completed.stderr
    assert manifest_path.read_text(encoding="utf-8") == original
    assert list(output_dir.iterdir()) == [manifest_path]


def test_exact_report_generation_retry_is_idempotent_and_byte_verified(tmp_path: Path):
    report = _evaluate(manifest_dict())
    paths = write_artifacts(report, tmp_path / "out")
    before = {path.name: path.read_bytes() for path in paths["generation"].iterdir()}
    retried = write_artifacts(report, tmp_path / "out")
    after = {path.name: path.read_bytes() for path in paths["generation"].iterdir()}
    assert retried == paths
    assert after == before


def test_partial_existing_generation_fails_closed_without_rewrite(tmp_path: Path):
    report = _evaluate(manifest_dict())
    output_dir = tmp_path / "out"
    generation = output_dir / control.artifact_stem(report)
    generation.mkdir(parents=True)
    partial = generation / "report.json"
    partial.write_text(render_json(report), encoding="utf-8")
    with pytest.raises(ManifestError, match="partial or contains unexpected"):
        write_artifacts(report, output_dir)
    assert list(generation.iterdir()) == [partial]


def test_tampered_existing_generation_fails_closed_without_rewrite(tmp_path: Path):
    report = _evaluate(manifest_dict())
    paths = write_artifacts(report, tmp_path / "out")
    paths["markdown"].write_text("tampered\n", encoding="utf-8")
    before = {path.name: path.read_bytes() for path in paths["generation"].iterdir()}
    with pytest.raises(ManifestError, match="does not match this run"):
        write_artifacts(report, tmp_path / "out")
    after = {path.name: path.read_bytes() for path in paths["generation"].iterdir()}
    assert after == before


@pytest.mark.parametrize("failure_boundary", [1, 2, 3, 4])
def test_publish_failure_leaves_no_partial_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_boundary: int
):
    report = _evaluate(manifest_dict())
    output_dir = tmp_path / f"out-{failure_boundary}"
    real_write = control.atomic_write_text
    calls = 0

    def injected_failure(path: Path, content: str) -> None:
        nonlocal calls
        calls += 1
        if calls == failure_boundary:
            raise OSError("injected publication failure")
        real_write(path, content)

    monkeypatch.setattr(control, "atomic_write_text", injected_failure)
    with pytest.raises(OSError, match="injected publication failure"):
        control.write_artifacts(report, output_dir)
    assert output_dir.exists()
    assert list(output_dir.iterdir()) == []
