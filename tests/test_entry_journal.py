import json
from types import SimpleNamespace as NS

import pytest

from broker_runtime.entry_journal import EntryJournal, initialize_new

DATE, REF, FP = "2026-09-08", "SPY|BUY|Algo|2026-09-08", ("SPY", "BUY", 10, "LMT", 100)


def test_missing_or_corrupt_journal_never_becomes_empty(tmp_path):
    path = tmp_path / "journal.json"
    with pytest.raises(ValueError, match="unavailable"):
        with EntryJournal(path): pass
    path.write_text("broken")
    with pytest.raises(ValueError, match="unavailable"):
        with EntryJournal(path): pass
    assert path.read_text() == "broken"


def test_complete_runner_lock_excludes_another_writer(tmp_path):
    path = tmp_path / "journal.json"
    initialize_new(path)
    with EntryJournal(path):
        with pytest.raises(OSError):
            with EntryJournal(path): pass


def test_pre_wire_claim_survives_ambiguous_send_and_restart(tmp_path, monkeypatch):
    monkeypatch.delenv("LEGEND_ETF_PRIMARY_ACCOUNT", raising=False)
    path = tmp_path / "journal.json"
    initialize_new(path)
    broker = NS(managedAccounts=lambda: ["TEST_PRIMARY"])
    order = NS(totalQuantity=10, account="")
    def wire(*a, **k):
        assert json.loads(path.read_text())[0]["state"] == "awaiting_reconciliation"
        raise TimeoutError("ambiguous wire")
    with EntryJournal(path) as journal:
        with pytest.raises(TimeoutError):
            journal.place(DATE, REF, FP, wire, broker, NS(conId=42), order)
    with EntryJournal(path) as journal:
        assert REF in journal.unresolved_refs()
        with pytest.raises(ValueError, match="prior entry intent"):
            journal.place(DATE, REF, FP, lambda *a, **k: pytest.fail("duplicate wire"), broker, NS(conId=42), order)


def test_parent_ack_does_not_attest_missing_children_and_old_claims_are_retained(tmp_path, monkeypatch):
    monkeypatch.delenv("LEGEND_ETF_PRIMARY_ACCOUNT", raising=False)
    path = tmp_path / "journal.json"
    initialize_new(path)
    broker = NS(managedAccounts=lambda: ["TEST_PRIMARY"])
    order = NS(totalQuantity=10, account="", orderId=7, clientId=9, permId=701)
    parent = NS(order=order, contract=NS(conId=42), orderStatus=NS(status="Submitted"))
    child = NS(order=NS(account="TEST_PRIMARY", orderId=8, clientId=9, permId=702, parentId=7),
               contract=NS(conId=42), orderStatus=NS(status="PendingSubmit"))
    with EntryJournal(path) as journal:
        journal.place(DATE, REF, FP, lambda *a, **k: parent, broker, parent.contract, order)
        journal.record(DATE, REF, FP, "parent_submitted")
        with pytest.raises(ValueError, match="incomplete"):
            journal.complete(DATE, REF, FP, [parent, child])
        assert REF in journal.unresolved_refs()
        child.orderStatus.status = "Submitted"
        journal.complete(DATE, REF, FP, [parent, child])
        assert REF not in journal.unresolved_refs()
        journal.record("2026-09-09", "new-ref", FP, "awaiting_reconciliation")
        assert len(journal.read()) == 2
        assert REF in journal.references("2026-09-09")[0]


def test_explicit_bootstrap_never_overwrites_existing_history(tmp_path):
    path = tmp_path / "journal.json"
    initialize_new(path)
    with pytest.raises(FileExistsError):
        initialize_new(path)
