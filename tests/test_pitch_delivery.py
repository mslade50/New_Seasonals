import json
import threading
from pathlib import Path

import pytest

import pitch_delivery as delivery
import pitch_journal


ASOF = "2026-08-31"


def records(title: str = "Long GLD") -> list[dict]:
    return [
        {"kind": "idea", "date": ASOF, "idea_id": f"{ASOF}-1",
         "rank": 1, "title": title},
        {"kind": "killed", "date": ASOF, "title": "Short TLT",
         "reason": "era split flipped"},
    ]


def reserve(path: Path, planned: list[dict] | None = None):
    return delivery.reserve_delivery(
        asof=ASOF, records=planned or records(), subject="Daily Pitch",
        html="<p>idea</p>", recipients=["m@example.com"], path=path,
        use_r2=False)


def install_fake_r2(monkeypatch, tmp_path):
    import cache_io

    state = {"body": None, "etag": None, "sequence": 0,
             "last_error": "404 Not Found", "writes": []}
    mutex = threading.Lock()
    monkeypatch.setattr(delivery, "R2_DOWNLOAD_DIR", tmp_path / "downloads")
    monkeypatch.setattr(cache_io, "is_configured", lambda: True)

    def head(key):
        with mutex:
            if state["body"] is None:
                return None
            return {"ETag": state["etag"], "ContentLength": len(state["body"])}

    def download(key, path):
        with mutex:
            body = state["body"]
        if body is None:
            state["last_error"] = "404 Not Found"
            return False
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        return True

    def conditional(path, key, *, create_only=False, expected_etag=None):
        body = Path(path).read_bytes()
        with mutex:
            if create_only and state["body"] is not None:
                return "precondition_failed", None
            if expected_etag is not None and expected_etag != state["etag"]:
                return "precondition_failed", None
            state["sequence"] += 1
            state["etag"] = f'"etag-{state["sequence"]}"'
            state["body"] = body
            state["writes"].append(json.loads(body.decode("utf-8")))
            return "uploaded", state["etag"]

    monkeypatch.setattr(cache_io, "head", head)
    monkeypatch.setattr(cache_io, "download_to_local", download)
    monkeypatch.setattr(cache_io, "last_download_error",
                        lambda: state["last_error"])
    monkeypatch.setattr(cache_io, "conditional_upload_from_local", conditional)
    return state


def test_verdict_digest_is_order_independent_and_ignores_write_stamp():
    first = records()
    second = [{**first[1], "written_at": "later"},
              {**first[0], "written_at": "earlier"}]
    assert delivery.verdict_digest(first) == delivery.verdict_digest(second)


def test_confirmed_send_is_skipped_on_a_matching_rerun(tmp_path):
    path = tmp_path / "receipt.json"
    receipt, should_send = reserve(path)
    assert should_send and receipt["status"] == "sending"
    delivery.complete_delivery(receipt, path, use_r2=False, sent=True)

    existing, should_send = reserve(path)
    assert not should_send
    assert existing["status"] == "sent"


def test_a_different_verdict_cannot_reuse_a_sent_receipt(tmp_path):
    path = tmp_path / "receipt.json"
    receipt, _ = reserve(path)
    delivery.complete_delivery(receipt, path, use_r2=False, sent=True)

    with pytest.raises(delivery.DeliveryReceiptError, match="digest differs"):
        reserve(path, records("Short GLD"))


@pytest.mark.parametrize("sent", [False])
def test_uncertain_smtp_outcome_blocks_automatic_resend(tmp_path, sent):
    path = tmp_path / "receipt.json"
    receipt, _ = reserve(path)
    updated = delivery.complete_delivery(
        receipt, path, use_r2=False, sent=sent, reason="socket reset")
    assert updated["status"] == "ambiguous"
    with pytest.raises(delivery.DeliveryReceiptError, match="ambiguous"):
        reserve(path)


def test_sending_receipt_blocks_automatic_resend(tmp_path):
    path = tmp_path / "receipt.json"
    reserve(path)
    with pytest.raises(delivery.DeliveryReceiptError, match="sending"):
        reserve(path)


def test_reconcile_journal_appends_only_missing_records(tmp_path):
    path = tmp_path / "journal.jsonl"
    planned = records()
    pitch_journal.append(planned[:1], path, push=False)

    assert delivery.reconcile_journal(planned, path) == 1
    assert delivery.reconcile_journal(planned, path) == 0
    final = pitch_journal.load(path, pull=False)
    assert delivery.verdict_digest(final, ASOF) == delivery.verdict_digest(planned)
    assert len(final) == 2


def test_concurrent_recovery_reruns_do_not_duplicate_journal_records(tmp_path):
    path = tmp_path / "journal.jsonl"
    results = []
    errors = []

    def reconcile():
        try:
            results.append(delivery.reconcile_journal(records(), path))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=reconcile) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors
    assert sorted(results) == [0, 2]
    final = pitch_journal.load(path, pull=False)
    assert len(final) == 2


def test_reconcile_journal_rejects_conflicting_same_day_state(tmp_path):
    path = tmp_path / "journal.jsonl"
    pitch_journal.append(records("Different idea"), path, push=False)
    with pytest.raises(delivery.DeliveryReceiptError, match="conflicting"):
        delivery.reconcile_journal(records(), path)


def test_production_journal_reconciliation_fails_when_r2_upload_fails(
        tmp_path, monkeypatch):
    import cache_io

    journal_path = tmp_path / "production-journal.jsonl"
    monkeypatch.setattr(pitch_journal, "JOURNAL_PATH", journal_path)
    monkeypatch.setattr(cache_io, "is_configured", lambda: True)
    monkeypatch.setattr(cache_io, "upload_from_local", lambda *args: False)

    with pytest.raises(delivery.DeliveryReceiptError, match="upload to R2"):
        delivery.reconcile_journal(records(), journal_path)


def test_production_journal_is_downloaded_and_digest_verified(
        tmp_path, monkeypatch):
    import cache_io

    journal_path = tmp_path / "production-journal.jsonl"
    cloud = {"body": None}
    monkeypatch.setattr(pitch_journal, "JOURNAL_PATH", journal_path)
    monkeypatch.setattr(delivery, "R2_JOURNAL_DOWNLOAD_DIR",
                        tmp_path / "cloud-download")
    monkeypatch.setattr(cache_io, "is_configured", lambda: True)

    def upload(path, key):
        cloud["body"] = Path(path).read_bytes()
        return True

    def download(key, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(cloud["body"])
        return True

    monkeypatch.setattr(cache_io, "upload_from_local", upload)
    monkeypatch.setattr(cache_io, "download_to_local", download)

    assert delivery.reconcile_journal(records(), journal_path) == 2
    cloud_records = delivery.load_cloud_journal()
    assert delivery.verdict_digest(cloud_records, ASOF) == delivery.verdict_digest(
        records())


def test_sent_stand_down_can_be_followed_by_one_directed_amendment(tmp_path):
    receipt_path = tmp_path / "receipt.json"
    journal_path = tmp_path / "journal.jsonl"
    stand_down = [
        {"kind": "stand_down", "date": ASOF,
         "candidates_considered": 24},
        {"kind": "killed", "date": ASOF, "title": "Long GLD",
         "reason": "era split flipped"},
    ]
    directed = [
        {"kind": "idea", "date": ASOF, "idea_id": f"{ASOF}-1",
         "rank": 1, "title": "McKinley-directed TLT",
         "directed_by": "McKinley asked for this cell"},
    ]

    first, should_send = delivery.reserve_delivery(
        asof=ASOF, records=stand_down, subject="NO TRADES",
        html="<p>stand down</p>", recipients=["m@example.com"],
        path=receipt_path, use_r2=False)
    assert should_send
    delivery.complete_delivery(first, receipt_path, use_r2=False, sent=True)
    assert delivery.reconcile_journal(stand_down, journal_path) == 2

    prior = pitch_journal.load(journal_path, pull=False)
    amendment, should_send = delivery.reserve_delivery(
        asof=ASOF, records=directed, subject="Directed idea",
        html="<p>directed</p>", recipients=["m@example.com"],
        path=receipt_path, use_r2=False, prior_records=prior)
    assert should_send
    assert amendment["delivery_id"] != first["delivery_id"]
    assert len(amendment["delivery_history"]) == 1
    final_receipt = delivery.complete_delivery(
        amendment, receipt_path, use_r2=False, sent=True)
    assert delivery.reconcile_journal(directed, journal_path) == 1
    assert delivery.reconcile_journal(directed, journal_path) == 0

    final_journal = pitch_journal.load(journal_path, pull=False)
    delivery.verify_sent_receipt(final_receipt, final_journal, ASOF)
    assert [record["kind"] for record in final_journal] == [
        "stand_down", "killed", "idea"]


def test_directed_amendment_recovery_accepts_a_partially_appended_batch(
        tmp_path):
    journal_path = tmp_path / "journal.jsonl"
    baseline = [
        {"kind": "stand_down", "date": ASOF,
         "candidates_considered": 24},
        {"kind": "killed", "date": ASOF, "title": "baseline kill",
         "reason": "failed control"},
    ]
    directed = [
        {"kind": "idea", "date": ASOF, "idea_id": f"{ASOF}-1",
         "rank": 1, "title": "Directed one", "directed_by": "McKinley"},
        {"kind": "idea", "date": ASOF, "idea_id": f"{ASOF}-2",
         "rank": 2, "title": "Directed two", "directed_by": "McKinley"},
    ]
    pitch_journal.append(baseline + directed[:1], journal_path, push=False)

    assert delivery.reconcile_journal(directed, journal_path) == 1
    assert delivery.reconcile_journal(directed, journal_path) == 0
    final = pitch_journal.load(journal_path, pull=False)
    assert len(delivery.verdict_records(final, ASOF)) == 4


def test_verify_sent_receipt_requires_exact_count_and_digest(tmp_path):
    path = tmp_path / "receipt.json"
    receipt, _ = reserve(path)
    receipt = delivery.complete_delivery(
        receipt, path, use_r2=False, sent=True)
    delivery.verify_sent_receipt(receipt, records(), ASOF)

    bad = json.loads(path.read_text(encoding="utf-8"))
    bad["verdict_count"] += 1
    with pytest.raises(delivery.DeliveryReceiptError, match="records"):
        delivery.verify_sent_receipt(bad, records(), ASOF)


def test_production_reservation_persists_sending_to_r2_before_smtp(
        tmp_path, monkeypatch):
    state = install_fake_r2(monkeypatch, tmp_path)
    path = tmp_path / "receipt.json"
    receipt, should_send = delivery.reserve_delivery(
        asof=ASOF, records=records(), subject="Daily Pitch", html="<p>x</p>",
        recipients=["m@example.com"], path=path, use_r2=True)

    assert should_send
    assert receipt["status"] == "sending"
    assert state["writes"][0]["status"] == "sending"
    delivery.complete_delivery(receipt, path, use_r2=True, sent=True)
    assert state["writes"][-1]["status"] == "sent"


def test_production_reservation_blocks_when_r2_cannot_persist(
        tmp_path, monkeypatch):
    import cache_io

    install_fake_r2(monkeypatch, tmp_path)
    monkeypatch.setattr(
        cache_io, "conditional_upload_from_local",
        lambda *args, **kwargs: ("error", None))

    with pytest.raises(delivery.DeliveryReceiptError, match="persist"):
        delivery.reserve_delivery(
            asof=ASOF, records=records(), subject="Daily Pitch",
            html="<p>x</p>", recipients=["m@example.com"],
            path=tmp_path / "receipt.json", use_r2=True)


def test_conditional_r2_claim_allows_only_one_concurrent_sender(
        tmp_path, monkeypatch):
    import cache_io

    state = install_fake_r2(monkeypatch, tmp_path)
    original = cache_io.conditional_upload_from_local
    barrier = threading.Barrier(2)

    def racing_claim(*args, **kwargs):
        if kwargs.get("create_only"):
            barrier.wait(timeout=5)
        return original(*args, **kwargs)

    monkeypatch.setattr(cache_io, "conditional_upload_from_local", racing_claim)
    outcomes = []

    def claim(path):
        try:
            receipt, should_send = delivery.reserve_delivery(
                asof=ASOF, records=records(), subject="Daily Pitch",
                html="<p>x</p>", recipients=["m@example.com"], path=path,
                use_r2=True)
            outcomes.append(("send", receipt["delivery_id"], should_send))
        except delivery.DeliveryReceiptError as exc:
            outcomes.append(("blocked", str(exc), False))

    threads = [
        threading.Thread(target=claim, args=(tmp_path / "machine-a.json",)),
        threading.Thread(target=claim, args=(tmp_path / "machine-b.json",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not any(thread.is_alive() for thread in threads)
    assert [outcome[0] for outcome in outcomes].count("send") == 1
    assert [outcome[0] for outcome in outcomes].count("blocked") == 1
    assert len(state["writes"]) == 1
