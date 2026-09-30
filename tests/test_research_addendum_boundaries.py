"""Offline coverage, manifest stability and ambiguous-delivery regressions."""
import json
import smtplib
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from research_delivery import DeliveryNotSent, DeliveryUncertain, deliver_once
from research_io import read_jsonl
from scripts import send_context_slack as context
from scripts import send_posts_email as posts


def test_delivery_claim_blocks_ambiguous_retry_and_changed_content(tmp_path):
    path = tmp_path / "receipts.jsonl"
    calls = []
    def accepted_but_lost_response():
        calls.append("accepted")
        raise OSError("fixture response lost")
    with pytest.raises(OSError):
        deliver_once(path, {"day": "fixture"}, {"text": "one"}, accepted_but_lost_response)
    for payload in ({"text": "one"}, {"text": "changed"}):
        with pytest.raises(DeliveryUncertain):
            deliver_once(path, {"day": "fixture"}, payload, accepted_but_lost_response)
    assert calls == ["accepted"]
    assert [r["state"] for r in read_jsonl(path)] == ["SENDING"]


def test_proven_presubmission_failure_can_retry_and_confirmed_send_deduplicates(tmp_path):
    path = tmp_path / "receipts.jsonl"
    def offline():
        raise DeliveryNotSent("fixture connection never opened")
    with pytest.raises(DeliveryNotSent):
        deliver_once(path, {"day": "fixture"}, {"text": "one"}, offline)
    calls = []
    first = deliver_once(path, {"day": "fixture"}, {"text": "one"}, lambda: calls.append(1))
    second = deliver_once(path, {"day": "fixture"}, {"text": "one"}, lambda: calls.append(2))
    assert first["status"] == "SENT" and second["status"] == "ALREADY_SENT" and calls == [1]


def test_corrupt_delivery_tail_is_preserved_before_any_transport(tmp_path):
    path = tmp_path / "receipts.jsonl"
    path.write_bytes(b'{"state":')
    calls = []
    with pytest.raises(ValueError):
        deliver_once(path, {}, {}, lambda: calls.append(1))
    assert path.read_bytes() == b'{"state":' and calls == []


def test_delivery_receipt_missing_identity_is_not_treated_as_empty(tmp_path):
    path = tmp_path / "receipts.jsonl"
    path.write_text('{"schema_version":"research-delivery.v1","state":"SENT"}\n')
    before = path.read_bytes()
    calls = []
    with pytest.raises(ValueError):
        deliver_once(path, {}, {}, lambda: calls.append(1))
    assert calls == [] and path.read_bytes() == before


def test_simultaneous_delivery_attempts_use_one_transport_call(tmp_path):
    path = tmp_path / "receipts.jsonl"
    calls = []
    def attempt():
        return deliver_once(path, {"day": "same"}, {"text": "same"}, lambda: calls.append(1))["status"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: attempt(), range(2)))
    assert sorted(results) == ["ALREADY_SENT", "SENT"] and calls == [1]


def test_posts_data_accepted_then_quit_disconnect_is_success_and_not_resent(tmp_path, monkeypatch):
    calls = []
    class SMTP:
        def __init__(self, *args): pass
        def starttls(self): pass
        def login(self, *args): pass
        def sendmail(self, *args): calls.append(1); return {}
        def quit(self): raise smtplib.SMTPServerDisconnected("fixture post-DATA disconnect")
        def close(self): pass
    monkeypatch.setattr(posts.smtplib, "SMTP", SMTP)
    send = lambda: posts.send_queue_email("fixture", "fixture", ["fixture"], SimpleNamespace(as_string=lambda: "synthetic"))
    for _ in range(2):
        deliver_once(tmp_path / "receipts.jsonl", {"day": "fixture"}, {"body": "fixture"}, send)
    assert calls == [1]


@pytest.mark.parametrize("mode", ["webhook", "token"])
def test_slack_uncertain_response_never_automatically_reposts(monkeypatch, mode):
    calls = []
    def accepted_but_timeout(*args, **kwargs):
        calls.append(1)
        raise urllib.error.URLError("fixture timeout after acceptance")
    monkeypatch.setattr(context.urllib.request, "urlopen", accepted_but_timeout)
    with pytest.raises(DeliveryUncertain):
        if mode == "webhook":
            context.post_via_webhook("https://fixture.invalid", "test", [], "purple")
        else:
            context.post_via_token("fixture", "fixture", "test", [], "purple")
    assert calls == [1]


def test_post_send_context_recovery_is_idempotent_and_preserves_bad_journal(tmp_path, monkeypatch):
    monkeypatch.setattr(context, "FLAG_STATE_PATH", tmp_path / "flags.json")
    monkeypatch.setattr(context, "JOURNAL_PATH", tmp_path / "journal.jsonl")
    sidecar = {"nuggets": [{"fingerprint": "fixture", "mean_pct": 1.2}]}
    for _ in range(2):
        context.advance_flag_state(sidecar, "2026-09-06", "same-delivery")
        context.append_journal(sidecar, "2026-09-06", Path("fixture.md"), "same-delivery")
    assert json.loads(context.FLAG_STATE_PATH.read_text())["flags"]["fixture"]["count"] == 1
    assert len(read_jsonl(context.JOURNAL_PATH)) == 1
    with context.JOURNAL_PATH.open("ab") as handle:
        handle.write(b'{"damaged":')
    before = context.JOURNAL_PATH.read_bytes()
    with pytest.raises(ValueError):
        context.append_journal(sidecar, "2026-09-06", Path("fixture.md"), "next-delivery")
    assert context.JOURNAL_PATH.read_bytes() == before
