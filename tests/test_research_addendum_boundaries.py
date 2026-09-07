"""Offline coverage, manifest stability and ambiguous-delivery regressions."""
import json
import smtplib
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from fundamental.config import FMP_ENDPOINTS
from fundamental.coverage import ready_coverage
from fundamental.run_manifest import freeze_sources
from fundamental.universe import select_balanced_enrichment_batch
from research_delivery import DeliveryNotSent, DeliveryUncertain, deliver_once
from research_io import read_jsonl
from scripts import run_fundamental_sleeve as runner
from scripts import send_context_slack as context
from scripts import send_posts_email as posts


def test_readiness_requires_each_endpoint_current_and_not_future():
    rows = [{"ticker": ticker, "endpoint": endpoint, "snapshot_as_of": day}
        for ticker, day in [("FRESH", "2026-09-01"), ("STALE", "2026-01-01"), ("FUTURE", "2026-09-07")]
        for endpoint in FMP_ENDPOINTS]
    rows += [{"ticker": "PARTIAL", "endpoint": endpoint,
              "snapshot_as_of": "2026-01-01" if endpoint == FMP_ENDPOINTS[0] else "2026-09-01"}
             for endpoint in FMP_ENDPOINTS]
    sec = pd.DataFrame([{"ticker": t, "snapshot_as_of": "2026-09-01"} for t in ["FRESH", "STALE", "FUTURE", "PARTIAL"]])
    baseline, deep, _ = ready_coverage(pd.DataFrame(rows), sec, as_of="2026-09-06")
    assert baseline == deep == {"FRESH"}
    sec.loc[sec.ticker == "FRESH", "snapshot_as_of"] = "2020-01-01"
    assert ready_coverage(pd.DataFrame(rows), sec, as_of="2026-09-06")[1] == set()


def test_source_change_or_added_archive_prevents_completed_manifest(tmp_path, monkeypatch):
    path = tmp_path / "input.json"
    path.write_text('{"generation":1}')
    paths = {"input": path}
    monkeypatch.setattr(runner, "_source_paths", lambda as_of: paths)
    before = freeze_sources(paths)
    runner._assert_sources_unchanged(before, "2026-09-06")
    path.write_text('{"generation":2}')
    with pytest.raises(RuntimeError, match="inputs changed"):
        runner._assert_sources_unchanged(before, "2026-09-06")
    before = freeze_sources(paths)
    paths["new_archive"] = tmp_path / "new.parquet"
    with pytest.raises(RuntimeError):
        runner._assert_sources_unchanged(before, "2026-09-06")


def test_run_plan_previews_actual_balanced_batch(monkeypatch):
    universe = pd.DataFrame([
        {"ticker": ticker, "research_eligible": True, "research_lane": "standard_company",
         "market_cap_band": "large", "sector": sector, "dollar_volume_63d": volume, "as_of": "2026-09-06"}
        for ticker, sector, volume in [("AAA", "Technology", 1), ("BBB", "Technology", 100), ("CCC", "Utilities", 100)]])
    monkeypatch.setattr(runner, "_eligible_universe", lambda: universe)
    monkeypatch.setattr(runner, "_read_parquet", lambda path: pd.DataFrame())
    monkeypatch.setattr(runner, "load_underwrite_decisions", lambda path: [])
    monkeypatch.setattr(runner, "load_research_controls", lambda *a, **kw: ({}, {}))
    monkeypatch.setattr(runner, "load_research_event_state", lambda **kw: {
        "thesis_events": [], "trigger_events": [], "completed_control_requests": {}, "health": {}})
    monkeypatch.setattr(runner, "load_portfolio_snapshot", lambda **kw: ({}, {}))
    plan = runner.build_run_plan(as_of="2026-09-06", batch_size=2, universe_refresh_days=7)
    expected = select_balanced_enrichment_batch(universe, 2, include_specialists=True)
    assert expected == ["BBB", "CCC"]
    assert plan["coverage"]["bounded_refresh_tickers"] == expected


def test_execute_run_refuses_manifest_when_report_consumes_changing_inputs(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    source.write_text('{"generation":1}')
    report_path = tmp_path / "daily.json"
    monkeypatch.setattr(runner, "DAILY_REPORT_CURRENT", report_path)
    monkeypatch.setattr(runner, "_source_paths", lambda as_of: {"source": source})
    published = []
    monkeypatch.setattr(runner, "append_decision_transitions", lambda **kw: published.append("transition"))
    monkeypatch.setattr(runner, "write_sleeve_run_manifest", lambda value: published.append("manifest"))
    def run(command):
        if "scripts/build_fundamental_report.py" in command:
            report_path.write_text('{"health":{},"underwrite_decisions":[]}')
            source.write_text('{"generation":2}')
    monkeypatch.setattr(runner, "_run", run)
    args = SimpleNamespace(refresh_universe=False, refresh=False, refresh_prices=False, verify=False,
                           as_of="2026-09-06", output=tmp_path / "report.html")
    with pytest.raises(RuntimeError, match="inputs changed"):
        runner.execute_run(args, {"universe": {"stale": False}, "coverage": {"baseline_gap": 0}})
    assert published == []


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
