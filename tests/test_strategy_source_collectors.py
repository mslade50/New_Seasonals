from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from research.strategy_discovery.contracts import validate_item, validate_manifest
from research.strategy_discovery.source_collectors import (
    CollectorError,
    collect_bundle,
    empty_state,
)
from scripts import collect_strategy_sources as cli

NOW = dt.datetime(2026, 9, 7, 12, 0, tzinfo=dt.timezone.utc)


class Response:
    def __init__(self, payload, status=200, headers=None):
        self._payload = payload
        self.status_code = status
        self.headers = headers or {}
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def source_config(*sources, mode="LIVE", item_budget=150):
    return {
        "schema_version": "strategy-source-config.v1",
        "run_mode": mode,
        "window_hours": 48,
        "request_budget": 12,
        "item_budget": item_budget,
        "x_bearer_token_env": "X_TOKEN",
        "crossref_contact_email_env": "CONTACT",
        "report_title": "Fixture research",
        "sources": list(sources),
    }


def x_source(kind="ACCOUNT", value="@alpha"):
    return {
        "source_id": "x:alpha",
        "platform": "X",
        "kind": kind,
        "value": value,
        "enabled": True,
        "max_pages": 2,
        "max_items": 50,
    }


def ssrn_source():
    return {
        "source_id": "ssrn:quant",
        "platform": "SSRN",
        "kind": "SSRN_QUERY",
        "value": "quantitative trading",
        "enabled": True,
        "max_pages": 2,
        "max_items": 75,
    }


def test_x_account_collection_uses_official_endpoint_and_preserves_raw_boundary():
    session = Session(
        [
            Response({"data": {"id": "42", "username": "alpha"}}),
            Response(
                {
                    "data": [
                        {
                            "id": "100",
                            "author_id": "42",
                            "conversation_id": "100",
                            "created_at": "2026-09-07T11:00:00Z",
                            "text": "Buy every dip (unverified source wording).",
                        }
                    ],
                    "includes": {"users": [{"id": "42", "username": "alpha"}]},
                    "meta": {"newest_id": "100"},
                },
                headers={"x-rate-limit-remaining": "98"},
            ),
        ]
    )
    manifest, items, discovery_config, telemetry = collect_bundle(
        source_config(x_source()),
        empty_state(),
        environment={"X_TOKEN": "secret", "CONTACT": "ops@example.com"},
        now=NOW,
        session=session,
    )
    validate_manifest(manifest)
    validate_item(items[0], 0)
    assert session.calls[0][0].endswith("/users/by/username/alpha")
    assert session.calls[1][0].endswith("/users/42/tweets")
    assert session.calls[1][1]["params"]["max_results"] >= 10
    assert items[0]["claims"] == [] and items[0]["strategy_proposal"] is None
    assert items[0]["text"].startswith("Buy every dip")
    assert "secret" not in json.dumps([manifest, items, discovery_config, telemetry])
    assert telemetry["requests_used"] == 2


def test_ssrn_crossref_collection_binds_doi_version_and_revision_window():
    work = {
        "DOI": "10.2139/ssrn.1234567",
        "title": ["A Costed Liquid Strategy"],
        "author": [{"given": "Ada", "family": "Quant"}],
        "abstract": "<jats:p>We test a liquid rule.</jats:p>",
        "created": {"date-time": "2026-09-06T10:00:00Z"},
        "published": {"date-parts": [[2026, 9, 6]]},
        "deposited": {"date-time": "2026-09-07T09:00:00Z"},
        "type": "posted-content",
        "subtype": "preprint",
    }
    session = Session([Response({"message": {"items": [work], "next-cursor": "next"}})])
    manifest, items, _, telemetry = collect_bundle(
        source_config(ssrn_source()),
        empty_state(),
        environment={"CONTACT": "ops@example.com"},
        now=NOW,
        session=session,
    )
    item = items[0]
    validate_item(item, 0)
    assert item["post_id"] == "ssrn:1234567"
    assert item["created_at"] == "2026-09-07T09:00:00Z"
    assert item["source_document"]["doi"] == "10.2139/ssrn.1234567"
    assert item["source_document"]["published_at"] == "2026-09-06T00:00:00Z"
    assert len(item["source_document"]["version_digest"]) == 64
    params = session.calls[0][1]["params"]
    assert "from-deposit-date" in params["filter"]
    assert params["mailto"] == "ops@example.com"
    assert manifest["sources"][0]["provider_status"] == "OK"
    assert telemetry["requests_used"] == 1


def test_ssrn_revision_uses_deposit_as_window_availability_time():
    work = {
        "DOI": "10.2139/ssrn.7654321",
        "title": ["An Older Paper Revised Today"],
        "author": [{"given": "Grace", "family": "Researcher"}],
        "created": {"date-time": "2020-01-02T10:00:00Z"},
        "published": {"date-parts": [[2020, 1, 2]]},
        "deposited": {"date-time": "2026-09-07T09:30:00Z"},
        "type": "posted-content",
    }
    session = Session([Response({"message": {"items": [work], "next-cursor": "next"}})])

    manifest, items, _, _ = collect_bundle(
        source_config(ssrn_source()),
        empty_state(),
        environment={"CONTACT": "ops@example.com"},
        now=NOW,
        session=session,
    )

    assert manifest["sources"][0]["window"]["start"] == "2026-09-05T12:00:00Z"
    assert items[0]["created_at"] == "2026-09-07T09:30:00Z"
    assert items[0]["source_document"]["published_at"] == "2020-01-02T00:00:00Z"
    assert (
        manifest["sources"][0]["window"]["start"]
        <= items[0]["created_at"]
        <= manifest["sources"][0]["window"]["end"]
    )


def test_missing_x_token_fails_without_network_call():
    session = Session([])
    with pytest.raises(CollectorError, match="bearer token"):
        collect_bundle(
            source_config(x_source()),
            empty_state(),
            environment={},
            now=NOW,
            session=session,
        )
    assert session.calls == []


def test_pending_capture_replays_then_complete_sources_advance(monkeypatch, tmp_path: Path):
    manifest = {
        "schema_version": "1.0",
        "provider": "fixture",
        "provider_version": "1",
        "sources": [
            {
                "source_id": "ssrn:quant",
                "platform": "SSRN",
                "discovery_only": True,
                "locator": {"kind": "SSRN_QUERY", "value": "quantitative trading"},
                "capture_id": "capture:fixture",
                "captured_at": "2026-09-07T12:00:00Z",
                "window": {"start": "2026-09-05T12:00:00Z", "end": "2026-09-07T12:00:00Z"},
                "cursor": {"in": None, "out": "ssrn-zero:2026-09-07T12:00:00Z", "exhausted": True},
                "provider_status": "OK",
                "expected_item_count": 0,
                "expected_min_items": 0,
                "observed_item_count": 0,
            }
        ],
    }
    discovery = {
        "schema_version": "1.0",
        "run_mode": "LIVE",
        "as_of": "2026-09-07T12:00:00Z",
        "source_max_age_hours": 54,
        "catalog_max_age_days": 30,
        "required_source_ids": ["ssrn:quant"],
        "source_locator_allowlist": {"ssrn:quant": {"kind": "SSRN_QUERY", "value": "quantitative trading"}},
        "report_title": "Fixture",
        "policy": {"auto_lifecycle_ceiling": "RESEARCH_READY", "x_discovery_only": True},
    }
    telemetry = {
        "schema_version": "strategy-source-telemetry.v1",
        "captured_at": "2026-09-07T12:00:00Z",
        "request_budget": 1,
        "requests_used": 1,
        "item_budget": 1,
        "items_observed": 0,
        "rate_limit_observations": [],
    }
    monkeypatch.setattr(cli, "collect_bundle", lambda *a, **k: (manifest, [], discovery, telemetry))
    monkeypatch.setattr(cli, "_verify_capture_journaled", lambda *a, **k: None)
    config_path = tmp_path / "sources.json"
    config_path.write_text(json.dumps(source_config(ssrn_source())), encoding="utf-8")
    state = tmp_path / "state.json"
    root = tmp_path / "captures"
    first = cli.collect(config_path, state, root, None)
    second = cli.collect(config_path, state, root, None)
    assert first == second
    assert json.loads(state.read_text())["pending"] is not None
    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps({
        "schema_version": "strategy-research-email-decision.v1",
        "source_bundle_digest": first.name,
        "positions_used": False,
        "email_required": False,
        "delivery_status": "NO_EMAIL",
    }), encoding="utf-8")
    accepted = cli.acknowledge(state, root, first, decision_path=decision)
    assert accepted["pending"] is None
    assert accepted["accepted"]["ssrn:quant"]["cursor_out"].startswith("ssrn-zero:")
    assert accepted["accepted"]["ssrn:quant"]["capture_digest"] == first.name


def test_partial_capture_acknowledges_evidence_without_advancing_cursor(monkeypatch, tmp_path: Path):
    base = empty_state()
    # Exercise the state transition helper directly; a partial observation is
    # accepted as processed evidence but cannot become the next cursor anchor.
    from research.strategy_discovery.source_collectors import accepted_from_manifest

    manifest = {
        "sources": [
            {
                "source_id": "x:alpha",
                "provider_status": "PARTIAL",
                "cursor": {"out": "101", "exhausted": False},
                "window": {"end": "2026-09-07T12:00:00Z"},
                "capture_id": "capture:partial",
            }
        ]
    }
    assert accepted_from_manifest(manifest) == {}
    assert base["accepted"] == {}


def test_acknowledgement_requires_a_committed_discovery_journal(monkeypatch, tmp_path: Path):
    # Reuse the pending-capture integration above to assert the critical
    # boundary independently: an agent cannot advance a cursor merely by
    # claiming that its discovery command succeeded.
    with pytest.raises((CollectorError, OSError, ValueError)):
        cli._verify_capture_journaled(
            {
                "sources": [
                    {
                        "source_id": "ssrn:quant",
                        "platform": "SSRN",
                        "discovery_only": True,
                        "locator": {"kind": "SSRN_QUERY", "value": "quantitative trading"},
                        "capture_id": "capture:missing",
                        "captured_at": "2026-09-07T12:00:00Z",
                        "window": {"start": "2026-09-05T12:00:00Z", "end": "2026-09-07T12:00:00Z"},
                        "cursor": {"in": None, "out": "cursor", "exhausted": True},
                        "provider_status": "OK",
                        "expected_item_count": 0,
                        "expected_min_items": 0,
                        "observed_item_count": 0,
                    }
                ]
            },
            [],
            tmp_path / "missing-journal.jsonl",
        )
