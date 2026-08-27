import argparse
import json
from pathlib import Path

from research.experiment_registry import load_records
from research.idea_miner.models import SourceRecord, dedupe_sources, load_source_files
from research.idea_miner.pipeline import build_weekly_queue
from research.idea_miner.report import render_weekly_inbox
from scripts.run_weekly_idea_miner import run


def row(number: int, archetype: str = "intraday", **overrides):
    keyword = {
        "intraday": "intraday opening return",
        "trend": "time-series trend momentum",
        "event": "earnings announcement event",
        "fundamental": "revenue margin estimate",
        "portfolio": "portfolio correlation risk",
        "other": "market observation",
    }[archetype]
    payload = {
        "source_type": "ssrn" if number % 2 else "x",
        "url": f"https://example.org/source/{number}",
        "title": f"Source {number}: {keyword}",
        "text": f"A testable claim about {keyword}.",
        "published_at": "2026-08-25T12:00:00Z",
        "retrieved_at": "2026-08-27T12:00:00Z",
        "claim": f"Claim {number} for {keyword}",
        "mechanism": "Slow capital and benchmarked flows create delayed adjustment.",
        "instruments": ["US equities"],
        "horizon": "same trading day",
        "test_idea": "Fix the decision time, enter next bar, exit close, and test held-out years.",
        "first_rejection": "Reject if the effect is absent before costs in a held-out period.",
        "data_requirements": ["point-in-time 15-minute OHLCV"],
    }
    payload.update(overrides)
    return payload


def _write_sources(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(item) for item in rows) + "\n", encoding="utf-8")


def _find_forbidden_keys(value):
    forbidden = {"quantity", "shares", "order_type", "target_weight", "approve"}
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in forbidden:
                found.append(str(key).lower())
            found.extend(_find_forbidden_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_find_forbidden_keys(child))
    return found


def test_source_loading_and_deduplication(tmp_path):
    source_path = tmp_path / "sources.jsonl"
    duplicate = row(1, url="https://example.org/source/1/")
    _write_sources(source_path, [row(1), duplicate, row(2, "trend")])
    loaded = load_source_files([source_path])
    unique, duplicate_count = dedupe_sources(loaded)
    assert len(loaded) == 3
    assert len(unique) == 2
    assert duplicate_count == 1


def test_queue_is_deterministic_bounded_and_archetype_balanced():
    records = [
        SourceRecord.from_mapping(row(1, "intraday")),
        SourceRecord.from_mapping(row(2, "intraday")),
        SourceRecord.from_mapping(row(3, "intraday")),
        SourceRecord.from_mapping(row(4, "trend")),
        SourceRecord.from_mapping(row(5, "event")),
        SourceRecord.from_mapping(row(6, "portfolio")),
    ]
    first = build_weekly_queue(records, as_of="2026-08-27", max_candidates=5, max_per_archetype=2)
    second = build_weekly_queue(reversed(records), as_of="2026-08-27", max_candidates=5, max_per_archetype=2)
    assert first["selected"] == second["selected"]
    assert len(first["selected"]) == 5
    lanes = [card["archetype"] for card in first["selected"]]
    assert lanes.count("intraday") <= 2
    assert not _find_forbidden_keys(first)
    assert first["research_only"] is True and first["no_order"] is True


def test_prior_hypothesis_is_suppressed_from_review():
    record = SourceRecord.from_mapping(row(1))
    initial = build_weekly_queue([record], as_of="2026-08-20")
    fingerprint = initial["all_hypotheses"][0]["hypothesis_fingerprint"]
    repeated = build_weekly_queue([record], as_of="2026-08-27", prior_fingerprints={fingerprint})
    assert repeated["selected"] == []
    assert repeated["funnel"]["duplicate_prior"] == 1


def test_missing_mechanism_is_a_blocker_not_a_fabricated_answer():
    record = SourceRecord.from_mapping(
        row(1, mechanism="", instruments=[], horizon="", test_idea="")
    )
    queue = build_weekly_queue([record], as_of="2026-08-27")
    card = queue["all_hypotheses"][0]
    assert card["status"] == "needs_source_diligence"
    assert card["mechanism"].startswith("Unresolved:")
    assert card["instruments"] == ["UNRESOLVED"]


def test_html_leads_with_guardrail_and_direct_source_links():
    queue = build_weekly_queue(
        [SourceRecord.from_mapping(row(1))], as_of="2026-08-27"
    )
    rendered = render_weekly_inbox(queue)
    assert "No production path" in rendered
    assert "not an investment recommendation" in rendered
    assert "https://example.org/source/1" in rendered
    assert "Falsifiable test" in rendered


def test_cli_is_dry_by_default_and_writes_only_to_artifacts(tmp_path):
    source_path = tmp_path / "sources.jsonl"
    _write_sources(source_path, [row(1), row(2, "trend")])
    output_dir = tmp_path / "artifacts" / "weekly"
    registry = tmp_path / "artifacts" / "registry" / "experiments.jsonl"

    dry = run(
        argparse.Namespace(
            input=[str(source_path)],
            output_dir=str(output_dir),
            registry=str(registry),
            as_of="2026-08-27",
            max_candidates=5,
            max_per_archetype=2,
            write=False,
        )
    )
    assert dry["write_requested"] is False
    assert not output_dir.exists()
    assert not registry.exists()

    written = run(
        argparse.Namespace(
            input=[str(source_path)],
            output_dir=str(output_dir),
            registry=str(registry),
            as_of="2026-08-27",
            max_candidates=5,
            max_per_archetype=2,
            write=True,
        )
    )
    assert Path(written["outputs"]["report_html"]).exists()
    assert Path(written["outputs"]["manifest"]).exists()
    records = load_records(registry)
    assert len(records) == 4  # two frozen sources + two hypotheses
    assert all(record["research_only"] and record["no_order"] for record in records)


def test_cli_rejects_non_artifact_destinations(tmp_path):
    source_path = tmp_path / "sources.jsonl"
    _write_sources(source_path, [row(1)])
    args = argparse.Namespace(
        input=[str(source_path)],
        output_dir=str(tmp_path / "output"),
        registry=None,
        as_of="2026-08-27",
        max_candidates=5,
        max_per_archetype=2,
        write=False,
    )
    try:
        run(args)
    except ValueError as exc:
        assert "artifacts" in str(exc)
    else:
        raise AssertionError("non-artifact output path should fail closed")

