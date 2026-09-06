import datetime as dt
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import repo_health_check as health


def test_journal_health_reads_configured_production_data_root(tmp_path, monkeypatch):
    data_root = tmp_path / "production-config" / "data"
    data_root.mkdir(parents=True)
    for name in ("pitch_journal.jsonl", "context_journal.jsonl", "posts_journal.jsonl"):
        (data_root / name).write_text('{"kind":"ok"}\n', encoding="utf-8")
    monkeypatch.setattr(health, "JOURNAL_DATA_ROOT", data_root)
    health.RESULTS.clear()

    health.check_journals()

    assert health.RESULTS == [
        ("OK", "journal:pitch_journal.jsonl", "parses clean"),
        ("OK", "journal:context_journal.jsonl", "parses clean"),
        ("OK", "journal:posts_journal.jsonl", "parses clean"),
    ]


def test_delivery_health_passes_canonical_production_evidence_paths(
    tmp_path, monkeypatch
):
    data_root = tmp_path / "production-config" / "data"
    monkeypatch.setattr(health, "JOURNAL_DATA_ROOT", data_root)
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((list(argv), kwargs))
        return SimpleNamespace(returncode=0, stdout="OK\n", stderr="")

    monkeypatch.setattr(health.subprocess, "run", fake_run)
    health.RESULTS.clear()

    health.check_delivery()

    assert len(calls) == 2
    expected_journals = {
        "check_pitch_delivered.py": data_root / "pitch_journal.jsonl",
        "check_context_delivered.py": data_root / "context_journal.jsonl",
    }
    for argv, kwargs in calls:
        script_name = Path(argv[1]).name
        journal_index = argv.index("--journal") + 1
        assert Path(argv[journal_index]) == expected_journals[script_name]
        assert kwargs["cwd"] == health.ROOT
        assert kwargs["timeout"] == 120

    pitch_argv = next(
        argv for argv, _kwargs in calls
        if Path(argv[1]).name == "check_pitch_delivered.py"
    )
    pitch_day = pitch_argv[pitch_argv.index("--asof") + 1]
    receipt_index = pitch_argv.index("--delivery-receipt") + 1
    assert Path(pitch_argv[receipt_index]) == (
        data_root / "pitch_delivery_receipts" / f"{pitch_day}.json"
    )
    assert "--require-r2" in pitch_argv
    assert data_root / f"pitch_journal.delivery.{pitch_day}.json" not in map(
        Path, pitch_argv
    )

    context_argv = next(
        argv for argv, _kwargs in calls
        if Path(argv[1]).name == "check_context_delivered.py"
    )
    assert "--delivery-receipt" not in context_argv
    assert "--require-r2" not in context_argv


def test_guard_collection_process_error_fails_loud(monkeypatch):
    health.RESULTS.clear()
    monkeypatch.setattr(
        health.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="No module named pytest",
        ),
    )

    health.check_test_collection()

    assert health.RESULTS == [
        (
            "FAIL",
            "tests:collect",
            "collection process failed (exit 1: No module named pytest); "
            "run pytest --collect-only tests for detail",
        )
    ]


def _receipt(job_id: str, day: dt.date, status: str = "success",
             source: str = "local") -> dict:
    return {
        "schema_version": health.AUTOMATION_RECEIPT_SCHEMA,
        "job_id": job_id,
        "run_date_et": day.isoformat(),
        "status": status,
        "source": source,
        "updated_at_utc": f"{day.isoformat()}T21:00:00+00:00",
    }


def test_automation_health_accepts_local_or_github_success(monkeypatch):
    today = dt.date(2026, 8, 27)
    monkeypatch.setattr(health, "CRITICAL_AUTOMATION_JOBS",
                        {"scan_am": 1, "risk_am": 1})
    health.RESULTS.clear()

    def fetch(job_id, day):
        if day != today:
            return None
        source = "github" if job_id == "risk_am" else "local"
        return _receipt(job_id, day, source=source)

    health.check_gha(fetch=fetch, today=today)

    assert health.RESULTS == [
        ("OK", "automation:scan_am",
         "success via local, 0 bd old (2026-08-27T21:00:00+00:00)"),
        ("OK", "automation:risk_am",
         "success via github, 0 bd old (2026-08-27T21:00:00+00:00)"),
    ]


def test_automation_health_fails_latest_failure(monkeypatch):
    today = dt.date(2026, 8, 27)
    monkeypatch.setattr(health, "CRITICAL_AUTOMATION_JOBS", {"scan_am": 1})
    health.RESULTS.clear()

    health.check_gha(
        fetch=lambda job_id, day: (_receipt(job_id, day, "failure")
                                   if day == today else None),
        today=today,
    )

    assert health.RESULTS[0][0:2] == ("FAIL", "automation:scan_am")
    assert "FAILED via local" in health.RESULTS[0][2]


def test_automation_health_fails_indeterminate_side_effect(monkeypatch):
    today = dt.date(2026, 8, 27)
    monkeypatch.setattr(health, "CRITICAL_AUTOMATION_JOBS", {"scan_am": 1})
    health.RESULTS.clear()
    receipt = _receipt("scan_am", today, source="local")
    receipt.update(
        status="indeterminate",
        detail="Sheets write completed but readback timed out",
    )

    health.check_gha(
        fetch=lambda job_id, day: receipt if day == today else None,
        today=today,
    )

    assert health.RESULTS[0][0:2] == ("FAIL", "automation:scan_am")
    assert "INDETERMINATE" in health.RESULTS[0][2]
    assert "manual resolution required" in health.RESULTS[0][2]


def test_automation_health_fails_stale_success(monkeypatch):
    today = dt.date(2026, 8, 27)
    old_day = dt.date(2026, 8, 24)
    # The production-wide lookup horizon is set by the longest allowance,
    # while the scan's own one-session freshness rule still decides status.
    monkeypatch.setattr(
        health,
        "CRITICAL_AUTOMATION_JOBS",
        {"scan_am": 1, "weekly_rundown": 8},
    )
    health.RESULTS.clear()

    def fetch(job_id, day):
        if job_id == "scan_am" and day == old_day:
            return _receipt(job_id, day)
        if job_id == "weekly_rundown" and day == today:
            return _receipt(job_id, day)
        return None

    health.check_gha(fetch=fetch, today=today)

    assert health.RESULTS[0][0:2] == ("FAIL", "automation:scan_am")
    assert "3 bd old" in health.RESULTS[0][2]


def test_receipt_lookup_honors_eight_nyse_sessions_across_long_calendar_gap(
    monkeypatch,
):
    today = dt.date(2026, 1, 5)
    receipt_day = dt.date(2025, 12, 22)
    assert (today - receipt_day).days == 14
    assert health.bdays_behind(receipt_day, today) == 8
    assert health.receipt_search_horizon_days(today, 8) >= 14

    monkeypatch.setattr(
        health, "CRITICAL_AUTOMATION_JOBS", {"weekly_rundown": 8}
    )
    health.RESULTS.clear()
    requested = []

    def fetch(job_id, day):
        requested.append(day)
        return _receipt(job_id, day) if day == receipt_day else None

    health.check_gha(fetch=fetch, today=today)

    assert receipt_day in requested
    assert health.RESULTS == [
        (
            "OK",
            "automation:weekly_rundown",
            "success via local, 8 bd old (2025-12-22T21:00:00+00:00)",
        )
    ]


def test_receipt_horizon_is_derived_from_maximum_allowed_age(monkeypatch):
    today = dt.date(2026, 1, 5)
    monkeypatch.setattr(
        health,
        "CRITICAL_AUTOMATION_JOBS",
        {"daily": 1, "weekly": 8},
    )
    requested = {"daily": [], "weekly": []}
    health.RESULTS.clear()

    def fetch(job_id, day):
        requested[job_id].append(day)
        return None

    health.check_gha(fetch=fetch, today=today)

    assert len(requested["weekly"]) == len(requested["daily"])
    assert len(requested["weekly"]) == (
        health.receipt_search_horizon_days(today, 8) + 1
    )
    assert "8-NYSE-session search horizon" in health.RESULTS[1][2]


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (dt.date(2026, 4, 2), dt.date(2026, 4, 6), 1),
        (dt.date(2026, 9, 4), dt.date(2026, 9, 8), 1),
        (dt.date(2026, 12, 31), dt.date(2027, 1, 4), 1),
        (dt.date(2026, 8, 28), dt.date(2026, 8, 31), 1),
    ],
)
def test_business_day_age_uses_project_nyse_calendar(start, end, expected):
    assert health.bdays_behind(start, end) == expected


def test_trigger_health_reads_pinned_runtime_log_tree(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs" / "2026-08-27"
    log_dir.mkdir(parents=True)
    (log_dir / "premarket-a1b2c3d4.log").write_text("ok\n", encoding="utf-8")
    monkeypatch.setattr(health, "AUTOMATION_LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(health, "LOCAL_PIPELINE_MAX_BD", {"premarket": 2})
    health.RESULTS.clear()

    health.check_trigger_logs()

    assert health.RESULTS[0][0:2] == ("OK", "triggers:premarket")
