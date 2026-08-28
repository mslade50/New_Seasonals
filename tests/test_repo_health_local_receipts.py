import datetime as dt

from scripts import repo_health_check as health


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
    monkeypatch.setattr(health, "CRITICAL_AUTOMATION_JOBS", {"scan_am": 1})
    health.RESULTS.clear()

    health.check_gha(
        fetch=lambda job_id, day: (_receipt(job_id, day)
                                   if day == old_day else None),
        today=today,
    )

    assert health.RESULTS[0][0:2] == ("FAIL", "automation:scan_am")
    assert "3 bd old" in health.RESULTS[0][2]


def test_trigger_health_reads_pinned_runtime_log_tree(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs" / "2026-08-27"
    log_dir.mkdir(parents=True)
    (log_dir / "premarket-a1b2c3d4.log").write_text("ok\n", encoding="utf-8")
    monkeypatch.setattr(health, "AUTOMATION_LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(health, "LOCAL_PIPELINE_MAX_BD", {"premarket": 2})
    health.RESULTS.clear()

    health.check_trigger_logs()

    assert health.RESULTS[0][0:2] == ("OK", "triggers:premarket")
