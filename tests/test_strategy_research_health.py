import datetime as dt
import json

import pytest

from scripts import repo_health_check as health


@pytest.mark.parametrize("kind", ["missing", "stale", "running", "false_success", "success"])
def test_health_requires_today_verified_research_after_deadline(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(health, "CONFIG_ROOT", tmp_path)
    results = []
    monkeypatch.setattr(health, "report", lambda *args: results.append(args))
    if kind != "missing":
        path = tmp_path / "artifacts/strategy_research_agent/last_run.json"
        path.parent.mkdir(parents=True)
        receipt = {"started_at": "2026-09-08T04:30:00+00:00", "status": "success", "exit_code": 0,
                   "phases": dict(source_collection=0, research_agent=0, completion_check=0)}
        if kind == "stale":
            receipt["started_at"] = "2026-09-07T04:30:00+00:00"
        if kind == "running":
            receipt["status"] = "running"
        if kind == "false_success":
            receipt["phases"].pop("completion_check")
        path.write_text(json.dumps(receipt))
    health.check_strategy_research(dt.datetime(2026, 9, 8, 11, 30, tzinfo=dt.timezone.utc))
    assert results[0][0] == ("OK" if kind == "success" else "FAIL")
