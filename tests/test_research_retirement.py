"""Retired research cannot issue requests, publish, or block earnings."""
from scripts import automation_supervisor as sup
from scripts import build_analyst_grades as grades


def test_discretionary_catalog_has_no_work():
    assert sup.build_catalog()["discretionary"].jobs == ()


def test_earnings_job_keeps_earnings_but_no_grades():
    job = next(j for j in sup.build_catalog()["postclose"].jobs if j.id == "earnings_and_grades")
    assert len(job.commands) == 1
    assert "earnings_calendar.py" in str(job.commands[0])
    assert "analyst" not in str(job.commands)
    assert len(job.outputs) == 1
    assert "earnings_calendar.parquet" in str(job.outputs)


def test_retired_grade_entry_point_never_loads_credentials_or_fetches(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("Retired collector attempted work")
    monkeypatch.setattr(grades, "load_env", forbidden)
    monkeypatch.setattr(grades, "build_grades", forbidden)
    grades.main()
    assert "retired" in capsys.readouterr().out
