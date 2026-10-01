"""Pins for the scan_am earnings-calendar pre-step (2026-09-30 incident).

A stale canonical calendar is repaired by the normal producer before the scan's
side-effect boundary, or fails before it (the fallback test pins what that does
and does not buy). No network, R2, or producer process runs here.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import sys

import pandas as pd
import pytest

from earnings_calendar_provider import SCOPE
from scripts import automation_supervisor as sup
from scripts import ensure_earnings_calendar as ensure


def _calendar(as_of: pd.Timestamp, generated: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "date": [as_of + pd.Timedelta(days=3)] * 2,
            "calendar_scope": [SCOPE] * 2,
            "calendar_as_of": [str(as_of.date())] * 2,
            "calendar_generated_at": [generated.isoformat()] * 2,
        }
    )


def _fresh() -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    today = now.tz_convert("America/New_York").tz_localize(None).normalize()
    return _calendar(today, now)


def _stale() -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    today = now.tz_convert("America/New_York").tz_localize(None).normalize()
    return _calendar(today - pd.Timedelta(days=10), now - pd.Timedelta(days=1))


@pytest.fixture
def wired(monkeypatch):
    state = {"frames": [], "fetches": 0, "producer_calls": 0, "producer_rc": 0}

    def fetch(dest):
        state["fetches"] += 1
        frame = state["frames"].pop(0)
        if isinstance(frame, Exception):
            raise frame
        return frame

    def produce(timeout_seconds=ensure.PRODUCER_TIMEOUT_SECONDS):
        state["producer_calls"] += 1
        return state["producer_rc"]

    monkeypatch.setattr(ensure, "fetch_canonical", fetch)
    monkeypatch.setattr(ensure, "run_producer", produce)
    return state


def test_the_incident_calendar_is_stale_for_the_next_morning_scan():
    # 2026-10-01 04:16 ET; the canonical object still carried 2026-09-29.
    frame = _calendar(pd.Timestamp("2026-09-29"), pd.Timestamp("2026-09-29T21:40:00+00:00"))
    problem = ensure.staleness(frame, pd.Timestamp("2026-10-01T08:16:00+00:00"))
    assert problem == "Earnings calendar missed the previous NYSE session; refresh required"
    fresh = _calendar(pd.Timestamp("2026-09-30"), pd.Timestamp("2026-09-30T21:40:00+00:00"))
    assert ensure.staleness(fresh, pd.Timestamp("2026-10-01T08:16:00+00:00")) is None


def test_fresh_calendar_exits_zero_without_running_the_producer(wired, capsys):
    wired["frames"] = [_fresh()]
    assert ensure.main() == 0
    assert wired["producer_calls"] == 0
    assert wired["fetches"] == 1
    assert "fresh; no refresh needed" in capsys.readouterr().out


def test_stale_calendar_repaired_by_the_producer_exits_zero(wired, capsys):
    wired["frames"] = [_stale(), _fresh()]
    assert ensure.main() == 0
    assert wired["producer_calls"] == 1
    assert wired["fetches"] == 2  # re-checks the canonical object, not the producer's word
    assert "repaired" in capsys.readouterr().out


def test_stale_calendar_with_failed_producer_exits_nonzero(wired, capsys):
    wired["frames"] = [_stale(), _stale()]
    wired["producer_rc"] = 1
    assert ensure.main() == 1
    assert wired["producer_calls"] == 1
    out = capsys.readouterr().out
    assert "still stale after producer exit 1" in out
    assert "nothing was staged" in out


def test_producer_success_that_leaves_r2_stale_still_fails(wired):
    wired["frames"] = [_stale(), _stale()]
    assert ensure.main() == 1
    assert wired["producer_calls"] == 1


def test_unavailable_calendar_fails_without_spending_the_alpha_request(wired, capsys):
    wired["frames"] = [ensure.CalendarUnavailable("canonical earnings_calendar.parquet could not be downloaded")]
    assert ensure.main() == 1
    assert wired["producer_calls"] == 0
    assert "check failed" in capsys.readouterr().out


def test_producer_command_is_the_earnings_and_grades_command():
    jobs = {job.id: job for pipeline in sup.CATALOG.values() for job in pipeline.jobs}
    producer = [cmd.argv[1:] for cmd in jobs["earnings_and_grades"].commands]
    assert producer == [ensure.PRODUCER]
    assert ensure.PRODUCER_TIMEOUT_SECONDS < _ensure_step(jobs["scan_am"]).timeout_seconds


# ------------------------------------------------------------- supervisor wiring


def _ensure_step(job: sup.JobSpec) -> sup.CommandSpec:
    return next(cmd for cmd in job.commands if "scripts/ensure_earnings_calendar.py" in cmd.argv)


def test_scan_am_runs_the_pre_step_before_its_side_effect_boundary():
    job = next(job for job in sup.CATALOG["premarket"].jobs if job.id == "scan_am")
    labels = [cmd.label for cmd in job.commands]
    step = _ensure_step(job)
    boundary = next(i for i, cmd in enumerate(job.commands) if cmd.side_effecting)
    assert not step.side_effecting
    assert not job.rerun_safe  # the boundary marker is written for this job
    assert labels.index(step.label) < boundary
    assert labels.index(step.label) == 0  # before the cache pull, which then gets the repaired object
    assert job.depends_on == ("master_prices_am", "risk_am")


def test_pre_step_is_scan_am_only_and_adds_no_catalog_job():
    ids = [job.id for pipeline in sup.CATALOG.values() for job in pipeline.jobs]
    assert "ensure_earnings_calendar" not in " ".join(ids)
    holders = [
        job.id
        for pipeline in sup.CATALOG.values()
        for job in pipeline.jobs
        for cmd in job.commands
        if "scripts/ensure_earnings_calendar.py" in cmd.argv
    ]
    assert holders == ["scan_am"]
    scan_pm = next(job for job in sup.CATALOG["postclose"].jobs if job.id == "scan_pm")
    assert scan_pm.depends_on == ("master_prices_pm", "risk_pm", "earnings_and_grades")


class _Process:
    def __init__(self, failing: str):
        self.failing = failing
        self.calls: list[list[str]] = []

    def stream(self, argv, *, cwd, env, timeout_seconds, logger):
        self.calls.append(list(argv))
        return 1 if self.failing in argv else 0

    def capture(self, argv, *, cwd, env, timeout_seconds):
        raise AssertionError("unexpected capture")


def test_stale_calendar_is_a_retryable_failure_not_indeterminate(tmp_path):
    now = dt.datetime(2026, 10, 1, 8, 16, tzinfo=dt.timezone.utc)
    job = dataclasses.replace(
        next(job for job in sup.CATALOG["premarket"].jobs if job.id == "scan_am"),
        local_gate=None,
        depends_on=(),
    )
    pipeline = dataclasses.replace(sup.CATALOG["premarket"], jobs=(job,))
    store = sup.InMemoryReceiptStore(now=lambda: now)
    process = _Process("scripts/ensure_earnings_calendar.py")
    supervisor = sup.AutomationSupervisor(
        catalog={pipeline.id: pipeline},
        repo_root=tmp_path,
        state_root=tmp_path / "state",
        python_executable=sys.executable,
        env={name: "present" for name in job.required_env},
        receipts=store,
        process=process,
        dispatcher=None,  # type: ignore[arg-type]
        validator=None,
        now=lambda: now,
    )
    with sup.RunLogger(tmp_path / "run.log", echo=False) as logger:
        outcome = supervisor.run_job(
            pipeline, job, run_date="2026-10-01", logger=logger, allow_fallback=False
        )
    assert outcome.status == "failure"
    receipt = store.latest("2026-10-01", "scan_am")
    assert receipt.status == "failure" and receipt.phase == "retryable"
    assert len(process.calls) == 1  # the scanner never started
    log = (tmp_path / "run.log").read_text(encoding="utf-8")
    assert "guard: persist side-effect boundary" not in log
    # Re-runnable only because fallback was disabled here. Production allows the
    # immediate GitHub fallback; see the next test for what happens when it fails.
    assert sup.effective_status(receipt, now) == "failure"


class _AcceptedButFailedDispatcher:
    def __init__(self):
        self.calls: list[str] = []

    def dispatch_and_wait(self, workflow, *, automation_token, logger):
        self.calls.append(workflow.workflow)
        raise sup.DispatchAcceptedError("synthetic: workflow accepted, run failed")


def test_failed_pre_step_plus_failed_github_fallback_is_still_indeterminate(tmp_path):
    # Pins the real production outcome so nobody relies on a guarantee that
    # does not exist: the pre-step avoids the local boundary, but a failed
    # GitHub fallback of non-rerun-safe scan_am ends manual_review, which the
    # 05:45 retry skips.
    now = dt.datetime(2026, 10, 1, 8, 16, tzinfo=dt.timezone.utc)
    job = dataclasses.replace(
        next(job for job in sup.CATALOG["premarket"].jobs if job.id == "scan_am"),
        local_gate=None,
        depends_on=(),
    )
    pipeline = dataclasses.replace(sup.CATALOG["premarket"], jobs=(job,))
    store = sup.InMemoryReceiptStore(now=lambda: now)
    process = _Process("scripts/ensure_earnings_calendar.py")
    dispatcher = _AcceptedButFailedDispatcher()
    supervisor = sup.AutomationSupervisor(
        catalog={pipeline.id: pipeline},
        repo_root=tmp_path,
        state_root=tmp_path / "state",
        python_executable=sys.executable,
        env={name: "present" for name in job.required_env},
        receipts=store,
        process=process,
        dispatcher=dispatcher,  # type: ignore[arg-type]
        validator=None,
        now=lambda: now,
    )
    with sup.RunLogger(tmp_path / "run.log", echo=False) as logger:
        outcome = supervisor.run_job(
            pipeline, job, run_date="2026-10-01", logger=logger, allow_fallback=True
        )
    assert dispatcher.calls == ["daily_screener.yml"]
    assert len(process.calls) == 1  # the local scanner never started
    assert outcome.status == "indeterminate" and outcome.source == "github"
    receipt = store.latest("2026-10-01", "scan_am")
    assert receipt.status == "indeterminate" and receipt.phase == "manual_review"
