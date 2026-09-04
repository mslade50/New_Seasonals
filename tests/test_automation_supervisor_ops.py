"""Pins for the 2026-09-04 ops fixes (plan D11, brief build_ops_supervisor).

Covers, against the receipt state table in the supervisor's module docstring:
the 05:45 local second chance (``run-pipeline --retry``), the read-side
``expired`` normalisation, the invocation-scoped exit code of the controller
and the retry, and ``resolve`` naming what is due afterwards.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import sys

import pytest

from scripts import automation_supervisor as sup

UTC = dt.timezone.utc


class FakeProcess:
    def __init__(self, codes=None):
        self.codes = list(codes or [])
        self.stream_calls = []
        self.capture_calls = []

    def stream(self, argv, *, cwd, env, timeout_seconds, logger):
        self.stream_calls.append({"argv": list(argv), "env": dict(env)})
        return self.codes.pop(0) if self.codes else 0

    def capture(self, argv, *, cwd, env, timeout_seconds):
        self.capture_calls.append(list(argv))
        raise AssertionError(f"unexpected capture call: {argv}")


class FakeDispatcher:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.calls = []
        self.reconcile_calls = []

    def _run(self, automation_token):
        return sup.GithubRun(
            database_id=123,
            status="completed",
            conclusion="success",
            title=f"automation {automation_token}",
            url="https://example.test/run/123",
        )

    def dispatch_and_wait(self, workflow, *, automation_token, logger):
        self.calls.append((workflow.workflow, automation_token))
        if self.fail:
            raise sup.DispatchError("synthetic dispatch failure")
        return self._run(automation_token)

    def reconcile_and_wait(self, workflow, *, automation_token, logger):
        self.reconcile_calls.append((workflow.workflow, automation_token))
        return self._run(automation_token)


class NoopValidator:
    def validate(self, outputs, *, repo_root, started_at_utc, logger):
        return None


def _receipt(
    job_id,
    status,
    *,
    token="prior",
    source="local",
    pipeline="test",
    run_date="2026-08-27",
    phase=None,
    lease_expires_at_utc=None,
    detail=None,
):
    return sup.Receipt(
        schema_version="automation-receipt.v1",
        pipeline=pipeline,
        job_id=job_id,
        run_date_et=run_date,
        status=status,
        source=source,
        automation_token=token,
        started_at_utc="2026-08-27T10:00:00+00:00",
        updated_at_utc="2026-08-27T10:01:00+00:00",
        phase=phase,
        lease_expires_at_utc=lease_expires_at_utc,
        detail=detail,
    )


NOW = dt.datetime(2026, 8, 27, 12, 0, tzinfo=UTC)


def _local_job(job_id, *, depends_on=(), rerun_safe=False, side_effecting=True):
    return sup.JobSpec(
        id=job_id,
        description=job_id,
        commands=(
            sup.CommandSpec("pull", ("{python}", "pull.py")),
            sup.CommandSpec("effect", ("{python}", "effect.py"), side_effecting=side_effecting),
        ),
        workflow=sup.WorkflowSpec(f"{job_id}.yml"),
        required_env=("NEEDED",),
        rerun_safe=rerun_safe,
        depends_on=tuple(depends_on),
    )


def _pipeline(*jobs):
    return sup.PipelineSpec(
        "test", "test", "weekdays", dt.time(1), dt.time(2), dt.time(3), tuple(jobs)
    )


def _supervisor(tmp_path, pipeline, process, dispatcher, receipts, *, catalog=None, now=NOW):
    return sup.AutomationSupervisor(
        catalog=catalog or {pipeline.id: pipeline},
        repo_root=tmp_path,
        state_root=tmp_path / "state",
        python_executable=sys.executable,
        env={"NEEDED": "present", "LOCAL_AUTOMATION_STRICT": "1"},
        receipts=receipts,
        process=process,
        dispatcher=dispatcher,
        validator=NoopValidator(),
        now=lambda: now,
    )


def _run(supervisor, tmp_path, pipeline_id="test", run_date="2026-08-27"):
    log_path = tmp_path / "run.log"
    with sup.RunLogger(log_path, echo=False) as logger:
        outcomes = supervisor.run_pipeline(
            pipeline_id, run_date=run_date, allow_fallback=True, logger=logger
        )
    return outcomes, log_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------- retry window


def test_premarket_declares_a_bounded_local_retry_window():
    premarket = sup.CATALOG["premarket"]
    assert premarket.retry_at_et == dt.time(5, 45)
    assert premarket.retry_until_et == dt.time(7, 0)
    # Inside the GitHub fallback window on purpose (receipt CAS arbitrates).
    assert premarket.fallback_at_et <= premarket.retry_at_et <= premarket.fallback_until_et
    friday = dt.date(2026, 9, 4)
    saturday = dt.date(2026, 9, 5)
    assert premarket.retry_is_due(dt.datetime.combine(friday, dt.time(5, 45), tzinfo=sup.ET))
    assert premarket.retry_is_due(dt.datetime.combine(friday, dt.time(6, 59, 59), tzinfo=sup.ET))
    assert not premarket.retry_is_due(dt.datetime.combine(friday, dt.time(7, 0), tzinfo=sup.ET))
    assert not premarket.retry_is_due(dt.datetime.combine(friday, dt.time(5, 44, 59), tzinfo=sup.ET))
    assert not premarket.retry_is_due(dt.datetime.combine(saturday, dt.time(5, 45), tzinfo=sup.ET))
    # Every other pipeline has no local retry and never reports due.
    for pipeline_id, pipeline in sup.CATALOG.items():
        if pipeline_id == "premarket":
            continue
        assert pipeline.retry_at_et is None
        assert not pipeline.retry_is_due(dt.datetime.combine(friday, dt.time(5, 45), tzinfo=sup.ET))


# ---------------------------------------------------------------- state table


def test_rerun_is_a_noop_for_every_job_holding_a_success_receipt(tmp_path):
    site = sup.JobSpec(
        id="site",
        description="site",
        workflow=sup.WorkflowSpec("site.yml"),
        dispatch_only=True,
        rerun_safe=True,
        depends_on=("effect_b",),
    )
    pipeline = _pipeline(
        _local_job("effect_a"), _local_job("effect_b", depends_on=("effect_a",)), site
    )
    store = sup.InMemoryReceiptStore(now=lambda: NOW)
    for job_id, source in (("effect_a", "local"), ("effect_b", "operator"), ("site", "github")):
        assert store.claim(_receipt(job_id, "success", token=f"{job_id}-done", source=source, phase="completed"))
    events_before = len(store.events)
    latest_before = dict(store.latest_values)
    process = FakeProcess()
    dispatcher = FakeDispatcher()

    outcomes, log = _run(_supervisor(tmp_path, pipeline, process, dispatcher, store), tmp_path)

    assert [o.status for o in outcomes] == ["success", "success", "success"]
    assert all(o.preexisting and o.detail == "existing receipt" for o in outcomes)
    assert process.stream_calls == []
    assert dispatcher.calls == [] and dispatcher.reconcile_calls == []
    assert len(store.events) == events_before
    assert store.latest_values == latest_before
    assert log.count("skip ") == 3


def test_rerun_reclaims_an_expired_pre_side_effect_lease_locally(tmp_path):
    # Exactly the 2026-09-03 scan_am shape (incident Appendix B): one local
    # claim, phase local_pre_side_effect, lease passed, no later event.
    pipeline = _pipeline(_local_job("scan_am"))
    store = sup.InMemoryReceiptStore(now=lambda: NOW)
    stalled = _receipt(
        "scan_am",
        "running",
        token="2026-08-27-scan_am-ca1f8cf470",
        phase="local_pre_side_effect",
        lease_expires_at_utc="2026-08-27T11:59:59+00:00",
    )
    assert store.claim(stalled)
    process = FakeProcess()
    dispatcher = FakeDispatcher()

    outcomes, log = _run(_supervisor(tmp_path, pipeline, process, dispatcher, store), tmp_path)

    assert outcomes[0].status == "success" and not outcomes[0].preexisting
    assert len(process.stream_calls) == 2  # pull + effect ran again
    assert dispatcher.calls == []
    latest = store.latest("2026-08-27", "scan_am")
    assert latest.status == "success" and latest.source == "local"
    assert latest.automation_token != stalled.automation_token
    assert (
        "expired scan_am: local_pre_side_effect lease from local expired at "
        "2026-08-27T11:59:59+00:00 token=2026-08-27-scan_am-ca1f8cf470; reclaiming"
    ) in log


def test_rerun_skips_a_live_pre_side_effect_lease(tmp_path):
    pipeline = _pipeline(_local_job("scan_am"))
    store = sup.InMemoryReceiptStore(now=lambda: NOW)
    live = _receipt(
        "scan_am",
        "running",
        phase="local_pre_side_effect",
        lease_expires_at_utc="2026-08-27T12:00:01+00:00",
    )
    assert store.claim(live)
    process = FakeProcess()

    outcomes, _ = _run(_supervisor(tmp_path, pipeline, process, FakeDispatcher(), store), tmp_path)

    assert outcomes[0].status == "running" and outcomes[0].detail == "live lease"
    assert process.stream_calls == []
    assert store.latest("2026-08-27", "scan_am") == live


@pytest.mark.parametrize(
    "phase, source, detail",
    [
        ("manual_review", "local", "AutomationError: send execution report exited 1"),
        ("local_side_effect", "local", "side-effecting step started: run unified scanner"),
        ("manual_review", "github", "DispatchAcceptedError: GitHub fallback concluded cancelled"),
    ],
)
def test_rerun_never_touches_an_indeterminate_or_post_side_effect_receipt(
    tmp_path, phase, source, detail
):
    pipeline = _pipeline(_local_job("scan_am"), _local_job("site", depends_on=("scan_am",)))
    store = sup.InMemoryReceiptStore(now=lambda: NOW)
    marker = _receipt("scan_am", "indeterminate", phase=phase, source=source, detail=detail)
    assert store.claim(marker)
    events_before = len(store.events)
    process = FakeProcess()
    dispatcher = FakeDispatcher()

    outcomes, log = _run(_supervisor(tmp_path, pipeline, process, dispatcher, store), tmp_path)

    assert outcomes[0].status == "indeterminate" and outcomes[0].preexisting
    assert outcomes[1].status == "blocked" and outcomes[1].preexisting
    assert process.stream_calls == []
    assert dispatcher.calls == [] and dispatcher.reconcile_calls == []
    assert len(store.events) == events_before
    assert store.latest("2026-08-27", "scan_am") == marker
    assert "skip scan_am: indeterminate receipt from" in log
    assert "never re-run automatically" in log
    assert "resolve --pipeline test --job scan_am --date 2026-08-27" in log


def test_blocked_by_a_fresh_failure_is_not_preexisting(tmp_path):
    pipeline = _pipeline(_local_job("scan_am"), _local_job("site", depends_on=("scan_am",)))
    store = sup.InMemoryReceiptStore(now=lambda: NOW)
    process = FakeProcess([1])  # pull fails before the side effect
    dispatcher = FakeDispatcher(fail=True)

    outcomes, _ = _run(_supervisor(tmp_path, pipeline, process, dispatcher, store), tmp_path)

    assert outcomes[0].status == "failure" and not outcomes[0].preexisting
    assert outcomes[1].status == "blocked" and not outcomes[1].preexisting


# ---------------------------------------------------------------- expired normalisation


def test_effective_status_reports_expired_without_rewriting_the_receipt(monkeypatch, capsys):
    now = dt.datetime(2026, 9, 3, 11, 30, 30, tzinfo=UTC)  # 07:30:30 ET, the v7 health run
    stalled = sup.dataclasses.replace(
        _receipt(
            "scan_am",
            "running",
            token="2026-09-03-scan_am-ca1f8cf470",
            pipeline="premarket",
            phase="local_pre_side_effect",
            lease_expires_at_utc="2026-09-03T08:44:22.193402+00:00",
        ),
        run_date_et="2026-09-03",
        started_at_utc="2026-09-03T08:14:22.193402+00:00",
        updated_at_utc="2026-09-03T08:14:22.193402+00:00",
    )
    assert sup.effective_status(None, now) == "missing"
    assert sup.effective_status(stalled, now) == "expired"
    assert sup.effective_status(stalled, dt.datetime(2026, 9, 3, 8, 30, tzinfo=UTC)) == "running"
    assert sup.effective_status(
        sup.dataclasses.replace(stalled, status="success", phase="completed"), now
    ) == "success"

    store = sup.InMemoryReceiptStore(now=lambda: now)
    assert store.claim(stalled)
    events_before = list(store.events)
    monkeypatch.setattr(sup, "_utc_now", lambda: now)
    monkeypatch.setattr(sup, "_make_runtime", lambda args, controller_only: (object(), store))

    assert sup.main(["status", "premarket", "--date", "2026-09-03"]) == 0
    text = capsys.readouterr().out
    assert "scan_am                      expired  local" in text

    assert sup.main(["status", "premarket", "--date", "2026-09-03", "--json"]) == 0
    rows = {row["job_id"]: row for row in json.loads(capsys.readouterr().out)}
    assert rows["scan_am"]["status"] == "expired"
    assert rows["scan_am"]["receipt_status"] == "running"
    assert rows["scan_am"]["phase"] == "local_pre_side_effect"
    assert rows["cboe_am"]["status"] == "missing"
    # Read-only: the receipt and its event trail are byte-identical afterwards.
    assert store.latest("2026-09-03", "scan_am") == stalled
    assert store.events == events_before


# ---------------------------------------------------------------- exit code scoping


def _controller_runtime(tmp_path, store, dispatcher, now):
    supervisor = sup.AutomationSupervisor(
        catalog=sup.CATALOG,
        repo_root=tmp_path,
        state_root=tmp_path / "state",
        python_executable=sys.executable,
        env={},
        receipts=store,
        process=FakeProcess(),
        dispatcher=dispatcher,
        validator=NoopValidator(),
        now=lambda: now,
    )
    return supervisor, store


def test_fallback_due_exit_code_ignores_a_preexisting_indeterminate_receipt(tmp_path, monkeypatch):
    # 2026-09-02 21:20Z = 17:20 ET, inside the execution fallback window
    # (16:50-20:00 ET). The 09-02 execution_report receipt was indeterminate
    # from 16:30 ET; every later controller tick exited 1 while doing nothing.
    now = dt.datetime(2026, 9, 2, 21, 20, tzinfo=UTC)
    store = sup.InMemoryReceiptStore(now=lambda: now)
    stuck = sup.dataclasses.replace(
        _receipt(
            "execution_report",
            "indeterminate",
            pipeline="execution",
            phase="manual_review",
            detail="AutomationError: send execution report exited 1",
        ),
        run_date_et="2026-09-02",
    )
    assert store.claim(stuck)
    dispatcher = FakeDispatcher()
    monkeypatch.setattr(sup, "_utc_now", lambda: now)
    monkeypatch.setattr(
        sup, "_make_runtime", lambda args, controller_only: _controller_runtime(tmp_path, store, dispatcher, now)
    )

    rc = sup.main(
        [
            "fallback-due",
            "--pipeline",
            "execution",
            "--state-root",
            str(tmp_path / "state"),
        ]
    )

    assert rc == 0
    assert dispatcher.calls == []
    assert store.latest("2026-09-02", "execution_report") == stuck
    logs = list((tmp_path / "state" / "logs" / "2026-09-02").glob("execution-*.log"))
    assert len(logs) == 1
    text = logs[0].read_text(encoding="utf-8")
    assert "mode=fallback-due" in text
    assert (
        "reported, not counted against this fallback-due run for 2026-09-02: "
        "execution/execution_report (indeterminate) (pre-existing receipts; "
        "operator resolution required)"
    ) in text


def test_fallback_due_exit_code_still_fails_on_its_own_dispatch_failure(tmp_path, monkeypatch):
    now = dt.datetime(2026, 9, 2, 21, 20, tzinfo=UTC)
    store = sup.InMemoryReceiptStore(now=lambda: now)
    dispatcher = FakeDispatcher(fail=True)
    monkeypatch.setattr(sup, "_utc_now", lambda: now)
    monkeypatch.setattr(
        sup, "_make_runtime", lambda args, controller_only: _controller_runtime(tmp_path, store, dispatcher, now)
    )

    rc = sup.main(
        ["fallback-due", "--pipeline", "execution", "--state-root", str(tmp_path / "state")]
    )

    assert rc == 1
    assert len(dispatcher.calls) == 1
    assert store.latest("2026-09-02", "execution_report").status == "failure"


def test_fallback_due_blocked_only_by_preexisting_receipts_is_not_fatal(tmp_path, monkeypatch):
    # The 2026-09-03 07:46 tick wrote scan_am indeterminate itself (fatal for
    # THAT tick). A later in-window tick finds it pre-existing: the two site
    # deploys are blocked by it, and that is reported, not counted.
    now = dt.datetime(2026, 9, 3, 12, 30, tzinfo=UTC)  # 08:30 ET
    store = sup.InMemoryReceiptStore(now=lambda: now)
    for job_id in ("cboe_am", "master_prices_am", "risk_am", "event_sleeve_am"):
        assert store.claim(
            sup.dataclasses.replace(
                _receipt(job_id, "success", pipeline="premarket", phase="completed", token=f"{job_id}-ok"),
                run_date_et="2026-09-03",
            )
        )
    assert store.claim(
        sup.dataclasses.replace(
            _receipt(
                "scan_am",
                "indeterminate",
                pipeline="premarket",
                source="github",
                phase="manual_review",
                token="2026-09-03-scan_am-d7f138ad30",
            ),
            run_date_et="2026-09-03",
        )
    )
    dispatcher = FakeDispatcher()
    monkeypatch.setattr(sup, "_utc_now", lambda: now)
    monkeypatch.setattr(
        sup, "_make_runtime", lambda args, controller_only: _controller_runtime(tmp_path, store, dispatcher, now)
    )

    rc = sup.main(
        ["fallback-due", "--pipeline", "premarket", "--state-root", str(tmp_path / "state")]
    )

    assert rc == 0
    assert dispatcher.calls == []
    text = next((tmp_path / "state" / "logs" / "2026-09-03").glob("premarket-*.log")).read_text(
        encoding="utf-8"
    )
    assert "premarket/scan_am (indeterminate)" in text
    assert "premarket/private_site_am (blocked)" in text
    assert "premarket/shared_site_am (blocked)" in text


# ---------------------------------------------------------------- run-pipeline --retry


def _premarket_all_success(now, run_date="2026-09-04"):
    store = sup.InMemoryReceiptStore(now=lambda: now)
    for job in sup.CATALOG["premarket"].jobs:
        source = "github" if job.dispatch_only else "local"
        assert store.claim(
            sup.dataclasses.replace(
                _receipt(job.id, "success", pipeline="premarket", phase="completed", source=source, token=f"{job.id}-ok"),
                run_date_et=run_date,
            )
        )
    return store


def test_retry_flag_is_gated_by_the_retry_window_and_noops_on_success(tmp_path, monkeypatch, capsys):
    friday_0545_et = dt.datetime(2026, 9, 4, 9, 45, tzinfo=UTC)  # 05:45 EDT
    store = _premarket_all_success(friday_0545_et)
    events_before = list(store.events)
    dispatcher = FakeDispatcher()
    process = FakeProcess()

    def runtime(args, controller_only=False):
        supervisor = sup.AutomationSupervisor(
            catalog=sup.CATALOG,
            repo_root=tmp_path,
            state_root=tmp_path / "state",
            python_executable=sys.executable,
            env={},
            receipts=store,
            process=process,
            dispatcher=dispatcher,
            validator=NoopValidator(),
            now=lambda: friday_0545_et,
        )
        return supervisor, store

    monkeypatch.setattr(sup, "_utc_now", lambda: friday_0545_et)
    monkeypatch.setattr(sup, "_make_runtime", runtime)
    argv = [
        "run-pipeline",
        "--pipeline",
        "premarket",
        "--retry",
        "--config-root",
        str(tmp_path),
        "--state-root",
        str(tmp_path / "state"),
    ]

    assert sup.main(argv) == 0
    assert process.stream_calls == [] and dispatcher.calls == []
    assert store.events == events_before
    text = next((tmp_path / "state" / "logs" / "2026-09-04").glob("premarket-*.log")).read_text(
        encoding="utf-8"
    )
    assert "mode=retry" in text
    assert text.count("skip ") == len(sup.CATALOG["premarket"].jobs)

    # The plain primary entry point at 05:45 is outside 04:10-05:20: unchanged.
    assert sup.main([a for a in argv if a != "--retry"]) == 0
    assert "outside its local ET window 04:10-05:20" in capsys.readouterr().out

    # Past 07:00 ET the retry replays as No action, before any runtime is built.
    monkeypatch.setattr(sup, "_utc_now", lambda: dt.datetime(2026, 9, 4, 11, 30, tzinfo=UTC))
    monkeypatch.setattr(sup, "_make_runtime", lambda *a, **k: pytest.fail("runtime must not be built"))
    assert sup.main(argv) == 0
    assert "outside its local retry ET window 05:45-07:00" in capsys.readouterr().out


def test_retry_refuses_a_pipeline_without_a_retry_window(monkeypatch, tmp_path):
    monkeypatch.setattr(sup, "_utc_now", lambda: dt.datetime(2026, 9, 4, 9, 30, tzinfo=UTC))
    with pytest.raises(sup.AutomationError, match="no local retry window"):
        sup.main(["run-pipeline", "--pipeline", "execution", "--retry", "--config-root", str(tmp_path)])


def test_retry_reports_but_does_not_fail_on_a_preexisting_indeterminate(tmp_path, monkeypatch):
    friday_0545_et = dt.datetime(2026, 9, 4, 9, 45, tzinfo=UTC)
    store = _premarket_all_success(friday_0545_et)
    key = ("2026-09-04", "scan_am")
    store.latest_values[key] = sup.dataclasses.replace(
        store.latest_values[key], status="indeterminate", phase="local_side_effect"
    )
    process = FakeProcess()
    dispatcher = FakeDispatcher()
    monkeypatch.setattr(sup, "_utc_now", lambda: friday_0545_et)
    monkeypatch.setattr(
        sup,
        "_make_runtime",
        lambda args, controller_only=False: _controller_runtime(tmp_path, store, dispatcher, friday_0545_et),
    )
    argv = [
        "run-pipeline", "--pipeline", "premarket", "--config-root", str(tmp_path),
        "--state-root", str(tmp_path / "state"),
    ]

    assert sup.main([*argv, "--retry"]) == 0
    assert process.stream_calls == [] and dispatcher.calls == []
    assert store.latest(*key).status == "indeterminate"

    # The primary (non-retry) entry point keeps the conservative red: an
    # indeterminate receipt in its own window is its problem.
    monkeypatch.setattr(sup, "_utc_now", lambda: dt.datetime(2026, 9, 4, 8, 10, tzinfo=UTC))
    assert sup.main(argv) == 1


def test_retry_exits_zero_when_the_primary_still_holds_the_lock(tmp_path, monkeypatch, capsys):
    friday_0545_et = dt.datetime(2026, 9, 4, 9, 45, tzinfo=UTC)
    state = tmp_path / "state"
    monkeypatch.setattr(sup, "_utc_now", lambda: friday_0545_et)
    monkeypatch.setattr(
        sup,
        "_make_runtime",
        lambda args, controller_only=False: (object(), sup.InMemoryReceiptStore()),
    )
    with sup.GlobalFileLock(state / "automation_supervisor.lock", timeout_seconds=0):
        rc = sup.main(
            ["run-pipeline", "--pipeline", "premarket", "--retry", "--config-root", str(tmp_path), "--state-root", str(state)]
        )
    assert rc == 0
    assert "No action: another automation supervisor holds" in capsys.readouterr().out


# ---------------------------------------------------------------- resolve prints what is due


def test_resolve_success_prints_dependents_and_exact_fallback_commands(tmp_path, monkeypatch, capsys):
    now = dt.datetime(2026, 9, 3, 11, 47, tzinfo=UTC)  # 07:47 ET, the real resolution time
    store = sup.InMemoryReceiptStore(now=lambda: now)
    assert store.claim(
        sup.dataclasses.replace(
            _receipt(
                "scan_am",
                "indeterminate",
                pipeline="premarket",
                source="github",
                phase="manual_review",
                token="2026-09-03-scan_am-d7f138ad30",
            ),
            run_date_et="2026-09-03",
        )
    )
    monkeypatch.setattr(sup, "_utc_now", lambda: now)
    monkeypatch.setattr(sup, "_make_runtime", lambda args, controller_only: (object(), store))

    rc = sup.main(
        [
            "resolve", "--pipeline", "premarket", "--job", "scan_am", "--date", "2026-09-03",
            "--disposition", "success", "--reason", "verified recovery run",
            "--config-root", "C:/cfg", "--ref", "automation-runtime-2026-09-03.1",
        ]
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert store.latest("2026-09-03", "scan_am").status == "success"
    assert "Now due for 2026-09-03 (premarket): private_site_am, shared_site_am. Not dispatched." in out
    for job_id in ("private_site_am", "shared_site_am"):
        assert (
            "python scripts/automation_supervisor.py fallback-due --pipeline premarket "
            f"--job {job_id} --date 2026-09-03 --config-root C:/cfg --ref automation-runtime-2026-09-03.1"
        ) in out
    assert "NOTE: fallback-due acts only inside" not in out  # 07:47 ET is inside 05:20-08:55

    # Same resolution at 12:20 ET (the manual deploy time): the window caveat is printed.
    store.latest_values[("2026-09-03", "scan_am")] = sup.dataclasses.replace(
        store.latest_values[("2026-09-03", "scan_am")], status="indeterminate", source="github", phase="manual_review"
    )
    monkeypatch.setattr(sup, "_utc_now", lambda: dt.datetime(2026, 9, 3, 16, 20, tzinfo=UTC))
    assert sup.main(
        [
            "resolve", "--pipeline", "premarket", "--job", "scan_am", "--date", "2026-09-03",
            "--disposition", "success", "--reason", "verified", "--ref", "main",
        ]
    ) == 0
    late = capsys.readouterr().out
    assert "NOTE: fallback-due acts only inside the premarket ET fallback window 05:20 ET-08:55 ET" in late
    assert "--config-root" not in late


def test_resolve_retryable_failure_prints_the_job_itself(tmp_path, monkeypatch, capsys):
    now = dt.datetime(2026, 9, 2, 21, 0, tzinfo=UTC)
    store = sup.InMemoryReceiptStore(now=lambda: now)
    assert store.claim(
        sup.dataclasses.replace(
            _receipt("execution_report", "indeterminate", pipeline="execution", phase="manual_review"),
            run_date_et="2026-09-02",
        )
    )
    monkeypatch.setattr(sup, "_utc_now", lambda: now)
    monkeypatch.setattr(sup, "_make_runtime", lambda args, controller_only: (object(), store))

    rc = sup.main(
        [
            "resolve", "--pipeline", "execution", "--job", "execution_report", "--date", "2026-09-02",
            "--disposition", "retryable_failure", "--reason", "email never left", "--ref", "main",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert store.latest("2026-09-02", "execution_report").status == "failure"
    assert "Now due for 2026-09-02 (execution): execution_report." in out
    assert "fallback-due --pipeline execution --job execution_report --date 2026-09-02 --ref main" in out


def test_dependents_closure_is_transitive_and_ordered():
    premarket = sup.CATALOG["premarket"]
    assert [job.id for job in sup._dependents_of(premarket, "scan_am")] == [
        "private_site_am",
        "shared_site_am",
    ]
    assert [job.id for job in sup._dependents_of(premarket, "master_prices_am")] == [
        "risk_am",
        "event_sleeve_am",
        "scan_am",
        "private_site_am",
        "shared_site_am",
    ]


# ------------------------------------------------- round 2 (2026-09-04 verify)


def test_a_live_lease_of_another_writer_is_not_this_run_s_failure(tmp_path):
    # verify finding 2: a job whose lease is still LIVE belongs to another
    # writer, so leaving it alone is the correct action; a dependent blocked
    # only behind it must not turn the retry or a controller tick red.
    pipeline = _pipeline(_local_job("scan_am"), _local_job("site", depends_on=("scan_am",)))
    store = sup.InMemoryReceiptStore(now=lambda: NOW)
    live = _receipt(
        "scan_am",
        "running",
        phase="local_pre_side_effect",
        lease_expires_at_utc="2026-08-27T12:30:00+00:00",
    )
    assert store.claim(live)
    process = FakeProcess()

    outcomes, log = _run(_supervisor(tmp_path, pipeline, process, FakeDispatcher(), store), tmp_path)

    assert outcomes[0].status == "running" and outcomes[0].preexisting
    assert outcomes[1].status == "blocked" and outcomes[1].preexisting
    assert process.stream_calls == []
    assert store.latest("2026-08-27", "scan_am") == live
    assert "skip scan_am: live running lease" in log


def test_fallback_due_blocked_only_by_a_live_lease_is_not_fatal(tmp_path, monkeypatch):
    now = dt.datetime(2026, 9, 3, 12, 30, tzinfo=UTC)  # 08:30 ET, inside the window
    store = sup.InMemoryReceiptStore(now=lambda: now)
    for job_id in ("cboe_am", "master_prices_am", "risk_am", "event_sleeve_am"):
        assert store.claim(
            sup.dataclasses.replace(
                _receipt(job_id, "success", pipeline="premarket", phase="completed", token=f"{job_id}-ok"),
                run_date_et="2026-09-03",
            )
        )
    assert store.claim(
        sup.dataclasses.replace(
            _receipt(
                "scan_am",
                "running",
                pipeline="premarket",
                source="github",
                phase="github_reconcile",
                token="2026-09-03-scan_am-live",
                lease_expires_at_utc="2026-09-03T13:30:00+00:00",
            ),
            run_date_et="2026-09-03",
        )
    )
    dispatcher = FakeDispatcher()
    monkeypatch.setattr(sup, "_utc_now", lambda: now)
    monkeypatch.setattr(
        sup, "_make_runtime", lambda args, controller_only: _controller_runtime(tmp_path, store, dispatcher, now)
    )

    rc = sup.main(
        ["fallback-due", "--pipeline", "premarket", "--state-root", str(tmp_path / "state")]
    )

    assert rc == 0
    assert dispatcher.calls == [] and dispatcher.reconcile_calls == []
    text = next((tmp_path / "state" / "logs" / "2026-09-03").glob("premarket-*.log")).read_text(
        encoding="utf-8"
    )
    assert "skip scan_am: live running lease" in text
    assert "premarket/private_site_am (blocked)" in text
    assert "premarket/shared_site_am (blocked)" in text


def test_lock_records_when_the_current_holder_acquired_it(tmp_path):
    path = tmp_path / "automation_supervisor.lock"
    first = sup.GlobalFileLock(path, timeout_seconds=0)
    first.acquire()
    stamped = path.stat().st_mtime
    assert first.holder_since_et().endswith(" ET")
    first.release()
    # A later acquire re-stamps: the phrase "since <mtime>" names the current
    # holder, not the day the lock file was first created.
    os.utime(path, (stamped - 7200, stamped - 7200))
    second = sup.GlobalFileLock(path, timeout_seconds=0)
    second.acquire()
    try:
        assert path.stat().st_mtime > stamped - 7200
    finally:
        second.release()


def test_fallback_due_reports_a_held_lock_instead_of_raising(tmp_path, monkeypatch, capsys):
    now = dt.datetime(2026, 9, 3, 11, 30, tzinfo=UTC)  # 07:30 ET, the health task
    state = tmp_path / "state"
    monkeypatch.setattr(sup, "_utc_now", lambda: now)
    monkeypatch.setattr(
        sup,
        "_make_runtime",
        lambda args, controller_only=False: (object(), sup.InMemoryReceiptStore()),
    )
    with sup.GlobalFileLock(state / "automation_supervisor.lock", timeout_seconds=0):
        rc = sup.main(
            [
                "fallback-due", "--pipeline", "premarket", "--job", "scan_am",
                "--config-root", str(tmp_path), "--state-root", str(state),
            ]
        )
    out = capsys.readouterr().out
    assert rc == 1  # a hung primary at 07:30 is red, but never a traceback
    assert "FAIL fallback-due: primary still holds the supervisor lock since " in out
    assert "no job was inspected or dispatched" in out


def test_health_still_runs_the_battery_when_the_lock_is_held(tmp_path, monkeypatch, capsys):
    now = dt.datetime(2026, 9, 3, 11, 30, tzinfo=UTC)  # 07:30 ET
    state = tmp_path / "state"
    process = FakeProcess()

    class _Runtime:
        env = {"EXISTING": "1"}

    runtime = _Runtime()
    runtime.process = process
    monkeypatch.setattr(sup, "_utc_now", lambda: now)
    monkeypatch.setattr(
        sup,
        "_make_runtime",
        lambda args, controller_only=False: (runtime, sup.InMemoryReceiptStore()),
    )
    argv = [
        "health", "--config-root", str(tmp_path), "--state-root", str(state),
        "--repo-root", str(tmp_path), "--skip-tests",
    ]

    # Unlocked: plain pass-through, the battery's own exit code decides.
    assert sup.main(argv) == 0
    assert len(process.stream_calls) == 1
    assert process.stream_calls[0]["argv"][-1] == "--skip-tests"

    with sup.GlobalFileLock(state / "automation_supervisor.lock", timeout_seconds=0):
        rc = sup.main(argv)
    out = capsys.readouterr().out
    assert rc == 1
    assert len(process.stream_calls) == 2  # the read-only battery still ran
    assert "FAIL health: primary still holds the supervisor lock since " in out
    assert "running the read-only battery without it" in out
    logs = [p.read_text(encoding="utf-8") for p in (state / "logs" / "2026-09-03").glob("health-*.log")]
    assert len(logs) == 2  # one per run, both under the stable state root
    assert sum("FAIL health: primary still holds the supervisor lock since " in t for t in logs) == 1


def test_resolve_prefers_the_runtime_marker_ref_over_the_main_default(tmp_path, monkeypatch, capsys):
    now = dt.datetime(2026, 9, 3, 11, 47, tzinfo=UTC)
    config_root = tmp_path / "cfg"
    (config_root / ".local").mkdir(parents=True)
    (config_root / ".local" / "automation-runtime.json").write_text(
        json.dumps({"fallback_ref": "automation-runtime-2026-09-03.1", "pinned_sha": "0" * 40}),
        encoding="utf-8",
    )

    def _store():
        store = sup.InMemoryReceiptStore(now=lambda: now)
        assert store.claim(
            sup.dataclasses.replace(
                _receipt("scan_am", "indeterminate", pipeline="premarket", phase="manual_review"),
                run_date_et="2026-09-03",
            )
        )
        return store

    monkeypatch.setattr(sup, "_utc_now", lambda: now)

    store = _store()
    monkeypatch.setattr(sup, "_make_runtime", lambda args, controller_only: (object(), store))
    assert sup.main(
        [
            "resolve", "--pipeline", "premarket", "--job", "scan_am", "--date", "2026-09-03",
            "--disposition", "success", "--reason", "verified", "--config-root", str(config_root),
        ]
    ) == 0
    out = capsys.readouterr().out
    assert "--ref automation-runtime-2026-09-03.1" in out
    assert "--ref main" not in out
    assert "taken from the runtime marker under" in out

    # An explicit --ref always wins over the marker.
    store = _store()
    monkeypatch.setattr(sup, "_make_runtime", lambda args, controller_only: (object(), store))
    assert sup.main(
        [
            "resolve", "--pipeline", "premarket", "--job", "scan_am", "--date", "2026-09-03",
            "--disposition", "success", "--reason", "verified", "--config-root", str(config_root),
            "--ref", "automation-runtime-2026-09-04.1",
        ]
    ) == 0
    assert "--ref automation-runtime-2026-09-04.1" in capsys.readouterr().out

    # An explicit `--ref main` is honoured VERBATIM (round 3): the operator
    # asked for the branch, so the marker must not silently replace it -- it
    # only earns the warning.
    store = _store()
    monkeypatch.setattr(sup, "_make_runtime", lambda args, controller_only: (object(), store))
    assert sup.main(
        [
            "resolve", "--pipeline", "premarket", "--job", "scan_am", "--date", "2026-09-03",
            "--disposition", "success", "--reason", "verified", "--config-root", str(config_root),
            "--ref", "main",
        ]
    ) == 0
    explicit_main = capsys.readouterr().out
    assert "--job private_site_am --date 2026-09-03 --config-root" in explicit_main
    assert "--ref main" in explicit_main
    assert "automation-runtime-2026-09-03.1" not in explicit_main
    assert "WARNING: --ref main is a moving branch" in explicit_main

    # The marker lives under the RUNTIME root in production; --repo-root is
    # that root under a scheduled task, and it is searched first.
    runtime_root = tmp_path / "rt-v9"
    (runtime_root / ".local").mkdir(parents=True)
    (runtime_root / ".local" / "automation-runtime.json").write_text(
        json.dumps({"fallback_ref": "automation-runtime-2026-09-05.2"}), encoding="utf-8"
    )
    store = _store()
    monkeypatch.setattr(sup, "_make_runtime", lambda args, controller_only: (object(), store))
    assert sup.main(
        [
            "resolve", "--pipeline", "premarket", "--job", "scan_am", "--date", "2026-09-03",
            "--disposition", "success", "--reason", "verified",
            "--config-root", str(config_root), "--repo-root", str(runtime_root),
        ]
    ) == 0
    from_runtime = capsys.readouterr().out
    assert "--ref automation-runtime-2026-09-05.2" in from_runtime
    assert "automation-runtime-2026-09-03.1" not in from_runtime  # runtime root wins

    # No marker under the config root: the commands still print, with a warning
    # that `main` is a moving branch rather than the pinned tag.
    store = _store()
    monkeypatch.setattr(sup, "_make_runtime", lambda args, controller_only: (object(), store))
    assert sup.main(
        [
            "resolve", "--pipeline", "premarket", "--job", "scan_am", "--date", "2026-09-03",
            "--disposition", "success", "--reason", "verified", "--config-root", str(tmp_path / "bare"),
        ]
    ) == 0
    bare = capsys.readouterr().out
    assert "--ref main" in bare
    assert "WARNING: --ref main is a moving branch" in bare


@pytest.mark.parametrize(
    "marker, expected",
    [
        ({"fallback_ref": "automation-runtime-2026-09-03.1"}, "automation-runtime-2026-09-03.1"),
        ({"fallback_ref": "main"}, None),          # the default is not an improvement
        ({"fallback_ref": "tag with space"}, None),
        ({"fallback_ref": ""}, None),
        ({"fallback_ref": 7}, None),
        ({}, None),
        ("not-an-object", None),
    ],
)
def test_marker_fallback_ref_is_defensive(tmp_path, marker, expected):
    root = tmp_path / "cfg"
    (root / ".local").mkdir(parents=True)
    (root / ".local" / "automation-runtime.json").write_text(json.dumps(marker), encoding="utf-8")
    assert sup._marker_fallback_ref(str(root)) == expected
    assert sup._marker_fallback_ref(str(tmp_path / "absent")) is None
    assert sup._marker_fallback_ref(None) is None
    # An unparseable marker is a hint that failed, never an exception.
    (root / ".local" / "automation-runtime.json").write_text("{", encoding="utf-8")
    assert sup._marker_fallback_ref(str(root)) is None


# ------------------------------------------------- round 3 (2026-09-04 verify 2)


def test_plan_shows_the_local_retry_window_for_pipelines_that_have_one():
    # The retry is a real scheduled writer of these receipts; a plan that hides
    # it under-describes who can claim them (round-2 verify).
    premarket = sup.render_plan(sup.CATALOG["premarket"])
    assert (
        "  local retry window: 05:45 ET-07:00 ET "
        "(run-pipeline --pipeline premarket --retry)"
    ) in premarket
    assert premarket.index("fallback window:") < premarket.index("local retry window:")
    for pipeline_id in sup.CATALOG:
        if pipeline_id == "premarket":
            continue
        assert "local retry window" not in sup.render_plan(sup.CATALOG[pipeline_id])


def test_marker_is_read_from_the_runtime_root_first(tmp_path):
    # Production truth: install_local_automation_tasks.ps1 writes the marker
    # under RuntimeRoot and run_local_automation.ps1 reads it there. The config
    # root is only a development-checkout convenience.
    runtime = tmp_path / "rt-v9"
    config = tmp_path / "cfg"
    for root, ref in ((runtime, "runtime-tag"), (config, "config-tag")):
        (root / ".local").mkdir(parents=True)
        (root / ".local" / "automation-runtime.json").write_text(
            json.dumps({"fallback_ref": ref}), encoding="utf-8"
        )
    assert sup._marker_fallback_ref(str(runtime), str(config)) == "runtime-tag"
    assert sup._marker_fallback_ref(None, str(config)) == "config-tag"
    # A root whose marker is unusable falls through to the next root.
    (runtime / ".local" / "automation-runtime.json").write_text("{", encoding="utf-8")
    assert sup._marker_fallback_ref(str(runtime), str(config)) == "config-tag"
    (runtime / ".local" / "automation-runtime.json").write_text(
        json.dumps({"fallback_ref": "main"}), encoding="utf-8"
    )
    assert sup._marker_fallback_ref(str(runtime), str(config)) == "config-tag"
    assert sup._marker_fallback_ref() is None


def test_ref_default_is_none_so_an_explicit_main_is_distinguishable():
    args = sup.build_parser().parse_args(
        ["fallback-due", "--pipeline", "premarket", "--config-root", "."]
    )
    assert args.ref is None
    explicit = sup.build_parser().parse_args(
        ["fallback-due", "--pipeline", "premarket", "--config-root", ".", "--ref", "main"]
    )
    assert explicit.ref == "main"
    # Every dispatch path still resolves a concrete ref.
    assert (args.ref or sup.DEFAULT_REF) == "main"
    source = pathlib.Path(sup.__file__).read_text(encoding="utf-8")
    assert "ref=args.ref or DEFAULT_REF" in source


def test_health_prints_the_fail_line_even_when_the_run_log_cannot_be_opened(
    tmp_path, monkeypatch, capsys
):
    now = dt.datetime(2026, 9, 3, 11, 30, tzinfo=UTC)  # 07:30 ET
    state = tmp_path / "state"
    state.mkdir(parents=True)
    (state / "logs").write_text("not a directory", encoding="utf-8")  # unusable state root
    process = FakeProcess()

    class _Runtime:
        env = {"EXISTING": "1"}

    runtime = _Runtime()
    runtime.process = process
    monkeypatch.setattr(sup, "_utc_now", lambda: now)
    monkeypatch.setattr(
        sup,
        "_make_runtime",
        lambda args, controller_only=False: (runtime, sup.InMemoryReceiptStore()),
    )
    argv = [
        "health", "--config-root", str(tmp_path), "--state-root", str(state),
        "--repo-root", str(tmp_path), "--skip-tests",
    ]
    with sup.GlobalFileLock(state / "automation_supervisor.lock", timeout_seconds=0):
        rc = sup.main(argv)
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL health: primary still holds the supervisor lock since " in out
    assert "WARN health: cannot open the run log" in out
    assert len(process.stream_calls) == 1  # the read-only battery still ran
    # The FAIL line reaches stdout exactly once: printed before the logger is
    # built, and not repeated by the stdout-only fallback logger.
    assert out.count("FAIL health: primary still holds") == 1
