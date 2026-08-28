from __future__ import annotations

import datetime as dt
import json
import sys

import pytest

from scripts import automation_supervisor as sup


class FakeProcess:
    def __init__(self, stream_codes=None, captures=None):
        self.stream_codes = list(stream_codes or [])
        self.captures = list(captures or [])
        self.stream_calls = []
        self.capture_calls = []

    def stream(self, argv, *, cwd, env, timeout_seconds, logger):
        self.stream_calls.append(
            {
                "argv": list(argv),
                "cwd": cwd,
                "env": dict(env),
                "timeout": timeout_seconds,
            }
        )
        return self.stream_codes.pop(0) if self.stream_codes else 0

    def capture(self, argv, *, cwd, env, timeout_seconds):
        self.capture_calls.append(list(argv))
        if not self.captures:
            raise AssertionError(f"unexpected capture call: {argv}")
        value = self.captures.pop(0)
        if isinstance(value, sup.CaptureResult):
            return value
        return sup.CaptureResult(0, value)


class FakeDispatcher:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.calls = []

    def dispatch_and_wait(self, workflow, *, automation_token, logger):
        self.calls.append((workflow, automation_token))
        if self.fail:
            raise sup.DispatchError("synthetic dispatch failure")
        return sup.GithubRun(
            database_id=123,
            status="completed",
            conclusion="success",
            title=f"automation {automation_token}",
            url="https://example.test/run/123",
        )


class NoopValidator:
    def __init__(self):
        self.calls = []

    def validate(self, outputs, *, repo_root, started_at_utc, logger):
        self.calls.append((outputs, repo_root, started_at_utc))


def _logger(tmp_path):
    return sup.RunLogger(tmp_path / "run.log", echo=False)


def _receipt(job_id, status, *, token="prior", source="local", pipeline="test"):
    return sup.Receipt(
        schema_version="automation-receipt.v1",
        pipeline=pipeline,
        job_id=job_id,
        run_date_et="2026-08-27",
        status=status,
        source=source,
        automation_token=token,
        started_at_utc="2026-08-27T10:00:00+00:00",
        updated_at_utc="2026-08-27T10:01:00+00:00",
    )


def _supervisor(tmp_path, pipeline, process, dispatcher, receipts=None, validator=None):
    return sup.AutomationSupervisor(
        catalog={pipeline.id: pipeline},
        repo_root=tmp_path,
        state_root=tmp_path / "state",
        python_executable=sys.executable,
        env={
            "NEEDED": "present",
            "LOCAL_AUTOMATION_STRICT": "1",
        },
        receipts=receipts or sup.InMemoryReceiptStore(),
        process=process,
        dispatcher=dispatcher,
        validator=validator,
        now=lambda: dt.datetime(2026, 8, 27, 12, 0, tzinfo=dt.timezone.utc),
    )


def test_catalog_has_all_et_pipelines_and_cloud_only_site_jobs():
    assert set(sup.CATALOG) == {
        "premarket",
        "discretionary",
        "execution",
        "postclose",
        "indicator",
        "weekly-rundown",
    }
    assert sup.CATALOG["premarket"].run_at_et == dt.time(4, 10)
    assert sup.CATALOG["execution"].run_at_et == dt.time(16, 30)
    assert sup.CATALOG["postclose"].run_at_et == dt.time(17, 10)

    site_jobs = [
        job
        for pipeline in sup.CATALOG.values()
        for job in pipeline.jobs
        if "site" in job.id
    ]
    assert site_jobs
    assert all(job.dispatch_only and not job.commands for job in site_jobs)
    assert {job.workflow.workflow for job in site_jobs} == {
        "deploy_site.yml",
        "deploy_shared_seasonals.yml",
    }


def test_catalog_keeps_event_and_trend_once_only_with_dedicated_fallbacks():
    jobs = {
        job.id: job
        for pipeline in sup.CATALOG.values()
        for job in pipeline.jobs
    }
    event = jobs["event_sleeve_am"]
    trend = jobs["trend_sleeve"]
    assert event.duplicate_sensitive
    assert event.workflow.workflow == "event_sleeve.yml"
    assert trend.duplicate_sensitive
    assert trend.workflow.workflow == "trend_sleeve.yml"

    am_scan = jobs["scan_am"].workflow.input_dict()
    pm_scan = jobs["scan_pm"].workflow.input_dict()
    assert am_scan == {
        "bookend": "am",
        "run_event_sleeve": "false",
        "deploy_after_scan": "false",
    }
    assert pm_scan["bookend"] == "pm"
    assert pm_scan["run_event_sleeve"] == "false"


def test_catalog_local_commands_mirror_critical_workflow_modes():
    jobs = {
        job.id: job
        for pipeline in sup.CATALOG.values()
        for job in pipeline.jobs
    }

    master_am = [value for command in jobs["master_prices_am"].commands for value in command.argv]
    master_pm = [value for command in jobs["master_prices_pm"].commands for value in command.argv]
    assert "--exclude-today" in master_am
    assert "--exclude-today" not in master_pm

    risk_am = [value for command in jobs["risk_am"].commands for value in command.argv]
    assert "--data-only" in risk_am and "--refresh-last" in risk_am

    for job_id in ("cboe_am", "cboe_pm", "risk_am", "risk_pm", "scan_am"):
        flattened = [value for command in jobs[job_id].commands for value in command.argv]
        if job_id.startswith("scan"):
            assert "exposure" in flattened
        assert "--local-primary" in flattened

    scan_pm = [value for command in jobs["scan_pm"].commands for value in command.argv]
    assert "daily_scan.py" in scan_pm and "--scope=all" in scan_pm
    assert "--local-primary" not in scan_pm  # PM scan does not publish exposure.


def test_hydrate_environment_uses_only_explicit_files_and_never_mutates_parent(tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    (config / ".env").write_text(
        "EMAIL_USER=sender@example.test\n"
        "R2_ACCOUNT_ID=acct\nR2_ACCESS_KEY_ID=key\n"
        "R2_SECRET_ACCESS_KEY=secret\nR2_BUCKET=bucket\n"
        "GH_PAT_NEW_SEASONALS=github-token\n",
        encoding="utf-8",
    )
    exec_env = tmp_path / "exec_agent.env"
    exec_env.write_text("STATUS_TOKEN=status-secret\n", encoding="utf-8")
    gcp = tmp_path / "credentials.json"
    gcp.write_text(json.dumps({"type": "service_account", "private_key": "private"}), encoding="utf-8")
    base = {"UNCHANGED": "yes"}

    env = sup.hydrate_environment(
        config_root=config,
        gcp_json_path=gcp,
        exec_env_path=exec_env,
        base_env=base,
    )

    assert env["EMAIL_USER"] == "sender@example.test"
    assert env["STATUS_TOKEN"] == "status-secret"
    assert json.loads(env["GCP_JSON"])["type"] == "service_account"
    assert env["GH_TOKEN"] == "github-token"
    assert env["LOCAL_AUTOMATION_STRICT"] == "1"
    assert base == {"UNCHANGED": "yes"}


def test_hydrate_environment_rejects_invalid_gcp_json(tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    (config / ".env").write_text("R2_BUCKET=bucket\n", encoding="utf-8")
    exec_env = tmp_path / "exec.env"
    exec_env.write_text("STATUS_TOKEN=x\n", encoding="utf-8")
    gcp = tmp_path / "credentials.json"
    gcp.write_text("not-json", encoding="utf-8")

    with pytest.raises(sup.AutomationError, match="not valid JSON"):
        sup.hydrate_environment(
            config_root=config,
            gcp_json_path=gcp,
            exec_env_path=exec_env,
            base_env={},
        )


def test_receipt_store_blocks_success_and_running_but_allows_failure_retry():
    store = sup.InMemoryReceiptStore()
    success = _receipt("event", "success")
    assert store.claim(success)
    assert not store.claim(_receipt("event", "running", token="new"))

    failure = _receipt("trend", "failure")
    assert store.claim(failure)
    retry = _receipt("trend", "running", token="retry")
    assert store.claim(retry)
    assert store.latest("2026-08-27", "trend").automation_token == "retry"


def test_local_success_sets_strict_child_env_and_success_receipt(tmp_path):
    job = sup.JobSpec(
        id="component",
        description="component",
        commands=(sup.CommandSpec("step", ("{python}", "tool.py")),),
        workflow=sup.WorkflowSpec("component.yml"),
        required_env=("NEEDED",),
    )
    pipeline = sup.PipelineSpec(
        "test", "test", "weekdays", dt.time(1), dt.time(2), dt.time(3), (job,)
    )
    process = FakeProcess([0])
    dispatcher = FakeDispatcher()
    receipts = sup.InMemoryReceiptStore()
    supervisor = _supervisor(tmp_path, pipeline, process, dispatcher, receipts=receipts)

    with _logger(tmp_path) as logger:
        outcomes = supervisor.run_pipeline(
            "test",
            run_date="2026-08-27",
            allow_fallback=True,
            logger=logger,
        )

    assert outcomes[0].status == "success"
    assert process.stream_calls[0]["env"]["LOCAL_AUTOMATION_PRIMARY"] == "1"
    assert process.stream_calls[0]["env"]["LOCAL_AUTOMATION_STRICT"] == "1"
    assert process.stream_calls[0]["env"]["LOCAL_AUTOMATION_RUN_TOKEN"]
    latest = receipts.latest("2026-08-27", "component")
    assert latest.status == "success" and latest.source == "local"
    assert not dispatcher.calls


def test_local_failure_immediately_dispatches_github_and_records_both_transitions(tmp_path):
    job = sup.JobSpec(
        id="component",
        description="component",
        commands=(sup.CommandSpec("bad step", ("{python}", "tool.py")),),
        workflow=sup.WorkflowSpec("component.yml", (("mode", "pm"),)),
    )
    pipeline = sup.PipelineSpec(
        "test", "test", "weekdays", dt.time(1), dt.time(2), dt.time(3), (job,)
    )
    process = FakeProcess([17])
    dispatcher = FakeDispatcher()
    receipts = sup.InMemoryReceiptStore()
    supervisor = _supervisor(tmp_path, pipeline, process, dispatcher, receipts=receipts)

    with _logger(tmp_path) as logger:
        outcome = supervisor.run_job(
            pipeline,
            job,
            run_date="2026-08-27",
            logger=logger,
            allow_fallback=True,
        )

    assert outcome == sup.JobOutcome("component", "success", "github")
    assert len(dispatcher.calls) == 1
    token = dispatcher.calls[0][1]
    assert token
    transitions = [(row.status, row.source) for row in receipts.events]
    assert ("failure", "local") in transitions
    assert ("running", "github") in transitions
    assert receipts.latest("2026-08-27", "component").status == "success"


def test_running_once_only_receipt_never_executes_or_dispatches(tmp_path):
    job = sup.JobSpec(
        id="event",
        description="event",
        commands=(sup.CommandSpec("event", ("{python}", "event.py")),),
        workflow=sup.WorkflowSpec("event_sleeve.yml"),
        duplicate_sensitive=True,
    )
    pipeline = sup.PipelineSpec(
        "test", "test", "weekdays", dt.time(1), dt.time(2), dt.time(3), (job,)
    )
    process = FakeProcess()
    dispatcher = FakeDispatcher()
    receipts = sup.InMemoryReceiptStore()
    assert receipts.claim(_receipt("event", "running", pipeline="test"))
    supervisor = _supervisor(tmp_path, pipeline, process, dispatcher, receipts=receipts)

    with _logger(tmp_path) as logger:
        outcome = supervisor.run_job(
            pipeline,
            job,
            run_date="2026-08-27",
            logger=logger,
            allow_fallback=True,
        )

    assert outcome.status == "running"
    assert not process.stream_calls
    assert not dispatcher.calls


def test_github_only_controller_dispatches_missing_and_honors_dependencies(tmp_path):
    first = sup.JobSpec(
        id="first",
        description="first",
        commands=(sup.CommandSpec("first", ("{python}", "first.py")),),
        workflow=sup.WorkflowSpec("first.yml"),
    )
    second = sup.JobSpec(
        id="second",
        description="second",
        commands=(sup.CommandSpec("second", ("{python}", "second.py")),),
        workflow=sup.WorkflowSpec("second.yml"),
        depends_on=("first",),
    )
    pipeline = sup.PipelineSpec(
        "test", "test", "weekdays", dt.time(1), dt.time(2), dt.time(3), (first, second)
    )
    process = FakeProcess()
    dispatcher = FakeDispatcher()
    supervisor = _supervisor(tmp_path, pipeline, process, dispatcher)

    with _logger(tmp_path) as logger:
        outcomes = supervisor.run_pipeline(
            "test",
            run_date="2026-08-27",
            allow_fallback=True,
            github_only=True,
            logger=logger,
        )

    assert [row.status for row in outcomes] == ["success", "success"]
    assert [row[0].workflow for row in dispatcher.calls] == ["first.yml", "second.yml"]
    assert dispatcher.calls[0][1] != dispatcher.calls[1][1]
    assert not process.stream_calls


def test_github_dispatcher_finds_unique_token_in_run_title_and_waits(tmp_path):
    token = "2026-08-27-job-unique"
    queued = json.dumps(
        [
            {
                "databaseId": 77,
                "status": "queued",
                "conclusion": None,
                "displayTitle": f"fallback {token}",
                "url": "https://example.test/77",
            }
        ]
    )
    completed = json.dumps(
        [
            {
                "databaseId": 77,
                "status": "completed",
                "conclusion": "success",
                "displayTitle": f"fallback {token}",
                "url": "https://example.test/77",
            }
        ]
    )
    process = FakeProcess(captures=["", queued, completed])
    ticks = iter([0, 0, 1, 1, 2, 2])
    dispatcher = sup.GithubDispatcher(
        process,
        repo_root=tmp_path,
        env={},
        repository="owner/repo",
        ref="main",
        sleep=lambda _: None,
        monotonic=lambda: next(ticks),
        poll_seconds=0,
    )

    with _logger(tmp_path) as logger:
        run = dispatcher.dispatch_and_wait(
            sup.WorkflowSpec("job.yml", (("mode", "am"),), timeout_seconds=30),
            automation_token=token,
            logger=logger,
        )

    assert run.database_id == 77
    dispatch_argv = process.capture_calls[0]
    assert f"automation_token={token}" in dispatch_argv
    assert "mode=am" in dispatch_argv


class HeadBackend:
    def __init__(self, heads):
        self.heads = heads

    def head(self, key):
        return self.heads.get(key)


def test_output_validator_checks_exact_r2_size_and_recent_upload(tmp_path):
    local = tmp_path / "data" / "producer.json"
    local.parent.mkdir()
    local.write_bytes(b"123456")
    started = dt.datetime(2026, 8, 27, 12, 0, tzinfo=dt.timezone.utc)
    backend = HeadBackend(
        {
            "producer.json": {
                "ContentLength": 6,
                "LastModified": started + dt.timedelta(seconds=1),
            }
        }
    )
    validator = sup.OutputValidator(backend)

    with _logger(tmp_path) as logger:
        validator.validate(
            (sup.OutputSpec("data/producer.json", r2_key="producer.json", require_recent_upload=True),),
            repo_root=tmp_path,
            started_at_utc=started,
            logger=logger,
        )

    backend.heads["producer.json"]["ContentLength"] = 5
    with _logger(tmp_path) as logger, pytest.raises(sup.ValidationError, match="size mismatch"):
        validator.validate(
            (sup.OutputSpec("data/producer.json", r2_key="producer.json"),),
            repo_root=tmp_path,
            started_at_utc=started,
            logger=logger,
        )


def test_output_validator_maps_every_intraday_glob_file(tmp_path):
    intraday = tmp_path / "data" / "intraday"
    intraday.mkdir(parents=True)
    (intraday / "SPY_15min.parquet").write_bytes(b"spy")
    (intraday / "QQQ_15min.parquet").write_bytes(b"qqqq")
    backend = HeadBackend(
        {
            "intraday/15min/SPY.parquet": {"ContentLength": 3},
            "intraday/15min/QQQ.parquet": {"ContentLength": 4},
        }
    )
    validator = sup.OutputValidator(backend)
    spec = sup.OutputSpec(
        "data/intraday/*_15min.parquet",
        r2_prefix="intraday/15min",
        strip_suffix="_15min",
    )

    with _logger(tmp_path) as logger:
        validator.validate(
            (spec,),
            repo_root=tmp_path,
            started_at_utc=dt.datetime.now(tz=dt.timezone.utc),
            logger=logger,
        )


def test_global_file_lock_rejects_second_holder(tmp_path):
    path = tmp_path / "global.lock"
    with sup.GlobalFileLock(path, timeout_seconds=0), pytest.raises(sup.LockUnavailable):
        sup.GlobalFileLock(path, timeout_seconds=0).acquire()


def test_plan_and_run_dry_run_do_not_require_secret_paths(capsys):
    assert sup.main(["plan", "execution"]) == 0
    plan_output = capsys.readouterr().out
    assert "GITHUB-DISPATCH-ONLY" not in plan_output
    assert "STATUS_TOKEN" in plan_output

    # Conditional CLI validation intentionally permits dry-run without any
    # credential files.  It prints the same plan and returns before R2 setup.
    assert sup.main(["run", "execution", "--dry-run"]) == 0
    assert "daily_execution_report.py" in capsys.readouterr().out


def test_task_scheduler_entry_point_and_cutover_guard_prevent_prior_day_dispatch(
    tmp_path, capsys
):
    # This must return before reading the deliberately missing config root or
    # constructing an R2/GitHub client.  It is the migration-night duplicate
    # guard for the 2026-08-27 production runs.
    rc = sup.main(
        [
            "run-pipeline",
            "--pipeline",
            "premarket",
            "--config-root",
            str(tmp_path / "missing-config"),
            "--date",
            "2026-08-27",
        ]
    )
    assert rc == 0
    assert "precedes automation cutover 2026-08-28" in capsys.readouterr().out


@pytest.mark.parametrize(
    "pipeline_id",
    [
        "premarket",
        "discretionary",
        "execution",
        "postclose",
        "indicator",
        "weekly-rundown",
    ],
)
def test_task_scheduler_entry_point_accepts_every_installed_pipeline_id(
    pipeline_id, tmp_path, capsys
):
    assert (
        sup.main(
            [
                "run-pipeline",
                "--pipeline",
                pipeline_id,
                "--config-root",
                str(tmp_path / "not-read-during-dry-run"),
                "--dry-run",
            ]
        )
        == 0
    )
    assert pipeline_id in capsys.readouterr().out
