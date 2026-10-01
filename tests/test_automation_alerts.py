"""Pins for the supervisor's operator failure email (2026-09-30 incident).

The SMTP sender is always monkeypatched; nothing here touches the network.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import sys

import pytest

from scripts import automation_supervisor as sup

UTC = dt.timezone.utc
POSTCLOSE_NOW = dt.datetime(2026, 9, 30, 21, 20, tzinfo=UTC)  # 17:20 ET Wednesday
RETRY_NOW = dt.datetime(2026, 10, 1, 10, 0, tzinfo=UTC)  # 06:00 ET Thursday
CREDS = {"EMAIL_USER": "sender@example.test", "EMAIL_PASS": "secret"}
CALENDAR_ERROR = "CalendarError: Earnings calendar missed the previous NYSE session; refresh required"


class LoggingProcess:
    """Fails any argv containing a key of ``failing`` after logging its line."""

    def __init__(self, failing: dict[str, str] | None = None):
        self.failing = failing or {}
        self.calls: list[list[str]] = []

    def stream(self, argv, *, cwd, env, timeout_seconds, logger):
        self.calls.append(list(argv))
        for script, line in self.failing.items():
            if script in argv:
                logger.line("Traceback (most recent call last):")
                logger.line(line)
                return 1
        return 0

    def capture(self, argv, *, cwd, env, timeout_seconds):
        raise AssertionError("unexpected capture")


class NoopValidator:
    def validate(self, outputs, *, repo_root, started_at_utc, logger):
        return None


def _job(job_id, *, depends_on=(), rerun_safe=True, script=None):
    return sup.JobSpec(
        id=job_id,
        description=job_id,
        commands=(
            sup.CommandSpec("pull", ("{python}", "pull.py")),
            sup.CommandSpec(
                "effect", ("{python}", script or f"{job_id}.py"), side_effecting=True
            ),
        ),
        workflow=sup.WorkflowSpec(f"{job_id}.yml"),
        rerun_safe=rerun_safe,
        depends_on=tuple(depends_on),
    )


def _wire(monkeypatch, tmp_path, pipeline_id, jobs, *, now, env=None, store=None, failing=None):
    pipeline = dataclasses.replace(sup.CATALOG[pipeline_id], jobs=tuple(jobs))
    store = store or sup.InMemoryReceiptStore(now=lambda: now)
    process = LoggingProcess(failing)
    supervisor = sup.AutomationSupervisor(
        catalog={pipeline_id: pipeline},
        repo_root=tmp_path,
        state_root=tmp_path / "state",
        python_executable=sys.executable,
        env=dict(CREDS if env is None else env),
        receipts=store,
        process=process,
        dispatcher=None,  # type: ignore[arg-type]
        validator=NoopValidator(),
        now=lambda: now,
    )
    sent: list[tuple[dict, str, str]] = []
    monkeypatch.setattr(sup, "_utc_now", lambda: now)
    monkeypatch.setattr(sup, "_make_runtime", lambda args, controller_only=False: (supervisor, store))
    monkeypatch.setattr(
        sup, "_send_alert_email", lambda env, subject, body: sent.append((dict(env), subject, body))
    )
    return sent, store


def _argv(tmp_path, pipeline_id, *extra):
    return [
        "run-pipeline", "--pipeline", pipeline_id, "--config-root", str(tmp_path),
        "--state-root", str(tmp_path / "state"), "--no-fallback", *extra,
    ]


def _postclose_jobs():
    return (
        _job("master_prices_pm"),
        _job("earnings_and_grades", script="scripts/refresh_earnings_calendar.py"),
        _job("portfolio_report", depends_on=("master_prices_pm", "earnings_and_grades"), rerun_safe=False),
        _job("scan_pm", depends_on=("master_prices_pm", "earnings_and_grades"), rerun_safe=False),
    )


def _log_text(tmp_path, run_date, pipeline_id):
    logs = list((tmp_path / "state" / "logs" / run_date).glob(f"{pipeline_id}-*.log"))
    assert len(logs) == 1
    return logs[0], logs[0].read_text(encoding="utf-8")


def test_the_incident_postclose_run_sends_one_email(tmp_path, monkeypatch):
    sent, _ = _wire(
        monkeypatch, tmp_path, "postclose", _postclose_jobs(), now=POSTCLOSE_NOW,
        failing={"scripts/refresh_earnings_calendar.py": "Earnings refresh stopped: Alpha candidate failed"},
    )
    rc = sup.main(_argv(tmp_path, "postclose"))
    assert rc == 1  # the alert never changes the exit code
    assert len(sent) == 1
    env, subject, body = sent[0]
    assert subject == "[New Seasonals] postclose 2026-09-30: 1 failed, 2 blocked"
    log_path, log = _log_text(tmp_path, "2026-09-30", "postclose")
    assert "Pipeline: postclose" in body
    assert "ET date:  2026-09-30" in body
    assert "Mode:     run-pipeline" in body
    assert f"Log:      {log_path}" in body
    assert "  earnings_and_grades: failure, source local" in body
    assert "    detail: effect exited 1" in body
    assert "    last error line: Earnings refresh stopped: Alpha candidate failed" in body
    assert "  portfolio_report: waiting on earnings_and_grades" in body
    assert "  scan_pm: waiting on earnings_and_grades" in body
    assert "master_prices_pm" not in body
    assert "resolve --pipeline" not in body  # nothing indeterminate
    assert f"alert: emailed {subject}" in log


def test_indeterminate_scan_carries_the_resolve_hint_and_last_error_line(tmp_path, monkeypatch):
    # The 2026-10-01 04:16 scan_am shape: the scanner raised after its boundary.
    jobs = (_job("scan_am", rerun_safe=False, script="daily_scan.py"),
            _job("private_site_am", depends_on=("scan_am",)))
    now = dt.datetime(2026, 10, 1, 8, 16, tzinfo=UTC)
    sent, _ = _wire(
        monkeypatch, tmp_path, "premarket", jobs, now=now, failing={"daily_scan.py": CALENDAR_ERROR}
    )
    assert sup.main(_argv(tmp_path, "premarket")) == 1
    (_, subject, body), = sent
    assert subject == "[New Seasonals] premarket 2026-10-01: 1 indeterminate, 1 blocked"
    assert "  scan_am: indeterminate, source local" in body
    assert f"    last error line: {CALENDAR_ERROR}" in body
    assert (
        "  python scripts/automation_supervisor.py resolve --pipeline premarket --job scan_am "
        "--date 2026-10-01 --disposition success|retryable_failure --reason ..."
    ) in body


def test_all_success_sends_nothing(tmp_path, monkeypatch):
    sent, _ = _wire(monkeypatch, tmp_path, "postclose", _postclose_jobs(), now=POSTCLOSE_NOW)
    assert sup.main(_argv(tmp_path, "postclose")) == 0
    assert sent == []
    _, log = _log_text(tmp_path, "2026-09-30", "postclose")
    assert "alert:" not in log


def test_dry_run_plan_status_and_fallback_due_never_send(tmp_path, monkeypatch, capsys):
    sent, store = _wire(monkeypatch, tmp_path, "postclose", _postclose_jobs(), now=POSTCLOSE_NOW,
                        failing={"scripts/refresh_earnings_calendar.py": "boom error"})
    assert sup.main(_argv(tmp_path, "postclose", "--dry-run")) == 0
    assert sup.main(["plan", "postclose"]) == 0
    assert sup.main(["status", "postclose", "--state-root", str(tmp_path / "state")]) == 0
    sup.main(["fallback-due", "--pipeline", "postclose", "--state-root", str(tmp_path / "state")])
    assert sent == []


def test_retry_alerts_only_on_its_own_non_success(tmp_path, monkeypatch):
    run_date = "2026-10-01"
    jobs = (_job("scan_am", rerun_safe=False), _job("private_site_am", depends_on=("scan_am",)))
    store = sup.InMemoryReceiptStore(now=lambda: RETRY_NOW)
    stuck = sup.Receipt(
        schema_version="automation-receipt.v1", pipeline="premarket", job_id="scan_am",
        run_date_et=run_date, status="indeterminate", source="local", automation_token="t",
        started_at_utc="2026-10-01T08:10:00+00:00", updated_at_utc="2026-10-01T08:16:00+00:00",
        phase="manual_review", detail=CALENDAR_ERROR,
    )
    assert store.claim(stuck)
    sent, _ = _wire(monkeypatch, tmp_path, "premarket", jobs, now=RETRY_NOW, store=store)
    assert sup.main(_argv(tmp_path, "premarket", "--retry")) == 0
    assert sent == []  # the 04:10 run that produced the indeterminate already alerted


def test_retry_that_fails_itself_alerts_and_lists_preexisting_receipts(tmp_path, monkeypatch):
    run_date = "2026-10-01"
    jobs = (
        _job("cboe_am", script="cboe.py"),
        _job("scan_am", rerun_safe=False),
        _job("private_site_am", depends_on=("scan_am",)),
    )
    store = sup.InMemoryReceiptStore(now=lambda: RETRY_NOW)
    assert store.claim(sup.Receipt(
        schema_version="automation-receipt.v1", pipeline="premarket", job_id="scan_am",
        run_date_et=run_date, status="indeterminate", source="local", automation_token="t",
        started_at_utc="2026-10-01T08:10:00+00:00", updated_at_utc="2026-10-01T08:16:00+00:00",
        phase="manual_review",
    ))
    sent, _ = _wire(monkeypatch, tmp_path, "premarket", jobs, now=RETRY_NOW, store=store,
                    failing={"cboe.py": "ERROR: CBOE fetch failed"})
    assert sup.main(_argv(tmp_path, "premarket", "--retry")) == 1
    (_, subject, body), = sent
    assert subject == "[New Seasonals] premarket 2026-10-01: 1 failed"
    assert "Mode:     retry" in body
    assert "  cboe_am: failure, source local" in body
    assert "  scan_am: indeterminate, source local (pre-existing receipt; reported, not counted)" in body
    assert "  private_site_am: waiting on scan_am (pre-existing receipt; reported, not counted)" in body
    assert "resolve --pipeline premarket --job scan_am --date 2026-10-01" in body


def test_switch_off_and_missing_credentials_degrade_to_a_log_line(tmp_path, monkeypatch):
    failing = {"scripts/refresh_earnings_calendar.py": "boom error"}
    sent, _ = _wire(monkeypatch, tmp_path, "postclose", _postclose_jobs(), now=POSTCLOSE_NOW,
                    env={**CREDS, "NEW_SEASONALS_ALERT_EMAIL": "0"}, failing=failing)
    assert sup.main(_argv(tmp_path, "postclose")) == 1
    assert sent == []
    _, log = _log_text(tmp_path, "2026-09-30", "postclose")
    assert "alert: email disabled by NEW_SEASONALS_ALERT_EMAIL=0; not sent: [New Seasonals] postclose" in log


def test_missing_credentials_never_reach_smtp(monkeypatch):
    import smtplib

    monkeypatch.setattr(smtplib, "SMTP", lambda *a, **k: pytest.fail("SMTP must not be opened"))
    lines: list[str] = []
    assert sup.deliver_alert({}, "subject", "body", lines.append) is False
    assert lines == ["WARNING: alert email not sent (AutomationError: EMAIL_USER/EMAIL_PASS not set): subject"]


def test_smtp_failure_never_raises_or_changes_the_exit_code(tmp_path, monkeypatch):
    _wire(monkeypatch, tmp_path, "postclose", _postclose_jobs(), now=POSTCLOSE_NOW,
          failing={"scripts/refresh_earnings_calendar.py": "boom error"})

    def explode(env, subject, body):
        raise OSError("network unreachable")

    monkeypatch.setattr(sup, "_send_alert_email", explode)
    assert sup.main(_argv(tmp_path, "postclose")) == 1
    _, log = _log_text(tmp_path, "2026-09-30", "postclose")
    assert "WARNING: alert email not sent (OSError: network unreachable)" in log


def test_sender_uses_the_reports_gmail_convention(monkeypatch):
    import smtplib

    seen: dict = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout=None):
            seen["server"] = (host, port)
            seen["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def starttls(self):
            seen["tls"] = True

        def login(self, user, password):
            seen["login"] = (user, password)

        def sendmail(self, sender, recipients, text):
            seen["mail"] = (sender, recipients, text)

    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    sup._send_alert_email(CREDS, "[New Seasonals] x", "plain body\n")
    assert seen["server"] == ("smtp.gmail.com", 587)
    assert isinstance(seen["timeout"], (int, float)) and 0 < seen["timeout"] <= 60
    assert seen["tls"] and seen["login"] == ("sender@example.test", "secret")
    sender, recipients, text = seen["mail"]
    assert recipients == ["mckinleyslade@gmail.com"]
    assert "Content-Type: text/plain" in text and "Subject: [New Seasonals] x" in text

    sup._send_alert_email(
        {**CREDS, "NEW_SEASONALS_ALERT_RECIPIENTS": "a@example.test, b@example.test"}, "s", "b"
    )
    assert seen["mail"][1] == ["a@example.test", "b@example.test"]


class _HealthProcess:
    def __init__(self, lines, rc):
        self.lines, self.rc = lines, rc

    def stream(self, argv, *, cwd, env, timeout_seconds, logger):
        for line in self.lines:
            logger.line(line)
        return self.rc


@pytest.mark.parametrize("rc", [0, 1])
def test_health_emails_its_summary_block_only_on_fail(tmp_path, monkeypatch, rc):
    now = dt.datetime(2026, 10, 1, 11, 30, tzinfo=UTC)  # 07:30 ET
    fails = "1 FAIL" if rc else "0 FAIL"
    lines = [
        "[FAIL] automation:scan_am: indeterminate via local" if rc else "[OK] automation:scan_am: ok",
        "",
        f"===== SUMMARY: {fails}, 0 WARN, 40 OK =====",
    ] + (["  [FAIL] automation:scan_am: indeterminate via local"] if rc else [])

    class _Runtime:
        env = dict(CREDS)
        process = _HealthProcess(lines, rc)

    sent: list = []
    monkeypatch.setattr(sup, "_utc_now", lambda: now)
    monkeypatch.setattr(sup, "_make_runtime",
                        lambda args, controller_only=False: (_Runtime(), sup.InMemoryReceiptStore()))
    monkeypatch.setattr(sup, "_send_alert_email",
                        lambda env, subject, body: sent.append((subject, body)))
    state = tmp_path / "state"
    assert sup.main(["health", "--config-root", str(tmp_path), "--state-root", str(state),
                     "--repo-root", str(tmp_path), "--skip-tests"]) == rc
    if not rc:
        assert sent == []
        return
    (subject, body), = sent
    assert subject == "[New Seasonals] health 2026-10-01: 1 FAIL"
    assert "===== SUMMARY: 1 FAIL, 0 WARN, 40 OK =====" in body
    assert "  [FAIL] automation:scan_am: indeterminate via local" in body
    assert "[OK]" not in body


def _lock_is_free(state) -> bool:
    try:
        with sup.GlobalFileLock(state / "automation_supervisor.lock", timeout_seconds=0):
            return True
    except sup.LockUnavailable:
        return False


def test_pipeline_alert_is_sent_only_after_the_lock_is_released(tmp_path, monkeypatch):
    _wire(monkeypatch, tmp_path, "postclose", _postclose_jobs(), now=POSTCLOSE_NOW,
          failing={"scripts/refresh_earnings_calendar.py": "boom error"})
    observed: list[bool] = []
    monkeypatch.setattr(sup, "_send_alert_email",
                        lambda env, subject, body: observed.append(_lock_is_free(tmp_path / "state")))
    assert sup.main(_argv(tmp_path, "postclose")) == 1
    assert observed == [True]


def test_health_alert_is_sent_only_after_the_lock_is_released(tmp_path, monkeypatch):
    now = dt.datetime(2026, 10, 1, 11, 30, tzinfo=UTC)
    state = tmp_path / "state"

    class _Runtime:
        env = dict(CREDS)
        process = _HealthProcess(["===== SUMMARY: 1 FAIL, 0 WARN, 1 OK ====="], 1)

    observed: list[bool] = []
    monkeypatch.setattr(sup, "_utc_now", lambda: now)
    monkeypatch.setattr(sup, "_make_runtime",
                        lambda args, controller_only=False: (_Runtime(), sup.InMemoryReceiptStore()))
    monkeypatch.setattr(sup, "_send_alert_email",
                        lambda env, subject, body: observed.append(_lock_is_free(state)))
    assert sup.main(["health", "--config-root", str(tmp_path), "--state-root", str(state),
                     "--repo-root", str(tmp_path), "--skip-tests"]) == 1
    assert observed == [True]


def test_a_raising_builder_never_changes_the_exit_code(tmp_path, monkeypatch):
    sent, _ = _wire(monkeypatch, tmp_path, "postclose", _postclose_jobs(), now=POSTCLOSE_NOW,
                    failing={"scripts/refresh_earnings_calendar.py": "boom error"})

    def explode(**kwargs):
        raise ValueError("synthetic builder bug")

    monkeypatch.setattr(sup, "build_failure_alert", explode)
    assert sup.main(_argv(tmp_path, "postclose")) == 1
    assert sent == []
    _, log = _log_text(tmp_path, "2026-09-30", "postclose")
    assert "WARNING: alert not built (ValueError: synthetic builder bug)" in log


def test_secrets_are_redacted_from_the_mailed_body(tmp_path, monkeypatch):
    leaked = "HTTPError: GET https://www.alphavantage.co/query?function=X&apikey=ABC123SECRET&horizon=3month"
    sent, _ = _wire(monkeypatch, tmp_path, "postclose", _postclose_jobs(), now=POSTCLOSE_NOW,
                    failing={"scripts/refresh_earnings_calendar.py": leaked})
    assert sup.main(_argv(tmp_path, "postclose")) == 1
    (_, _, body), = sent
    assert "ABC123SECRET" not in body
    assert "apikey=<redacted>&horizon=3month" in body
    assert sup.redact_secrets("password=hunter2 Token=abc api_key=k secret=s") == (
        "password=<redacted> Token=<redacted> api_key=<redacted> secret=<redacted>"
    )


def test_health_with_the_lock_held_emails_even_when_the_battery_passes(tmp_path, monkeypatch):
    now = dt.datetime(2026, 10, 1, 11, 30, tzinfo=UTC)
    state = tmp_path / "state"

    class _Runtime:
        env = dict(CREDS)
        process = _HealthProcess(["===== SUMMARY: 0 FAIL, 0 WARN, 40 OK ====="], 0)

    sent: list = []
    monkeypatch.setattr(sup, "_utc_now", lambda: now)
    monkeypatch.setattr(sup, "_make_runtime",
                        lambda args, controller_only=False: (_Runtime(), sup.InMemoryReceiptStore()))
    monkeypatch.setattr(sup, "_send_alert_email",
                        lambda env, subject, body: sent.append((subject, body)))
    with sup.GlobalFileLock(state / "automation_supervisor.lock", timeout_seconds=0):
        rc = sup.main(["health", "--config-root", str(tmp_path), "--state-root", str(state),
                       "--repo-root", str(tmp_path), "--skip-tests"])
    assert rc == 1
    (subject, body), = sent
    assert subject == "[New Seasonals] health 2026-10-01: supervisor lock still held"
    assert "FAIL health: primary still holds the supervisor lock since " in body
    assert "===== SUMMARY: 0 FAIL, 0 WARN, 40 OK =====" in body
