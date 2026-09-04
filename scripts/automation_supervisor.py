"""Local-primary production automation with receipt-gated GitHub fallback.

This module is intentionally usable from both Windows Task Scheduler and a
GitHub-hosted fallback controller.  The local machine runs the production
commands; GitHub Actions is dispatched only when a local component fails or a
fallback controller finds a component missing after its due time.

Safety properties:

* every component is claimed through an R2 receipt before side effects start;
* Event and Trend are separately receipted, duplicate-sensitive components;
* a single OS file lock prevents overlapping local pipelines;
* secrets are loaded only from explicitly supplied paths and are never logged;
* producer jobs validate their R2 object sizes (and freshness where useful);
* private/shared site builds are dispatch-only and can never run locally here.

Run ``python scripts/automation_supervisor.py --help`` for the operational
commands.  ``plan`` and ``run --dry-run`` have no external side effects.

Receipt state table (what ``run_job`` does with the existing ``latest.json``
for the job and ET date it is asked to run; the same table governs the
04:10 primary, the 05:45 ``run-pipeline --retry`` second chance, and the
GitHub-only ``fallback-due`` controller):

===============================  ==========  ======================================
existing latest receipt          effective   action
===============================  ==========  ======================================
absent                           missing     claim and run (local) / dispatch (github)
success (any source)             success     skip; nothing runs, nothing is written
indeterminate (manual_review,    indeterm.   skip; NEVER re-run. A side effect may
  or the local_side_effect                    have happened. Only ``resolve`` clears
  marker written before the                   it. Pre-existing ones are reported and
  first side-effecting step)                  do not fail a retry/controller run.
running, lease live              running     skip ("live lease"); another writer owns it
running, lease expired,          expired     adopt the GitHub run by token; never a
  phase github_reconcile                      second dispatch
running, lease expired, phase    expired     RE-RUN with a new token. The child never
  local_pre_side_effect /                     reached its side-effecting step (the
  local_retryable                             2026-09-03 scan_am shape), so a fresh
                                              run is safe and is the whole point of
                                              the 05:45 retry task.
failure (retryable, or an        failure     re-run with a new token
  operator retryable_failure)
===============================  ==========  ======================================

``effective_status`` is the read-side normalisation used by ``status``; a
``running`` receipt whose lease has passed reads ``expired`` there, in the
run log ("expired <job>: ... reclaiming"), and in the health battery
(``scripts/repo_health_check.py`` reports an EXPIRED running lease as FAIL).
The receipt itself is never rewritten by a reader.

Exit codes: ``run-pipeline`` (primary) fails on any failure/blocked/
indeterminate outcome. ``run-pipeline --retry`` and ``fallback-due`` fail only
on outcomes THIS invocation produced; a receipt that was already
indeterminate before the run started is printed with the ``resolve`` command
and does not turn the run red (an older ambiguity is the operator's, not the
controller's). A dependent blocked only by another writer's LIVE lease is
pre-existing too: leaving a live claim alone is the correct action.

A held supervisor lock is never a traceback. ``run-pipeline --retry`` prints
``No action`` and exits 0 (the primary is still working); ``fallback-due`` and
``health`` print a FAIL line naming how long the lock has been held and exit
1, and ``health`` still runs its read-only battery without the lock.

Runtime logs go to ``<state-root>/logs/<ET date>/<pipeline>-<stub>.log``.
The Task Scheduler runner passes ``--state-root <ConfigRoot>/artifacts/automation``
so logs, the lock, and the health receipt cache live OUTSIDE the pinned
runtime worktree and survive a cutover.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import glob
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol
from zoneinfo import ZoneInfo

# Direct script execution sets sys.path[0] to the scripts directory. Ensure
# repository-root modules such as cache_io remain importable in Task Scheduler
# and GitHub workflow subprocesses.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ET = ZoneInfo("America/New_York")
UTC = dt.timezone.utc
RECEIPT_PREFIX = "automation/receipts/v1"
DEFAULT_CUTOVER_DATE_ET = dt.date(2026, 8, 28)
R2_REQUIRED = (
    "R2_ACCOUNT_ID",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "R2_BUCKET",
)
LEASE_GRACE_SECONDS = 900


class AutomationError(RuntimeError):
    """Base class for fail-closed supervisor errors."""


class LockUnavailable(AutomationError):
    """Raised when another supervisor process holds the global lock."""


class ValidationError(AutomationError):
    """Raised when a supposedly successful producer cannot be verified."""


class DispatchError(AutomationError):
    """Raised when GitHub fallback cannot be dispatched or does not succeed."""


class DispatchAcceptedError(DispatchError):
    """The workflow was submitted, but its terminal result is not successful."""


class DispatchRunNotFound(DispatchAcceptedError):
    """A previously accepted automation token is not visible in GitHub runs."""


@dataclass(frozen=True)
class CommandSpec:
    label: str
    argv: tuple[str, ...]
    timeout_seconds: int = 3600
    side_effecting: bool = False


@dataclass(frozen=True)
class WorkflowSpec:
    workflow: str
    inputs: tuple[tuple[str, str], ...] = ()
    timeout_seconds: int = 3600

    def input_dict(self) -> dict[str, str]:
        return dict(self.inputs)


@dataclass(frozen=True)
class OutputSpec:
    """Map one local file (or glob) to one or more canonical R2 objects.

    ``r2_key`` is used for a single file.  For a glob, ``r2_prefix`` is
    joined with each basename after optionally removing ``strip_suffix``.
    ContentLength must exactly match the local file for every match.
    """

    local_pattern: str
    r2_key: str | None = None
    r2_prefix: str | None = None
    strip_suffix: str = ""
    required: bool = True
    min_bytes: int = 1
    require_recent_upload: bool = False

    def __post_init__(self) -> None:
        if bool(self.r2_key) == bool(self.r2_prefix):
            raise ValueError("OutputSpec requires exactly one of r2_key or r2_prefix")


@dataclass(frozen=True)
class JobSpec:
    id: str
    description: str
    commands: tuple[CommandSpec, ...] = ()
    workflow: WorkflowSpec | None = None
    required_env: tuple[str, ...] = ()
    env_overrides: tuple[tuple[str, str], ...] = ()
    local_gate: str | None = None
    outputs: tuple[OutputSpec, ...] = ()
    dispatch_only: bool = False
    duplicate_sensitive: bool = False
    rerun_safe: bool = False
    depends_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.dispatch_only and self.commands:
            raise ValueError(f"dispatch-only job {self.id} cannot have local commands")
        if self.dispatch_only and self.workflow is None:
            raise ValueError(f"dispatch-only job {self.id} requires a workflow")
        if not self.dispatch_only and not self.commands:
            raise ValueError(f"local job {self.id} requires at least one command")


@dataclass(frozen=True)
class PipelineSpec:
    id: str
    description: str
    cadence: str
    run_at_et: dt.time
    fallback_at_et: dt.time
    fallback_until_et: dt.time
    jobs: tuple[JobSpec, ...]
    # Optional local second-chance window (``run-pipeline --retry``). It may
    # overlap the GitHub fallback window: the R2 receipt CAS arbitrates, a
    # live lease from either side blocks the other, and an expired
    # pre-side-effect lease is reclaimed by whichever wakes first.
    retry_at_et: dt.time | None = None
    retry_until_et: dt.time | None = None

    def active_on(self, day: dt.date) -> bool:
        if self.cadence == "weekdays":
            return day.weekday() < 5
        if self.cadence == "monday":
            return day.weekday() == 0
        if self.cadence == "sunday":
            return day.weekday() == 6
        raise ValueError(f"unsupported cadence: {self.cadence}")

    def fallback_is_due(self, now_et: dt.datetime) -> bool:
        if now_et.tzinfo is None:
            raise ValueError("fallback due checks require a timezone-aware datetime")
        local = now_et.astimezone(ET)
        clock = local.timetz().replace(tzinfo=None)
        return (
            self.active_on(local.date())
            and self.fallback_at_et <= clock <= self.fallback_until_et
        )

    def local_is_due(self, now_et: dt.datetime) -> bool:
        """Whether a Task Scheduler launch is still safe to begin locally.

        ``StartWhenAvailable`` is useful for short sleep/reboot interruptions,
        but Windows can otherwise replay a missed trigger many hours or days
        late. The local primary owns only the interval before the guarded
        GitHub fallback window; after that boundary the remote controller owns
        recovery and duplicate prevention.
        """
        if now_et.tzinfo is None:
            raise ValueError("local due checks require a timezone-aware datetime")
        local = now_et.astimezone(ET)
        clock = local.timetz().replace(tzinfo=None)
        return (
            self.active_on(local.date())
            and self.run_at_et <= clock < self.fallback_at_et
        )

    def retry_is_due(self, now_et: dt.datetime) -> bool:
        """Whether the local second-chance task may re-examine today's receipts.

        Bounded like ``local_is_due`` so a ``StartWhenAvailable`` replay hours
        late cannot start a premarket pipeline at lunchtime. Pipelines without
        a retry window never return True.
        """
        if self.retry_at_et is None or self.retry_until_et is None:
            return False
        if now_et.tzinfo is None:
            raise ValueError("retry due checks require a timezone-aware datetime")
        local = now_et.astimezone(ET)
        clock = local.timetz().replace(tzinfo=None)
        return (
            self.active_on(local.date())
            and self.retry_at_et <= clock < self.retry_until_et
        )


def _py(
    label: str,
    *args: str,
    timeout: int = 3600,
    side_effecting: bool = False,
) -> CommandSpec:
    return CommandSpec(
        label=label,
        argv=("{python}", *args),
        timeout_seconds=timeout,
        side_effecting=side_effecting,
    )


def _out(
    local: str,
    key: str,
    *,
    minimum: int = 1,
    recent: bool = True,
    required: bool = True,
) -> OutputSpec:
    return OutputSpec(
        local_pattern=local,
        r2_key=key,
        min_bytes=minimum,
        require_recent_upload=recent,
        required=required,
    )


R2_ENV = R2_REQUIRED
SHEETS_ENV = ("GCP_JSON",)
EMAIL_ENV = ("EMAIL_USER", "EMAIL_PASS")


def build_catalog() -> dict[str, PipelineSpec]:
    """Return the authoritative New York-time local pipeline catalog.

    The command ordering mirrors the former scheduled workflow ordering.  A
    workflow input named ``automation_token`` is added by the dispatcher, not
    repeated in this catalog.
    """

    publish = lambda group: _py(
        f"publish {group} private-site input",
        "scripts/site_r2_pipeline.py",
        "publish-group",
        "--group",
        group,
        "--local-primary",
        timeout=600,
        side_effecting=True,
    )
    pull_master = _py(
        "pull canonical master prices",
        "scripts/automation_supervisor.py",
        "_r2-download",
        "master_prices.parquet",
        "data/master_prices.parquet",
        "--required",
        timeout=600,
    )
    pull_scan = _py(
        "pull fail-closed scanner inputs",
        "scripts/pull_scan_caches.py",
        timeout=900,
    )
    pull_risk = _py(
        "pull canonical append-only risk state",
        "scripts/pull_scan_caches.py",
        "--set",
        "risk",
        timeout=300,
    )
    pull_cboe = _py(
        "pull canonical CBOE put/call state",
        "scripts/automation_supervisor.py",
        "_r2-download",
        "cboe_putcall.parquet",
        "data/cboe_putcall.parquet",
        "--required",
        timeout=300,
    )

    premarket = PipelineSpec(
        id="premarket",
        description="Settled-cache refresh, event sleeve, full scan, and cloud site builds",
        cadence="weekdays",
        run_at_et=dt.time(4, 10),
        fallback_at_et=dt.time(5, 20),
        fallback_until_et=dt.time(8, 55),
        # Local second chance (2026-09-03 stall): the 05:45 ET S4U task re-runs
        # this pipeline through the receipt table; every success is a no-op.
        # 05:45, not 05:30: master_prices_am holds a 70-minute lease, so a
        # 04:10 claim is still live at 05:30 and the retry would skip the very
        # job most likely to have stalled. Late replays past 07:00 leave
        # recovery to the 07:30 health probe and the GitHub controller.
        retry_at_et=dt.time(5, 45),
        retry_until_et=dt.time(7, 0),
        jobs=(
            JobSpec(
                id="cboe_am",
                description="Refresh prior-session CBOE put/call cache",
                commands=(
                    pull_cboe,
                    _py(
                        "backfill CBOE put/call",
                        "-m",
                        "cboe_putcall",
                        "--start",
                        "2024-01-01",
                        "--assert-fresh-bd",
                        "2",
                        side_effecting=False,
                    ),
                    publish("cboe"),
                ),
                workflow=WorkflowSpec(
                    "update_cboe_putcall.yml", (), 1200
                ),
                required_env=R2_ENV,
                rerun_safe=True,
                outputs=(_out("data/cboe_putcall.parquet", "cboe_putcall.parquet", minimum=512),),
            ),
            JobSpec(
                id="master_prices_am",
                description="Refresh settled daily bars without today's placeholder",
                commands=(
                    pull_master,
                    _py(
                        "increment master prices (exclude today)",
                        "scripts/update_master_prices.py",
                        "--exclude-today",
                        timeout=2700,
                        side_effecting=True,
                    ),
                ),
                workflow=WorkflowSpec(
                    "update_master_prices.yml", (("mode", "am"),), 3600
                ),
                required_env=R2_ENV,
                rerun_safe=True,
                outputs=(_out("data/master_prices.parquet", "master_prices.parquet", minimum=1_000_000),),
            ),
            JobSpec(
                id="risk_am",
                description="Correct the final risk row with settled prices (no email)",
                commands=(
                    pull_risk,
                    _py(
                        "risk data-only correction",
                        "daily_risk_report.py",
                        "--data-only",
                        "--refresh-last",
                        timeout=2700,
                    ),
                    publish("risk"),
                ),
                workflow=WorkflowSpec("risk_report.yml", (("mode", "data_only"),), 3600),
                required_env=R2_ENV,
                rerun_safe=True,
                outputs=(
                    _out("data/rd2_fragility.parquet", "rd2_fragility.parquet", minimum=1_000),
                    _out("data/rd2_environment.json", "rd2_environment.json", minimum=50),
                    _out("data/dial_sleeve_paper.json", "dial_sleeve_paper.json", minimum=50),
                ),
                depends_on=("master_prices_am",),
            ),
            JobSpec(
                id="event_sleeve_am",
                description="Once-only event-sleeve staging before the scanner",
                commands=(
                    pull_scan,
                    _py(
                        "stage event sleeve",
                        "event_sleeve.py",
                        timeout=1800,
                        side_effecting=True,
                    ),
                ),
                workflow=WorkflowSpec("event_sleeve.yml", (), 1800),
                required_env=R2_ENV + SHEETS_ENV,
                outputs=(
                    _out("data/event_sleeve_state.json", "event_sleeve_state.json", minimum=20),
                    _out(
                        "data/event_sleeve_last_actions.json",
                        "event_sleeve_last_actions.json",
                        minimum=20,
                    ),
                    _out(
                        "data/event_sleeve_journal.jsonl",
                        "event_sleeve_journal.jsonl",
                        minimum=1,
                        recent=False,
                        required=False,
                    ),
                ),
                duplicate_sensitive=True,
                depends_on=("master_prices_am",),
            ),
            JobSpec(
                id="scan_am",
                description="Unified liquid + overflow premarket scan",
                commands=(
                    pull_scan,
                    _py(
                        "run unified scanner",
                        "daily_scan.py",
                        "--scope=all",
                        "--bookend=am",
                        timeout=5400,
                        side_effecting=True,
                    ),
                    publish("exposure"),
                ),
                workflow=WorkflowSpec(
                    "daily_screener.yml",
                    (
                        ("bookend", "am"),
                        ("run_event_sleeve", "false"),
                        ("deploy_after_scan", "false"),
                    ),
                    7200,
                ),
                required_env=R2_ENV + SHEETS_ENV + EMAIL_ENV,
                env_overrides=(("OVERFLOW_UNIVERSE_ACTIVE", "0"),),
                outputs=(_out("data/exposure_state.json", "exposure_state.json", minimum=50),),
                depends_on=("master_prices_am", "risk_am"),
            ),
            JobSpec(
                id="private_site_am",
                description="Cloud-only private-site build and deployment",
                workflow=WorkflowSpec("deploy_site.yml", (), 7200),
                dispatch_only=True,
                rerun_safe=True,
                depends_on=("scan_am",),
            ),
            JobSpec(
                id="shared_site_am",
                description="Cloud-only teammate-safe seasonality deployment",
                workflow=WorkflowSpec("deploy_shared_seasonals.yml", (), 3600),
                dispatch_only=True,
                rerun_safe=True,
                depends_on=("scan_am",),
            ),
        ),
    )

    discretionary = PipelineSpec(
        id="discretionary",
        description="Research-only 0-2 name premarket attention list",
        cadence="weekdays",
        run_at_et=dt.time(8, 35),
        fallback_at_et=dt.time(8, 50),
        fallback_until_et=dt.time(9, 20),
        jobs=(
            JobSpec(
                id="discretionary_focus",
                description="Build, publish, and email the research-only focus list",
                commands=(
                    _py(
                        "gate NYSE delivery window",
                        "scripts/check_discretionary_focus_session.py",
                        "--delivery-window",
                        "--require-allowed",
                    ),
                    _py("pull discretionary inputs", "scripts/pull_discretionary_focus_inputs.py"),
                    _py(
                        "refresh isolated overflow prices",
                        "scripts/build_overflow_prices.py",
                        "--no-upload",
                        timeout=2700,
                    ),
                    _py(
                        "refresh isolated overflow earnings",
                        "scripts/build_earnings_calendar.py",
                        "--overflow-staging",
                        "--fail-on-fetch-errors",
                        "--no-upload",
                        timeout=2700,
                    ),
                    _py(
                        "build FINAL research shortlist",
                        "scripts/build_discretionary_focus.py",
                        "--phase",
                        "FINAL",
                        "--fetch-news",
                        "--output",
                        "data/discretionary_focus/current.json",
                        timeout=2700,
                    ),
                    _py(
                        "recheck NYSE delivery window",
                        "scripts/check_discretionary_focus_session.py",
                        "--delivery-window",
                        "--require-allowed",
                    ),
                    _py(
                        "publish discretionary focus",
                        "scripts/publish_discretionary_focus.py",
                        "--input",
                        "data/discretionary_focus/current.json",
                        side_effecting=True,
                    ),
                    _py(
                        "send at-most-once focus email",
                        "scripts/send_discretionary_focus_email.py",
                        "--input",
                        "data/discretionary_focus/current.json",
                        "--receipt",
                        "data/discretionary_focus/email_receipt.json",
                        "--persist-receipt-r2",
                        side_effecting=True,
                    ),
                ),
                workflow=WorkflowSpec(
                    "discretionary_focus.yml", (("delivery_mode", "publish_and_email"),), 2700
                ),
                required_env=R2_ENV + ("FMP_API_KEY",) + EMAIL_ENV,
                local_gate="discretionary_delivery_window",
                outputs=(
                    _out(
                        "data/discretionary_focus/current.json",
                        "discretionary_focus/current.json",
                        minimum=50,
                    ),
                    _out(
                        "data/discretionary_focus/email_receipt.json",
                        "discretionary_focus/email_receipt.json",
                        minimum=50,
                    ),
                ),
                duplicate_sensitive=True,
            ),
        ),
    )

    execution = PipelineSpec(
        id="execution",
        description="Live-account execution status email",
        cadence="weekdays",
        run_at_et=dt.time(16, 30),
        fallback_at_et=dt.time(16, 50),
        fallback_until_et=dt.time(20, 0),
        jobs=(
            JobSpec(
                id="execution_report",
                description="Send the nightly execution report once at 16:30 ET",
                commands=(
                    _py(
                        "send execution report",
                        "daily_execution_report.py",
                        "--force",
                        side_effecting=True,
                    ),
                ),
                workflow=WorkflowSpec("execution_report.yml", (), 1800),
                required_env=R2_ENV + EMAIL_ENV + ("STATUS_TOKEN",),
            ),
        ),
    )

    postclose = PipelineSpec(
        id="postclose",
        description="Post-close caches, reports, sleeves, scan, and cloud deployments",
        cadence="weekdays",
        run_at_et=dt.time(17, 10),
        fallback_at_et=dt.time(18, 45),
        fallback_until_et=dt.time(23, 55),
        jobs=(
            JobSpec(
                id="master_prices_pm",
                description="Append the completed session to canonical daily prices",
                commands=(
                    pull_master,
                    _py(
                        "increment master prices (include close)",
                        "scripts/update_master_prices.py",
                        timeout=2700,
                        side_effecting=True,
                    ),
                ),
                workflow=WorkflowSpec(
                    "update_master_prices.yml", (("mode", "pm"),), 3600
                ),
                required_env=R2_ENV,
                rerun_safe=True,
                outputs=(_out("data/master_prices.parquet", "master_prices.parquet", minimum=1_000_000),),
            ),
            JobSpec(
                id="risk_pm",
                description="Full daily risk report, email, and private-site inputs",
                commands=(
                    pull_risk,
                    _py(
                        "run full risk report",
                        "daily_risk_report.py",
                        timeout=2700,
                        side_effecting=True,
                    ),
                    publish("risk"),
                ),
                workflow=WorkflowSpec("risk_report.yml", (("mode", "full"),), 3600),
                required_env=R2_ENV + EMAIL_ENV,
                outputs=(
                    _out("data/rd2_fragility.parquet", "rd2_fragility.parquet", minimum=1_000),
                    _out("data/rd2_environment.json", "rd2_environment.json", minimum=50),
                    _out("data/dial_sleeve_paper.json", "dial_sleeve_paper.json", minimum=50),
                ),
                depends_on=("master_prices_pm",),
            ),
            JobSpec(
                id="verify_fills",
                description="Verify post-close staged-order fills in Sheets",
                commands=(
                    _py(
                        "verify fills",
                        "verify_fills.py",
                        timeout=1800,
                        side_effecting=True,
                    ),
                ),
                workflow=WorkflowSpec("verify_fills.yml", (), 1800),
                required_env=SHEETS_ENV,
            ),
            JobSpec(
                id="harvest_fills",
                description="Copy the broker's executions ring into the durable fills store",
                commands=(
                    _py(
                        "harvest broker fills",
                        "scripts/harvest_fills.py",
                        "--summary-json",
                        "data/live_fills_status.json",
                        timeout=900,
                        side_effecting=True,
                    ),
                ),
                # Local-only by design: the ring lives behind the broker's read
                # token, and a GitHub runner has no reason to hold it.
                required_env=R2_ENV + ("STATUS_TOKEN",),
                rerun_safe=True,
                outputs=(
                    _out("data/live_fills.parquet", "live_fills.parquet", minimum=1_000),
                    _out("data/live_fills_status.json", "live_fills_status.json", minimum=50),
                ),
            ),
            JobSpec(
                id="earnings_and_grades",
                description="Refresh FMP earnings calendar and analyst-grade history",
                commands=(
                    _py(
                        "build earnings calendar",
                        "scripts/build_earnings_calendar.py",
                        timeout=3600,
                        side_effecting=True,
                    ),
                    _py(
                        "build analyst grades",
                        "scripts/build_analyst_grades.py",
                        timeout=3600,
                        side_effecting=True,
                    ),
                ),
                workflow=WorkflowSpec("build_earnings_calendar.yml", (), 7200),
                required_env=R2_ENV + ("FMP_API_KEY",),
                rerun_safe=True,
                outputs=(
                    _out("data/earnings_calendar.parquet", "earnings_calendar.parquet", minimum=10_000),
                    _out("data/analyst_grades.parquet", "analyst_grades.parquet", minimum=1_000),
                ),
            ),
            JobSpec(
                id="portfolio_report",
                description="Portfolio health email and open-position snapshot",
                commands=(
                    _py(
                        "pull portfolio-report caches",
                        "scripts/pull_scan_caches.py",
                        "--set",
                        "report",
                    ),
                    _py(
                        "run portfolio report",
                        "daily_portfolio_report.py",
                        timeout=7200,
                        side_effecting=True,
                    ),
                ),
                workflow=WorkflowSpec("portfolio_report.yml", (), 9000),
                required_env=R2_ENV + SHEETS_ENV + EMAIL_ENV,
                env_overrides=(("OVERFLOW_UNIVERSE_ACTIVE", "0"),),
                depends_on=("master_prices_pm", "earnings_and_grades"),
            ),
            JobSpec(
                id="cboe_pm",
                description="Post-close CBOE backstop refresh",
                commands=(
                    pull_cboe,
                    _py(
                        "backfill CBOE put/call",
                        "-m",
                        "cboe_putcall",
                        "--start",
                        "2024-01-01",
                        "--assert-fresh-bd",
                        "2",
                    ),
                    publish("cboe"),
                ),
                workflow=WorkflowSpec(
                    "update_cboe_putcall.yml", (), 1200
                ),
                required_env=R2_ENV,
                rerun_safe=True,
                outputs=(_out("data/cboe_putcall.parquet", "cboe_putcall.parquet", minimum=512),),
            ),
            JobSpec(
                id="trend_sleeve",
                description="Once-only month-end trend-sleeve rebalance gate",
                commands=(
                    pull_master,
                    _py(
                        "run trend sleeve",
                        "trend_sleeve.py",
                        timeout=1800,
                        side_effecting=True,
                    ),
                ),
                workflow=WorkflowSpec(
                    "trend_sleeve.yml",
                    (("force", "false"), ("reset_state", "false")),
                    2400,
                ),
                required_env=R2_ENV + SHEETS_ENV,
                outputs=(
                    _out(
                        "data/trend_sleeve_state.json",
                        "trend_sleeve_state.json",
                        minimum=20,
                        recent=False,
                        required=False,
                    ),
                ),
                duplicate_sensitive=True,
                depends_on=("master_prices_pm",),
            ),
            JobSpec(
                id="intraday_prices",
                description="Pull, increment, and mirror all 15-minute parquets",
                commands=(
                    _py(
                        "pull canonical intraday parquets",
                        "scripts/automation_supervisor.py",
                        "_pull-intraday",
                        timeout=3600,
                    ),
                    _py(
                        "update intraday parquets",
                        "scripts/update_intraday_yfinance.py",
                        "--upload",
                        timeout=5400,
                        side_effecting=True,
                    ),
                ),
                workflow=WorkflowSpec("update_intraday_prices.yml", (), 7200),
                required_env=R2_ENV,
                rerun_safe=True,
                outputs=(
                    OutputSpec(
                        "data/intraday/*_15min.parquet",
                        r2_prefix="intraday/15min",
                        strip_suffix="_15min",
                        min_bytes=100,
                    ),
                    _out(
                        "data/intraday/_meta.parquet",
                        "intraday/15min/_meta.parquet",
                        minimum=100,
                    ),
                ),
            ),
            JobSpec(
                id="scan_pm",
                description="Unified post-close liquid + overflow scan",
                commands=(
                    pull_scan,
                    _py(
                        "run unified scanner",
                        "daily_scan.py",
                        "--scope=all",
                        "--bookend=pm",
                        timeout=5400,
                        side_effecting=True,
                    ),
                ),
                workflow=WorkflowSpec(
                    "daily_screener.yml",
                    (
                        ("bookend", "pm"),
                        ("run_event_sleeve", "false"),
                        ("deploy_after_scan", "false"),
                    ),
                    7200,
                ),
                required_env=R2_ENV + SHEETS_ENV + EMAIL_ENV,
                env_overrides=(("OVERFLOW_UNIVERSE_ACTIVE", "0"),),
                depends_on=("master_prices_pm", "risk_pm", "earnings_and_grades"),
            ),
            JobSpec(
                id="macro_releases",
                description="Refresh normalized U.S. macro release history",
                commands=(
                    _py(
                        "build macro release history",
                        "scripts/build_macro_releases.py",
                        side_effecting=True,
                    ),
                ),
                workflow=WorkflowSpec("build_macro_releases.yml", (("full", "false"),), 3600),
                required_env=R2_ENV + ("FMP_API_KEY",),
                rerun_safe=True,
                outputs=(
                    _out(
                        "data/macro_release_history.parquet",
                        "macro_release_history.parquet",
                        minimum=1_000,
                    ),
                ),
            ),
            JobSpec(
                id="private_site_pm",
                description="Cloud-only private-site build and deployment",
                workflow=WorkflowSpec("deploy_site.yml", (), 7200),
                dispatch_only=True,
                rerun_safe=True,
                depends_on=("scan_pm",),
            ),
            JobSpec(
                id="shared_site_pm",
                description="Cloud-only teammate-safe seasonality deployment",
                workflow=WorkflowSpec("deploy_shared_seasonals.yml", (), 3600),
                dispatch_only=True,
                rerun_safe=True,
                depends_on=("scan_pm",),
            ),
        ),
    )

    weekly_indicator = PipelineSpec(
        id="indicator",
        description="Precompute and upload strategy indicator caches",
        cadence="monday",
        run_at_et=dt.time(3, 0),
        fallback_at_et=dt.time(3, 30),
        fallback_until_et=dt.time(23, 55),
        jobs=(
            JobSpec(
                id="indicator_cache",
                description="Build liquid and overflow indicator cache passes",
                commands=(
                    _py(
                        "build indicator cache",
                        "scripts/build_indicator_cache.py",
                        timeout=7200,
                        side_effecting=True,
                    ),
                ),
                workflow=WorkflowSpec("build_indicator_cache.yml", (), 7200),
                required_env=R2_ENV,
                rerun_safe=True,
                outputs=(
                    OutputSpec(
                        "data/bt_indicator_cache/*.parquet",
                        r2_prefix="bt_indicator_cache",
                        min_bytes=100,
                        require_recent_upload=False,
                    ),
                ),
            ),
        ),
    )

    weekly_rundown = PipelineSpec(
        id="weekly-rundown",
        description="Sunday market-rundown PDF and email",
        cadence="sunday",
        run_at_et=dt.time(8, 0),
        fallback_at_et=dt.time(8, 30),
        fallback_until_et=dt.time(23, 55),
        jobs=(
            JobSpec(
                id="weekly_rundown",
                description="Render and send the weekly market rundown",
                commands=(
                    _py(
                        "run weekly market rundown",
                        "weekly_market_rundown.py",
                        timeout=5400,
                        side_effecting=True,
                    ),
                ),
                workflow=WorkflowSpec("weekly_rundown.yml", (), 7200),
                required_env=EMAIL_ENV,
            ),
        ),
    )

    catalog = {
        p.id: p
        for p in (
            premarket,
            discretionary,
            execution,
            postclose,
            weekly_indicator,
            weekly_rundown,
        )
    }
    _validate_catalog(catalog)
    return catalog


def _validate_catalog(catalog: Mapping[str, PipelineSpec]) -> None:
    seen: set[str] = set()
    for pipeline in catalog.values():
        ids = {job.id for job in pipeline.jobs}
        if len(ids) != len(pipeline.jobs):
            raise ValueError(f"duplicate job id in pipeline {pipeline.id}")
        for job in pipeline.jobs:
            if job.id in seen:
                raise ValueError(f"job id must be globally unique: {job.id}")
            seen.add(job.id)
            missing = set(job.depends_on) - ids
            if missing:
                raise ValueError(f"{job.id} has unknown dependencies: {sorted(missing)}")
            if ("site" in job.id) and not job.dispatch_only:
                raise ValueError(f"site job must be dispatch-only: {job.id}")


CATALOG = build_catalog()


def _parse_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise AutomationError(f"required environment file is missing: {path}")
    try:
        from dotenv import dotenv_values  # type: ignore

        raw = dotenv_values(path)
        return {str(k): str(v) for k, v in raw.items() if k and v is not None}
    except ImportError:
        result: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            if value.startswith("export "):
                value = value[7:].lstrip()
            if "=" not in value:
                continue
            key, raw_value = value.split("=", 1)
            key = key.strip()
            raw_value = raw_value.strip()
            if len(raw_value) >= 2 and raw_value[0] == raw_value[-1] and raw_value[0] in "\"'":
                raw_value = raw_value[1:-1]
            result[key] = raw_value
        return result


def hydrate_environment(
    *,
    config_root: Path,
    gcp_json_path: Path,
    exec_env_path: Path,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build a child environment from explicit secret locations.

    No values are printed, returned in errors, or copied into the repository.
    The caller passes this mapping directly to subprocesses and R2/GitHub
    clients; the parent process environment is not mutated.
    """

    env = dict(base_env if base_env is not None else os.environ)
    env.update(_parse_env_file(config_root.resolve() / ".env"))
    env.update(_parse_env_file(exec_env_path.resolve()))

    if not gcp_json_path.is_file():
        raise AutomationError(f"required GCP JSON file is missing: {gcp_json_path}")
    try:
        gcp = json.loads(gcp_json_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AutomationError(f"GCP JSON is not valid JSON: {gcp_json_path}") from exc
    if not isinstance(gcp, dict) or not gcp:
        raise AutomationError(f"GCP JSON must contain a non-empty object: {gcp_json_path}")
    env["GCP_JSON"] = json.dumps(gcp, separators=(",", ":"))

    if not env.get("GH_TOKEN") and env.get("GH_PAT_NEW_SEASONALS"):
        env["GH_TOKEN"] = env["GH_PAT_NEW_SEASONALS"]
    # Windows creates redirected Python stdout with the active ANSI code page
    # unless these are explicit. The supervisor decodes child output as UTF-8,
    # so every child must emit UTF-8 as well.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["LOCAL_AUTOMATION_STRICT"] = "1"
    return env


def resolve_external_secret_paths(
    *,
    config_root: Path,
    gcp_json_path: Path | None,
    exec_env_path: Path | None,
    base_env: Mapping[str, str] | None = None,
) -> tuple[Path, Path]:
    """Resolve non-repository secret files without exposing paths in tasks.

    Explicit CLI paths win.  Otherwise the passed config root's ``.env`` may
    carry ``LOCAL_AUTOMATION_GCP_JSON_PATH`` and
    ``LOCAL_AUTOMATION_EXEC_ENV_PATH``.  The final machine-local default is
    the existing OneDrive ``trading_ibkr`` credential directory.  Only paths,
    never file contents, are resolved here.
    """

    bootstrap = dict(base_env if base_env is not None else os.environ)
    bootstrap.update(_parse_env_file(config_root.resolve() / ".env"))
    trading_root = Path.home() / "OneDrive" / "trading_ibkr"
    gcp = gcp_json_path or Path(
        bootstrap.get("LOCAL_AUTOMATION_GCP_JSON_PATH", str(trading_root / "credentials.json"))
    )
    execution = exec_env_path or Path(
        bootstrap.get("LOCAL_AUTOMATION_EXEC_ENV_PATH", str(trading_root / "exec_agent.env"))
    )
    return gcp.expanduser().resolve(), execution.expanduser().resolve()


class GlobalFileLock:
    """Non-reentrant cross-process lock backed by native OS file locking."""

    def __init__(self, path: Path, *, timeout_seconds: float = 1.0, poll_seconds: float = 0.1):
        self.path = path
        self.timeout_seconds = timeout_seconds
        self.poll_seconds = poll_seconds
        self._handle: Any = None

    def _try_lock(self) -> bool:
        assert self._handle is not None
        try:
            if os.name == "nt":
                import msvcrt

                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except (OSError, BlockingIOError):
            return False

    def acquire(self) -> GlobalFileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            self._handle.write(b"0")
            self._handle.flush()
        deadline = time.monotonic() + self.timeout_seconds
        while not self._try_lock():
            if time.monotonic() >= deadline:
                self._handle.close()
                self._handle = None
                raise LockUnavailable(f"another automation supervisor holds {self.path}")
            time.sleep(self.poll_seconds)
        # Stamp the mtime on every successful acquire so a blocked reader can
        # say how long the CURRENT holder has held it (the file itself is
        # created once and outlives every generation). Best effort: a failure
        # here must never cost the caller its lock.
        try:
            os.utime(self.path, None)
        except OSError:  # pragma: no cover - filesystem refusal is not fatal
            pass
        return self

    def holder_since_et(self) -> str | None:
        """ET timestamp of the last successful acquire, or None if unknowable."""
        try:
            stamp = self.path.stat().st_mtime
        except OSError:
            return None
        return (
            dt.datetime.fromtimestamp(stamp, tz=UTC)
            .astimezone(ET)
            .strftime("%Y-%m-%d %H:%M:%S ET")
        )

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> GlobalFileLock:  # noqa: PYI034
        return self.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


class RunLogger:
    def __init__(self, path: Path, *, echo: bool = True):
        self.path = path
        self.echo = echo
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8", buffering=1)

    def line(self, value: str = "") -> None:
        text = value.rstrip("\r\n")
        self._handle.write(text + "\n")
        if self.echo:
            print(text, flush=True)

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> RunLogger:  # noqa: PYI034
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class StdoutLogger:
    """Last-resort logger when the run log cannot be opened.

    An unusable state root (``logs`` occupied by a file, a revoked ACL, a full
    disk) must never swallow a FAIL line: the health probe still has to say why
    the morning is red. Interface-compatible with ``RunLogger``; writes nowhere
    but stdout.
    """

    path = None

    def line(self, value: str = "") -> None:
        print(value.rstrip("\r\n"), flush=True)

    def close(self) -> None:
        return None

    def __enter__(self) -> StdoutLogger:  # noqa: PYI034
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


@dataclass(frozen=True)
class CaptureResult:
    returncode: int
    stdout: str


class ProcessClient(Protocol):
    def stream(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        timeout_seconds: int,
        logger: RunLogger,
    ) -> int: ...

    def capture(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        timeout_seconds: int,
    ) -> CaptureResult: ...


class SubprocessClient:
    """No-shell subprocess seam with line-by-line stdout/stderr streaming."""

    def stream(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        timeout_seconds: int,
        logger: RunLogger,
    ) -> int:
        display = subprocess.list2cmdline(list(argv)) if os.name == "nt" else shlex.join(argv)
        logger.line(f"$ {display}")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        proc = subprocess.Popen(
            list(argv),
            cwd=str(cwd),
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            shell=False,
            creationflags=creationflags,
        )
        assert proc.stdout is not None
        output: queue.Queue[str | None] = queue.Queue()

        def reader() -> None:
            try:
                for line in proc.stdout:
                    output.put(line)
            finally:
                output.put(None)

        thread = threading.Thread(target=reader, name="automation-output-reader", daemon=True)
        thread.start()
        deadline = time.monotonic() + timeout_seconds
        done_reading = False
        while not done_reading or proc.poll() is None:
            if time.monotonic() >= deadline and proc.poll() is None:
                logger.line(f"ERROR: command exceeded {timeout_seconds}s; terminating")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                thread.join(timeout=2)
                return 124
            try:
                item = output.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                done_reading = True
            else:
                logger.line(item)
        thread.join(timeout=2)
        return int(proc.wait())

    def capture(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        timeout_seconds: int,
    ) -> CaptureResult:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            completed = subprocess.run(
                list(argv),
                cwd=str(cwd),
                env=dict(env),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                shell=False,
                creationflags=creationflags,
                check=False,
            )
            return CaptureResult(completed.returncode, completed.stdout or "")
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ""
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            return CaptureResult(124, str(output))


class R2Backend:
    """Small S3-compatible client used for receipts and output verification."""

    def __init__(
        self,
        env: Mapping[str, str],
        *,
        client: Any | None = None,
    ):
        missing = [name for name in R2_REQUIRED if not env.get(name)]
        if missing:
            raise AutomationError(f"missing required R2 environment names: {', '.join(missing)}")
        self.bucket = env["R2_BUCKET"]
        if client is None:
            try:
                import boto3  # type: ignore
            except ImportError as exc:
                raise AutomationError("boto3 is required for automation receipts") from exc
            endpoint = f"https://{env['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
            client = boto3.client(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=env["R2_ACCESS_KEY_ID"],
                aws_secret_access_key=env["R2_SECRET_ACCESS_KEY"],
                region_name="auto",
            )
        self.client = client

    @staticmethod
    def _is_missing(exc: Exception) -> bool:
        response = getattr(exc, "response", {}) or {}
        code = str((response.get("Error") or {}).get("Code", ""))
        return code in {"404", "NoSuchKey", "NotFound"} or "NoSuchKey" in type(exc).__name__

    @staticmethod
    def _is_precondition(exc: Exception) -> bool:
        response = getattr(exc, "response", {}) or {}
        code = str((response.get("Error") or {}).get("Code", ""))
        status = str((response.get("ResponseMetadata") or {}).get("HTTPStatusCode", ""))
        return code in {"412", "PreconditionFailed"} or status == "412"

    def head(self, key: str) -> Mapping[str, Any] | None:
        try:
            return self.client.head_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            if self._is_missing(exc):
                return None
            raise AutomationError(f"R2 HEAD failed for {key}: {type(exc).__name__}") from exc

    def get_json(self, key: str) -> tuple[dict[str, Any] | None, str | None]:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            if self._is_missing(exc):
                return None, None
            raise AutomationError(f"R2 GET failed for {key}: {type(exc).__name__}") from exc
        raw = response["Body"].read()
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AutomationError(f"R2 receipt is invalid JSON: {key}") from exc
        if not isinstance(value, dict):
            raise AutomationError(f"R2 receipt must be an object: {key}")
        # Preserve the HTTP entity-tag quotes. boto3 forwards IfMatch verbatim,
        # and R2/S3 conditional writes require the quoted ETag form.
        return value, str(response.get("ETag", "")).strip() or None

    def put_json(
        self,
        key: str,
        value: Mapping[str, Any],
        *,
        if_none_match: bool = False,
        if_match: str | None = None,
    ) -> bool:
        kwargs: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": key,
            "Body": json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"),
            "ContentType": "application/json",
        }
        if if_none_match:
            kwargs["IfNoneMatch"] = "*"
        if if_match:
            normalized = if_match.strip()
            if not (normalized.startswith('"') and normalized.endswith('"')):
                normalized = f'"{normalized.strip(chr(34))}"'
            kwargs["IfMatch"] = normalized
        try:
            self.client.put_object(**kwargs)
            return True
        except Exception as exc:
            if self._is_precondition(exc):
                return False
            raise AutomationError(f"R2 PUT failed for {key}: {type(exc).__name__}") from exc


@dataclass(frozen=True)
class Receipt:
    schema_version: str
    pipeline: str
    job_id: str
    run_date_et: str
    status: str
    source: str
    automation_token: str
    started_at_utc: str
    updated_at_utc: str
    phase: str | None = None
    lease_expires_at_utc: str | None = None
    workflow: str | None = None
    github_run_id: int | None = None
    github_url: str | None = None
    detail: str | None = None
    duplicate_sensitive: bool = False

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def lease_expired(self, now_utc: dt.datetime) -> bool:
        if not self.lease_expires_at_utc:
            return False
        try:
            expires = dt.datetime.fromisoformat(
                self.lease_expires_at_utc.replace("Z", "+00:00")
            )
        except (TypeError, ValueError):
            return False
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=UTC)
        return now_utc.astimezone(UTC) >= expires.astimezone(UTC)


def effective_status(receipt: Receipt | None, now_utc: dt.datetime) -> str:
    """Read-side normalisation shared by every reporter of a receipt.

    ``running`` with a lease that has already passed reads ``expired``: the
    claim is reclaimable (see ``ReceiptStore.claim``) and the health battery
    counts it as FAIL, so ``status`` must not print it as a live ``running``.
    Nothing is written; the receipt stays exactly as R2 holds it. An
    unparseable lease is treated as live, matching ``Receipt.lease_expired``
    (the claim path never reclaims what it cannot date).
    """
    if receipt is None:
        return "missing"
    if receipt.status == "running" and receipt.lease_expired(now_utc):
        return "expired"
    return receipt.status


class ReceiptStore(Protocol):
    def latest(self, run_date: str, job_id: str) -> Receipt | None: ...

    def claim(self, receipt: Receipt) -> bool: ...

    def transition(self, receipt: Receipt, *, update_latest: bool) -> None: ...


class R2ReceiptStore:
    def __init__(
        self,
        backend: R2Backend,
        *,
        now: Callable[[], dt.datetime] = lambda: dt.datetime.now(tz=UTC),
    ):
        self.backend = backend
        self.now = now

    @staticmethod
    def _job_prefix(run_date: str, job_id: str) -> str:
        return f"{RECEIPT_PREFIX}/{run_date}/{job_id}"

    @classmethod
    def _latest_key(cls, run_date: str, job_id: str) -> str:
        return f"{cls._job_prefix(run_date, job_id)}/latest.json"

    @classmethod
    def _event_key(cls, receipt: Receipt) -> str:
        stamp = receipt.updated_at_utc.replace(":", "").replace("+", "_")
        return (
            f"{cls._job_prefix(receipt.run_date_et, receipt.job_id)}/events/"
            f"{stamp}-{receipt.status}-{receipt.source}-{receipt.automation_token}.json"
        )

    @staticmethod
    def _decode(value: Mapping[str, Any]) -> Receipt:
        fields = {field.name for field in dataclasses.fields(Receipt)}
        return Receipt(**{name: value.get(name) for name in fields})

    def latest(self, run_date: str, job_id: str) -> Receipt | None:
        value, _ = self.backend.get_json(self._latest_key(run_date, job_id))
        return self._decode(value) if value else None

    def claim(self, receipt: Receipt) -> bool:
        key = self._latest_key(receipt.run_date_et, receipt.job_id)
        current, etag = self.backend.get_json(key)
        if current:
            decoded = self._decode(current)
            if decoded.status in {"success", "indeterminate"}:
                return False
            if decoded.status == "running":
                # GitHub claims require token reconciliation, never a blind
                # replacement. Local retryable/pre-side-effect claims may be
                # CAS-recovered only after their full declared lease expires.
                if decoded.phase == "github_reconcile":
                    return False
                if not decoded.lease_expired(self.now()):
                    return False
        ok = self.backend.put_json(
            key,
            receipt.as_dict(),
            if_none_match=current is None,
            if_match=etag if current is not None else None,
        )
        if not ok:
            return False
        self.backend.put_json(self._event_key(receipt), receipt.as_dict())
        return True

    def transition(self, receipt: Receipt, *, update_latest: bool) -> None:
        if not update_latest:
            self.backend.put_json(self._event_key(receipt), receipt.as_dict())
            return
        key = self._latest_key(receipt.run_date_et, receipt.job_id)
        # Retry a same-token CAS race once. This matters when the local and
        # hosted controllers both wake on an expired GitHub reconciliation
        # lease: they may adopt the same run, but neither may regress a success
        # written by the other controller.
        for _attempt in range(2):
            current, etag = self.backend.get_json(key)
            if not current or current.get("automation_token") != receipt.automation_token:
                raise AutomationError(f"lost receipt claim for {receipt.job_id}")
            current_receipt = self._decode(current)
            if current_receipt.source == "operator" and current_receipt.phase == "manual_resolved":
                if (
                    current_receipt.status == receipt.status
                    and current_receipt.source == receipt.source
                    and current_receipt.phase == receipt.phase
                ):
                    return
                raise AutomationError(
                    f"operator-resolved receipt is terminal for {receipt.job_id}"
                )
            if current_receipt.status == "success":
                if receipt.status == "success":
                    return
                raise AutomationError(
                    f"successful receipt is terminal for {receipt.job_id}"
                )
            if current_receipt.status == "failure":
                raise AutomationError(
                    f"failed receipt must be reclaimed before transition: {receipt.job_id}"
                )
            operator_resolution = (
                receipt.source == "operator" and receipt.phase == "manual_resolved"
            )
            if (
                current_receipt.status == "indeterminate"
                and receipt.status not in {"indeterminate", "success"}
                and not operator_resolution
            ):
                raise AutomationError(
                    f"indeterminate receipt requires operator resolution: {receipt.job_id}"
                )
            if self.backend.put_json(key, receipt.as_dict(), if_match=etag):
                # Latest is the safety gate, so make it durable before the
                # append-only audit event. If the event write fails, callers
                # stop; the latest marker still prevents duplicate execution.
                self.backend.put_json(self._event_key(receipt), receipt.as_dict())
                return
        raise AutomationError(f"receipt transition raced for {receipt.job_id}")


class InMemoryReceiptStore:
    """Deterministic test seam; also useful for plan-only integrations."""

    def __init__(
        self,
        *,
        now: Callable[[], dt.datetime] = lambda: dt.datetime.now(tz=UTC),
    ):
        self.latest_values: dict[tuple[str, str], Receipt] = {}
        self.events: list[Receipt] = []
        self.now = now

    def latest(self, run_date: str, job_id: str) -> Receipt | None:
        return self.latest_values.get((run_date, job_id))

    def claim(self, receipt: Receipt) -> bool:
        key = (receipt.run_date_et, receipt.job_id)
        current = self.latest_values.get(key)
        if current:
            if current.status in {"success", "indeterminate"}:
                return False
            if current.status == "running":
                if current.phase == "github_reconcile":
                    return False
                if not current.lease_expired(self.now()):
                    return False
        self.latest_values[key] = receipt
        self.events.append(receipt)
        return True

    def transition(self, receipt: Receipt, *, update_latest: bool) -> None:
        if update_latest:
            key = (receipt.run_date_et, receipt.job_id)
            current = self.latest_values.get(key)
            if not current or current.automation_token != receipt.automation_token:
                raise AutomationError(f"lost receipt claim for {receipt.job_id}")
            if current.source == "operator" and current.phase == "manual_resolved":
                if (
                    current.status == receipt.status
                    and current.source == receipt.source
                    and current.phase == receipt.phase
                ):
                    return
                raise AutomationError(
                    f"operator-resolved receipt is terminal for {receipt.job_id}"
                )
            if current.status == "success":
                if receipt.status == "success":
                    return
                raise AutomationError(
                    f"successful receipt is terminal for {receipt.job_id}"
                )
            if current.status == "failure":
                raise AutomationError(
                    f"failed receipt must be reclaimed before transition: {receipt.job_id}"
                )
            operator_resolution = (
                receipt.source == "operator" and receipt.phase == "manual_resolved"
            )
            if (
                current.status == "indeterminate"
                and receipt.status not in {"indeterminate", "success"}
                and not operator_resolution
            ):
                raise AutomationError(
                    f"indeterminate receipt requires operator resolution: {receipt.job_id}"
                )
            self.latest_values[key] = receipt
        self.events.append(receipt)


class OutputValidator:
    def __init__(self, backend: R2Backend, *, freshness_slack_seconds: int = 300):
        self.backend = backend
        self.freshness_slack = dt.timedelta(seconds=freshness_slack_seconds)

    @staticmethod
    def _key_for(spec: OutputSpec, path: Path) -> str:
        if spec.r2_key:
            return spec.r2_key
        name = path.stem
        if spec.strip_suffix and name.endswith(spec.strip_suffix):
            name = name[: -len(spec.strip_suffix)]
        return f"{spec.r2_prefix}/{name}{path.suffix}"

    def validate(
        self,
        outputs: Sequence[OutputSpec],
        *,
        repo_root: Path,
        started_at_utc: dt.datetime,
        logger: RunLogger,
    ) -> None:
        for spec in outputs:
            pattern = str(repo_root / spec.local_pattern)
            matches = [Path(p) for p in sorted(glob.glob(pattern))]
            if not matches:
                if spec.required:
                    raise ValidationError(f"required producer output missing: {spec.local_pattern}")
                logger.line(f"validation: optional output absent: {spec.local_pattern}")
                continue
            if spec.r2_key and len(matches) != 1:
                raise ValidationError(f"single R2 key mapped to {len(matches)} files: {spec.local_pattern}")
            for path in matches:
                size = path.stat().st_size
                if size < spec.min_bytes:
                    raise ValidationError(f"producer output too small: {path.name} ({size} bytes)")
                key = self._key_for(spec, path)
                metadata = self.backend.head(key)
                if not metadata:
                    raise ValidationError(f"producer output missing from R2: {key}")
                remote_size = int(metadata.get("ContentLength", -1))
                if remote_size != size:
                    raise ValidationError(
                        f"R2 size mismatch for {key}: local={size}, remote={remote_size}"
                    )
                if spec.require_recent_upload:
                    modified = metadata.get("LastModified")
                    if not isinstance(modified, dt.datetime):
                        raise ValidationError(f"R2 LastModified missing for {key}")
                    if modified.tzinfo is None:
                        modified = modified.replace(tzinfo=UTC)
                    if modified.astimezone(UTC) < started_at_utc - self.freshness_slack:
                        raise ValidationError(f"R2 object was not refreshed by this run: {key}")
                logger.line(f"validation: r2://{key} size={size} verified")


@dataclass(frozen=True)
class GithubRun:
    database_id: int
    status: str
    conclusion: str | None
    title: str
    url: str | None


class GithubDispatcher:
    def __init__(
        self,
        process: ProcessClient,
        *,
        repo_root: Path,
        env: Mapping[str, str],
        repository: str,
        ref: str,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        poll_seconds: float = 8.0,
    ):
        self.process = process
        self.repo_root = repo_root
        self.env = env
        self.repository = repository
        self.ref = ref
        self.sleep = sleep
        self.monotonic = monotonic
        self.poll_seconds = poll_seconds

    def _capture(self, argv: Sequence[str], *, timeout: int = 60) -> str:
        result = self.process.capture(
            argv,
            cwd=self.repo_root,
            env=self.env,
            timeout_seconds=timeout,
        )
        if result.returncode != 0:
            summary = (result.stdout or "").strip().splitlines()
            tail = summary[-1] if summary else "no output"
            raise DispatchError(f"GitHub CLI failed ({result.returncode}): {tail}")
        return result.stdout

    def _find_by_token(
        self, workflow: WorkflowSpec, automation_token: str
    ) -> GithubRun | None:
        payload = self._capture(
            [
                "gh",
                "run",
                "list",
                "--workflow",
                workflow.workflow,
                "--event",
                "workflow_dispatch",
                "--repo",
                self.repository,
                "--limit",
                "100",
                "--json",
                "databaseId,status,conclusion,displayTitle,url",
            ]
        )
        try:
            rows = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise DispatchError("GitHub run list returned invalid JSON") from exc
        for row in rows:
            title = str(row.get("displayTitle", ""))
            if automation_token in title:
                return GithubRun(
                    database_id=int(row["databaseId"]),
                    status=str(row.get("status", "")),
                    conclusion=row.get("conclusion"),
                    title=title,
                    url=row.get("url"),
                )
        return None

    def _wait_for_token(
        self,
        workflow: WorkflowSpec,
        *,
        automation_token: str,
        logger: RunLogger,
        require_visible_initially: bool,
    ) -> GithubRun:
        deadline = self.monotonic() + workflow.timeout_seconds
        first = True
        while self.monotonic() < deadline:
            run = self._find_by_token(workflow, automation_token)
            if first and require_visible_initially and run is None:
                raise DispatchRunNotFound(
                    f"accepted GitHub token is not visible: {automation_token}"
                )
            first = False
            if run is None:
                self.sleep(self.poll_seconds)
                continue
            if run.status == "completed":
                if run.conclusion == "success":
                    logger.line(f"GitHub fallback succeeded: {run.url or run.database_id}")
                    return run
                raise DispatchAcceptedError(
                    f"GitHub fallback concluded {run.conclusion}: "
                    f"{run.url or run.database_id}"
                )
            logger.line(f"GitHub run {run.database_id}: {run.status}")
            self.sleep(self.poll_seconds)
        raise DispatchAcceptedError(
            f"timed out waiting for {workflow.workflow} token={automation_token}"
        )

    def dispatch_and_wait(
        self,
        workflow: WorkflowSpec,
        *,
        automation_token: str,
        logger: RunLogger,
    ) -> GithubRun:
        # Re-adopt a token if a prior caller submitted it but failed before its
        # receipt transition became visible. A token is never submitted twice.
        existing = self._find_by_token(workflow, automation_token)
        if existing is not None:
            logger.line(
                f"GitHub fallback: adopt {workflow.workflow} token={automation_token}"
            )
            return self._wait_for_token(
                workflow,
                automation_token=automation_token,
                logger=logger,
                require_visible_initially=True,
            )
        argv = [
            "gh",
            "workflow",
            "run",
            workflow.workflow,
            "--repo",
            self.repository,
            "--ref",
            self.ref,
            "-f",
            f"automation_token={automation_token}",
        ]
        for name, value in workflow.inputs:
            argv.extend(("-f", f"{name}={value}"))
        logger.line(
            f"GitHub fallback: dispatch {workflow.workflow} token={automation_token}"
        )
        # A nonzero CLI exit does not prove the server rejected the request: a
        # connection can fail after GitHub accepted it. Treat that boundary as
        # ambiguous so non-rerun-safe jobs stop for review and rerun-safe jobs
        # alone may later receive a new token.
        try:
            self._capture(argv)
        except Exception as exc:
            try:
                existing = self._find_by_token(workflow, automation_token)
            except Exception as reconcile_exc:
                raise DispatchAcceptedError(
                    "GitHub submission outcome is ambiguous and token lookup "
                    f"also failed: {reconcile_exc}"
                ) from exc
            if existing is not None:
                return self._wait_for_token(
                    workflow,
                    automation_token=automation_token,
                    logger=logger,
                    require_visible_initially=True,
                )
            raise DispatchAcceptedError(
                "GitHub submission outcome is ambiguous; no matching token "
                "was visible in the immediate reconciliation check"
            ) from exc
        try:
            return self._wait_for_token(
                workflow,
                automation_token=automation_token,
                logger=logger,
                require_visible_initially=False,
            )
        except DispatchAcceptedError:
            raise
        except Exception as exc:
            raise DispatchAcceptedError(
                f"GitHub dispatch accepted but reconciliation failed: {exc}"
            ) from exc

    def reconcile_and_wait(
        self,
        workflow: WorkflowSpec,
        *,
        automation_token: str,
        logger: RunLogger,
    ) -> GithubRun:
        """Adopt an already-submitted run; this method never dispatches."""
        try:
            return self._wait_for_token(
                workflow,
                automation_token=automation_token,
                logger=logger,
                require_visible_initially=True,
            )
        except DispatchAcceptedError:
            raise
        except Exception as exc:
            raise DispatchAcceptedError(
                f"GitHub token reconciliation failed: {exc}"
            ) from exc


@dataclass(frozen=True)
class JobOutcome:
    job_id: str
    status: str
    source: str | None = None
    detail: str | None = None
    # True when the outcome merely restates a receipt that existed before this
    # invocation started (or is blocked only by such receipts). A retry or
    # controller run must not turn red for an ambiguity it did not create.
    preexisting: bool = False

    @property
    def satisfied(self) -> bool:
        return self.status == "success"


class AutomationSupervisor:
    def __init__(
        self,
        *,
        catalog: Mapping[str, PipelineSpec],
        repo_root: Path,
        state_root: Path,
        python_executable: str,
        env: Mapping[str, str],
        receipts: ReceiptStore,
        process: ProcessClient,
        dispatcher: GithubDispatcher,
        validator: OutputValidator | None,
        now: Callable[[], dt.datetime] = lambda: dt.datetime.now(tz=UTC),
    ):
        self.catalog = dict(catalog)
        self.repo_root = repo_root
        self.state_root = state_root
        self.python_executable = python_executable
        self.env = dict(env)
        self.receipts = receipts
        self.process = process
        self.dispatcher = dispatcher
        self.validator = validator
        self.now = now

    def _token(self, run_date: str, job: JobSpec) -> str:
        return f"{run_date}-{job.id}-{uuid.uuid4().hex[:10]}"

    def _receipt(
        self,
        *,
        pipeline: PipelineSpec,
        job: JobSpec,
        run_date: str,
        token: str,
        status: str,
        source: str,
        started: dt.datetime,
        workflow: str | None = None,
        github_run: GithubRun | None = None,
        detail: str | None = None,
        phase: str | None = None,
        lease_seconds: int | None = None,
    ) -> Receipt:
        current_dt = self.now().astimezone(UTC)
        current = current_dt.isoformat()
        lease_expires = (
            (current_dt + dt.timedelta(seconds=lease_seconds)).isoformat()
            if lease_seconds is not None
            else None
        )
        return Receipt(
            schema_version="automation-receipt.v1",
            pipeline=pipeline.id,
            job_id=job.id,
            run_date_et=run_date,
            status=status,
            source=source,
            automation_token=token,
            started_at_utc=started.astimezone(UTC).isoformat(),
            updated_at_utc=current,
            phase=phase,
            lease_expires_at_utc=lease_expires,
            workflow=workflow,
            github_run_id=github_run.database_id if github_run else None,
            github_url=github_run.url if github_run else None,
            detail=detail,
            duplicate_sensitive=job.duplicate_sensitive,
        )

    @staticmethod
    def _local_lease_seconds(job: JobSpec) -> int:
        commands = list(job.commands)
        if not job.rerun_safe:
            before_effect: list[CommandSpec] = []
            for command in commands:
                if command.side_effecting:
                    break
                before_effect.append(command)
            commands = before_effect
        declared = sum(command.timeout_seconds for command in commands)
        return max(300, declared + LEASE_GRACE_SECONDS)

    @staticmethod
    def _github_lease_seconds(job: JobSpec) -> int:
        assert job.workflow is not None
        return job.workflow.timeout_seconds + LEASE_GRACE_SECONDS

    def _resolve_command(self, command: CommandSpec) -> list[str]:
        return [
            self.python_executable if value == "{python}" else value
            for value in command.argv
        ]

    def _preflight(self, job: JobSpec) -> None:
        missing = [name for name in job.required_env if not self.env.get(name)]
        if missing:
            raise AutomationError(
                f"required environment names missing for {job.id}: {', '.join(missing)}"
            )

    def _local_gate(self, job: JobSpec) -> tuple[bool, str]:
        if job.local_gate is None:
            return True, ""
        if job.local_gate == "discretionary_delivery_window":
            from scripts.check_discretionary_focus_session import delivery_window_gate

            allowed, market_date = delivery_window_gate(self.now())
            return allowed, f"NYSE delivery gate for {market_date.isoformat()}"
        raise AutomationError(f"unknown local job gate: {job.local_gate}")

    def _run_github(
        self,
        *,
        pipeline: PipelineSpec,
        job: JobSpec,
        run_date: str,
        token: str,
        started: dt.datetime,
        logger: RunLogger,
    ) -> JobOutcome:
        assert job.workflow is not None
        running = self._receipt(
            pipeline=pipeline,
            job=job,
            run_date=run_date,
            token=token,
            status="running",
            source="github",
            started=started,
            workflow=job.workflow.workflow,
            phase="github_reconcile",
            lease_seconds=self._github_lease_seconds(job),
        )
        self.receipts.transition(running, update_latest=True)
        try:
            github_run = self.dispatcher.dispatch_and_wait(
                job.workflow, automation_token=token, logger=logger
            )
            success = self._receipt(
                pipeline=pipeline,
                job=job,
                run_date=run_date,
                token=token,
                status="success",
                source="github",
                started=started,
                workflow=job.workflow.workflow,
                github_run=github_run,
                phase="completed",
            )
            self.receipts.transition(success, update_latest=True)
            return JobOutcome(job.id, "success", "github")
        except Exception as exc:  # noqa: BLE001 - receipt every fallback outcome
            status = (
                "indeterminate"
                if isinstance(exc, DispatchAcceptedError) and not job.rerun_safe
                else "failure"
            )
            failure = self._receipt(
                pipeline=pipeline,
                job=job,
                run_date=run_date,
                token=token,
                status=status,
                source="github",
                started=started,
                workflow=job.workflow.workflow,
                detail=f"{type(exc).__name__}: {exc}",
                phase=("manual_review" if status == "indeterminate" else "retryable"),
            )
            self.receipts.transition(failure, update_latest=True)
            logger.line(f"ERROR: {job.id} GitHub fallback {status}: {exc}")
            return JobOutcome(job.id, status, "github", str(exc))

    def _resume_github(
        self,
        *,
        pipeline: PipelineSpec,
        job: JobSpec,
        receipt: Receipt,
        logger: RunLogger,
    ) -> JobOutcome:
        """Reconcile an expired GitHub lease without submitting a second run."""
        assert job.workflow is not None
        try:
            github_run = self.dispatcher.reconcile_and_wait(
                job.workflow,
                automation_token=receipt.automation_token,
                logger=logger,
            )
            success = self._receipt(
                pipeline=pipeline,
                job=job,
                run_date=receipt.run_date_et,
                token=receipt.automation_token,
                status="success",
                source="github",
                started=dt.datetime.fromisoformat(receipt.started_at_utc),
                workflow=job.workflow.workflow,
                github_run=github_run,
                phase="completed",
            )
            self.receipts.transition(success, update_latest=True)
            return JobOutcome(job.id, "success", "github", "reconciled existing run")
        except Exception as exc:  # noqa: BLE001 - ambiguity must be persisted
            status = "failure" if job.rerun_safe else "indeterminate"
            resolved = self._receipt(
                pipeline=pipeline,
                job=job,
                run_date=receipt.run_date_et,
                token=receipt.automation_token,
                status=status,
                source="github",
                started=dt.datetime.fromisoformat(receipt.started_at_utc),
                workflow=job.workflow.workflow,
                detail=f"{type(exc).__name__}: {exc}",
                phase=("retryable" if status == "failure" else "manual_review"),
            )
            self.receipts.transition(resolved, update_latest=True)
            logger.line(f"ERROR: {job.id} GitHub reconciliation {status}: {exc}")
            return JobOutcome(job.id, status, "github", str(exc))

    def run_job(
        self,
        pipeline: PipelineSpec,
        job: JobSpec,
        *,
        run_date: str,
        logger: RunLogger,
        allow_fallback: bool,
        github_only: bool = False,
    ) -> JobOutcome:
        existing = self.receipts.latest(run_date, job.id)
        if existing and existing.status in {"success", "indeterminate"}:
            note = ""
            if existing.status == "indeterminate":
                note = (
                    "; pre-existing, never re-run automatically; clear with: "
                    f"resolve --pipeline {pipeline.id} --job {job.id} --date {run_date} "
                    "--disposition success|retryable_failure --reason ..."
                )
            logger.line(
                f"skip {job.id}: {existing.status} receipt from {existing.source} "
                f"token={existing.automation_token}{note}"
            )
            return JobOutcome(
                job.id, existing.status,
                existing.source,
                "existing receipt",
                preexisting=True,
            )
        if existing and existing.status == "running":
            if not existing.lease_expired(self.now().astimezone(UTC)):
                logger.line(
                    f"skip {job.id}: live running lease from {existing.source} "
                    f"token={existing.automation_token}"
                )
                # Another writer owns it and its lease has not passed: the
                # correct action is to leave it alone, so neither this job nor
                # anything blocked behind it is a failure THIS run produced.
                # (2026-09-04 verify: a live lease on scan_am turned the 05:45
                # retry and every controller tick red while nothing was wrong.)
                return JobOutcome(
                    job.id, "running", existing.source, "live lease", preexisting=True
                )
            if existing.phase == "github_reconcile":
                logger.line(
                    f"reconcile {job.id}: expired GitHub lease "
                    f"token={existing.automation_token}"
                )
                return self._resume_github(
                    pipeline=pipeline,
                    job=job,
                    receipt=existing,
                    logger=logger,
                )
            # Same vocabulary as ``status``/health: the local claim died before
            # its side-effecting step (phase local_pre_side_effect) or is
            # rerun-safe (local_retryable), and its lease has passed. Reclaim.
            logger.line(
                f"expired {job.id}: {existing.phase} lease from {existing.source} "
                f"expired at {existing.lease_expires_at_utc} "
                f"token={existing.automation_token}; reclaiming"
            )

        if github_only and job.workflow is None:
            # A local-only job (no migrated workflow) has no GitHub backup to
            # dispatch. The hourly fallback controller must step over it rather
            # than assert its way into a crashed sweep of the whole pipeline.
            logger.line(f"skip {job.id}: local-only job, no GitHub workflow backup")
            return JobOutcome(job.id, "skipped", None, "no GitHub workflow backup")

        started = self.now().astimezone(UTC)
        token = self._token(run_date, job)
        source = "github" if (github_only or job.dispatch_only) else "local"
        claim = self._receipt(
            pipeline=pipeline,
            job=job,
            run_date=run_date,
            token=token,
            status="running",
            source=source,
            started=started,
            workflow=job.workflow.workflow if source == "github" and job.workflow else None,
            phase=(
                "github_reconcile"
                if source == "github"
                else "local_retryable" if job.rerun_safe else "local_pre_side_effect"
            ),
            lease_seconds=(
                self._github_lease_seconds(job)
                if source == "github"
                else self._local_lease_seconds(job)
            ),
        )
        if not self.receipts.claim(claim):
            current = self.receipts.latest(run_date, job.id)
            state = current.status if current else "concurrent claim"
            logger.line(f"skip {job.id}: {state}")
            return JobOutcome(job.id, "running", current.source if current else None, state)

        # Deterministic applicability gates govern both the local primary and
        # GitHub backup. Evaluate them after the CAS claim (one terminal receipt
        # per day) but before either execution branch so a holiday cannot turn
        # a clean local no-op into a failing/ambiguous remote side effect.
        allowed, gate_detail = self._local_gate(job)
        if not allowed:
            success = self._receipt(
                pipeline=pipeline,
                job=job,
                run_date=run_date,
                token=token,
                status="success",
                source=source,
                started=started,
                workflow=(job.workflow.workflow if source == "github" and job.workflow else None),
                detail=f"not applicable: {gate_detail}",
                phase="completed",
            )
            self.receipts.transition(success, update_latest=True)
            logger.line(f"success {job.id}: not applicable ({gate_detail})")
            return JobOutcome(job.id, "success", source, "not applicable")

        if github_only or job.dispatch_only:
            return self._run_github(
                pipeline=pipeline,
                job=job,
                run_date=run_date,
                token=token,
                started=started,
                logger=logger,
            )

        child_env = dict(self.env)
        child_env["LOCAL_AUTOMATION_PRIMARY"] = "1"
        child_env["LOCAL_AUTOMATION_RUN_TOKEN"] = token
        child_env["LOCAL_AUTOMATION_STRICT"] = "1"
        child_env.update(job.env_overrides)
        logger.line(f"start {job.id}: {job.description}; token={token}")
        indeterminate_marked = False
        try:
            self._preflight(job)
            for command in job.commands:
                if command.side_effecting and not job.rerun_safe and not indeterminate_marked:
                    # CAS the durable marker before the process can touch
                    # Sheets, SMTP, orders, or other non-idempotent state.
                    # A crash from this point onward requires human review;
                    # neither immediate nor hourly fallback may run it twice.
                    marker = self._receipt(
                        pipeline=pipeline,
                        job=job,
                        run_date=run_date,
                        token=token,
                        status="indeterminate",
                        source="local",
                        started=started,
                        detail=f"side-effecting step started: {command.label}",
                        phase="local_side_effect",
                    )
                    logger.line(
                        f"guard: persist side-effect boundary for {job.id}: {command.label}"
                    )
                    try:
                        self.receipts.transition(marker, update_latest=True)
                    except Exception as marker_exc:  # noqa: BLE001 - fail closed at the gate
                        # The child process has not started. Do not guess whether
                        # a failed/ambiguous R2 request reached the durable latest
                        # marker, and do not dispatch a second writer here.
                        logger.line(
                            f"ERROR: side-effect boundary for {job.id} was not confirmed; "
                            f"child process was not started and automatic fallback was "
                            f"suppressed: {type(marker_exc).__name__}: {marker_exc}"
                        )
                        return JobOutcome(
                            job.id,
                            "failure",
                            "local",
                            f"side-effect boundary not confirmed: {marker_exc}",
                        )
                    indeterminate_marked = True
                    logger.line(f"guard: side-effect boundary durable for {job.id}")
                logger.line(f"step: {command.label}")
                rc = self.process.stream(
                    self._resolve_command(command),
                    cwd=self.repo_root,
                    env=child_env,
                    timeout_seconds=command.timeout_seconds,
                    logger=logger,
                )
                if rc != 0:
                    raise AutomationError(f"{command.label} exited {rc}")
            if job.outputs:
                if self.validator is None:
                    raise ValidationError("output validator is required for producer jobs")
                self.validator.validate(
                    job.outputs,
                    repo_root=self.repo_root,
                    started_at_utc=started,
                    logger=logger,
                )
            success = self._receipt(
                pipeline=pipeline,
                job=job,
                run_date=run_date,
                token=token,
                status="success",
                source="local",
                started=started,
                phase="completed",
            )
            self.receipts.transition(success, update_latest=True)
            logger.line(f"success {job.id} (local)")
            return JobOutcome(job.id, "success", "local")
        except Exception as exc:  # noqa: BLE001 - local failures share one fallback path
            if indeterminate_marked:
                attention = self._receipt(
                    pipeline=pipeline,
                    job=job,
                    run_date=run_date,
                    token=token,
                    status="indeterminate",
                    source="local",
                    started=started,
                    detail=f"{type(exc).__name__}: {exc}",
                    phase="manual_review",
                )
                try:
                    self.receipts.transition(attention, update_latest=True)
                except Exception as receipt_exc:  # noqa: BLE001 - preserve local evidence
                    logger.line(
                        f"ERROR: local {job.id} failed after the durable side-effect "
                        f"boundary ({type(exc).__name__}: {exc}); unable to persist the "
                        f"manual-review detail: {type(receipt_exc).__name__}: {receipt_exc}; "
                        "automatic fallback suppressed"
                    )
                    return JobOutcome(job.id, "indeterminate", "local", str(exc))
                logger.line(
                    f"ERROR: local {job.id} is indeterminate after side effects: {exc}; "
                    "automatic fallback suppressed"
                )
                return JobOutcome(job.id, "indeterminate", "local", str(exc))
            logger.line(
                f"ERROR: local {job.id} failed before side effects: "
                f"{type(exc).__name__}: {exc}"
            )
            local_failure = self._receipt(
                pipeline=pipeline,
                job=job,
                run_date=run_date,
                token=token,
                status="failure",
                source="local",
                started=started,
                detail=f"{type(exc).__name__}: {exc}",
                phase="retryable",
            )
            try:
                self.receipts.transition(local_failure, update_latest=False)
            except Exception as receipt_exc:  # noqa: BLE001 - control plane is unavailable
                logger.line(
                    f"ERROR: unable to persist the retryable failure for {job.id}: "
                    f"{type(receipt_exc).__name__}: {receipt_exc}; automatic fallback "
                    "suppressed"
                )
                return JobOutcome(job.id, "failure", "local", str(exc))
            if allow_fallback and job.workflow is not None:
                logger.line(f"immediate GitHub fallback for {job.id}")
                return self._run_github(
                    pipeline=pipeline,
                    job=job,
                    run_date=run_date,
                    token=token,
                    started=started,
                    logger=logger,
                )
            self.receipts.transition(local_failure, update_latest=True)
            return JobOutcome(job.id, "failure", "local", str(exc))

    def run_pipeline(
        self,
        pipeline_id: str,
        *,
        run_date: str,
        allow_fallback: bool,
        github_only: bool = False,
        only_jobs: set[str] | None = None,
        logger: RunLogger,
    ) -> list[JobOutcome]:
        pipeline = self.catalog[pipeline_id]
        selected: set[str] | None = None
        if only_jobs is not None:
            jobs_by_id = {job.id: job for job in pipeline.jobs}
            unknown = only_jobs - jobs_by_id.keys()
            if unknown:
                raise AutomationError(
                    f"unknown job(s) for {pipeline_id}: {', '.join(sorted(unknown))}"
                )
            selected = set()

            def include_with_dependencies(job_id: str) -> None:
                if job_id in selected:
                    return
                selected.add(job_id)
                for dependency in jobs_by_id[job_id].depends_on:
                    include_with_dependencies(dependency)

            for requested in only_jobs:
                include_with_dependencies(requested)

        outcomes: dict[str, JobOutcome] = {}
        for job in pipeline.jobs:
            if selected is not None and job.id not in selected:
                continue
            unsatisfied = [dep for dep in job.depends_on if not outcomes.get(dep, JobOutcome(dep, "failure")).satisfied]
            if unsatisfied:
                detail = f"unsatisfied dependencies: {', '.join(unsatisfied)}"
                logger.line(f"skip {job.id}: {detail}")
                # Blocked only by receipts that predate this run is itself a
                # pre-existing condition, not a new failure of this run.
                inherited = all(
                    dep in outcomes and outcomes[dep].preexisting for dep in unsatisfied
                )
                outcomes[job.id] = JobOutcome(
                    job.id, "blocked", detail=detail, preexisting=inherited
                )
                continue
            outcomes[job.id] = self.run_job(
                pipeline,
                job,
                run_date=run_date,
                logger=logger,
                allow_fallback=allow_fallback,
                github_only=github_only,
            )
        return list(outcomes.values())


def _format_time(value: dt.time) -> str:
    return value.strftime("%H:%M ET")


def render_plan(pipeline: PipelineSpec, *, python_executable: str = "python") -> str:
    lines = [
        f"{pipeline.id}: {pipeline.description}",
        f"  schedule: {pipeline.cadence} at {_format_time(pipeline.run_at_et)}",
        (
            "  fallback window: "
            f"{_format_time(pipeline.fallback_at_et)}-{_format_time(pipeline.fallback_until_et)}"
        ),
    ]
    if pipeline.retry_at_et is not None and pipeline.retry_until_et is not None:
        # The local second chance is a real scheduled writer; a plan that hides
        # it under-describes who can claim these receipts (2026-09-04 verify).
        lines.append(
            "  local retry window: "
            f"{_format_time(pipeline.retry_at_et)}-{_format_time(pipeline.retry_until_et)}"
            f" (run-pipeline --pipeline {pipeline.id} --retry)"
        )
    for job in pipeline.jobs:
        mode = "GITHUB-DISPATCH-ONLY" if job.dispatch_only else "LOCAL-PRIMARY"
        flags = []
        if job.duplicate_sensitive:
            flags.append("once-only")
        if job.depends_on:
            flags.append("after=" + ",".join(job.depends_on))
        suffix = f" [{' '.join(flags)}]" if flags else ""
        lines.append(f"  - {job.id}: {mode}{suffix}")
        for command in job.commands:
            argv = [python_executable if v == "{python}" else v for v in command.argv]
            display = subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)
            lines.append(f"      {command.label}: {display}")
        if job.required_env:
            lines.append("      required env names: " + ", ".join(job.required_env))
        if job.env_overrides:
            lines.append(
                "      fixed env: " + ", ".join(f"{name}={value}" for name, value in job.env_overrides)
            )
        if job.local_gate:
            lines.append(f"      local gate: {job.local_gate}")
        if job.outputs:
            for output in job.outputs:
                target = output.r2_key or f"{output.r2_prefix}/<files>"
                lines.append(f"      validates: {output.local_pattern} -> r2://{target}")
        if job.workflow:
            inputs = ", ".join(f"{k}={v}" for k, v in job.workflow.inputs)
            input_text = f" ({inputs})" if inputs else ""
            lines.append(f"      GitHub fallback: {job.workflow.workflow}{input_text}")
    return "\n".join(lines)


def _et_run_date(now: dt.datetime) -> str:
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return now.astimezone(ET).date().isoformat()


def _utc_now() -> dt.datetime:
    """Clock seam for ``main`` so window gates are testable without patching datetime."""
    return dt.datetime.now(tz=UTC)


def _dependents_of(pipeline: PipelineSpec, job_id: str) -> list[JobSpec]:
    """Jobs downstream of ``job_id`` in catalog order (transitive closure)."""
    downstream: set[str] = {job_id}
    ordered: list[JobSpec] = []
    for job in pipeline.jobs:
        if job.id != job_id and any(dep in downstream for dep in job.depends_on):
            downstream.add(job.id)
            ordered.append(job)
    return ordered


RUNTIME_MARKER_RELPATH = Path(".local") / "automation-runtime.json"
DEFAULT_REF = "main"


def _marker_fallback_ref(*roots: str | None) -> str | None:
    """The pinned immutable tag from the first runtime marker found in ``roots``.

    The marker lives under the RUNTIME root -- ``install_local_automation_tasks.ps1``
    writes ``<RuntimeRoot>\\.local\\automation-runtime.json`` and
    ``run_local_automation.ps1`` reads it there -- so callers pass the runtime
    root first and the config root second (a development checkout may hold one,
    the production config root does not). Returns None when no marker is
    present, none can be parsed, or the ref names a branch rather than a
    tag-shaped ref. Never raises: this only decorates a printed hint.
    """
    for root in roots:
        if not root:
            continue
        try:
            marker = json.loads(
                (Path(root) / RUNTIME_MARKER_RELPATH).read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            continue
        candidate = marker.get("fallback_ref") if isinstance(marker, Mapping) else None
        if not isinstance(candidate, str) or not candidate or candidate == DEFAULT_REF:
            continue
        if not all(ch.isalnum() or ch in "._/-" for ch in candidate):
            continue
        return candidate
    return None


def _print_due_after_resolution(
    *,
    pipeline: PipelineSpec,
    job_id: str,
    run_date: str,
    disposition: str,
    receipts: ReceiptStore,
    now_utc: dt.datetime,
    config_root: str | None,
    ref: str | None,
    runtime_root: str | None = None,
) -> list[str]:
    """After ``resolve``, name what is now due and the exact command. Never dispatches.

    Returns the job ids printed as due (for tests). A success resolution
    unblocks the job's dependents; a retryable_failure resolution makes the
    job itself due again.
    """
    if disposition == "success":
        candidates = _dependents_of(pipeline, job_id)
    else:
        candidates = [job for job in pipeline.jobs if job.id == job_id]
    due: list[str] = []
    for job in candidates:
        state = effective_status(receipts.latest(run_date, job.id), now_utc)
        if state in {"missing", "failure", "expired"}:
            due.append(job.id)
    if not due:
        print(f"Nothing downstream of {job_id} is due for {run_date}.")
        return due
    window = (
        f"{_format_time(pipeline.fallback_at_et)}-{_format_time(pipeline.fallback_until_et)}"
    )
    in_window = pipeline.fallback_is_due(now_utc)
    print(
        f"Now due for {run_date} ({pipeline.id}): {', '.join(due)}. "
        "Not dispatched. Run by hand (each command dispatches that job plus its "
        "prerequisite closure, honouring existing receipts):"
    )
    # The printed command is meant to be pasted. An explicit ``--ref`` is
    # honoured verbatim -- including ``--ref main``, which only earns a warning.
    # When none was given, fall back to the pinned immutable tag in the runtime
    # marker (runtime root first, then the config root), because that is what
    # every scheduled dispatch resolves (2026-09-04 verify).
    searched = [runtime_root, config_root]
    marker_ref = _marker_fallback_ref(*searched) if ref is None else None
    effective_ref = ref or marker_ref or DEFAULT_REF
    config = f" --config-root {config_root}" if config_root else ""
    for dependent in due:
        print(
            f"  python scripts/automation_supervisor.py fallback-due "
            f"--pipeline {pipeline.id} --job {dependent} --date {run_date}"
            f"{config} --ref {effective_ref}"
        )
    looked_in = ", ".join(
        str(Path(root) / RUNTIME_MARKER_RELPATH) for root in searched if root
    ) or str(RUNTIME_MARKER_RELPATH)
    if marker_ref:
        print(
            f"  --ref {marker_ref} taken from the runtime marker under "
            f"{looked_in} (the immutable tag the tasks and the controller "
            "dispatch)."
        )
    elif effective_ref == DEFAULT_REF:
        print(
            f"  WARNING: --ref {DEFAULT_REF} is a moving branch, not the pinned "
            "immutable tag the scheduled runs dispatch. The marker lives at "
            f"<RuntimeRoot>\\{RUNTIME_MARKER_RELPATH}; none was readable at "
            f"{looked_in}. Read fallback_ref from the enabled generation's "
            "marker and pass it explicitly before running these."
        )
    if not in_window:
        print(
            f"  NOTE: fallback-due acts only inside the {pipeline.id} ET fallback "
            f"window {window}; outside it the command prints 'No pipeline is inside "
            "its ET fallback window' and does nothing. Use the child workflow's "
            "workflow_dispatch by hand instead, or wait for the window."
        )
    return due


def _make_runtime(
    args: argparse.Namespace, *, controller_only: bool = False
) -> tuple[AutomationSupervisor, R2ReceiptStore]:
    repo_root = Path(args.repo_root).resolve()
    state_root = Path(args.state_root).resolve()
    if controller_only:
        # A GitHub-hosted fallback controller already receives scoped secrets
        # in its environment and has no local OneDrive credential files.
        # A local status/fallback task may still pass config-root to augment
        # that environment from the machine's private .env.
        env = dict(os.environ)
        config_root = Path(args.config_root).resolve() if args.config_root else repo_root
        env_path = config_root / ".env"
        if env_path.is_file():
            env.update(_parse_env_file(env_path))
        if not env.get("GH_TOKEN") and env.get("GH_PAT_NEW_SEASONALS"):
            env["GH_TOKEN"] = env["GH_PAT_NEW_SEASONALS"]
        env["LOCAL_AUTOMATION_STRICT"] = "1"
    else:
        if not args.config_root:
            raise AutomationError("--config-root is required for a live local run")
        config_root = Path(args.config_root).resolve()
        gcp_path, exec_path = resolve_external_secret_paths(
            config_root=config_root,
            gcp_json_path=Path(args.gcp_json) if args.gcp_json else None,
            exec_env_path=Path(args.exec_env) if args.exec_env else None,
        )
        env = hydrate_environment(
            config_root=config_root,
            gcp_json_path=gcp_path,
            exec_env_path=exec_path,
        )
    backend = R2Backend(env)
    receipts = R2ReceiptStore(backend)
    process = SubprocessClient()
    dispatcher = GithubDispatcher(
        process,
        repo_root=repo_root,
        env=env,
        repository=args.repository,
        ref=args.ref or DEFAULT_REF,
    )
    supervisor = AutomationSupervisor(
        catalog=CATALOG,
        repo_root=repo_root,
        state_root=state_root,
        python_executable=args.python,
        env=env,
        receipts=receipts,
        process=process,
        dispatcher=dispatcher,
        validator=OutputValidator(backend),
    )
    return supervisor, receipts


def _add_runtime_arguments(
    parser: argparse.ArgumentParser, *, config_required: bool = True
) -> None:
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--repo-root", default=str(root))
    parser.add_argument("--state-root", default=str(root / "artifacts" / "automation"))
    parser.add_argument("--config-root", required=config_required)
    parser.add_argument("--gcp-json", help="optional external GCP JSON path override")
    parser.add_argument("--exec-env", help="optional external exec_agent.env path override")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--repository", default="mslade50/New_Seasonals")
    # Default None, not "main": ``resolve`` must be able to tell an operator
    # who typed ``--ref main`` (honoured verbatim, with a warning) from one who
    # gave no ref at all (the runtime marker's pinned tag is substituted). Every
    # dispatch path reads ``args.ref or DEFAULT_REF``.
    parser.add_argument("--ref", default=None)
    parser.add_argument(
        "--cutover-date-et",
        default=os.environ.get(
            "LOCAL_AUTOMATION_CUTOVER_DATE_ET", DEFAULT_CUTOVER_DATE_ET.isoformat()
        ),
        help="earliest ET receipt date allowed to run or dispatch",
    )


def _internal_r2_download(key: str, local: str, *, required: bool) -> int:
    from cache_io import download_to_local  # imported after child env hydration

    ok = bool(download_to_local(key, local))
    if required and not ok:
        print(f"ERROR: required R2 input unavailable: {key}")
        return 1
    return 0


def _internal_pull_intraday() -> int:
    from cache_io import download_to_local

    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas is required for intraday cache pull")
        return 1
    target = Path("data/intraday")
    target.mkdir(parents=True, exist_ok=True)
    meta_path = target / "_meta.parquet"
    if not download_to_local("intraday/15min/_meta.parquet", str(meta_path)):
        print("ERROR: canonical intraday metadata is unavailable")
        return 1
    meta = pd.read_parquet(meta_path)
    if "ticker" not in meta.columns or meta.empty:
        print("ERROR: canonical intraday metadata has no tickers")
        return 1
    failed: list[str] = []
    for ticker in meta["ticker"].astype(str):
        key = f"intraday/15min/{ticker}.parquet"
        local = target / f"{ticker}_15min.parquet"
        ok = False
        for attempt in range(3):
            if download_to_local(key, str(local)):
                ok = True
                break
            if attempt < 2:
                time.sleep(2)
        if not ok:
            failed.append(ticker)
    if failed:
        print(f"ERROR: failed to pull {len(failed)} intraday objects: {failed[:20]}")
        return 1
    print(f"pulled {len(meta)} canonical intraday parquet(s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="print the side-effect-free pipeline plan")
    plan.add_argument("pipeline", choices=sorted(CATALOG), nargs="?")
    plan.add_argument("--python", default="python")

    run = sub.add_parser("run", help="run one local-primary pipeline")
    # The legacy positional form keeps --config-root conditionally optional
    # so ``run PIPELINE --dry-run`` can remain credential-free.  A live call
    # fails closed in _make_runtime when it is omitted.
    _add_runtime_arguments(run, config_required=False)
    run.add_argument("pipeline", choices=sorted(CATALOG))
    run.add_argument("--date", help="ET receipt date override (YYYY-MM-DD)")
    run.add_argument("--dry-run", action="store_true", help="print plan only; no secrets or I/O")
    run.add_argument("--no-fallback", action="store_true")

    run_pipeline = sub.add_parser(
        "run-pipeline", help="Task Scheduler-compatible local-primary entry point"
    )
    _add_runtime_arguments(run_pipeline)
    run_pipeline.add_argument("--pipeline", choices=sorted(CATALOG), required=True)
    run_pipeline.add_argument("--date", help="ET receipt date override (YYYY-MM-DD)")
    run_pipeline.add_argument("--dry-run", action="store_true", help="print plan only; no secrets or I/O")
    run_pipeline.add_argument("--no-fallback", action="store_true")
    run_pipeline.add_argument(
        "--retry",
        action="store_true",
        help=(
            "local second-chance mode: gated by the pipeline's retry window instead "
            "of its primary window; every job with a success receipt is a no-op, an "
            "expired pre-side-effect lease is re-run, indeterminate is never re-run; "
            "pre-existing indeterminate receipts do not fail the run"
        ),
    )

    fallback = sub.add_parser(
        "fallback-due", help="dispatch missing jobs during their bounded ET fallback window"
    )
    _add_runtime_arguments(fallback, config_required=False)
    fallback.add_argument(
        "--pipeline",
        choices=[*sorted(CATALOG), "all"],
        default="all",
        help="limit the receipt check to one pipeline (default: all)",
    )
    fallback.add_argument("--date", help="ET receipt date override (YYYY-MM-DD)")
    fallback.add_argument(
        "--job",
        help="dispatch only this job and its prerequisite closure (requires one pipeline)",
    )

    status = sub.add_parser("status", help="show latest R2 receipts")
    _add_runtime_arguments(status, config_required=False)
    status.add_argument("pipeline", choices=sorted(CATALOG), nargs="?")
    status.add_argument("--date", help="ET receipt date override (YYYY-MM-DD)")
    status.add_argument("--json", action="store_true")

    health = sub.add_parser(
        "health", help="run the receipt/data/delivery health battery from the pinned runtime"
    )
    _add_runtime_arguments(health)
    health.add_argument("--skip-tests", action="store_true")
    health.add_argument("--skip-automation", action="store_true")

    resolve = sub.add_parser(
        "resolve",
        help=(
            "explicitly resolve an indeterminate or expired pre-side-effect "
            "receipt after operator review"
        ),
    )
    _add_runtime_arguments(resolve, config_required=False)
    resolve.add_argument("--pipeline", choices=sorted(CATALOG), required=True)
    resolve.add_argument("--job", required=True)
    resolve.add_argument("--date", required=True, help="ET receipt date (YYYY-MM-DD)")
    resolve.add_argument(
        "--disposition",
        choices=("success", "retryable_failure"),
        required=True,
    )
    resolve.add_argument("--reason", required=True)

    download = sub.add_parser("_r2-download", help=argparse.SUPPRESS)
    download.add_argument("key")
    download.add_argument("local")
    download.add_argument("--required", action="store_true")
    sub.add_parser("_pull-intraday", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "_r2-download":
        return _internal_r2_download(args.key, args.local, required=args.required)
    if args.command == "_pull-intraday":
        return _internal_pull_intraday()
    if args.command == "plan":
        pipelines = [CATALOG[args.pipeline]] if args.pipeline else CATALOG.values()
        print("\n\n".join(render_plan(p, python_executable=args.python) for p in pipelines))
        return 0
    if args.command in {"run", "run-pipeline"} and args.dry_run:
        print(render_plan(CATALOG[args.pipeline], python_executable=args.python))
        return 0

    now = _utc_now()
    run_date = getattr(args, "date", None) or _et_run_date(now)
    retry_mode = bool(getattr(args, "retry", False))
    if args.command in {"run", "run-pipeline", "fallback-due", "resolve", "health"}:
        try:
            selected_day = dt.date.fromisoformat(run_date)
            cutover_day = dt.date.fromisoformat(args.cutover_date_et)
        except ValueError as exc:
            raise AutomationError("--date and --cutover-date-et must use YYYY-MM-DD") from exc
        if selected_day < cutover_day:
            print(
                f"No action: ET receipt date {selected_day} precedes automation "
                f"cutover {cutover_day}."
            )
            return 0
    if args.command == "fallback-due" and args.job:
        if not args.pipeline or args.pipeline == "all":
            raise AutomationError("--job requires --pipeline with one concrete pipeline")
        known_jobs = {job.id for job in CATALOG[args.pipeline].jobs}
        if args.job not in known_jobs:
            raise AutomationError(
                f"job {args.job!r} is not in pipeline {args.pipeline!r}"
            )
    if args.command == "run-pipeline" and retry_mode:
        pipeline = CATALOG[args.pipeline]
        if pipeline.retry_at_et is None or pipeline.retry_until_et is None:
            raise AutomationError(f"pipeline {pipeline.id} has no local retry window")
        if not pipeline.retry_is_due(now.astimezone(ET)):
            print(
                f"No action: retry launch for {pipeline.id} is outside its local "
                f"retry ET window {pipeline.retry_at_et.strftime('%H:%M')}-"
                f"{pipeline.retry_until_et.strftime('%H:%M')} or active cadence."
            )
            return 0
    elif args.command == "run-pipeline":
        pipeline = CATALOG[args.pipeline]
        now_et = now.astimezone(ET)
        if not pipeline.local_is_due(now_et):
            print(
                f"No action: Task Scheduler launch for {pipeline.id} is outside "
                f"its local ET window {pipeline.run_at_et.strftime('%H:%M')}-"
                f"{pipeline.fallback_at_et.strftime('%H:%M')} or active cadence."
            )
            return 0
    supervisor, receipts = _make_runtime(
        args,
        controller_only=args.command in {"fallback-due", "status", "resolve", "health"},
    )
    state_root = Path(args.state_root).resolve()

    if args.command == "health":
        log_path = state_root / "logs" / run_date / f"health-{uuid.uuid4().hex[:8]}.log"
        health_env = dict(supervisor.env)
        health_env["NEW_SEASONALS_AUTOMATION_STATE_ROOT"] = str(state_root)
        health_env["NEW_SEASONALS_CONFIG_ROOT"] = str(Path(args.config_root).resolve())
        health_argv = [
            args.python,
            "-u",
            str(Path(args.repo_root).resolve() / "scripts" / "repo_health_check.py"),
        ]
        if args.skip_tests:
            health_argv.append("--skip-tests")
        if args.skip_automation:
            health_argv.append("--skip-automation")
        # The battery itself only READS (R2 receipts, local files, the test
        # collector), so a held lock is not a reason to skip it. It is a
        # finding: at 07:30 ET a live lock means the 04:10 primary never
        # returned. Report that as FAIL, run the battery anyway, stay red.
        # (2026-09-04 verify: an uncaught LockUnavailable made a hung primary
        # invisible to the health task -- no battery, no log line, only Task
        # Scheduler's 0x1.)
        rc = 1  # bound before anything can raise: a red default, never NameError
        lock = GlobalFileLock(state_root / "automation_supervisor.lock")
        locked_out: str | None = None
        try:
            lock.acquire()
        except LockUnavailable as exc:
            since = lock.holder_since_et() or "an unknown time"
            locked_out = (
                f"FAIL health: primary still holds the supervisor lock since "
                f"{since} ({exc}); running the read-only battery without it"
            )
            # Straight to stdout FIRST. The run log lives under the state root,
            # and an unusable state root is exactly the case where the finding
            # must survive (2026-09-04 verify, finding 3).
            print(locked_out, flush=True)
        try:
            try:
                run_logger: RunLogger | StdoutLogger = RunLogger(log_path)
            except OSError as exc:
                print(
                    f"WARN health: cannot open the run log {log_path} ({exc}); "
                    "logging to stdout only",
                    flush=True,
                )
                run_logger = StdoutLogger()
            with run_logger as logger:
                logger.line(f"health date_et={run_date} log={log_path}")
                if locked_out and not isinstance(logger, StdoutLogger):
                    logger.line(locked_out)  # already on stdout; this is the file copy
                rc = supervisor.process.stream(
                    health_argv,
                    cwd=Path(args.repo_root).resolve(),
                    env=health_env,
                    timeout_seconds=3600,
                    logger=logger,
                )
        finally:
            lock.release()
        if locked_out:
            return 1
        return 0 if rc == 0 else 1

    if args.command == "status":
        pipeline_ids = [args.pipeline] if args.pipeline else list(CATALOG)
        rows: list[dict[str, Any]] = []
        for pipeline_id in pipeline_ids:
            for job in CATALOG[pipeline_id].jobs:
                receipt = receipts.latest(run_date, job.id)
                # ``status`` is the normalised view (an expired running lease
                # reads ``expired``); ``receipt_status`` is the raw R2 field.
                rows.append(
                    {
                        "pipeline": pipeline_id,
                        "job_id": job.id,
                        "status": effective_status(receipt, now),
                        "receipt_status": receipt.status if receipt else "missing",
                        "phase": receipt.phase if receipt else None,
                        "lease_expires_at_utc": (
                            receipt.lease_expires_at_utc if receipt else None
                        ),
                        "source": receipt.source if receipt else None,
                        "updated_at_utc": receipt.updated_at_utc if receipt else None,
                        "automation_token": receipt.automation_token if receipt else None,
                    }
                )
        if args.json:
            print(json.dumps(rows, indent=2, sort_keys=True))
        else:
            for row in rows:
                print(
                    f"{row['pipeline']:<18} {row['job_id']:<28} "
                    f"{row['status']:<8} {row['source'] or '-'}"
                )
        return 0

    if args.command == "resolve":
        jobs = {job.id: job for job in CATALOG[args.pipeline].jobs}
        if args.job not in jobs:
            raise AutomationError(
                f"job {args.job!r} is not in pipeline {args.pipeline!r}"
            )
        current = receipts.latest(run_date, args.job)
        expired_pre_effect = bool(
            current
            and current.status == "running"
            and current.phase == "local_pre_side_effect"
            and current.lease_expired(now)
        )
        if current is None or (
            current.status != "indeterminate" and not expired_pre_effect
        ):
            state = current.status if current else "missing"
            raise AutomationError(
                "only an indeterminate receipt or an expired local_pre_side_effect "
                f"receipt can be resolved; current={state}"
            )
        resolved = dataclasses.replace(
            current,
            status=("success" if args.disposition == "success" else "failure"),
            source="operator",
            updated_at_utc=dt.datetime.now(tz=UTC).isoformat(),
            phase="manual_resolved",
            lease_expires_at_utc=None,
            detail=args.reason,
        )
        receipts.transition(resolved, update_latest=True)
        print(
            f"Resolved {args.pipeline}/{args.job} {run_date} as "
            f"{resolved.status}: {args.reason}"
        )
        # F10 (2026-09-03): the AM site deploys stayed `missing` all day after
        # the operator resolved scan_am. Say what is due now; never dispatch.
        _print_due_after_resolution(
            pipeline=CATALOG[args.pipeline],
            job_id=args.job,
            run_date=run_date,
            disposition=args.disposition,
            receipts=receipts,
            now_utc=now,
            config_root=args.config_root,
            ref=args.ref,
            # Under a scheduled task ``--repo-root`` IS the pinned runtime
            # worktree, which is where the marker lives.
            runtime_root=args.repo_root,
        )
        return 0

    lock = GlobalFileLock(state_root / "automation_supervisor.lock")
    pipeline_ids: list[str]
    if args.command in {"run", "run-pipeline"}:
        pipeline_ids = [args.pipeline]
    else:
        candidates = (
            [args.pipeline]
            if args.pipeline and args.pipeline != "all"
            else list(CATALOG)
        )
        now_et = now.astimezone(ET)
        pipeline_ids = [pid for pid in candidates if CATALOG[pid].fallback_is_due(now_et)]
        if not pipeline_ids:
            print("No pipeline is inside its ET fallback window.")
            return 0

    # A retry or controller run answers for what IT did. A receipt that was
    # already indeterminate when it started is reported, not counted (F2: the
    # 2026-09-02 execution_report ambiguity turned every later controller tick
    # red while each tick did nothing).
    scoped_exit = args.command == "fallback-due" or retry_mode
    mode = "retry" if retry_mode else args.command
    failures = 0
    try:
        lock.acquire()
    except LockUnavailable as exc:
        since = lock.holder_since_et() or "an unknown time"
        if retry_mode:
            print(
                f"No action: {exc}; the primary run still holds the supervisor "
                f"lock since {since}.",
                flush=True,
            )
            return 0
        if args.command == "fallback-due":
            # The 07:30 health task runs this first. A traceback there hid a
            # hung primary from both the recovery and the battery (2026-09-04
            # verify). Say it plainly and stay red; never a stack trace, and
            # straight to stdout (the run log lives under the state root, which
            # is itself a thing that can be unusable).
            print(
                f"FAIL fallback-due: primary still holds the supervisor lock "
                f"since {since} ({exc}); no job was inspected or dispatched.",
                flush=True,
            )
            return 1
        raise
    try:
        for pipeline_id in pipeline_ids:
            token_stub = uuid.uuid4().hex[:8]
            log_path = state_root / "logs" / run_date / f"{pipeline_id}-{token_stub}.log"
            reported: list[str] = []
            with RunLogger(log_path) as logger:
                logger.line(
                    f"pipeline={pipeline_id} date_et={run_date} mode={mode} log={log_path}"
                )
                outcomes = supervisor.run_pipeline(
                    pipeline_id,
                    run_date=run_date,
                    allow_fallback=not getattr(args, "no_fallback", False),
                    github_only=args.command == "fallback-due",
                    only_jobs=(
                        {args.job}
                        if args.command == "fallback-due" and args.job
                        else None
                    ),
                    logger=logger,
                )
                for outcome in outcomes:
                    if outcome.status not in {"failure", "blocked", "indeterminate"}:
                        continue
                    if scoped_exit and outcome.preexisting:
                        reported.append(f"{pipeline_id}/{outcome.job_id} ({outcome.status})")
                        continue
                    failures += 1
                if reported:
                    logger.line(
                        f"reported, not counted against this {mode} run for {run_date}: "
                        + ", ".join(reported)
                        + " (pre-existing receipts; operator resolution required)"
                    )
    finally:
        lock.release()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
