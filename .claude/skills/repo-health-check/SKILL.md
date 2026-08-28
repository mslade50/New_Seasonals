---
name: repo-health-check
description: Scheduled health sweep of the New_Seasonals local-primary pipeline - R2 automation receipts, pinned-runtime logs, data freshness, fragility PIT integrity, agent-product delivery, and guard-test collection. Use when running the scheduled check or when McKinley asks whether everything is running as intended.
---

# Repo Health Check

Verify the whole pipeline is running as intended, investigate anything that
is not, and leave a legible verdict. This skill is designed to run headless
on a schedule (see "Scheduling" below) and interactively on request.

## Stage 1 - run the battery

```
python scripts/repo_health_check.py
```

It prints one `[OK] / [WARN] / [FAIL]` line per check plus a summary, and
exits 1 on any FAIL. The checks, and what each failure actually means:

| Check | What a FAIL means |
|---|---|
| `automation:*` | A critical component's latest R2 receipt is missing, failed, indeterminate, has an expired running lease, or is older than its cadence allows. `indeterminate` requires operator review because an external side effect may already have happened. |
| `data:*` | A core parquet is stale or unreadable. Locally this can also mean the repo/R2 copy just hasn't been pulled - the detail line says so. |
| `pit:rd2_fragility` | Frozen point-in-time fragility history CHANGED between runs. This is the most serious single failure the battery can detect: it means the append-only contract broke (most likely daily_risk_report's silent full-rewrite fallback fired) and live `frag_risk_bands` sizing is on a drifted recompute vintage. |
| `journal:*` | An append-only journal has unparseable lines. |
| `delivery:pitch` / `delivery:context` | The last expected Daily Pitch morning / Market Context evening produced no journaled delivery. A green Task Scheduler entry means nothing here - these checks are the durable evidence. |
| `triggers:*` | A pinned local-primary pipeline log under the operational worktree went quiet. The hourly GitHub controller is backup, but a stale local trigger is still a production degradation. |
| `tests:collect` | Guard-test collection errors, or guard files contributing zero collectable tests (they silently never run in CI - `test_eod_dd.py` and `test_olv_fill_window.py` were found in exactly this state on 2026-08-12). |

## Stage 2 - investigate every FAIL (and suspicious WARNs)

Do not report a bare failure line; find the cause first. Playbook by check:

- **automation failure**: read the latest receipt and matching pinned-runtime
  log first. For a GitHub-sourced receipt, use `gh run view <github_run_id>
  --log-failed`. Distinguish a local producer failure, GitHub backup failure,
  expired lease, missing Task Scheduler trigger, and `indeterminate` external
  state. Never resolve `indeterminate` without verifying the named side effect.
- **data staleness**: compare the canonical R2 object with the pinned runtime's
  local copy. The operational runtime never uses `git pull`; a stale local copy
  may need the normal bounded R2 pull on the next producer run.
- **pit:rd2_fragility**: treat as an incident. `git log -p --follow
  data/rd2_fragility.parquet` to find the rewriting commit, check the
  risk_report run logs for the "unreadable fragility cache" warning path in
  `daily_risk_report.py` (~line 815). Until restored from git history, note
  that FAMILY4 / 3x Bear Fade / Monthly Weak Close sizing is on the wrong
  vintage. Do NOT fix by re-running the report; the fix is restoring the
  frozen rows from the last good commit.
- **delivery failure**: read `scripts/logs/daily_pitch_<date>.log` or
  `market_context_<date>.log` (also `*_last_run.log`). Distinguish agent
  gave up politely, upstream state build failed, or the machine was off.
- **tests**: run the named file directly (`python -m pytest tests/<file> -q`)
  to reproduce, and say whether the pinned invariant is live-money-relevant.

Read-only investigation only. Never fix data files, re-run producers, or
push commits from this skill - the deliverable is the diagnosis. The one
exception: the battery script maintains its own tripwire state file
(`data/health_check_state.json`); leave it alone.

## Stage 3 - the verdict

Write the report to `scripts/logs/health_check_<YYYY-MM-DD>.md` and print it.
Format:

```
# Health check <date> <time ET>
VERDICT: HEALTHY | DEGRADED | BROKEN

## Failures (if any)
- <check>: <what happened> -> <root cause found> -> <what needs doing, by whom>

## Warnings worth a look
- ...

## All clear
<one line per green subsystem group>
```

- **HEALTHY**: zero FAILs. WARNs listed but explained.
- **DEGRADED**: FAILs exist but live order flow is safe (e.g. a report email
  missed, a stale intraday cache).
- **BROKEN**: anything touching live sizing or order staging - the PIT
  tripwire, a failed daily_screener/update_master_prices/risk_report chain
  on a trading morning, an unreadable master_prices.
- Lead the verdict line with WHY in one sentence. A scheduled run's log may
  be the only thing McKinley reads - it must stand alone.

## Scheduling

House rule: eyeball several manual runs before registering the task.

Manual: `scripts\run_repo_health_check.bat` (logs to
`scripts/logs/health_check_last_run.log` + a dated copy).

The phased installer in `scripts/install_local_automation_tasks.ps1` registers
`New Seasonals Local - health` at 07:30 ET from the pinned runtime. Do not use
the old `schtasks /Create` command or the superseded `Repo Health Check` task;
cutover disables that entry without deleting it.

The headless run uses the scoped allowlist in
`scripts/health_headless_settings.json` (read-only + gh/git/pytest; no web,
no writes outside logs), the market-context permission pattern.
