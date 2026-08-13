---
name: repo-health-check
description: Scheduled health sweep of the New_Seasonals pipeline - GHA workflow status, data freshness, fragility PIT integrity, agent-product delivery, trigger chain, guard-test collection. Use when running the scheduled check or when McKinley asks whether everything is running as intended.
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
| `gha:*` | A weekday-critical workflow's latest run failed, or is older than its cadence allows. GitHub sheds scheduled crons under load and NEVER backfills, so "too old" usually means the cron never fired - no red X exists anywhere. |
| `data:*` | A core parquet is stale or unreadable. Locally this can also mean the repo/R2 copy just hasn't been pulled - the detail line says so. |
| `pit:rd2_fragility` | Frozen point-in-time fragility history CHANGED between runs. This is the most serious single failure the battery can detect: it means the append-only contract broke (most likely daily_risk_report's silent full-rewrite fallback fired) and live `frag_risk_bands` sizing is on a drifted recompute vintage. |
| `journal:*` | An append-only journal has unparseable lines. |
| `delivery:pitch` / `delivery:context` | The last expected Daily Pitch morning / Market Context evening produced no journaled delivery. A green Task Scheduler entry means nothing here - these checks are the durable evidence. |
| `triggers:*` | A local AM dispatch log in `C:\Scripts\logs` went quiet (WARN only - the GHA fallback crons cover a missed dispatch, at worse latency). |
| `tests:collect` | Guard-test collection errors, or guard files contributing zero collectable tests (they silently never run in CI - `test_eod_dd.py` and `test_olv_fill_window.py` were found in exactly this state on 2026-08-12). |

## Stage 2 - investigate every FAIL (and suspicious WARNs)

Do not report a bare failure line; find the cause first. Playbook by check:

- **gha failure**: `gh run list --workflow=<file> --limit 5`, then
  `gh run view <id> --log-failed`. Distinguish (a) the job ran and broke,
  (b) the cron was shed (no recent run at all), (c) the local dispatch task
  didn't fire (cross-check the matching `triggers:*` line and
  `C:\Scripts\logs\trigger_*.log`).
- **data staleness**: check whether the GHA writer succeeded (if yes, the
  local copy needs `git pull` / an R2 pull and the pipeline itself is fine -
  say so explicitly). `scripts/pull_context_prices.py` refreshes
  master_prices from R2.
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

Register (weekdays 7:30 AM ET is a good slot - after the 4:10-5:10 AM chain
and the pitch, before the open):

```powershell
schtasks /Create /TN "Repo Health Check" /SC WEEKLY /D MON,TUE,WED,THU,FRI `
  /ST 07:30 /TR "C:\Users\McKinley Slade\dev\New_Seasonals\scripts\run_repo_health_check.bat" /F
```

The headless run uses the scoped allowlist in
`scripts/health_headless_settings.json` (read-only + gh/git/pytest; no web,
no writes outside logs), the market-context permission pattern.
