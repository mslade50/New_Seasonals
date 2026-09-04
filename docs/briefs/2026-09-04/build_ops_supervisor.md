# Brief: build_ops_supervisor (D11: recovery that does not depend on GitHub's cron, plus receipt and log hygiene)

Date 2026-09-04. Plan: `docs/plan_2026-09-04.md` (section 0, D1, D11, section 6). Incident record: `docs/incidents/2026-09-03_scan_am_stall.md` (read it first). Type: BUILD (production automation: a VERIFY agent follows).

## Decision and why
On 2026-09-03 a stalled premarket scan had no automatic recovery for 3h23m because the only recovery path was GitHub's shared cron, which fires roughly six times a day, not hourly. The local 07:30 health task now dispatches `fallback-due` for scan_am, but still only to GitHub. Decision: add a LOCAL second chance that needs nothing from GitHub, make the controller's red signal specific to the date it is recovering, keep runtime logs across cutovers, and write the cutover cadence rule into the operating doc. All of it ships in the single v9 cutover (plan D1); nothing here is applied to the live task set by you.

## Files you own
`scripts/automation_supervisor.py`, `scripts/install_local_automation_tasks.ps1`, `scripts/run_local_automation.ps1`, `.github/workflows/local_automation_fallback.yml`, `docs/local_automation_task_scheduler.md`, and the tests that cover them (`tests/test_automation_supervisor*.py`, `tests/test_automation_runtime_safety.py` is owned by the tests-hygiene builder running in parallel: if you must touch it, the tree may be dirty there, build on what is present and say so). Nothing else.

## Hard rules
Section 0 of the plan. Never register, enable, disable or delete a Task Scheduler task; never run `run-pipeline`, `fallback-due` (except `--dry-run` / `plan` forms that the code documents as side-effect free), or `resolve`; never dispatch a workflow; never write to R2 except through a test double. The 2026-09-02 `execution_report` receipt is the mind's to resolve, not yours.

## Intent
1. Local second chance (F1). The installer registers one additional S4U task, `premarket-retry`, weekdays 05:30 ET, running `run-pipeline --pipeline premarket`. Confirm in code (and pin in a test) that `run-pipeline` on a pipeline whose jobs already hold `success` receipts is a no-op for every job, that a job holding an EXPIRED `running` pre-side-effect receipt is re-run locally (that is the 09-03 shape; the scan never started, so re-running is safe), and that a job holding `indeterminate` or a post-side-effect state is NOT re-run. If the second and third behaviours are not what the code does today, implement them; document the state table in the module docstring.
2. Lease-expiry sweep. `status` and `health` currently disagree with the controller on an expired `running` receipt. Add a read-side normalisation so all three report `expired` for that state, and have `health` count it as FAIL with the job name, so the 07:30 email is unambiguous.
3. Controller exit code (F2). `fallback-due` exits non-zero only for the date it is recovering; an older date's indeterminate receipt is reported, not fatal. Pin in a test.
4. Log retention (F4). Runtime logs write to a stable path outside the pinned worktree (choose `artifacts/automation_logs/<pipeline>/<date>.log` or the existing `scripts/logs/` convention, whichever the code already favours; say which) and the cutover phase in the installer copies the outgoing worktree's logs forward before removing it. Add a health WARN when the current pipeline log is missing.
5. Controller cadence (F7). Add cron lines to `local_automation_fallback.yml` inside the premarket fallback window (05:20-08:55 ET, both DST regimes) at about 20-minute spacing, so a dropped tick is not a lost morning; keep the existing lines.
6. After an operator `resolve --disposition success`, print the dependent jobs that are now due and the exact `fallback-due` command for them (F10). Do not auto-dispatch.
7. Docs: `docs/local_automation_task_scheduler.md` gains a "Cutover cadence" section (one cutover per day, never 04:00-09:35 ET, never before the incident write-up exists, always outside a trading window), a "Current generation" note that points at the marker file instead of naming a version, and the new `premarket-retry` task in its task table.
8. Prune switch (F6). The installer gains `-PruneSuperseded` that unregisters every `New Seasonals Local v*` task not belonging to the current generation; default off; dry-run listing when `-WhatIf`. You do not run it.

## Recon first
`artifacts/build_2026-09-04/ops_supervisor/00_plan.md`: the receipt state machine as the code implements it today (states, transitions, which command reads which), where logs are written, how the installer names generations. Then build.

## Verification
`python -m pytest -q tests/test_automation_supervisor*.py tests/test_automation_runtime_safety.py -p no:cacheprovider` plus the full suite. `python scripts/automation_supervisor.py plan --pipeline premarket` and `status --date 2026-09-03` (read-only) before and after, saved to your artifacts folder; the 09-03 scan_am row must now read `expired`-aware in status output while the underlying receipt is untouched. PowerShell `-WhatIf` run of the installer showing the new task and the prune list, saved as text. `artifacts/build_2026-09-04/ops_supervisor/checks.json`: `{"tests_failed": int, "rerun_noop_on_success": bool, "rerun_on_expired_pre_side_effect": bool, "no_rerun_on_indeterminate": bool, "controller_exit_scoped_to_date": bool, "log_path_outside_worktree": bool, "cron_lines_in_window": int, "prune_switch_default_off": bool, "tasks_registered_by_you": 0, "files_touched": [...]}`.
No screenshots.

## Report
Section 6 format. Handoff: the `git add` list; the exact operator steps for the v9 cutover (tag, prepare, cutover time window, the `wevtutil` command to enable the Task Scheduler Operational log which needs an admin shell, the prune command); anything the verify agent should attack first.

## Round 2 (2026-09-04, after verify_ops_supervisor returned PASS with gaps)
The worktree holds round 1 uncommitted (scripts/automation_supervisor.py, install_local_automation_tasks.ps1, run_local_automation.ps1, the fallback workflow, the doc, four tests). Build on it. Evidence: `artifacts/verify_2026-09-04/ops_supervisor/` (checks.json, attack_detail.json, findings 1-8).
1. `health` and `fallback-due` must not crash on LockUnavailable: the health handler catches it, emits a FAIL line "primary still holds the supervisor lock since <lock mtime ET>", and still runs the battery parts that need no lock; the runner's health branch continues to the battery when fallback-due fails on the lock.
2. A dependent blocked only by another writer's LIVE lease (outcome status "running") counts as preexisting for the scoped exit code.
3. Move `premarket-retry` from 05:30 to 05:45 (master_prices_am lease is 70 min); window end stays 07:00.
4. Installer: Prune message says "add -PruneSuperseded -WhatIf to list without deleting"; Copy-RetiredRuntimeLogs prints per-source counts; write the new cutover-state.json BEFORE Copy-RetiredRuntimeLogs so the enabled generation is mirrored at its own cutover.
5. `resolve`'s printed fallback-due commands default `--ref` to the runtime marker's fallback_ref when the config root holds one, and warn when the ref would be `main`.
6. Docs: "one lock covers every generation" -> "from v9 onward" plus a warning that hand-run supervisor commands from the dev checkout must pass --state-root elsewhere; fix the cutover-state mirror wording; in the workflow comment document the EDT 13:07Z tick landing at 09:07 ET inside the discretionary window and the off-window ticks.
Re-run the five test files and the full suite; refresh checks.json; report in section 6 format with a diff summary.
