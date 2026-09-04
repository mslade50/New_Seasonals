# Brief: verify_ops_supervisor (independent verification of the v9 supervisor/installer changes and the ledger-sha half of build_ops_fills_ledger)

Date 2026-09-04. Plan: `docs/plan_2026-09-04.md` (section 0, rule 5, D1, D11, section 6). Type: VERIFY. You did not build this; try to break it.

## Decision and why
`build_ops_supervisor` changed `scripts/automation_supervisor.py`, `scripts/install_local_automation_tasks.ps1`, `scripts/run_local_automation.ps1`, `.github/workflows/local_automation_fallback.yml`, `docs/local_automation_task_scheduler.md` and four test files (read its brief and its artifacts under `artifacts/build_2026-09-04/ops_supervisor/`). `build_ops_fills_ledger` changed the provenance block of `scripts/build_trade_ledger.py` (`_git_sha`, `GITHUB_SHA` first) plus `tests/test_ledger_provenance.py` (that file also carries ANOTHER session's staged overlay-lab hunks: `git diff --cached` shows theirs, `git diff` shows the ledger-sha hunk on top; verify only the provenance hunk). These run the production scan pipeline and the one v9 cutover. The mind reviews your verdict and checks only.

## Files you own
None. Scratch only: `artifacts/verify_2026-09-04/ops_supervisor/`.

## Hard rules
Section 0 of the plan. Never register, enable, disable or delete a task; never dispatch; never resolve; never run `run-pipeline` or `fallback-due` except with `--dry-run`/`-WhatIf` forms the code documents as side-effect free; R2 reads only through `status`. Set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` for pytest on this machine.

## Intent
Attack, in order:
1. `preexisting` propagation: construct receipt states where a genuine NEW failure in this invocation sits next to a pre-existing indeterminate on the same date; the exit code must be non-zero. Construct the reverse (only pre-existing) and it must be zero. Check `blocked` inheritance when one dependency is pre-existing and another is a fresh failure.
2. Retry semantics: `run-pipeline --retry` at 05:30 with (a) all success (no-op, exit 0), (b) scan_am holding an EXPIRED `local_pre_side_effect` lease (re-run), (c) scan_am `indeterminate` (no re-run, loud), (d) a `local_retryable` lease on a rerun-safe job mid-side-effect (what happens?), (e) lock held by a live 04:10 run (`LockUnavailable` -> exit 0 "No action"; is a hung primary then invisible until the 07:30 health? say so plainly). Also: with the window 05:30-07:00, a stall found at 07:05 has no local retry; state whether the 07:30 health `fallback-due` covers it.
3. Log root move: confirm the runner passes `--state-root <ConfigRoot>\artifacts\automation` on every supervisor call, that `scripts/repo_health_check.py` finds logs there via `NEW_SEASONALS_AUTOMATION_STATE_ROOT`, that the lock file now shared across generations cannot deadlock a v8->v9 cutover (old generation's lock vs new), and that the Cutover copy-forward regex (`Get-RuntimeRootFromTask`) handles a quoted or spaced RuntimeRoot.
4. Installer: run every `-WhatIf` form yourself (RegisterDisabled, Prune with `-PruneSuperseded`, Cutover with `-ConfirmCutover -PruneSuperseded`) and diff against the builder's saved outputs; confirm no enabled or running task is ever a prune candidate; confirm the v8 generation is retained under the documented `-RetireTaskNamePrefix`.
5. Workflow: the two new cron lines fire inside 05:20-08:55 ET in BOTH DST regimes (compute); the `AUTOMATION_RUNTIME_REF` pin is unchanged; the test that asserts the pin still matches.
6. Ledger sha: with `GITHUB_SHA` set, unset with a git repo, unset in a dir without `.git`; the parquet metadata round trip; the deploy workflow's ledger step does not scrub the env.
7. Full suite once; read the docs section for a statement that contradicts the code.

## Verification
`artifacts/verify_2026-09-04/ops_supervisor/checks.json`: `{"new_failure_never_masked": bool, "retry_cases": {"a": "...", "b": "...", "c": "...", "d": "...", "e": "..."}, "stall_after_0700_covered_by_health": bool, "state_root_on_all_calls": bool, "health_check_finds_logs": bool, "cross_generation_lock_safe": bool, "cutover_regex_quoted_paths": bool, "whatif_outputs_match_builder": bool, "prune_never_enabled_or_running": bool, "cron_in_window_both_dst": bool, "pin_unchanged": bool, "sha_cases_ok": bool, "tests_failed": int, "doc_contradictions": [...]}`.
No screenshots.

## Report
Section 6 format. Findings ranked with file:line and a one-line fix. Handoff: PASS or FAIL first, then the reasons, then the operator cutover sequence you would actually hand McKinley (verbatim commands, elevated or not marked).
