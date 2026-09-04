# Brief: build_tests_hygiene (make the suite's red honest so verify agents have a green baseline)

Date 2026-09-04. Plan: `docs/plan_2026-09-04.md` (section 0, section 6). Type: BUILD (tests only).

## Decision and why
`python -m pytest -q tests/` on main has 7 failures. Six are tests of execution-tab features the 2026-09-02 guard patch deliberately disabled (`add_to_position` / `trim_readd` until they get an atomic exact-identity broker lifecycle; the scheduled-options executor), acknowledged in commit 43521a51's message; one is a 30-second network timeout in a runtime-safety test that spawns `_pull-intraday`. A red suite hides the next real failure and every verify agent this fortnight needs a clean baseline. Decision: mark the six as expected failures with the reason string pointing at the disabling commit, and mark the network test so it is skipped when R2 credentials are absent or the network is unreachable. Do not delete tests, do not change production code, do not weaken any assertion.

## Files you own
`tests/test_execution_time_stop_trim.py`, `tests/test_scheduled_options.py`, `tests/test_automation_runtime_safety.py`. Nothing else. If a `conftest.py` change is unavoidable for a marker, say so in the report and make the smallest possible edit to `tests/conftest.py`.

## Hard rules
Section 0 of the plan. No edits outside the three files (plus the conftest exception). No production code changes. Never commit.

## Intent
- The four `test_agent_accepts_time_stop_only_for_add_and_readd[...]` / `test_executor_accepts_time_stop_only_for_add_and_readd[...]` cases and the two `test_scheduled_options` cases (`test_dynamic_executor_resolves_and_submits_market_order`, `test_signed_command_handler_persists_and_cancels_schedule`): `pytest.mark.xfail(strict=True, reason="disabled by 43521a51 until <feature> has an atomic broker lifecycle; re-enable with that work")`. Strict, so that when the feature returns the xfail turns into a failure and someone removes the mark.
- `test_direct_script_pull_intraday_imports_repo_module_without_pythonpath`: keep the test, but gate it so it is skipped (with a reason) when R2 credentials are not present in the environment or when the subprocess would need network; if the test's real intent (import without PYTHONPATH) can be met by a variant that does not touch R2, add that variant as a new always-on test beside it.
- Run the full suite before and after; the after-run must be 0 failed, with the six listed as xfailed and the one as skipped (or replaced by the offline variant passing).

## Recon first
Write `artifacts/build_2026-09-04/tests_hygiene/00_plan.md` with the exact marks you will add and why each is xfail vs skip, before editing.

## Verification
Run `python -m pytest -q tests/ -p no:cacheprovider -rA 2>&1 | tail -40` before and after; save both to `artifacts/build_2026-09-04/tests_hygiene/`. Produce `artifacts/build_2026-09-04/tests_hygiene/checks.json` from a script that parses the after-run: `{"before_failed": int, "after_failed": int, "after_xfailed": int, "after_skipped": int, "after_passed": int, "xfail_strict_all": bool, "production_files_touched": []}` where `production_files_touched` is computed from `git status --porcelain` minus your owned files and must be empty.

## Report
Section 6 format. Handoff: the exact `git add` list for the mind.
