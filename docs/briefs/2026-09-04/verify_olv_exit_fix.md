# Brief: verify_olv_exit_fix (independent verification of the OLV exit runner fix)

Date 2026-09-04. Plan: `docs/plan_2026-09-04.md` (section 0, rule 5, section 6). Type: VERIFY. You did not build this; try to break it.

## Decision and why
The builder (`docs/briefs/2026-09-04/build_olv_exit_fix.md`, read it) changed `C:\Users\McKinley Slade\OneDrive\trading_ibkr\olv_exit_moo.py` and `test_olv_exits.py`. This runner is OLV's only stop. It runs Monday 09:10 ET on both accounts. The mind reviews only your verdict and checks.

## Files you own
None. Scratch only: `artifacts/verify_2026-09-04/olv_exit_fix/`. You may not edit anything in OneDrive or the repo.

## Hard rules
Section 0 of the plan. NEVER run the runner or any OneDrive script other than `test_olv_exits.py` (read it first to confirm it cannot connect). No TWS.

## Intent
1. Diff `_backup_20260904_olv_exit_prepatch\olv_exit_moo.py` against the current file and against `_backup_20260902_legend_reservation_guard_prepatch_v2\olv_exit_moo.py`. Confirm the placement path is cancel-legs -> standalone sell -> poll -> verify-reject -> re-arm, with no `ocaGroup`/`ocaType` on the exit order, and that nothing outside the placement/journal/status logic changed (leg matching, qty clamp, PA full-leg rule, cutoffs, clientIds).
2. Walk `legend_reservation_guard.guarded_place_order` and confirm the standalone sell is accepted in both guard states, by reading the guard, not by trusting the builder's test.
3. Read the new broker fake and every new test; try to construct a sequence the tests miss: cancel confirmation never arrives; reject arrives AFTER the poll timeout; `openTrades()` returns the order under a different orderRef suffix; the re-arm itself is rejected; a leg that filled between the book read and the placement; two legs on the same ticker (stacked positions share ticker, differ by Time_Exit_Date); the PA account with no matching leg. For each, say whether the runner fails safe (position keeps an exit, run exits non-zero, journal state honest) and cite the line.
4. Run `python -m pytest -q test_olv_exits.py -p no:cacheprovider` in the OneDrive directory; 0 failed.
5. Confirm the journal schema change (new states) does not break `olv_exit_moo`'s own reader, the daily scan email's OLV-EXIT warning path (`daily_scan.py`, grep `olv_exit_placed`), or `daily_execution_report.py`.

## Verification
`artifacts/verify_2026-09-04/olv_exit_fix/checks.json`: `{"placement_path_is_cancel_then_sell": bool, "oca_join_absent": bool, "unrelated_logic_unchanged": bool, "guard_accepts_standalone_both_states": bool, "sequences_tried": int, "sequences_failing_unsafe": [...], "tests_failed": int, "journal_consumers_ok": bool}`.
No screenshots.

## Report
Section 6 format. Findings: every defect with file:line and a one-line fix, ranked. Handoff: PASS or FAIL in one word, then the reasons, then the exact Monday-morning operator check (what to look at in the 09:10 email and TWS).
