# Brief: build_olv_exit_fix (OLV pre-market exit runner: back to cancel-then-sell, with reject verification)

Date 2026-09-04. Plan: `docs/plan_2026-09-04.md` (section 0, rule 5, section 6). Type: BUILD in OneDrive (live money: a VERIFY agent follows; you never run the runner).

## Decision and why
On 2026-09-03 09:10 the runner `C:\Users\McKinley Slade\OneDrive\trading_ibkr\olv_exit_moo.py` placed both due OLV exits (POWI 637 sh, CMI 54 sh) as SELL MKT TIF=OPG orders that JOIN the bracket's existing OCA group. IBKR rejected both with error 201 "Invalid OCA handling method"; the runner printed `[OK] MOO exit placed`, summary `SENT_OPG`, exit code 0, and journaled both as `ACKNOWLEDGED` in `olv_exit_placed.json`. Nothing sold. The version in `_backup_20260902_legend_reservation_guard_prepatch_v2/olv_exit_moo.py` (lines ~290-334) cancelled the bracket's SELL legs, placed a standalone MKT/OPG sell, and re-armed a time exit on failure; that design filled POWI on 2026-09-02 (see `data/live_fills.parquet`). Decision: restore the cancel-then-sell design, add the verify-the-reject guard the event and pitch runners already have, make an empty or unknown status fail loud instead of counting as success, and let a re-run retry a leg whose journal row is not a confirmed placement. Everything else in the runner (leg matching, calendar-desync tolerance, primary qty clamp min(staged, leg, held), PA sells the full matched leg, 9:25 OPG cutoff then MKT DAY, clientIds 99/98, idempotency journal) stays as is.

## Files you own
`C:\Users\McKinley Slade\OneDrive\trading_ibkr\olv_exit_moo.py` and `C:\Users\McKinley Slade\OneDrive\trading_ibkr\test_olv_exits.py`. Nothing else in OneDrive; nothing in the repo except your scratch folder `artifacts/build_2026-09-04/olv_exit_fix/`.

## Hard rules
Section 0 of the plan. NEVER run `olv_exit_moo.py` or any OneDrive runner; run only `test_olv_exits.py` (read it first; it must not connect to a broker). trading_ibkr is not under git yet: BEFORE your first edit copy the two files to `C:\Users\McKinley Slade\OneDrive\trading_ibkr\_backup_20260904_olv_exit_prepatch\` (create it; this is the folder's existing backup convention). Do not touch `legend_reservation_guard.py`, `eq_order_entry.py`, `pa_order_entry.py`, or any journal/flag/env file. Do not print secret values.

## Intent
1. Placement sequence per due leg, per account: (a) match the working bracket as today; (b) cancel that bracket's SELL child legs (target, stop if any, time) as the owning clientId and wait for cancel confirmation with a bounded timeout; (c) place a standalone SELL MKT, TIF=OPG before the 9:25 cutoff else MKT DAY, with NO `ocaGroup` and no `ocaType`, orderRef in the runner's existing format; (d) poll the order status with a bounded timeout.
2. Verify-the-reject (the 2026-08-21 lesson, see `event_moo.py` and `pitch_moo.py`): on any terminal reject or cancel, look up `openTrades()` for the orderRef; a surviving order is cancelled; then RE-ARM a protective time exit for the leg (MKT, goodAfterTime at the leg's original time-exit session 15:59, as the prepatch version did) so the position is never left without an exit; journal the leg as `FAILED_REARMED` (or `FAILED_NO_REARM` if the re-arm itself fails, which must produce a loud email line); the run exits non-zero.
3. Status semantics: `ACKNOWLEDGED` only when the polled status is Submitted, PreSubmitted or Filled. An empty or unknown status after the poll timeout is journaled `UNKNOWN`, printed loudly, and exits non-zero; it is never success.
4. Re-run behaviour: a journal row in `FAILED_*` or `UNKNOWN` state allows ONE retry on the next run that day (bounded), after re-reading the live book so a filled or already-flat leg is skipped; only a row in `ACKNOWLEDGED` state produces the existing reconcile-manual outcome.
5. The reservation guard: `guarded_place_order` in `legend_reservation_guard.py` blocks new group-bound market exits when the guard is active. The standalone sell must pass in BOTH guard states; read the guard's call contract and prove it in a test with the guard active and inactive (the guard is inactive in production today; do not change it).
6. Summary and email lines: per leg, print the sequence actually taken (cancelled N legs, placed order id, final status, journal state). Replace the misleading `[OK] MOO exit placed` on a non-confirmed status.
7. Tests in `test_olv_exits.py`: keep every existing test; ADD a broker fake that implements the subset of the ib_insync surface the runner uses (placeOrder, cancelOrder, openTrades, order status transitions) and drive `guarded_place_order` through it rather than monkeypatching it away. Cases: happy path (cancel legs, place, Submitted -> ACKNOWLEDGED, exit 0); reject 201 -> survivor cancelled, time exit re-armed, FAILED_REARMED, exit non-zero; empty status -> UNKNOWN, exit non-zero; re-run after FAILED_REARMED retries once; re-run after ACKNOWLEDGED does not re-place; no `ocaGroup` on the placed exit; guard active and inactive both allow the standalone sell; PA path sells the full matched leg; 9:25 cutoff fallback still MKT DAY.

## Recon first
`artifacts/build_2026-09-04/olv_exit_fix/00_plan.md`: a diff summary between the prepatch backup and the current file (what the 09-02 rewrite changed and why, as far as the code and comments say), the guard's call contract, the journal schema, and your exact-edit plan. Then build.

## Verification
`python -m pytest -q test_olv_exits.py -p no:cacheprovider` from the OneDrive directory: 0 failed. A static check that the file contains no `ocaGroup =` assignment on the exit order path. `artifacts/build_2026-09-04/olv_exit_fix/checks.json`: `{"backup_created": bool, "tests_passed": int, "tests_failed": int, "oca_join_removed": bool, "reject_rearm_covered": bool, "unknown_status_fails_loud": bool, "rerun_retry_covered": bool, "guard_both_states_covered": bool, "runner_executed": false, "files_touched": [...]}`.
No screenshots.

## Report
Section 6 format. Handoff: what the verify agent should attack first; the exact operator step to reconcile any leg still open when the fix lands.

## Round 2 (2026-09-04, after verify_olv_exit_fix returned FAIL)
Apply these fixes in the same two files (backup already exists in `_backup_20260904_olv_exit_prepatch/`; take a second copy to `_backup_20260904_olv_exit_round1/` before editing). Evidence: `artifacts/verify_2026-09-04/olv_exit_fix/` (test_attack_sequences.py, checks.json, the report's F2-F5).
- F2 (HIGH): on a same-day re-run, when `journal_rows` is non-empty, match the leg by the journaled identity (`source_client_id`, `source_order_id` from the latest row's detail) instead of `pick_time_leg`'s single-candidate date fallback; if no match, take `_retry_without_bracket`. A stacked sibling's bracket must never be cancelled under another row's identity.
- F3 (HIGH): in `_execute_exit`, immediately before placing, re-read the exact held position and set `exit_qty = min(exit_qty, held_now - already_queued)`; if <= 0 journal `SKIPPED_FLAT` and return without placing. Applies on every path, including post-9:25 MKT DAY.
- F4 (MEDIUM): after the reject-verify pause, re-read the placed trade's status before classifying, so a terminal status that arrives during the pause takes the FAILED_REARMED branch, not UNKNOWN.
- F5 (LOW): survivor detection also matches on `_order_key(t.order) == _order_key(placed.order)` when `placed` is not None, not only on exact orderRef.
- F6 (LOW): when a bracket AND a standalone sell are both working, print a `[WARN] two exits working` line (keep the leave-both behaviour).
Add the verifier's four failing sequences to `test_olv_exits.py` (port them from `artifacts/verify_2026-09-04/olv_exit_fix/test_attack_sequences.py`) so they pass, keep every existing test green, refresh checks.json, report in section 6 format.
