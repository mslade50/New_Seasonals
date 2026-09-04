# Brief: recon_onedrive (read-only audit of C:\Users\McKinley Slade\OneDrive\trading_ibkr)

Date 2026-09-04. Plan: `docs/plan_2026-09-04.md` (read section 0 first). Type: RECON. You write nothing outside your scratch folder.

## Decision and why
The directory `C:\Users\McKinley Slade\OneDrive\trading_ibkr` places every live order for this book and is not in any git repository. Before anything in it is edited this fortnight (pitch_moo arming under D5, book_snapshot margin tags under D11) the mind needs an inventory, the armed-runner list, the test state, and a concrete, safe way to put it under version control in place without breaking the Task Scheduler paths that point at it. This brief produces that; it changes nothing.

## Files you own
None in the repo. Scratch only: `artifacts/recon_2026-09-04/onedrive/` (create it; it is gitignored). Put your exact recon plan, scripts and outputs there.

## Hard rules
Section 0 of the plan, plus: NEVER execute any `.py`, `.bat` or `.ps1` in trading_ibkr except a test file you have first read in full and confirmed mocks the broker (no live `connect()` on ib_insync / ib_async; no Sheets client). If in doubt, do not run it and say so. Do not open TWS. Do not print secret values; report key NAMES and file paths only. Do not create files inside trading_ibkr.

## Intent
1. Inventory every file (`.py .bat .ps1 .json .jsonl .flag .csv .env .txt`) with size, mtime, one-line purpose, and class: live runner (name its Task Scheduler task via `Get-ScheduledTask` read-only), library, test, journal/state, secret, other. List every OneDrive conflict/duplicate copy (names containing a hostname, `-copy`, `(1)`, etc.).
2. Armed state right now: which activation flags exist (`pitch_moo_enabled.flag`, `event_moo_enabled.flag`, `radar_trail_enabled.flag`, any other `*.flag`), and therefore which runners would transmit if their task fired. Cross-check against the registered tasks and their enabled state and last run result.
3. Version control: is there a `.git`? Any backup folders or dated copies? How would last week's `order_staging.py` be recovered today (OneDrive version history, a backup, nothing)? Design the in-place git plan: the exact `.gitignore` (every secret, journal, flag, credential, `.env`, `*.json` state that must not be committed), whether OneDrive's file locking or Files-On-Demand can corrupt a repo in that folder (cite what you observe: are files hydrated? attributes?), and the alternative of a mirror repo elsewhere with a sync script. Recommend one, with the command sequence McKinley would run himself.
4. Tests: run every test file that meets the safety rule, from that directory, `python -m pytest -q <file> -p no:cacheprovider`. Verbatim results. List runners with no test.
5. Config drift vs the repo. For each, give file, line, value, and whether it matches the repo's source of truth: per-strategy daily cap 250 bps; OVS scale-out 0.4 near at 1 ATR; the EOD-DD Friday-only gate; any remaining NO_STOP_SHORT_NOTIONAL_CAP (CLAUDE.md memory says the 25% cap was removed; confirm); Account_Value parity abort at 750000; clientIds (OLV 99/98, event 147, pitch 148); RADAR_STRATEGY string; native MOC/DAY encoding in event_moo and pitch_moo and the verify-the-reject guard; Fill_Window_Days / Entry_Expire_Time; GapDerate handling; T1_Open_Filters; Manual_Limit; any hardcoded `C:\Users\mckin` path; where `credentials.json`, Sheets IDs, TWS host/port are read from.
6. Safety review of the order path, read-only: order_staging -> eq_order_entry / pa_order_entry dedup and idempotency (what happens if the 9:31 chain runs twice; if the staging tab is stale; if TWS is logged out; if account value differs), olv_exit_moo's 9:25 cutoff fallback, event_moo / pitch_moo reject verification. Severity-rate each gap.
7. For D5 specifically: read `pitch_moo.py` and its test in full and list, with line references, every place the mind must check before arming it: order type encoding per leg type, the activation flag, the Manual_Only refusal, the partial-approval refusal, the $15k basket and ATR-percent-of-price band, the 9:05 / 9:32 pass routing, the clientId, and what it journals. Note anything that reads a file the repo's `daily_pitch.py` TAB_COLUMNS does not write.
8. Journals: what the placed-orders journals, `olv_exit_placed.json`, event/pitch logs show for the last 10 trading days (orders per runner per day, errors).

## Recon first
Write `artifacts/recon_2026-09-04/onedrive/00_plan.md` listing the files you will read and the scripts you will write, before reading beyond the directory listing.

## Verification
Produce `artifacts/recon_2026-09-04/onedrive/checks.json` from a script with keys:
`{"inventory_count": int, "git_present": bool, "conflict_copies": [..], "flags_present": [..], "armed_runners": [..], "tests_run": [{"file":..., "passed":int, "failed":int, "skipped":int}], "tests_skipped_unsafe": [..], "drift_mismatches": [{"item":..., "onedrive_value":..., "repo_value":...}], "stale_mckin_paths": [..], "secrets_files": [..]}`.
No screenshots required.

## Report
Use section 6 of the plan. Under Findings, rank by money at risk. Under Handoff, give the mind (a) the recommended git plan as a numbered command list for McKinley, (b) the pitch_moo pre-arming checklist, (c) anything that must be fixed in OneDrive before D11's book_snapshot edit.
