# Brief: doc_incident_0903 (incident write-up for the 2026-09-03 premarket scan stall)

Date 2026-09-04. Plan: `docs/plan_2026-09-04.md` (section 0, D1, D11, section 6). Type: DOCUMENT. You create exactly one file.

## Decision and why
Plan D1 allows one runtime cutover this fortnight and requires the 09-03 incident write-up to exist first. The ops audit reconstructed the timeline from receipts, GitHub runs and Task Scheduler; the runtime logs themselves were destroyed by the two cutovers that morning. This document is the record a future session reads before touching the supervisor.

## Files you own
`docs/incidents/2026-09-03_scan_am_stall.md` (new; create the directory). Nothing else.

## Hard rules
Section 0 of the plan. Read-only everywhere else. You may run `python scripts/automation_supervisor.py status --date 2026-09-03` and `--date 2026-09-02` (read-only), `gh run list` / `gh run view` (read-only), `git show 0c912f58 5e70c489`, and PowerShell `Get-ScheduledTask` / `Get-ScheduledTaskInfo` (read-only). Do not resolve any receipt, do not dispatch, do not enable or disable a task.

## Intent
Source of record: the ops audit at `C:\Users\McKinley Slade\AppData\Local\Temp\claude\C--Users-McKinley-Slade-dev-New-Seasonals\d2be6b41-9d9f-4549-a8b2-b9dfbff9ea0b\scratchpad\ops_health\REPORT.md` (read it in full; it is the mind's saved copy of the audit) plus the live sources above. Write the incident in this structure, every timestamp in ET with the source named:
1. Summary (five lines): what failed, when, impact on trading (none: Order_Staging was written 07:42, the 09:31 chain ran), who recovered it and how.
2. Timeline 2026-09-03 04:10 to 08:03, one row per event, with source (receipt event, gh run id, task result, commit hash).
3. Root cause: what is KNOWN (the receipt never left `phase=local_pre_side_effect`, the lease expired at 04:44, no immediate GitHub dispatch, the only controller tick in the window was 07:44) and what is INFERRED (an R2 receipt transition raising inside the failure handler; the fix's shape is the evidence). Say which is which.
4. Fix shipped: what `0c912f58` / `5e70c489` change (guarded transitions, `resolve` accepting expired running receipts, the 07:30 health task running `fallback-due` for scan_am), and that the two commits are byte-identical (why that happened).
5. Verification: the 2026-09-04 04:10 v8 run succeeded end to end (cite `status --date 2026-09-04`), and whether the 07:30 v8 health task fired and its result.
6. Still open, mapped to the ops audit's findings F1-F13 by number, each one line: no GitHub-independent recovery; the 09-02 execution_report receipt still indeterminate; harvest_fills absent from the pinned runtime; logs destroyed per cutover and the Task Scheduler operational log disabled; CI red on 09-02/03 (fixed 09-04 by the mind, say so if `gh run list --workflow tests.yml --limit 3` shows green); runtime churn; controller cadence; CBOE backup untested since its fix; interactive-logon dependency; AM site deploys after operator resolution; Option Surface / RadarPackExport tasks failing; ledger_git_sha unknown; hygiene.
7. Rules adopted (from plan D1 and D11): one cutover per day, never 04:00-09:35 ET, never before the write-up.

## Recon first
None beyond reading the sources; list them at the top of the file under "Sources".

## Verification
No JSON. Named checks in your report: the file exists, every timestamp has a named source, sections 1-7 present, and no statement in it contradicts `status --date 2026-09-03` output (paste that output into an appendix).

## Report
Section 6 format. Handoff: anything in the sources that contradicted each other, and any fact you could only infer.
