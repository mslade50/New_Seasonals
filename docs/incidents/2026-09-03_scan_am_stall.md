# Incident: 2026-09-03 premarket scan_am stall (local-primary runtime v6)

Written 2026-09-04 under plan D1 (`docs/plan_2026-09-04.md`): this record has to exist before the fortnight's one remaining runtime cutover. All times are US Eastern (EDT, UTC-4) unless a raw UTC value is quoted beside them. Every timestamp names its source.

## Sources

- Ops audit saved by the mind, read in full: `C:\Users\McKinley Slade\AppData\Local\Temp\claude\C--Users-McKinley-Slade-dev-New-Seasonals\d2be6b41-9d9f-4549-a8b2-b9dfbff9ea0b\scratchpad\ops_health\REPORT.md` (plus `tasks.json`, `r2_exposure_state.json` in the same folder). Audit run 2026-09-03 ~22:10 ET.
- R2 receipts, read directly (read-only `list_objects_v2` / `get_object` on `automation/receipts/v1/<date>/<job>/`): `2026-09-03/scan_am`, `2026-09-03/event_sleeve_am`, `2026-09-02/execution_report`, `2026-09-04/scan_am`. Script: scratchpad `read_receipts.py`.
- `python scripts/automation_supervisor.py status --date 2026-09-03` / `--date 2026-09-02` / `--date 2026-09-04` (dev checkout, read-only; verbatim in Appendix A).
- GitHub: `gh run list` (2026-09-03, 2026-09-04, `local_automation_fallback.yml`, `tests.yml`), `gh run view` on 33750722451 / 33751349896 / 33751370947 (jobs + logs), `gh api .../actions/runs/<id>` (actors), `gh pr list --head codex/fix-scan-recovery-v8`.
- Git: `git log --graph`, `git show --stat`, `git diff 0c912f58~1 0c912f58`, `git diff 0c912f58 5e70c489`, `git merge-base --is-ancestor`, `git tag --format`.
- Task Scheduler: `Get-ScheduledTask` / `Get-ScheduledTaskInfo` (read-only) at 07:27, 07:30 and 07:31 ET on 2026-09-04; `wevtutil gl Microsoft-Windows-TaskScheduler/Operational`.
- Surviving runtime worktrees: `C:\Users\McKinley Slade\dev\New_Seasonals-automation-runtime-v7\` (`.local\automation-runtime.json`, `artifacts\automation\cutover-state.json`, `artifacts\automation\logs\2026-09-03\health-867b3aa3.log`, `artifacts\automation\health-receipts\2026-09-03\scan_am.json`) and `...-runtime-v8\` (marker, `cutover-state.json`, `logs\2026-09-04\premarket-6bc339f8.log`, `premarket-ee07fca8.log`, `health-cd00ce09.log`). The v6 worktree (the one that stalled) no longer exists on disk.
- `data/pitch_journal.jsonl` (line 265, the 2026-09-03 `stand_down` record); `data/backtest_trades_full.parquet` schema metadata.

## 1. Summary

1. What failed: the `scan_am` job of the v6 `premarket` pipeline (Task Scheduler `New Seasonals Local v6 - premarket`, pinned `1119efae`, tag `automation-runtime-2026-09-02.1`) claimed its receipt at 04:14:22 and died before its side-effecting step; the process exited 1 with no failure receipt, no GitHub dispatch, and a `running` lease that expired at 04:44:22 (R2 receipt events; `Get-ScheduledTaskInfo` LastResult 1 at 04:10:10).
2. When: 2026-09-03, dead from 04:14 until the operator dispatch completed at 07:42:46 (gh run 33750722451). Nothing automatic recovered it in those 3h28m: the fallback controller's only tick in the window came at 07:44:36, seven minutes after the operator had already acted.
3. Impact on trading: none. The operator-dispatched scan staged 3 liquid and 3 overflow rows, synced 887 signal rows, sent the scan email at 07:42:41 and published `exposure_state.json` at 07:42:43 (run 33750722451 log; receipt resolution detail), 1h49m before the 09:31 `IBKR Daily Order Chain`, which ran with result 0 (`Get-ScheduledTaskInfo`). The 05:10 Daily Pitch ran on time (stand-down, `pitch_journal.jsonl` line 265) but before the scan existed.
4. Who recovered it and how: McKinley (GitHub `mslade50`) dispatched `daily_screener.yml` by hand at 07:37:45 against ref `automation-runtime-2026-09-02.1`, resolved the receipt to `success` (source `operator`) at 07:47:34, committed the fix at 07:49:51 (`5e70c489`), merged it via PR #15 at 07:52:32 (`0c912f58`), cut tag `automation-runtime-2026-09-03.1`, prepared runtime v8 at 07:55:08 and cut over at 08:03:04.
5. Side effects of the recovery: the controller tick at 07:44 dispatched a second scan (run 33751370947) that was cancelled 97 s in; the controller then marked the receipt `indeterminate` and exited 1. The 09-03 AM site deploys never ran through the pipeline (`private_site_am` / `shared_site_am` still `missing` for 2026-09-03 in `status`); a manual private-site deploy at 12:20:55 (run 33778072898) covered the private site only.

## 2. Timeline, 2026-09-03 04:10 to 08:03 ET

| ET | UTC | Event | Source |
|---|---|---|---|
| 02:30:53 (pre-window) | 06:30:53Z | Last fallback controller tick before the stall (run 33723564770, success; nothing was due). The next tick is 07:44:36, so no controller ran during the dead window. | `gh run list --workflow local_automation_fallback.yml` |
| 04:10:10 | 08:10:10Z | `New Seasonals Local v6 - premarket` fires (runtime `...-runtime-v6`, pinned `1119efae`). Final task result 0x1. | `Get-ScheduledTaskInfo` (LastRun 9/3/2026 04:10:10, LastResult 1); audit s.2 |
| 04:10:50 | 08:10:50Z | `cboe_am` success (local). | R2 receipt via v7 health log line `[OK] automation:cboe_am ... 2026-09-03T08:10:50Z` |
| 04:12:51 | 08:12:51Z | `master_prices_am` success (local). | same log, `master_prices_am` |
| 04:13:28 | 08:13:28Z | `risk_am` success (local). | same log, `risk_am` |
| 04:13:29 | 08:13:29Z | `event_sleeve_am` claims receipt (`local_pre_side_effect`); side-effect marker durable 04:14:13; success 04:14:21. | R2 events `2026-09-03/event_sleeve_am/events/` |
| 04:14:22 | 08:14:22.19Z | `scan_am` claims receipt: `status=running`, `phase=local_pre_side_effect`, token `2026-09-03-scan_am-ca1f8cf470`, `lease_expires_at_utc=08:44:22.19Z`. This is the LAST event the local process ever wrote. | R2 event `.../scan_am/events/2026-09-03T081422.193402_0000-running-local-...json` |
| 04:14 to 04:44 | | Local process exits 1 somewhere inside the pre-side-effect step. No `failure` event, no `latest.json` update, no `immediate GitHub fallback` dispatch (no `github` receipt event and no `daily_screener.yml` run exists between 04:14 and 07:37). | R2 event listing (one local event only until 11:44Z); `gh run list --created 2026-09-03` |
| 04:44:22 | 08:44:22Z | `scan_am` lease expires. Receipt is now "running with an expired lease", a state the v6/v7 `resolve` command could not act on (it accepted `indeterminate` only). | receipt `lease_expires_at_utc`; `git diff 0c912f58~1 0c912f58` (the `resolve` change) |
| 05:10 | | `Daily Pitch (agent)` runs; result stand-down (12 candidates, 22 check scripts). No scan rows existed yet. | `data/pitch_journal.jsonl` line 265 (`"kind": "stand_down", "date": "2026-09-03"`); audit s.2 (task 0x0) |
| 06:56:03 | 10:56:03Z | Runtime v7 cutover (pinned `1119efae`, same code as v6; prepared 21:32:12 ET on 09-02). Routine cutover, done while `scan_am` was dead and unnoticed. | `...-runtime-v7\artifacts\automation\cutover-state.json` `cutover_at`; `.local\automation-runtime.json` `prepared_at` |
| 07:30:30 | 11:30:30Z | `New Seasonals Local v7 - health` fires, result 0x1. Health battery: 3 FAIL (`scan_am` EXPIRED running lease via local; `execution_report` 09-02 INDETERMINATE; `data:master_prices.parquet` file missing in the fresh worktree), 8 WARN, 29 OK. Report only: v7 health had no recovery step. | `Get-ScheduledTaskInfo` (LastRun 9/3 07:30:30, LastResult 1); `...-runtime-v7\artifacts\automation\logs\2026-09-03\health-867b3aa3.log` |
| 07:37:45 | 11:37:45Z | Operator dispatch: `Daily Strategy Scan (am) - operator-recovery-2026-09-03-scan-am-v7`, run 33750722451, `triggering_actor=mslade50`, `head_branch=automation-runtime-2026-09-02.1`. `run-scanner` job 07:38:26 to 07:42:46, success. | `gh run view 33750722451 --json jobs`; `gh api .../runs/33750722451` |
| 07:42:41 | 11:42:41Z | Scanner stdout flushed: `[EMAIL] Email sent successfully`; `moc_orders` cleared. Order_Staging / Overflow written inside the 07:38:26 to 07:42:41 window (the GHA log buffers the scanner's stdout to one flush, so the write cannot be pinned finer than that). | run 33750722451 log (scratchpad `recovery_33750722451.log`) |
| 07:42:43 | 11:42:43Z | `exposure_state.json` uploaded to R2 (`computed_at 2026-09-03T11:42:38Z`, asof 2026-09-02, mult 0.0, Rule 1 raw-21d 61.4 > 50). | run log `[cache_io] uploaded ... exposure_state.json (1,156 bytes)`; audit `r2_exposure_state.json` |
| 07:44:36 | 11:44:36Z | Fallback controller tick, run 33751349896, checks out `automation-runtime-2026-08-31.4` (the controller's own checked-in pin was one tag BEHIND the local runtime's `09-02.1`). | `gh run view 33751349896 --log` (`ref: automation-runtime-2026-08-31.4`) |
| 07:44:46 | 11:44:46Z | Controller claims `scan_am` for GitHub: `running`, `phase=github_reconcile`, token `d7f138ad30`, lease to 09:59:46. | R2 events `...-running-github-2026-09-03-scan_am-d7f138ad30.json` (two writes, :46.41 and :46.77) |
| 07:44:50 | 11:44:50Z | `GitHub fallback: dispatch daily_screener.yml token=2026-09-03-scan_am-d7f138ad30`; run 33751370947 created 07:44:52, `run-scanner` started 07:45:03. | controller log; `gh run view 33751370947 --json jobs` |
| 07:46:45 | 11:46:45Z | Run 33751370947 cancelled (`run-scanner` cancelled, deploy jobs skipped). The API records `actor=github-actions[bot]`; the canceller is not exposed. | `gh api .../runs/33751370947`; `gh run view --json jobs` |
| 07:46:51 | 11:46:51Z | Controller: `ERROR: scan_am GitHub fallback indeterminate: GitHub fallback concluded cancelled`; writes `indeterminate` / `manual_review` receipt; `skip private_site_am` / `skip shared_site_am: unsatisfied dependencies: scan_am`; process exit 1 (workflow conclusion failure). | controller log; R2 event `...-indeterminate-github-...d7f138ad30.json` |
| 07:47:34 | 11:47:34Z | Operator `resolve`: receipt to `status=success`, `source=operator`, `phase=manual_resolved`, detail "Verified v7 recovery run 33750722451 completed successfully; staged 3 liquid and 3 overflow rows, synced 887 signal rows, sent email, and published exposure. Delayed duplicate run 33751370947 was cancelled before any staging/email completion marker." `latest.json` updated 07:47:35. | R2 event `...-success-operator-...d7f138ad30.json`; `latest.json` LastModified |
| 07:49:51 | 11:49:51Z | Commit `5e70c489` "Recover stalled premarket scans safely" (author McKinley) on branch `codex/fix-scan-recovery-v8`, parent `1119efae`, tree `fe9791a6`. | `git log --format='%h %p %t %an %ad'` |
| 07:50:25 | 11:50:25Z | PR #15 opened from that branch; its Tests run 33751886124 fails (pre-existing unicode-guard failure, see s.6 F5). | `gh pr list --head codex/fix-scan-recovery-v8`; `gh run list --workflow tests.yml` |
| 07:52:31 | 11:52:31Z | PR #15 merged with a merge commit: `0c912f58` (author `mslade50`, parents `1119efae` + `5e70c489`, tree `fe9791a6`). Tag `automation-runtime-2026-09-03.1` -> `0c912f58`, same second. Tests run 33752075528 on it fails (same guard). | `git log` on `0c912f58`; `git tag --format='%(creatordate:iso)'`; `gh pr list` `mergedAt 11:52:32Z` |
| 07:55:08 | 11:55:08Z | Runtime v8 prepared at `...-runtime-v8` (`pinned_sha 0c912f58...`, `fallback_ref automation-runtime-2026-09-03.1`, branch `codex/local-primary-runtime-v8`). | `...-runtime-v8\.local\automation-runtime.json` `prepared_at` |
| 08:03:04 | 12:03:04Z | Runtime v8 cutover: seven `New Seasonals Local v8 - *` tasks registered and enabled, v7 set disabled. | `...-runtime-v8\artifacts\automation\cutover-state.json` `cutover_at`; `Get-ScheduledTask` (v7 Disabled, v8 Ready) |

After the window, for impact only: `New Seasonals Local v8 - discretionary` 08:35:35 result 0; `IBKR Event Sleeve Auction Orders` 09:05:05 result 0; `IBKR OLV Pre-Market Exits` 09:10:10 result 0; `IBKR Daily Order Chain` 09:31:31 result 0; v8 execution 16:30 and postclose 17:10 result 0 (`Get-ScheduledTaskInfo`, 2026-09-04 07:27 ET). Manual private-site deploy run 33778072898 at 12:20:55 (`gh run list`). The shared-seasonality site had no AM deploy on 09-03 at all (only the PM one at 18:44:16, run 33814442433).

## 3. Root cause

### Known (evidence in hand)

- The `scan_am` receipt was claimed at 04:14:22.19 with `phase=local_pre_side_effect` and never transitioned again by the local process. The only later events are the controller's `github` writes from 07:44:46 onward. Source: R2 event listing for `2026-09-03/scan_am` (five events total, one local).
- The lease expired at 04:44:22.19 (`lease_expires_at_utc` in the claim event). Lease arithmetic: for a non-rerun-safe job the lease covers the pre-side-effect commands only, `max(300, sum(timeouts) + LEASE_GRACE_SECONDS 900)`; `scan_am`'s single pre-side-effect command is `pull fail-closed scanner inputs` (`scripts/pull_scan_caches.py`, timeout 900), so 900 + 900 = 1800 s = exactly 30 min. Source: `scripts/automation_supervisor.py` (`LEASE_GRACE_SECONDS`, `_local_lease_seconds`, the `scan_am` catalog entry at ~line 393).
- The scanner never started. The side-effect marker (`indeterminate` / `local_side_effect`, "side-effecting step started: run unified scanner") that precedes `daily_scan.py` is absent for 09-03 and present for 09-04 at 04:15:07. Source: R2 events for both dates.
- The immediate GitHub fallback did not fire: there is no `github`-source receipt event and no `daily_screener.yml` run between 04:14 and 07:37 (`gh run list --created 2026-09-03`).
- The only fallback-controller tick in the dead window was 07:44:36 (run 33751349896); the previous one was 02:30:53 (run 33723564770), a 5h14m gap. Source: `gh run list --workflow local_automation_fallback.yml`.
- The Task Scheduler result for the v6 premarket task was 0x1, so the PowerShell wrapper saw a non-zero supervisor exit. Source: `Get-ScheduledTaskInfo`.
- The v7 07:30 health task saw the expired lease and reported FAIL, and nothing in v7 could act on it: the v7 `resolve` accepted only `indeterminate` receipts, and the v7 health branch of `run_local_automation.ps1` ran `health` alone. Sources: `health-867b3aa3.log`; the pre-fix side of `git diff 0c912f58~1 0c912f58`.
- The v6 worktree and its 04:10 log are gone (`ls C:\Users\McKinley Slade\dev\` shows only `...-runtime-v7` and `...-runtime-v8`), so the supervisor's own error line for the crash cannot be read.

### Inferred (stated as inference)

- Why no failure receipt and no dispatch: in the pre-fix code path a pre-side-effect exception goes `self.receipts.transition(local_failure, update_latest=False)` and only then `immediate GitHub fallback`. If that R2 write itself raised (network, auth, R2 5xx, a conditional-write conflict), the exception escaped the failure handler, the dispatch line was never reached, and the supervisor exited non-zero with `latest.json` still showing the original `running` claim. That is exactly the state observed, and it is exactly what the fix guards (try/except around every transition with "automatic fallback suppressed" logging). The fix's shape is the evidence; there is no log line proving it. Alternative not excluded by the evidence: a hard process kill (OOM, host-level termination) between the claim and the failure handler, which would leave the same receipt state; the 0x1 task result argues for a Python-level exit rather than a kill, but only weakly.
- Which pre-side-effect step died: `pull_scan_caches.py` is the only command before the side-effect boundary, so "died in pull_scan_caches" (audit s.3) is the only consistent reading, but it is inferred from the catalog, not from a log.
- Who cancelled run 33751370947 at 07:46:45: presumed McKinley by hand (a duplicate scan was about to rewrite Order_Staging on top of the 07:42 rows); the GitHub API does not expose the cancelling actor.

## 4. Fix shipped (`5e70c489` / `0c912f58`, tag `automation-runtime-2026-09-03.1`)

Diff: 6 files, +265/-11 (`git show --stat`). Source for everything below: `git diff 0c912f58~1 0c912f58` (saved at scratchpad `fix_0c912f58.diff`).

1. **Guarded receipt transitions in the job runner** (`scripts/automation_supervisor.py`): the side-effect-boundary write is wrapped; if it raises, the child is not started and the job returns `failure` with "side-effect boundary not confirmed" (fail closed at the gate, no second writer). The post-side-effect `manual_review` write and the pre-side-effect `retryable` failure write are each wrapped; a failed write logs "automatic fallback suppressed" and returns a `JobOutcome` instead of letting the exception escape. New `guard:` log lines mark "persist side-effect boundary" and "side-effect boundary durable" (both visible in the 09-04 v8 premarket log, lines 284-285).
2. **`resolve` accepts an expired pre-side-effect receipt**: previously "only an indeterminate receipt can be resolved"; now also `status=running` + `phase=local_pre_side_effect` + lease expired. This is the state the operator faced at 07:30 to 07:44 on 09-03 and could not clear until the controller had converted it to `indeterminate`.
3. **`fallback-due --job <id>`**: dispatches one job plus its prerequisite closure (`only_jobs` in `_run_pipeline`; requires a single concrete `--pipeline`).
4. **The 07:30 health task recovers before it reports** (`scripts/run_local_automation.ps1`): the `health` branch now runs `fallback-due --pipeline premarket --job scan_am` first, then `health`; the task exit code is 1 if either fails. Recovery is still GitHub-only (it dispatches `daily_screener.yml`; it does not re-run the pipeline locally).
5. **Controller pin advanced**: `.github/workflows/local_automation_fallback.yml` `AUTOMATION_RUNTIME_REF` 2026-08-31.4 -> 2026-09-03.1 (with the matching assertion in `tests/test_local_automation_workflows.py`). Note the 07:44 controller tick had been running the 08-31.4 code while the local runtime was on 09-02.1.
6. Tests: `tests/test_automation_supervisor.py` +144 lines, `tests/test_local_automation_powershell.py` +4.

**Why two byte-identical commits.** `5e70c489` (07:49:51, author McKinley) is the branch commit on `codex/fix-scan-recovery-v8`, parent `1119efae` (then `origin/main`). PR #15 was merged at 07:52:32 with GitHub's "merge commit" button rather than a fast-forward, producing `0c912f58` (author `mslade50`, parents `1119efae` and `5e70c489`). Because the branch was a single commit directly on top of the base, the merge commit's tree is the branch commit's tree (`fe9791a6` for both; `git diff 0c912f58 5e70c489` is empty). The tag and the v8 pin point at the merge commit `0c912f58`. Both then reached the local `main` through `95929d52` (12:20:37, "Merge remote-tracking branch 'origin/main'"), whose other parent `43521a51` sits on the separate local lineage (`a0a079f1`, `76a82f55`, `03875aac`, ...) that `1119efae` is not an ancestor of. Consequence recorded for F3: `03875aac` (harvest_fills) is NOT an ancestor of `0c912f58` (`git merge-base --is-ancestor` says no), so the pinned runtime v8 has no `harvest_fills` job.

## 5. Verification (2026-09-04)

- **04:10 v8 premarket, end to end, all local**: `New Seasonals Local v8 - premarket` LastRun 9/4/2026 04:10:10, LastResult 0 (`Get-ScheduledTaskInfo`). `status --date 2026-09-04` (Appendix A2): `cboe_am`, `master_prices_am`, `risk_am`, `event_sleeve_am`, `scan_am` all `success local`; `private_site_am`, `shared_site_am` `success github`. Receipts: `scan_am` claimed 04:14:21.39, side-effect marker durable 04:15:07.41 ("side-effecting step started: run unified scanner"), success 04:19:32.54 (R2 events `2026-09-04/scan_am`). Runtime log `...-runtime-v8\artifacts\automation\logs\2026-09-04\premarket-6bc339f8.log` (8,552 lines): `guard: persist side-effect boundary for scan_am` / `guard: side-effect boundary durable for scan_am` (lines 284-285), `success scan_am (local)` (line 8474), then the two site dispatches: `deploy_site.yml` token `3e54f750e1` = run 33852895307 (04:19:40, success) and `deploy_shared_seasonals.yml` token `af0aa2bc60` = run 33853789084 (04:30:42, success).
- **07:30 v8 health task fired and passed**: `New Seasonals Local v8 - health` was `Running` at 07:30:09 with LastRun 9/4/2026 07:30:30, and `Ready` with LastResult 0 at 07:31:18 (`Get-ScheduledTaskInfo`, two reads). Its recovery pre-step wrote `premarket-ee07fca8.log`: `skip master_prices_am ... skip risk_am ... skip scan_am: success receipt from local token=2026-09-04-scan_am-d5de1b06e4` (the `--job scan_am` closure is scan_am + master_prices_am + risk_am; nothing dispatched). Its health battery wrote `health-cd00ce09.log`: `0 FAIL, 2 WARN, 38 OK` (the two WARNs: no local runtime log yet for `indicator` and `weekly-rundown`, which have not had a v8 run). This is the first time the v8 health/recovery path has run; it exercised the no-op branch only, not an actual dispatch.
- **Fallback controller ticks since the cutover**: 09-03 at 11:14, 12:41, 14:53, 17:54 and 20:27 ET, 09-04 at 01:08 and 05:58 ET, all success (`gh run list --workflow local_automation_fallback.yml`; runs 33771481133 through 33860938428).
- **CI**: `tests.yml` green on `5dcde30e` at 07:26:13 ET 09-04 (run 33867935090, success) after nine consecutive failures from `f7fdc83b` (09-02 17:24 ET) through `85d43f00` (09-04 07:21 ET).

## 6. Still open (mapped to the ops audit's F1-F13)

- **F1 (HIGH) no GitHub-independent recovery**: unchanged in substance; v8's 07:30 `fallback-due` dispatches to GitHub only and ran a no-op today. D11's 05:30 S4U `run-pipeline --pipeline premarket` task is not built.
- **F2 (HIGH) 09-02 `execution_report` indeterminate**: still `indeterminate local` in `status --date 2026-09-02` today (Appendix A3); the R2 events show the side-effect marker written at 16:30:02.59 and "send execution report exited 1" at 16:30:04.61 ET on 09-02, 2 s after the marker. Whether that email went out is unverified.
- **F3 (HIGH) `harvest_fills` absent from the pinned runtime**: verified, `03875aac` is not an ancestor of `0c912f58`; `harvest_fills` reads `missing` for 09-02, 09-03 and 09-04 in `status`. This is the cutover D1 allows.
- **F4 (HIGH) logs destroyed per cutover; Task Scheduler Operational log disabled**: the v6 worktree and its 04:10 log are gone, which is why section 3 has an inferred cause. Partial correction to the audit: the v7 worktree survived the v8 cutover (directory present, dated 09-03 07:30, holding the 07:30 health log used above). `wevtutil gl Microsoft-Windows-TaskScheduler/Operational` reports `enabled: false` as of 2026-09-04.
- **F5 (MED) CI red on 09-02/03**: fixed 09-04 by the mind; `gh run list --workflow tests.yml --limit 3` shows the newest run (33867935090, `5dcde30e`, 07:26 ET) green after two failures earlier that morning.
- **F6 (MED) runtime churn**: five tags in four days (`08-31.1/.2/.3/.4`, `09-01.1`, `09-02.1`, `09-03.1`); two cutovers on the incident morning (06:56 v7, 08:03 v8), the second inside the 04:00-09:35 window. No cutover since. Section 7 is the rule.
- **F7 (MED) controller cadence**: the 09-03 ticks landed at 21:30 (09-02), 02:30, 07:44, 11:14, 12:41, 14:53, 17:54 ET; the stall fell in the 5h14m gap. Unchanged.
- **F8 (MED) CBOE GitHub backup untested since its fix**: `4ec5f78f` (08-31 08:05) has not been exercised by a real dispatch since; unchanged.
- **F9 (MED) interactive-logon dependency**: unchanged; every broker-side task remains Interactive/Limited (audit s.2, s.7).
- **F10 (MED) AM site deploys after operator resolution**: demonstrated on 09-03: `private_site_am` / `shared_site_am` still `missing` for 2026-09-03; the private site was covered by hand at 12:20:55, the shared site never got an AM deploy that day.
- **F11 (LOW) Option Surface / RadarPackExport**: `IBKR Option Surface` LastRun 9/3 18:00 LastResult 1; `RadarPackExport` LastRun 8/29 10:00 LastResult 1 (`Get-ScheduledTaskInfo` today). `risk_dashboard_signal_state.json` frozen 2026-05-07 (audit s.6).
- **F12 (LOW) `ledger_git_sha` unknown**: still `unknown` in `data/backtest_trades_full.parquet` built `gha:33852895307` at 04:24:57 ET 09-04 (`ledger_rows 4701`).
- **F13 (LOW) hygiene**: 94 worktrees (`git worktree list`), 409 `git status --porcelain` lines of which 392 untracked, measured 09-04; the duplicate-commit pattern in section 4 is a symptom of the same thing.

## 7. Rules adopted (plan D1 and D11)

1. One runtime cutover per day, at most.
2. Never cut over between 04:00 and 09:35 ET (the premarket-to-order-chain window). On 09-03 both cutovers (06:56 and 08:03) broke this rule, and the 06:56 one happened while `scan_am` was already dead.
3. Never cut over before the incident write-up for the previous failure exists in `docs/`. This document is that write-up for 2026-09-03; the next cutover is the `harvest_fills` one (D1) and it happens on a day with no other change.
4. (D11) The cadence rule is to be written into `docs/local_automation_task_scheduler.md`; that edit is not part of this document.

## Appendix A. `automation_supervisor.py status` output, dev checkout, 2026-09-04 ~07:28 ET (read-only)

### A1. `--date 2026-09-03`

```
premarket          cboe_am                      success  local
premarket          master_prices_am             success  local
premarket          risk_am                      success  local
premarket          event_sleeve_am              success  local
premarket          scan_am                      success  operator
premarket          private_site_am              missing  -
premarket          shared_site_am               missing  -
discretionary      discretionary_focus          success  local
execution          execution_report             success  local
postclose          master_prices_pm             success  local
postclose          risk_pm                      success  local
postclose          verify_fills                 success  local
postclose          harvest_fills                missing  -
postclose          earnings_and_grades          success  local
postclose          portfolio_report             success  local
postclose          cboe_pm                      success  local
postclose          trend_sleeve                 success  local
postclose          intraday_prices              success  local
postclose          scan_pm                      success  local
postclose          macro_releases               success  local
postclose          private_site_pm              success  github
postclose          shared_site_pm               success  github
indicator          indicator_cache              missing  -
weekly-rundown     weekly_rundown               missing  -
```

### A2. `--date 2026-09-04`

```
premarket          cboe_am                      success  local
premarket          master_prices_am             success  local
premarket          risk_am                      success  local
premarket          event_sleeve_am              success  local
premarket          scan_am                      success  local
premarket          private_site_am              success  github
premarket          shared_site_am               success  github
discretionary      discretionary_focus          missing  -
execution          execution_report             missing  -
postclose          master_prices_pm             missing  -
postclose          risk_pm                      missing  -
postclose          verify_fills                 missing  -
postclose          harvest_fills                missing  -
postclose          earnings_and_grades          missing  -
postclose          portfolio_report             missing  -
postclose          cboe_pm                      missing  -
postclose          trend_sleeve                 missing  -
postclose          intraday_prices              missing  -
postclose          scan_pm                      missing  -
postclose          macro_releases               missing  -
postclose          private_site_pm              missing  -
postclose          shared_site_pm               missing  -
indicator          indicator_cache              missing  -
weekly-rundown     weekly_rundown               missing  -
```

(Read at 07:28 ET; the 08:35 / 16:30 / 17:10 pipelines had not yet run.)

### A3. `--date 2026-09-02`, the one line that differs from a clean day

```
execution          execution_report             indeterminate local
```

(All ten postclose producer jobs `success local`, both PM sites `success github`, `harvest_fills missing`, premarket all success.)

## Appendix B. `scan_am` receipt events for 2026-09-03 (R2, `automation/receipts/v1/2026-09-03/scan_am/`)

| Event key (UTC) | status | source | phase | token | lease_expires_at_utc |
|---|---|---|---|---|---|
| `events/2026-09-03T081422.193402_0000-running-local-...ca1f8cf470.json` | running | local | local_pre_side_effect | ca1f8cf470 | 08:44:22.19Z |
| `events/2026-09-03T114446.413983_0000-running-github-...d7f138ad30.json` | running | github | github_reconcile | d7f138ad30 | 13:59:46.41Z |
| `events/2026-09-03T114446.766532_0000-running-github-...d7f138ad30.json` | running | github | github_reconcile | d7f138ad30 | 13:59:46.77Z |
| `events/2026-09-03T114651.855394_0000-indeterminate-github-...d7f138ad30.json` | indeterminate | github | manual_review | d7f138ad30 | null |
| `events/2026-09-03T114734.674898_0000-success-operator-...d7f138ad30.json` | success | operator | manual_resolved | d7f138ad30 | null |
| `latest.json` (LastModified 11:47:35.74Z) | success | operator | manual_resolved | d7f138ad30 | null |

No `failure` event and no local event after the claim.
