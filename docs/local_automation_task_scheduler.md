# Local-primary Task Scheduler install

The scheduler is installed in deliberate phases. Production code runs from a
dedicated branch worktree pinned to a tested `origin/main` SHA and from a
dedicated virtual environment. The normal development workspace is used only
as the configuration root, so its `.env` and external credential references
remain machine-local; no secret is copied into the runtime worktree or task
definition.

## Current generation

Do not look for the live generation number in this document; it is stale the
day after every cutover. The authoritative record is the marker file of the
runtime whose tasks are enabled:

```powershell
Get-ScheduledTask | Where-Object { $_.TaskName -like 'New Seasonals Local*' -and $_.State -ne 'Disabled' } |
  Select-Object TaskName, State
# then, for that generation's runtime root (the -RuntimeRoot in the task action):
Get-Content "$runtime\.local\automation-runtime.json"   # pinned_sha, fallback_ref, config_root, prepared_at
Get-Content "$runtime\artifacts\automation\cutover-state.json"   # cutover_at, task states at cutover
```

`fallback_ref` in that marker is the immutable tag the GitHub controller and
every child dispatch resolve; `pinned_sha` is the commit the tasks execute.
The installer's `-Phase Status` prints the enabled/next-run state of the
prefix you name plus the stable state root and how many other generations
are still registered.

## Cutover cadence

Adopted 2026-09-04 (plan D1/D11, incident
`docs/incidents/2026-09-03_scan_am_stall.md` section 7). Two failed
recoveries in six days came with five cutovers in a week, two of them on the
incident morning, one of them while `scan_am` was already dead.

1. **One runtime cutover per day, at most.** Prepare and RegisterDisabled may
   happen earlier; the enable/disable flip happens once.
2. **Never between 04:00 and 09:35 ET.** That is the premarket-to-order-chain
   window (04:10 premarket, 05:45 retry, 07:30 health, 09:05 event auction,
   09:10 OLV exits, 09:31 order chain). A cutover inside it can disable the
   writer of the morning's staging rows.
3. **Never before the incident write-up for the previous failure exists in
   `docs/incidents/`.** A cutover is not a way to make a red morning go away.
4. **Always outside a trading window**: not while any `New Seasonals Local*`
   task, the order chain, or the broker-side tasks are running (the installer
   refuses a queued/running task; it cannot see the broker tasks, so look).
   The clean slots are roughly 10:00-16:00 ET, or after 18:30 ET once
   `postclose` has finished.
5. Copy the outgoing generation's logs forward (Cutover does this
   automatically since 2026-09-04) before deleting its worktree by hand.

Run these commands from an elevated **Windows PowerShell 5.1** prompt after the
scheduler change is merged and verified on `origin/main`. `<N>` is the next
generation number (read the current one from the marker, see above):

```powershell
$repo = 'C:\Users\McKinley Slade\dev\New_Seasonals'
$runtime = 'C:\Users\McKinley Slade\dev\New_Seasonals-automation-runtime-v<N>'
$fallbackTag = 'automation-runtime-<YYYY-MM-DD>.<k>'
$runtimeBranch = 'codex/local-primary-runtime-v<N>'
$taskPrefix = 'New Seasonals Local v<N> - '
$retirePrefix = 'New Seasonals Local v<N-1> - '
git -C $repo fetch origin main
$sha = git -C $repo rev-parse origin/main

& "$repo\scripts\install_local_automation_tasks.ps1" -Phase Prepare -SourceRepository $repo -RuntimeRoot $runtime -ConfigRoot $repo -PinnedSha $sha -FallbackRef $fallbackTag -RuntimeBranch $runtimeBranch -TaskNamePrefix $taskPrefix -RetireTaskNamePrefix $retirePrefix
& "$repo\scripts\install_local_automation_tasks.ps1" -Phase RegisterDisabled -SourceRepository $repo -RuntimeRoot $runtime -ConfigRoot $repo -TaskNamePrefix $taskPrefix -RetireTaskNamePrefix $retirePrefix
& "$repo\scripts\install_local_automation_tasks.ps1" -Phase Status -SourceRepository $repo -RuntimeRoot $runtime -ConfigRoot $repo -TaskNamePrefix $taskPrefix -RetireTaskNamePrefix $retirePrefix
```

The fallback tag must already exist on `origin` and resolve to exactly the
tested `origin/main` SHA. The pinned supervisor dispatches backup workflows at
that immutable tag, so a later workflow-input edit on `main` cannot silently
break an older operational runtime.

`RegisterDisabled` creates the following eight disabled tasks, all evaluated on
the machine's Eastern local clock (add `-WhatIf` to list what it would
register without elevation or any change):

| Pipeline | Schedule | Notes |
|---|---|---|
| `premarket` | weekdays 04:10 | local primary window 04:10-05:20 |
| `premarket-retry` | weekdays 05:45 | `run-pipeline --pipeline premarket --retry`; window 05:45-07:00; needs nothing from GitHub |
| `discretionary` | weekdays 08:35 | |
| `execution` | weekdays 16:30 | |
| `postclose` | weekdays 17:10 | |
| `indicator` | Monday 03:00 | |
| `weekly-rundown` | Sunday 08:00 | |
| `health` | weekdays 07:30 | runs `fallback-due --pipeline premarket --job scan_am` (GitHub) then the health battery |

`premarket-retry` (2026-09-04, after the 2026-09-03 stall) re-walks the same
R2 receipts the 04:10 run wrote, through the state table in the
`automation_supervisor.py` module docstring: every job with a `success`
receipt is a no-op (nothing runs, nothing is written); a job whose local
`running` lease expired before its side-effecting step (the 09-03 shape,
`local_pre_side_effect`) is re-run with a new token; an `indeterminate`
receipt is never re-run. Its exit code counts only what it did itself: a
receipt that was already indeterminate, or a job blocked only behind another
writer's still-live lease, is printed and does not turn the task red. If the
04:10 run still holds the supervisor lock, the retry prints `No action` and
exits 0. Its window overlaps the GitHub controller's on purpose; the receipt
CAS arbitrates, and whichever side wakes first on an expired lease takes it.

It fires at 05:45 rather than 05:30 because `master_prices_am` claims a
70-minute lease: a 04:10 claim is still live at 05:30, so a 05:30 retry would
skip exactly the job most likely to have stalled and then find every
downstream job blocked.

**05:45 does not guarantee a clear board.** Jobs claim when their predecessor
finishes, not at 04:10, so a lease claimed later is still live at 05:45:
`cboe_am` (90 min) claimed at 04:15 or later, `risk_am` (75 min) at 04:30 or
later, `master_prices_am` (70 min) at 04:35 or later. The retry skips those
and says so. What bounds the damage is the lease itself plus the probes after
it: an ordinary premarket run cannot hold a lease past roughly 06:55 (the
primary's launch window closes at 05:20 and the longest local lease is
`cboe_am`'s 90 minutes), and anything still `running` on an expired lease is a
FAIL in the 07:30 health battery and reclaimable by the next controller tick.
A badly delayed chain can push a late job's lease past 06:55; that is the case
the 07:30 FAIL exists for, not one the retry can fix. Evidence:
`artifacts/verify_2026-09-04/ops_supervisor/round2/attack_r2_edge_detail.json`
`retry_lease_coverage`. The retry window still ends at 07:00 so a
`StartWhenAvailable` replay cannot start a premarket pipeline at lunchtime.

The 07:30 `health` task runs `fallback-due` first and the battery second. If
the 04:10 primary is hung and still holds the supervisor lock, both print a
`FAIL ... primary still holds the supervisor lock since <time ET>` line
straight to stdout and exit 1; the battery still runs (every check in it is
read-only), and it still runs when the run log itself cannot be opened, in
which case the log lines go to stdout and a `WARN health: cannot open the run
log` precedes them. The runner reaches the battery because `fallback-due`
returns an exit code rather than raising; its `try`/`catch` around that call
covers only a *launch* failure (a missing interpreter, say) -- PowerShell's
`&` on a native executable does not throw on a non-zero exit, so the
non-zero-exit path was always going to continue and the guard is for the
other one.

Each task wakes the machine, retries up to three times at five-minute intervals,
and ignores a new trigger while the prior copy is still running. A missed
trigger starts locally only while it is still inside that pipeline's bounded
primary window; late/day-mismatched replays exit without side effects and leave
recovery to the receipt-guarded GitHub backup. Tasks use S4U, so no Windows
password is stored. The runner
hydrates `GH_TOKEN` in process from the existing user-scoped
`GH_PAT_NEW_SEASONALS` only when no ambient `GH_TOKEN` is present; it never
prints or persists either value. The runner validates the pinned SHA and refuses tracked code
changes before it calls `automation_supervisor.py`. Scheduled runs never fetch,
merge, reset, check out, or upgrade code.

The operational worktree marks only the explicit generated-state allowlist
(risk, CBOE, exposure, and analyst-grade snapshots) as `skip-worktree`.
Those files are runtime data published to R2; all code and reference inputs
remain covered by the clean pinned-SHA guard.

## Runtime logs and the stable state root

Since 2026-09-04 the runner passes
`--state-root <ConfigRoot>\artifacts\automation` (the development checkout,
gitignored) and exports the same path as `NEW_SEASONALS_AUTOMATION_STATE_ROOT`
for the health battery. Runtime logs
(`logs\<ET date>\<pipeline>-<stub>.log`, `health-<stub>.log`), the supervisor
lock, and the health receipt cache therefore survive the deletion of a
retired runtime worktree, and **from v9 onward** one lock covers every
generation. The layout is unchanged; only the root moved. Generations before
v9 wrote under `<RuntimeRoot>\artifacts\automation` and locked there, so a v8
run and a v9 run hold *different* lock files and do not exclude each other:
during a cutover overlap the only guards are the installer's idle check and
the 04:00-09:35 ET rule above. The Cutover phase writes the incoming
generation's `cutover-state.json`, then copies logs forward (never
overwriting) from every `-RetireTaskNamePrefix` task's `-RuntimeRoot` and from
the new runtime's own tree. For each of those sources it mirrors
`artifacts\automation\cutover-state.json` into `<state root>\cutovers\`
**before** it checks whether that source has a logs directory at all -- the
incoming generation is a freshly prepared worktree that has none (it writes
its logs to the stable state root), so a mirror behind that check never ran
for it and the enabled generation's own record was missing for its whole life.
A retired task whose registered action carries no quoted `-RuntimeRoot` cannot
be located; that is printed, not silently dropped. The health battery WARNs
`triggers:<pipeline>: no local runtime log found` when a pipeline has no log
under that root.

> **Hand-run supervisor commands from the development checkout take the
> production lock.** `--state-root` defaults to `<repo>\artifacts\automation`,
> which since v9 *is* the production state root, so an ad hoc
> `run-pipeline`/`health` from `C:\Users\McKinley Slade\dev\New_Seasonals`
> blocks (or is blocked by) the scheduled tasks and writes into their log
> tree. Read-only commands (`plan`, `status`, `--dry-run`) never take the
> lock. For anything else, pass an explicit throwaway root, e.g.
> `--state-root "$env:TEMP\supervisor-scratch"`.

To turn on the Task Scheduler Operational log (it was disabled during the
2026-09-03 incident, so the 04:10 task's own history was invisible), from an
elevated shell:

```powershell
wevtutil sl Microsoft-Windows-TaskScheduler/Operational /e:true
```

Only the explicit cutover phase enables the eight new local tasks and disables
the prior local prefix, the legacy GitHub-dispatch tasks, and the superseded
`Repo Health Check` entry. The new pinned `health` task replaces that malformed
legacy action. The installer never deletes or overwrites a Task Scheduler
entry, except through the explicit prune switch described below:

```powershell
& "$repo\scripts\install_local_automation_tasks.ps1" -Phase Cutover -SourceRepository $repo -RuntimeRoot $runtime -ConfigRoot $repo -TaskNamePrefix $taskPrefix -RetireTaskNamePrefix $retirePrefix -ConfirmCutover
& "$repo\scripts\install_local_automation_tasks.ps1" -Phase Status -SourceRepository $repo -RuntimeRoot $runtime -ConfigRoot $repo -TaskNamePrefix $taskPrefix -RetireTaskNamePrefix $retirePrefix
```

Run cutover outside a scheduled window. Its pre-change legacy enabled states
are recorded under `artifacts/automation/cutover-state.json` in the runtime
worktree. If any enable/disable operation fails, the installer automatically
restores both task sets to their pre-cutover enabled states and fails loud.
Code upgrades are also explicit. Do not mutate a pinned runtime under an active
task set. Prepare and register a separate versioned runtime/prefix, then pass
the prior prefix through `-RetireTaskNamePrefix`; cutover snapshots and rolls
back both local task sets before either can be left in a partial writer state.

### Pruning superseded generations

Disabled generations accumulate (35 disabled `New Seasonals Local*` tasks
across v1/v4/v5/v6/v7 on 2026-09-04). `-PruneSuperseded` is OFF by default and
unregisters only tasks that are named like a local-primary generation
(`New Seasonals Local - x` or `New Seasonals Local vN - x`), are not the
current `-TaskNamePrefix`, are not the `-RetireTaskNamePrefix` rollback set,
are disabled, and are idle. An enabled task is never pruned; disable it
through a cutover first. `-WhatIf` lists without deleting and needs no
elevation:

```powershell
& "$repo\scripts\install_local_automation_tasks.ps1" -Phase Prune -SourceRepository $repo -RuntimeRoot $runtime -ConfigRoot $repo -TaskNamePrefix $taskPrefix -RetireTaskNamePrefix $retirePrefix -PruneSuperseded -WhatIf
& "$repo\scripts\install_local_automation_tasks.ps1" -Phase Prune -SourceRepository $repo -RuntimeRoot $runtime -ConfigRoot $repo -TaskNamePrefix $taskPrefix -RetireTaskNamePrefix $retirePrefix -PruneSuperseded
```

`-Phase Cutover ... -PruneSuperseded` prunes the same set immediately after a
fully successful cutover (the just-retired prefix stays for rollback).

## Receipt and fallback contract

Every component writes a latest receipt plus append-only events under
`automation/receipts/v1/<YYYY-MM-DD-ET>/<job>/` in R2. `success` is terminal;
`failure` may be retried; a live `running` lease blocks overlap. Before a
non-rerun-safe component touches Sheets, SMTP, or another external system, its
latest receipt becomes `indeterminate`. Any crash or ambiguous result after
that point suppresses both immediate and hourly GitHub fallback until an
operator verifies the external state and runs:

```powershell
& "$runtime\.venv\Scripts\python.exe" "$runtime\scripts\automation_supervisor.py" resolve `
  --pipeline postclose --job portfolio_report --date 2026-08-28 `
  --disposition success --reason "verified email and Portfolio sheet readback" `
  --config-root $repo
```

Use `retryable_failure` only after confirming that the side effect did not
occur. `resolve` also accepts a `running` receipt whose lease expired before
the side-effecting step. After a `success` resolution it prints the
dependents that are now due (on 2026-09-03 the AM site deploys stayed
`missing` all day after `scan_am` was resolved) and the exact `fallback-due
--job` command for each; it never dispatches. An explicit `--ref` is echoed
verbatim into those commands, `--ref main` included (with a WARNING, because
`main` is a moving branch and every scheduled dispatch resolves the pinned
immutable tag). With no `--ref` at all they carry `--ref <fallback_ref>` read
from the runtime marker `<RuntimeRoot>\.local\automation-runtime.json` --
`--repo-root` first, which under a scheduled task IS the pinned runtime, then
`--config-root` -- and fall back to `main` plus the WARNING when neither holds
a readable marker. `status` shows
such a receipt as `expired` (the raw field stays `running`; `--json` carries
both `status` and `receipt_status`).

The sole migrated-job GitHub cron is
`.github/workflows/local_automation_fallback.yml`; it consults the same R2
receipts and dispatches only missing/retryable jobs during bounded ET windows.
Its ticks are `47 * * * *` plus `7,27 9-13 * * 1-5` (about 20-minute spacing
across the premarket window in both DST regimes, added 2026-09-04 because
GitHub sheds most scheduled ticks) and the 08:50 ET discretionary probe. Six
of the added ticks land outside the premarket window and exit "No pipeline is
inside its ET fallback window"; one of those, the EDT `13:07Z` tick at 09:07
ET, lands inside the 08:50-09:20 ET discretionary window under the `general`
concurrency group rather than the separate `discretionary` one. That is an
extra chance at the discretionary backup, not a second writer, but it is the
tick to remove first if that window ever needs an isolated controller. A
controller tick exits non-zero only for outcomes it produced itself; a receipt
that was already indeterminate when the tick started, or a job blocked only
behind another writer's live lease, is reported in its log and does not fail
the run. A held local supervisor lock makes it print a `FAIL fallback-due:`
line and exit 1 rather than raise. Child workflows have no independent
schedules.

Private and shared site production builds remain GitHub/Cloudflare-only. The
local pipelines publish canonical inputs to R2 and dispatch the cloud build;
they never use local `data/` or `dist/` as production evidence.
