# Local-primary Task Scheduler install

The scheduler is installed in deliberate phases. Production code runs from a
dedicated branch worktree pinned to a tested `origin/main` SHA and from a
dedicated virtual environment. The normal development workspace is used only
as the configuration root, so its `.env` and external credential references
remain machine-local; no secret is copied into the runtime worktree or task
definition.

Run these commands from an elevated **Windows PowerShell 5.1** prompt after the
scheduler change is merged and verified on `origin/main`:

```powershell
$repo = 'C:\Users\McKinley Slade\dev\New_Seasonals'
$runtime = 'C:\Users\McKinley Slade\dev\New_Seasonals-automation-runtime'
$fallbackTag = 'automation-runtime-2026-08-28.2'
git -C $repo fetch origin main
$sha = git -C $repo rev-parse origin/main

& "$repo\scripts\install_local_automation_tasks.ps1" -Phase Prepare -SourceRepository $repo -PinnedSha $sha -FallbackRef $fallbackTag
& "$repo\scripts\install_local_automation_tasks.ps1" -Phase RegisterDisabled -SourceRepository $repo
& "$repo\scripts\install_local_automation_tasks.ps1" -Phase Status -SourceRepository $repo
```

The fallback tag must already exist on `origin` and resolve to exactly the
tested `origin/main` SHA. The pinned supervisor dispatches backup workflows at
that immutable tag, so a later workflow-input edit on `main` cannot silently
break an older operational runtime.

`RegisterDisabled` creates the following seven disabled tasks, all evaluated on
the machine's Eastern local clock:

| Pipeline | Schedule |
|---|---|
| `premarket` | weekdays 04:10 |
| `discretionary` | weekdays 08:35 |
| `execution` | weekdays 16:30 |
| `postclose` | weekdays 17:10 |
| `indicator` | Monday 03:00 |
| `weekly-rundown` | Sunday 08:00 |
| `health` | weekdays 07:30 |

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

Only the explicit cutover phase enables the seven local tasks and disables the
four legacy GitHub-dispatch tasks plus the superseded `Repo Health Check`
entry. The new pinned `health` task replaces that malformed legacy action. The
installer never deletes or overwrites a Task Scheduler entry:

```powershell
& "$repo\scripts\install_local_automation_tasks.ps1" -Phase Cutover -SourceRepository $repo -ConfirmCutover
& "$repo\scripts\install_local_automation_tasks.ps1" -Phase Status -SourceRepository $repo
```

Run cutover outside a scheduled window. Its pre-change legacy enabled states
are recorded under `artifacts/automation/cutover-state.json` in the runtime
worktree. If any enable/disable operation fails, the installer automatically
restores both task sets to their pre-cutover enabled states and fails loud.
Code upgrades are also explicit. Do not mutate the pinned runtime under an
active task set. This installer intentionally does not implement an in-place
upgrade or disable a previously installed local-primary prefix; a future
upgrade must use a separately reviewed handoff that snapshots and rolls back
both local task sets before either can become a writer.

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
occur. The sole migrated-job GitHub cron is
`.github/workflows/local_automation_fallback.yml`; it consults the same R2
receipts and dispatches only missing/retryable jobs during bounded ET windows.
Child workflows have no independent schedules.

Private and shared site production builds remain GitHub/Cloudflare-only. The
local pipelines publish canonical inputs to R2 and dispatch the cloud build;
they never use local `data/` or `dist/` as production evidence.
