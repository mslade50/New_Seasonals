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
git -C $repo fetch origin main
$sha = git -C $repo rev-parse origin/main

& "$repo\scripts\install_local_automation_tasks.ps1" -Phase Prepare -SourceRepository $repo -PinnedSha $sha
& "$repo\scripts\install_local_automation_tasks.ps1" -Phase RegisterDisabled -SourceRepository $repo
& "$repo\scripts\install_local_automation_tasks.ps1" -Phase Status -SourceRepository $repo
```

`RegisterDisabled` creates the following six disabled tasks, all evaluated on
the machine's Eastern local clock:

| Pipeline | Schedule |
|---|---|
| `premarket` | weekdays 04:10 |
| `discretionary` | weekdays 08:35 |
| `execution` | weekdays 16:30 |
| `postclose` | weekdays 17:10 |
| `indicator` | Monday 03:00 |
| `weekly-rundown` | Sunday 08:00 |

Each task wakes the machine, starts after a missed trigger, retries up to three
times at five-minute intervals, and ignores a new trigger while the prior copy
is still running. Tasks use S4U, so no Windows password is stored. The runner
hydrates `GH_TOKEN` in process from the existing user-scoped
`GH_PAT_NEW_SEASONALS` only when no ambient `GH_TOKEN` is present; it never
prints or persists either value. The runner validates the pinned SHA and refuses tracked code
changes before it calls `automation_supervisor.py`. Scheduled runs never fetch,
merge, reset, check out, or upgrade code.

Only the explicit cutover phase enables the six local tasks and disables the
four legacy local tasks that merely dispatched GitHub workflows. It never
deletes or overwrites a Task Scheduler entry:

```powershell
& "$repo\scripts\install_local_automation_tasks.ps1" -Phase Cutover -SourceRepository $repo -ConfirmCutover
& "$repo\scripts\install_local_automation_tasks.ps1" -Phase Status -SourceRepository $repo
```

Run cutover outside a scheduled window. Its pre-change legacy enabled states
are recorded under `artifacts/automation/cutover-state.json` in the runtime
worktree for operator-led rollback. Code upgrades are also explicit: validate a
new `origin/main` SHA, prepare a new runtime worktree/branch, register a new
disabled task set, exercise it, and only then cut over. Do not mutate the pinned
runtime under an active task set.
