# Workspace hygiene

This repository contains both durable research evidence and large amounts of
reproducible output. Keep those categories separate so Git status remains a
useful safety signal.

## One worktree per task

Concurrent tasks must not share a branch or working directory. Create a task
worktree with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/new_task_worktree.ps1 -Task execution-scheduler
```

This creates a sibling directory under `New_Seasonals-worktrees/` and a branch
named `codex/execution-scheduler`. Open that directory for the task. The helper
refuses to overwrite an existing directory or branch.

## Protect scheduled runtimes during cleanup

Before proposing worktree or branch removal, inventory both Windows Task
Scheduler actions and active Codex automations under
`$CODEX_HOME/automations/*/automation.toml` (normally `~/.codex/automations/`).
Codex heartbeat prompts can name a runtime even when the task itself belongs to
the main checkout. A merged branch or an unchanged Windows task inventory does
not establish that a worktree is unused.

Keep every referenced runtime and its branch. Mark permanent runtime worktrees
with `git worktree lock --reason` identifying the active automation. Treat a lock
as an active dependency: never unlock it or override it for routine cleanup.
Retirement requires an explicit migration or shutdown, verification of the
replacement schedule, and the owner's deletion approval.

The September 18 cleanup removed the active EP runtime because the protected
set omitted its Codex heartbeat. See
`artifacts/recon_2026-09-17/cleanup_log.md` and the current runtime inventory in
[operations_current.md](operations_current.md).

## Generated output

Put disposable downloads, logs, screenshots, browser profiles, temporary
datasets, and rendered reports under `artifacts/`. The whole directory is
ignored by Git. A script can create a category directory with:

```powershell
python scripts/workspace_hygiene.py artifact-dir browser/execution-qa
```

`scratch/` remains a research evidence area. Python and Markdown files there
are intentionally visible to Git. Common machine outputs under `scratch/` are
ignored, but new research code and notes are not.

Presentation builder source remains tracked under `presentation/`; rendered
HTML, figures, PDFs, PowerPoints, and generated specs are ignored.

## Task preflight and postflight

Record the workspace before starting a task:

```powershell
python scripts/workspace_hygiene.py start --force
```

At the end, name the files or directories the task was expected to change:

```powershell
python scripts/workspace_hygiene.py check `
  --allow site `
  --allow tests/test_execution_site.py
```

The check permits dirtiness that already existed at preflight. It fails when a
new path appears, or an existing dirty path changes, outside the declared
scope. This makes concurrent edits and accidental generated files visible
without deleting or resetting anybody's work.

The baseline is stored at `.local/workspace_hygiene/baseline.json`. Set
`NEW_SEASONALS_HYGIENE_BASELINE` when several independent processes in the
same worktree need separate baselines. Separate worktrees are still preferred.

## Versioned reference data

Files under `fundamental/reference/` and the explicitly tracked files under
`data/` are repository evidence, not runtime caches. Refresh jobs should write
their candidate output under `artifacts/` first. Updating a versioned reference
should be an explicit task whose diff is reviewed and committed.

Never clean a dirty workspace by deleting files or resetting Git history
without first reviewing ownership and obtaining the required approval.
