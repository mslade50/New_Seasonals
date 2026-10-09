# Packaging checkpoint — stop requested for agent handoff

Stopped on 2026-10-06 at **17:38:07.651994 UTC / 13:38:07.651994 EDT**.
This directory is **generated, not tested and not ready for deployment**. Do not
run `stage --apply`, `activate --apply`, `deactivate --apply` or `rollback --apply`
against production from this checkpoint. No install/activation/worker launch or
broker command was performed. The original combined candidate is unchanged.

## Completed files

- `deployment-manifest.json`: 17 narrowly allowed source/config/pointer payload
  targets, 24 pinned production prerequisites, before/after SHA256 and byte counts.
- `payload/`: combined reviewed source with artifact-only CLI guards excluded;
  original OpenBreakout CLI plus a startup bridge; two default-disabled configs;
  external Legend worker plus explicit repository-root pointer.
- `installer.py`: implemented read-only preflight/plans, explicit `--apply`
  stage/activate/deactivate/rollback, fixed CLI roots, allowlist/path/base-hash
  checks, per-file atomic writes/readback, backup verification, transactional
  exception rollback, install lock, source-drift refusal, preservation of all
  journals/risk/attempts/ledger/flags/Scheduler. Mutation gates conservatively
  refuse other Python process IDs and 07:30–16:15 New York; they do not inspect
  process command lines or stop any process.
- `payload/repo/open_breakout/deployment_bootstrap.py`: startup settings override
  inherited coordination env; live OpenBreakout and Legend share the local DB;
  shadow/replay/status and Legend kill/verify/dry-run keep coordination disabled.
  Activation pins runtime hashes and starts no worker. Native close qualification
  remains None. Activation uses explicit operator confirmation of fresh flat
  exact-owned books; this is an operator assertion, not broker proof.
- `test_deployment.py`: written but **not run**. Covers baseline/path/allowlist
  refusals, active process/time gate, default-disabled full manifest/rollback,
  backup/source drift, injected write failure, activation/session/manual gates,
  ledger/journal preservation, disabled/active bootstrap, import probing,
  isolated original Legend BAT with inert worker substitution, and isolated
  PowerShell Start-Process child/environment contract for OB launchers.
- `validate_deployment.py`: written but **not run**. Native-path-denied staged
  imports, PowerShell AST parsing, pytest/report orchestration, production hash
  comparison. It never executes the original production launchers.
- `combined-baseline.json`, `combined-offline-verification.json`,
  `combined-source-review.patch`: provenance from the previously completed
  **423 passed / 1 skipped** combined candidate; that result does not qualify
  this newly generated deployment packaging.
- `checkpoint.json`: exact last lightweight verification and production hashes.

Builder/template sources live one directory up: `build_deployment.py`,
`deployment_bootstrap.py`, `installer.py`, `test_deployment.py`,
`validate_deployment.py`. Builder generated this package and never wrote to the
repository or OneDrive. Continue edits in these templates and rebuild, or keep
them synchronized with package files to avoid overwriting later fixes.

## Checks actually performed

1. Builder verified original production OB baselines and parsed payload Python.
2. At checkpoint, all Python files in this package passed AST parsing.
3. Read-only `installer.Deployment().report()` passed: all 24 production
   prerequisites and every expected target hash/absence matched.
4. At **17:38:07.651994 UTC**, every entry from the original combined
   `production_base_hashes` and `legend_production_baseline` matched production.
   No production source/config/launcher/helper was changed.

## Unfinished / review before any deployable claim

- Run and fix the deployment verifier/tests; no test or isolated mock launch has
  run. Confirm the PowerShell Parser invocation and Start-Process argument quoting.
- Fully validate active as well as disabled external Legend startup and actual
  bootstrap/import paths using inert substitutes. Tests presently exercise active
  shared-env bootstrap separately; BAT mock-launch is presently disabled only.
- Re-run the aggregate 423-case suite against the actual deployment payload in an
  isolated fixture with native paths blocked. Current provenance refers to the
  older bundle, before the startup bridge additions.
- Review mutation/backup/receipt/settings validation, partial transaction recovery,
  and runtime/settings tamper behavior; add missing failure cases as appropriate.
  Python process-name gate is intentionally conservative; do not weaken it using
  the previously denied process-command-line inspection.
- Review project hygiene requirements for a user-applied source promotion; the
  installer does not currently invoke workspace_hygiene.py.
- Prepare final exact user commands and reconciliation attestation wording, a
  README, complete package file integrity manifest, zip and checksum. None exist
  yet for this deployment package. Keep stage default disabled and automatic
  STOP-to-MKT closing unavailable; preserve today's latched halt.
- Recheck all recorded production hashes before concluding; another agent may
  now own the remaining work. Do not apply, restart, activate or change Scheduler
  without new explicit authority.

The only presently verified user command is read-only baseline preflight:

```powershell
Set-Location -LiteralPath 'C:\Users\McKinley Slade\Documents\Codex\2026-10-06\task-3\open-breakout-deployment'
& 'C:\Users\McKinley Slade\AppData\Local\Programs\Python\Python310\python.exe' -B installer.py preflight
```

To continue **offline testing only** (not yet verified successful):

```powershell
& 'C:\Users\McKinley Slade\AppData\Local\Programs\Python\Python310\python.exe' -B validate_deployment.py
```
