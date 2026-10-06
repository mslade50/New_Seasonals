Start with C:\Users\McKinley Slade\Documents\Codex\2026-10-06\task-3\open-breakout-deployment\CHECKPOINT.md.

# Finished preparation handoff — October 6, 2026

**Code complete and offline-tested; production not installed, not enabled,
not broker-validated. Automatic close-out remains disabled.**

Continue with this directory's `README.md`, `deployment-manifest.json`,
`package-integrity.json` and `evidence/`. The original stopped-editing checkpoint
is preserved as `CHECKPOINT-ORIGINAL.md`; its remaining work has been completed
in the maintained repository package `deployments/open_breakout_20261006`.
Do not run the historical parent `build_deployment.py` over this finished package.

Verified evidence:

- 432 aggregate offline tests passed against actual staged payload bytes,
  including eight new feed-loss/fill/stop/Legend/recovery integration cases.
  The former skipped live-config test now runs against an isolated staged config.
- 44 installer/startup/launcher/rollback/recovery tests passed. Active and
  disabled external Legend imports, both workers' shared DB and inherited
  launcher environment were exercised with inert fixtures.
- Both original candidates, 41 original bundle files, its zip, 12 review-patch
  targets and recorded production baselines matched. Deployment preflight pins
  28 prerequisites and 17 allowlisted targets; all production source/config/
  launcher/helper hashes remained unchanged during verification.
- Package has SHA256 inventories, exact before/after hashes, verified backups,
  atomic individual writes, exception rollback, interrupted-transaction recovery,
  lock/drift/tamper checks and project hygiene integration. Mutations refuse
  other Python processes and 07:30–16:15 New York; none stop or start workers.
- Public-remote source is sanitized. The account token is rendered only from
  hash-verified local config and must reproduce reviewed production bytes.
- No installer apply, activation, trading worker, test order, real-order cancel,
  broker connection, Scheduler edit or journal/reset/re-arm was performed.
- Final workspace hygiene reported a concurrently changed, pre-existing
  `scratch/idea_checks/_state/queue.json`, outside this task. It was preserved
  and excluded from staging; task source review and production hashes passed.

Remaining operator-dependent work:

1. Review exact owned orders/executions/positions and preserve today's halt.
2. After safe worker quiescence/settlement and outside the mutation window,
   use the README's disabled `stage --apply`, then installed preflight.
3. Run user-approved broker reads for the selected future session. Existing
   `daily_launch.ps1 -DryRun` is broker preflight, **not an offline test**.
   Legend `--kill`/`--verify-only` are order-mutating safeguards, not read-only QA.
4. Choose manual filled-position fallback or wait for a separately prepared,
   broker-qualified owned-close mechanism. Same-ID STOP-to-MKT remains an
   offline prototype; never enable it from mock guarantees. Manual fallback
   can cause Legend to miss its 09:31:20 transmit deadline.
5. Truthfully attest fresh exact-owned settlement, enable only an unjournalled
   future full session, and authorize normal worker launch separately. Verify
   real startup/runtime/broker evidence before claiming the changes are live.

Rollback and interruption-recovery commands are in README.md. They preserve
all journals, ownership/attempt/risk state, flags and the shared ledger. They
are file operations, cannot reverse fills and require quiescent workers.
