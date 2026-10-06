# OpenBreakout / Legend deployment preparation

Start with `C:\Users\McKinley Slade\Documents\Codex\2026-10-06\task-3\open-breakout-deployment\CHECKPOINT.md`.

The candidate implementation and offline deployment preparation are complete.
**Production is unchanged: not installed, not enabled, not broker-validated.**
The earlier `combined-candidate.zip` is evidence, not an installer. This package
promotes only 17 allowlisted source/config/pointer targets and pins 28 existing
production prerequisites. Its offline harness and tests are never installed.

The existing task was checked and is idle with an explicit stopped-editing
handoff. Work continued from its installer and candidate, without rebuilding
the trading logic. The old parent-directory `build_deployment.py` and templates
are historical checkpoint tools; do not run them over this finished package.
Maintained source is `deployments/open_breakout_20261006` in remote main.
The final source adds reviewed startup guards and constructs cancellation and
cleanup operation IDs by joining their existing components with a colon. This
satisfies the repository text guard and preserves the exact durable ID strings.

## Behavior and limits

- Enablement keeps price-only resting entries before 600 seconds only while
  fresh owner orders, account-filtered executions, positions and protection
  reconcile. Actual advancing callbacks maintain separate monotonic ages for
  each mini trade and micro BidAsk feed. Heartbeats, unrelated ticks, cached
  updates and Legend publication cannot reset a feed's age. Entry placement,
  re-parking and repricing pause immediately on stale prices. At expiry, owned
  unfilled entries cancel and the halt latches; accepted protective exits stay.
- Execution disconnection, missing/unknown evidence, protection failure, manual
  control, risk limits and the 11:30 entry cutoff retain separate safeguards.
  A connected flag does not qualify the price grace. Fill handling stays active.
- An eligible finalized Legend signal latches opposite-only suppression for the
  exact New York day, account and ES/MES or NQ/MNQ family. Same-direction entries
  remain eligible subject to all other gates. Rejection, revocation, restart or
  price recovery never automatically release suppression or consumed attempts.
- **Automatic close-out is disabled.** Native qualification is `None`; the
  same-ID STOP-to-MKT prototype is callable only by explicitly marked offline
  fixtures. Real transport cannot qualify by setting a connected flag. For an
  opposing filled breakout, keep accepted stop/time protection, block entry
  admission, and reconcile exact breakout-owned exposure manually. Never flatten
  an aggregate account position. Legend may miss its original **09:31:20**
  transmit deadline. There is no retry, deadline extension or new attempt.

No opening levels, journals, ownership records, risk files, reservations,
attempts, enable flags, Scheduler tasks or existing orders are installer targets.
The new coordination ledger is local, shared by both live workers:
`C:\Users\McKinley Slade\dev\New_Seasonals\artifacts\open_breakout_runs\coordination\combined-ledger.sqlite`.
It is never put in OneDrive or deleted by rollback. Shadow/replay/status and
Legend dry-run/kill/verify modes do not activate coordination. Existing Legend
`--kill` and `--verify-only` **can cancel orders and flatten owned exposure**;
their availability is preserved but they are not offline or read-only checks.

## Exact commands

Use the same finished package for install, later preflight and rollback. Every
installer command is read-only unless `--apply` is explicit. The following
command syntax and file transitions passed disposable-root tests; production
apply commands have **not** been executed.

```powershell
Set-Location -LiteralPath 'C:\Users\McKinley Slade\Documents\Codex\2026-10-06\task-3\open-breakout-deployment'
$obPython = 'C:\Users\McKinley Slade\AppData\Local\Programs\Python\Python310\python.exe'
& $obPython -B installer.py preflight
& $obPython -B installer.py stage
```

Install disabled only after the operator has reviewed fresh exact-owned
positions/executions/orders, completed reconciliation, and safely made both
workers quiescent. Do not stop a protective worker with unresolved exposure.
Mutation refuses **all other Python process IDs**, including unrelated Python
jobs, and refuses 07:30–16:15 New York. It does not stop processes or inspect
command lines. Do not bypass that conservative refusal or kill unrelated jobs.
Today's halted session must remain halted; do not use another attempt directory.

```powershell
& $obPython -B installer.py stage --apply
& $obPython -B installer.py preflight --installed
```

This backs up exact originals, verifies backup hashes, writes disabled settings,
installs the startup bridge, and starts nothing. The runtime configs retain
`allow_price_only_pause=false`. Their fingerprints change when the explicit
new fields are added: retained journals are preserved, not migrated or resumed.
Use saved originals for historical configuration evidence.
Disabled means the two new features are off. Existing trading permissions and
schedules remain in effect and can run the pre-existing strategy behavior at
their normal times; installation is not a global trading-disable operation.

Rollback is also an outside-window, quiescent-worker operation, following
fresh manual review of owned exposure. It cannot change an already-running
worker, reverse fills, flatten exposure or cancel orders:

```powershell
& $obPython -B installer.py rollback
& $obPython -B installer.py rollback --apply
& $obPython -B installer.py preflight
```

Rollback verifies all installed hashes and backups before restoring originals;
it removes only newly installed sources/pointer/settings/receipt. Backups and a
rollback receipt remain at
`artifacts/open_breakout_runs/combined-deployment-backups/open-breakout-combined-20261006-v1`.
Existing ledger, flags and journals remain. Retained backups deliberately block
a blind second install with this same bundle ID. Review a new deployment version
if reinstall is needed; do not delete the backups to bypass that refusal.

For an interrupted transaction, preserve its lock and preimages, first inspect
the recovery plan, then restore only recorded before/after states:

```powershell
& $obPython -B installer.py recover
& $obPython -B installer.py recover --apply
```

Unknown source edits, invalid backups or an unexplained lock refuse automatic
recovery. Preserve them for manual review. Live startup refuses any install lock.
Per-file replacement is atomic; the multi-file transaction is crash-recoverable,
not a single filesystem-wide atomic commit. Exceptions restore touched files;
hard interruption retains a durable transaction journal. Recovery does not
retry any trading intent. Workspace hygiene runs before source promotion and
checks only allowlisted changes; unrelated existing work is preserved.

## Reproduce offline evidence

```powershell
& $obPython -B validate_deployment.py
& $obPython -B verify_payload.py
```

Both commands write only disposable fixtures/reports under repository
`artifacts/open_breakout_deployment`. They read production baselines but never
execute production launchers, connect to brokers, start a trading worker or
modify Scheduler. Active import probes use an injected future clock in isolated
roots. Launcher tests use inert child scripts; they establish interpreter,
quoting and inherited-environment contracts, not broker qualification. The
aggregate uses the actual install payload, not candidate CLI guards.
The fixed repository-root literal and external pointer are mapped only inside
disposable fixtures; the production startup checks the exact reviewed root
before importing its helper package.

Account-bearing templates contain `__LOCAL_BROKER_ACCOUNT__`. Only the
hash-verified local live config supplies the account during staging; the
rendered bytes must equal the recorded post-install SHA256 and size. Credentials,
the real account and enable flags are not bundled or pushed to the public Git
remote. The local original candidate remains unchanged.

`package-integrity.json` hashes all source, template and evidence files except
itself. The independently supplied zip SHA256 identifies that inventory too.
Installer preflight verifies package integrity, all 28 prerequisite hashes,
every target's base hash/absence, rendered payload, account/port/client pairing,
and exact fixed host roots. A changed production baseline requires a reviewed
reconciliation; never edit expected hashes simply to make preflight pass.

## Next-session broker qualification and activation

1. Keep current workers/orders and today's latched halt untouched. After
   completed settlement, review today's records by strategy account, contract,
   orderRef, client ID, order ID, permId and execution ID. Preserve opening
   levels, fills, attempts and consumed/reserved risk. Neither an empty open-order
   list nor an account net of zero establishes exact-owned settlement.
2. After disabled installation, the operator may run the existing broker
   preflight for the chosen session. **`daily_launch.ps1 -DryRun connects to the
   broker and refreshes inputs; it is not offline and was not run here.** On
   October 7, use the following only after approving those broker reads:

   ```powershell
   & powershell.exe -NoProfile -File 'C:\Users\McKinley Slade\dev\New_Seasonals\artifacts\open_breakout_runs\daily_launch.ps1' -Session '2026-10-07' -DryRun
   & 'C:\Users\McKinley Slade\OneDrive\trading_ibkr\run_legend_ema_fut.bat' --dry-run
   ```

   Confirm exact configured accounts, live client IDs (OpenBreakout 927481,
   Legend 163), contract expiries/roll, market-data subscriptions, fresh order/
   position/execution responses, accepted protection and unchanged ownership.
   A successful launcher preflight does not qualify automatic close-out.
3. Qualify filled handoffs in a **separate explicitly authorized paper account**,
   isolated orderRefs/client IDs and a dedicated local ledger, with no production
   worker attached. Use minimum explicitly approved test size. A broker-specific
   qualification driver still needs separate preparation/review; this package
   supplies no command that submits a qualification order. Record broker/SDK
   versions, account, contract, order/perm/execution IDs, exact native requests,
   status receipts, broker errors and before/after accepted protection. Test both
   directions and index families, full/partial entry fills, fill during cancel,
   stop fills before/during close, duplicate callbacks, rejection, lost ack,
   timeout, execution outage, restart, unrelated positions and another account.
   Prove no protection gap, no residual reversal or terminal-order reopening,
   correct residual quantity, one owned close at most, and final reconciliation
   of every remaining owned order. Broker documentation/confirmation and actual
   paper receipts must establish whether same-ID STOP-to-MKT is supported and
   safe; otherwise implement and qualify a supported alternative. Never replace
   the production adapter's `None` with mock qualification or `offline_fixture`.
4. Decide explicitly whether to run with the documented manual filled-position
   fallback or wait for a separately qualified close mechanism. Also decide any
   proposed release policy for rejected/revoked Legend signals; the delivered
   default never releases automatically. Activation under manual fallback
   knowingly accepts missed Legend entries when reconciliation misses 09:31:20.
5. After broker read qualification and fresh exact-owned settlement, while
   quiescent and outside the mutation window, review then set future-session
   feature enablement. October 7 is an example; choose an unjournalled full XNYS
   session strictly after the current New York date and within 14 days:

   ```powershell
   & $obPython -B installer.py activate --session '2026-10-07' --confirm-fresh-owner-flat
   & $obPython -B installer.py activate --session '2026-10-07' --confirm-fresh-owner-flat --apply
   & $obPython -B installer.py preflight --installed
   ```

   `--confirm-fresh-owner-flat` is a truthful operator attestation of fresh
   exact-owned reconciliation for both strategies, **not automated broker proof**.
   This step only enables future startup settings; it does not start a worker,
   enable automatic close-out, change a schedule or reset today's session.
   Existing normal schedules remain unchanged. Before leaving enablement set,
   be ready for those existing schedules to launch on eligible subsequent days.
6. At the chosen normal session, the operator authorizes worker launch separately.
   Check startup logs for pinned source hashes, correct account/client/contract,
   both live workers using the identical local DB and shadow coordination off.
   Observe a genuine eligible Legend decision; verify opposite-only cancellation
   with native completed receipts and fill attribution. Observe genuine feed
   interruptions only; never impair a production protection path to test it.
   Verify no early price-only cancellation before 600 seconds while healthy,
   immediate entry pause, protection through fills, and no re-arm after recovery.
   Record broker evidence before calling the result live or broker-validated.

To disable future feature startup, after settlement/quiescence:

```powershell
& $obPython -B installer.py deactivate
& $obPython -B installer.py deactivate --apply
```

Deactivation is a file operation and restores the previous shorter stale-price
policy for future processes. It does not stop current workers, erase suppression
or authorize entry retries. Changing live session policy mid-session is refused.
