# Legend EMA ETF sleeve runbook

## Current release status

The SPY/QQQ-only release is described in [ETF-native integration](legend_spy_qqq_2026-09-16.md).
The default signal command now uses raw SPY/QQQ history from read-only IBKR.
CME/Databento data is not required. The ETF research and broker runner are
connected with 09:31 entry and 10:30 time exit. Live execution remains gated
by exact-date configuration, native signal parity, shared-executor capacity
and integration evidence, and actual paper broker proof.

The historical activation notes below describe the superseded futures setup.

### September 6 baseline

The ETF implementation is built and its signal replay passes, but operational
setup is incomplete and live activation remains disabled. As of September 6:

- no Windows scheduled task has been installed;
- the dated live gate is disabled;
- the Databento cost ceiling is $0.00, so no recurring spend is authorized;
- no paper or live broker order has been sent by this rollout;
- guarded shared-executor support was deployed September 2 and repaired for
  legacy account discovery; its Legend activation marker remains absent;
- the purchased ES/NQ/RTY data is installed in the machine-global cache;
- Primary TWS currently refuses its configured read-only API connection;
- Primary-only runtime configuration and the daily capacity producer remain
  unfinished; the $0.50/day data allowance request is awaiting an answer;
- IBKR's exact OCA type-2/GAT behavior at 10:30 has not yet been proven in the
  PA paper account.

The owner has authorized progressing toward activation. Complete and verify the
broker and shared-capacity requirements before arming the live date. Do not
interpret passing source tests as observed broker execution.

Use a dedicated stable Git checkout for scheduled execution, pinned to a reviewed
commit. A checkout under `artifacts/worktrees` or `artifacts/task_worktrees` is
for review only. Run `scripts/configure_legend_etf_shadow.py --executor-root
<executor-path>` there to verify Primary identity read-only; add `--apply` to
create the new Primary-only shadow configuration. This keeps paid data and live
execution disabled and refuses to overwrite an existing configuration. Register
shadow tasks only after configuration and current signal inputs are ready.

## Pinned strategy

This release trades only SPY and QQQ. Each ETF creates its own setup:

- Use raw IBKR TRADES bars, RTH only, from the preceding 20 calendar days.
  Fetch 21 days to ensure a complete first day, then select the exact window.
- Require every expected XNYS 15-minute bar in that window and at least 200
  warmup bars before the setup day. The immediately preceding exchange session
  must be a full 26-bar session; known early closes fail closed.
- Require `abs(close - open) / (high - low) >= 0.75` and no bar whose inclusive
  low/high range touches its finalized same-bar EMA20. This is the original
  ETF research qualification rule; candle direction does not add a new filter.
- Carry EMA20 continuously through that window's RTH closes. The bounded seed
  makes preparation and execution deterministic without downloading decades
  each morning. Historical comparison uses this same window on both paths.
- Hash the prior OHLC history and recompute it from IBKR before entry. A changed
  history or setup blocks entry and requires a new plan. Current-session bars
  never enter the prior-session signal.

Execution is ETF-native and uses raw, unadjusted RTH prices:

- Subscribe to ordered IBKR `Last` trade ticks before the open.
- Build the exact 09:30 ETF minute from those trade ticks. If the ETF opens
  below its seeded raw RTH EMA20, the fade is long; if it opens above, the fade
  is short. An open equal to the EMA is rejected.
- Reject the trade if the penny-away EMA target was touched in the 09:30
  minute. The first `Last` tick at or after 09:31 is the decision and entry
  reference; the runner then prepares an IOC market-entry parent. Immediately before
  transmission, re-read all ticks since that decision and reject the entry if
  the target has crossed. No entry may be transmitted after 09:31:20, and the
  actual broker fill is reconciled separately.
- Keep the validated 5-second ETF bar feed for causal 15-minute EMA revisions.
  Revise the penny-away limit after completed windows at 09:46, 10:01, and
  10:16.
- Exit at 10:30. There is no stop. Ex-dividend ETFs are excluded, and shorts
  require the live borrow checks to pass.

Neither IBKR CME market data nor Databento is used by the default signal path.
IBKR supplies SPY/QQQ history, ticks, bars, account and borrow/dividend references,
and order state. ES/NQ/RTY research remains available behind the explicit
`--source databento-research` option; its plans cannot enter live execution in
this release. Recovery still audits all legacy symbols, including IWM, so an
old unresolved lot cannot disappear from supervision.

## Sizing and portfolio capacity

The sizing denominator is a stress convention, not a stop loss:

| Account | Long/root | Short/root | Legend gross cluster cap | Denominator |
|---|---:|---:|---:|---|
| Primary | 10 bps NLV | 5 bps NLV | 30 bps NLV | 1.25 x prior raw ETF Wilder ATR14 |
| PA | 5 bps NLV | 2.5 bps NLV | 15 bps NLV | 1.25 x prior raw ETF Wilder ATR14 |

ES/NQ/RTY are one correlated equity-index cluster. There is no overlap boost.
Runtime values may reduce the reviewed limits but cannot increase them. Set the
selected PA account's `MAX_SHARES_PER_ROOT=1` for the one-share paper drills.

Legend's own caps are only one layer. Live preflight also requires one atomic
shared equity-index portfolio budget for the exact account set. That artifact
must:

- use protocol `legend-equity-index-risk-budget-v3` and risk basis
  `stress_atr_bps`;
- be generated for the entry date between 08:30 and 09:25 ET;
- expire at the next midnight ET, so the same atomic ledger covers later
  mapped-ETF entries from the existing executor (Legend itself still has the
  independent hard 09:31:20 entry deadline);
- name the exact deployment-manifest SHA256; and
- provide remaining long, short, and gross basis-point capacity per account.

V3 is a mutable, machine-global reservation ledger rather than a frozen hash.
Immediately before the first entry mutation, Legend holds the shared OS lock
and atomically debits the complete selected-account batch under one idempotent
owner token. The lock stays held through the transmit phase. Reserved capacity
is intentionally not returned on cancel/replace, so a crash or retry cannot
double-spend the cluster budget. Any inconsistent owner token or insufficient
long, short, or gross capacity blocks every entry in the batch. The existing
execution stack does not yet publish or participate in this artifact, so live
is currently impossible by design.

## Machine-global runtime

All mutable runtime material belongs under:

```text
%LOCALAPPDATA%\NewSeasonals\legend_etf
```

This includes `runtime.env`, the immutable daily `signal_plan.json`, the
frozen ETF history inputs, optional legacy futures cache/charge ledger, live/dry state and audit
files, critical alerts, scheduler receipts, logs, reservations, quarantine
markers, deployment manifests, and daily portfolio budgets. It is shared by
all checkouts on the machine, so moving between a Git checkout and a worktree
cannot create a second independent execution state.

Install the pinned dependencies and copy the template from a stable checkout:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-legend-etf.txt
$legendRuntime = Join-Path $env:LOCALAPPDATA "NewSeasonals\legend_etf"
New-Item -ItemType Directory -Force -Path $legendRuntime | Out-Null
Copy-Item docs\legend_etf_runtime.env.example (Join-Path $legendRuntime "runtime.env")
```

Replace every account and path placeholder. Do not store an IBKR password,
Databento API key, or other secret in `runtime.env`. The Databento key remains
in `DATABENTO_API_KEY` or the Windows credential vault service
`New_Seasonals.Databento`, username `prod-001`.

The PA paper Gateway default in the template is port 4002. The primary and
authoritative ETF-feed ports are deliberately operator-configured: verify the
actual TWS/Gateway mode rather than assuming 7496/4001. Client IDs must be
unique.

## ETF plan preparation

At 08:45 ET, create the immutable prior-session ETF signal plan:

```powershell
python scripts\prepare_legend_etf_signals.py
```

For an intraday read-only check that does not create an executable plan:

```powershell
python scripts\prepare_legend_etf_signals.py --check-data --output artifacts\native-check\signal_plan.json
```

The normal producer only writes plans between the prior close and the entry
session's open. Missing/stale history fails closed. Frozen raw inputs are
retained beside the plan. Client ID 156 is read-only and separate from the
session feed (154) and Primary account session (155).

The old Databento cache and charge protections are retained exclusively for
explicit research runs. No Databento allowance is needed for SPY/QQQ operation.

## Shadow workflow

Prerequisites for every scheduled or manual run:

- Windows timezone is `Eastern Standard Time` and the clock is synchronized;
- the user is signed in, wake timers are permitted, and the machine can wake;
- IB Gateway/TWS is already running and logged in;
- the configured paper/live ports, account IDs, client IDs, and SPY/QQQ/IWM
  market-data permissions are verified; and
- the stable checkout and pinned Python environment have not changed since the
  reviewed manifest was built.

Validate today's plan and read-only connectivity:

```powershell
python scripts\run_legend_etf_session.py --accounts primary pa --check
```

Start a complete shadow session by 09:28:

```powershell
python scripts\run_legend_etf_session.py --accounts primary pa --shadow-through-exit
```

Dry run may read market and account state but refuses place, cancel, and modify
calls. A release-quality shadow must run through 10:30 and preserve its state,
audit, receipt, and log evidence in the machine-global runtime directory.

## Shared-executor deployment boundary

Before live use, every active Python path under the external IBKR executor root
that can call `placeOrder`, `cancelOrder`, or `reqGlobalCancel` must participate
in the same reservation protocol. The external root must also contain the
reviewed support modules `legend_reservation_guard.py` and
`legend_portfolio_budget.py`.

All raw `.placeOrder`, `.cancelOrder`, and `.reqGlobalCancel` calls must live
only in the central `legend_reservation_guard.py`; active executor modules call
that wrapper. The inventory includes every active Python source, including
`*_selftest.py` files that can mutate when invoked with a live flag, plus the
reviewed `contract_reference.json`. The `legend_portfolio_budget.py` support
module must expose the V3 atomic capacity reservation function.

The deployment handshake is deliberately one-way. The manifest builder first
writes `.legend_reservation_guard_required.json`, then the shared reservation
configuration. Once the marker exists, every guarded executor must load the
exact attested guard bytes and fail closed if the manifest, source hash,
configuration, interpreter, required tests, or marker validation fails. Never
remove the marker as a rollback; restore the reviewed backup as a unit.

Every mutation uses an explicit broker account and positive `conId`. Fresh
position snapshots and raw all-client open-order snapshots are taken twice for
startup and periodic reconciliation; a missing, changing, or ambiguous identity
blocks the action. Reconciliation itself requires the connection to expose the
one configured managed account; an explicit account argument cannot make a
wrong Gateway endpoint authoritative. Dollar-risk conversion requires exactly
one account-bound NetLiquidation row. A new exit may not join an existing OCA
group: IBKR stock orders have no atomic reduce-only guarantee if a sibling fills
before the new order reaches the broker. Those group-bound close helpers fail
before any broker mutation and preserve the original protection. The generic
cancel, modify, flatten,
trim-and-readd, and add-to-position executor commands are disabled because they
cannot preserve these invariants. Mapped option execution is limited to a
single long 1x option or a canonical 1:1 same-expiry, same-right vertical;
direction and bounded loss come from the qualified broker contracts, not from
the request payload. Scheduled dynamic market-option execution and one-leg
option/FOP/BAG position closes are disabled. `close_only`, `pa_positions.py`,
and `pa_flatten.py` may mutate only the guarded SPY/QQQ/IWM/DIA index clusters;
general-stock use is deliberately rejected until it has an equivalent durable
mutation lifecycle.

The execution relay durably locks, appends, flushes, and fsyncs each command ID
before a live subprocess can start. If the receipt cannot be read or written,
the command is rejected before execution. Any nonterminal, negative, malformed,
or missing child result is reported as unknown with `DO NOT RETRY`; it is never
reported as executed.

Only after those sources and support modules are patched and tested, produce a
reviewed integration receipt containing the exact executor source-tree hash
and the full required integration-test set, then build the deployment guard
with the current passing `--candidate-parity-evidence` via
`scripts\build_legend_executor_guard_manifest.py`. The builder discovers all
active sources, proves raw mutations are centralized, validates and hashes that
receipt, hashes the complete Legend runtime source set and critical Python
runtime, and writes the shared reservation configuration. Put the external
root, resulting manifest path, and manifest SHA256 in `runtime.env`. Any source,
interpreter, package, path, receipt, or reservation-directory drift then blocks
a new live entry.

The guarded patch has not been applied to the external live executor. That is
a real-money boundary and requires explicit approval immediately before copying
the frozen, audited files. Copying code does not create the marker, arm the live
gate, install tasks, connect to IBKR, or authorize an order.

The OLV book-cap path uses a durable two-phase lifecycle. It atomically records
`PENDING` with the exact account, `conId`, OCA identities, attributable lot,
discretionary baseline, and preassigned trim order ID before its first mutation.
On restart it takes two stable broker snapshots and either proves the invariant,
repairs the exact exit quantity, or cancels only the exact trim and enters a
manual/critical state. A submitted trim remains pending until terminal broker
state and the resulting exits are reconciled; journal presence alone never
authorizes a skip. Both configured accounts must connect and pass read-only
planning before either can submit. A cap trim is automatic only when the exact
contract has one filled OLV lot and the total live position exactly equals that
lot; overlap, a second lot (including one due to exit today), or any nonzero
discretionary baseline is manual/no-mutation. If recovery finds the OLV lot
flat with its exits still working, it validates the complete recorded OCA
topology before mutation, then guarded-cancels every exact sibling separately
and proves the group is gone. New trims stop at 15:58 ET, with a final clock
assertion inside the guard immediately before the broker wire call. If that
final check crosses the cutoff, the proved no-wire path immediately restores
the original exit quantities before the executor may continue. After that,
`--reconcile-only` may contain unresolved work through the 16:05 cutoff.
Multiple OLV lots in one symbol, a working entry requiring resize, corrupt
state, or an ambiguous identity fails closed for manual handling.

The separate OLV pre-market exit runner also binds both endpoints to distinct
configured accounts before either can mutate. Its legacy group-bound close is
deliberately rejected by the active shared guard because a fresh OCA snapshot
cannot make a new stock order atomic. The original bracket remains working and
the runner reports a protected failure; do not enable this exit path until an
in-place or broker-proven lifecycle has passed paper proof. A journal row never
authorizes a blind retry.

## Scheduler

Preview the three weekday tasks:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\register_legend_etf_tasks.ps1 -Accounts primary,pa
```

The shadow preview is:

- 08:45 — prepare the futures signal plan;
- 09:28 — run the shadow or dated-gated live session; and
- 10:40 — watchdog receipt validation and, in live mode, broker
  `--reconcile-only` recovery.

A live registration gives the same watchdog task additional 12:00, 14:00,
15:45, and 16:10 triggers. Each reruns the independent correction audit and
exact-lot reconciliation. The intraday sweeps can contain an attributable bust
while SPY/QQQ/IWM are still in RTH; the 16:10 sweep is detection/quarantine
first because outside-RTH market execution is not assumed. The next scheduled
live startup audits all three ETFs again before any new-entry gate.

The 10:40 live watchdog re-proves zero exact-owned shares and zero attributed
working orders, but does not release the durable account-symbol quarantine on
the entry date. Release requires a subsequent-session, all-three-ETF clear
correction audit for that account. Both scheduled tasks run that independent
read-only audit on non-full sessions, so late busts/corrections are still
detected. Until the clear proof, patched executors continue to block the
account-symbol.

The watchdog lease records the exact process ID, process creation time, and run
ID. A takeover may fence only that exact scheduled child and must reconcile
broker state afterward; a lock loser cannot overwrite the active owner's lease.

Task installation is a separate operational change. The installer refuses to
install from `artifacts\worktrees`, refuses unreviewed name conflicts, builds
all three definitions before the first mutation, and verifies the installed
set. It uses an interactive principal, `WakeToRun`, start-when-available, and
three one-minute restart attempts. Those settings do not replace the login,
Gateway, or clock prerequisites above.

The three shadow tasks were installed September 16 from the stable checkout.
They begin September 17. New shadow installations use `-Install`. A live task additionally requires `-Live` and
the literal acknowledgement printed by the installer, but it still cannot
trade without the independent exact-date runtime gate.

## Live gate and staged rollout

ETF-native candidate parity is a release invariant. Run
`scripts/verify_legend_etf_candidate_parity.py --data-dir <15min-history> --output <evidence.json>`
on the exact release tree. It compares the retained original ETF research rule
with the new production evaluator over 2012 through August 2026, including
rejected dates. Complete exchange grids are required; missing historical inputs
are counted explicitly. Passing evidence requires both SPY and QQQ, at least
2,500 evaluated sessions per ETF, zero qualification mismatches, and bounded
EMA/ratio differences. Inputs, source files, interpreter, and packages are
hashed. Source/input changes during replay invalidate the result.

The release manifest requires this native proof. The old 342-candidate futures
replay remains a useful execution regression but cannot authorize this version.
Broker fill behavior still requires the drills below.

The rollout gates are:

1. Exact current-tree historical replay is green and its input hashes are
   retained.
2. At least five complete full-session shadows prove the exact 09:30 tick
   minute, first 09:31 tick, pre-transmit no-cross check, 5-second feed quality,
   all three target revisions, 10:30 management, receipts, and 10:40 watchdog.
3. On the PA paper Gateway at port 4002, run reviewed one-share long and short
   drills. Prove borrow handling, partial target fills, restart during
   submission and revision, disconnect/reconnect recovery, exact `orderRef`
   echo in both open-order and execution callbacks, and no over-exit.
4. Explicitly prove the OCA type-2 target plus 10:30 GAT market sibling in the
   paper account. In particular, verify that the target does not remain routed
   in a way that blocks the time exit, that the time exit fills at the intended
   cutoff, and that partial fills reduce the sibling correctly. Retain the
   broker events as the paper proof artifact.
   The artifact must bind to the exact deployment-manifest SHA256, prove the
   IOC parent leaves partial-fill children active, keep all target revisions at
   or below 1,000ms, and prove the 10:30 exit within five seconds.
5. Run several bounded PA paper sessions. Only then consider a separately
   reviewed, exact-date PA activation. Primary activation comes later and
   requires its own explicit review.

The code submits the target/time pair as OCA type 2, but broker-held 10:30
reliability is **not claimed until gate 4 is proven**. At 10:30:05 the running
application takes tight-timeout account-wide execution/order/position
snapshots, batches exact-order cancellations, durably records every safe
residual exit, transmits all required market exits before slow per-lot proof,
and never flattens a collided/netted symbol. That is a fail-safe, not a
substitute for proving broker behavior during a local or network outage.

For one broker-mutating paper or live date, the machine-global `runtime.env`
must contain the exact dated gate, exact account IDs, the reviewed reservation
directory and deployment manifest hash, and that date's shared portfolio
budget. Then invoke `--live` only for the explicitly approved account labels.
Enabling shorts is separate from enabling longs; primary is separate from PA;
the date expires after the session.

## Failure, recovery, and halt behavior

- Missing/stale ETF setup history, a bad required-bar grid, a roll, stale or invalid ETF
  ticks/bars, stale EMA/ATR history, clock skew, dividend-reference failure,
  delayed/missing borrow data, account/contract/client mismatch, portfolio
  budget failure, guard-manifest drift, or a late start blocks new exposure.
- Any existing SPY/QQQ/IWM position or working order in the selected account
  blocks that root. V1 never nets another sleeve's position.
- State is persisted before broker submission and target modification. A
  restart reconciles exact account, client ID, `conId`, `orderRef`, order IDs,
  and fills. A prepared target revision is abandoned; a transmit-authorized
  revision is resolved from broker state (pending target, prior target, or
  prior-target repair after the deadline). A stale pending target is never
  applied late. It never blindly retries an ambiguous submit or revision.
- The named process mutex prevents two Legend sessions from racing. The shared
  reservation and durable quarantine prevent a *patched* external executor
  from racing the same account-symbol. Until every external mutator is patched,
  this cross-sleeve isolation guarantee does not exist and live remains NO-GO.
- If protection cannot be proved, recovery cancels only exact attributed
  orders, recomputes the signed virtual lot, and market-exits only those owned
  shares. It never sends an ambiguous whole-symbol flatten. Over-exit,
  unprotected lots, or unresolved working orders create a durable critical
  state and alert.

To halt new entries, set `LEGEND_ETF_LIVE_ENABLED=0` and keep the live session
and 10:40 watchdog available. The kill switch blocks new entries, but durable
same-day broker-mutated records are still reconciled exit-only; gate or build
attestation failure is reported after containment. Do not kill Gateway/TWS or
delete runtime state as a halt procedure, and never use a generic whole-symbol
flatten for this sleeve.

## Verification

```powershell
python -m pytest -q tests/test_legend_etf_native.py `
  tests/test_legend_ema_backtest.py tests/test_legend_etf_signal.py `
  tests/test_legend_etf_execution.py `
  tests/test_legend_etf_recovery.py `
  tests/test_legend_etf_parity.py `
  tests/test_legend_etf_hardening.py
python -m ruff check legend_etf scripts/prepare_legend_etf_signals.py `
  scripts/run_legend_etf_session.py scripts/check_legend_etf_calendar.py `
  scripts/verify_legend_etf_candidate_parity.py `
  scripts/build_legend_executor_guard_manifest.py
```

Automated tests and historical parity do not replace the five-session shadow
evidence, external-executor integration, or PA broker paper drills.
