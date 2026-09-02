# Legend EMA ETF sleeve runbook

## Current release status

The local implementation is shadow-ready, but live activation is currently a
**NO-GO**. As of the 2026-09-02 handoff:

- no Windows scheduled task has been installed;
- the dated live gate is disabled;
- the Databento cost ceiling is $0.00, so no recurring spend is authorized;
- no paper or live broker order has been sent by this rollout;
- the existing shared IBKR executor has not yet been patched with the common
  account-symbol reservation guard or the shared portfolio-budget publisher;
- IBKR's exact OCA type-2/GAT behavior at 10:30 has not yet been proven in the
  PA paper account.

Do not register live tasks or arm a live date until every release gate below is
complete. Merging code is not authorization to trade.

## Pinned strategy

This is a dedicated SPY/QQQ/IWM execution sleeve for the original, validated
Legend EMA rule. Futures data creates the prior-session candidate only:

- ES -> SPY, NQ -> QQQ, RTY -> IWM.
- Use the complete prior CME equity-index RTH session, 09:30-16:00 New York
  time, with exactly 26 left-labelled 15-minute bars on a normal session.
- Require `abs(close - open) / (high - low) >= 0.75`.
- Every bar must remain strictly on the session-direction side of its same-bar
  RTH EMA20. Equality is a touch and rejects the setup.
- The setup and current contract instrument IDs must match. A missing setup
  anchor or 15-minute bin, an intervening observed futures session,
  insufficient causal EMA history, a roll, and known XNYS early-close entry
  dates fail closed. Databento trade bars may legitimately omit no-trade
  minutes or an exchange-closure date; those omissions do not invent bars or
  invalidate an otherwise complete observed-bar EMA segment.

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

IBKR CME market data is not needed for this ETF implementation. The futures
candidate is prepared pre-open from the Databento cache; IBKR supplies only
SPY/QQQ/IWM ticks, bars, account state, borrow/dividend references, and order
state.

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

- use protocol `legend-equity-index-risk-budget-v2` and risk basis
  `stress_atr_bps`;
- be generated for the entry date between 08:30 and 09:25 ET;
- expire exactly at 09:31:20 ET;
- name the exact deployment-manifest SHA256; and
- provide remaining long, short, and gross basis-point capacity per account.

V2 is a mutable, machine-global reservation ledger rather than a frozen hash.
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
150-calendar-day futures cache and charge ledger, live/dry state and audit
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

## Databento plan and cost boundary

At or shortly after 08:45 ET, create the immutable signal plan:

```powershell
python scripts\prepare_legend_etf_signals.py --max-cost-usd 0
```

The script maintains a machine-global 150-day rolling cache and seeds an empty
cache from the already-purchased archive when available. Every parquet write
has a SHA256 integrity sidecar; a missing/mismatched pair blocks the run. Each
refresh quotes and re-fetches a 30-calendar-day overlap (or an older append
gap), uses fresh rows to heal additions/corrections, and blocks if the fresh
response omits a previously cached row. This protects the observed-bar EMA
without pretending no-trade minutes exist. The default hard cost ceiling is
exactly $0.00; a positive overlap or append quote therefore makes no data
request. An ambiguous request or charge-ledger state also fails closed.
An exact request ID already recorded as persisted is served from its verified
cache and is never downloaded or counted as a new authorization again.

Any recurring Databento allowance requires separate approval. After that
approval, set both the reviewed numeric ceiling and the literal paid
confirmation in `runtime.env`. Historical quotes are not a promise of future
cost. Do not raise the ceiling merely to make a failed run pass.

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
`*_selftest.py` files that can mutate when invoked with a live flag. The
`legend_portfolio_budget.py` support module must expose the V2 atomic capacity
reservation function.

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

This work has not yet been applied to the external live executor. That is a
real-money boundary and requires explicit approval immediately before editing
and validating it.

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

No task is currently installed. When the stable checkout is reviewed, shadow
installation uses `-Install`. A live task additionally requires `-Live` and
the literal acknowledgement printed by the installer, but it still cannot
trade without the independent exact-date runtime gate.

## Live gate and staged rollout

Historical candidate parity is a release invariant. The exhaustive baseline
for 2016-01-01 through 2026-08-31 covered 2,660 sessions and matched 342 of 342
candidates: ES 110, NQ 139, RTY 93, with zero candidate mismatches (maximum
trend-ratio delta `1.11e-16`, maximum ATR delta `5.68e-14`). The parity evidence
records SHA256 hashes for every input parquet and the complete candidate
pipeline, plus the exact interpreter and package versions. The verifier rejects
a source/runtime, historical-engine, golden-file, or archive change during its
run. The deployment-manifest builder re-hashes every input and then binds that
passing evidence to its current Legend tree; the live validator keeps checking
the evidence, source, runtime, and input metadata. Re-run the verifier against
the exact stable release tree and archive; stale or worktree-bound evidence is
not sufficient.

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

- Missing/stale futures data, a bad required-bar grid, a roll, stale or invalid ETF
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
python -m pytest -q tests/test_legend_etf_signal.py `
  tests/test_legend_etf_execution.py `
  tests/test_legend_etf_recovery.py `
  tests/test_legend_etf_parity.py `
  tests/test_legend_etf_hardening.py
python -m ruff check legend_etf scripts/prepare_legend_etf_signals.py `
  scripts/run_legend_etf_session.py scripts/check_legend_etf_calendar.py `
  scripts/verify_legend_futures_candidate_parity.py `
  scripts/build_legend_executor_guard_manifest.py
```

Automated tests and historical parity do not replace the five-session shadow
evidence, external-executor integration, or PA broker paper drills.
