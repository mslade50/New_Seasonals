# Legend activation readiness — September 16, 2026

Legend is not live. Primary-only shadow configuration now exists in the
machine-global runtime, but tasks are not installed because the current
futures feed cannot satisfy the existing freshness requirements.

## Completed

- Verified Primary TWS account identity read-only on the configured port 7496.
  Created `runtime.env` with live execution, directions, account allowlist,
  live date, and paid data disabled. Account identifiers remain local.
- Verified live SPY/QQQ/IWM Last ticks and five-second bars without submitting
  orders. The repaired adapter retained 125/92/113 ticks spanning about 15.5
  seconds in a bounded September 16 broker probe. Clock skew was 0.50 seconds.
- Fixed tick retention: ib_insync clears `Ticker.tickByTicks` on each network
  packet. The adapter now captures each update into a bounded session tape,
  rejects overflow, and detaches callbacks on cancellation/disconnect. The
  regression test reproduced the loss before the fix and passes afterward.
- All 93 Legend tests passed, including the 342-candidate ETF execution replay.
  Ruff and diff whitespace checks passed. These are not paper execution proof.
- Checked 104 external executor sources: raw broker mutation calls are
  centralized in the structurally valid reservation guard. The activation
  marker is absent. This structural check does not replace integration tests
  or authorize enabling that guard for the existing live strategies.
- Previewed the Primary-only shadow task definitions at 08:45, 09:28, and
  10:40. No tasks were registered or started.

## Data blocker discovered during preparation

At 13:49 UTC, Databento's account-scoped metadata reported GLBX.MDP3 and
`ohlcv-1m` availability only through 05:49 UTC, approximately eight hours old.
The metadata endpoint returns availability for the caller's entitlements:
https://databento.com/docs/api-reference-historical

The current strategy requires a same-day 08:30–09:25 ET contract probe, and
the latest probe must be no more than ten minutes old. Updating the cached
history alone cannot meet these requirements. Do not change the timestamp,
treat a metadata request as an observed trade, or weaken the roll/freshness
checks to make a run pass. Resolve sufficiently fresh data access first.

The three-root 30-day refresh was quoted at $0.325573310256 on September 16;
all existing cache integrity checks passed and the daily charge ledger was
clear. No paid request was made. The $0.50/day allowance remains an unanswered
owner approval request, and the runtime cap remains zero. A future quote may
differ. This price does not establish access to timely futures data.

## Remaining before live activation

1. Resolve the futures feed delay and prepare a genuinely fresh morning plan.
2. Install and observe at least five complete shadow sessions. Today's opening
   window was missed; September 17–23 is the earliest five-weekday window,
   conditional on resolving the blockers and every session completing.
3. Complete the real shared portfolio-capacity producer and exact-source
   deployment/integration attestation. The existing budget writer is a
   library; no reviewed live capacity source/publisher is configured.
4. Verify a paper endpoint and conduct the runbook's one-share broker drills,
   including IOC partial fills, OCA/GAT timed exit, revisions, and recovery.
   The expected paper Gateway port 4002 was not listening at this check.
5. Review the evidence and obtain exact-date, account, direction, and sizing
   approval immediately before enabling live execution.

Diagnostic artifacts are under the main workspace's ignored
`artifacts/legend-activation-20260916/`. No broker orders, paid data requests,
shared-executor activation, or site publication were performed.
