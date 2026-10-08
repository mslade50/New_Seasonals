Open Breakout reliability v2 is an isolated offline source candidate. It has not
been installed, activated, connected to IBKR, or qualified on native paper orders.
The execution reconnect candidate is separate. All rollout flags default off.

The order lifecycle records intent before transmission, fences each cancellation
once, consumes native decoder callbacks independently of SDK status, and requires
complete current owner orders, positions, executions and completed-order evidence.
Immediate executions bind exact durable identity and enter the journal before
protection; failed journal commits remain eligible for execution-query redelivery.
An immutable initiating cause links subsequent symptoms without replacing it.

Entry acknowledgment waits run outside the shared risk lock. A peer market can
proceed only when a fresh complete owner cycle bounds the existing exposure.
Attempts and daily reservations remain consumed after an uncertain order outcome.
Price recovery can clear its own pause after reconciliation; it cannot clear an
order incident, restore an attempt, or rearm a breakout. The price-only grace is
600 seconds measured from the last actual local callback.

Known late entry fills continue to be journaled and protected while halted.
Protection revisions require a current native body echo. Native terminal status
plus a later complete execution cycle proves closure; SDK Cancelled and order
omission alone do not. Normal and opposed-entry net-zero exits use distinct,
durable flat-exit cancellation fences, with persistent closure/deadline tracking.

Automatic emergency flattening and native coordination close are unqualified and
disabled in reliability mode. Native OCA quantity reductions without a matching
qualified revision, execution corrections, exit-induced position reversals, and
late entries with uncertain old exits require loud manual reconciliation. Exact
proven closed exits permit new protective identities for a subsequent late fill.
These are explicit limits, not successful-paper qualification claims.

See INSTALL-ROLLBACK.md for separate gated rollout, QUALIFICATION.md for the
paper requirements, CASE-MAP.md for deterministic coverage, and evidence/ for the
final local results. CI verifies public offline fixtures and does not certify the
operator's local baseline or paper order behavior. Frozen inventory and archive
hashes identify the reviewed bytes.
