# Day-trade futures order control — September 30, 2026

Yesterday's MES Close/adjust-exits and both manual edits failed before any
broker mutation. The day-trade service retains IBKR client 927481 throughout
the session. The old executor attempted a second connection with that occupied
client ID; its empty TimeoutError produced `Nothing changed:` / `Nothing sent:`.
The subsequent close_only rejection correctly refused the two working exits.

The executor now uses an authenticated loopback handoff when an active owner
registers its connection. The handoff supports snapshots and modification or
cancellation of existing exact account/contract/client/order/perm identities.
It cannot place a new order. Close submissions retain the executor's existing
gates, durable journal and fill reconciliation. All owners are resolved before
the first exit changes. A failed/stale registration never becomes a second
connection attempt. Connection failures now name the client and exception.

Close adjusts OCA exits proportionally before submitting the requested close;
a full close confirms exit cancellations first. Manual Save edits only the
selected order's quantity or prices, preserving timing, OCA, routing and other
terms. Save remains direct. A fill race or uncertain acknowledgement stops the
workflow, and an attempted edit is not automatically resent.

An operator mutation transfers that futures market to manual control for the
rest of the session. The day-trade strategy retains existing broker exits but
cannot re-enter, overwrite protection, or emergency-flatten against the user's
action. Other markets retain their strategy behavior. The journal records the
handoff and later owner fills as manual evidence; strategy R is unavailable
until attribution is reconciled. The next day's new journal starts normally.
The Execution ticket/editor displays this consequence before submission.

Activation uses the existing checkout and scheduled day-trade launcher. Install
only `execution_lifecycle.py` and `owner_connection.py` into the existing broker
runtime, preserving hash-verified backups. The executor imports these in fresh
command subprocesses, so ExecAgent needs no restart. The day-trade side starts
its handoff on the next normal launch; an already-running session does not load
new Python modules. Do not restart an active trading session just to load it.
The registry lives under the existing user's LocalAppData, is loopback-only and
uses a random per-launch secret. No broker credentials are stored in it.

Verification includes real local client/server protocol tests with inert broker
dataclasses, long/short and full-close OCA cases, both STP and scheduled MKT
edits, exact identity failures, fill races, lost acknowledgements, stale owner
registration, retained order terms, and manual-control strategy boundaries.
No live test trade or historical retry is part of verification. Native broker
acknowledgement through the handoff remains to be observed after its next launch.

IBKR requires the placing client for API order modifications; see
[IBKR modifying orders](https://interactivebrokers.github.io/tws-api/modifying_orders.html).
An occupied client ID is error 326 in
[IBKR error codes](https://ibkrcampus.eu/docs/tws-api/doc/error-handling/error-codes).

The two broker helpers were installed at 10:53 ET on September 30 with exact
candidate SHA-256 verification. The prior lifecycle and manifest are preserved
under `.runtime_backups/daytrade_order_control_20260930T105304` in the existing
broker checkout. No executor child was active, no agent/service was restarted,
and no order was submitted. The current day-trade process predates this repair;
the unchanged 08:12 ET daily launcher uses this checkout and will load it on
October 1. Preparation and installation receipts are retained under
`artifacts/execution-daytrade-repair-20260930/`.
