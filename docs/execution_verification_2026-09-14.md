# Execution verification — 14 September 2026

This audit covers the private-site Execution commands, dashboard controls, their
broker adapter, and the expected-exit status publisher. Trading checks use inert
IBKR objects and fake brokers. Production checks read existing broker state;
no live trade, modification, cancellation, or schedule was submitted as a test.

## Reported failure

The SNA limit-order edit from 218 to 171 reached IBKR, but the executor then
emitted two incompatible results. The lifecycle returned through a JSON-printing
`_out` function whose actual return value is an integer. The caller treated it
as a dictionary and failed after the mutation. The agent correctly classified
the malformed response as unknown. The repaired caller collects a result,
persists it, and emits one terminal JSON response. Real subprocess tests exercise
the actual `_out`/CLI boundary, including replay and delivery failure.

Read-only reconciliation matched the exact account, contract, client, order and
permanent ID, quantity 171, zero fills, and unchanged other order terms. The
receipt was marked executed without resubmitting the edit. Recovery publication
must be checked after installing the agent repair.

## Command and control coverage

| Area | Exercised behavior |
| --- | --- |
| Entry brackets | Primary/PA; long/short; stock LMT, STP LMT, MKT, MOO, MOC; supported futures and FX types; native parent/child topology and acknowledgement |
| Position actions | Primary/PA Add, Close with exit adjustment, Close only, full/partial Flatten; quantity/percentage; long/short; exact-account contract qualification |
| Exit handling | Stop, target, time stop and combinations; owner-client cancellation; partial fills; retained coverage; fill during cancellation; position direction changes |
| Modify/Cancel | Exact account/contract/client/order identity; quantity and price changes; all six displayed working-order type layouts; one terminal CLI result and durable replay |
| Options | Calls, puts, debit/credit verticals; scheduled capped-limit selection; quote age; routed market-rule increments; rounded-limit premium cap; schedule persistence, expiry and exactly-once trigger |
| Error paths | Rejections, unacknowledged orders, invalid/nonfinite input, stale source data, unavailable owners, output delivery failure and uncertain transmission |
| Ticket UI | 60 account/instrument/type/side combinations; invalid combinations blocked; side/instrument/month retained; exact-position selection; fractional quantity rejection |
| Async UI | Account changes invalidate sizing responses; old front-month responses cannot overwrite newer requests; schedule cancellation retains its owning account |
| Display | Futures expiry grouping, quoted average cost, contract multipliers in Trade Log, raw vs aggregated fills, unavailable notional explicitly labeled |
| Other controls | Hedge attribution/scenarios/preferences, activity escaping and uncertain results, navigation, expected-exit freshness/error projection |

Unsupported combinations retain their explicit rejection. Options account
authorization is unchanged; the scheduled option ticket remains Primary-only.
Option-position closing and legacy `trim_readd` are not exposed as supported
actions; the supported stock re-add control uses the unified position lifecycle.

## Rendered browser checks

An isolated HTTP fixture serves the actual frontend with synthetic positions,
orders and command responses. It has no broker connection or network client.
The original ticket reset was reproduced in the rendered page. With fresh
candidate assets, SELL/FUT/MES/202612 survives entry → echo → entry. Close
LMT/GTC/outside-RTH settings also survive switching away and back.

Primary quantity 100 → 78 and PA quantity 100 → 80 each required only Modify,
quantity, Save. Each produced one request carrying the exact order identity and
`new_qty`, no additional confirmation dialog, and the new quantity appeared in
the table. Switching accounts while an editor was open closed the old editor
and rendered the selected account's orders. Browser console had no errors.

Authenticated production navigation and data loading were checked for Execution,
Portfolio, Seasonal, Risk, Radar, Focus, Events, Trade Log, Signals, Orders,
Options, Charts, Status, Futures Lab, Entry Lab, Monte Carlo, Fundamentals and
Theo vs Actual. Trade Log Today/PA/SNA filters, options workspace tabs, streamed
R2 chart navigation, SPY seasonal analysis and Macro tab were exercised.
These are navigation/data/control smoke checks, not a claim to have recomputed
every research result. Historical research pages disclose their dated snapshots.

## Expected-exit dependency gap

The producer's after-16:10 JSON serialization failure is reproduced and fixed.
The repaired publisher collects coherent read-only inputs and preserves the full
algorithm scope. It publishes current degraded status when coverage is missing.
Archived receipts do not bridge the 10 September reviewed opening inventory
through the missing September 10 session. Saved execution rows alone cannot
prove completeness. Exit attribution therefore remains unverified until genuine
history coverage or a separately reviewed opening allocation is available.

## Release evidence

Final local/CI counts, tested commit, runtime hashes, cloud build/freshness result
and authenticated post-deployment observations are recorded in the release
handoff. Passing simulations do not prove that IBKR will accept every future
order under every market, account, liquidity or connectivity condition. Unknown
transmission outcomes remain explicitly unverified and are not automatically
retried.
