# Execution runtime verification — 2026-09-14

This audit exercised 193 parameterized order-path cases and the adjacent recovery,
CLI, scheduled-option, and edit suites: **404 passed, 1 skipped** in 55.65 seconds.
These are transport simulations using native `ib_insync==0.9.86` order, contract,
trade, and immutable position objects. No order was sent to a broker.

## Coverage matrix

| Ticket / operation | Verified combinations | Evidence |
| --- | --- | --- |
| Entry bracket | STK: LMT, STP LMT, MKT, MOO, MOC; FUT: LMT, MKT, MOO; CASH: LMT, MKT. Every combination on Primary and PA, BUY and SELL. | `test_execution_all_orders.py`: native parent type/TIF, stop/target/time children, OCA grouping, final transmit release, complete acknowledgement. |
| Unsupported entry modes | FUT MOC/STP LMT, CASH MOO/MOC/STP LMT, standalone STK STP entry. | Reject before any simulated submission. |
| Attach exits | STK/FUT/CASH, Primary/PA, stop only, target only, time only, and all three. | Exact held contract; correct quantities, account, order classes and acknowledgement. Additional no-stop and time-stop suites. |
| Close only | STK/FUT/CASH, Primary/PA, MKT and LMT. | Native close order and exact account/contract; pending acknowledgement and terminal partial-fill cases. |
| Close + resize | STK/FUT/CASH, Primary/PA, MKT and LMT. | Native closes; proportional existing exit adjustment; partial fills, cancellation, recovery, owner failures, OCA allocation and replay suites. |
| Flatten | STK/FUT/CASH, Primary/PA, MKT and LMT. | Cancel-first behavior retained; fresh exact-account inventory; pending-entry fill during cancellation; same conId in two accounts; wrong-account-only holdings; partial restoration acknowledgement; direction flip before restoration. |
| Add / close + re-add | STK, Primary/PA, long/short; MKT Add and DAY LMT re-add. | Native parent and attached children; inherited exits; confirmed-fill quantity only; no late-session re-add; restart does not submit again. FUT/CASH Add/re-add remain unsupported. |
| Modify / Cancel | STK/FUT/CASH, Primary/PA, LMT, STP, STP LMT, scheduled MKT exits. | Exact order/client/permId; quantity reductions; preserved schedule; one JSON terminal result; durable replay and ambiguous-send handling. Increases and OPT/BAG edits have separate integration coverage, described below. |
| Single option / spread | Primary: long call, long put, call/put debit and credit verticals. | Actual OPT/BAG and LMT objects, trusted topology and risk checks; no fractional quantities/ratios or nonfinite prices. |
| Scheduled option | Primary capped LMT with `capped_limit_v1`. | Existing capped-options suite: fresh exact-contract quote, exchange tick bands, premium budget, contract cap, expired/legacy intent rejection, uncertain acknowledgement. Scheduled persistence/cancel/due-once agent tests. |
| Time stops and GTD | Future calendar date, explicit US/Eastern timestamp, invalid/past date and incompatible inherited legs. | Entry/Attach tests plus `test_execution_time_stop_trim.py`. |
| Result delivery / recovery | Actual `_out` prints a single JSON document and returns 0; broken output pipes; unknown sends; repeated command IDs. | Matrix parses the entire stdout buffer. CLI subprocess tests exercise main/result boundaries. Completed and acknowledged pending position-action receipts survive output loss and reconcile without repeating sends. |
| Read helpers | Option, workbench, futures-front and book subprocess timeouts. | Kills and reaps the timed-out child; does not leave abandoned quote subprocesses. |

The integrated edit review adds 69 derivative and scheduled-MKT cases. Scheduled
MKT increases require a fresh exact-contract reference quote. Futures retain
their existing account-specific caps; USD currency pairs use the correct USD
notional basis. OPT/BAG edits derive risk and direction from qualified contracts
and supported topology, preserve signed credit prices, enforce option-specific
caps, and check covered-exit capacity. Unknown or unsupported structures reject
before sending. No additional user-entered edit qualifiers are required.
Frontend intent/account/draft tests and authenticated browser checks supplement
this runtime audit.

## Concrete runtime corrections

- Final placement success now requires Submitted, PreSubmitted, or Filled, with
  a bounded acknowledgement wait. PendingSubmit alone is not success.
- Flatten now resolves the selected account and exact contract before any
  cancellation, preflights every owner, and reads inventory again afterward.
  A pending entry's cancellation-time fill is included in a full close.
- Partial Flatten validates restored exits, caps restoration by confirmed fills,
  and stops if position direction changed. Partial-filled terminal closes and
  unacknowledged restoration produce an unknown result with reconciliation detail.
- Close and Attach use qualified copies of the held contract. Entry/Attach dates
  and option quantities/prices reject malformed or nonfinite inputs before sends.
- A position action failing before its first mutation now persists a terminal
  rejection; it cannot leave a false unresolved-edit blocker. Output delivery
  happens outside the mutation exception handler, preserving durable results.

Production source touched by this audit: `position_actions.py`, plus generated
executor functions `_do_flatten`, `_cancel_via_owners`, `_placement_problem`,
`_do_close_only`, `_do_exit_attach`, `_do_entry_bracket`, `_do_option_spread`,
`_parse_spread_legs`, and new `_execution_deadline`. Generated agent changes are
the four `_fetch_*` timeout helpers. Preparation still requires the exact pinned
source hashes; the original broker runtime is never imported by this harness.

## Reproducibility and limits

`scripts/prepare_execution_test_fixtures.py --source <reviewed-original-runtime>`
checks source hashes, applies the current patcher, and extracts function bodies
only. The checked-in fixtures contain no live imports, environment loading, or
entrypoint. Tests inject fake transport, account configuration, filesystem state,
and guard boundaries. The source-to-fixture AST drift check ran with
`IBKR_REVIEW_SOURCE` pointing to the reviewed original source set.

Older tests were moved from importing the installed broker runtime and reapplying
an obsolete patcher to these current function fixtures. Six expected-failure
markers were replaced with tests of the supported capped-option and position
action contracts. Some older failures were real missing promotions—Flatten's
account lookup and subprocess timeout cleanup—not merely source-version drift.
The single skip in this run is the legacy `eq_order_entry.py` source-preflight
test: that non-dashboard source is absent from the seven reviewed source files.

The before-fix inert reproduction recorded six failures: five ticket handlers
reported success on PendingSubmit, and a Primary Flatten request could select the
same conId held only in PA. Evidence is retained in the task artifact
`artifacts/audit/previous-runtime-regressions.json`; the current matrix covers all
six corrected cases. Independent review also reproduced and rechecked the
restoration direction-flip and lost-output receipt defects.

The targeted run used these files:

```text
test_execution_all_orders.py        test_broker_runtime_fixes.py
test_execution_attach_nostop.py     test_execution_time_stop_trim.py
test_scheduled_options.py           test_unified_position_actions.py
test_execution_runtime_audit.py     test_execution_ops_fixes.py
test_execution_cli_results.py       test_execution_capped_options.py
test_execution_repair_dispatch.py   test_position_action_recovery.py
```

Simulation verifies constructed orders, state transitions, guard arguments,
account isolation, acknowledgement interpretation, and result delivery. It does
not establish broker/exchange acceptance, auction eligibility, live permissions,
liquidity, price fills, or trading-session behavior. Full integrated tests, CI,
runtime installation, cloud site deployment, and authenticated production checks
are separate rollout gates; this document is not a deployment receipt.
