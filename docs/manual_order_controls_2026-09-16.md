# Manual order controls and exit reconciliation

Manual Cancel and Save on Primary and PA send the selected order instruction
without portfolio-risk, notional, quantity-cap, protection, OCA-balance,
strategy-ownership or option-topology vetoes. The broker decides whether the
requested quantity and prices are valid. The UI has no extra confirmation or
risk/purpose form. A finite numeric payload and exact order address are still
necessary to encode and route the request.

The new `manual_order_actions` module is called only by the executor's explicit
`cancel` and `modify` handlers. Automated trading and position actions retain
their existing policy checks. Authentication, live arming, explicit dry-run,
exact account/contract/placing-client identity, durable command receipts and
broker acknowledgment handling are unchanged.

An unresolved earlier edit does not block a new explicit manual instruction.
The same command ID is never transmitted twice. An uncertain outcome is reported
as unknown rather than success. Pending automatic close/re-add recovery for the
same account/contract is put into attention before sending the manual edit, so
that recovery cannot subsequently undo it. This does not cancel other orders or
disable strategy schedulers. Changes preserve all unedited broker order fields.

## Reconcile position exits

The Reconcile button appears only on position rows with existing closing orders
whose effective remaining coverage differs from the live holding, or whose OCA
siblings have unequal remaining quantities. A linked OCA group counts once, at
its largest remaining quantity; independent exits count separately. Filled and
cancelled orders, other contracts/accounts, entries and children of still-working
entry orders do not contribute. Positions with no exits or already matching exits
have no Reconcile button.

The broker re-reads inventory and open orders on click, then scales the groups
proportionally using largest-remainder rounding. It sets every sibling in a group
to the group's new remaining size, adding back that leg's already-filled quantity
when modifying total quantity. Whole quantities stay whole; fractional positions
use the finest decimal unit present in the holding and group weights. Zero-sized
groups are cancelled. All reductions precede increases; all placing clients are
resolved before changing anything. Prices, schedules, OCA membership and other
order fields remain unchanged. This never creates an entry, close or re-add.

Position changes, new fills, concurrent edits or missing acknowledgments stop the
operation and report the uncertainty. There is no automatic replay or restoration
from a stale plan. A new click after a refresh builds a new plan from the actual
broker state. The same command ID cannot transmit twice. Primary and PA share this
behavior. Manual cancel/modify remain available even when this bounded multi-order
calculation cannot establish an allocation.

## Candidate and rollout

`python -m broker_runtime.prepare_manual_order_actions --source <runtime> --output <new artifact directory>`
checks the reviewed runtime source hashes before producing only
`execute_order.py`, `exec_agent.py`, `manual_order_actions.py` and
`reconcile_position_exits.py`. The first two
are narrowly patched from the running source, not replaced with older repository
versions. Keep generated files under ignored artifacts because the running source
may contain private configuration.

Activation requires backing up those files, rechecking source hashes, installing
the candidate and restarting the execution agent while its command executor is
idle, and adding only `reconcile_exits` to the existing `LIVE_TYPES` list. Account
arming and all existing limits remain unchanged. The frontend is published only through the R2-backed GitHub Actions site
workflow. Rollback restores the backed-up runtime and previous site source;
neither rollback nor cancellation can undo broker fills. Removing manual risk
caps means operator edits can increase exposure or remove exits in either account;
there is no application-defined maximum loss for those edits.

Offline verification covers both accounts, policy-independent cancel and modify,
prior unresolved state, strategy-tagged orders, quantity increases, signed prices,
fractional quantities, exact identity/owner routing, preservation of OCA/time/
transmit fields, duplicate requests, lost acknowledgments and fills racing cancel.
No live order is submitted or changed as a test.
