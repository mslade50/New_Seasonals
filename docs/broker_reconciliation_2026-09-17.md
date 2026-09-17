# Broker evidence reconciliation

Prepared and tested; activation requires the separate live-runtime approval.

The execution agent previously left interrupted position actions in `attention`
indefinitely. It did not query the broker again, and later manual edits paused
recovery without a path to retire the old workflow. A new Add was consequently
blocked even when the current position and orders were readable and consistent.

The new shared observer reads fresh positions, raw open-order callbacks, completed
orders and executions. The existing pinned `_fresh_open_trades` collector avoids
ib_insync's stale structural fields. A second position/open-order read compares
prices, quantities, parent/OCA structure, timing, TIF, transmit and outside-RTH
fields to reject a moving snapshot. Blank accounts require an exact single-account
mapping; otherwise evidence is ambiguous. It uses
account, contract and permanent order identity; working-order fallback uses the
placing client, order ID and command reference. Missing requests, transitions and unavailable
submission evidence remain explicit uncertainty.

The agent checks stopped workflows and manual edits every 30 seconds. New
position actions check the same evidence before treating an old receipt as a
block. A repeated manual command can recover its receipt without resending.
Normal pending close management retains its existing fill/exit handling.

An interrupted pre-close workflow can be retired because the journal records a
close identity before submission and marks attached additions before sending.
Retirement preserves all current broker changes and reports the old workflow as
stopped, never as a completed trade. Submitted closes, additions and manual edits
require matching broker evidence. Partially completed exit-allocation workflows
are retired without continuing their old plan. No observer places, modifies,
cancels or restores an order, nor does it replay a close or re-add. Original
errors remain in the receipt alongside the resolution and evidence.

Current exit coverage is checked against actual holdings before retiring a
position workflow that managed exits. OCA groups count once and must have
consistent quantities; protection required by the original bracket must remain.
Cancelled addition children do not count as protection. A known execution with
insufficient coverage is explicitly reported as a reconciled broker outcome plus
a current exit discrepancy, remains visible, and never triggers stale restoration.
Explicit manual edits themselves retain their existing unrestricted semantics.

IBKR completed-order callbacks create default zero fill counters in ib_insync;
the observer uses completed-order filled quantity where available, full quantity
for a confirmed Filled status, and explicit unknown quantity otherwise. Contradictory
filled quantities and corrected executions retain an explicit discrepancy. Execution-only
recovery requires complete, unique, uncorrected fills; maximum cumulative quantity
alone cannot establish a fill.

## Evidence and verification

- The original code reproduces the stale block with a readable broker fixture.
- 182 tests passed; one old source-pinning test skipped because its reviewed
  pre-upgrade fixture no longer matches the runtime.
- Five legacy adapter tests fail identically on the unchanged baseline and this
  candidate: they apply an obsolete patcher to the current executor. New tests
  extract the installed close/Add handlers and exercise them without importing
  or running the configured live executor.
- Coverage includes both accounts, new Add after an interrupted close, duplicate
  command IDs, manual changes, exact identities, partial fills, completed orders,
  missing history, pending transitions, failed requests, changing snapshots,
  attached additions, preserved original orders and zero observer mutations.
- Read-only live IBKR preview on September 17 resolved all five stale PA
  workflows: MCHP, ENTG, HXL and SNA were flat with no working orders; RTX held
  100 shares with two working orders. This preview changed no journals or orders.
- Independent contrarian review rejected the first candidate for hidden coverage
  discrepancies, stale/incomplete structural snapshots and blank-account omission.
  A separate adversarial check found execution-correction overstatement. All have
  targeted regression coverage in the revised candidate. The second review found
  two related orphan-exit cases (flat exit-allocation journals lacking a direction
  field, and old recorded exits after a position reversal); these are corrected
  with long/short regression fixtures. Final verification of those fixes pending.

## Runtime activation and rollback

Candidate: `artifacts/broker-reconciliation-candidate-v4/`. Its manifest pins the
current executor, agent and dependencies and hashes the four candidate modules.
Preparation refuses an existing candidate directory or changed live sources.

After explicit activation approval: verify all source hashes again, confirm the
execution child is idle, stop the ExecAgent scheduled task, back up the three
existing modules and position-action journal, install `broker_reconciliation.py`,
`position_actions.py`, `position_action_agent.py`, and `manual_order_actions.py`,
then restart the same scheduled task. No account arming, risk caps, environment
variables, website files, or broker orders are changed. Verify the installed
hashes (including the raw broker reader), agent connection and evidence-bearing terminal receipts for the five
stale workflows. Do not send a trade as a test.

Before activation also check whether a configured Legend reservation guard pins
the runtime source tree. Preserve its protections; if active, include the reviewed
runtime hash update in the activation plan rather than bypassing that check.
The September 17 preflight found no required marker, no reservation-config override
in the launch environment file or user/machine settings, and no default config.

There is no direct order exposure or charge from the observer; installing code
in the live execution runtime nevertheless affects whether future user actions
are admitted. Rollback restores the backed-up three modules and restarts the
agent. Leave the unused new module in place; deletion is unnecessary. Resolved
receipts should retain their verified broker evidence. Only restore individual
journal backups after confirming that no newer command or resolution supersedes
them. Neither code nor journal rollback reverses broker fills.

Remaining limitations: an attempted submission with missing durable broker
identity and no matching execution evidence remains unknown. A working or
partially filled entry is not falsely marked completed. These cases need actual
broker evidence; age or an empty cache cannot resolve them.
