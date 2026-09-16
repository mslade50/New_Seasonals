# Manual cancel and modify

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

## Candidate and rollout

`python -m broker_runtime.prepare_manual_order_actions --source <runtime> --output <new artifact directory>`
checks the reviewed runtime source hashes before producing only
`execute_order.py`, `exec_agent.py` and `manual_order_actions.py`. The first two
are narrowly patched from the running source, not replaced with older repository
versions. Keep generated files under ignored artifacts because the running source
may contain private configuration.

Activation requires backing up those files, rechecking source hashes, installing
the candidate and restarting the execution agent while its command executor is
idle. The frontend is published only through the R2-backed GitHub Actions site
workflow. Rollback restores the backed-up runtime and previous site source;
neither rollback nor cancellation can undo broker fills. Removing manual risk
caps means operator edits can increase exposure or remove exits in either account;
there is no application-defined maximum loss for those edits.

Offline verification covers both accounts, policy-independent cancel and modify,
prior unresolved state, strategy-tagged orders, quantity increases, signed prices,
fractional quantities, exact identity/owner routing, preservation of OCA/time/
transmit fields, duplicate requests, lost acknowledgments and fills racing cancel.
No live order is submitted or changed as a test.
