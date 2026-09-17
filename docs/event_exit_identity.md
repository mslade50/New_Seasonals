# Event auction cycle identity

Event entries and exits now share an explicit entry date. The broker reference uses that stable cycle date rather than the day a retry runs. Reconciliation supports legacy submission-date exit references within a single strategy cycle, while rejecting ambiguous cycles, invalid execution metadata and reversed inventory.

The shared repair covers T1/T2 SPY FOMC, T3/T4 IWM calendar and V2/V4 SVXY strategies. Research schedules, filters, sizes and auction types are unchanged. Synthetic regressions cover all six strategies, full/partial fills, repeated cycles and shared symbols.

## Event-only broker candidate

Use `python broker_runtime/prepare.py --event-only --source <reviewed broker source> --output <new ignored candidate directory>`. Preparation checks the source hash and compiles the candidate; it does not activate it. Install the producer and runner as one reviewed release. Keep private broker evidence, deployment receipts, backups and candidate files under ignored artifacts.

The runner validates trade/ticker/action/auction and explicit Entry_Date/Execution_ID, scopes broker reads to Primary, and writes an exclusive durable claim before submission. Retries, changed quantities, partial fills and ambiguous acknowledgements cannot release that claim automatically. Terminal-looking acknowledgements never trigger automatic cancellation. A fresh deadline/date check applies immediately before the actual broker call, including shared-guard paths that skip callbacks.

## Required legacy migration

An installed `auction_intents/event_migration.json` receipt must have schema `event-migration.v1`, the exact Primary account, policy `hold_all_legacy_cycles`, and a reviewed immutable `through_entry_date` in YYYY-MM-DD form. Every cycle entered on or before that cutoff is held for explicit reconciliation, including completed or unknown legacy attempts that stale state might resurrect. Missing or malformed receipts fail closed. The cutoff is never automatically advanced.

Inspect current broker executions, open orders, stock holdings, canonical state and historic cycle evidence before setting the fence. A staged exit is not proof of a fill, and net symbol holdings do not establish strategy ownership. Do not erase claims or invent fill allocations to manufacture flat inventory. Reversed attributed inventory stops Event staging until separately reconciled.

## Verification and rollout

The focused suite passes 98 tests covering attribution, all six order paths, cross-day duplicate protection, migration boundary/error cases, uncertain outcomes, cancellation isolation and delays across auction cutoffs. An independent contrarian review identified migration, cross-account cancellation and deadline gaps; corrected behavior passed independent adversarial checks. The complete candidate imported using the deployed interpreter and real helper dependencies without running main or contacting the broker.

Before activation, preserve prior source and runtime-marker backups, verify exact candidate hashes, install the legacy fence first, promote only the tested pinned runtime, and keep the local marker/cloud fallback tag aligned. Verify installed hashes and read-only validation afterward. Production installation never implies authority to place a compensating trade or alter historic allocations.
