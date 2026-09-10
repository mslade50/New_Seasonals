# Inventory inputs: repair and remaining decisions

**Later September 9 update:** the owner kept the D sale discretionary, the
dividend audit explains the target discrepancy, and the older D tranche has
since exited. The following observations remain historical. Use the
[current cutover record](olv_inventory_cutover_2026-09-09.md) for the five-tranche
candidate, completed repairs, validation, and remaining activation prerequisites.

## Completed

- Owner policy: manual TWS trades are discretionary unless explicitly assigned.
- Added reviewed, exact-execution allocations to the pure inventory adapter.
  This source change is tested but not installed in the automation runtime.
- Added a read-only reconciliation report. It compares broker holdings with
  tagged exit claims, counts OCA siblings once, distinguishes stock/options
  contracts sharing a symbol, and retains unassigned balances.
- Installed only the read-only `book_snapshot.py` upgrade. Primary and PA
  routing remains intact. Live `/book` verification confirmed both account
  queries complete, fresh timestamps, and remaining/filled quantities for
  all 24 Primary orders and 11 PA orders in the observed snapshot.
- 148 targeted tests passed across inventory, manual assignments, snapshot
  preparation, fill harvesting, pending orders and OLV stop/handoff contracts.

## Opening inventory evidence (not yet approved)

| OLV entry reference | Claimed remaining shares | Broker aggregate |
|---|---:|---:|
| D 2026-08-25 | 520 | |
| D 2026-08-31 | 1,739 | D 2,259 |
| SNA 2026-09-03 | 271 | |
| SNA 2026-09-09 | 392 | SNA 663 |
| RTX 2026-09-04 | 358 | |
| RTX 2026-09-09 | 507 | RTX 865 |

The ring contains a 753-share untagged D sale on September 8. Original entry
fills total 693 and 2,319 shares. Assigning reductions of 173 and 580 would
match today's exit orders, but that assignment is a pending owner question.
Do not infer the answer from the fact that the exits match net holdings.
The older D signal log ATR and current target orders disagree; do not infer a
raw ATR from the target or describe the difference as a verified adjustment.
AMZN, FTI, UVXY, futures and option balances remain separate evidence, not
automatically assigned to OLV. The report also identifies working USO entry
brackets, which must not be counted as held inventory.

## Still required before inventory-based activation

1. Resolve the D assignment and review all opening algorithmic inventory.
2. Establish canonical fill continuity from the reviewed cutoff. The live
   `/fills` response still lacks the completeness contract; the existing R2
   status lacks continuous account coverage. A current-session query success
   is not proof of uninterrupted history. No canonical history was rewritten.
3. Reconcile the older frozen ATR/entry metadata, then test the actual cap and
   volume-stop inputs together. The Primary-only exit candidate must not
   replace the shared PA route silently.
4. Promote the reviewed inventory adapter/runtime and coordinated relay changes
   only after the complete input contract is proven. Early-close deadlines
   remain the separately recorded open item.

The optional-overlay fallback is unchanged. No daily scan, trade, email,
order runner, scheduler change, canonical fill upload, Pages rebuild or
inventory seed activation occurred in this task.

## Evidence and rollback

Isolated source branch: `codex/inventory-inputs-20260909`, based on `e9e568f`.
Private evidence is under its ignored `artifacts/inventory-inputs/`: captured
book/ring/canonical inputs, review report, prepared snapshot manifests and
live verification snapshots. These contain private broker data and are not
committed.

Installed snapshot SHA-256:
`43b38e726e6ce09c6acfedd197fb84e9e462c942278b06cb85bcd140bd482798`.
Original snapshot SHA-256:
`6a01d281d38c35352fa310ca6908201b063f9a9d784ad6a903697fd33a95060a`.
Original bytes are retained at the broker directory's
`.runtime_backups/inventory_snapshot_20260909/book_snapshot.py`; two subsequent
backups retain the timestamp precision iterations. Restoring the original file
reverts only this read-only collector update. The running agent launches the
file on each refresh, so no trading process restart was needed.
