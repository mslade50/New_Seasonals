# Prepared execution and operations repairs

This package prepares a separate candidate from the exact source hashes in `source_hashes.json`. It does not install, arm, connect, submit orders, change scheduler tasks, or modify the OneDrive source.

Run `python broker_runtime/prepare.py --source <reviewed checkout> --output <new ignored artifact directory>`. Preparation fails before writing any candidate when a reviewed source hash changes. Candidate files retain the source checkout's private configuration: keep them in ignored artifacts, never commit them. The package itself contains no copied credential literals.

Prepared changes cover exact-account flattening; close recovery capped to actual holdings minus other working closes and confirmed fills; all-owner preflight; exact cancel/modify; independently attached additions; fill-gated trim/re-add; durable cross-day auction claims; explicit per-account execution-history attestation; helper kill/reap; truthful entry/batch exit codes.

Submission acknowledgements are not fills. Partial or ambiguous mutations remain unknown and require broker reconciliation before retry. A failed bracket release is never reported as a clean rejection. The existing arming controls and scheduled-option execution gate remain in place. No snapshot-age gate disables position controls: the executor resolves the current exact broker identity.

Missed auctions retain their next-auction intent. A durable claim is written before submission and survives date changes; a prior uncertain or submitted attempt is held for reconciliation rather than duplicated. Existing legacy auction rows without a stable entry date and earlier fills must be reconciled before promotion. No market-order fallback is introduced.

## Verification and promotion prerequisites

137 targeted tests passed together; the additional lagging-position recovery assertion passes in the subsequent 15-test broker subset. Tests exercise real patched functions by AST extraction only, fake brokers, local candidate compilation, canonical history CAS, corrections, partial closes, account mismatches, unknown outcomes, auction claims, supervisor deadlines, and producer health. The deployed modules are never imported. Tests requiring reviewed external source skip when that source is unavailable; portable lifecycle helper tests still run.

Promotion remains an operational step requiring the exact candidate review, compatible root broker/UI changes, the shared `sheets_io.py` helper, explicit account mapping on multi-account endpoints, and current source hashes. Native IB bracket activation and owner-client coexistence need paper/TWS verification before enabling repaired generic controls. No such verification or live activation has happened in this task.

Canonical fill initialization requires a confirmed missing object plus an explicit initialization flag. Legacy Trend inventory requires a reviewed seed/bootstrap; targets and zero reported rows cannot supply it. The new sleeve consumer accepts only a complete Primary receipt and matching canonical fill-file digest. Data outside verified history is an exception, not inferred inventory.

## Actual Primary OLV exit handoff

The candidate now includes hash-pinned `olv_exit_moo.py` and `olv_contract.py`. Required columns are `Symbol`, `Quantity`, `Time_Exit_Date`, `Execute_On`, `Strategy_Ref`, `account_key`, `broker_account`, `con_id`, `tranche_id`, `ref_date`, and `entry_order_ref`. The original entry reference must come from the reviewed seed or actual opening fill. Optional `source_time_client_id`, `source_time_order_id`, and `source_time_perm_id` narrow the match further. Missing metadata never falls back to the nearest time leg. Two staged tranches cannot claim one account/contract/entry-ref/time-date bracket.

Only Primary is connected. Past `Execute_On` remains eligible for the next opening auction; after the cutoff it raises `CRITICAL_MISSED_AUCTION_PENDING` before broker connection or cancellation. A cutoff crossed during cancellation attempts one verified re-arm of the original future time exit and preserves the pending auction exception. A missing/past time deadline is not converted into a market order. There is no MKT/DAY substitute.

The remaining live bracket must agree with the actual staged tranche quantity. Effective bracket executions are read before and after cancellation; new fills reduce the owned remainder even when another strategy keeps aggregate holdings long. Unreadable or conflicting positions stop the new sell and surface an immediate exception. Unknown delivery never triggers a replacement or re-arm. A bracketless incomplete obligation requires reconciliation, rather than guessing ownership from a journal or net symbol holdings. Journal records survive date changes.

The follow-up verification ran 63 tests across OLV, monitor, tagged inventory and broker lifecycle. Fake brokers exercise patched execution functions, including an 80-share owned tranche reduced to 70 after 10 new cancel-window fills while 180 net shares remain. Candidate preparation verified all 8 source hashes and compiled every Python candidate. Native OPG handling, bracket callbacks, TWS/client ownership and the new staging producer still require coordinated paper/runtime verification before promotion.

## Entry journal and retired auction fallbacks

The prepared `eq_order_entry.py` now holds a native OS lock for its complete run. Missing, corrupt or duplicated entry history fails before connection; it never becomes an empty deduplication set. `entry_journal.py` writes each wire intent atomically before the broker call, retains prior dates, binds exact Primary/account/conId, and marks a bracket acknowledged only after its parent and every linked child have accepted statuses and distinct durable order identities. Parent placement alone remains unresolved. A later run preserves that claim and reports an unresolved duplicate instead of re-entering or reporting a successful skip. No automatic cancel/rebuild is attempted after partial assembly or uncertain transmission.

The obsolete naked-MOO CSV routes for Trend and hand-staged OLV are explicit pending-auction exceptions. They cannot silently become MKT/DAY orders. The dedicated Event/Trend/OLV runners and compatible staging producer must be installed together during the reviewed runtime rollout so those obligations continue through their intended auction route. Historical journal reconciliation is a rollout prerequisite. The helper's separate `--initialize-reviewed-empty` command only creates a missing file, fails if it exists, and is never called automatically or by this repair.

The final unnumbered operations repair also strengthens `OutputValidator`: it streams SHA-256 over the local artifact and the exact R2 HEAD generation using conditional GET. Equal byte lengths with different content fail validation; concurrent replacement or unreadable content is an exception. Successful component receipts retain each object's key, size, digest and ETag in `artifact_evidence`. Producer-specific health remains separate from completed side effects and deduplication.
