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
