# Financing-history ledger

Local research support for the frozen 2023–2025 offering pilot. No paid data, trading, scheduling, publishing, or production-state changes.

## September 23, 2026 result

The ledger normalizes 7,046 as-filed financing cash-flow facts from all 90 pilot issuers. These are distinct period/concept/accession/value vintages, not 7,046 independent financing events. Audited funding evidence contains 41 assertions across 24 financing identities, including 13 receipt assertions and revisions.

Reconciliation of the 20 original strength-plus-short-runway observations finds:

- Six with a financing announcement known at the signal, but no reviewed receipt confirmation yet available.
- Four with confirmed receipts after the reported cash balance: two NKTR observations and two BBAI observations. NKTR has $107.5m of approximately reported net proceeds; BBAI has gross-only evidence, so fees are not invented and gross is excluded from the net bridge.
- Ten with incomplete funding audits and no documented intervening funding in the reviewed subset. This is not a finding that no financing occurred, nor clearance for a short.

The count of ten funding flags is a data-quality result, not an offering prediction hit rate. The original pilot and its outcome labels remain untouched. No negative event windows, adjusted runway estimates, incidence statistics or trading returns are certified by this layer.

Report: `artifacts/cash_runway/ledger_20260923_v1/financing_ledger.html`.

## Data contract

`fundamental/financing_ledger.py` separates economic dates from public availability. Every assertion has an issuer CIK, stable funding ID, record ID, stage, status, source URLs and timestamp/date. Stages are announcement, pricing, receipt, capacity and cancellation. Monetary basis is net, estimated net, gross, unspecified or capacity.

Date-only source information is ambiguous during that Eastern Time calendar day and becomes safely known the next day. SEC acceptance establishes a conservative known-by time; it does not claim to be the earliest press-release timestamp. The NKTR June30,2025 launch has a documented 16:02 ET wire timestamp and precedes the pilot's 16:30 ET cutoff.

Each receipt needs an explicit incremental tranche ID, actual receipt interval and public confirmation. Expected closing is never treated as received. A later filing confirming an earlier closing cannot leak into the earlier signal. Revisions of one receipt supersede previous values only once known. The latest net evidence survives a later gross-only repeat; contradictory contemporaneous net evidence fails closed.

Receipts fully inside the reported balance period are not added again. Intervals spanning the balance date remain unallocated. Only documented net and estimated-net receipts wholly after the balance enter the limited `reported_plus_known_net_before_burn` bridge. Gross, unspecified amounts, future tranches and undrawn capacity are excluded. The bridge omits subsequent burn and other flows; `current_cash_estimate` stays null and `fully_reconciled` stays false.

The imported 21-event registry supplies announcement identities/dates only. Its later-priced headline amounts are deliberately omitted from announcement assertions. Partnership-linked funding and ordinary warrant exercises support funding reconciliation but are not promoted to standalone offering outcomes.

SEC cash-flow facts retain accession, acceptance, concept label, units and exact fiscal duration. Alternate concepts, overlapping YTD periods and amended vintages are retained as audit evidence, not summed. Facts missing from standardized companyfacts remain missing; issuer-specific tags and financing footnotes still need review.

## Sources and coverage

The output source index records 50 ledger source URLs, of which 42 have hash-verified archived bytes. Eight inherited issuer/wire sources were not archived; prior web verification is not represented as a successful raw download. All 18 supplemental SEC timestamp assertions were matched to saved submission acceptance timestamps.

The review queue inventories financial filings and 8-Ks for all 90 companies, including accessions not captured by the original keyword searches. A captured accession is not marked audited. The known 54-of-90 usable historical price coverage limitation is unchanged.

Important audit examples:

- KYMR June2025 base receipt was $237.294m net on June30, confirmed in an August11 filing. The July option was a separate approximately $37.6m gross receipt. The aggregate $288.4m gross is not another receipt.
- NKTR July2 close totaled $115m gross, including its option. The approximately $107.5m net amount was known in August; later revisions cannot rewrite the earlier historical information set.
- VYGR's January2023 Neurocrine announcement was conditional. The $39m equity consideration and $136m licensing payment arrived February23, confirmed February24. Contract effectiveness on February21 is a different date.
- GERN receipt confirmations were found later than the January2023 and March/April2024 signals. Actual earlier settlement is not assumed to have been known from a pricing announcement.
- ELDN's November2025 deal ultimately closed at $57.5m gross and approximately $53.6m net, including its option. Its pricing release only expected the next day's closing.

Each assertion links to its own sources in the report and `supplemental_records.json`; exact captured evidence remains under the prior/new artifact directories.

## Reproduce

The builder is offline. It reuses the frozen pilot and reviewed supplemental assertions; it cannot certify or fetch missing event evidence automatically.

```powershell
python scripts/build_financing_ledger.py --pilot-dir artifacts/cash_runway/history_20260923_v1 --output-dir artifacts/cash_runway/ledger_20260923_v1 --supplemental-records artifacts/cash_runway/ledger_20260923_v1/supplemental_records.json
python -m pytest tests/test_financing_ledger.py tests/test_financing_history.py tests/test_financing_opportunity.py tests/test_cash_runway.py -q
```

Use a new artifact directory and versioned assertions for new research rounds. The builder rejects writing into the original pilot directory or outside `artifacts/`. The manifest hashes inputs, implementation files, outputs and the exact report. Rebuilding resets pending visual QA, which must be performed again.

## Verification and next gate

73 focused tests passed, including the 21 new ledger tests. Saved-run integrity checks cover all 3,240 observations, source hashes, tranche identities, availability cutoffs and exclusion of embedded receipts. Desktop/mobile render checks and visual inspection cover the overview, filters, empty states, source links and financing-stage detail.

Next: complete funding-note and no-offering-window reviews for the target and matched comparison companies, then expand a newly frozen historical sample. Resolve delisted/security identity price gaps before making population-level incidence or short-return claims. This ledger is a reusable research layer, not an operational overnight offering-alert service.
