# OLV sizing-cap repair — September 15, 2026

Status: implemented candidate; independent review and activation pending.

The pushed September 14 fallback computed held notional, but the scanner still
required known reconciled inventory to enter the cap branch. NAV and pending
entry readers also refused that inventory. The installed runtime predated the
fallback. Unit tests of attribution did not establish working production sizing.

The scanner now accepts a separate sizing Capacity containing NAV, held value,
pending-entry reservations, timestamp, source, and known/unknown status. It
does not set TaggedInventory to known or create entry prices, ATRs, ownership
allocations, exit rows or broker orders. A failed inventory bridge remains an
explicit exit limitation while valid capacity can constrain new scan quantities.

When reconciled capacity is unavailable, the conservative fallback counts all
current stock holdings in the candidate's ticker, at broker market value. This
includes other-sleeve and untagged shares: it may tighten the cap but does not
free capacity on the assumption that absent tags prove absent OLV ownership.
It retains actual Primary NAV, remaining OLV parent-order limit reservations,
ETF exemptions, and the existing fail-open policy for unavailable capacity.
The fallback measures absolute broker NET stock exposure, not verified gross
OLV ownership. An offsetting short from another sleeve can hide gross OLV
exposure inside the account net; this cannot be reconstructed without the
attribution ledger. Do not call the fallback a complete inventory repair or
universally conservative across offsetting sleeves.

The 16:05 job saves separate sizing evidence under dated
`ops/olv_capacity/YYYY-MM-DD.json` keys and preserves content-addressed
generations. Capture no longer requires a seed or execution-coverage chain.
It attempts reconciled exit inventory separately and reports its status.
The component's required output is now the sizing capture; success alone does
not certify exit reconciliation. Bookend scans use the required completed
session until the next cash-session open; otherwise a fresh broker observation
must pass validation. The dedicated read-only collector stamps its collection boundaries and
requires a completed current-day execution request. BUY fills during collection
reserve additional value so a fill cannot disappear between copied positions
and remaining orders. This is bounded current-query proof, not an archived
execution-coverage chain. Live freshness is checked after the query completes.
A logged-in collector is still required when obtaining
new broker evidence, but not when reading a valid saved closing observation.

Initial local verification: 195 focused tests passed, including executing the scanner's
actual capacity setup and cap branch with unknown inventory. The regression
case cuts 100 proposed shares at $100 to 50 with $280,000 held, $15,000 pending,
and $600,000 NAV. Missing capture/offline broker, flat accounts, partial fills,
stale or malformed inputs, account mismatch, duplicate positions/orders,
weekends, holidays, half-days, and ETF exemptions are covered. Exit inventory
remains unknown throughout fallback sizing. No live scan or order submission
is used for validation.

Activation must update the tested pinned local runtime and guarded cloud
fallback together, preserve the prior runtime and marker, and verify installed
code plus a read-only broker observation. The next genuine scheduled capture
and scan must be distinguished from a fixture replay or manual read-only probe.

Contrarian review found and prompted corrections for the stale scan-start
clock, fills during collection, two older scanner fixtures, independent capture
failure reporting, and share-class symbol normalization. The corrected code
passed 262 broader tests before the last symbol/clock regressions were added;
updated exact-candidate results are recorded with rollout evidence.
