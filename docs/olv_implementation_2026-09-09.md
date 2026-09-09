# OLV approved target and capacity changes

September 9, 2026. Prepared in branch `codex/olv-parity-review-20260909`.
Implementation was verified locally before rollout. The owner authorized
deployment on September 9. This release promotes the model/scanner source
and regenerates Portfolio through the cloud; the inventory-dependent live cap
remains explicitly unavailable until its inputs are reconciled. Broker exit
candidates remain uninstalled. See the dated deployment record for final status.

## Owner decisions and resulting behavior

The owner approved both review recommendations: retain the submitted entry
limit as the profit-target anchor, and count pending entries toward the existing
per-stock OLV notional budget. Primary remains the account scope.

- The model now targets submitted limit plus 2.5 ATR. A 99.50 limit filled at
  97 still targets 104.50 when ATR is 2. The volume-loss threshold continues
  to use the actual modeled fill, and the accepted exclusion of entry-day
  target credit remains.
- Pending OLV entry orders reserve remaining shares times submitted limit.
  Filled holdings consume their tagged entry notional. New signals consume
  the remaining budget before their eventual fill outcome is known.
  Model reservations convert to filled exposure or release after expiry/exit.
  Live orders reserve until the broker reports fill or cancellation, even
  after a nominal expiry time.
- The existing 50% of configured sizing-capital threshold and ETF exemptions
  remain. At configured capital of $750,000, that threshold is $375,000;
  it is not 50% of live account NLV.
- OLV daily caps floor final shares like the installed broker code and
  calculate PnL from those shares. The original staged risk amount remains
  the daily-cap denominator unless the stock capacity actually clips shares.

The scanner reads the existing authenticated, read-only book endpoint. It
requires exact Primary identity, fresh per-account order observations, valid
remaining/filled quantities, and inventory caught up to the order observation.
Unknown inputs remain explicitly unavailable. Under the owner's existing
fallback policy, valid trades still proceed without the optional overlay.
Consequently this is not a hard exposure guarantee during feed failures.
No manual trading control is gated by this change.

## Production prerequisites found by read-only inspection

The installed collector does not export remaining/filled order quantities or
the per-account order timestamp needed here. A hash-pinned preparer adds those
read-only fields to the reviewed collector source. It also prepares the
existing exact-tranche OLV runner; it never installs files or connects to IBKR.

The actual-inventory adapter is still unverified in production. In addition
to the missing reviewed opening seed identified in the initial review, the
current fills endpoint and canonical R2 status do not provide the completeness
attestation required by the inventory contract. A false gap flag alone is not
proof of complete history.

Read-only broker and signal-log checks found six active Primary OLV tranches.
Their opening metadata is not uniformly reconstructible without reconciliation:
older D signals retain only rounded ATR, current quantities include reductions,
and working targets differ from a direct reconstruction using that log.
The reason for that target difference has not been established. Do not invent
frozen ATR, assume all reductions retain the original entry reference, or
bootstrap actual holdings from the theoretical Portfolio table.

The prepared strict exit runner is Primary-only. The currently installed
runner also processes PA. Installing it over the existing runner would remove
that PA route; this candidate is therefore not an install-ready cutover.
Preserve PA explicitly or obtain a separately scoped account decision.

## Verification and limits

Final local suite: **2,010 passed, 55 skipped, 2 expected failures**, with
three existing warnings. The focused new reservation/feed/preparer suite
passed 34 tests. Source whitespace and workspace hygiene checks passed.
Full output is retained privately in
`artifacts/olv-review/full-tests-final.log`.

Focused checks cover gap improvement, fill-based loss confirmation, no same-day
target credit, simultaneous future fills, unfilled-order expiry, ETF exemptions,
integer cap/PnL behavior, and preservation of staged-risk remainders. Read-only
adapter tests cover partial fills, stale account data, missing quantities,
duplicate identities, contract ambiguity, pending cancellation, and inventory
that has not caught up with the book. Scanner arithmetic is exercised by
extracting its block; the scanner entry point is never invoked.

The broker preparer verifies source hashes before writing a new candidate
directory and compiles every candidate. Its tests execute the exported
remaining-quantity behavior against inert broker fixtures.

The full-suite broker-patcher fixtures require the pre-install source versions
recorded in `broker_runtime/source_hashes.json`. The existing installed
Execution changes correctly fail those older hashes. Verification uses a
private fixture assembled from byte-verified prior installation backups and
unchanged installed files, selected through `IBKR_REVIEW_SOURCE`. This does
not imply the older candidates should replace current installed Execution code.

The model retains unrounded relative levels on adjusted bars to preserve its
existing dividend scale-invariance rule; live submitted prices use pennies.
Intraday fills, partial fills and auction behavior remain model limitations.
Reservations account for the prior per-strategy daily cap; later pooled/net
cap interactions are not a certified exact replay and may reserve conservatively.
The separate fixed-15:59 early-close defect remains open.

## Next bounded step

Complete the existing actual-inventory input path: reconcile and review opening
tranches, supply continuous attested fills and coherent order observations,
and validate known, empty, stale, ambiguous and partial-fill states through
the actual producer/consumer contracts. Then prepare a Primary cutover that
preserves PA, with source hashes, backups and rollback. Obtain the required
financially consequential activation approval immediately before installation.
The approved source deployment preserves the current unknown-inventory fallback;
it does not activate or certify the inventory-dependent cap or stop handoff.
The local producer and cloud fallback must use the same immutable
`automation-runtime-2026-09-09.5` release to preserve the model change on later
scheduled builds. Do not run a daily scan,
place a test trade, or mark the volume-stop pipeline live as a verification shortcut.
