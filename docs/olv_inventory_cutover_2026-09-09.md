# Primary OLV inventory cutover

Status: implementation and deployment candidate prepared; **not activated**.
This continues priority 3. Priority 4 remains queued.

## Owner decisions and reconciled inputs

The September 8 D sale remains discretionary. No execution allocation was
created for it. The proposed starting snapshot describes the currently held
shares claimed by exact tagged exits; it does not rewrite historical trades.
Future unassigned discretionary reductions that leave algorithm quantities
above broker holdings require reconciliation and disable inventory overlays.

The read-only Primary query at September 9, 21:09:11 UTC reconciled:

| Entry | Remaining shares | Actual entry price | Frozen ATR | Time exit date |
| --- | ---: | ---: | ---: | --- |
| D August 31 | 1,739 | 64.853523 | 0.95 | September 15 |
| SNA September 3 | 271 | 379.14 | 5.792215 | September 18 |
| SNA September 9 | 392 | 377.87 | 5.622853 | September 23 |
| RTX September 4 | 358 | 199.36 | 4.399643 | September 21 |
| RTX September 9 | 507 | 197.72 | 4.346430 | September 23 |

D's older 520-share tranche exited through its existing scheduled order at
15:59 ET during this review. The five observed fills total 520 shares. This
task did not cause that trade. Other holdings remain outside these OLV claims;
USO pending entries belong to OVS and are excluded from the OLV reservation.

The D target discrepancy is explained by the existing September 3 dividend
adjustment audit: the $0.6675 distribution changed 69.18 to 68.52 and 67.69
to 67.03. Frozen ATR is supported by the saved signal and staging inputs;
it was not inferred from an adjusted target. The live loss rule remains based
on actual entry and raw bars.

## Implemented behavior

- Local and backup scanners read the shared reviewed R2 seed at
  `ops/tagged_inventory_seed.json`. An explicit local seed remains available
  for validation. Missing inputs remain unknown, never modeled holdings.
- Fresh executions and pending orders come from the same completed broker
  observation. Old seeds require digest-verified, overlapping canonical fill
  coverage. The harvester retains proven coverage as the live ring expires.
- The collector attests only the execution interval supported by its reviewed
  TWS settings. The relay extends overlapping intervals; it cannot turn a
  current-session query or a list of old trades into proof of continuity.
- New OLV entries retain their original ATR, contract and actual time deadline,
  matched to staged inputs and broker exits, for later volume-stop decisions.
- Primary exit proposals use `OLV_Exits_Primary` and the strict exact-tranche
  consumer. The existing PA consumer keeps its prior table and behavior.
  The shared scheduled entry point invokes both independently.
- Primary OLV entry expiry and time exits use the NYSE session close minus
  one minute, including half-days. Existing orders are not retimed by preparing
  or installing this code; new staging uses the corrected dates and times.
- The backup scanner receives its existing read token and calendar dependency.
  Unknown inventory still permits valid scans without the optional overlay;
  the 50% per-stock cap is not guaranteed during source outages.

## Validation

The current five tranches passed the actual inventory reader with a completed
read-only broker query and the proposed seed. The relay response contract was
assembled from that query for local validation; this is **not** a claim that
the undeployed relay has been verified live.

The isolated volume-stop staging function read September 9 raw OHLCV and wrote
only to a fake Primary sheet. It produced zero proposals: D 65.10, SNA 377.42
and RTX 197.55 remained above their older tranches' loss levels. The two new
entries arm on the following session. Pending OLV entry exposure was zero.
No daily scan, live sheet write, order runner, test trade or email was invoked.

Regression coverage includes partial fills, corrected executions, conflicting
manual inventory, missing/old coverage, canonical digest mismatch, interval
gaps, new-entry metadata, OCA identity, pending cancellations, exact exits,
missed auctions, holiday/half-day timing, and preservation of PA functions.
The full JavaScript suite passed all 26 files. The final Python run passed
2,057 tests, with 55 skips, two expected failures and three existing warnings.
It used hash-verified historical sources for legacy broker-patcher fixtures;
the new production candidate was also checked against current installed files.
Repository CI results are recorded separately in the release handoff.

## Remaining prerequisites and activation sequence

1. Set the active TWS Trade Log's history to seven days and verify the saved
   setting plus a successful API query. The inspected setting is one day;
   today's direct query returned nine executions. The site's accumulated log
   preserves earlier collected trades but does not recover an uncollected day.
   Do not attest seven-day coverage before this step succeeds.
2. Refresh and approve the opening snapshot immediately before activation;
   reconcile any intervening fills. Keep the D sale discretionary.
3. Obtain explicit approval for the coordinated financial activation. The
   candidate enables inventory-dependent entry sizing and future automatic
   OLV exits in Primary (about $535,000 of current entry notional), while
   preserving PA. No test order or daily scan is part of the cutover.
4. Back up exact broker files, seed/status objects, current runtime markers,
   and the old exit journal. Install the eight hash-verified candidate files;
   preserve the old journal and initialize the distinct Primary journal from
   reviewed existing receipts. Create the distinct Primary exit table without
   changing PA's table. Publish the approved seed conditionally, preserving any
   prior seed as a versioned review artifact.
5. Deploy the relay, promote the tested immutable local/fallback runtime, and
   install the reviewed history policy. Verify source hashes and fresh live
   coverage, inventory and pending-order reads. Exercise the stop producer
   with inert writes and consumer with inert brokers. Do not invoke the scan
   or live exit runner as a shortcut to validation.

The prepared candidate is under ignored
`artifacts/inventory-inputs/priority3-candidate-v2/`; its manifest records source
and candidate hashes. Private captured evidence is under `artifacts/priority3/`,
including `opening-seed-210912.json` and `collector-probe-210911.json`.
No account identifiers, tokens or private evidence are committed.

Rollback restores saved broker bytes and runtime markers, switches the relay
to its preceding version, and restores the prior reviewed seed pointer. Keep
new receipts for reconciliation. Rollback cannot undo a fill already executed;
existing live orders must be reviewed separately if activation has run.

The expected-exit monitor can now use the shared seed, but this release does
not claim a new scheduled monitor or certify the site's existing warning.
PA's route is preserved, not newly certified end to end.
