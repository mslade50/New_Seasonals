# OLV closing inventory and put/call status — September 10, 2026

Status: implemented and locally verified; production activation pending.

## Incident and owner decisions

The September 10 premarket scan queried actual inventory at approximately
04:15 ET. The live collector runs 05:00–21:00; the separate broker refresh
failed and the optional OLV cap was bypassed. The owner confirmed Primary's
Gateway was not logged in overnight. Gateway's launcher records a restart
at 08:32 ET. The old exception handling retained only `RuntimeError`, so the
specific broker failure cannot be reconstructed from the scan log.

Separately, the cap used 50% of the fixed $750,000 strategy-sizing reference.
The observed Primary NAV was about $605,447. Those are $375,000 versus about
$302,724 of permitted entry notional. The user adjusted SNA themselves;
this work does not alter that position or its orders.

The owner directed a 16:05 ET broker capture on T−1 and approved actual Primary
broker NAV for this cap. Scope remains OLV-tagged holdings plus pending OLV
entries, with the existing ETF exemptions. Untagged trades remain discretionary.
The owner declined a policy that holds new entries whenever inventory is unknown.

## Resulting behavior

- The existing scheduler gains a trading-day `inventory-close` pipeline at
  16:05 ET. It queries Primary read-only with client ID 8122, separate from the
  site's collector. It captures reconciled tagged inventory, pending entries,
  NAV, reviewed attribution, and execution-coverage evidence.
- Evidence is stored in R2 under a dated session key, with content-addressed
  generations preserved. A failed query cannot publish a successful capture.
- Morning scans read the immediately preceding completed session's capture
  without requiring a live Gateway connection. The NYSE calendar handles
  weekends, holidays and half-days. A snapshot cannot be used after the next
  cash-session open or following a change to reviewed attribution.
- The live cap uses recorded Primary broker NAV. The theoretical engine uses
  its own simulated equity; the fixed strategy risk-sizing reference is unchanged.
  The post-close scan continues to use a current verified broker observation.
- Missing or invalid capture remains an explicit exception, preserving the
  owner's existing fail-open decision. The cap cannot be guaranteed when its
  required capture is absent. Gateway must be logged in for the 16:05 capture.
- System status gains a CBOE put/call data card with source session, equity,
  total and index ratios, freshness, and the scraping schedule. Data freshness
  does not certify that a particular scheduler run completed.

## Verification and rollout

Focused tests cover SNA-sized cap arithmetic and the actual scanner cap block,
partial/pending reservations, ETF exemptions, missing/invalid NAV, coherent
account identity, failed queries, source age, weekend/holiday boundaries,
review changes, scheduler wiring, and CBOE missing/partial/stale data rendering.
A separate Primary-only read-only query completed successfully with positions,
executions and NAV. No scan, trading runner, order change or email was invoked.

Activation must coordinate the pinned local runtime, the 16:05 scheduled task,
and the guarded fallback reference `automation-runtime-2026-09-10.1`.
The private site must deploy through its existing cloud-only R2 workflow.
Rollback preserves the prior runtime commit/marker and disables the new task;
saved snapshot generations remain available. The first real post-close capture
and following-morning consumption remain to be observed after activation.
