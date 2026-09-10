# OLV closing inventory and put/call status — September 10, 2026

Status: runtime activated September 10; private-site deployment in progress.

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
- Both scheduled bookend scans read the latest completed session's capture
  without requiring a live Gateway connection. The NYSE calendar handles
  weekends, holidays and half-days. A snapshot cannot be used after the next
  cash-session open or following a change to reviewed attribution.
- The live cap uses recorded Primary broker NAV. The theoretical engine uses
  its own simulated equity; the fixed strategy risk-sizing reference is unchanged.
  A manual intraday scan continues to use a current verified broker observation.
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
and the guarded fallback reference `automation-runtime-2026-09-10.2`.
The private site must deploy through its existing cloud-only R2 workflow.
Rollback preserves the prior runtime commit/marker and disables the new task;
saved snapshot generations remain available. The first real post-close capture
and following-morning consumption remain to be observed after activation.

## Activation evidence

- PR #40 merged at `f0793c0ed075f3010325c3b1c43c2ff966dba9a5` after both
  GitHub checks passed. Local verification passed 269 Python tests and the
  JavaScript status/freshness checks.
- The initial v9 promotion advanced only this fix from its prior version to
  `c06a37ab86d0352abb6fa1b9d5e3da0d11094d1f`, tagged
  `automation-runtime-2026-09-10.1`. The separate scoped candidate passed
  188 targeted tests. Unrelated changes from main were not promoted.
- The final bookend correction is installed at
  `6d214733041fff1c7f942f5e595927c86b855d4b`, tagged
  `automation-runtime-2026-09-10.2`. It makes evening and morning scans use
  the same completed-session capture. All 58 follow-up tests passed on both
  the review and runtime candidates. Premarket, postclose and capture runtime
  validation passed without executing their jobs.
- Runtime-only validation passed for both `premarket` and `inventory-close`.
  `New Seasonals Local v9 - inventory-close` is enabled, with its first run
  scheduled for September 10 at 16:05 ET. The existing eight tasks remain enabled.
- The prior marker is preserved under the runtime's
  `.local/runtime_promotions/closing_inventory_20260910T162049Z/` directory.
- Cloud-only private-site run `34501659994` completed generation but correctly
  stopped before deployment at the risk-study gate. Four historical matches
  were below the five-completed-observation minimum, so all return statistics
  were withheld. The gate incorrectly classified that completed study as absent.
- PR #42 preserves that minimum and emits explicit insufficient-sample status
  and counts. The gate requires coherent counts and complete window coverage;
  missing, malformed, or stale results still fail. The same four-episode fixture
  fails the previous gate and passes the corrected gate without invented stats.
  All 46 focused tests and both GitHub CI jobs passed.
- Replacement cloud-only run `34505682036` builds merged commit
  `b2cb741533c8d7a1b5f7d6bd8226c191f267fd21`. All stages passed and Cloudflare
  production deployment `492c0902-2b18-44b5-a514-8b1694c7c304` serves that commit.
  Authenticated Risk QA confirmed the four-episode insufficient-sample message.
- Authenticated Status QA caught the put/call card reading the output directory
  rather than the R2 input directory. The path now follows the same input-constant
  pattern as other source artifacts. Regression fixtures separate R2 input and
  site output folders; 40 focused tests and JavaScript freshness checks passed.
  PR #43 passed both GitHub checks and merged as
  `c5568577f2843fe2591c62111975be1e6e848195`.
- Final cloud run `34508728737` passed all generation, R2 provenance, freshness,
  and deployment stages. Cloudflare production deployment
  `ecc0cfe5-0647-401b-b439-34a52be62567` serves that exact main commit.
  Authenticated Status QA shows CBOE FRESH through September 9, with equity
  0.67, total 0.88 and index 0.95. These agree with the authoritative R2 cache.
  Portfolio, Seasonal and Execution were reloaded and visually checked on the
  final deployment; Execution is online. This checks rendering/connectivity,
  not order submission or fill behavior.
- No daily scan, trading runner, existing-order modification or email was
  started during activation. The real 16:05 capture and next-morning use are
  future scheduled events, not yet observed successes.
