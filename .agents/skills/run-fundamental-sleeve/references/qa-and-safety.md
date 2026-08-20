# QA, Safety, and Learning

## Enforce source hierarchy

Prefer:

1. McKinley's explicit instructions and supplied portfolio/research context.
2. SEC filings and filing acceptance timestamps.
3. Company earnings releases, presentations, transcripts, and official IR material.
4. Immutable FMP standardized facts and estimate snapshots.
5. Current market data and adjusted local price history.
6. High-quality external context only when it materially changes the decision.

Label facts, management claims, consensus, estimates, assumptions, inferences, model outputs, and PM judgments. Cite current or retrieved facts. Do not cite calculations as sourced facts; preserve their formulas and inputs.

## Fail closed on decision-critical freshness

Require current:

- security identity and listing status;
- price and valuation timestamp;
- diluted shares and material capital-structure changes;
- latest filing/earnings evidence;
- company guidance after the latest event;
- estimates when the thesis depends on consensus or revisions;
- manual portfolio/NAV snapshot before portfolio sizing.

Keep the broad baseline running when a noncritical source is stale, but block `REVIEW_READY` when the missing source owns the conclusion. State the exact limitation.

## Run quantitative integrity checks

Check:

- ticker, share class, ADR ratio, exchange, currency, and issuer/CIK mapping;
- fiscal versus calendar periods and pre/post-print estimates;
- current diluted shares, buybacks, issuance, convertibles, M&A, and discontinued operations;
- raw current-price versus adjusted historical-price usage;
- enterprise-value bridge and sign conventions;
- FCF calculation, working-capital effects, capitalized costs, SBC, and recurring add-backs;
- sector-appropriate denominators and valuation methods;
- bear/base/bull arithmetic, implied expectations, and return horizons;
- duplicate sources, revised facts, restatements, and unresolved contradictions;
- source availability time for point-in-time claims.

## Red-team every promoted name

Build the strongest coherent case that the market is right. Use at least one material disconfirming fact or explicitly state that it remains missing. Ask:

- Does the screen capitalize peak or trough earnings incorrectly?
- Does company growth reach each diluted share?
- Does cash flow reflect underinvestment or temporary working capital?
- Has the price already moved more than the estimate change?
- Does the catalyst change value or only attention?
- Can financing, dilution, regulation, competition, or governance prevent realization?
- Does trend reflect a structural issue the model has not captured?
- Is the apparent edge merely stale consensus or a known fact?

Demote to `WAIT_TRIGGER` or `PASS` when the opposing case remains unresolved.

## Test behavior and safety

Run relevant existing tests, and add coverage when changing implementation. Protect these invariants:

- A screen rank cannot create `QUICK REVIEW`.
- Specialist lanes use their own frameworks and can graduate only through them.
- Missing or stale critical evidence blocks promotion.
- State transitions are valid, append-only, and idempotent.
- `WATCH` fires only from its trigger; `PASS` reopens only from its condition.
- At most three names appear in the reader-facing inbox.
- Background ticker queues remain behind aggregate audit counts.
- Missing portfolio context never renders as actual zero exposure.
- Research controls never create orders, allocations, messages, uploads, or broker mutations.
- Fundamental failures never contaminate Portfolio, Seasonal, Risk, or Execution production data.
- Immutable snapshots reject historical rewrites.
- Point-in-time tests use decision-available timestamps and survivorship-safe universes.

Maintain adversarial fixtures for peak-cycle FCF, SBC-heavy software, a bank scored on generic FCF, a REIT scored on GAAP earnings, pre-revenue biotech, a miner valued only at spot, acquisition roll-ups, working-capital retailers, recent spins, restatements, and serial dilution.

## Inspect reader-facing artifacts

Render changed HTML with a local headless browser. Inspect:

- the opening viewport and primary conclusion;
- all review cards and exact requested decisions;
- valuation, downside, proof, and kill fields;
- source timestamps and stale/conflict badges;
- mobile/narrow layout when frontend code changed;
- aggregate pass/freshness drawers;
- absence of execution or live-action controls.

Store screenshots and browser profiles under `artifacts/`. Never use a local rendering as proof of production freshness.

## Keep the process calibrated

Append every surfaced hypothesis before outcomes are known. Measure by archetype:

- screen-to-underwrite and underwrite-to-review conversion;
- 3/6/12/24-month absolute and sector-relative return;
- maximum drawdown and time to proof/kill;
- KPI, estimate, and valuation forecast error;
- false-positive cause;
- pass false negatives from a small random audit sample;
- selection, valuation, timing, and conditional-sizing contribution.

Version policy changes. Use point-in-time forward evidence and adequate independent samples before changing thresholds. Do not optimize rules on a handful of successful anecdotes.

## Complete only after every check passes

Before handing off, verify:

- source coverage and gaps are truthful;
- each surfaced claim maps to evidence or a labeled judgment;
- decision gates and output caps pass;
- tests pass or failures are reported;
- changed HTML is visually inspected;
- no unrelated files changed;
- no prohibited external or capital action occurred;
- the user can understand the exact decision in one reading.
