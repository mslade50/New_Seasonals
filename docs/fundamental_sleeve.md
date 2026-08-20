# Fundamental Sleeve — V2 Research Operating System

Status: **research-only**. Live actions are disabled.

Canonical orchestration lives in `.agents/skills/run-fundamental-sleeve/`.
The empirical and practitioner evidence behind the design is summarized in
`docs/fundamental_research_foundations.md`.

## Mandate

Build a long-only, concentrated quality-at-reasonable-expectations sleeve. It
may eventually hold up to ten positions and 30% of total account NAV, subject
to an account-level capital budget shared with the existing slow trend sleeve.

The operating objective is narrower: find the rare security where economic
reality is likely better than the path embedded in price, downside is tolerable,
and observable evidence can resolve the disagreement. The screen only allocates
research effort. It cannot recommend a security, surface a review, size a
position, or change portfolio state.

## Broad-universe research funnel

The sleeve is not a mega-cap screen.  Discovery is intentionally wide and
separated from expensive diligence:

1. **Discovery:** FMP's company screener currently returns roughly 2,000
   actively traded operating companies above $300 million of market value.
2. **Price/liquidity eligibility:** the isolated research price view combines
   `master_prices.parquet` and `overflow_prices.parquet`.  A company needs at
   least 252 adjusted bars, a $3 price, and $5 million of 63-day average dollar
   volume.  Missing prices remain visible as a coverage gap rather than being
   silently discarded.
3. **Balanced statement enrichment:** batches rotate across sector and
   small/mid/large/mega-cap strata.  The FMP subscription does not include its
   bulk statement endpoints, so the queue grows incrementally from four
   per-company screen endpoints instead of repeatedly refreshing only the
   largest companies.
4. **Independent discovery axes:** standard companies receive separate
   business-quality, owner-alignment, valuation-support, fundamental-change,
   and trend observations. A legacy composite remains an audit diagnostic; it
   is not the decision rule.
5. **Specialist lanes:** financials, real estate, biotech/pipeline companies,
   and special situations remain in the broad universe but are not forced
   through an economically inappropriate general-company scorecard.
6. **Deep diligence:** current primary evidence, filed-fact reconciliation,
   price-implied expectations, an archetype-specific operating model, valuation
   cases, catalysts, falsifiers, and an adversarial case are reserved for the
   names that survive the broad funnel.

The broad-universe source of truth is
`data/fundamental/current/broad_universe_latest.parquet`.  Current FMP and SEC
views are cumulative across incremental run dates; immutable dated parts remain
the audit trail.

## Draft controls

These are future guardrails only. The current project produces research ranks
and evidence packets; it does not recommend an allocation, size a position,
stage an order, or send a trade. Any capital decision remains manual.

- Target deployment: 27% of live account NAV.
- Hard sleeve cap: 30% of current market value.
- Combined fundamental/trend slow-sleeve cap: 30% until interaction research
  explicitly authorizes a larger allocation.
- Maximum positions: 10; six to eight is normal; cash is valid.
- Starter: 0.75–1.25%; core: 2.5–3.5%; single-name hard cap: 4.5%.
- Sector cap: 9%; correlated-thesis cluster cap: 10%.
- Strict no-overlap rule between technical and fundamental tickers until
  physical broker positions can be attributed safely or a subaccount is used.
- Trend is a realization and risk governor, never thesis proof. The primary
  context is the 200-day average, its slope, and 12-minus-1 relative strength.
  A 200-week average is optional secular context for cyclicals and turnarounds,
  not a second daily gate.

The machine-readable source of truth is `fundamental/config.py`.

## Source hierarchy

1. SEC filing and filing acceptance timestamp.
2. FMP standardized statements and metrics, retained with raw payload digest.
3. Daily immutable FMP consensus snapshots.
4. Adjusted price cache for recomputed relative trend and liquidity measures.
5. Analyst interpretation and thesis evidence, kept separate from sourced facts.

An SEC package count means only that a filed source is present. It must never be
described as a tie-out. Decision readiness requires a reconciliation ledger that
matches the consequential metric, period, units, accession, amendment state,
and tolerance; that ledger is a pending v2 engine.

Raw source payloads are content-addressed under `data/fundamental/raw/`.
Point-in-time snapshot parts are written once under
`data/fundamental/snapshots/as_of=...`. Derived current views can be rebuilt and
are stored under `data/fundamental/current/`.

An attempt to change an existing snapshot part fails. This is intentional: a
vendor revision must become a new dated snapshot rather than rewriting the
historical research record.

## Environment

- `FMP_API_KEY`: existing FMP credential.
- `FUNDAMENTAL_SEC_USER_AGENT`: SEC-compliant identity with a real contact,
  for example `New Seasonals research name@example.com`.
- Existing `R2_*` variables are optional unless `--upload` is used.

## Local commands

```powershell
python scripts/build_fundamental_universe.py
python scripts/build_overflow_prices.py --no-upload --exclude-today
python scripts/update_fundamentals.py --balanced-batch 75 --bundle-depth screen
python scripts/update_fundamentals.py --tickers AAPL MSFT --bundle-depth deep --with-sec
python scripts/validate_fundamental_underwrites.py
python scripts/build_fundamental_report.py
python scripts/build_fundamental_company_maps.py
```

The manual GitHub workflow first runs `scripts/pull_fundamental_inputs.py`
because the symbol master, price caches, and cumulative current research views
are distributed through R2 rather than committed to Git.  It then rebuilds the
broad discovery universe, enriches a balanced batch, and renders the report.

The report output is `reports/fundamental/fundamental_daily.html`. It contains
no execution route or live-order button. Each A/B research candidate links to a
standalone evidence packet under `reports/fundamental/tearsheets/`; those
packets keep reported facts, derived metrics, consensus, research judgment, and
missing evidence visibly separate.

The reader-facing brief shows no more than three names. It leads with whether
anything needs attention and classifies surfaced names as `QUICK REVIEW`,
`KEEP DIGGING`, or `PASS`. The complete universe, source register, and candidate
table stay collapsed under optional diagnostics; they are research machinery,
not a user review queue.

## Private-site inbox

`site/fundamentals.html` is the normal reader-facing surface. It shows one
primary state: either no review is needed or up to three underwrites have
cleared the `QUICK_REVIEW` bar. At most three unfinished names remain below as
optional active research. The broad candidate queue is deliberately absent;
only aggregate pass reasons, coverage counts, and source vintages appear in
collapsed audit drawers. Pass-reason counts assign one primary reason to each
background company so the categories add cleanly; trend is also shown as a
separate overlapping lens because it can gate a company whose primary issue is
valuation, leverage, dilution, or missing specialist work.

The `DEEPEN`, `WATCH`, and `PASS` buttons alter research priority only. On the
deployed Access-protected site they write a reversible state file to
`fundamental/site_state.json` through `/fundamental-state`; local previews fall
back to browser storage. The daily research workflow pulls that state when it
is available. These controls never allocate capital, stage orders, modify a
portfolio, send messages, or connect to a broker. Repository orchestration now
consumes the state file: `DEEPEN` receives the next bounded diligence slot,
`WATCH` waits for its recorded trigger, and `PASS` remains suppressed until
thesis-changing evidence is explicitly supplied to the reducer.

Fundamental research is also deliberately outside the private-site production
gate. Its R2 inputs and rendered payload are best effort: missing or stale
research makes only the Fundamentals tab unavailable or old, and never blocks
a Portfolio, Seasonal, Risk, or Execution deployment. The core site freshness
and R2-provenance checks remain mandatory for their own inputs.

## Founder-led and circle-of-competence offshoot

`reports/fundamental/company_maps.html` maintains two separate research maps:

- a strict current founder-CEO roster backed by a recent proxy, annual filing,
  or official leadership page; founder-chair-only companies are excluded and a
  mismatch against the current local CEO snapshot is held out for recheck;
- a personalized, mostly consumer-facing product circle. Inclusion requires
  observed customer behavior to illuminate demand, frequency, pricing,
  retention, throughput, unit economics, or brand strength. Direct contact
  with an enterprise tool, exchange, broker, or infrastructure product is not
  enough when the decisive revenue and margin drivers remain opaque. The report
  keeps those false-positive familiar names in a visible exclusion table.

The versioned source records live in `fundamental/reference/`. The generated
support view is `data/fundamental/current/company_maps_latest.json`. Founder
leadership and understandability are research tags only: neither can promote a
screen result to `QUICK REVIEW`, imply attractive valuation, or create an
allocation or order.

## Research routes and decision states

- `HYPOTHESIS_TEST`: independent discovery axes justify testing an expectations gap.
- `WATCH_FOR_CHANGE`: one dimension is interesting, but the setup is not aligned.
- `SPECIALIST_MODEL`: baseline is current; dedicated economics are required.
- `EVIDENCE_GAP`: comparable history is too incomplete for reliable routing.
- `BACKGROUND`: no compelling current setup.
- `REJECT`: outside the eligible universe.

Every screen record remains `NOT_DECISION_GRADE` and
`screen_can_surface_review=false`. Underwriting separately tracks company-thesis
status, security readiness, and the reader decision. A v2 `QUICK_REVIEW` must
pass every deterministic promotion gate in `fundamental/underwrite.py`; a
legacy or incomplete JSON record is rejected.

## Two-key promotion rule

A security reaches the reader only when both keys turn:

1. **Fundamental edge:** a source-backed, falsifiable variant; current security
   bridge; causal operating model; price-implied expectations; two valid
   valuation anchors; acceptable return/downside skew; and explicit financing,
   dilution, red-team, proof, and kill evidence.
2. **Realization edge:** at least two of positive or reversing estimates, an
   observable catalyst, and green trend confirmation.

The authoritative contract is `fundamental/schema/underwrite.v2.json`. The
loader and site both fail closed. Missing evidence, stale prices, unsupported
valuation methods, unresolved source IDs, and unresolved contradictions block
promotion rather than becoming neutral observations.

## Point-in-time rules

- Historical screens use information whose `accepted_at` is no later than the
  decision timestamp.
- Rows without an acceptance timestamp fail closed in historical research.
- Filing period end is never substituted for market availability.
- Estimates are usable historically only from immutable daily snapshots.
- Current-active symbol history is not a valid survivorship-free backtest.
- Backtest claims require delisted securities, corporate actions, realistic
  decision lags, costs, taxes, and fixed portfolio slot constraints.

## Canonical run commands

The daily planner is read-only by default:

```powershell
python scripts/run_fundamental_sleeve.py --as-of YYYY-MM-DD
```

An authorized local research build remains non-publishing and non-executable:

```powershell
python scripts/run_fundamental_sleeve.py --execute --as-of YYYY-MM-DD
```

Add `--refresh` only when the dry run identifies missing or stale baseline
inputs. The command has no upload, deploy, messaging, broker, order, or capital
path. After real browser inspection, attest the exact report digest with
`scripts/record_fundamental_visual_qa.py`.

## Before live capital

The project remains not implementation-ready until all of the following pass:

1. SEC tie-out and source freshness gates.
2. Written thesis and valuation schema populated for each security.
3. Point-in-time and survivorship-safe validation.
4. Live NAV, sector, cluster, and combined-sleeve capital checks.
5. Strict ticker overlap enforcement or separate broker subaccount.
6. Append-only action/order/fill ledger and daily reconciliation.
7. Server-enforced dry-run preview followed by separate live confirmation.
8. Eight to twelve weeks of shadow operation including at least one earnings
   cycle for the tracked candidates.
