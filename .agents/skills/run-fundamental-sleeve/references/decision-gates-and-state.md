# Decision Gates, State Machine, and Contracts

## Keep state dimensions separate

Track these axes independently:

| Axis | Allowed values |
| --- | --- |
| Research stage | `DISCOVERED`, `ELIGIBLE`, `COVERAGE_GAP`, `BASELINE_READY`, `HYPOTHESIS`, `TRIAGE`, `UNDERWRITE_ACTIVE`, `WAIT_TRIGGER`, `REVIEW_READY`, `PASS`, `RETIRED` |
| Company thesis | `untested`, `strengthening`, `intact`, `watch`, `impaired`, `broken`, `changed`, `retired` |
| Security readiness | `not decision-grade`, `wait for proof`, `wait for price`, `conditional`, `review-ready`, `re-underwrite` |
| User disposition | `none`, `deepen`, `watch`, `pass`, `ready list` |
| Portfolio state | `unavailable`, `not held`, `manually held`, `manually closed` |

Never infer portfolio state from research state. Never infer a capital action from user disposition.

## Enforce allowed research transitions

Use this primary path:

```text
DISCOVERED
  -> ELIGIBLE | COVERAGE_GAP | RETIRED
ELIGIBLE
  -> BASELINE_READY | COVERAGE_GAP
BASELINE_READY
  -> HYPOTHESIS | PASS
HYPOTHESIS
  -> TRIAGE | PASS | WAIT_TRIGGER
TRIAGE
  -> UNDERWRITE_ACTIVE | WAIT_TRIGGER | PASS
UNDERWRITE_ACTIVE
  -> WAIT_TRIGGER | REVIEW_READY | PASS
WAIT_TRIGGER
  -> UNDERWRITE_ACTIVE only when its trigger fires
REVIEW_READY
  -> WAIT_TRIGGER | PASS | REUNDERWRITE through changed evidence
PASS
  -> TRIAGE only when its stored reopening condition fires
```

Represent `REUNDERWRITE` as `security_readiness=re-underwrite` plus the prior research stage until the new work is complete. Apply a `stale` blocker overlay whenever a required source breaches freshness.

Require every transition to carry:

- transition ID and timestamp;
- issuer/security and prior/new state;
- reason code and concise rationale;
- supporting and disconfirming evidence IDs;
- fired trigger ID when applicable;
- source freeze time;
- code, policy, schema, and validator versions;
- agent/run ID;
- authority: deterministic engine, research agent, or McKinley.

Append transitions. Never rewrite history.

## Maintain evidence-first contracts

Preserve immutable raw artifacts and rebuild current views from versioned records.

### Source artifact

Store `source_id`, issuer/security, source class, provider, accession or URL, primary-source flag, filing acceptance time, publication time, retrieval time, reporting period, content hash, immutable path, parser version/status, and superseded source ID.

### Fact observation

Store `fact_id`, source ID, metric, value, unit/currency, period, scope/segment, GAAP or non-GAAP basis, reported or derived status, formula and input fact IDs, confidence, and superseded fact ID.

### Estimate and market snapshots

Store exact freeze times. For estimates, store metric, fiscal period, analyst count, distribution, and 1/7/30/90-day revisions. For market data, separate raw current price and capital structure from adjusted return/trend history.

### Claim and evidence ledger

Store `claim_id`, pillar ID, falsifiable claim, expected timing, linked KPI/model line, status, and confirm/warning/break tests.

Append evidence rows with evidence type (`fact`, `management claim`, `consensus`, `market data`, `assumption`, `inference`, `model output`, or `PM judgment`), source ID, availability time, confirm/disconfirm direction, magnitude, materiality, reliability, freshness, model/valuation/status impact, contradiction group, follow-up, and owner/run ID.

Block decision readiness while a material contradiction remains unresolved.

### Candidate hypothesis

Store issuer/security, business-model lane, idea archetype, why it surfaced, possible variant wedge, denominator risk, first cheap rejection test, next evidence, trigger, research stage, source posture, and implementation readiness.

### Underwrite

Store version, current price/capital-structure as-of, driver tree, three through five pillars, estimate path, market-implied path, valuation cases, catalysts, downside mechanisms, financing/dilution, governance/capital allocation, trend, portfolio fit, opposing case, evidence gaps, readiness gates, next review, and decision log.

### Trigger

Store trigger ID, claim/pillar, event or metric, comparator, threshold, source, earliest/latest date, status, fired time, required response, and expiry/reopen behavior.

Use `fundamental-trigger.v1` in `fundamental/schema/trigger.v1.json`. A trigger
cannot fire without an in-cutoff observation and at least one source ID.
`WATCH` reopens on a fired trigger or genuinely thesis-changing evidence;
`PASS` reopens only on thesis-changing evidence. `CLEAR` removes the override.

Use `fundamental-evidence.v1` for the current evidence-ledger projection and
append decision changes to `research_transitions.jsonl`. Every completed local
build writes `fundamental-run-manifest.v1` with exact input/output digests,
versions, state health, and a pending visual-QA status.

### Portfolio snapshot and proposal

Accept portfolio holdings and NAV only from a dated manual or authorized read-only source. Keep any proposed weight in a separate non-executable record with `no_order=true`, scenario loss, binding constraint, portfolio overlap, and missing inputs.

## Gate `REVIEW_READY`

Require all of these:

1. Verify issuer/security identity and current diluted capital structure.
2. Use a current price and explicit market-data timestamp.
3. Tie latest financials and decision KPIs to primary sources.
4. Normalize the relevant earnings, cash flow, capital intensity, and per-share economics.
5. Explain the causal business model with three through five observable drivers.
6. State a specific variant wedge distinct from a known fact.
7. Quantify what the current price appears to imply.
8. Reconcile guidance, consensus, revisions, and house assumptions, or mark unavailable consensus as decision-blocking when essential.
9. Build archetype-appropriate bear, base, and bull cases with current inputs.
10. Identify the earnings/estimate path and evidence window required for the stock to work.
11. Define the downside mechanism, financing/dilution risk, and survival/liquidity posture.
12. Define dated proof, warning, break, and kill conditions.
13. State trend/timing and the rule it changes.
14. Present the strongest opposing case and disconfirming evidence.
15. Assess liquidity, causal-driver clusters, cross-sleeve overlap, and conditional portfolio fit.
16. Resolve every decision-blocking source gap or contradiction.

Require an explicit validator pass. Never promote from a research score, founder tag, product-circle tag, price move, or management narrative.

## Define wait and pass states precisely

For `WAIT_TRIGGER`, store one or more observable triggers with source, threshold, date/window, and response. Do not re-run unchanged work daily.

For `PASS`, store:

- primary structured reason;
- hard or soft pass classification;
- evidence IDs;
- reopening condition and expiry for a soft pass;
- permanent rationale for a hard pass;
- outcome-audit eligibility.

Permit thesis-changing evidence to reopen a passed name. Do not let routine price noise reopen it.

## Keep user-facing reviews scarce

Surface at most three `REVIEW_READY` records. Hold additional completed work in the research queue until higher-priority decisions clear. Give manually held names with warning or kill events priority over new ideas.

Ask only for a research disposition. Do not create an executable ticket. If portfolio/NAV data are missing, withhold sizing and label portfolio fit conditional.

When a current portfolio exists, cap any research-only proposed weight by the minimum of:

- 4.5% single-name cap;
- 75 basis-point NAV scenario-loss budget divided by bear-case loss percentage;
- sector and causal-driver-cluster headroom;
- liquidity/capacity constraint;
- cross-sleeve overlap constraint;
- trend and security-readiness limit.

Treat 30% sleeve exposure as a hard cap and cash as valid. Never force deployment or ten positions.
