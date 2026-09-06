# Quant controls and research automation plan — 2026-09-05

Status: antagonist-reviewed; repo-local implementation in progress. Live
broker, email, X access, scheduler and production-data activation are excluded
until the owner gates below are closed.

## Executive decision

Proceed now with five bounded changes:

1. change only liquid OVS from 0.5x to the owner-directed 0.7x and independently
   verify the money path;
2. correct and guard OLV's actual sizing/entry semantics without changing its
   live risk policy;
3. build a file-only, non-authoritative expected-flat evaluator and daily
   JSON/Markdown/HTML report;
4. build a file-only, non-authoritative strategy-discovery funnel and daily
   JSON/Markdown/HTML report, ready for a later approved read-only X adapter;
5. close three deterministic green-on-failure bugs that do not require a new
   trading decision.

Do not describe this slice as “top-tier fund standard” while the high-severity
owner decisions and cross-system controls in the final section remain open.

The plan was challenged by an independent antagonist after separate OLV,
repository-wide, and daily-control audits. Its verdict was conditional. Every
blocking condition is part of the contracts below: source completeness,
non-operational shadow banners, immutable/revisioned identities, exact Decimal
quantities, comprehensive account-contract reconciliation, lifecycle authority,
fixed OVS arithmetic, exact OLV semantics, three explicit hygiene fixes, visible
deferrals, and credential rotation before activation.

## 1. OLV: validated behavior

OLV is **not** flat-sized. It is flat-risk only inside one tier × recency rung
× earnings state, before integer shares and caps. Shares are ATR-normalized.

At the current $750,000 reference NAV and GRM 1.5:

| Base state | Effective base bps | 0 prior signals, 0.5x | 1 prior, 0.7x | 2+ prior, 1.0x |
|---|---:|---:|---:|---:|
| Liquid normal | 52.5 | $1,968.75 | $2,756.25 | $3,937.50 |
| Overflow normal | 37.5 | $1,406.25 | $1,968.75 | $2,812.50 |
| Either tier, earnings -10..0 TD | 15.0 | $562.50 | $787.50 | $1,125.00 |

The earnings value replaces the base and still receives the recency multiplier.
Recency counts prior **signals**, not fills or open positions, for that ticker
in 21 ticker sessions. OLV has no active fragility, cycle, rank-mean, open-leg,
or same-day overlay. Quantity is approximately:

```text
floor(target risk dollars / (1.25 × ATR))
```

before applicable ADDV, single-name notional and per-strategy daily caps.

Pivot entry policy (kept by the owner as an appetite/drawdown control):

| Nearest eligible context | Entry action |
|---|---|
| no/high-invalid/low-nearest, high d<=2, or high 3<d<=4 | close - 0.25 ATR |
| nearest high 2<d<=3 | close - 0.50 ATR |
| nearest high 4<d<=5 | close - 0.75 ATR |
| nearest high d>5 | skip |

Pivots are causal 40/40 close pivots; each source expires after more than 252
ticker sessions. The `(3,4]` return to 0.25 ATR is deliberate and tested.

No live OLV risk rule changes in this slice. The audit identified two material
policies requiring owner approval: the 50%-NAV single-name cap currently uses a
modeled/fail-open Portfolio snapshot and does not reserve working GTC orders;
missing stock earnings coverage silently restores full size.

## 2. OVS liquid 0.7x contract

Only `tier_risk_mults['Liquid']` changes from 0.5 to 0.7. This supersedes the
old implementation as an owner risk-appetite decision; it does not rewrite the
registered D9/D12 evidence.

| OVS state at GRM 1.5 | P1 effective bps before caps | P2 effective bps before cap |
|---|---:|---:|
| Liquid normal | 42.0 | 8.4 |
| Liquid non-midterm, rank mean <94 | 29.4 | 5.88 |
| Liquid midterm | 31.5 | 6.3 |
| Overflow normal | 60.0 | 12.0 |

Overflow remains 1.0x. Rank-mean remains 0.7x and is exempt in midterms. The
midterm cycle multiplier remains 0.75x. P1/P2 base bps and fixed P2 aggregate
cap stay unchanged. The multiplier changes staged P2 weights, never cap dollars.

Required proof: carrier isolation, all four arithmetic cells, both gap paths,
binding mixed-tier P2 cap, scan/engine ordering, frozen replay with input hashes
and row reconciliation, full suite, then independent money-path verification.

## 3. Expected-flat control foundation

The control question is signal/lot-specific:

> For every complete, in-scope producer obligation due by the cutoff, does the
> exact account + contract + signal/tranche have zero signed residual, and do
> all attributed signals plus the effective baseline reconcile to the broker's
> aggregate position?

A symbol-level “must be zero” rule is prohibited because legitimate strategies
and discretionary inventory can overlap.

The offline v1 accepts local, immutable inputs only:

- producer receipts, including explicit zero-obligation receipts and sequence
  coverage;
- append-only obligation/revision events with immutable producer event IDs;
- exact account + `conId` contract identity, with only demonstrably unique
  normalized fallbacks;
- exact Decimal fill quantities and explicitly parsed IBKR correction families;
- effective-dated baseline plus append-only adjustments;
- a fresh-looking book fixture and fill/gap evidence for deterministic replay.

It reconciles every account-contract touched by a due obligation, attributed
open signal, baseline inventory, or fill after the watermark. Ambiguous
corrections, contracts, allocation, corporate actions, missing producers,
sequence gaps, stale/missing accounts, fill gaps, or aggregate mismatches are
`UNKNOWN`, never clean. Proven partial exits and reversals are `ALERT`.

Top-level states are `CLEAR`, `ALERT`, `UNKNOWN`, and `NOT_SCHEDULED`, but
`run_mode=DISABLED|FIXTURE|SHADOW|LIVE` and
`operationally_authoritative=false` prevent offline output from presenting a
live operational `CLEAR`. V1 writes atomically and only to a caller-selected
local directory. It never places, cancels, changes, or flattens an order.

Every daily report leads with:

1. state and a prominent non-operational/shadow banner;
2. what changed today;
3. required human action;
4. evidence/source completeness;
5. due obligations and exact residuals;
6. account-contract reconciliation;
7. unknowns and reason codes;
8. code/config/input digests.

Live rollout comes later: all scoped producers write intents, owner signs the
baseline, then 10–20 shadow sessions achieve 100% receipt coverage and exact
reconciliation before read-only broker acquisition or email can be enabled.
There is no automatic remediation phase.

## 4. Strategy-discovery foundation

X is a **lead source, never evidence of edge**. The offline v1 consumes an
owner-supplied source snapshot and coverage manifest. It records platform/post
IDs, canonical thread/quote/repost relationships, content hashes, author/time,
query/list, collection window/cursor and errors. A zero-candidate day can be
`COMPLETE`; an empty file without all configured source receipts cannot.

Candidate processing is deliberately bounded:

```text
DISCOVERED -> SPECIFIED -> TRIAGED -> RESEARCH_READY
```

Automatic processing stops at `RESEARCH_READY`. `VALIDATED_RESEARCH` requires
a referenced reproducible artifact; `OWNER_REVIEW` requires an explicit human
transition. There is no automatic ACTIVE or production state.

Each card separates source-claimed metrics from internally validated metrics
and includes deterministic rule translation, timing/data availability,
universe/direction/entry/exit, costs/borrow/capacity, mechanism, structural
fingerprint, duplicate/dead-end match, strongest counterargument, kill
criteria, portfolio role/overlap hypothesis, unknowns, and the next research
action. “Why it belongs” is phrased as “why it is worth testing” until a
reproducible incremental after-cost portfolio study exists.

V1 receives digested strategy-book and dead-end catalog snapshots rather than
importing production strategy/order code. Stale/missing catalogs reduce source
authority. It has the same explicit run mode, non-authoritative shadow banner,
atomic local artifacts and append-only journal discipline as the expected-flat
control. It has no network, browser, API, email, R2, Sheets, scheduler, broker,
or strategy-config mutation path.

Later activation requires an approved read-only X access method, possible cost
ceiling, source/query universe, retention policy, schedule, recipients,
research universe, model/time budget and shadow-quality review. The preferred
unattended source is the official read-only API if its cost and terms are
approved; the file provider remains the deterministic replay boundary.

## 5. Low-risk code hygiene implemented now

Three high-confidence green-on-failure paths are in scope:

- portfolio report: backtest/chart/export failure returns nonzero and performs
  no subsequent Sheet or email work;
- health receipts: lookup horizon derives safely from allowed age and age uses
  NYSE sessions across weekends/holidays;
- intraday updater: an all-empty acquisition fails before metadata rebuild or
  upload. A nonempty, already-current, zero-added fetch may pass when freshness
  is proven. Partial coverage is reported; no unregistered threshold is added.

## Owner decision and activation register

| Item | Consequence today | Recommendation / decision needed | Activation blocker |
|---|---|---|---|
| Exposed Gmail application credential in external OneDrive script | Credential must be treated as compromised | Revoke/rotate; move replacement to environment/secret storage. Source/backups/history cleanup is separate destructive work. | **All new SMTP activation** |
| OLV single-name cap source | Modeled snapshot can miss broker positions and working orders; lookup failure disables cap | Use fresh broker positions + reserved working OLV orders; choose UNKNOWN behavior | OLV cap hardening |
| OLV missing earnings coverage | Missing stock/calendar data restores full risk | Explicit stock fail-closed/haircut policy; named ETF/index exemptions | OLV earnings hardening |
| Post-close scope | No complete signal-level intent registry exists | Choose accounts, producers, cutoff, recipients, baseline/adjustment owner, correction policy and intent-write failure policy | Live control/email |
| X access and policy | No approved unattended X source | Choose API/browser/export, cost ceiling, sources/queries, retention, cadence, recipients, asset universe and model budget | Daily X collection/email |
| F1 scanner universe completeness | Tiny partial universe can green-run and replace staging | Add pre-write requested/resolved coverage gate and explicit attrition allowlist | Top-tier scan control |
| F2 Sheets replacement | `clear` then fallible `update` can erase sole operational copy | Generation/commit pointer or verified restore protocol | Top-tier operational integrity |
| F4 fill verification outage | Systemic vendor failure can become row-level manual review + green receipt | Separate DATA_UNAVAILABLE, preserve prior state, fail nonzero | Top-tier fill control |
| D3.4 open-leg source | Live 0.8/1.2 branch reads modeled positions and UNKNOWN as zero | Broker-attributed state; choose conservative/blocking UNKNOWN policy | Promote D3.4 with broker truth |
| F6 executor release boundary | Live OneDrive executor lacks a valid Git/CI release identity; legacy OVS fallbacks drift | Put it under Git/CI in place; require stamped policy version and fail missing stamps | Reproducible live release |
| Historical universe | Long-history results use current constituents | Point-in-time membership/delistings; keep explicit model-risk caveat meanwhile | Capital-inference standard |
| Dependencies and legacy schedulers | Builds are not locked; retired runners remain executable | Pin Python/hashed locks and make retired entry points fail-fast under separate tested migration | Reproducible operations |

## Acceptance and handoff

The repo-local slice is complete only when:

- each workstream's focused tests pass;
- the full suite is run and any baseline failure is independently reproduced;
- OVS receives an independent money-path PASS;
- both offline controls demonstrate complete-zero, outage, malicious/ambiguous,
  partial/reversal and deterministic-rerun fixtures as applicable;
- every changed path passes workspace-hygiene allowlisting and `git diff --check`;
- no external writes occurred; and
- the final report separates implemented behavior, measured impact, residual
  risk and exact owner decisions.
