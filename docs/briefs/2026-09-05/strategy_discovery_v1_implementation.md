# Strategy Discovery V1 — implementation and verification record

Date: 2026-09-05

Branch: `codex/strategy-discovery-v1-20260905`

Scope: offline/file-provider-only research foundation

## Outcome

Implemented a deterministic, fail-closed strategy-discovery processor around
strict local JSON/JSONL contracts. It produces a daily human funnel without
browsing X or enabling any network, email, scheduler, strategy, order, storage,
or broker action.

The implementation enforces the antagonist-approved authority model:

- X inputs are untrusted and `DISCOVERY_ONLY`.
- Source-claimed metrics remain separate from internally validated metrics.
- Automatic lifecycle promotion stops at `RESEARCH_READY`.
- `VALIDATED_RESEARCH` requires a structured reproducible-research artifact.
- `OWNER_REVIEW` requires an explicit recorded human transition from validated
  research.
- Both promotions require a prior journaled run; no first-run artifact or
  same-run artifact-plus-transition shortcut is accepted.
- Validation and owner authority are bound to an exact canonical research-spec
  digest, independently from the broad structural dedupe fingerprint.
- No lifecycle state authorizes capital, strategy mutation, order staging, or
  execution.

## Implemented controls

- strict, unknown-field-rejecting contracts for config, source manifest, X
  item/lineage, claims, proposals, catalog snapshots, validation artifacts,
  owner transitions, candidates, and reports;
- expected-versus-observed coverage with exact source/locator allowlisting, bounded UTC
  windows, source freshness, exact/minimum counts, provider state, cursor
  continuity, and explicit `COMPLETE | PARTIAL | UNKNOWN` outcomes;
- complete-zero versus outage semantics;
- canonical post/reply/quote/repost rules, cross-capture native-content dedupe,
  complete observation provenance, and conflict handling;
- structural fingerprinting across normalized direction, universe, signal,
  entry, and exit, independent of marketing prose, data-vendor wording,
  performance claims, costs, and borrow assumptions; duplicate predicates and
  equivalent numeric spellings cannot evade a catalog match;
- injected, SHA-256-digested strategy-book and dead-end snapshots with
  freshness checks and no strategy/order-module imports;
- prompt-injection quarantine plus signal, bounded-exit, causality/timing,
  cost, market-impact, borrow, data, point-in-time universe/delisting, capacity,
  substantive rationale/falsifier, and investability gates; every proposal in a
  deduplicated group must pass rather than inheriting the primary item's gates;
- explicit condition operator/value grammar, entry order/timing/price-rule
  semantics, bounded numeric magnitudes, and controlled invalid-Unicode/JSON
  failures;
- portfolio-role hypotheses, why-now, variant wedge, numeric research
  assumptions, falsifiers, explicit unknowns, downstream workflow, first
  rejection, and next-research step in the human report;
- deterministic JSON, Markdown, and HTML output; Markdown neutralizes raw HTML,
  HTML escapes untrusted values, and SHADOW mode has a prominent
  non-authoritative banner; human views expose complete validation-artifact
  provenance and state that integrity was verified without executing replay;
- append-only, idempotent, SHA-256 hash-chained journal with strict corruption
  failure, one lock namespace across every writer, safe cursor-anchor state,
  and bounded interprocess locking across verify-plus-append;
- repo-artifact-root-only CLI with `.json`/`.jsonl` input enforcement, UNC and
  collision rejection, immutable run generations, and a last-written,
  journal-bound `latest.json` commit pointer.

## Verification evidence

Focused adversarial suite:

```text
python -m pytest -q tests\test_strategy_discovery.py
120 passed in 1.89s
```

Focused plus adjacent Daily Posts/pitch grammar regression suite:

```text
python -m pytest -q tests\test_strategy_discovery.py tests\test_daily_posts.py tests\test_pitch_grammar.py
262 passed in 3.39s
```

Syntax and patch hygiene:

```text
python -m compileall -q research\strategy_discovery scripts\run_strategy_discovery.py
exit 0

git diff --check
exit 0
```

The expanded adversarial suite covers complete zero versus provider outage, provider
partial state, required-source gaps, stale catalogs, expected/observed count
contradictions, cursor replay and cursor gaps, item conflict removal,
post/thread/quote/repost semantics, cross-source structural dedupe, current-book
and dead-end matches, impossible close/open timing, missing and placeholder
cost/borrow assumptions, prompt injection, source/internal metric separation,
automatic lifecycle ceiling, reproducible-artifact requirements, explicit human
authority, lifecycle persistence, orphan historical state, journal tampering,
cross-process journal writers, mixed-lock rejection, cursor-anchor laundering
and content-ID state binding, prerequisite lifecycle runs,
real artifact path/hash/time/root checks, PIT/survivorship gating, Markdown
remote-content neutralization, HTML escaping, local/UNC/path-collision
enforcement, immutable generation failure/tamper behavior, exact research-spec
authority, multi-spec rejection, strict nested report validation, strict
unknown-field rejection, digest failure, out-of-window items, timestamps after
the reporting boundary, inactive/future catalog records, table-driven
placeholder rationale/falsifier/signal/data/investability controls, ambiguous
operator/value and entry-order semantics, bounded extreme numbers, invalid
Unicode/UTF-8/duplicate JSON, ambiguous intraday timing, and non-X links.

Fixture CLI proof, first run:

```text
strategy discovery SHADOW COMPLETE: 1 candidate(s), 1 research-ready
journal: ...\artifacts\strategy_discovery\example_run_v3\journal.jsonl (3 new event(s))
report: ...\runs\<run_id>\strategy_discovery_report.json
report: ...\runs\<run_id>\strategy_discovery_report.md
report: ...\runs\<run_id>\strategy_discovery_report.html
latest: ...\latest.json
```

Exact replay of the same capture:

```text
strategy discovery SHADOW COMPLETE: 1 candidate(s), 1 research-ready
journal: ...\artifacts\strategy_discovery\example_run_v3\journal.jsonl (0 new event(s))
```

The ignored fixture output visibly separates the author's claimed 58% win rate
from an empty internally validated section and labels the candidate
`RESEARCH_READY`, not validated or owner-approved.

## Exact committed file set

1. `docs/briefs/2026-09-05/strategy_discovery_v1_implementation.md`
2. `docs/strategy_discovery_runbook.md`
3. `research/strategy_discovery/__init__.py`
4. `research/strategy_discovery/contracts.py`
5. `research/strategy_discovery/journal.py`
6. `research/strategy_discovery/pipeline.py`
7. `research/strategy_discovery/render.py`
8. `research/strategy_discovery/examples/config.example.json`
9. `research/strategy_discovery/examples/source_manifest.example.json`
10. `research/strategy_discovery/examples/items.example.jsonl`
11. `research/strategy_discovery/examples/strategy_catalog_snapshot.example.json`
12. `research/strategy_discovery/examples/dead_end_catalog_snapshot.example.json`
13. `scripts/run_strategy_discovery.py`
14. `tests/test_strategy_discovery.py`

## Deliberately unresolved operational decisions

No workaround was implemented for the missing external authority. Before this
can become a real daily bot, the owner must choose and approve:

- official read-only X API access, including any spend/request budget;
- the source accounts/lists/searches and capture cadence;
- the authoritative strategy-book/dead-end exporters and their independent
  verification;
- shadow acceptance window and completeness service levels;
- retention, host, credential, and escalation policies;
- recipients and a separate email delivery mechanism;
- scheduler activation and the named humans permitted to record owner review.

The validation-artifact JSON is an explicit local research authority record; V1
requires a prior `RESEARCH_READY` journal observation, verifies an existing
regular local artifact under the approved root against its SHA-256 and time
boundary, requires its exact research-spec digest, and journals its identity,
but does not execute its reproduction command or independently rerun the research. That
independent replay remains a required human/reviewer control before an owner
should record `OWNER_REVIEW`.

The committed package contains no X adapter, HTTP client, browser control,
SMTP, R2, Sheets, scheduler registration, strategy import, order import, or
trading path.
