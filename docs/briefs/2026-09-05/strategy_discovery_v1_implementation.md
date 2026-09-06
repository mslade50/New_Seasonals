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
- No lifecycle state authorizes capital, strategy mutation, order staging, or
  execution.

## Implemented controls

- strict, unknown-field-rejecting contracts for config, source manifest, X
  item/lineage, claims, proposals, catalog snapshots, validation artifacts,
  owner transitions, candidates, and reports;
- expected-versus-observed coverage with required-source registry, bounded UTC
  windows, source freshness, exact/minimum counts, provider state, cursor
  continuity, and explicit `COMPLETE | PARTIAL | UNKNOWN` outcomes;
- complete-zero versus outage semantics;
- canonical post/reply/quote/repost rules and replay/conflict handling;
- structural fingerprinting across normalized direction, universe, signal,
  entry, and exit, independent of marketing prose, data-vendor wording,
  performance claims, costs, and borrow assumptions;
- injected, SHA-256-digested strategy-book and dead-end snapshots with
  freshness checks and no strategy/order-module imports;
- prompt-injection quarantine plus signal, bounded-exit, causality/timing,
  cost, market-impact, borrow, and data-feasibility gates;
- portfolio-role hypotheses, falsifiers, first rejection, and next-research
  step in the human report;
- deterministic JSON, Markdown, and HTML output; Markdown neutralizes raw HTML,
  HTML escapes untrusted values, and SHADOW mode has a prominent
  non-authoritative banner;
- append-only, idempotent, SHA-256 hash-chained journal with strict corruption
  failure;
- local-only CLI with `.json`/`.jsonl` input enforcement and same-directory
  atomic report replacement.

## Verification evidence

Focused adversarial suite:

```text
python -m pytest -q tests\test_strategy_discovery.py
37 passed in 0.24s
```

Focused plus adjacent Daily Posts/pitch grammar regression suite:

```text
python -m pytest -q tests\test_strategy_discovery.py tests\test_daily_posts.py tests\test_pitch_grammar.py
179 passed in 1.72s
```

Syntax and patch hygiene:

```text
python -m compileall -q research\strategy_discovery scripts\run_strategy_discovery.py
exit 0

git diff --check
exit 0
```

The adversarial suite covers complete zero versus provider outage, provider
partial state, required-source gaps, stale catalogs, expected/observed count
contradictions, cursor replay and cursor gaps, item conflict removal,
post/thread/quote/repost semantics, cross-source structural dedupe, current-book
and dead-end matches, impossible close/open timing, missing and placeholder
cost/borrow assumptions, prompt injection, source/internal metric separation,
automatic lifecycle ceiling, reproducible-artifact requirements, explicit human
authority, lifecycle persistence, orphan historical state, journal tampering,
HTML and Markdown escaping, local-path enforcement, atomic CLI outputs, strict
unknown-field rejection, digest failure, out-of-window items, timestamps after
the reporting boundary, and non-X links.

Fixture CLI proof, first run:

```text
strategy discovery SHADOW COMPLETE: 1 candidate(s), 1 research-ready
journal: ...\artifacts\strategy_discovery\example_run_v3\journal.jsonl (3 new event(s))
report: ...\strategy_discovery_report.json
report: ...\strategy_discovery_report.md
report: ...\strategy_discovery_report.html
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
validates its reproducibility metadata and journals its identity but does not
execute its reproduction command or independently rerun the research. That
independent replay remains a required human/reviewer control before an owner
should record `OWNER_REVIEW`.

The committed package contains no X adapter, HTTP client, browser control,
SMTP, R2, Sheets, scheduler registration, strategy import, order import, or
trading path.
