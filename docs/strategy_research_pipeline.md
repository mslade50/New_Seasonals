# Strategy research-to-email pipeline

Status: implemented, tested, and ready for an explicit scheduler cutover. The
repository code does not register the task by itself.

The daily pipeline collects strategy ideas from supported public sources,
converts only executable ideas into strict research specifications, tests them
with frozen repository data, compares them with the configured algorithm book,
and sends an email only when a candidate clears every deterministic gate.
Current positions are never an input.

## Sources

`scripts/collect_strategy_sources.py` implements a bounded read-only collector:

- X uses the official v2 API for configured accounts, lists, or recent-search
  queries. `X_BEARER_TOKEN` is read from the environment. The committed X
  entries remain disabled until an owner list id/query is supplied.
- SSRN coverage uses Crossref's REST metadata for DOI prefix `10.2139`.
  Deposit time captures both new papers and revised metadata. The default two
  queries are enabled and require no paid service. A contact email may be
  supplied through `STRATEGY_RESEARCH_CONTACT_EMAIL`. SSRN item `created_at`
  records this deposit-time availability event so it matches the source cursor
  and capture window; the original paper date remains in
  `source_document.published_at`.

The source registry is `config/strategy_research_sources.json`. It fixes the
sources, global and per-source item ceilings, request limits, maximum pages,
freshness window, and run mode. The disabled X entries are each capped at 50
returned posts per run; that ceiling should be reviewed against the selected X
access tier before either entry is enabled.
Source text never changes the registry.

Collection and cursor acceptance are a two-phase transaction. A capture is
written immutably under
`artifacts/strategy_discovery/source_captures/<sha256>` and recorded as
`pending`. The next run reuses a pending capture without calling a provider.
Only `collect_strategy_sources.py acknowledge` clears it, and that command
first verifies the exact source/capture/content digest in the validated
discovery journal. The acknowledgement receives the normalized item file and
proves that it differs from the immutable raw capture only in `claims` and
`strategy_proposal`. For a complete capture it also requires a terminal email
or `NO_EMAIL` decision bound to the source-bundle digest. Journal coverage,
rather than the provider's optimistic status alone, decides completeness; only
sources journaled `COMPLETE` advance their cursor. A failed or merely assertive
research agent therefore cannot skip unread posts or papers; a crash after SMTP acceptance
replays safely through the delivery receipt and then acknowledges.

Partial coverage is journaled and acknowledged without a quiet-day decision.
Its cursor does not advance, the next run retries from the last complete
high-water mark, and the scheduled runner emits an operational failure alert.

## Discovery and research

The scheduled `/strategy-research` skill owns the interpretation and empirical
work. Its full contract is in
`.claude/skills/strategy-research/SKILL.md`. The important boundaries are:

1. Raw source items are immutable and treated as untrusted.
2. Only bounded, machine-readable rules receive a strategy proposal.
3. A first discovery run journals `RESEARCH_READY` before testing begins.
4. Reproducible artifacts carry frozen data hashes, code revision, replay
   command, methodology, and a required metric set.
5. A later run attaches the exact-spec artifact and creates the algorithm
   family assessment.

`scripts/build_strategy_discovery_catalogs.py` snapshots all 15 configured
primary strategies for direct native-rule deduplication and a structured
dead-end registry when present. `scripts/build_algorithm_family_catalog.py`
covers the broader configured algorithm baseline: primary strategies, Event,
Trend, and separate research/reference ideas. This broad catalog is the
portfolio-fit authority. Cash-gated algorithms stay in the baseline because
positions are irrelevant to strategy duplication and complementarity.

## Email decision

`scripts/finalize_strategy_research.py` validates the final package and writes
`data/strategy_research/latest_decision.json` on every successful run. It
returns success without opening SMTP when no candidate qualifies. It never
sends a stand-down email.

The deterministic gate requires:

- a complete LIVE source/discovery run and journaled reproducible artifact;
- at least 40 costed observations, positive net and recent-era means, gross
  edge at least five times modeled round-trip costs, two of three neighboring
  specifications positive, and bootstrap probability of a nonpositive mean no
  higher than 10%;
- absolute daily correlation to the configured active book no higher than
  0.60, at least 0.05 incremental portfolio Sharpe, marginal capital occupancy
  no higher than 35%, and estimated capacity of at least $750,000;
- every proposed instrument verified liquid and IBKR-tradeable and a
  point-in-time or explicitly fixed universe;
- for an already represented strategy family, the stricter limits of 0.35
  correlation and 0.10 incremental Sharpe.

Qualified emails contain the executable strategy, the repository-data
validation, measured algorithm-book fit, implementation steps, failure modes,
and source links. `research_delivery.deliver_once` creates a durable `SENDING`
claim before SMTP and blocks ambiguous automatic retries.

## Scheduling and operations

`scripts/run_strategy_research.bat` is the unattended entry point. It collects
sources, invokes the pinned Opus/xhigh skill, and verifies that the pending
capture was acknowledged and a new decision was written. Logs live under the
ignored `artifacts/strategy_research_agent/` directory.

A source, agent, or final completion failure sends a separate deduplicated
operational alert through `send_strategy_research_failure_email.py`. This keeps
a broken run distinguishable from a successful `NO_EMAIL` research day.

`scripts/register_strategy_research_task.ps1` registers a daily 12:30 AM ET
Windows task with a three-hour deadline and start-when-available behavior. The
timing keeps normal research clear of the 4:10 AM premarket pipeline. The
registration script is intentionally inert until the owner approves the
production cutover.

Manual non-email verification:

```powershell
python scripts/collect_strategy_sources.py `
  --state artifacts/manual-research/cursors.json `
  --capture-root artifacts/manual-research/captures collect

python scripts/finalize_strategy_research.py `
  --report <report.json> --family-fit <family-fit.json> `
  --package <email-package.json> --html-out artifacts/manual-research/email.html
```

The finalizer sends only when `--send` is explicitly passed and at least one
candidate clears every gate.

## Verification completed during implementation

- The Crossref adapter was exercised against the live public API: two requests
  produced 78 DOI-bound SSRN observations with complete cursor windows.
- Collector, pending/acknowledgement, discovery, family-fit, research gate,
  catalog, scheduling-wire, cloud rank migration, predecessor backup, and R2
  promotion tests pass.
- SMTP, X, Task Scheduler registration, R2 mutation, private-site deployment,
  and any trading action were not exercised during development.
