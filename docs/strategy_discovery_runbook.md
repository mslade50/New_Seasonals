# Strategy Discovery V1 — offline foundation and activation runbook

The source repair integration adds an optional algorithm-family companion;
see [family-fit scope and commands](strategy_family_fit_mvp_2026-09-06.md).
The V1 report and lifecycle below remain unchanged.

Status: the strict processor remains offline and file-only. The separate,
bounded source collector and worthwhile-only delivery workflow are documented
in [the strategy research pipeline](strategy_research_pipeline.md). This
processor still cannot browse, send email, mutate the strategy book, stage
orders, or touch a broker.

## What one daily output says

Each committed run publishes one immutable generation plus a tamper-evident
append-only journal:

- `runs/<run_id>/strategy_discovery_report.json` is the machine contract.
- `runs/<run_id>/strategy_discovery_report.md` is the compact human review.
- `runs/<run_id>/strategy_discovery_report.html` is the standalone human
  review with all source text HTML-escaped.
- `runs/<run_id>/bundle_manifest.json` binds every immutable file to its
  SHA-256 and byte count.
- `journal.jsonl` records runs, source captures, candidate observations,
  validation artifacts, and explicit owner transitions in a SHA-256 hash chain.
- `latest.json` is replaced last. It points to a complete generation and the
  verified journal head; an unreferenced generation is not a committed run.

The report and run content ID include an explicit processor version (currently
`1.0.6`). Any
content-affecting contract/classification/rendering release must bump it, so a
new implementation cannot collide with an immutable generation produced by an
older one from identical input snapshots.

Development journals written before `1.0.6` do not carry the required
run-bound source events and event-specific payload contracts. They are
incompatible and must not be reused as shadow-acceptance evidence. Preserve
them as development artifacts if needed; begin shadow acceptance with a new,
empty journal rather than rewriting or migrating history in place.

The human report is deliberately ordered as a decision funnel:

1. run mode, authority boundary, and `COMPLETE | PARTIAL | UNKNOWN` status;
2. expected-versus-observed source coverage, windows, and cursor state;
3. strategy-book and dead-end catalog freshness/digest health;
4. candidate count and dispositions;
5. for each candidate: executable structure, incremental portfolio-role
   hypothesis, falsifiers, blocking gate, source-claimed metrics, internally
   validated metrics, compact artifact path/hash/spec/code/data/replay evidence,
   provenance, and the next research action;
6. limitations that prevent over-reading the report.

`COMPLETE` means only that every configured source window was exhaustively
captured and the injected catalogs were current. It does not mean “all of X” or
“all possible strategies.” A complete window with zero items is reported as a
verified zero. A provider outage with zero items is `UNKNOWN`, never a clean day.

## Trust and authority boundary

```text
official read-only X/Crossref collector
           |
           | strict local JSON / JSONL only
           v
source manifest + normalized items ----+
digested strategy-book snapshot --------+--> offline validator --> reports
digested dead-end snapshot --------------+         |
reproducible validation artifact manifest+         +--> append-only journal
explicit HUMAN transition file ----------+
```

X is `DISCOVERY_ONLY`. It may supply a hypothesis or an author-claimed metric;
it cannot supply internally validated edge. Source claims stay under
`source_claimed_metrics` with `evidence_class=SOURCE_CLAIMED`. Only a separately
produced `REPRODUCIBLE_RESEARCH` artifact with code revision, frozen data
digests, reproduction command, methodology, and metrics can populate
`internally_validated_metrics` and advance a candidate to
`VALIDATED_RESEARCH`. The artifact must be an existing regular JSON/JSONL file
beneath the approved validation-artifact root, cannot escape through traversal
or symlinks, and must match its declared SHA-256. Its creation timestamp cannot
be after the run boundary or before the candidate's preregistration.
The report verifies artifact integrity but deliberately does not execute the
recorded replay command; the Markdown and HTML views say that next to the
validated metrics.

The broad structural fingerprint remains the deduplication identity. A separate
canonical `research_spec_digest` binds validation authority to the precise
execution, signal, data/PIT, cost, borrow, capacity, falsifier, unknown, and
investability specification. A candidate observation journals that digest; an
artifact and human transition must carry the same digest. Changing any bound
field drops inherited validation/owner authority. Source narrative such as the
name, thesis, why-now, wedge, or portfolio-role wording remains provenance and
does not manufacture a second validation spec. If one structural group still
contains more than one research spec, it remains `DISCOVERED/NEEDS_SPEC` and no
artifact or owner transition can overwrite that failed consistency gate.

Automation can advance a new, feasible, non-duplicate idea only as far as
`RESEARCH_READY`. Promotion is intentionally multi-run: run 1 journals
`RESEARCH_READY`; a later run may attach a verified artifact and journal
`VALIDATED_RESEARCH`; only another later run may accept an explicit human
transition to `OWNER_REVIEW`. An artifact and transition cannot shortcut these
preregistered states in the same run. `OWNER_REVIEW` requires `actor_type=HUMAN`,
a named actor, reason, timestamp no later than `as_of`, and a prior accepted
validation. Even `OWNER_REVIEW` does not authorize a strategy change, capital
allocation, order staging, or execution. Machine output always carries
`operationally_authoritative=false`.

The journal is tamper-evident, not cryptographically authenticated. Its hash
chain and per-run commitments detect truncation, reordering, or byte edits that
do not also replace the downstream chain and recompute the run identities. They
cannot prove authorship against a local principal who can replace the journal
and recompute every digest. V1 therefore trusts the
filesystem identity allowed to write the approved output root. Production
activation requires OS-level write isolation, access logging, backup/retention,
and an independently chosen signing or append-service design if protection
against an authorized local writer is required. No journal hash is represented
as a signature.

## Input contracts

The example bundle is in
`research/strategy_discovery/examples/`. Unknown fields, missing required
fields, invalid enumerations, non-finite numbers, malformed timestamps, bad
catalog digests, ambiguous lineage, and non-X permalinks fail closed.

### Run config

The config fixes the `as_of`, mode, exact required-source allowlist, source-ID
to locator registry, freshness limits, title, and authority policy. Extra
manifest sources are rejected, missing sources make completeness `UNKNOWN`,
and each present source's locator kind/value must exactly equal its registry
entry. Modes are:

- `DISABLED`: accepts no source captures or items and reports `UNKNOWN`.
- `FIXTURE`: synthetic/offline development data.
- `SHADOW`: real read-only inputs, prominently labeled non-authoritative.
- `LIVE`: an operational research report only. The trading boundary remains
  disabled and X remains discovery-only.

### Source manifest

Every approved account, list, or search has a unique source and capture ID,
UTC capture/window timestamps, provider status, exact or minimum expected item
count, observed count, pagination cursor in/out, and an `exhausted` flag.
Coverage and candidate provenance retain provider name/version/status, approved
locator, capture ID/digest/time/window, item ID, content digest, permalink, and
full parent/quote/repost lineage.

Coverage rules are intentionally asymmetric:

- provider `ERROR`, count contradiction, cursor gap, capture-ID content change,
  conflicting post payload, or out-of-window item => `UNKNOWN`;
- provider `PARTIAL`, stale window, unmet expected count/minimum, or
  non-exhausted cursor => `PARTIAL`;
- provider `OK`, matching counts, fresh window, valid cursor continuity, and
  exhausted pagination => `COMPLETE`, including a real zero.

Replaying the exact same capture is idempotent. Reusing its ID with different
content fails closed. A new capture must continue the latest journaled
`cursor.out`; the first local capture must start with `cursor.in=null` so the
continuity anchor is explicit. Only a `COMPLETE` capture advances the accepted
cursor anchor; `PARTIAL` and `UNKNOWN` captures remain audit observations. The
original accepted-anchor context is persisted for exact replay and included in
the run content ID, so a wrong-cursor capture cannot become complete by being
replayed and identical current files evaluated against different prior anchors
cannot share a run ID.

Every source observation carries the current `run_id`, and its event key binds
the run, source, and capture identities. A `RUN` opens one journal transaction;
all source and candidate child events must reference that current run and
cannot be appended later to an older run. An accepted cursor anchor must name
an exact earlier `COMPLETE` source event from a prior run; a same-run fabricated
anchor or backfilled child fails closed.

### Source items, claims, and X lineage

Items are normalized into `POST`, `REPLY`, `QUOTE`, or `REPOST`:

- posts, replies, and quotes preserve their own post identity;
- replies require thread and parent IDs;
- quotes require the quoted post ID and do not inherit the quoted post's claims;
- reposts canonicalize to the original post, contain no copied text, claims, or
  proposal, and are lineage-only;
- the same native post observed by multiple approved sources/captures
  deduplicates by stable native content while preserving every observation;
  item/capture metadata does not manufacture a conflict;
- genuinely conflicting native content or lineage under one post ID is removed
  from candidacy and makes every involved source `UNKNOWN`.

Every claim is explicitly `SOURCE_CLAIMED`. A proposal is structured rather
than inferred from prose: direction, listed-equity universe, signal conditions,
observation timing, entry, bounded exits, data requirements, numeric costs,
borrow, capacity, point-in-time universe/delisting basis, why-now, variant
wedge, investability conditions, explicit unknowns, downstream workflow,
portfolio-role hypothesis, and falsifiers are all explicit.
Strings containing only common sentinels such as `TBD`, `unknown`, `N/A`,
`none`, or `not applicable` never satisfy a readiness gate. This applies to
every proposal merged into a structural candidate, including nested condition
arrays and every falsifier; a complete primary post cannot launder an
incomplete corroborating post.

### Catalog snapshots

The strategy-book and dead-end catalogs are injected snapshots; the discovery
package never imports `strategy_config`, scanners, backtesters, order modules,
or broker code. Each snapshot carries a SHA-256 digest of its canonical
`records` array. The canonicalization is UTF-8 JSON with sorted keys and compact
separators (the implementation is `sha256_json(records)`).

Each strategy record contains its precomputed structural fingerprint. Each
strategy-book row must be active; inactive rows make the snapshot contract
invalid rather than being mislabeled as active. Each dead-end record also
carries the dated rejection reason, and its decision timestamp cannot be after
the catalog's own point-in-time boundary. A stale catalog makes the run partial
and blocks automatic promotion; a digest mismatch blocks the run.
Snapshot ID, catalog type, canonical records digest, `generated_at`, and
point-in-time `as_of` are all run-ID material. Consequently two otherwise
identical catalogs evaluated as fresh versus stale cannot share a content ID or
immutable report generation.

The upstream catalog exporter is intentionally not included in V1. Before
shadow activation, build and independently verify a read-only exporter that
normalizes the current book and research-negative registry to these contracts
without importing either into this package.

## Candidate gates and deduplication

The structural fingerprint hashes normalized direction, universe, signal,
entry, and exit. It ignores marketing name, prose, claimed performance, data
vendor/field phrasing, costs, and borrow assumptions. Those remain visible and
gated, but they do not make the same executable strategy look new. Two posts
describing the same strategy therefore form one candidate with combined
provenance. Condition order, duplicate identical predicates, and equivalent
integer/float notation also cannot evade catalog deduplication.

The automatic research gate blocks or quarantines:

- instruction-like/prompt-injection payloads;
- missing/placeholder signal fields or values, unsupported operators, or empty
  condition arrays;
- operator/value type mismatches: relational and crossing operators require one
  numeric scalar, `between` requires two increasing numeric bounds, and
  membership arrays must have one homogeneous scalar type;
- no bounded exit;
- final-close data used to enter at that same close;
- open/intraday data used to claim the already-fixed same-session open;
- same-session close execution without at least five minutes of declared lead;
- same-session intraday observation and entry without explicit ordered clocks
  (the V1 schema has no clock fields, so this remains `NEEDS_SPEC`);
- an unknown entry order type, a MOO/LOO or MOC/LOC timing mismatch, a missing
  price rule for a limit/stop order, or an ignored price rule on a market order;
- missing or placeholder commission, slippage, or market-impact assumptions;
- shorts without explicit availability and borrow-fee assumptions;
- missing or placeholder data field/cadence/availability declarations;
- current/static constituents, or missing point-in-time membership,
  delisting-security, and delisting-return controls (an explicit fixed set of
  instruments is the only non-PIT exception);
- missing numeric capacity methodology, why-now, variant wedge, investability
  conditions, explicit unknowns, or downstream research workflow;
- any `PARTIAL`/`UNKNOWN` source coverage or stale/uncertain injected catalog;
- matches to the current strategy book or a recorded dead end.

Passing these gates means only “ready to spend research time.” It is not a
positive expected-return finding.

All accepted numeric values have a finite magnitude bound before canonical
JSON or hashing. Invalid UTF-8, lone Unicode surrogates, non-finite numbers,
extreme integers, and duplicate JSON object keys become controlled
`ContractError` failures rather than tracebacks.

## Run the example

From the repository root:

```powershell
python scripts/run_strategy_discovery.py `
  --config research/strategy_discovery/examples/config.example.json `
  --source-manifest research/strategy_discovery/examples/source_manifest.example.json `
  --items research/strategy_discovery/examples/items.example.jsonl `
  --strategy-catalog research/strategy_discovery/examples/strategy_catalog_snapshot.example.json `
  --dead-end-catalog research/strategy_discovery/examples/dead_end_catalog_snapshot.example.json `
  --output-dir artifacts/strategy_discovery/example_run
```

Optional `--validation-artifacts` accepts a local JSON manifest. Optional
`--owner-transitions` accepts a local JSONL file. Every input must be local and
must end in `.json` or `.jsonl`; URLs and UNC/network paths are rejected. By
default, output must resolve within repository `artifacts/strategy_discovery/`,
and no input or validation artifact may live inside the chosen output
directory. Tests may inject a separate approved local root; the production CLI
has no flag that broadens it.

The CLI takes one cross-process transaction lock before loading the journal and
holds it through report validation, immutable generation publication, journal
append, journal re-verification, and the final `latest.json` replace. A second
writer waits for the first and then reloads the new head; if the lock cannot be
obtained within the bound, it fails closed. Locks are never stolen or guessed
stale. A crashed writer can leave a lock or ignored staging generation; an
operator must inspect both rather than truncating audit state.
Every journal writer uses the same `journal.jsonl.lock` domain, and append calls
accept only a verifiable lease for that exact path. Existing journal symlinks or
Windows reparse points are rejected before read or append.

Within the locked append, event payloads and event keys are validated by event
type, then the entire prospective chain is validated across events before any
byte is written. The transaction order is `RUN`, source observations, any
newly accepted validation artifact, any newly accepted human transition, then
candidate observations. Validation still requires exact-spec `RESEARCH_READY`
from a prior run, and a human transition still requires exact-spec
`VALIDATED_RESEARCH` from another prior run. Higher lifecycle states and
`NEW_RESEARCH_CANDIDATE` must agree; incomplete runs render candidates
`DISCOVERED/NEEDS_COVERAGE` even when older validation exists. Replays may be
idempotent, but a later append cannot backfill a child into an already recorded
run.

Before a run group can seed any later lifecycle, its required-source registry
must reconcile to its source children, `COMPLETE` requires every source child
to be complete, and `PARTIAL` forbids an unknown source child. The journal also
recomputes every candidate-derived summary count from the candidate children.
Validation artifacts, human transitions, and all three authority-bearing
candidate states (`RESEARCH_READY`, `VALIDATED_RESEARCH`, `OWNER_REVIEW`) are
accepted only inside an enabled `COMPLETE` run. Candidate and validation state
is committed to the in-memory authority history only after all sibling/source/
summary checks for that run have passed; an inconsistent run cannot seed the
next transition.

Each `RUN` also carries an `input_material_digest` and a canonical
`transaction_commitment`. The latter covers processor/mode/as-of/completeness,
the required-source registry and summary, every full source-capture child,
every newly attached validation or human transition, and every candidate
observation. Child run IDs and event keys are excluded only to avoid a circular
hash. The run ID is then derived from the input digest plus transaction
commitment, and the journal recomputes the commitment after collecting all
children. Relabeling provider or completeness state, changing a sibling, or
reconciling a forged summary therefore invalidates both the commitment and run
identity. An already-journaled validation/transition may be resubmitted only
for an exact same-run replay; on a genuinely new run, omit it and rely on its
validated historical authority rather than creating an unpersistable duplicate
global event key.

## Official X and Crossref adapters

The separate collector in `scripts/collect_strategy_sources.py` now implements
the original adapter design:

1. Read-only scopes only; no post, like, follow, direct-message, or account-write
   permission. Secrets come from the existing secret manager/environment, never
   source, config, logs, reports, or exception text.
2. A reviewed source registry fixes source IDs and exact account/list/search
   locators. The adapter
   must never allow a post's content to modify queries, cadence, recipients,
   commands, or later workflow state.
3. Bounded UTC windows and exhaustive pagination. Write the items to a staged
   local file first, compute observed counts, and emit the manifest only after
   the capture has a definitive provider result.
4. On rate limit, authentication failure, malformed response, cursor loss, or
   any uncertain page, set provider status `PARTIAL` or `ERROR`. Never turn an
   exception into an empty successful capture.
5. Resolve native X lineage fields into the canonical post/thread/quote/repost
   contract. Do not copy quote/repost text into a new claim, and retain the
   stable post ID and permalink.
6. Treat every field from X as untrusted. The downstream package already
   escapes HTML and gates common instruction payloads; the adapter must not
   execute, browse links from, summarize commands in, or interpolate source
   text into prompts with authority.
7. Keep raw provider receipts under an ignored artifact location with retention
   and access controls chosen before activation. Never promote raw X content to
   versioned strategy evidence.
8. Add fixture recordings for deletion, edit, quote-with-comment, repost,
   multi-post thread, pinned old post, pagination, zero-result, rate-limit, and
   cursor-reset behavior. No real API call belongs in unit tests.

The collector is read-only, secret values stay in environment/.env inputs, the
source registry is fixed in versioned configuration, and every request/item/
page budget is bounded. Native X lineage and SSRN DOI/version metadata cross a
strict contract. Provider errors fail the run rather than becoming zero-result
days. Captures are immutable and cursor advancement requires a later explicit
acknowledgement after discovery commits that exact bundle.

The remaining owner inputs are limited to X service configuration: the approved
account/list/search locators, bearer token, and chosen spend ceiling. X entries
remain disabled until those are supplied. Crossref/SSRN collection, empirical
validation, active-algorithm fit, no-finding silence, durable email delivery,
and the daily Task Scheduler entry point are implemented in
[strategy_research_pipeline.md](strategy_research_pipeline.md). Registering the
task and running the first production email remain explicit cutover actions.

## Operator response to failures

- `UNKNOWN`: do not infer absence or promote new candidates. Repair provider,
  count, cursor, conflict, or time-boundary evidence and rerun with a new
  capture ID where appropriate.
- `PARTIAL`: review observed candidates as leads only, but do not call the daily
  discovery set complete. Repair stale/incomplete sources or catalogs.
- hash-chain, event-contract, or cross-event provenance failure: preserve the
  journal and investigate. Never truncate, rewrite, or regenerate it to make a
  run green. Remember that the chain is tamper-evident rather than a signature;
  investigate the writer identity and filesystem audit trail as well.
- transaction-lock failure: inspect the lock owner, immutable generations,
  `latest.json`, and journal hash chain. Never steal or automatically age out a
  lock; rerun only after an operator establishes the last committed state.
- quarantined item: retain provenance for audit, do not follow embedded links or
  instructions, and manually inspect only in an isolated research context.
- stale/digest-failing catalog: rebuild from the authoritative source and have
  the snapshot independently verified before use.

This file-only V1 processor still has no transport responsibility. The separate
finalizer owns delivery claims and the scheduled runner verifies a confirmed
email or an explicit `NO_EMAIL` decision.
