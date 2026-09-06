# Strategy Discovery V1 — offline foundation and activation runbook

Status: implemented as an offline, research-only file processor. It does not
browse X, call an API, use a browser, send email, schedule itself, mutate the
strategy book, stage orders, or touch a broker. Those boundaries are deliberate.

## What one daily output says

Each run writes the same three deterministic views plus a tamper-evident
append-only journal:

- `strategy_discovery_report.json` is the authoritative machine contract.
- `strategy_discovery_report.md` is the compact human review.
- `strategy_discovery_report.html` is the standalone human review with all
  source text HTML-escaped.
- `journal.jsonl` records runs, source captures, candidate observations,
  validation artifacts, and explicit owner transitions in a SHA-256 hash chain.

The human report is deliberately ordered as a decision funnel:

1. run mode, authority boundary, and `COMPLETE | PARTIAL | UNKNOWN` status;
2. expected-versus-observed source coverage, windows, and cursor state;
3. strategy-book and dead-end catalog freshness/digest health;
4. candidate count and dispositions;
5. for each candidate: executable structure, incremental portfolio-role
   hypothesis, falsifiers, blocking gate, source-claimed metrics, internally
   validated metrics, provenance, and the next research action;
6. limitations that prevent over-reading the report.

`COMPLETE` means only that every configured source window was exhaustively
captured and the injected catalogs were current. It does not mean “all of X” or
“all possible strategies.” A complete window with zero items is reported as a
verified zero. A provider outage with zero items is `UNKNOWN`, never a clean day.

## Trust and authority boundary

```text
future official read-only X adapter
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
`VALIDATED_RESEARCH`.

Automation can advance a new, feasible, non-duplicate idea only as far as
`RESEARCH_READY`. `OWNER_REVIEW` requires an explicit journaled transition with
`actor_type=HUMAN`, a named actor, reason, timestamp, and prior
`VALIDATED_RESEARCH` state. Even `OWNER_REVIEW` does not authorize a strategy
change, capital allocation, order staging, or execution.

## Input contracts

The example bundle is in
`research/strategy_discovery/examples/`. Unknown fields, missing required
fields, invalid enumerations, non-finite numbers, malformed timestamps, bad
catalog digests, ambiguous lineage, and non-X permalinks fail closed.

### Run config

The config fixes the `as_of`, mode, required sources, freshness limits, title,
and authority policy. Modes are:

- `DISABLED`: accepts no source captures or items and reports `UNKNOWN`.
- `FIXTURE`: synthetic/offline development data.
- `SHADOW`: real read-only inputs, prominently labeled non-authoritative.
- `LIVE`: an operational research report only. The trading boundary remains
  disabled and X remains discovery-only.

### Source manifest

Every configured account, list, or search has a unique source and capture ID,
UTC capture/window timestamps, provider status, exact or minimum expected item
count, observed count, pagination cursor in/out, and an `exhausted` flag.

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
continuity anchor is explicit.

### Source items, claims, and X lineage

Items are normalized into `POST`, `REPLY`, `QUOTE`, or `REPOST`:

- posts, replies, and quotes preserve their own post identity;
- replies require thread and parent IDs;
- quotes require the quoted post ID and do not inherit the quoted post's claims;
- reposts canonicalize to the original post, contain no copied text, claims, or
  proposal, and are lineage-only;
- identical post replays deduplicate; conflicting payloads with one post ID are
  removed from candidacy and make source completeness unknown.

Every claim is explicitly `SOURCE_CLAIMED`. A proposal is structured rather
than inferred from prose: direction, listed-equity universe, signal conditions,
observation timing, entry, bounded exits, data requirements, costs, borrow,
portfolio-role hypothesis, and falsifiers are all explicit.

### Catalog snapshots

The strategy-book and dead-end catalogs are injected snapshots; the discovery
package never imports `strategy_config`, scanners, backtesters, order modules,
or broker code. Each snapshot carries a SHA-256 digest of its canonical
`records` array. The canonicalization is UTF-8 JSON with sorted keys and compact
separators (the implementation is `sha256_json(records)`).

Each strategy record contains its precomputed structural fingerprint. Each
dead-end record also carries the dated rejection reason. A stale catalog makes
the run partial; a digest mismatch blocks the run.

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
provenance.

The automatic research gate blocks or quarantines:

- instruction-like/prompt-injection payloads;
- missing structured signal conditions;
- no bounded exit;
- final-close data used to enter at that same close;
- open/intraday data used to claim the already-fixed same-session open;
- same-session close execution without at least five minutes of declared lead;
- missing or placeholder commission, slippage, or market-impact assumptions;
- shorts without explicit availability and borrow-fee assumptions;
- missing data field/cadence/availability declarations;
- matches to the current strategy book or a recorded dead end.

Passing these gates means only “ready to spend research time.” It is not a
positive expected-return finding.

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
must end in `.json` or `.jsonl`; URLs are rejected. The CLI writes only beneath
the selected output directory. Report files use same-directory atomic replace;
the journal uses flushed append-only records and never rewrites history.

## Future official X adapter: required design

The collector is a separate, later change. It should use the official X
read-only API rather than HTML scraping or a session-bearing browser. Minimum
controls:

1. Read-only scopes only; no post, like, follow, direct-message, or account-write
   permission. Secrets come from the existing secret manager/environment, never
   source, config, logs, reports, or exception text.
2. A reviewed source registry fixes account/list/search locators. The adapter
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

## Decisions required before any operational activation

V1 intentionally leaves these to the owner:

- whether to procure/pay for official X API access and the acceptable daily
  request/cost budget;
- exact accounts, private/public lists, and bounded search queries;
- capture window, cadence, timezone, holiday/weekend behavior, and retention;
- whether the digested catalog exporter is allowed to read the live book and
  which negative-research registry is authoritative;
- shadow acceptance period and service-level thresholds for complete captures;
- report recipients and the separate email delivery design;
- scheduler/host identity, credential storage, alert escalation, and retry
  policy;
- named human actors allowed to record `OWNER_REVIEW` transitions.

Recommended rollout is fixture validation, then at least 20 market sessions in
`SHADOW`, then a reviewed completeness/cursor/dedupe scorecard. Change to
`LIVE`, add scheduling, and add email only through separate explicit owner
approvals. No stage should enable trading or automatic strategy mutation.

## Operator response to failures

- `UNKNOWN`: do not infer absence or promote new candidates. Repair provider,
  count, cursor, conflict, or time-boundary evidence and rerun with a new
  capture ID where appropriate.
- `PARTIAL`: review observed candidates as leads only, but do not call the daily
  discovery set complete. Repair stale/incomplete sources or catalogs.
- hash-chain failure: preserve the journal and investigate. Never truncate,
  rewrite, or regenerate it to make a run green.
- quarantined item: retain provenance for audit, do not follow embedded links or
  instructions, and manually inspect only in an isolated research context.
- stale/digest-failing catalog: rebuild from the authoritative source and have
  the snapshot independently verified before use.

This V1 has no delivery guarantee because email and scheduling are out of
scope. The files are the result until those separately controlled layers are
approved and built.
