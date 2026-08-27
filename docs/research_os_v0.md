# Research OS v0 — isolated strategy research

This layer is intentionally disconnected from production. It supplies common
research contracts for four lanes:

1. intraday strategy research;
2. the broad-universe Opportunity Book;
3. Trend V2 experiments;
4. weekly external hypothesis intake from X, SSRN, papers, and other sources.

## Hard boundary

Research OS code may read explicitly supplied local research inputs and write
only to an explicit `artifacts/` directory. It does not import or call order
staging, broker, Google Sheets, production deployment, R2 upload, or live-site
code. It does not modify `STRATEGY_BOOK` or production trend state.
The runnable CLIs resolve paths and refuse outputs outside the active isolated
worktree's own ignored `artifacts/` root; a similarly named directory elsewhere
is not accepted.

An idea can move through:

```text
source -> hypothesis -> preregistration -> trial -> result -> disposition
```

The append-only `research/experiment_registry.py` contract can record sources,
hypotheses, preregistrations, parameter trials, results, and dispositions. The
weekly intake writes its source and hypothesis records today. Intraday and
Trend bundles remain self-contained artifacts until a human explicitly opens
a formal registry experiment; v0 does not pretend those results are already
registered. Every registry record is stamped `research_only=true` and
`no_order=true`. Preregistrations must define both promotion and kill gates and
a positive trial budget. Identical reruns are idempotent under a cross-process
file lock; changed work becomes a new record rather than rewriting history.

Every completed run manifest uses the same safety envelope:
`schema_version`, `research_only=true`, `no_order=true`,
`production_writes=false`, and `automatic_promotion=false`.

## Weekly external idea intake

The v0 miner deliberately separates source collection from evaluation. X and
SSRN access methods can change, X may require a signed-in session, and a paper
can be revised. The exact URL, source text, publication time, and retrieval
time therefore enter through a frozen local JSON, JSONL, or CSV record. The
miner does not fetch or silently reconstruct source content.

Minimum source record:

```json
{
  "source_type": "x or ssrn",
  "url": "https://direct-source-url",
  "title": "Source title",
  "text": "Post text or paper abstract",
  "published_at": "2026-08-25T14:30:00Z",
  "retrieved_at": "2026-08-27T13:00:00Z",
  "claim": "The exact proposed market effect",
  "mechanism": "Why the effect could persist",
  "instruments": ["US equities"],
  "horizon": "10:30 ET to close",
  "test_idea": "A falsifiable test with a fixed decision time",
  "first_rejection": "The cheapest decisive reason to stop",
  "data_requirements": ["point-in-time 15-minute OHLCV"]
}
```

Dry run:

```powershell
python scripts/run_weekly_idea_miner.py `
  --input artifacts/research-sources/sources.jsonl `
  --output-dir artifacts/weekly-idea-miner
```

Write a local HTML inbox, frozen source snapshot, run manifest, and registry:

```powershell
python scripts/run_weekly_idea_miner.py `
  --input artifacts/research-sources/sources.jsonl `
  --output-dir artifacts/weekly-idea-miner `
  --registry artifacts/research-registry/experiments.jsonl `
  --write
```

The queue is archetype-balanced and capped at five by default. Its score only
allocates research attention. It is never a recommendation, approved
position, or entry signal.

## Codex scheduled-task handoff

OpenAI's scheduled-task documentation says local project tasks can run in an
isolated worktree, require the computer and desktop app to be running when
local files are needed, and should be tested manually before being scheduled:
<https://learn.chatgpt.com/docs/automations>.

After several manual runs, the following is the intended task prompt. Do not
arm it until the curated X/SSRN source list and cadence have been reviewed.

```text
Run the weekly external strategy-research intake in a new isolated worktree.
This is research-only. Do not modify production code or state; do not commit,
push, upload, deploy, email, message, stage, allocate, or contact a broker.

Collect at most 30 new source records from the reviewed X account/topic list
and SSRN searches. Preserve direct URL, exact title/post/abstract, authors,
published_at, and retrieved_at. Do not treat source claims as evidence. Save
the frozen records under artifacts/research-sources/<YYYY-MM-DD>/sources.jsonl.

For each source, state the claim, possible mechanism, instruments, horizon,
falsifiable test, data requirements, first rejection, and what would kill it.
Deduplicate against the append-only research registry. Then run:

python scripts/run_weekly_idea_miner.py
  --input <frozen source jsonl>
  --output-dir artifacts/weekly-idea-miner/<YYYY-MM-DD>
  --registry artifacts/research-registry/experiments.jsonl
  --write

Return no more than five hypothesis cards. A card may be ready for
preregistration or may need source diligence; neither status is a trade
recommendation. Do not automatically code or backtest more than the bounded
trial budget, and never promote a result into STRATEGY_BOOK.
```

The reliable always-on portion should remain deterministic data collection in
the existing scheduler infrastructure. Codex is used for source synthesis and
hypothesis formulation, with isolated worktrees and human promotion.

## Promotion contract

Nothing in v0 promotes automatically. A future research candidate must have:

- a stable hypothesis and economic/structural mechanism;
- a point-in-time universe and decision-time definition;
- a fixed cost and execution model;
- an explicit trial budget counting every tested variation;
- independent time and ticker/sector holdouts;
- day/event-clustered inference where signals share a shock;
- predefined promotion and kill gates;
- a shadow period and explicit human decision.
