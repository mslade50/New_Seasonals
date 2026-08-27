# Codex handoff for the Wide Opportunity Book

This handoff is research only. It enriches the deterministic 1,025-name
coverage pass; it does not call `daily_pitch.py`, write the Pitch tab, send an
email, change a portfolio, or create an executable idea.

## Funnel

```text
full liquid + overflow universe (~1,025)
  -> deterministic coverage and five archetype ranks
  -> 75-name review queue
  -> 10-name deep-test queue
  -> at most 3 source-diligence notes
  -> human decision whether to preregister a bounded experiment
```

The first three stages use only the supplied local adjusted price cache. The
Codex stage may inspect current public information, but a source claim is not
evidence of an edge. It may reject all ten names. Reader-facing diligence is
capped at three names per run so breadth does not turn into shallow conviction.

## Required diligence card

Every selected name must retain its deterministic ticker, archetype, score,
cutoff date, and first-rejection test, then add:

- direct primary-source URLs and retrieval timestamps;
- what changed, with fact separated from inference;
- a potential variant wedge, explicitly marked unproven;
- why the setup is researchable now;
- the cheapest decisive rejection check;
- what would kill the hypothesis;
- the next bounded workflow and trial budget;
- unresolved earnings, corporate-action, liquidity, borrow, or data issues.

Forbidden fields include action, side, quantity, shares, sizing, target weight,
entry/limit/stop/target price, approval, or an instruction to stage or place an
order. “No candidates survived” is a valid and useful result.

## Manual trial prompt

Run this only in a new isolated worktree after the local price cache is known
to be complete. Output must stay beneath that worktree's ignored `artifacts/`
root.

```text
This is a research-only Wide Opportunity Book run. Do not modify production
code or state; do not commit, push, upload, deploy, email, message, publish,
write Sheets, allocate capital, stage an order, or contact a broker.

Read the existing adjusted master_prices.parquet without changing it. Run:

python scripts/build_wide_opportunity_book.py
  --prices <absolute read-only master_prices path>
  --output-dir artifacts/opportunity-book/<market-cutoff>

The runner must choose the latest local SPY bar as its cutoff. Confirm that all
~1,025 requested tickers received a coverage verdict. Inspect the ten-name
deep-test queue and the seeded audit sample before doing source diligence.

Using current primary sources where available, create source-diligence cards
for no more than three names. Preserve the deterministic fields and use the
contract in research/opportunity_book/CODEX_HANDOFF.md. A mechanical rank is
not a recommendation, and current news is not proof of a repeatable edge.
Reject names that fail the first check. Save a standalone local HTML report,
the structured JSON cards, frozen source metadata, and a run manifest under
the same artifacts run directory. Return the coverage counts, rejections, and
next research workflows. Do not invoke the production Daily Pitch pipeline.
```

After several manually reviewed runs, this prompt can be used as a local Codex
scheduled task in an isolated worktree. Per OpenAI's scheduled-task guidance,
the desktop app and machine must be running for local files, and the task
should be manually tested before scheduling. No task is armed by this project.
