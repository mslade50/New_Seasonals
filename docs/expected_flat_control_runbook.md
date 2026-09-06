# Expected-Flat Control — Offline Foundation Runbook

## Current status

This control is **implemented for local, offline evaluation only**. It is not
connected to IBKR, the execution broker, R2, Google Sheets, SMTP, Task
Scheduler, or any trading process. Nothing in this implementation can create,
stage, place, cancel, or modify an order.

The operational default is `DISABLED`. The V2 file schema rejects
`operationally_authoritative=true`, and the file-only CLI rejects `LIVE`
entirely. A `FIXTURE` or `SHADOW` result is prominently labeled
non-authoritative and can never produce an overall `CLEAR`. The repository
does not register a daily task or send an email.

## What the control answers

For each append-only expected-flat obligation that is due by the run cutoff:

1. Which account, signal, and tranche created the obligation?
2. Which exact contract is involved?
3. What was the signed allocation quantity at an explicit baseline timestamp?
4. Which deduplicated, correction-aware executions changed it after that
   baseline?
5. Does the sum of every allocation on the touched account/contract equal the
   fresh aggregate position snapshot?
6. Is the due signal/tranche flat, long, short, reversed, or unknowable?

It intentionally does **not** ask whether a ticker-level modeled portfolio is
flat. Multiple strategies and tranches can legitimately net in one broker
position, so a ticker-only comparison can both miss real residuals and create
false alarms.

## State contract

| State | Meaning | Required response |
|---|---|---|
| `CLEAR` | Every due obligation is exactly flat, every source is complete, allocation totals equal fresh aggregate positions, and a future trusted adapter—not a V2 file—constructs an explicitly authoritative `LIVE` run. | Retain evidence and continue the independent broker review. |
| `ALERT` | Complete evidence proves a non-zero due residual. The signed quantity identifies long, short, or reversal. | Investigate immediately. Do not auto-trade from this report. |
| `UNKNOWN` | Identity, freshness, correction, source completeness, or aggregate reconciliation is not provable; also the mandatory overall state for otherwise-clean `FIXTURE`/`SHADOW` runs. | Repair the evidence gap before drawing a flatness conclusion. |
| `NOT_SCHEDULED` | The control is disabled or complete producer receipts prove no obligations are due. | Confirm that this was the intended schedule/mode. |

Precedence is fail-closed: a proven `ALERT` remains visible; otherwise any
material uncertainty prevents `CLEAR`.

## Identity and quantity invariants

- Account is part of every key. Two accounts never net against each other.
- `con_id` is exact when present. A normalized
  `(symbol, sec_type, currency, exchange, expiry)` fallback is allowed only
  when it maps to one contract for that account. Ambiguity is `UNKNOWN`.
- Signal and tranche remain separate even when they share a contract.
- Quantities are signed `Decimal` values encoded as JSON strings or integers.
  JSON floats are rejected. Long is positive; short is negative; zero is flat.
- An obligation `event_id` is an opaque immutable identity. It must not embed
  mutable quantity. Amendments append a monotonically increasing revision;
  they never replace or delete prior revisions.
- A `baseline_id` is globally immutable across its append-only history.
  Revision 1 is `INITIAL`; later contiguous revisions use `CORRECTION`,
  `CORPORATE_ACTION`, or `POSITION_TRANSFER`. `recorded_at` and
  `source_sequence` must strictly increase, and `effective_at` cannot regress.
  The latest revision recorded and effective by the cutoff supplies the full
  replacement quantity. Only executions strictly after its `effective_at` are
  applied, preventing double counting while allowing splits and corrections.
- The aggregate snapshot must be fresh **and** at least as recent as all
  baseline/execution evidence that it reconciles.
- Every relevant supplied account/contract—across baseline histories, active
  executions, aggregate snapshots, and due obligations—must satisfy:

  `sum(current signed quantity across all supplied signal/tranche allocations) == aggregate signed broker quantity`

  This invariant is required even when the due tranche itself calculates to
  zero. Missing overlapping allocations therefore produce `UNKNOWN`, not a
  false `CLEAR`.

## Append-only source receipts

Four source classes must prove completeness through the run cutoff:

1. One receipt for every producer listed in `run.required_producers`, including
   a positive zero-obligation receipt when it ran but emitted nothing.
2. One execution receipt.
3. One allocation-baseline receipt.
4. One aggregate-position receipt.

Each receipt declares its exact revision-qualified record IDs, count,
zero-event flag, complete timestamp, session, and contiguous ordered
source-sequence range. The sequence array must equal the sequences on supplied
records; it cannot claim phantom rows. A zero-event receipt uses `event_ids=[]`,
`event_count=0`, `source_sequence_first=null`, `source_sequence_last=null`, and
`source_sequences=[]`. Future as well as stale `complete_through` timestamps
fail closed. Distinct source records cannot share a sequence number.

Execution corrections are explicit families. Revision 1 is the original;
later contiguous revisions replace it, and a latest `VOID` removes the whole
family. Duplicate identical execution records are harmless. Conflicting
duplicates, missing correction revisions, multiple records claiming one
revision, changing allocation identity, or non-monotonic correction history
produce `UNKNOWN`.

## Frozen local manifest shape

`schema_version` is currently `2`. Unknown fields, missing fields, naive
timestamps, non-finite quantities, and JSON floats are rejected.

```json
{
  "schema_version": 2,
  "run": {
    "run_id": "postclose-20260905",
    "run_mode": "FIXTURE",
    "operationally_authoritative": false,
    "as_of": "2026-09-05T21:30:00Z",
    "session_date": "2026-09-05",
    "required_producers": ["time-stop-producer"],
    "max_position_age_seconds": 300
  },
  "producer_receipts": [{
    "source_id": "time-stop-producer",
    "receipt_id": "producer-receipt-1",
    "session_date": "2026-09-05",
    "complete": true,
    "complete_through": "2026-09-05T21:30:00Z",
    "zero_events": true,
    "event_ids": [],
    "event_count": 0,
    "source_sequence_first": null,
    "source_sequence_last": null,
    "source_sequences": []
  }],
  "execution_receipt": {
    "source_id": "execution-ledger",
    "receipt_id": "execution-receipt-1",
    "session_date": "2026-09-05",
    "complete": true,
    "complete_through": "2026-09-05T21:30:00Z",
    "zero_events": true,
    "event_ids": [],
    "event_count": 0,
    "source_sequence_first": null,
    "source_sequence_last": null,
    "source_sequences": []
  },
  "baseline_receipt": {
    "source_id": "allocation-baselines",
    "receipt_id": "baseline-receipt-1",
    "session_date": "2026-09-05",
    "complete": true,
    "complete_through": "2026-09-05T21:30:00Z",
    "zero_events": true,
    "event_ids": [],
    "event_count": 0,
    "source_sequence_first": null,
    "source_sequence_last": null,
    "source_sequences": []
  },
  "position_receipt": {
    "source_id": "position-snapshot",
    "receipt_id": "position-receipt-1",
    "session_date": "2026-09-05",
    "complete": true,
    "complete_through": "2026-09-05T21:30:00Z",
    "zero_events": true,
    "event_ids": [],
    "event_count": 0,
    "source_sequence_first": null,
    "source_sequence_last": null,
    "source_sequences": []
  },
  "obligation_revisions": [],
  "position_baselines": [],
  "execution_revisions": [],
  "aggregate_positions": []
}
```

The executable fixture at `tests/fixtures/expected_flat/shadow_clean.json` and
the adversarial builders in
`tests/test_expected_flat_control.py` are the canonical schema examples until
real producers exist.

Non-zero receipt event IDs identify append-log revisions, for example
`olv-aapl-t1-flat@1`, `exec-1@2`, and `baseline-1@2`. Aggregate snapshots are
not revisioned, so their immutable `snapshot_id` is the receipt event ID.

## Offline command

```powershell
python scripts/run_expected_flat_control.py `
  --input artifacts/expected_flat/input.json `
  --output-dir artifacts/expected_flat/output `
  --run-mode FIXTURE
```

The CLI reads only the named local JSON file. It writes JSON, Markdown, HTML,
and `completion.json` into a private staging directory, then atomically renames
that directory into an immutable generation. The generation name includes the
session, sanitized run ID, input digest prefix, and report digest prefix. An
exact retry verifies every report byte and the completion manifest, then returns
the already-published generation idempotently. A partial, tampered, malformed,
or unexpected existing generation fails closed and is never rewritten. The
input must resolve outside the output directory, and any failed artifact write
removes the unpublished staging directory, so no partial generation is exposed.

Exit codes:

| Code | Meaning |
|---:|---|
| `0` | `NOT_SCHEDULED` for the file-only CLI (`CLEAR` remains reserved for a future trusted adapter) |
| `10` | `ALERT` |
| `20` | `UNKNOWN` |
| `2` | CLI/schema error; no evaluation conclusion |

The CLI mode must exactly match the manifest and must be `DISABLED`, `FIXTURE`,
or `SHADOW`. `LIVE` is not a CLI choice. The frozen parser itself rejects
`operationally_authoritative=true`, so another local-file caller cannot bypass
the CLI and turn hand-authored JSON into operational `CLEAR`. There is no
override flag.

## Daily human output contract

Markdown and HTML begin with these decision sections:

1. **What changed** — obligations evaluated and result mix.
2. **Required action** — the precise human response and an explicit prohibition
   on auto-trading.
3. Run summary with schema and algorithm versions.
4. Positive source-completeness table, including zero-event sources.
5. Per-account/contract reconciliation table with attributed and aggregate
   signed quantities plus baseline, execution, and snapshot evidence times.
6. Obligation table with account, signal/tranche, contract, deadline, state,
   and signed residual.
7. Visible obligation and run-level findings with stable reason codes.
8. Provenance with deterministic code, config, input, and semantic-report
   SHA-256 digests.

The JSON contains the same evidence for durable audit and diffing. All three
report formats are deterministic for the same manifest. The semantic
`report_sha256` hashes the canonical report with that recursive field blank;
`completion.json` separately hashes the exact bytes and records the size of
each published report file. Manifest-derived Markdown and HTML text is escaped
and newline-neutralized.

## Test and incident matrix

The offline suite covers:

- authoritative clear, fixture/shadow non-clear, disabled, and genuine
  zero-obligation days;
- missing, duplicate, stale, wrong-session, wrong-event-set, zero-flag, and
  sequence-incomplete receipts;
- long and short residuals, partial fills, both reversal directions, and exact
  decimal fractions;
- exact duplicate fills, conflicting duplicates, corrections, voids, gaps,
  ambiguous families, and changed correction allocation;
- baseline cutoff behavior, append-only correction/corporate-action revisions,
  immutable identity, revision gaps/regressions/conflicts, future records, and
  source completeness;
- account/signal/tranche isolation, overlapping allocations, aggregate
  invariant failures, missing/duplicate/stale/pre-evidence snapshots;
- exact `con_id`, unique normalized fallback, ambiguous fallback, and separate
  same-symbol contracts;
- obligation revision gaps, conflicts, identity mutation, time/sequence
  monotonicity, and quantity-independent event identity;
- exact/no-phantom and explicit zero-event sequence receipts, future receipts,
  and revision-qualified execution receipt identities;
- deterministic, escaped human artifacts, tracking-pixel/newline attacks,
  categorical file/CLI authority rejection, input/output collisions,
  byte-verified idempotent retries, partial/tampered generation rejection, and
  failure injection at every publication boundary.

Run:

```powershell
python -m pytest -q tests/test_expected_flat_control.py
```

## Required work before any operational activation

Do not schedule or email this control until all of the following are reviewed
and approved:

1. Every live exit producer emits the immutable obligation and a complete
   daily receipt, including zero-obligation runs.
2. A durable allocation ledger seeds all existing positions by account,
   `con_id`, signal, and tranche. Unseeded legacy inventory must remain
   `UNKNOWN`; it cannot be declared clean.
3. The execution source preserves stable execution IDs and explicit correction
   families, with evidence that its session is complete.
4. A fresh broker aggregate snapshot carries exact account/contract identity
   and a completeness receipt.
5. Shadow runs cover normal days, overlapping-strategy days, partial fills,
   rejected exits, restarts, stale books, corrected fills, short positions,
   reversals, and zero-obligation days. Every result is manually reconciled to
   broker records.
6. Alert/unknown email recipients, escalation timing, dedupe/idempotency, and
   delivery-failure behavior are owner-approved and separately tested.
7. A money-path verifier confirms producer tagging cannot alter order payloads
   or executor behavior.
8. Only then should `LIVE` authority, SMTP, and scheduling be proposed as a
   separate, explicitly approved change.

There is intentionally no automatic remediation. Detection and trading remain
separate control domains.
