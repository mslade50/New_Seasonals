# Expected-Flat Control — Offline Foundation Runbook

## Current status

This control is **implemented for local, offline evaluation only**. It is not
connected to IBKR, the execution broker, R2, Google Sheets, SMTP, Task
Scheduler, or any trading process. Nothing in this implementation can create,
stage, place, cancel, or modify an order.

The operational default is `DISABLED`. A `FIXTURE` or `SHADOW` result is
prominently labeled non-authoritative and can never produce an overall
`CLEAR`. The repository does not register a daily task or send an email.

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
| `CLEAR` | Every due obligation is exactly flat, every source is complete, allocation totals equal fresh aggregate positions, and the run is explicitly authoritative `LIVE`. | Retain evidence and continue the independent broker review. |
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
- Baselines have an explicit `effective_at`. Only executions strictly after
  that time are applied, preventing double counting.
- The aggregate snapshot must be fresh **and** at least as recent as all
  baseline/execution evidence that it reconciles.
- Every touched account/contract must satisfy:

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

Each receipt declares its exact event IDs, count, zero-event flag, complete
timestamp, session, and a contiguous ordered source-sequence range. Event IDs,
counts, flags, sequences, session, and cutoff are checked against the supplied
records. Distinct source events cannot share a sequence number.

Execution corrections are explicit families. Revision 1 is the original;
later contiguous revisions replace it, and a latest `VOID` removes the whole
family. Duplicate identical execution records are harmless. Conflicting
duplicates, missing correction revisions, multiple records claiming one
revision, changing allocation identity, or non-monotonic correction history
produce `UNKNOWN`.

## Frozen local manifest shape

`schema_version` is currently `1`. Unknown fields, missing fields, naive
timestamps, non-finite quantities, and JSON floats are rejected.

```json
{
  "schema_version": 1,
  "run": {
    "run_id": "postclose-20260905",
    "run_mode": "FIXTURE",
    "operationally_authoritative": false,
    "as_of": "2026-09-05T21:30:00Z",
    "session_date": "2026-09-05",
    "required_producers": ["time-stop-producer"],
    "max_position_age_seconds": 300
  },
  "producer_receipts": [],
  "execution_receipt": {},
  "baseline_receipt": {},
  "position_receipt": {},
  "obligation_revisions": [],
  "position_baselines": [],
  "execution_revisions": [],
  "aggregate_positions": []
}
```

The objects shown as empty above are required strict receipt objects. The
executable adversarial fixtures in
`tests/test_expected_flat_control.py` are the canonical schema examples until
real producers exist.

## Offline command

```powershell
python scripts/run_expected_flat_control.py `
  --input artifacts/expected_flat/input.json `
  --output-dir artifacts/expected_flat/output `
  --run-mode FIXTURE
```

The CLI reads only the named local JSON file and atomically writes deterministic
JSON, Markdown, and HTML. Filenames include the session and sanitized run ID.
No partial `.tmp` artifact is exposed.

Exit codes:

| Code | Meaning |
|---:|---|
| `0` | `CLEAR` or `NOT_SCHEDULED` |
| `10` | `ALERT` |
| `20` | `UNKNOWN` |
| `2` | CLI/schema error; no evaluation conclusion |

The CLI mode must exactly match the manifest. An authoritative `LIVE` manifest
also requires `--acknowledge-authoritative-live`. That second gate is only a
local misuse guard; it does not make the current offline foundation production
ready.

## Daily human output contract

Markdown and HTML begin with these decision sections:

1. **What changed** — obligations evaluated and result mix.
2. **Required action** — the precise human response and an explicit prohibition
   on auto-trading.
3. Run summary.
4. Obligation table with account, signal/tranche, contract, deadline, state,
   and signed residual.
5. Visible obligation findings with stable reason codes.
6. Run-level source/identity findings.

The JSON contains the same fields for durable audit and diffing. All three
formats are deterministic for the same manifest.

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
- baseline cutoff behavior, missing/duplicate/future baselines, and source
  completeness;
- account/signal/tranche isolation, overlapping allocations, aggregate
  invariant failures, missing/duplicate/stale/pre-evidence snapshots;
- exact `con_id`, unique normalized fallback, ambiguous fallback, and separate
  same-symbol contracts;
- obligation revision gaps, conflicts, identity mutation, time/sequence
  monotonicity, and quantity-independent event identity;
- deterministic, escaped human artifacts and CLI live-mode double gating.

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
