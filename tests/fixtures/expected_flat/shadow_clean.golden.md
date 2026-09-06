# Expected-Flat Control — UNKNOWN

> **SHADOW / NON-AUTHORITATIVE — DO NOT TREAT AS OPERATIONAL CLEARANCE**

## What changed

- Evaluated 1 due expected-flat obligation(s) for 2026-09-05.
- Result mix: 0 ALERT, 0 UNKNOWN, 1 CLEAR.
- Validated 4/4 source(s) and reconciled 1/1 account-contract(s).
- Recorded 1 run-level issue(s).

## Required action

- The supplied evidence reconciled; record the manual shadow comparison and continue the approved shadow-review sequence.
- This non-authoritative result is validation evidence only, not operational clearance.

## Run summary

- Run: `golden-shadow`
- Mode: `SHADOW`
- Operationally authoritative: `false`
- Session: `2026-09-05`
- As of: `2026-09-05T21:30:00Z`
- State: **UNKNOWN**
- Schema version: `2`
- Algorithm version: `expected-flat/2.0.0`

## Source completeness

| State | Kind | Source | Receipt | Events | Sequence | Complete through | Issues |
|---|---|---|---|---:|---|---|---|
| CLEAR | OBLIGATION_PRODUCER | time-stop-producer | producer-receipt-1 | 1 | 1..1 | 2026-09-05T21:30:00Z | None |
| CLEAR | EXECUTION | execution-ledger | execution-receipt-1 | 1 | 1..1 | 2026-09-05T21:30:00Z | None |
| CLEAR | BASELINE | allocation-baselines | baseline-receipt-1 | 1 | 1..1 | 2026-09-05T21:30:00Z | None |
| CLEAR | POSITION | position-snapshot | position-receipt-1 | 1 | 1..1 | 2026-09-05T21:30:00Z | None |

## Account-contract reconciliation

| State | Account | Contract | Attributed | Aggregate | Allocations | Baseline effective / recorded | Latest execution | Snapshot | Issues |
|---|---|---|---:|---:|---:|---|---|---|---|
| CLEAR | U111 | AAPL/STK/USD/SMART conId=1001 | 0 | 0 | 1 | 2026-09-05T20:00:00Z / 2026-09-05T20:00:00Z | 2026-09-05T21:20:00Z | 2026-09-05T21:29:00Z | None |

## Obligations

| State | Strategy | Account | Signal / tranche | Contract | Signed residual | Flat by |
|---|---|---|---|---|---:|---|
| CLEAR | OLV | U111 | OLV:AAPL:2026-09-05 / t1 | AAPL/STK/USD/SMART conId=1001 | 0 | 2026-09-05T21:15:00Z |

## Obligation findings

- None.

## Run-level issues

- `NON_AUTHORITATIVE_RUN` (run): offline/shadow evidence cannot provide operational clearance

## Provenance and digests

- Algorithm: `expected-flat/2.0.0`
- Schema: `2`
- Code SHA-256: `2222222222222222222222222222222222222222222222222222222222222222`
- Config SHA-256: `7615c921de5772ce38b3f39d60bbe65036c133d40081d88a3afe7ff808b743ec`
- Input SHA-256: `1111111111111111111111111111111111111111111111111111111111111111`
- Report SHA-256: `f4f49b1613a213f4af8d0a0bc2a85aca3df4b394b0c3cc926b8464ff47f86dde`

---
This artifact is observational only. It cannot stage, place, cancel, or modify an order.
