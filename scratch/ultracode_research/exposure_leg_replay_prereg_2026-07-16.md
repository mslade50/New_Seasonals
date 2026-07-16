# Pre-registration: exposure_leg raw-21d kill-rule replay

Registered 2026-07-16, before any replay has run. Source: RISK_DIALS_2026-07-16.md B2.

## What is being tested

The exposure leg (25% NAV VOO/QQQ) currently has two global kill rules:
- Rule 1: flat when RAW 21d dial > 50 — **the rule under test**
- Rule 3: flat when 10d-MA 63d dial > 50 — PIT-pedigreed (same statistic as frag_risk_bands), NOT under test

Rule 2 (1.25x boost when both raw dials < 5) was removed 2026-07-16 on existing
evidence (mirrors the per-trade low-frag boost killed unanimously 2026-07-02);
it is not part of this replay's decision.

Hypothesis to test: Rule 1 adds nothing over Rule 3 alone. The claimed support
("raw has no signal, t=+0.72"; headline SPY -0.99%/21d edge) exists in no
committed file, so removing Rule 1 — which is a PROTECTION rule — requires this
replay to run first. Until it does, Rule 1 stays.

## Pre-registered comparison

Variants, identical everything else (`compute_exposure_leg_backtest`):
1. **Current stack** (Rule 1 + Rule 3, boost_mult=1.0)
2. **Rule-3-only** (the candidate simplification)
3. Reference only, not a decision input: pre-removal stack (boost_mult=1.25)

Metrics, full common history of the fragility series and both tickers:
- Total return, CAGR, max drawdown, Sharpe of the leg's daily PnL
- Time-flat fraction and number of flat episodes per variant
- Per-episode table for every day Rule 1 fires while Rule 3 does not
  (the marginal days — the entire difference lives here)

## Decision rule (registered now)

Drop Rule 1 only if ALL of:
1. Rule-3-only total return >= current stack on the marginal-day set
   (Rule 1 must be shown to cost money, not just be redundant), AND
2. Rule-3-only max drawdown no worse than current by more than 1% of NAV, AND
3. The conclusion holds on the PIT segment alone (2026-07-02 onward) in sign —
   acknowledged as low-power; if the PIT segment is too short to read, the
   full-history result decides and the PIT check is recorded for the record.

Otherwise Rule 1 stays and this pre-registration is closed with a negative result.

## Ops constraints (stated up front)

- **Vintage contamination**: the PIT segment starts 2026-07-02. Pre-2026-07
  episodes near the hard 50 edge are recompute-vintage and may re-date under
  a re-scored series; the episode table must flag which episodes are
  vintage-era.
- The replay uses `data/rd2_fragility.parquet` (5d-smoothed basis) exactly as
  the live rules read it. No re-derivation from raw signals.
- The staleness guard (DIAL_STALE_TD=3, added 2026-07-16) is orthogonal to
  this replay and ships regardless.

## Status

- [x] Rule 2 boost removed (2026-07-16, no replay needed)
- [x] Staleness guard added (2026-07-16)
- [ ] Replay run (this document's comparison)
- [ ] Decision recorded; if Rule 1 drops, CLAUDE.md B6 "only portfolio-level
      risk state" clause can then be written (it is false while Rule 1
      consumes the raw series)
