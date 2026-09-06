# Brief: verify_sizing_d33_clone_clamps (independent money-path verification)

Date: 2026-09-05. Mind session: Codex `/root`.

## Decision and why

D3.3 changes scanner and ledger sizing and therefore requires an independent
verifier. Verify the frozen IOB clone rule, the exact five-pair extension, and
the builder's engine multi-pair collision fix. Try to falsify live/engine
parity rather than repeating the builder's assertions.

## Files you own

- `artifacts/verify_2026-09-05/d33_clone_clamps/`

All source, tests, docs, data, and builder artifacts are read-only. If a defect
is found, report an exact reproduction; do not patch it.

## Hard rules

Follow section 0 of `docs/plan_2026-09-04.md`. Do not run `daily_scan.py`,
`order_staging.py`, an order runner, or a workflow. Do not write Sheets, R2,
Cloudflare, IBKR, or Task Scheduler. Do not commit, stash, checkout, push,
install dependencies, rename, move, or delete files.

## Intent

Independently establish all of the following:

1. Source config contains the existing IOB+MonFri pair plus exactly the five
   D3.3 additions, each 20 bps nominal and 30 bps effective at GRM 1.5.
2. IOB alone gets a 0.5x multiplier when both of its two staged index signals
   fire; a single signal stays 1.0x. MonFri has no clone multiplier. Existing
   3x Bear staged-count behavior is byte-equivalent at representative counts.
3. `^GSPC -> SPY` and `^NDX -> QQQ` aliasing is used consistently for
   cross-strategy collisions.
4. Cross-strategy clamps are absolute and occur before the IOB multiplier.
   A 52.5-bps IOB row that is both cross-strategy-clamped and cloned lands at
   15 effective bps; a row already below 30 bps is not raised.
5. Construct an adversarial triple-strategy same-date/same-ticker collision
   where two configured pairs share one member. Every strategy participating
   in at least one fired pair must receive its minimum applicable clamp in the
   engine, matching scan semantics; pair iteration order must not matter.
6. Candidate counts are pre-fill/staged and live remains per-tier. A failed
   fill on either clone does not restore the other to full risk in the engine.
7. No unrelated strategy config, entry/exit logic, caps, earnings overrides,
   pivot behavior, or OneDrive file changed.
8. Independently reproduce or audit the builder's full-history replay and
   identify any unexplained trade-key or PnL differences.

## Recon first

Write an attack plan to
`artifacts/verify_2026-09-05/d33_clone_clamps/00_attack_plan.md` before running
probes.

## Verification

- Inspect the complete diff and source call order.
- Run the builder's focused tests plus independent probes written only under
  the verifier artifact directory.
- Run the full test suite.
- Audit the D3.3-only replay inputs, scenario isolation, row keys and metrics.
- Produce `artifacts/verify_2026-09-05/d33_clone_clamps/checks.json` by script
  with at least: `verdict`, `pair_table_exact`, `grm_scaling_exact`,
  `iob_single_full`, `iob_double_half`, `monfri_untouched`,
  `absolute_then_clone`, `below_clamp_unchanged`, `triple_collision_safe`,
  `pair_order_invariant`, `staged_not_filled`, `scan_engine_parity`,
  `unrelated_diff_paths`, `tests_failed`, `replay_audited`, and
  `external_writes`.

## Report

Use section 6 of `docs/plan_2026-09-04.md` verbatim. Verdict must be exactly
PASS or FAIL; gaps are a FAIL until the builder closes them.
