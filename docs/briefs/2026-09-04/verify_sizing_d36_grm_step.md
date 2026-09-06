# Brief: verify_sizing_d36_grm_step (independent money-path verification)

Date: 2026-09-05. Mind session: Codex `/root`.

## Decision and why

D3.6 changes book-wide live sizing and requires independent verification.
Try to falsify one-time GRM scaling, the four overflow-long exemptions,
special-case earnings sizing, all three consumers, and the engine gate.

## Files you own

- `artifacts/verify_2026-09-05/d36_grm_step/`

All source, tests, docs, data, and builder artifacts are read-only. Report an
exact reproduction for any defect; never patch it.

## Hard rules

Follow section 0 of `docs/plan_2026-09-04.md`. Do not run `daily_scan.py`,
`daily_portfolio_report.py`, `order_staging.py`, an order runner, or a
workflow. Do not write Sheets, R2, Cloudflare, IBKR, Task Scheduler, or
production data. Do not commit, stash, checkout, push, install dependencies,
rename, move, or delete files.

## Intent

Independently establish all of the following:

1. GRM is exactly 1.875 and every GRM-denominated field scales exactly once;
   D3.2-D3.5 dimensionless multipliers do not scale.
2. The source nominal overflow table is exactly OLV 20, LT Trend 24, St OS 32,
   52wh 28. After the 52wh 0.70 base tilt and GRM, exported effective overflow
   base bps remain exactly 37.5, 45, 60, and 36.75.
3. OLV/St OS overflow earnings overrides remain exactly 15/9 effective bps,
   while liquid values take the step to 18.75/11.25. Attack direct engine
   `overflow_active` calls as well as both book builders.
4. Liquid rows and overflow shorts take the 1.25x step before caps; fixed caps
   remain fixed. D3.3-D3.5 behavior, OVS P2 parity, and noncarrier behavior are
   unchanged.
5. Independently reproduce/audit every replay arm and all four registered gate
   decisions, including the 2016+ worst-drawdown trough date.

## Recon first

Write an attack plan to
`artifacts/verify_2026-09-05/d36_grm_step/00_attack_plan.md` before probes.

## Verification

- Inspect the complete cumulative diff and isolate D3.6 from verified D3.1-D3.5.
- Run focused tests plus verifier-owned synthetic scanner/report/engine probes.
- Run the full suite and reproduce any claimed baseline failure on main.
- Independently audit or rerun the frozen engine arms, input hashes, changed-key
  reconciliation, overflow sleeve attribution, and the four engine gates.
- Produce scripted `checks.json` with at least: `verdict`, `grm_exact`,
  `nominal_overflow_table_exact`, `effective_overflow_base_unchanged`,
  `overflow_earnings_unchanged`, `liquid_and_overflow_shorts_take_step`,
  `grm_scaled_fields_once`, `dimensionless_fields_unchanged`,
  `three_consumers_match`, `prior_sizing_invariants_unchanged`, `tests_failed`,
  `replay_audited`, `engine_gate_pass`, `unrelated_diff_paths`, and
  `external_writes`.

## Report

Use section 6 of `docs/plan_2026-09-04.md` verbatim. Verdict must be exactly
PASS or FAIL; an unclosed gap is a FAIL.

