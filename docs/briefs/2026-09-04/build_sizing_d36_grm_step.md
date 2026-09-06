# Brief: build_sizing_d36_grm_step (GRM 1.875 + overflow-long exclusion)

Date: 2026-09-05. Mind session: Codex `/root`.

## Decision and why

Fortnight D3.6 is unlocked by independent PASS verdicts on D3.1-D3.5. Raise
the book-wide global risk multiplier from 1.5 to 1.875. Keep the four
overflow-long sleeves at their D3.5 effective risk by multiplying their source
nominal overrides by 0.8: OLV 20, LT Trend ST OS 24, St OS Sznl 32, and 52wh
Breakout 28 bps. Liquid rows and overflow shorts take the 1.25x step.

## Files you own

- `strategy_config.py`
- `daily_scan.py`
- `daily_portfolio_report.py`
- `pages/strat_backtester.py`
- `tests/test_grm_step.py` (new)
- `tests/test_base_bps_tilt.py` (overflow expectations only)
- `CLAUDE.md` (only the GRM / overflow override contract)
- `docs/running_list.md` (only this build's status)
- `artifacts/build_2026-09-05/d36_grm_step/`

Everything else is read-only. Build on independently verified D3.5 in
`codex/fortnight-d33-clamps`; do not alter or re-litigate D3.1-D3.5.

## Hard rules

Follow section 0 of `docs/plan_2026-09-04.md`. Do not run `daily_scan.py`,
`daily_portfolio_report.py`, `order_staging.py`, an order runner, or a
workflow. Do not write Sheets, R2, Cloudflare, IBKR, Task Scheduler, or
production data. Do not commit, stash, checkout, push, install dependencies,
rename, move, or delete files.

## Intent

1. Set `GLOBAL_RISK_MULTIPLIER = 1.875` exactly. Preserve all existing GRM
   mechanics: strategy `risk_bps`, OVS path bps/cap, earnings size overrides,
   and absolute overlap clamps scale once; dimensionless overlays do not.
2. Preserve the source nominal overflow-base table separately and exactly:
   OLV 20, LT Trend 24, St OS Sznl 32, 52wh 28. The existing D3.2 tilt still
   folds into the exported runtime map; therefore 52wh's runtime override is
   28 x 0.70 = 19.6 before consumers multiply by GRM. Effective overflow base
   bps must remain D3.5-exact: OLV 37.5, LT Trend 45, St OS 60, 52wh 36.75.
3. “Overflow longs excluded” applies to final special-case risk too, not only
   ordinary base risk. Add a narrow nominal earnings-size override table for
   the two excluded carriers that have one: OLV 8 and St OS Sznl 4.8. Scanner,
   report-book builder, and engine `overflow_active` path must keep their
   effective overflow earnings sizes at 15 and 9 bps respectively. Liquid
   earnings overrides take the GRM step to 18.75 and 11.25.
4. Wire both overflow tables through `daily_scan.build_effective_strategy_book`,
   `daily_portfolio_report.build_full_strategy_book`, and the engine's direct
   `overflow_active` path. Deep copies only; liquid config must not mutate.
5. D3.5 OVS remains exact: liquid 0.5x, overflow 1.0x; rank-mean 0.7x outside
   midterms; P2 denominator parity. OVS and ATR Extended Gap Up are overflow
   shorts and take the 1.25x GRM step. Fixed per-strategy daily caps remain
   fixed effective bps and may make realized ratios less than 1.25.
6. Do not change base tilts, D3.3 clamps/clone rule, D3.4 open-leg rule, D3.5
   configs, earnings windows, OVS paths, entry/exit behavior, cap thresholds,
   OneDrive, or any order runner. The older WP5 ADV-participation proposal is
   outside fortnight D3.6 and is not changed here.
7. No production action follows this build. The fortnight rule keeps the work
   uncommitted/unpublished; live-task or broker state is untouched.

## Recon first

Write the exact-edit plan before source edits to
`artifacts/build_2026-09-05/d36_grm_step/00_exact_edit_plan.md`.

## Verification

- `python -m pytest -q tests/test_grm_step.py tests/test_base_bps_tilt.py tests/test_ovs_risk_mults.py tests/test_open_leg_mults.py tests/test_clone_clamps.py`
- `python -m pytest -q tests/`
- Frozen-input engine replay with identical candidate data and all production
  caps/overlays: pre-D3 “today” at GRM 1.5 from the archived pre-D3 source,
  D3.5 at GRM 1.5, uniform D3.6 at 1.875 without the overflow exclusion, and
  production D3.6. Compare 2010+ and 2016-07+ annual PnL, Sharpe, maxDD, worst
  day and worst 21d. Reconcile every changed key and break out all four
  excluded overflow sleeves, including ordinary versus earnings-override rows.
- Score the frozen engine gate exactly: (a) D3.6 2010+ annual PnL improves at
  least $30,000 (4 points of $750k NAV) over pre-D3 today; (b) D3.5's GRM-1.5
  equivalent maxDD is no more than $7,500 worse than pre-D3 today; (c) D3.6
  worst-21d loss is no more than 110% of pre-D3 today; and (d) D3.6's 2016+
  worst drawdown trough is not in June-July 2026. A failed gate is a FAIL and
  stops acceptance; do not invent a replacement multiplier.
- The archived pre-D3 source and any input-vintage mismatch must be hash-stamped
  and disclosed. All current arms must use the same frozen local inputs with
  R2 refresh disabled.
- Produce `artifacts/build_2026-09-05/d36_grm_step/checks.json` by script with
  at least: `grm_exact`, `nominal_overflow_table_exact`,
  `effective_overflow_base_unchanged`, `overflow_earnings_unchanged`,
  `liquid_and_overflow_shorts_take_step`, `grm_scaled_fields_once`,
  `dimensionless_fields_unchanged`, `three_consumers_match`,
  `prior_sizing_invariants_unchanged`, `narrow_tests_failed`,
  `full_tests_failed`, `replay_complete`, `engine_gate_pass`, and
  `external_writes`.

## Report

Use section 6 of `docs/plan_2026-09-04.md` verbatim.
