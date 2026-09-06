# Brief: build_sizing_d33_clone_clamps (IOB index-clone cut and cross-strategy clamp extension)

Date: 2026-09-05. Mind session: Codex `/root`.

## Decision and why

Fortnight decision D3.3 is frozen: halve both Indices Oversold Bounce rows when
SPY and QQQ fire together on the same signal date, and extend the existing
same-date/same-tradeable absolute clamp to five additional dip-buy pairs at 20
bps nominal per side. The IOB rule is a variance-only clone control; it does
not apply to SPY QQQ MonFri Reversion. Cross-strategy clamps remain absolute,
GRM-scaled at import, keyed on staged signals rather than fills, and applied
before the same-day multiplier.

## Files you own

- `strategy_config.py`
- `daily_scan.py`
- `pages/strat_backtester.py`
- `tests/test_same_day_derate.py`
- `tests/test_clone_clamps.py` (new)
- `CLAUDE.md` (only the overlap/clone and OLV-pivot documentation touched by this brief)
- `docs/plan_2026-09-04.md` (only the OLV-pivot owner decision)
- `docs/running_list.md` (only O21/O34 and this build's status)
- `artifacts/build_2026-09-05/d33_clone_clamps/`

Everything else is read-only.

## Hard rules

Follow section 0 of `docs/plan_2026-09-04.md`. Do not run `daily_scan.py`,
`order_staging.py`, an order runner, or a workflow. Do not write Sheets, R2,
Cloudflare, IBKR, or Task Scheduler. Do not commit, stash, checkout, push,
install dependencies, rename, move, or delete files. Preserve adjusted-bar
engine/live conventions, cap ordering, and staged-signal semantics.

## Intent

1. Add exactly these five nominal-20-bps entries to
   `CROSS_STRATEGY_OVERLAP_OVERRIDES`, preserving the existing IOB+MonFri pair:
   - Monday Dip + Weak Close Decent Sznls
   - SPY QQQ MonFri Reversion + Weak Close Decent Sznls
   - Monthly Weak Close + SPY QQQ MonFri Reversion
   - Monthly Weak Close + Indices Oversold Bounce
   - Monday Dip + Indices Oversold Bounce
2. Configure only Indices Oversold Bounce with
   `same_day_signal_derate: 0.5` and `same_day_derate_floor: 0.5`. Its exact
   two-ticker spot universe makes the generic staged candidate count equivalent
   to the SPY+QQQ condition. SPY QQQ MonFri Reversion must not carry this field.
3. Keep the sizing order on both sides: all earlier per-row overlays ->
   cross-strategy absolute clamp -> IOB same-day 0.5x -> later gap derate/caps.
4. Make the engine overlap pre-pass correct for a triple collision in which a
   strategy participates in more than one configured pair. Resolve the minimum
   applicable clamp per `(signal date, tradeable ticker, strategy)`; never let
   the last pair overwrite an earlier affected strategy.
5. Reuse the existing scan and engine same-day derate machinery. Generalize its
   comments/docstring to both carriers without changing 3x Bear behavior or
   per-tier live counting.
6. Do not edit OneDrive: order staging already consumes the reduced scanner
   `Shares` / `Risk_Amt` / `Notional`.
7. Record the owner decision to KEEP the OLV pivot policy. Correct the old
   evidence claim: +8.68R was v2 vs v1, not policy vs no policy; the 2026-09-04
   replay found no per-signal edge (affected diff -4.6R, clustered t -0.33;
   total OLV PnL approximately unchanged), but materially smaller worst-21d and
   max drawdown (-$37k vs -$60k and -$41k vs -$65k). State that it is retained
   as an appetite/fewer-fills/drawdown control; basis classification flipped on
   1 of 19 policy assignments in the stability audit.

## Recon first

Write the exact-edit plan before source edits to
`artifacts/build_2026-09-05/d33_clone_clamps/00_exact_edit_plan.md`.

## Verification

- `python -m pytest -q tests/test_clone_clamps.py tests/test_same_day_derate.py tests/test_base_bps_tilt.py`
- `python -m pytest -q tests/`
- Run a full-history D3.3-only engine replay at GRM 1.5. Compare current D3.3
  with a counterfactual that removes only the five new pairs and the IOB
  same-day fields. Report affected candidate/trade rows and 2010+/2016-07+
  PnL, Sharpe, maxDD, worst day and worst-21d deltas.
- Produce `artifacts/build_2026-09-05/d33_clone_clamps/checks.json` by script
  with at least: `pair_table_exact`, `all_clamps_effective_bps`,
  `iob_derate_exact`, `monfri_has_no_iob_derate`, `triple_collision_safe`,
  `scan_engine_order_match`, `narrow_tests_failed`, `full_tests_failed`,
  `replay_complete`, `replay_rows_affected`, and `external_writes`.

## Report

Use section 6 of `docs/plan_2026-09-04.md` verbatim.
