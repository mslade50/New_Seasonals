# Brief: build_sizing_d35_ovs_risk_mults (OVS extremity + liquid tier)

Date: 2026-09-05. Mind session: Codex `/root`.

## Decision and why

Fortnight D3.5 is frozen. OVS signals whose signal-close mean of the 2d, 5d,
10d and 21d performance ranks is below 94 size at 0.7x, except midterm years
(`year % 4 == 2`) where this rule is 1.0x because the existing 0.75 cycle
multiplier already applies. There is no top-cell boost. D3.5b independently
sizes liquid-tier OVS at 0.5x on both gap paths; overflow is unchanged. The
pre-registered D9 decision met every gate and the owner's “Get to work” ended
the veto window before this brief was written.

## Files you own

- `strategy_config.py`
- `daily_scan.py`
- `pages/strat_backtester.py`
- `tests/test_ovs_risk_mults.py` (new)
- `CLAUDE.md` (only the OVS sizing contract / sizing order)
- `docs/running_list.md` (only this build's status)
- `artifacts/build_2026-09-05/d35_ovs_risk_mults/`

Everything else is read-only. Build on the independently verified D3.3+D3.4
tree in `codex/fortnight-d33-clamps`; do not alter or re-litigate earlier items.

## Hard rules

Follow section 0 of `docs/plan_2026-09-04.md`. Do not run `daily_scan.py`,
`order_staging.py`, an order runner, or a workflow. Do not write Sheets, R2,
Cloudflare, IBKR, Task Scheduler, or production data. Do not commit, stash,
checkout, push, install dependencies, rename, move, or delete files.

## Intent

1. OVS alone carries `rank_mean_risk` with exact config:
   `windows=[2,5,10,21]`, `threshold=94.0`, `below_mult=0.7`,
   `cycle_exempt=[2]`. The cutoff is strict `< 94`; 94 is full size. All four
   values come from the signal-close `rank_ret_{window}d` columns. A missing or
   non-finite value is unclassifiable and returns 1.0; OVS's signal mask already
   requires all four ranks >85, so a production candidate normally cannot take
   this fallback. Add a generic pure validated helper in `strategy_config.py`.
2. OVS alone carries generic `tier_risk_mults={"Liquid": 0.5}`. Missing tier
   entries return 1.0. Add a generic pure validated helper. Neither field is
   GRM-scaled and neither changes path bps or cap thresholds.
3. Scanner: apply tier and rank-mean multipliers to unrounded row risk before
   shares, ADV, concurrent-notional, cross-strategy, same-day, and downstream
   staging caps. Tier comes from `_scan_source`; ranks come from `last_row`.
   Stamp tier, mean/cell, exemption and multipliers in `Sizing_Notes`. The
   bottom-extremity rule has no 2026 live effect because 2026 is midterm; the
   liquid 0.5x rule does.
4. Engine: carry the four point-in-time ranks into `signal_data`; classify tier
   by `ticker in LIQUID_PLUS_COMMODITIES`, matching the disjoint full-book
   passes. Apply both multipliers before shares and every cap. They must affect
   P1 and P2.
5. The OVS P2 aggregate-cap pre-pass must include the same tier/extremity/cycle
   risk that the main sizing loop stages. Do not multiply the fixed cap dollars:
   only the staged-risk denominator changes. This is required for live-engine
   parity because order_staging aggregates the scanner's already-sized P2 rows.
6. Do not change `path1_bps`, `path2_bps`, `path2_daily_cap_pct`, the existing
   `cycle_risk_mults`, scale-out, earnings blackout, EOD-DD, precedence, any
   non-OVS strategy, OneDrive, or `daily_portfolio_report.py`.
7. D12 monitoring remains: re-examine after +40 new liquid OVS positions and
   retire the tier cut if that cell's avgR exceeds +0.3.

## Recon first

Write the exact-edit plan before source edits to
`artifacts/build_2026-09-05/d35_ovs_risk_mults/00_exact_edit_plan.md`.

## Verification

- `python -m pytest -q tests/test_ovs_risk_mults.py tests/test_ovs_scaleout.py tests/test_open_leg_mults.py tests/test_clone_clamps.py tests/test_base_bps_tilt.py`
- `python -m pytest -q tests/`
- Run a frozen-input, four-arm full-history engine replay at GRM 1.5 on top of
  verified D3.4: D3.4 control (both new fields removed), extremity only, liquid
  tier only, and combined D3.5. Compare 2010+ and 2016-07+ book PnL, annual PnL,
  Sharpe, maxDD, worst day and worst 21d. Reconcile every changed position key
  and risk/PnL delta. Break OVS results out by liquid/overflow, P1/P2, and
  midterm/non-midterm; prove the extremity arm changes zero midterm rows.
- Produce `artifacts/build_2026-09-05/d35_ovs_risk_mults/checks.json` by script
  with at least: `carrier_sets_exact`, `configs_exact`, `cutoff_strict`,
  `midterm_exempt`, `tier_liquid_half`, `tier_overflow_full`,
  `point_in_time_rank_snapshot`, `p1_p2_both_scaled`,
  `p2_prepass_parity`, `scan_engine_order_match`, `noncarriers_unchanged`,
  `narrow_tests_failed`, `full_tests_failed`, `replay_complete`,
  `replay_rows_affected`, `midterm_extremity_rows_affected`, and
  `external_writes`.

## Report

Use section 6 of `docs/plan_2026-09-04.md` verbatim.
