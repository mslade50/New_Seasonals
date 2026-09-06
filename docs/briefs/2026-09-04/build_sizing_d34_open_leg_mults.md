# Brief: build_sizing_d34_open_leg_mults (WCDS / LT Trend solo-add sizing)

Date: 2026-09-05. Mind session: Codex `/root`.

## Decision and why

Fortnight decision D3.4 is frozen: Weak Close Decent Sznls and LT Trend ST OS
size at 0.8x on a true solo signal and 1.2x when either the same strategy has
two or more staged candidates in that tier/day or it already has at least one
filled leg open at signal time. The same-day re-key is required because the
evidence lives in cluster days and every row must be sized consistently before
fills are known. Same-sector clustering is irrelevant and working limits do
not count under the fortnight's narrower "prior open leg" decision.

## Files you own

- `strategy_config.py`
- `daily_scan.py`
- `pages/strat_backtester.py`
- `tests/test_open_leg_mults.py` (new)
- `tests/test_wcds_size_tiers.py` (D3.1 invariant updated for the new
  rank-independent D3.4 solo baseline)
- `CLAUDE.md` (only the D3 sizing hierarchy / open-leg contract)
- `docs/running_list.md` (only this build's status)
- `artifacts/build_2026-09-05/d34_open_leg_mults/`

Everything else is read-only. Build on the already-verified uncommitted D3.3
tree in `codex/fortnight-d33-clamps`; do not alter or re-litigate D3.3.

## Hard rules

Follow section 0 of `docs/plan_2026-09-04.md`. Do not run `daily_scan.py`,
`order_staging.py`, an order runner, or a workflow. Do not write Sheets, R2,
Cloudflare, IBKR, or Task Scheduler. Do not commit, stash, checkout, push,
install dependencies, rename, move, or delete files. Preserve adjusted-bar
engine/live conventions, staged-signal semantics, and all cap ordering.

## Intent

1. Add exactly `open_leg_mults: {"none_open": 0.8, "adds": 1.2}` to the
   execution dicts for Weak Close Decent Sznls and LT Trend ST OS. No other
   strategy carries the field.
2. Add one generic pure helper in `strategy_config.py`. It returns `adds` when
   `staged_count >= 2 OR prior_open_count >= 1`; otherwise `none_open`. Missing
   config returns 1.0. Counts must be non-negative and multipliers finite and
   positive.
3. Scanner: reuse the Portfolio snapshot loader. Same-day candidate counts are
   per `(Strategy_Name, Scan_Source)` so liquid and overflow tabs remain
   independent; prior filled-open count is strategy-wide across tickers and
   tiers. Apply the multiplier to every staged row before cross-strategy
   absolute clamps and IOB/Bear same-day derates. Stamp the state/counts/mult
   in `Sizing_Notes`. A snapshot failure is `{}` and therefore treats a
   one-signal day as solo 0.8x; same-day clusters still get 1.2x.
4. Engine: count post-blackout staged candidates per `(strategy pass, signal
   date)` before fills. At each signal, count only already-filled same-strategy
   positions open at signal time, strategy-wide (not same ticker); do not count
   unfilled working candidates. Apply the helper before the D3.3 overlap clamp
   and before later same-day/gap/cap layers. Every row on a two-plus candidate
   day gets 1.2x even if another candidate never fills.
5. Do not change the old dormant `ladder_multipliers` semantics. It remains
   ticker-specific and has no carriers.
6. `daily_portfolio_report.py` needs no edit: it consumes the shared strategy
   book and engine. OneDrive needs no edit: it consumes staged sizes.
7. Re-score once with a D3.4-only full-history engine replay at GRM 1.5 on top
   of verified D3.3. Report both book metrics and WCDS/LT affected-row results;
   do not claim the earlier research effect if the engine replay differs.

## Recon first

Write the exact-edit plan before source edits to
`artifacts/build_2026-09-05/d34_open_leg_mults/00_exact_edit_plan.md`.

## Verification

- `python -m pytest -q tests/test_open_leg_mults.py tests/test_clone_clamps.py tests/test_same_day_derate.py tests/test_base_bps_tilt.py`
- `python -m pytest -q tests/`
- Run the frozen-input D3.4-only replay at GRM 1.5. The control is the verified
  D3.3 tree with only both `open_leg_mults` fields removed. Compare 2010+ and
  2016-07+ trades, PnL, annual PnL, Sharpe, maxDD, worst day and worst-21d;
  reconcile every changed trade key and risk/PnL delta.
- Produce `artifacts/build_2026-09-05/d34_open_leg_mults/checks.json` by script
  with at least: `carrier_set_exact`, `config_exact`, `solo_08`, `cluster_12`,
  `prior_open_12`, `working_not_counted`, `staged_not_filled`,
  `scan_engine_order_match`, `narrow_tests_failed`, `full_tests_failed`,
  `replay_complete`, `replay_rows_affected`, and `external_writes`.

### Required repair after verifier round 1

Round 1 correctly returned FAIL on two scanner-path defects. The repair must
make the scanner multiply pre-round target risk and then floor by stop distance,
matching the engine, and must preserve every hard share ceiling already applied
by ADV or concurrent-notional sizing. Private calculation fields must be removed
before a staging row leaves the scanner. Add direct regressions for the observed
`600 / 2.70 * 0.8` quantity case (177, not 178) and a 100-share ADV-capped LT
overflow cluster (stays 100, not 120), then run a fresh independent verifier.
Round 2 found the boundary form: a 110-share ceiling above a 100-share base row
is still binding against the later 120-share cluster target. Therefore record
every computed positive ADV and concurrent-notional ceiling even when it is
initially slack, add both boundary regressions, and require verifier round 3.

## Report

Use section 6 of `docs/plan_2026-09-04.md` verbatim.
