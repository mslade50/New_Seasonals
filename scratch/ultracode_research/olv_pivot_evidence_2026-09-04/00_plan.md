# study_olv_pivot_evidence — exact plan (2026-09-04)

Read-only research. Everything written lives in this folder. No config edits,
no ledger rebuild for production, no upload, no price-updater run, no commit.

## Recon findings that shape the scripts

- Policy lives in `strategy_config.py` OLV `execution['pivot_entry_policy']`
  (version `olv_close_pivot_40_v2_20260901`, 40/40, max_source_age_bars 252,
  default 0.25, rules gt5 skip / 4-5 -> 0.75 / 2-3 -> 0.50). Resolver shared by
  scan + engine: `olv_pivot_entry.resolve_olv_pivot_entry(_from_row)`; indicator
  columns from `olv_pivot_entry.causal_close_pivot_context` wired in
  `indicators.py` line ~396. Engine hook: `pages/strat_backtester.py`
  ~1174-1196 (runs before sizing; `skip` -> `continue`). Documented off value:
  `enabled: False` (kill switch; tests `test_backtester_kill_switch_restores_legacy_entry`).
  With `enabled=False` the resolver still returns `proposed_action` /
  `matched_rule` for audit, which is exactly what tags "affected signals".
- The "359 completed-fill sample, +8.68R" text entered with commit a3036527
  (2026-09-01). The numbers come from `artifacts/olv-pivot-age-252-20260901/`
  (gitignored: `.gitignore:44 artifacts/`), `summary.json` ->
  `policy_counterfactual`: age-capped policy 247.474R vs unlimited-age policy
  238.791R = +8.68R. That is policy-v2 vs policy-v1, NOT policy vs no-policy.
  The no-policy baseline on the same 359 fills is 253.354R (avgR 0.706), from
  `artifacts/olv-level-proximity-20260831/summary.json`. Sample = OLV rows of
  the 2026-08-31 13:07Z ledger (gha:33394874677), censored rows removed.
- Current ledger (gha:33852895307, 2026-09-04 08:24Z) already carries the
  policy: 316 OLV rows (88 Liquid / 228 Overflow) with Pivot* audit columns.
- Cache `data/master_prices.parquet`: 1120 tickers, last bar 2026-09-03.
  Machine time at start 07:50 EDT 2026-09-04 (pre-open), so the "09-04" live
  view is the 2026-09-03 close (what the 09-03 PM and 09-04 AM scans read) and
  "09-03" is the 2026-09-02 close (09-03 AM scan).
- Other builders are editing `strategy_config.py`, `daily_scan.py`,
  `pages/strat_backtester.py`. At recon time the only working-tree diff in the
  engine was the overlay-lab `portfolio_overlay_names` plumbing (no OLV path).
  Script 02 records `git rev-parse HEAD`, `git diff --stat` and the OLV
  pivot block at run time into `tree_state.txt`.

## Scripts (run in order)

1. `01_evidence_search.py` -> `evidence_search.json`
   git log -S searches (8.68 / ClosePivot / pivot_entry_policy), grep hits,
   and the quoted numbers from the artifacts summaries (paths cited).
2. `02_engine_replay.py` -> `replay_with_policy.csv`, `replay_without_policy.csv`,
   `candidates_pivot_audit.csv`, `replay_summary.json`, `tree_state.txt`,
   `ledger_parity.json`
   Book = `daily_portfolio_report.build_full_strategy_book()` filtered to
   "Oversold Low Volume" (liquid pass + overflow pass, OLV overflow bps
   override applied there). Data = `data_provider.get_history` from 2000-01-01.
   `precompute_all_indicators` (disk cache) -> `generate_candidates_fast` from
   2003-01-01 (ledger BT_START, so the recency-ladder warm-up matches the
   ledger) -> two `process_signals_fast` flat passes, cap_bps=250,
   overflow_active=True, flat_sizing=True, pooled caps None (ledger prod
   settings). Arm A = as configured; arm B = deepcopy with
   `pivot_entry_policy['enabled']=False`. Reporting windows: signal dates in
   [era, 2026-08-31] for eras 2010-01-01, 2016-07-01, 2024-01-01; censored
   rows (Exit Type Time and Exit Date < Time Stop) excluded from R stats.
   Affected signals = candidates whose proposed matched_rule != default.
   Per-signal diff = R_with (0 if skipped/unfilled) - R_without (0 if
   unfilled); clustered t by signal date: t = sum(d) / sqrt(sum_c S_c^2).
   Parity: with-policy arm vs ledger OLV rows on (Ticker, Signal Date,
   Entry Price, Exit Date).
3. `yf_pull.py` (helper) -> `yf_adjusted_3y.parquet`, `yf_pull_meta.json`
   ONE `yfinance.download` of the OLV liquid universe
   (`LIQUID_PLUS_COMMODITIES`, 3 years, auto_adjust=True), MultiIndex handled
   per CLAUDE.md, cached to parquet so later scripts never re-pull.
4. `03_basis_stability.py` -> `basis_signals.csv`, `basis_top10_shifts.csv`,
   `basis_summary.json`
   Liquid-tier OLV signals from the audit with signal date >= 2023-09-04.
   Three contexts per signal: cache full history (production), cache
   truncated to the yf window (same-window control), yf fully adjusted.
   Band = matched_rule (skip counts as its own band). Primary flip stat =
   cache-window vs yf on signals with >= 300 yf bars before the signal
   (full pivot context on both sides); the production-vs-yf view and the
   window-limited rows are reported separately. Ten largest |d_yf - d_cache|
   with the implied dividend from the cache/yf close-ratio step inside
   (pivot source date, signal date].
5. `04_live_status.py` -> `live_status.json`, `live_status.csv`
   For closes 2026-09-02 and 2026-09-03, every liquid OLV name: band on the
   cache and on yf, fired flag from the audit, flips.
6. `05_checks.py` -> `checks.json` (the brief's keys, produced by script).
