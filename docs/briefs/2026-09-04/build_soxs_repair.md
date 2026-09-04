# Brief: build_soxs_repair (repair the SOXS island in the price cache and make the basis guard see it)

Date 2026-09-04. Plan: `docs/plan_2026-09-04.md` (section 0, D13, section 6). Type: BUILD (money-path data: a VERIFY agent follows; the mind uploads to R2, not you). DO NOT START until the mind says the sizing verify is complete (the replay there must see an unchanged price file).

## Decision and why
The 09-04 data recon measured SOXS in `data/master_prices.parquet` at 15.03x inflated from 2026-03-19 to 2026-05-22 (cliffs +1377% on 03-19 and -94.6% on 05-26), validated by the SOXL mirror (median |r_SOXS + r_SOXL| 0.25% off the cliff days; ratio 1.002 before, 1.000 after). The 2026-07-17 repair recorded in memory has been overwritten. Today the 126d and 252d ranks are only mildly wrong (23.0 vs 26.2; 22.6 vs 29.4) and the SMA200 is 3.5x too high; from about 2026-09-18 the 126-day return window enters the island and the ranks go wrong in earnest, on a ticker with live 3x Bear Fade legs. The updater's basis guard is a median over the ~83-row overlap, so the 12 island rows inside the re-fetch window cannot trip it, and the island can be re-imported nightly. Decision: (1) repair the island by dividing OHLC (and multiplying volume) over 2026-03-19..2026-05-22 by the factor that makes the SOXL mirror hold, written to the local parquet with a dated backup; (2) make the guard segment-aware (a per-day or short-window basis test in addition to the median) so a re-pull that disagrees with the cache by more than 2% on any contiguous segment of 5+ sessions is rejected for that ticker; (3) prove on a no-upload dry run of the updater that the repair survives one update cycle.

## Files you own
`scripts/update_master_prices.py`, a new `scripts/repair_price_island.py` (generic: ticker, start, end, factor or mirror-derived factor, backup, report), tests for both (`tests/test_update_master_prices*.py` if present, else new `tests/test_price_island_repair.py`), and the LOCAL `data/master_prices.parquet` (gitignored). Nothing else. R2 stays untouched by you.

## Hard rules
Section 0 of the plan. Never upload. The updater may be run ONLY with its no-upload / dry-run flags (read the code to find them; if none exists, add `--no-upload` and use it). Back up the parquet to `data/master_prices.parquet.bak_20260904_soxs` before writing. Never touch any other ticker's rows.

## Intent
1. `repair_price_island.py`: derive the factor from the mirror when given `--mirror SOXL` (median of the ratio needed so daily returns satisfy r_SOXS = -r_SOXL over the island, excluding the cliff days), else take `--factor`; apply to Open/High/Low/Close (divide) and Volume (multiply) for the ticker and date range; print before/after cliff checks (largest |return| in the window +/- 5 sessions), the mirror residual, and the rank_126/rank_252/SMA200 on the last session before and after; write the parquet atomically (temp file then replace) with the backup already taken.
2. Guard: in `update_master_prices.py`, alongside the existing median overlap test, add a segment test: over the overlap, compute the ratio new/old per session; if any run of 5+ consecutive sessions has |ratio - 1| > 2%, treat the vendor series as broken for that ticker (same action as today's `novel_cliff_dates` / basis-broken path: keep the cache, log the ticker and segment). Keep the cap on full re-pulls. Log lines must name the ticker, segment dates and max ratio.
3. Tests: island repair on a synthetic frame (factor and mirror modes, other tickers untouched, volume scaled, atomic write); guard on a synthetic overlap with a 12-row 15x segment (rejected) and with a uniform 1.5% dividend shift (accepted, as today).
4. Dry run: after the repair, run the updater for SOXS only (or the smallest scope the CLI allows) with no upload against the live yfinance feed, and show that the guard rejects the re-import if yfinance still serves the island, or that yfinance now agrees. Save the log.

## Recon first
`artifacts/build_2026-09-04/soxs_repair/00_plan.md`: the updater's fetch window, guard code, CLI flags, and how the 07-17 repair was done if any record exists (memory note path: `C:\Users\McKinley Slade\.claude\projects\C--Users-McKinley-Slade-dev-New-Seasonals\memory\soxs-yfinance-feed-bug.md`, read-only).

## Verification
`artifacts/build_2026-09-04/soxs_repair/checks.json`: `{"backup_path": "...", "factor_used": float, "island_rows": int, "max_abs_ret_window_before": float, "max_abs_ret_window_after": float, "mirror_residual_before": float, "mirror_residual_after": float, "rank126_after": float, "rank252_after": float, "sma200_after": float, "other_tickers_changed": 0, "guard_rejects_synthetic_island": bool, "guard_accepts_dividend_shift": bool, "dry_run_outcome": "rejected|agrees|other", "tests_failed": int, "uploaded": false}`.
No screenshots.

## Report
Section 6 format. Handoff: the exact upload command for the mind (`cache_io.upload_from_local`) and what to check on the next nightly run's log.
