# Brief: recon_data_window (price-cache adjustment window and the scan-vs-ledger splits)

Date 2026-09-04. Plan: `docs/plan_2026-09-04.md` (read section 0 first). Type: RECON. You write nothing outside your scratch folder.

## Decision and why
In one fortnight the scan fired SPG twice while the ledger never fired it, and staged a UNH limit 126 bps away from the ledger's. The due-diligence report called this a data-vintage problem with real money on it. The most likely mechanism is structural: `scripts/update_master_prices.py` re-adjusts only a rolling window (default 120 days), so a dividend going ex re-scales rows inside the window and leaves older rows on the old basis, and the scan and the ledger build may read the file at different times or from different copies. The mind will decide the fix (print-and-flag vs a cache change) only after the mechanism is measured. This brief measures it and changes nothing.

## Files you own
None in the repo. Scratch only: `artifacts/recon_2026-09-04/data_window/` (create it).

## Hard rules
Section 0 of the plan. Do not run the updater. A SMALL live yfinance pull (at most 40 tickers, daily bars, read-only) is allowed for the diagnostic in intent item 2 only; handle the MultiIndex columns exactly as CLAUDE.md's "yfinance MultiIndex Bug" section says. Never write to R2. Reading R2 is allowed only through `cache_io.download_to_local` into your scratch folder, if creds are present in `.env`; otherwise skip and say so.

## Intent
1. Cache health on `data/master_prices.parquet`: schema, rows, tickers, date range, last-bar-per-ticker distribution, tickers in `LIQUID_PLUS_COMMODITIES` or the LEV3X lists with a last bar older than 3 sessions before 2026-09-03, duplicate (ticker, date) rows, non-positive prices, High < Low or Close outside [Low, High], single-day |return| > 40% in the last 250 sessions, and whether the SOXS series before 2026-05-26 still shows the known feed bug (memory note: `C:\Users\McKinley Slade\.claude\projects\C--Users-McKinley-Slade-dev-New-Seasonals\memory\soxs-yfinance-feed-bug.md`, read-only).
2. The adjustment window. Read `scripts/update_master_prices.py` in full and describe precisely what is re-pulled and re-written on each run, what `--max-lookback-days` bounds, and what the basis-change guard (full re-pull above 2% overlap divergence, cap 40 per run) and the `novel_cliff_dates` guard do. Then measure: for 30 high-yield names present in the liquid universe (SPY, XLU, XLP, VNQ, TLT, HYG, LQD, KO, PG, MO, VZ, T, XOM, CVX, JNJ, PFE, IBM, O, EPD, ET and ten more you pick by yield), compare the cache's adjusted Close against a fresh fully-adjusted yfinance series over the last 400 sessions, and report per ticker the max ratio deviation and the date it sits at. State whether there is a step at the window boundary. Then reason about which scan inputs span the boundary (ATR 14, ranks 2/5/10/21/126/252, SMA 200) and how large the induced error is at the observed deviations.
3. Vintages. Establish which copy of master_prices each consumer reads and when: the local pinned runtime's premarket pull (find the pull step in `scripts/automation_supervisor.py` / the pipeline catalog), the PM scan, the ledger build in `deploy_site.yml` (R2 pull), `daily_portfolio_report.py`, `scripts/build_trade_ledger.py`. If two consumers can read different vintages on the same day, say which and by how many hours.
4. Reproduce SPG and UNH. Find the scan rows (scan logs under `scripts/logs/` and the runtime worktrees, `artifacts/`, Trade_Signals_Log exports, `data/` staging snapshots, `scratch/live_vs_ledger_2026/`, the sizing DD's dd_live folder) and the ledger's inputs for those dates. Recompute the OLV signal and limit for SPG and UNH on the scan's bar and on the ledger's bar; show the two closes, two ATRs, two limits, and which input differs. If the artifacts do not exist locally, say exactly what is missing.
5. Guard logs: find the basis-change guard's and cliff guard's output in the last two weeks of logs; count tickers hitting each per run.
6. Freshness of the other caches: `earnings_calendar.parquet` (coverage of liquid + overflow universes, forward dates, last refresh), `cboe_putcall.parquet` (last date, lag-1), `rd2_fragility.parquet` (append-only PIT: dates monotonic, no trading-day gaps since 2026-07-02, metadata keys present, last row; diff the working copy against `git show HEAD:data/rd2_fragility.parquet` and confirm only the newest rows differ).
7. Live-universe tickers absent from the parquet (every strategy_config universe, `^GSPC ^NDX`, SVXY, LEV3X names, event and trend sleeve names): these depend on a pre-market yfinance pull and were the cause of a silently zeroed tier on 2026-06-11.

## Recon first
`artifacts/recon_2026-09-04/data_window/00_plan.md` first.

## Verification
`artifacts/recon_2026-09-04/data_window/checks.json` from a script:
`{"rows": int, "tickers": int, "last_date": "YYYY-MM-DD", "stale_liquid_tickers": [..], "dup_rows": int, "bad_ohlc_rows": int, "big_moves_250d": [{"ticker":..., "date":..., "ret":...}], "soxs_pre_0526_bug_present": bool, "window_days": int, "basis_deviation": [{"ticker":..., "max_ratio_dev":..., "at_date":...}], "step_at_boundary": bool, "consumers": [{"name":..., "source":..., "time_et":...}], "spg_unh_reproduced": bool, "spg_unh_cause": "...", "fragility_pit_ok": bool, "missing_live_tickers": [..]}`.
No screenshots required.

## Report
Section 6 format. Findings ranked by money at risk. Handoff: recommend ONE of (a) print scan close/ATR beside the R2 vintage and flag limit gaps > 25 bps, (b) widen or remove the re-adjust window, (c) move the cache to raw OHLCV with read-time adjustment (CLAUDE.md's deferred "Tier 2"), with the cost of each in hours and the residual error each leaves.
