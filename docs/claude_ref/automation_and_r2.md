# Local-primary automation and Cloudflare R2

Moved close to verbatim from CLAUDE.md on 2026-09-23. CLAUDE.md keeps the live rules;
this file keeps the history, evidence and detail. Path fixes applied on the move.

## Local-primary Automation + Cloudflare R2 (effective 2026-08-28)

The production data, scan, sleeve, report, and weekly jobs run on this machine through seven Windows Task Scheduler entries backed by a dedicated clean worktree, immutable Git fallback tag, and dedicated virtual environment. The development checkout is never the production runtime, and scheduled runs never update their own code.

| Task pipeline | Eastern schedule | Scope |
|---|---|---|
| `premarket` | Weekdays 04:10 | CBOE, NYSE breadth amendments (`breadth_am`, after `cboe_am`), settled prices, risk correction, event sleeve, AM scan, cloud site handoffs |
| `discretionary` | Weekdays 08:35 | Research-only Discretionary Focus |
| `execution` | Weekdays 16:30 | Live-position execution email |
| `postclose` | Weekdays 17:10 | PM prices, NYSE breadth (`breadth_pm`, between `master_prices_pm` and `risk_pm` so the evening dial is floored), risk/fills (verify + broker harvest)/earnings/portfolio/CBOE/trend/intraday/scan/macro/sites |
| `indicator` | Monday 03:00 | Backtester indicator cache |
| `weekly-rundown` | Sunday 08:00 | Weekly PDF email |
| `health` | Weekdays 07:30 | R2 receipts, data, delivery, and runtime logs |

`scripts/automation_supervisor.py` owns component-level R2 receipts, leases, strict producer validation, and GitHub fallback. Successful and indeterminate receipts block duplicates. Non-rerun-safe commands are durably marked `indeterminate` immediately before external side effects; crashes and ambiguous outcomes require explicit operator resolution and never blind-dispatch a second copy.

A command may declare `degraded_exit_codes` (2026-09-21, first carrier `breadth_pm`). Such an exit does not fail its job: the run continues and the receipt is written `status=success, health_status=degraded`, which `effective_status` reports as `degraded` and `repo_health_check` reports as WARN. Use it only for a shortfall the consumer already falls back on by itself, never for an outcome nothing downstream can detect.

Migrated child workflows are `workflow_dispatch`-only backups. `.github/workflows/local_automation_fallback.yml` is their sole cron and dispatches only missing/retryable receipt components during bounded ET windows. Production private/shared site builds remain cloud-only; local pipelines publish bounded canonical inputs to R2 and dispatch the site workflows. See `docs/local_automation_task_scheduler.md` for install, cutover, status, and rollback.


### R2 secrets (machine `.env` and GitHub backup secrets)
- `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET=seasonals-cache`

### Bucket contents (key-value)
- `master_prices.parquet` — full ~2000 ticker × 25-yr OHLCV (~50-200 MB). Read by `daily_scan` (ALL scopes — **cache-first for every ticker**, incl. the liquid + 3x-ETF universes; yfinance is only a fallback for names the cache lacks, e.g. carets/delisted) and `daily_portfolio_report.py`. As of 2026-06-11 the 42 LEV3X names (DUST/JDST/TQQQ/…) were backfilled in so the liquid scan no longer depends on a live pre-market yfinance pull (that pull returned a stale bar on 2026-06-11 and silently zeroed the liquid tier). Written by `update_master_prices.yml` twice on weekdays (AM via local workflow_dispatch ~4:17 AM ET + PM via 21:10 UTC cron = 17:10 ET EDT / 16:10 ET EST, post-close year-round; it was 20:30 UTC until the EST slot landed 15:30 ET, i.e. BEFORE the close, and appended yfinance's in-progress bar as the canonical daily close); its universe = whatever tickers already exist in the parquet, so backfilled names are auto-maintained. Pre-market runs pass `--exclude-today` so yfinance placeholder bars never enter the cache.
- `earnings_calendar.parquet` — FMP-backfilled (117k rows, 946 tickers). Read by `daily_scan` (any scope, OVS filter) and `daily_portfolio_report.py`. Written by `build_earnings_calendar.yml` weekdays at 21:30 UTC + the local belt-and-suspenders entry at the same slot.
- `intraday/15min/{TICKER}.parquet` + `intraday/15min/_meta.parquet` — 15min OHLCV cache. Historical depth backfilled from FMP (2003-present), ongoing maintenance via yfinance (60d rolling, no API key). Target universe is `LIQUID_PLUS_COMMODITIES` (~197 tickers, ~3 MB each, ~600 MB total). Read by `intraday_data.py` (lazy R2 refresh on stale local copies, 18h staleness window) which feeds Day Trade Limit modes in `pages/backtester.py`. Written by `update_intraday_prices.yml` weekdays at 20:45 UTC. Caret tickers (^GSPC, ^NDX) excluded — FMP doesn't serve them. Full architecture in `docs/intraday_data_plan.md`.

### `cache_io.py` API
```python
from cache_io import upload_from_local, download_to_local, is_configured

upload_from_local("data/foo.parquet", "foo.parquet")   # local → R2
download_to_local("foo.parquet", "data/foo.parquet")   # R2 → local
is_configured()                                          # bool: R2_* env vars set?
```
Both helpers no-op gracefully when R2 isn't configured (returns False, prints a notice). ASCII-only output to avoid Windows cp1252 crashes when running locally.


### Sunday Pipeline
1. **8:00 AM ET (local-primary)**: `weekly_market_rundown.py` generates the tabloid landscape PDF and emails it. `weekly_rundown.yml` is receipt-gated backup only.

**Radar digest retired 2026-08-04, path removed 2026-08-18.** `radar_weekly_summary.py` and `data/radar_weekly_summary.md` were deleted in the retirement; the generating agent (`trig_015YZLdjj3LxvyY25fnq1zch` in radar-briefings) was disabled at the 2026-07-13 cutover. The rundown kept reading the file it no longer refreshed, so the Sunday email shipped a digest frozen at 2026-07-12 for three weeks while the workflow reported green. The whole body path is now GONE from `weekly_market_rundown.py` (`RADAR_SUMMARY_PATH`, `_build_email_body()`, the call site, and the `markdown` / `MIMEText` imports it alone needed): the email is PDF-only by construction rather than by fallback. `scripts/run_radar_weekly.ps1` and `scripts/setup_radar_weekly_task.ps1` are deleted; `RadarWeeklySummary` was never actually registered in Task Scheduler. The cloud routine stays disabled and can only be deleted from https://claude.ai/code/routines.
