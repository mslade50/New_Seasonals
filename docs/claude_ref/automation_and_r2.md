# Local-primary automation and Cloudflare R2

Moved close to verbatim from CLAUDE.md on 2026-09-23. CLAUDE.md keeps the live rules;
this file keeps the history, evidence and detail. Path fixes applied on the move.

## Local-primary Automation + Cloudflare R2 (effective 2026-08-28)

The production data, scan, sleeve, report, and weekly jobs run on this machine through seven Windows Task Scheduler entries backed by a dedicated clean worktree, immutable Git fallback tag, and dedicated virtual environment. The development checkout is never the production runtime, and scheduled runs never update their own code.

| Task pipeline | Eastern schedule | Scope |
|---|---|---|
| `premarket` | Weekdays 04:10 | CBOE, NYSE breadth amendments (`breadth_am`, after `cboe_am`), settled prices, risk correction, event sleeve, AM scan, cloud site handoffs |
| `discretionary` | Weekdays 08:35 | Retired 2026-09-23 (`2ef9fa21`): task disabled, pipeline kept as an inert entry with no jobs. See `docs/fmp_retirement_2026-09-23.md` |
| `execution` | Weekdays 16:30 | Live-position execution email |
| `postclose` | Weekdays 17:10 | PM prices, NYSE breadth (`breadth_pm`, between `master_prices_pm` and `risk_pm` so the evening dial is floored), risk/fills (verify + broker harvest)/earnings/portfolio/CBOE/trend/intraday/scan/macro/sites |
| `indicator` | Monday 03:00 | Backtester indicator cache |
| `weekly-rundown` | Sunday 08:00 | Weekly PDF email |
| `health` | Weekdays 07:30 | R2 receipts, data, delivery, and runtime logs |

`scripts/automation_supervisor.py` owns component-level R2 receipts, leases, strict producer validation, and GitHub fallback. Successful and indeterminate receipts block duplicates. Non-rerun-safe commands are durably marked `indeterminate` immediately before external side effects; crashes and ambiguous outcomes require explicit operator resolution and never blind-dispatch a second copy.

A command may declare `degraded_exit_codes` (2026-09-21, first carrier `breadth_pm`). Such an exit does not fail its job: the run continues and the receipt is written `status=success, health_status=degraded`, which `effective_status` reports as `degraded` and `repo_health_check` reports as WARN. Use it only for a shortfall the consumer already falls back on by itself, never for an outcome nothing downstream can detect.

Migrated child workflows are `workflow_dispatch`-only backups. `.github/workflows/local_automation_fallback.yml` is their sole cron and dispatches only missing/retryable receipt components during bounded ET windows. Production private/shared site builds remain cloud-only; local pipelines publish bounded canonical inputs to R2 and dispatch the site workflows. See `docs/local_automation_task_scheduler.md` for install, cutover, status, and rollback.

### Current runtime pin and how releases happen

Live as of 2026-10-01 10:11 ET: runtime `New_Seasonals-automation-runtime-v9`, branch `codex/local-primary-runtime-v9-20260912`, pin `a7f49f00865fdaf6ef598845b6a1504a9899478b`, tag `automation-runtime-2026-10-01.earnings-alpha-only`. `AUTOMATION_RUNTIME_REF` in `local_automation_fallback.yml` names the same tag (main commit `f65eb5db`), so the GitHub fallback and the local runtime run the same code. Previous pin `3ee156c3`, tag `automation-runtime-2026-09-30.earnings-retries-issuer-review.v2`. Record: `docs/earnings_alpha_only_release_2026-10-01.md`.

The runtime is a curated branch, not main. A release cherry-picks named main commits onto the current runtime commit, runs the focused tests there, tags the new runtime commit, pushes the tag, edits the local marker (pin, fallback ref and a dated `*_release_at_utc` key) and moves `AUTOMATION_RUNTIME_REF` on main to the new tag. Releases follow the cutover cadence in `docs/local_automation_task_scheduler.md` (never 04:00 to 09:35 ET, clean slots about 10:00 to 16:00 ET or after `postclose` finishes). A commit on main is not live until a release carries it.

On main, not yet released (as of 2026-10-01):

- `7cb1560a` supervisor failure email. After `run` / `run-pipeline`, one email when any counted job ends failed, indeterminate or blocked; `health` emails its SUMMARY block on any FAIL or when the lock is held. Gmail SMTP via `EMAIL_USER` / `EMAIL_PASS`; recipients from `NEW_SEASONALS_ALERT_RECIPIENTS` (default is the operator address the scan email already uses); off switch `NEW_SEASONALS_ALERT_EMAIL=0`. Sent after the supervisor lock is released, secrets redacted, never changes an exit code, nothing on dry-run, plan, status or fallback-due.
- `7cb1560a` `scan_am` pre-step `scripts/ensure_earnings_calendar.py`, before the side-effect boundary: checks the canonical R2 calendar with `validate_freshness`, runs the normal producer once if stale, exits 1 without calling the producer if R2 is unreadable. Same step in `.github/workflows/daily_screener.yml` for the AM bookend. The `scan_am` local lease grows from 1800 s to 3000 s. If the local step and the GitHub fallback both fail, `scan_am` still ends `indeterminate` and the 05:45 retry skips it. A morning repair spends that date's Alpha request, so the 17:10 job reuses the morning snapshot.
- `9344b8b0` stale Discretionary-retirement automation tests fixed (the two failures on the runtime line); `scripts/repo_health_check.py` resolves the live runtime as the highest `New_Seasonals-automation-runtime-vN` with a marker, reads data from the pinned runtime when run from another checkout, and reports `ib_insync`-only collection errors as a warning.
- `5c63a3a4` (2026-09-24) `cache_io` lock-safe download.

### Earnings calendar job (`earnings_and_grades`, postclose 17:10)

Live rule since the 2026-10-01 release: Alpha Vantage only. `config/earnings_calendar.json` sets provider `alpha`, `alpha_fallback` `stop`, `confirmation_provider` `calendar`; no FMP bootstrap, confirmation or fallback. The job requires `ALPHA_VANTAGE_API_KEY` (plus the R2 keys). The receipt ID keeps the old name; grades were retired 2026-09-23. Producer: `scripts/refresh_earnings_calendar.py`. GitHub backup: `build_earnings_calendar.yml`, same script, same tag.

- One Alpha request per NY date. `alpha_calendar_snapshot.py` claims R2 `provider_snapshots/alpha_earnings/<NY date>.json`, shared by the 17:10 producer, the 06:30 monitor and any manual run. Transient failures retry after 60 s and 180 s; a terminal failure locks the date.
- Events that vanish from Alpha (past, same-day or near-term) are kept with `event_status` `schedule_unverified`. A new date for the same fiscal period replaces the old one.
- Guards (`8bfe68b9`): a re-dated elapsed or same-day unconfirmed event does not publish both dates; `coverage_gate` ignores forward `schedule_unverified` rows on both sides (`NEAR_TERM_COVERAGE_MIN` 0.80); `forward_shrink_gate` refuses when fresh Alpha `expected` rows fall below `FORWARD_SHRINK_MIN` 0.80 of the prior's; the receipt's `unverified_schedule_rows` counts only rows dated on or after `as_of`.
- Known gaps: a row already `schedule_unverified` from an earlier night is not superseded if Alpha re-lists that period; a single far-dated vanished event is dropped silently; a relabelled fiscal period publishes both dates; no actuals, EPS or surprise values are populated after the cutover (only `pages/backtester.py` and filters that are off use them); the universe is still the symbol_master frozen 2026-06-05.
- Single point of failure: the calendar gates both scans, `scan_pm` through its supervisor dependency on `earnings_and_grades` and `scan_am` through the scanner's freshness gate at `daily_scan.py:3005`. See `docs/incidents/2026-10-01_earnings_fmp_scan_skip.md`.

Aligned sites, change together: `config/earnings_calendar.json`, `scripts/refresh_earnings_calendar.py`, `earnings_calendar_provider.py`, `alpha_calendar_snapshot.py`, the `earnings_and_grades` and `scan_am` job specs in `scripts/automation_supervisor.py`, `scripts/ensure_earnings_calendar.py` (main only until released), `.github/workflows/build_earnings_calendar.yml`, `.github/workflows/daily_screener.yml`, and `AUTOMATION_RUNTIME_REF` in `.github/workflows/local_automation_fallback.yml`.


### R2 secrets (machine `.env` and GitHub backup secrets)
- `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET=seasonals-cache`

### Bucket contents (key-value)
- `master_prices.parquet` — full ~2000 ticker × 25-yr OHLCV (~50-200 MB). Read by `daily_scan` (ALL scopes — **cache-first for every ticker**, incl. the liquid + 3x-ETF universes; yfinance is only a fallback for names the cache lacks, e.g. carets/delisted) and `daily_portfolio_report.py`. As of 2026-06-11 the 42 LEV3X names (DUST/JDST/TQQQ/…) were backfilled in so the liquid scan no longer depends on a live pre-market yfinance pull (that pull returned a stale bar on 2026-06-11 and silently zeroed the liquid tier). Written by `update_master_prices.yml` twice on weekdays (AM via local workflow_dispatch ~4:17 AM ET + PM via 21:10 UTC cron = 17:10 ET EDT / 16:10 ET EST, post-close year-round; it was 20:30 UTC until the EST slot landed 15:30 ET, i.e. BEFORE the close, and appended yfinance's in-progress bar as the canonical daily close); its universe = whatever tickers already exist in the parquet, so backfilled names are auto-maintained. Pre-market runs pass `--exclude-today` so yfinance placeholder bars never enter the cache.
- `earnings_calendar.parquet`: Alpha Vantage only since 2026-10-01 (147,801 rows on the 2026-10-01 publish; history before the cutover was FMP-backfilled and is preserved). Read by `daily_scan` (any scope, OVS filter) and `daily_portfolio_report.py`. Written by the local `earnings_and_grades` job in `postclose` (17:10 ET), with `build_earnings_calendar.yml` as the receipt-gated GitHub backup. Conditional upload against the prior ETag, then readback. Job detail above.
- `provider_snapshots/alpha_earnings/<NY date>.json`: the day's single Alpha earnings response and its claim, written by `alpha_calendar_snapshot.py`. Read by the producer, the 06:30 monitor and `scripts/prepare_earnings_issuer_review.py --alpha-r2`.
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
