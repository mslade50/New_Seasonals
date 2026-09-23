# Retired automation (GitHub-first era) and the retired radar digest

Moved close to verbatim from CLAUDE.md on 2026-09-23. CLAUDE.md keeps the live rules;
this file keeps the history, evidence and detail. Path fixes applied on the move.

## Historical GitHub-first Schedule (retired 2026-08-28; do not operate from this table)

The table below is retained only for incident archaeology. The child schedules are disabled; the local-primary catalog above is authoritative. Order staging remains local and IBKR-bound.

| Workflow file | Schedule | What it does |
|---|---|---|
| `daily_screener.yml` | Weekdays 2x: AM via local workflow_dispatch at 4:47 AM ET (fallback GHA cron at 10:30 UTC, auto-skipped if dispatch succeeded today) + PM cron at 22:00 UTC | Unified scan, both runs `--scope=all` (full liquid + overflow, ~7-10 min). AM run also writes `data/exposure_state.json` and commits it back to main. Intraday MOC slots were retired when the strategy book lost its last Signal Close entry; restore them if MOC strategies are added back. |
| `build_earnings_calendar.yml` | Weekdays 21:30 UTC (5:30 PM ET) | FMP `/stable/earnings` pull → writes `data/earnings_calendar.parquet` → uploads to R2. Local `EarningsCalendarRefresh` Task Scheduler entry mirrors this for redundancy (last write wins). |
| `update_master_prices.yml` | Weekdays 2x: AM via local workflow_dispatch at 4:17 AM ET (fallback GHA cron at 9:30 UTC, auto-skipped if dispatch succeeded today) + PM cron at 21:10 UTC (17:10 ET EDT / 16:10 ET EST — post-close in BOTH, which 20:30 UTC was not) | Pulls `master_prices.parquet` from R2, fetches today's bars from yfinance for ~2000 tickers, appends, dedupes, writes back to R2. PM cron pulls today's close; every other trigger (AM dispatch, AM fallback cron, manual dispatch) passes `--exclude-today`. |
| `update_intraday_prices.yml` | Weekdays 21:45 UTC (5:45 PM EDT / 4:45 PM EST — moved from 20:45 on 2026-08-12, which was 3:45 PM EST = pre-close in winter) | Pulls per-ticker 15min parquets + meta from R2, runs `scripts/update_intraday_yfinance.py --upload` — fetches recent bars from yfinance for every ticker in meta, converts UTC→ET, appends, dedupes, writes back. yfinance has 60d rolling intraday history so this must run at least every ~50 days to avoid gaps; weekday cadence is fine in practice. |
| `portfolio_report.yml` | Weekdays 21:30 UTC (5:30 PM ET) | Pulls master_prices + earnings caches from R2, runs `daily_portfolio_report.py`, sends HTML email + writes Portfolio Sheets tab. |
| `bootstrap_caches.yml` | workflow_dispatch only | One-shot: builds `master_prices.parquet` from scratch via yfinance (~10-15 min for ~2000 tickers, 25-yr history) and uploads to R2. Used to seed the bucket (already run during Phase 2 setup). |
| `risk_report.yml` | Weekdays 2x: PM cron 21:15 UTC (5:15 PM ET, full run w/ email) + AM correction at 4:30 AM ET via local workflow_dispatch (fallback GHA cron 9:00 UTC), `--data-only --refresh-last` | Daily risk dashboard email (fragility dials + signals + forward returns). Writes `data/rd2_fragility.parquet` APPEND-ONLY (since 2026-07-02): history is frozen point-in-time, only new dates append. The PM run's newest row comes from a just-closed yfinance bar that can be provisional, so the AM correction (2026-07-17) refreshes ONLY the last session's row with settled prices before daily_scan sizes off it at ~4:47 AM — `merge_fragility_history(refresh_from=prev_bday)`; older rows never mutate. Same correction re-evaluates the dial-sleeve paper track's provisional day (dial_sleeve rollback) and the simple-dial shadow. Guards: `tests/test_fragility_append.py`, `tests/test_dial_sleeve.py`. This series sizes live orders (`frag_risk_bands`) — do not revert to full rewrites; recompute vintages drifted up to ~7 pts on the 63d dial. |
| `verify_fills.yml` | Weekdays 21:15 UTC | Post-close fill verification — updates Trade_Signals_Log. |
| `deploy_site.yml` | Reusable workflow (`workflow_call`), invoked by the `deploy-site` job at the tail of `daily_screener.yml` (`needs: run-scanner`) so it runs in the SAME run, right after the scan succeeds — 2x/trading day (after the ~4:47 AM ET dispatch scan and the PM bookend). Replaced the old best-effort `workflow_run` chain, which was silently not firing. A skipped (AM fallback) or failed scan skips the deploy and the prior deploy stays up. `workflow_dispatch` retained for manual rebuilds. | Builds + deploys the private analytics site to Cloudflare Pages (behind Cloudflare Access). Pipeline: R2 caches → `scripts/build_trade_ledger.py` (full-history ledger) → `scripts/build_signal_charts.py --all --upload --skip-existing` (renders only NEW per-trade charts to R2, best effort) → `daily_seasonal_ideas.py` (REQUIRED — the step is fail-loud in the workflow as a freshness input; doc said best-effort until 2026-08-12) → `scripts/build_risk_json.py` (best effort) → `scripts/build_site.py` (JSON payloads + `site/` assets → `dist/`) → wrangler Pages deploy (config-driven via `wrangler.toml`; no positional dir, so the CHARTS R2 binding applies). Needs `CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID` secrets. One-time setup: `docs/private_site_setup.md`. Operational runbook (failure modes, decisions log,
trigger chain, out-of-repo file map): `docs/site_runbook.html`. |
| `weekly_rundown.yml` | Sundays 12:00 UTC (8 AM EDT / 7 AM EST) | Tabloid PDF with all risk charts. |
| `trend_sleeve.yml` | Weekdays 21:35 UTC (no-ops except the month's last trading day) | Monthly trend-following ballast rebalance — writes MOO orders to the `Trend` Sheets tab + state to R2. See "Trend Sleeve" section. |
| `execution_report.yml` | Weekdays 20:30 AND 21:30 UTC — `daily_execution_report.py` gates on WHICH cron fired (`GHA_SCHEDULE` = `github.event.schedule`) + the date's DST regime, so exactly one sends at ~4:30 PM ET year-round even when GHA cron lag starts the run an hour+ late (the old hour==16 gate silently dropped the 2026-07-08/09 emails). Cron strings must match `EDT_CRON`/`EST_CRON` in the script; unknown cron fails open (sends). | Nightly email of LIVE primary-account positions (mirrors the site Execution tab). Pulls the execution-broker DO's `/book` snapshot (Bearer `STATUS_TOKEN`; URL via `EXEC_BROKER_URL` secret, defaults to the workers.dev URL), excludes OPT rows, and enriches each position from its own working exit legs: target = closing LMT `lmt`, stop = closing STP `aux` (NA if none), time stop = closing MKT leg's `goodAfterTime` date, strategy = 3rd pipe field of any leg's `orderRef` (requires `order_ref` in `book_snapshot.py` — added 2026-07-08 in `OneDrive\trading_ibkr`). No strategy-tagged legs → "Event Sleeve (V4)" (symbol in `event_sleeve_state.json` on R2 — event orders are parent-only, added 2026-08-21) → "Trend Sleeve" (symbol in `trend_sleeve_state.json` on R2) else "Discretionary"; both sleeve states are reconciled against the live book (event: shortfall-only, entered-today skipped). Recipients: repo variable `EXECUTION_REPORT_RECIPIENTS` (comma-separated; defaults to mckinleyslade@gmail.com). Guard: `tests/test_execution_report.py`. |

### Historical Task Scheduler State (retired at local-primary cutover)

| Task | State | Notes |
|---|---|---|
| `EarningsCalendarRefresh` | NOT ACTUALLY REGISTERED (found 2026-08-12) | Documented as belt-and-suspenders for the GHA equivalent, but no such task exists in Task Scheduler and the wrapper's hardcoded `C:\Users\mckin` path was dead post-migration (fixed to `$PSScriptRoot`-relative 2026-08-12). Re-register if the redundancy is still wanted; until then the GHA writer is the ONLY writer. |
| `Trigger Update Master Prices (GHA workflow_dispatch)` | Enabled | Weekdays 4:17 AM ET — fires `update_master_prices.yml` via the GitHub REST API to bypass shared-cron queue lag at 8-9 UTC. See "AM Trigger Architecture" below. |
| `Trigger Daily Screener (GHA workflow_dispatch)` | Enabled | Weekdays 4:47 AM ET, 30 min after the parquet trigger — fires `daily_screener.yml` via the GitHub REST API. Same mechanism. |
| `Trigger CBOE Put-Call (GHA workflow_dispatch)` | Enabled (Interactive logon) | Weekdays 4:10 AM ET, FIRST in the pre-market chain — fires `update_cboe_putcall.yml` so the put/call cache holds the prior session before the risk correction, scan and pitch read it. The 21:30 UTC run fires before CBOE publishes and can only ever collect D-1. Scripts: `C:\Scripts\trigger_cboe_putcall.ps1` + `register_cboe_putcall_trigger.ps1`. NOTE: no `/` in the task name — Task Scheduler treats it as a folder separator. Registered Interactive (like the risk-correction trigger) because S4U needs an elevated shell; fires whenever the machine is on and logged in, locked included. |
| `Trigger Risk Report AM Correction (GHA workflow_dispatch)` | Enabled | Weekdays 4:30 AM ET — fires `risk_report.yml` with `mode=data_only` so the fragility row daily_scan sizes off (~4:47 AM) reflects settled prices, not the provisional 5:15 PM bar. Scripts: `C:\Scripts\trigger_risk_report.ps1` + `_task.xml`. |
| `IBKR OLV Pre-Market Exits` | Enabled | Weekdays 9:10 AM ET — `run_olv_exit_moo.bat` -> `olv_exit_moo.py` (OneDrive trading_ibkr): reads the `OLV_Exits` tab and places TRUE market-on-open (TIF=OPG) SELLs for confirmed OLV stop legs on BOTH accounts before the 9:28 auction cutoff. Registered 2026-07-30 via `register_olv_exit_task.ps1`; must clear the 9:31 order chain (shares clientIds 99/98). |
| `IBKR OLV Book Cap (EOD)` | DISABLED 2026-08-25 (registered, never fired live) | Weekdays 3:40 PM ET — `run_olv_book_cap.bat` -> `olv_book_cap.py` (OneDrive trading_ibkr): trims every open OLV leg pro-rata (native MOC/DAY, exit legs resized first, owner clientIds 99/98) so the OLV sub-book closes at or below 100% of NAV on BOTH accounts (flat $750k primary / live NLV PA) and shrinks working OLV entries into the remaining room. Registered 2026-08-24 via `register_olv_book_cap_task.ps1`; 15:48 MOC cutoff -> MKT DAY, 15:58 hard stop; `--dry-run` for inspection. See "OLV Book Cap". |
| `Daily Pitch (agent)` | Enabled (Interactive logon) | Weekdays 5:10 AM ET — `scripts\run_daily_pitch.bat`: grade, assemble state, run `/daily-pitch` headless, verify delivery. Writes files and sends the pitch email; places no orders. Register with `scripts\register_daily_pitch_task.ps1` after eyeballing several manual runs. |
| `IBKR Daily Pitch Approvals (auction)` / `(open)` | NOT REGISTERED (inert) | Weekdays 9:05 and 9:32 AM ET — `pitch_moo.py` places `Pitch` rows marked `Y`. Live money: registration script also writes `pitch_moo_enabled.flag`; delete the flag to disarm without unregistering. |
| `RadarMorningBriefing` | Enabled | Lives in separate `last30days-radar` project — not yet migrated. |
| `Radar Weekly Sync` | Registered by `scripts/register_radar_sync_task.ps1`; PREVIEW until armed | Mondays 8:50 AM ET — `scripts\run_radar_sync.bat`: publish the momentum radar's recs to R2, then reconcile live trail stops. Transmits only when `radar_trail_enabled.flag` exists in OneDrive/trading_ibkr. See Momentum Radar. |
| `DailyPortfolioReport` | Disabled | Replaced by `portfolio_report.yml`. Re-enable as fallback if GHA breaks. |
| `MasterPricesUpdate` | Disabled | Replaced by `update_master_prices.yml`. |
| `OverflowDailyScan` | Disabled | Replaced by the unified `daily_screener.yml --scope=all` post-close run. |

Order staging (`C:\Users\McKinley Slade\OneDrive\trading_ibkr\order_staging.py`) is a manual / scheduled local launch — talks to IBKR TWS on `127.0.0.1:7496`. Reads `Order_Staging` + `Overflow` Sheets tabs and submits orders pre-market.

### Historical AM Dispatch Architecture (retired 2026-08-28)

GitHub's shared cron scheduler had 1-3h queue delays at 8:47 UTC, pushing the AM scan past pre-market staging deadlines. Fix: fire the AM runs from this machine via the GitHub REST API (`workflow_dispatch`), which has near-zero queue lag.

**Daily flow (weekdays):**
- **4:10 AM ET** local task -> POST `.../update_cboe_putcall.yml/dispatches` -> collects the prior session's put/call, which the 21:30 UTC run cannot see (CBOE publishes after it)
- **4:17 AM ET** local task → POST `…/update_master_prices.yml/dispatches` → GHA queues immediately, runs ~5 min
- **4:47 AM ET** local task → POST `…/daily_screener.yml/dispatches` → GHA queues immediately, runs ~7-10 min

**Fallback** (machine off / network outage): both workflows keep an early GHA cron (parquet 9:30 UTC, screener 10:30 UTC). Each workflow's first job (`check`) queries the GitHub API for today's `workflow_dispatch` runs and short-circuits if a successful one already exists; otherwise the main job runs. The fallback cron is subject to GHA's queue lag but still beats market open by ~3h in the worst case.

**Local artifacts:**
- Trigger scripts: `C:\Scripts\trigger_update_master_prices.ps1`, `C:\Scripts\trigger_daily_screener.ps1`, `C:\Scripts\trigger_cboe_putcall.ps1` (+ `register_cboe_putcall_trigger.ps1`)
- Task XMLs: `C:\Scripts\*_task.xml` (S4U principal, WakeToRun, no AC required, restart-on-failure 5min × 3)
- Logs: `C:\Scripts\logs\trigger_*.log` (one line per dispatch attempt)
- PAT: `HKCU\Environment\GH_PAT_NEW_SEASONALS` (fine-grained, scoped to `mslade50/New_Seasonals`, permissions: Actions/Workflows/Contents — read+write, Metadata read). Rotate annually.

**Current maintenance:** inspect R2 component receipts and pinned-runtime logs. The central hourly receipt controller supplies the GitHub backup; the former independent child crons no longer exist.


### Radar Weekly Digest Framework
The Claude prompt enforces heavy filtration via two required gates:
- **Variant Perception**: Must articulate specific disagreement with market pricing. If thesis = consensus, idea is killed.
- **Who's on the other side**: Must identify why the opportunity exists (forced selling, informed disagreement, or neglect). If can't identify, idea is killed.

Supporting lenses (non-dogmatic, context-dependent): catalyst magnitude, valuation vs forward reality, trend/market structure, persistence across the week, crowd positioning.
