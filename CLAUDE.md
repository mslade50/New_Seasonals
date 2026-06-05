# CLAUDE.md — Project Guide for New_Seasonals

## What This Project Is

A quantitative equity trading platform built on Streamlit. Three pillars:
1. **Strategy system** — backtesting, scanning, and order staging for directional equity strategies (1-63 day hold)
2. **Risk monitoring** — multi-layer market regime dashboard (volatility, internals, credit/macro)
3. **Dispersion analytics** — S&P 500 absolute return dispersion (Nomura methodology)

## Repo Structure

```
├── app.py                          # Main Streamlit entry point
├── strategy_config.py              # Strategy definitions (STRATEGY_BOOK)
├── daily_scan.py                   # Unified scanner — supports --scope=liquid|overflow|all (--moc-only flag retained for future use; no MOC strategies in book currently)
├── daily_risk_report.py            # Daily risk email (fragility dials + signals + forward returns)
├── daily_portfolio_report.py       # Daily portfolio health report (imports from strat_backtester)
├── weekly_market_rundown.py        # Weekly PDF rundown (tabloid landscape, 11 chart pages)
├── radar_weekly_summary.py         # Weekly radar digest (reads daily briefs, Claude distills best-of)
├── verify_fills.py                 # Post-close fill verification (updates Google Sheets)
├── indicators.py                   # Shared indicator library
├── earnings_filter.py              # Shared OVS earnings blackout helpers (load parquet, compute offset)
├── cache_io.py                     # Cloudflare R2 read/write wrapper (boto3) — graceful no-op without creds
├── abs_return_dispersion.py        # S&P 500 dispersion metric (~505 tickers)
├── local_overflow_scan.py          # DEPRECATED stub — forwards to `daily_scan.py --scope=overflow`
├── risk_dashboard_clean_sheet.md   # Risk Dashboard V2 design doc
├── pages/                          # Streamlit pages (FLAT — no subfolders)
│   ├── risk_dashboard_v2.py        # Multi-layer regime monitor (standalone)
│   ├── backtester.py               # Strategy backtesting UI
│   ├── strat_backtester.py         # Extended backtester
│   ├── heatmaps.py                 # Market heatmap inspector
│   ├── correlation_heatmaps.py     # Correlation analysis
│   ├── macro_seasonality.py        # Macro seasonality (formerly sector_trends)
│   ├── seasonal_sigs.py            # Seasonal signals
│   └── user_input.py               # User input page
├── .github/workflows/              # GitHub Actions — see "Automated Pipeline" below
│   ├── daily_screener.yml          # 2x/day unified scan — pre-market (08:47 UTC) and post-close (22:00 UTC) bookends, both --scope=all
│   ├── build_earnings_calendar.yml # Nightly FMP refresh → R2
│   ├── update_master_prices.yml    # Nightly yfinance incremental → R2
│   ├── update_intraday_prices.yml  # Nightly 15min yfinance incremental → R2 (intraday cache)
│   ├── portfolio_report.yml        # Daily portfolio email
│   ├── bootstrap_caches.yml        # workflow_dispatch only — one-shot full master_prices rebuild
│   ├── risk_report.yml             # Daily risk dashboard email
│   ├── verify_fills.yml            # Post-close fill verification
│   └── weekly_rundown.yml          # Sunday weekly PDF
├── scripts/                        # Task Scheduler PowerShell wrappers (most disabled post-Phase-2)
│   ├── run_radar_weekly.ps1        # Sundays 8:30 AM ET — runs radar digest, commits + pushes
│   ├── run_earnings_calendar.ps1   # Weekdays 5:30 PM ET — local backup of GHA build (dual writers OK)
│   ├── build_earnings_calendar.py  # FMP earnings backfill (used by both local + GHA)
│   ├── update_master_prices.py     # yfinance incremental update (used by both local + GHA)
│   ├── build_master_prices.py      # One-shot full rebuild (used by bootstrap_caches.yml)
│   └── (DISABLED locally: run_overflow_scan.ps1, run_daily_portfolio_report.ps1, run_master_prices_update.ps1)
├── data/                           # Persistent cache (parquet files + radar digest) — gitignored
├── docs/                           # Documentation
└── tests/                          # Tests
```

## Critical Rules

### yfinance MultiIndex Bug
ALL multi-ticker yfinance downloads return MultiIndex columns `(Price, Ticker)`. You MUST handle this:
```python
# For multi-ticker downloads:
if isinstance(raw.columns, pd.MultiIndex):
    df = raw.xs(ticker, level='Ticker', axis=1)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df.columns = [c.capitalize() for c in df.columns]
```
Skipping this causes silent crashes. Every data function must handle it.

### Pages Directory
The `pages/` directory must remain **FLAT** — no subfolders. Streamlit discovers pages by scanning this directory.

### Path Setup Pattern
All pages that import from the project root use:
```python
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
```

### Caching Pattern
- `@st.cache_data(ttl=3600)` for data downloads (1-hour TTL)
- `@st.cache_resource` for static data (seasonal maps)
- Parquet files in `data/` for expensive computations (S&P 500 prices)

## Module Boundaries

**risk_dashboard_v2.py is STANDALONE.** It must never import from:
- `strategy_config.py`
- `strat_backtester.py`
- `daily_scan.py`
- `indicators.py`

It may optionally import `SP500_TICKERS` from `abs_return_dispersion.py` (with try/except fallback).

**Strategy modules** (`strat_backtester.py`, `daily_scan.py`, `daily_portfolio_report.py`) all depend on `strategy_config.py` for `STRATEGY_BOOK` and `ACCOUNT_VALUE`.

**daily_portfolio_report.py** imports backtesting logic from `strat_backtester.py`. Both must stay in sync with `daily_scan.py` for signal detection, sizing, and trade processing. `ACCOUNT_VALUE` from `strategy_config.py` is the single source of truth for portfolio sizing across all three. Runs in **GitHub Actions** (weekdays 21:30 UTC = 5:30 PM ET) — pulls `data/master_prices.parquet` and `data/earnings_calendar.parquet` from Cloudflare R2 before running. Reports cover both liquid (LIQUID_PLUS_COMMODITIES) and overflow (CSV_UNIVERSE − LIQUID_PLUS_COMMODITIES) universes — overflow-eligible strategies get a second deep-copied pass with `OVERFLOW_RISK_OVERRIDES` (only OLV 35→25 bps remains; OVS uses path-1 nominal 40 bps for both tiers). Workflow: `.github/workflows/portfolio_report.yml`.

**daily_scan.py** is the single unified scanner (post-2026-04-30 merge with the retired `local_overflow_scan.py`). CLI flags:
- `--scope=liquid` (default) — scans every strategy against its native universe (typically LIQUID_PLUS_COMMODITIES)
- `--scope=overflow` — only the 5 overflow-eligible strategies, swapped to CSV_UNIVERSE − LIQUID_PLUS_COMMODITIES with OLV bps override
- `--scope=all` — both passes concatenated, signals stamped with `Scan_Source='Liquid'` or `'Overflow'`
- `--moc-only` — restricts to strategies with `entry_type='Signal Close'`. Skips the overflow tier entirely (overflow doesn't MOC by convention). Currently a no-op since the strategy book has no MOC entries; the flag is retained for future use if a Signal Close strategy is added back.

Per-tier tab routing inside `save_staging_orders`: Liquid rows → `Order_Staging`, Overflow rows → `Overflow`. Both tabs are read by `order_staging.py` (which lives in `C:\Users\mckin\OneDrive\trading_ibkr\` — IBKR-bound, stays local).

## Risk Dashboard V2 — Current State

**Phases 1 & 2 complete** (Layers 0–4). See `notes.md` for full details.

### Executive Summary — Signal-Based Three-Question Framework
One-screen briefing at the top of the page. Three sections:

**Section A: Price Context Banner** — SPY price, 12mo return, extension vs 200d SMA, drawdown from 52w high, regime label (e.g. "Healthy uptrend", "Correction underway"). Plus "What Changed" line tracking signal activations/deactivations since last session via JSON persistence (`data/risk_dashboard_signal_state.json`).

**Section B: Three Questions + Risk Dial** (3:1 column split)
- **Is liquidity real?** — Vol Suppression (low AR + low RV), VRP Compression (negative or <15th pctile)
- **Is everyone on the same side?** — Breadth Divergence (SPY near high, <55% sectors above 200d), Extended Calm (compound complacency counters), Vol Compression (>60 consecutive days below expanding median RV)
- **Are correlations stable?** — Credit-Equity Divergence (HY z >0.75 while SPX flat), Rates-Equity Vol Gap (MOVE elevated, VIX calm), Vol Uncertainty (VVIX/VIX ratio >80th pctile)
- Each question shows CLEAR/WATCH/WARNING badge. Each signal ON/OFF with explanatory detail when active.
- **Risk Dial** — Plotly gauge, 0-100 fragility score driven by (active signal count / total) × 80 × regime multiplier (0.6-1.8x based on price context). Labels: Robust → Neutral → Fragile.

**Section C: Stored Energy** (conditional — only when 2+ signals active)
- Vol compression duration & depth, calm streak, estimated drawdown range based on extension + compression + signal count.

Legacy point system preserved in collapsed expander for reference. Alert = +1, Alarm = +2.
- 0 pts = Normal | 1-2 = Caution | 3-4 = Stress | 5+ = Crisis

### Layer 1: Volatility State
- 1A: HAR-RV (Yang-Zhang at 1d/5d/22d)
- 1B: VRP = (VIX/100)^2 - RV_22d^2
- 1C: VIX Term Structure (VIX/VIX3M)
- 1D: VVIX

### Layer 2: Equity Market Internals
- 2A: Breadth (sector ETF proxy — % above 200d/50d SMA)
- 2B: Absorption Ratio (PCA on 63d sector returns). **Display-only** — removed from composite scoring. Red line at 0.40. Measures % of sector variance explained by first PC; low AR (<0.4) historically precedes below-avg returns (Minsky dynamic). Backtested: AR <0.4 → 5d avg -0.40% (vs +0.29% baseline), 63d avg +0.82% (vs +3.53%), N=17 deduped episodes over 10 years.
- 2C: Cross-sectional dispersion + avg pairwise correlation (2x2 grid)
- 2D: Hurst exponent (DFA, **126d window**, box sizes [8,16,32,48,63]). **Smoothed**: 11d rolling median → 15d EMA. Empirical percentile bands (P20/P80 of smoothed series). Alert > 80th pctile, alarm > 95th. 5d ΔH from smoothed series is the primary signal.
- 2E: Complacency Counters — two primary signals: days since 5% SPX drawdown + days since VIX > 28. 10% drawdown also displayed for context. Compound scoring: either > 80th pctile = alert (+1), BOTH > 80th = alarm (+2). Sawtooth charts for each counter.

### Layer 3: Cross-Asset Plumbing (4-column layout)
- 3A: Credit Spreads — LQD/HYG vs IEF price ratio z-scores (63d rolling). Alert: IG or HY z > 1.0. Alarm: both > 1.5.
- 3B: Yield Curve — 10Y-3M spread (^TNX - ^IRX). 21d change z-score is the signal. Alert: inverted OR z < -1.5. Alarm: inverted AND z < -2.0.
- 3C: MOVE Index — raw level with bands at 80/120/150. Alert: > 120. Alarm: > 150. Graceful fallback if ^MOVE unavailable on yfinance.
- 3D: Dollar Dynamics — UUP 21d momentum as DXY proxy. Alert: |chg| > 3%. Alarm: |chg| > 5%.

### Layer 4: Tail Risk & Cost of Protection (auto-expands when 2+ signals active)
- 4A: SKEW Index — time series with 120/140 bands. Disorderly stress detection: flags when SKEW falling (>3pts in 5d) while VIX rising (>3pts in 5d).
- 4B: Protection Cost Proxy — VIX3M × (SKEW/130), percentile-ranked over 5yr trailing window. Plotly gauge display (green/yellow/orange/red).
- 4C: Hedge Recommendation — decision tree based on regime × protection cost percentile. Outputs: sizing guidance, collar vs puts vs exposure reduction.

### Chart Defaults
- HAR-RV and VRP charts default to last 1 year. Double-click to zoom out to full history.
- Layer 3 charts use compact 200px height (vs 250px for Layers 1/2).

### Phase 3 TODO
- Signal event study: backtest each of the 8 signals individually to calibrate hit rates (currently placeholder estimates)
- Historical regime backtesting
- FRED data source for MOVE (more reliable than yfinance)

## Ticker Constants

| Variable | Location | Count | Description |
|----------|----------|-------|-------------|
| `SP500_TICKERS` | `abs_return_dispersion.py` | ~505 | Full S&P 500 constituents |
| `LIQUID_PLUS_COMMODITIES` | `strategy_config.py` | ~190 | Liquid universe — daily_scan default scope |
| `CSV_UNIVERSE` | `strategy_config.py` | ~1060 | Full universe (liquid + overflow tier ~870) |
| `OVERFLOW_ELIGIBLE_STRATEGIES` | `daily_scan.py` | 5 | OVS, OLV, LT Trend ST OS, St OS Sznl, 52wh Breakout |
| `OVERFLOW_RISK_OVERRIDES` | `daily_scan.py`, `daily_portfolio_report.py` | 1 | OLV: 35→25 bps for overflow tier |
| `SECTOR_ETFS` | `risk_dashboard_v2.py` | 11 | SPDR sector ETFs |
| `VOL_TICKERS` | `risk_dashboard_v2.py` | 4 | SPY, ^VIX, ^VIX3M, ^VVIX |
| `CROSS_ASSET_TICKERS` | `risk_dashboard_v2.py` | 7 | LQD, HYG, IEF, UUP, ^MOVE, ^TNX, ^IRX |
| `TAIL_RISK_TICKERS` | `risk_dashboard_v2.py` | 1 | ^SKEW |
| `SIGNAL_CACHE_PATH` | `risk_dashboard_v2.py` | — | `data/risk_dashboard_signal_state.json` |

## OVS Strategy — Earnings Blackout + 2-Path Sizing + Friday-only EOD-DD

Overbot Vol Spike has special-cased execution as of the 2026-04-30 merge.

### Earnings blackout (±10 trading days)
The OVS execution dict in `strategy_config.py` carries `earnings_blackout_td: 10`. Signals within ±10 trading days of an earnings announcement are dropped. Tickers with no earnings data in `data/earnings_calendar.parquet` (commodity ETFs, indices, futures, FX) **pass through** — NaN-as-True, mirroring the `Not Between` behavior in `pages/backtester.py`.

Implementation:
- `earnings_filter.py` — shared module with `load_earnings_dates_map()`, `signed_offset()`, `in_blackout(window=10)`. Loads `data/earnings_calendar.parquet`.
- `daily_scan.py` — applies the filter inline during the strategy loop (drops the signal before the dict is built).
- `pages/strat_backtester.py` — pre-pass that drops candidates from the chronological loop entirely, so the daily portfolio report's PnL reflects what live would do.

### Two-path execution (replaces the prior 30/20 bps + 1.3× ATR-sznl-5d sizer)
The OVS execution dict carries:
- `path1_bps: 40` — full size on a decisive open gap
- `path2_bps: 8` — reduced size on a mild gap
- `path2_daily_cap_pct: 1.0` — 1% of ACCOUNT_VALUE aggregate cap on path-2 risk

Decision happens in `order_staging.py` (in `C:\Users\mckin\OneDrive\trading_ibkr\`) using IBKR's T+1 session open vs the signal's close + 0.25 ATR threshold. Same scheme for liquid AND overflow universes.

| T+1 open vs close | Path | Per-trade size |
|---|---|---|
| Open > Close + 0.25 ATR | **Path 1: Decisive** | 40 bps (full) |
| Close < Open ≤ Close + 0.25 ATR | **Path 2: Mild** | 8 bps, capped at 1% aggregate (pro-rata scale-down across all path-2 rows that day) |
| Open ≤ Close | **Skip** | 0 |

Scanner-side stamps `Path1_Bps`, `Path2_Bps`, `Path2_Daily_Cap_Pct` columns on every OVS staging row so order_staging can compute the multiplier without importing strategy_config.

### Entry-day drawdown stop (EOD-DD, Friday entries only)
The OVS execution dict carries `eod_dd_atr: 0.25` and `eod_dd_weekdays: [4]`. If a Friday-entered OVS trade is more than 0.25 ATR offside vs the entry-day fill by 15:58 ET, exit at the entry-day close. Mon-Thu entries skip the check entirely — those positions get the full hold window instead. Weekday list uses Python conventions (Mon=0..Fri=4); empty/missing = all weekdays.

Aligned across four systems — change `eod_dd_weekdays` in one place and they all move together:
- `strategy_config.py` — execution dict (single source of truth)
- `pages/strat_backtester.py` — reads `execution['eod_dd_weekdays']`, gates the EOD-DD block on `df.index[entry_idx].weekday() in [...]`. Drives both the backtester page and `daily_portfolio_report.py`.
- `pages/backtester.py` — UI multiselect lets you override per-run for exploration (separate from the prod-locked rule above).
- `order_staging.py` (in `C:\Users\mckin\OneDrive\trading_ibkr\`) — hardcoded `weekday() == 4` gate on the STP-with-goodAfterTime=15:58 leg. Update both sides if you change the rule.
- Regression coverage: `tests/test_eod_dd.py` Cases C/D assert Fri fires + Tue skipped under `[4]`.

### Reference
- Trading-day arithmetic: `compute_signed_earnings_offsets()` in `pages/backtester.py` (np.busday_count + USFederalHolidayCalendar).
- Earnings parquet: `data/earnings_calendar.parquet` — 117k rows, 946 tickers, FMP-backfilled, includes forward dates.
- 2-path validation note (2026-04-29): 12 of 13 OVS signals on that date would have been killed by the blackout — only USO survived because no earnings data.

## Cloudflare R2 Cache + GHA Migration

As of 2026-04-30, the nightly pipeline runs entirely in GitHub Actions. The local Task Scheduler retains the radar tasks plus (as of 2026-05-13) two AM `workflow_dispatch` triggers that bypass GitHub's congested 8-9 UTC cron-queue lag. R2 is the persistence layer that lets cloud workflows share parquet caches.

### R2 secrets (in GHA repo settings)
- `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET=seasonals-cache`

### Bucket contents (key-value)
- `master_prices.parquet` — full ~2000 ticker × 25-yr OHLCV (~50-200 MB). Read by `daily_scan --scope=overflow|all` and `daily_portfolio_report.py`. Written by `update_master_prices.yml` twice on weekdays (AM via local workflow_dispatch ~4:17 AM ET + PM via 20:30 UTC cron). Pre-market runs pass `--exclude-today` so yfinance placeholder bars never enter the cache.
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

## Automated Pipeline

All five trading-day workflows now run in GHA. Order staging stays local (IBKR-bound).

| Workflow file | Schedule | What it does |
|---|---|---|
| `daily_screener.yml` | Weekdays 2x: AM via local workflow_dispatch at 4:47 AM ET (fallback GHA cron at 10:30 UTC, auto-skipped if dispatch succeeded today) + PM cron at 22:00 UTC | Unified scan, both runs `--scope=all` (full liquid + overflow, ~7-10 min). AM run also writes `data/exposure_state.json` and commits it back to main. Intraday MOC slots were retired when the strategy book lost its last Signal Close entry; restore them if MOC strategies are added back. |
| `build_earnings_calendar.yml` | Weekdays 21:30 UTC (5:30 PM ET) | FMP `/stable/earnings` pull → writes `data/earnings_calendar.parquet` → uploads to R2. Local `EarningsCalendarRefresh` Task Scheduler entry mirrors this for redundancy (last write wins). |
| `update_master_prices.yml` | Weekdays 2x: AM via local workflow_dispatch at 4:17 AM ET (fallback GHA cron at 9:30 UTC, auto-skipped if dispatch succeeded today) + PM cron at 20:30 UTC (4:30 PM ET) | Pulls `master_prices.parquet` from R2, fetches today's bars from yfinance for ~2000 tickers, appends, dedupes, writes back to R2. PM cron pulls today's close; every other trigger (AM dispatch, AM fallback cron, manual dispatch) passes `--exclude-today`. |
| `update_intraday_prices.yml` | Weekdays 20:45 UTC (4:45 PM ET) | Pulls per-ticker 15min parquets + meta from R2, runs `scripts/update_intraday_yfinance.py --upload` — fetches recent bars from yfinance for every ticker in meta, converts UTC→ET, appends, dedupes, writes back. yfinance has 60d rolling intraday history so this must run at least every ~50 days to avoid gaps; weekday cadence is fine in practice. |
| `portfolio_report.yml` | Weekdays 21:30 UTC (5:30 PM ET) | Pulls master_prices + earnings caches from R2, runs `daily_portfolio_report.py`, sends HTML email + writes Portfolio Sheets tab. |
| `bootstrap_caches.yml` | workflow_dispatch only | One-shot: builds `master_prices.parquet` from scratch via yfinance (~10-15 min for ~2000 tickers, 25-yr history) and uploads to R2. Used to seed the bucket (already run during Phase 2 setup). |
| `risk_report.yml` | Weekdays 21:15 UTC (5:15 PM ET) | Daily risk dashboard email (fragility dials + signals + forward returns). |
| `verify_fills.yml` | Weekdays 21:15 UTC | Post-close fill verification — updates Trade_Signals_Log. |
| `weekly_rundown.yml` | Sundays 14:00 UTC (9 AM ET) | Tabloid PDF with all risk charts + radar digest body. |

### Local Task Scheduler (post-Phase-2)

| Task | State | Notes |
|---|---|---|
| `EarningsCalendarRefresh` | Enabled | Belt-and-suspenders for the GHA equivalent. Both write to R2. |
| `Trigger Update Master Prices (GHA workflow_dispatch)` | Enabled | Weekdays 4:17 AM ET — fires `update_master_prices.yml` via the GitHub REST API to bypass shared-cron queue lag at 8-9 UTC. See "AM Trigger Architecture" below. |
| `Trigger Daily Screener (GHA workflow_dispatch)` | Enabled | Weekdays 4:47 AM ET, 30 min after the parquet trigger — fires `daily_screener.yml` via the GitHub REST API. Same mechanism. |
| `RadarMorningBriefing` | Enabled | Lives in separate `last30days-radar` project — not yet migrated. |
| `RadarWeeklySummary` | Enabled | Sundays 8:30 AM ET — depends on radar briefs from above. Not yet migrated. |
| `DailyPortfolioReport` | Disabled | Replaced by `portfolio_report.yml`. Re-enable as fallback if GHA breaks. |
| `MasterPricesUpdate` | Disabled | Replaced by `update_master_prices.yml`. |
| `OverflowDailyScan` | Disabled | Replaced by the unified `daily_screener.yml --scope=all` post-close run. |

Order staging (`C:\Users\mckin\OneDrive\trading_ibkr\order_staging.py`) is a manual / scheduled local launch — talks to IBKR TWS on `127.0.0.1:7496`. Reads `Order_Staging` + `Overflow` Sheets tabs and submits orders pre-market.

### AM Trigger Architecture (added 2026-05-13)

GitHub's shared cron scheduler had 1-3h queue delays at 8:47 UTC, pushing the AM scan past pre-market staging deadlines. Fix: fire the AM runs from this machine via the GitHub REST API (`workflow_dispatch`), which has near-zero queue lag.

**Daily flow (weekdays):**
- **4:17 AM ET** local task → POST `…/update_master_prices.yml/dispatches` → GHA queues immediately, runs ~5 min
- **4:47 AM ET** local task → POST `…/daily_screener.yml/dispatches` → GHA queues immediately, runs ~7-10 min

**Fallback** (machine off / network outage): both workflows keep an early GHA cron (parquet 9:30 UTC, screener 10:30 UTC). Each workflow's first job (`check`) queries the GitHub API for today's `workflow_dispatch` runs and short-circuits if a successful one already exists; otherwise the main job runs. The fallback cron is subject to GHA's queue lag but still beats market open by ~3h in the worst case.

**Local artifacts:**
- Trigger scripts: `C:\Scripts\trigger_update_master_prices.ps1`, `C:\Scripts\trigger_daily_screener.ps1`
- Task XMLs: `C:\Scripts\*_task.xml` (S4U principal, WakeToRun, no AC required, restart-on-failure 5min × 3)
- Logs: `C:\Scripts\logs\trigger_*.log` (one line per dispatch attempt)
- PAT: `HKCU\Environment\GH_PAT_NEW_SEASONALS` (fine-grained, scoped to `mslade50/New_Seasonals`, permissions: Actions/Workflows/Contents — read+write, Metadata read). Rotate annually.

**Maintenance:** if the local task or PAT breaks, the fallback cron picks up the slack the same day. If both break, the PM cron at 20:30 / 22:00 UTC still runs (independent of any of this).

### Sunday Pipeline (two-step, still partially local)
1. **8:30 AM ET (local)**: `radar_weekly_summary.py` reads last 7 days of radar briefs from `C:\Users\mckin\projects\last30days-radar\output\briefs\`, pulls yfinance snapshots for all tickers, pipes to Claude Code subprocess with PM-style distillation framework (variant perception required, "who's on the other side" required). Output committed + pushed to `data/radar_weekly_summary.md`.
2. **9:00 AM ET (Actions)**: `weekly_market_rundown.py` generates tabloid (17x11") landscape PDF with all risk charts, reads the radar digest and includes it as styled HTML email body alongside the PDF attachment.

### Daily Risk Report — Forward Returns Table
Uses `compute_similar_reading_returns()` from `risk_dashboard_v2.py`. Forward returns at similar fragility readings include:
- Mean and Median conditional returns
- **Mean Z / Median Z** — z-scores vs unconditional sample (mean via z-test, median via bootstrap SE with 1000 resamples)
- % Negative and Baseline (unconditional mean)
- Mean column color follows Mean Z thresholds (green >= 0, yellow > -1, red <= -1)

### Radar Weekly Digest Framework
The Claude prompt enforces heavy filtration via two required gates:
- **Variant Perception**: Must articulate specific disagreement with market pricing. If thesis = consensus, idea is killed.
- **Who's on the other side**: Must identify why the opportunity exists (forced selling, informed disagreement, or neglect). If can't identify, idea is killed.

Supporting lenses (non-dogmatic, context-dependent): catalyst magnitude, valuation vs forward reality, trend/market structure, persistence across the week, crowd positioning.

Framework doc: `C:\Users\mckin\Documents\vault\trading\decisions\radar_weekly_digest_framework.md`

## Google Sheets Integration

Tab layout in the `Trade_Signals_Log` workbook:
- `Order_Staging` — Liquid-tier signals (Limits, T+1 Open, Persistent GTC). Cleared + rewritten by every `daily_scan` run with `Scan_Source='Liquid'`.
- `Overflow` — Overflow-tier signals (same entry types, no MOC). Cleared + rewritten by `daily_scan --scope=overflow|all` with `Scan_Source='Overflow'`.
- `moc_orders` — MOC entries from liquid tier only (`save_moc_orders` skips overflow rows). Currently vestigial: the strategy book has no Signal Close entries, so this tab is never written. Reactivates automatically if any strategy is set to `entry_type='Signal Close'`.
- `Trade_Signals_Log` (sheet1) — append-only signal history.
- `Portfolio` — open-positions snapshot from `daily_portfolio_report.py`.
- `execution`, `execution_2` — order_staging.py output for primary + small-account execution.

`daily_scan.py` writes both `Order_Staging` and `Overflow` via `save_staging_orders(..., tier_filter='Liquid'|'Overflow')`. The function clears+rewrites only the tier it's responsible for (so a `--scope=liquid` run never touches `Overflow`).

`order_staging.py` (in `C:\Users\mckin\OneDrive\trading_ibkr\`) reads BOTH tabs and concatenates with `Scan_Source` distinguishing tier. Applies the OVS 2-path gap-tier sizer + path-2 daily aggregate cap + global 2.5% daily risk cap before submitting to IBKR.

`verify_fills.py` updates Trade_Signals_Log with fill status post-close.

Auth: `gspread` with GCP service account from Streamlit secrets / `GCP_JSON` env var (GHA) / `credentials.json` (local).
