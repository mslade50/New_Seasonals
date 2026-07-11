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
│   ├── deploy_site.yml             # Private-site build + Pages deploy — reusable workflow (workflow_call) invoked by daily_screener's deploy-site job, same run (2x/day)
│   └── weekly_rundown.yml          # Sunday weekly PDF
├── scripts/                        # Task Scheduler PowerShell wrappers (most disabled post-Phase-2)
│   ├── run_radar_weekly.ps1        # Sundays 8:30 AM ET — runs radar digest, commits + pushes
│   ├── run_earnings_calendar.ps1   # Weekdays 5:30 PM ET — local backup of GHA build (dual writers OK)
│   ├── build_earnings_calendar.py  # FMP earnings backfill (used by both local + GHA)
│   ├── update_master_prices.py     # yfinance incremental update (used by both local + GHA)
│   ├── build_master_prices.py      # One-shot full rebuild (used by bootstrap_caches.yml)
│   ├── build_trade_ledger.py       # Full-history trade ledger (data/backtest_trades_full.parquet)
│   ├── build_site.py               # Private-site JSON payloads + static assets -> dist/
│   ├── build_signal_charts.py      # Per-trade candlestick charts -> charts/ + R2 (lazy-served on the site)
│   ├── signal_chart_common.py      # Shared chart key + MAE/MFE helpers (build_signal_charts + build_site)
│   ├── build_risk_json.py          # Condensed risk summary for the site (best effort, exits 0)
│   ├── backtester_html_report.py   # Legacy single-file HTML view (reports/portfolio/)
│   ├── refresh_view.py             # Local one-command ledger + HTML refresh
│   └── (DISABLED locally: run_overflow_scan.ps1, run_daily_portfolio_report.ps1, run_master_prices_update.ps1)
├── site/                           # Private-site frontend (static HTML/CSS/JS, committed)
├── functions/                      # Cloudflare Pages Functions — chartimg/[[path]].js streams chart PNGs from R2
├── wrangler.toml                   # Pages config: pages_build_output_dir=dist + CHARTS R2 binding (TOML — action's wrangler 3.90.0 ignores .jsonc)
├── dist/                           # Site build output — gitignored, deployed to Cloudflare Pages
├── charts/                         # Per-trade chart PNGs — gitignored; R2 (charts/ prefix) is the source of truth
├── data/                           # Persistent cache (parquet files + radar digest) — gitignored
├── docs/                           # Documentation (private_site_setup.md = Cloudflare one-time setup)
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

### Dividend-Adjustment Basis (raw vs adjusted) — book-wide invariant
The rule, applied per surface:
- **Compare a FROZEN dollar level against RAW bars** (`auto_adjust=False`). A limit/stop/entry that was computed once and stored (sheet `Limit_Price`/`Entry`/`ATR`, a ledger entry, a live working order) lives in the as-traded basis it was minted in. Re-pulling ADJUSTED bars re-scales history down whenever a later dividend goes ex, dropping a past low below a limit that was never touched live (the EWZ 33.51 ex-div phantom fill, 2026-06). `verify_fills.py` pulls raw for exactly this reason.
- **RECOMPUTE a relative level each run → ADJUSTED bars are safe.** The backtest engines (`pages/backtester.py`, `pages/strat_backtester.py`) derive the limit from the same adjusted series each run (`Close ± k·ATR`) and compare to that series' forward bars. Both sides scale by the dividend factor `f`, so the fill decision is exactly scale-invariant — no phantom, and returns stay on the correct total-return basis. The engines do NOT round the limit (rounding is the one thing that could break invariance; `verify_fills` rounds, but it's moot there since it uses raw).
- **This holds only while every entry/exit level in the book is RELATIVE.** The moment an ABSOLUTE dollar level is added to the engine path (a hard limit price, a `$`-pivot, a fixed stop), scale-invariance breaks and that level must follow the frozen-level rule (raw bars), or move the cache to raw-OHLCV + read-time adjustment (the deferred "Tier 2" fix). Guard: `tests/test_verify_fills_exdiv.py`.
- **Cache note:** `master_prices.parquet` stores ADJUSTED OHLCV and `update_master_prices.py` re-adjusts a rolling window (`--max-lookback-days`, default 120 — capped above the 63-day max hold + ATR lookback so recent signals stay uniformly adjusted). Per-trade returns are unaffected by the cap; only buy-and-hold accounting past the cap drifts. Do NOT converge the engine basis (adjusted) with the `verify_fills` basis (raw).

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

**daily_portfolio_report.py** imports backtesting logic from `strat_backtester.py`. Both must stay in sync with `daily_scan.py` for signal detection, sizing, and trade processing. `ACCOUNT_VALUE` from `strategy_config.py` is the single source of truth for portfolio sizing across all three. Runs in **GitHub Actions** (weekdays 21:30 UTC = 5:30 PM ET) — pulls `data/master_prices.parquet` and `data/earnings_calendar.parquet` from Cloudflare R2 before running. Reports cover both liquid (LIQUID_PLUS_COMMODITIES) and overflow (CSV_UNIVERSE − LIQUID_PLUS_COMMODITIES) universes — overflow-eligible strategies get a second deep-copied pass with `OVERFLOW_RISK_OVERRIDES` (only OLV 35→25 bps nominal remains; OVS uses path-1 nominal 40 bps for both tiers; all nominals scale by `GLOBAL_RISK_MULTIPLIER` — see "Sizing Conventions"). Workflow: `.github/workflows/portfolio_report.yml`.

**daily_scan.py** is the single unified scanner (post-2026-04-30 merge with the retired `local_overflow_scan.py`). CLI flags:
- `--scope=liquid` (default) — scans every strategy against its native universe (typically LIQUID_PLUS_COMMODITIES)
- `--scope=overflow` — only the 6 overflow-eligible strategies, swapped to CSV_UNIVERSE − LIQUID_PLUS_COMMODITIES with OLV bps override
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
| `OVERFLOW_ELIGIBLE_STRATEGIES` | `daily_scan.py` | 6 | OVS, OLV, LT Trend ST OS, St OS Sznl, 52wh Breakout, ATR Extended Gap Up (no override — native 40 bps nominal on overflow) |
| `OVERFLOW_RISK_OVERRIDES` | `daily_scan.py`, `daily_portfolio_report.py` | 1 | OLV: 35→25 bps nominal for overflow tier (52.5→37.5 effective) |
| `GLOBAL_RISK_MULTIPLIER` | `strategy_config.py` | 1.5 | Book-wide risk scaler applied at import — see "Sizing Conventions" |
| `SECTOR_ETFS` | `risk_dashboard_v2.py` | 11 | SPDR sector ETFs |
| `VOL_TICKERS` | `risk_dashboard_v2.py` | 4 | SPY, ^VIX, ^VIX3M, ^VVIX |
| `CROSS_ASSET_TICKERS` | `risk_dashboard_v2.py` | 7 | LQD, HYG, IEF, UUP, ^MOVE, ^TNX, ^IRX |
| `TAIL_RISK_TICKERS` | `risk_dashboard_v2.py` | 1 | ^SKEW |
| `SIGNAL_CACHE_PATH` | `risk_dashboard_v2.py` | — | `data/risk_dashboard_signal_state.json` |

## Sizing Conventions — GLOBAL_RISK_MULTIPLIER + overlays (2026-05-27)

`strategy_config.GLOBAL_RISK_MULTIPLIER` (currently **1.5**) scales the whole
book at import time: every execution `risk_bps`, OVS `path1_bps` / `path2_bps`
/ `path2_daily_cap_pct`, the OLV `earnings_size_override.risk_bps`, and the
`OVERFLOW_RISK_OVERRIDES` in daily_scan / daily_portfolio_report. The dicts in
strategy_config SOURCE are nominal; everything downstream (scan, engines,
reports, staged `Risk_Amt`/`Risk_Bps`) sees SCALED values. **All bps in this
doc are nominal unless marked effective** — e.g. OLV liquid 35 nominal = 52.5
effective, overflow 25 = 37.5.

daily_scan per-signal sizing order (mirrored in strat_backtester step 3b):
base bps (tier x GRM) -> 2b fragility band -> 2c ladder rung -> 2c2 cycle-year
mult -> 2d earnings size override (flat REPLACE, itself GRM-scaled: OLV signals
-10..0 TD from earnings get 10 bps nominal / 15 effective) -> shares -> 5c
same-day signal de-rate (post-pass; 3x Bear fade — see its section below).

## Daily Risk Caps (aligned 2026-07-10)

Two stacked pro-rata caps on STAGED (pre-fill) risk, applied after all
per-signal sizing. All values are EFFECTIVE (not GRM-scaled):
- **Per-strategy: 250 bps/day.** Each strategy's staged risk per signal date
  independently capped; rows scale by cap/total. Live:
  `order_staging.PER_STRAT_DAILY_CAP_BPS` (was 200 from inception —
  raised to 250 on 2026-07-10 to close a silent live-vs-ledger divergence:
  the engine always modeled 250, so live trimmed 200-250 bps days up to
  20% while the ledger booked full size; the drift came from the engine
  default being anchored to the POOLED 2.5% cap, not the per-strategy
  one). Engine: `process_signals_fast(cap_bps=...)`, default 250.
  `PER_STRAT_DAILY_CAP_DOLLARS` in order_staging holds per-strategy dollar
  overrides (currently empty).
- **Pooled per-direction: long 500 bps, short 250 bps/day.** Total staged
  risk across ALL strategies per side, per signal date. Live:
  `order_staging.MAX_DAILY_RISK_PCT_LONG` (5.0) / `_SHORT` (2.5), applied
  post-open after the per-strategy cap. Engine: `max_long_risk_bps` /
  `max_short_risk_bps` — passed by `scripts/build_trade_ledger.py`
  (POOLED_LONG/SHORT_CAP_BPS) and `daily_portfolio_report.py` since
  2026-07-10 (before that the ledger did NOT model the pooled caps, so
  cross-strategy short cluster days ran optimistic vs live). The
  strat_backtester UI defaults now mirror prod (250/500/250).

Aligned sites — change together: `order_staging.py` (OneDrive) constants,
`scripts/build_trade_ledger.py` POOLED_*_CAP_BPS, `daily_portfolio_report.py`
call site, `pages/strat_backtester.py` UI defaults + `cap_bps` fallback (250)
in `process_signals_fast`.

## Ladder Sizing (OLV, 2026-04-22)

`execution['ladder_multipliers'] = [0.85, 1.00, 1.15]` (OLV only; OVS had it
at launch, removed in the OVS sizing overhaul): a repeat signal on the same
(ticker, strategy) while prior positions are still OPEN sizes at the next
rung — rung = min(open_count, last): 0 open = 0.85x, 1 open = 1.00x,
2+ open = 1.15x. Scales into a working cluster instead of equal-sizing every
re-signal. Aligned sites — change together:
- `strategy_config.py` execution `ladder_multipliers` (source of truth)
- `daily_scan.py` sizing step 2c — open counts from the `Portfolio` Sheets tab
  (`load_open_position_counts`; tab written nightly by daily_portfolio_report;
  lookup failure = no overlay, everything sizes at rung 1)
- `pages/strat_backtester.py` sizing (~line 1675) — replays generically for
  any strategy with the field, using the in-backtest open count (drives the
  ledger + daily_portfolio_report)

## Cross-Strategy Overlap Clamp (2026-05-12)

`strategy_config.CROSS_STRATEGY_OVERLAP_OVERRIDES`: when the named strategies
fire on the SAME signal date and SAME tradeable ticker (compared after
`SPOT_TO_TRADEABLE` aliasing, ^GSPC->SPY ^NDX->QQQ), each side's risk is
clamped to `risk_bps_when_overlapping`. Currently one pair: Indices Oversold
Bounce + SPY QQQ MonFri Reversion -> 20 bps nominal each. Applied in
daily_scan step 5b and replayed in `pages/strat_backtester.py` (~line 2258).

## OVS Strategy — Earnings Blackout + 2-Path Sizing + Friday-only EOD-DD

Overbot Vol Spike has special-cased execution as of the 2026-04-30 merge.

### Earnings blackout (±10 trading days)
The OVS execution dict in `strategy_config.py` carries `earnings_blackout_td: 10`. Signals within ±10 trading days of an earnings announcement are dropped. Tickers with no earnings data in `data/earnings_calendar.parquet` (commodity ETFs, indices, futures, FX) **pass through** — NaN-as-True, mirroring the `Not Between` behavior in `pages/backtester.py`.

Implementation:
- `earnings_filter.py` — shared module with `load_earnings_dates_map()`, `signed_offset()`, `in_blackout(window=10)`. Loads `data/earnings_calendar.parquet`.
- `daily_scan.py` — applies the filter inline during the strategy loop (drops the signal before the dict is built).
- `pages/strat_backtester.py` — pre-pass that drops candidates from the chronological loop entirely, so the daily portfolio report's PnL reflects what live would do.

### Two-path execution (replaces the prior 30/20 bps + 1.3× ATR-sznl-5d sizer)
The OVS execution dict carries (nominal; ×GRM 1.5 at import → 60 / 12 / 1.125%):
- `path1_bps: 40` — full size on a decisive open gap
- `path2_bps: 8` — reduced size on a mild gap
- `path2_daily_cap_pct: 0.75` — 0.75% of ACCOUNT_VALUE aggregate cap on path-2 risk (was 1.0 pre-2026-05-01)

Decision happens in `order_staging.py` (in `C:\Users\mckin\OneDrive\trading_ibkr\`) using IBKR's T+1 session open vs the signal's close + 0.25 ATR threshold. Same scheme for liquid AND overflow universes.

| T+1 open vs close | Path | Per-trade size |
|---|---|---|
| Open > Close + 0.25 ATR | **Path 1: Decisive** | 40 bps nominal / 60 effective (full) |
| Close < Open ≤ Close + 0.25 ATR | **Path 2: Mild** | 8 bps nominal / 12 effective, capped at the 0.75% nominal path-2 aggregate (pro-rata scale-down across all path-2 rows that day) |
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

## Cycle-Year Risk Tilt (OVS, 2026-06-10)

OVS runs at 0.75x risk in midterm years (year%4==2). Evidence: all six
midterm years 2006-2026 underperform (avgR +0.19 vs +0.49 non-midterm),
leave-one-year-out stable, damage concentrated in P1 decisive-gap entries
(+0.63 -> +0.23 avgR). ~1.5 sigma after episode clustering -> shrunk-Kelly
0.75x, not the full-conviction 0.4x. Validated by LOYO, NOT by re-running
the backtest with the rule on (in-sample rules flatter themselves).

Three aligned sites -- change together:
- `strategy_config.py` OVS execution `cycle_risk_mults: {2: 0.75}` (source of truth)
- `pages/strat_backtester.py` sizing step 3b2 (generic: any strategy with the field)
- `daily_scan.py` sizing step 2c2 (stamps the mult into Sizing notes)
(order_staging needs nothing since 2026-06-11: the OVS P1 fixed-dollar
target and its `OVS_CYCLE_MULTS` were removed -- P1 takes the scanner's
staged size as-is, so the tilt flows through like every other overlay.)

The live-vs-backtest divergence found during this work (P1-only live with a
fixed $3,000 target, mild gaps dropped) was RESOLVED 2026-06-11:
order_staging trades both paths again, P1 at scanner qty x1.0 and P2 at
scanner qty x (Path2_Bps / Path1_Bps) from the row stamps plus the P2
aggregate daily cap, matching the 2-path scheme the ledger models
(P2 = 407 trades, +0.20 avgR, +82R/24y). The engine's `ovs_p1_only`
parameter remains for counterfactuals.

## Fragility Risk Bands (2026-07-02)

Per-strategy fragility sizing via `execution['frag_risk_bands']` =
`[[lo, hi, mult], ...]` on the 10d-MA 63d risk-dial score as of signal date
(first match wins, `lo <= score < hi`; missing/stale score or no bands = 1.0x).
REPLACED the retired book-wide ramp (1.25x boost -> 0.10x floor): the boost had
no edge case, and only specific pockets degrade at high fragility. Current
bands: the dip-buy FAMILY4 (Weak Close Decent Sznls, SPY QQQ MonFri Reversion,
Monday Dip, Indices Oversold Bounce) run `[[50, 999, 0.25]]`; the rest of the
book (including OVS) is 1.0x at all scores. Unlike the old ramp, the ENGINE
REPLAYS the bands point-in-time, so ledger and live agree (finding #26 closed
for this scheme). Evidence: scratch/ultracode_research/PORTFOLIO_RESEARCH_2026-07-02.md.

PIT edge-weight gate (roadmap step 5, run 2026-07-03, scratch/pit_reestimate.py
+ pit_extract_signals.py): the fragility composite's signal weights were
re-estimated on expanding windows (vintage Y-1 weights score year Y, 2018+)
to remove calibration lookahead. Results: PIT-vs-current series corr 0.94,
>=50 day agreement 92%. FAMILY4 throttle SURVIVED (hi -0.10 vs lo +0.63,
clustered t=-1.96 p=0.057; LOYO floor ~1.4-1.5 sigma; negative in 6 of 9
years) — stands, conviction one notch lower than the current-weights grading.
The OVS [21,44) 0.75x tilt FAILED (PIT t=-1.34; even current weights only
t=-0.63 on 2018+ — its z=-3.0 lived in untestable 2016-17) and was REMOVED
per the pre-agreed gate; OVS is fully exempt again. The aggregate book-wide
>=50 effect also fails PIT (t=-0.23), vindicating the family-only design.
Residual lookahead the PIT gate cannot cure: signal definitions/parameters
are today's code. Re-examine FAMILY4 at +20 high-frag family trades (~2029).

Three aligned sites -- change together (order_staging needs nothing: it takes
scanner-staged sizes as-is since 2026-06-11):
- `strategy_config.py` execution `frag_risk_bands` (source of truth)
- `pages/strat_backtester.py` sizing step 3b3 (`frag_band_mult_at`, reads
  data/rd2_fragility.parquet point-in-time; pre-2016 signals -> 1.0x)
- `daily_scan.py` sizing step 2b (`frag_band_mult`, today's score, stamps
  Sizing notes; scan-summary email shows active band tilts)
- Guard: `tests/test_frag_risk_bands.py` (config invariants + boundary
  behavior + engine/live parity). Replay parity vs the research cells:
  scratch/parity_check_frag_bands.py (FAMILY4 74@0.25x exact, OVS 226/230
  @0.75x exact with 4 cap-interaction deviations of 0.0004 on one day).

## 3x Bear ETF Overbot Fade + Same-Day Signal De-rate (2026-07-07)

The 13 bear-equity 3x names (`strategy_config.LEV3X_BEAR_EQ`) were carved out
of the generic 3x ETF Overbot Fade (now 29 tickers) into a looser bear-only
fade: short thresholds 85->80, 21d consec 3->1, SAME 126/252d < 65 leader
exclusion — that filter is LOAD-BEARING (it keeps the fade from shorting
sustained bear markets; dropping it collapsed avgR +0.66 -> +0.28 with
-14R/-16R years in 2020/2022). Universes are disjoint by construction so the
two fades can never fire the same ticker on the same day. 25 bps nominal
(vs parent 40) because every signal in the loosened config lands 2020+
(one-regime sample). Fading an overbought inverse ETF = buying a market
selloff, so the strategy carries the FAMILY4 `frag_risk_bands` [[50,999,0.25]].
Evidence: scratch/lev3x_fade_class_study.py + lev3x_fade_bear_episodes.py.

Same-day signal de-rate — new generic sizing overlay, currently bear-fade
only: `execution['same_day_signal_derate'] = 0.10` sizes each of the day's
signals at `max(floor, 1 - 0.10*(n-1))` where n = that strategy's SIGNAL
count that day (ex-ante staged count, NOT fills — only ~1/3 of signals fill,
but high signal count itself marks the violent-selloff days where per-trade
edge degrades). `same_day_derate_floor` = 0.30. Composes multiplicatively
with frag bands (April 2024: 5 signals x high fragility -> 0.15x). Evidence:
scratch/lev3x_fade_bear_sizing_rule.py (same totR, worst 2-day window
-6.2R -> -4.5R).

Aligned sites -- change together (order_staging needs nothing: takes staged
sizes as-is):
- `strategy_config.py` execution `same_day_signal_derate` /
  `same_day_derate_floor` + shared formula `same_day_derate_mult()`
- `pages/strat_backtester.py` sizing step 3b4 (counts staged candidates
  per (strategy, day) in a pre-loop pass, post earnings-blackout)
- `daily_scan.py` post-pass 5c (after the cross-strategy overlap clamp;
  runs post-loop because n is only known after the strategy's ticker loop;
  counts per (Strategy_Name, Scan_Source); rescales Shares/Risk_Amt/Notional
  and stamps Sizing notes)
- Guard: `tests/test_same_day_derate.py` (carve-out partition, filter
  invariants, formula boundaries, single-carrier assertion)

## 3x Leader Gap Fade (pilot, 2026-07-10)

Capitulation fade on 3x ETFs whose UNDERLYING is spiking on fear. Universe =
`LEV3X_ALL` minus `LEV3X_BULL_EQ` (21 names: 13 bear-eq + TMF/TMV + 6 cmdty).
Filters: 2/5/10/21d rank > 80 (consec 1) AND 252d rank > 95 — the leader is
REQUIRED, the inverse of the other two 3x fades' <65 exclusion, so same-day
same-ticker cross-fire with them is impossible by construction (a ticker
cannot be <65 and >95 at once). Tape gate: T+1 open > close + 0.25 ATR,
resolved LIVE by order_staging's generic `T1_Open_Filters` gate (fail-closed;
scanner only stamps the JSON spec — it cannot see tomorrow's open). Entry:
Limit (Open + 0.75 ATR), OVS convention. 2-day time exit, NO STOP: stops
1.0-2.0 ATR (day-1 and day-2 armed) all destroyed the edge — adverse
excursion > 1 ATR is the normal path before the reversal (non-bull +23.7R ->
-39.8R at 1.0 ATR). The demanding entry IS the risk control (worst no-stop
trade -2.95R). Bull-eq exclusion is STRUCTURAL: every selectivity layer makes
bull-eq worse (strictest cell 0-for-7, avgR -1.28; losses span five bull
regimes 2018-2026) — do not re-add.

Sizing: 25 bps nominal (x GRM). Deliberately EXEMPT from frag_risk_bands and
same_day_signal_derate — the edge lives on exactly the high-fragility
multi-signal days those overlays would cut (Sept 2022, Apr 2025). Tail risk
is bounded instead by the per-strategy 250 bps daily cap (engine + live
aligned book-wide 2026-07-10 — see "Daily Risk Caps"; a 7-signal day at
37.5 eff = 262.5 bps trims ~5%). Validation
(2026-07-10): 31 trades / 15 episodes 2011-2025, avgR +0.80, PF 2.82,
episode-clustered t = 2.17, LOYO floor 1.55, drop-best-episode +9.4R @
t = 1.79, bootstrap P(<=0) = 2.1%. Pilot conviction — consider 40 bps only
after clean out-of-sample quarters.

Aligned sites — change together:
- `strategy_config.py` — the entry + `LEV3X_BULL_EQ` (source of truth)
- `pages/strat_backtester.py` / `daily_scan.py` — nothing bespoke; flows
  through generic paths (perf filters, T1_Open_Filters stamp, 0.75 ATR
  limit parse, max_one_pos, per-strategy daily cap)
- `order_staging.py` (OneDrive) — generic T1 gate enforces the gap at the
  IBKR T+1 open; per-strategy cap via the book-wide 250 bps default (a
  strategy-specific override existed for a few hours on 2026-07-10 and was
  removed when the book default aligned at 250)
- Guard: `tests/test_lev3x_leader_gap_fade.py`. Studies:
  `scratch/lev3x_fade_leader_*.py` (expansion, stops, entries, ovs_entry,
  class_split, bulleq_clusters, bulleq_strict, validation, capcheck,
  book_parity)

## Trend Sleeve (pilot, 2026-07-02)

`trend_sleeve.py` + `.github/workflows/trend_sleeve.yml`: monthly 12-ETF
trend-following ballast at 0.6x NAV (combo = 12-1 momentum AND 10-month MA,
long/flat, inverse-vol slots capped 20%, cash otherwise). Universe = SPY QQQ
IWM EFA EEM FXI VNQ GLD SLV DBC TLT LQD — NO USO (roll decay) and NO UUP
(capital inefficiency: 20% slot for +0.00%/mo contribution + K-1; costs 2022
+0.5% -> -1.9%, accepted). Sector-ETF / intl-single expansion tested and
REJECTED (equity slots crowd out diversifiers, 2008/2022 flip negative);
exhaustion scale-down overlay REJECTED (Sharpe flat). Signals on the month's
last trading-day close, staged MOO (TIF=OPG) to the `Trend` Sheets tab for
next-session execution; held-share state in `trend_sleeve_state.json` (R2 —
the month-end run computes DELTAS against it; if staged orders were never
executed, clear the state or the next rebalance is wrong). The workflow runs
weekdays 21:35 UTC (AFTER update_master_prices' 21:10 PM cron — the script
hard-fails if today's close is missing) and no-ops except on the last trading
day; `Execute_On` (next ET trading day after the run) gates submission.
FULLY AUTOMATED end-to-end: order_staging.py (`load_trend_rows`) reads the
tab on Execute_On morning and emits naked-MOO rows (appended AFTER risk caps,
excluded from PA/execution_2); eq_order_entry.py places them as MKT/OPG
parent-only (Exit_Condition_Time='NONE' -> no exit legs — positions unwind
via future rebalance SELL rows). Ballast ONLY — it loses ~-0.4%/mo in
high-fragility months (frag_risk_bands handles that hole). Scale to 1.0x of
the fraction only after 2 clean quarters. Studies: scratch/tf_universe_study.py,
scratch/ultracode_research/trend-following.md + trend_prework_gates.md.

## OLV Sector Loss Gate (2026-07-02)

`execution['sector_loss_gate'] = {'window_td': 10, 'max_realized_r': -2.0}`
(OLV only): skip a new signal when the strategy's realized R in the SAME
SECTOR over the trailing 10 trading days is -2R or worse -- the sector dip is
demonstrably trending, not bouncing. Motivated by June 2026: one oil cluster
re-signaled ~30x lost -20.4R (worst DD in the ledger) entirely below fragility
50, out of any dial's jurisdiction. Count caps and MTM gates were tested and
REJECTED (sector clustering is usually OLV's WINNING mode -- a 2-cap costs
+52R over 20y; live stops truncate unrealized pain before an MTM gate can
see it). With the corrected sector map (2026-07-03) the gate drops 20
net-losing trades (-5.3R, i.e. it ADDS R) over 20y and removes ~50% of a
June-type cluster -- the drop list is precisely the oil complex.
UNKNOWN-sector tickers PASS THROUGH in both gate sites: never pool no-sector
names into one pseudo-sector (the 2026-07-03 live bug: USO was gated off
unrelated UNKNOWN names' losses). Study: scratch/olv_cap_study*.py.

Aligned sites -- change together:
- `strategy_config.py` OLV execution `sector_loss_gate` (source of truth)
- `pages/strat_backtester.py` candidate gate (chronological loop keeps its own
  closed-trade log; exits strictly before signal date)
- `daily_scan.py` `sector_gate_blocked()` -- reads recent closed trades from
  `data/backtest_trades_full.parquet` (deploy_site's ledger build now mirrors
  it to R2; `daily_screener.yml` pulls it). Missing ledger/sector map = gate
  off with a printed notice (fail-open overlay).
- `data/sector_map.parquet` (committed): ticker->sector union of yfinance
  sector_overrides + FMP symbol_master + a curated sector/commodity ETF
  table, 1,460 tickers. Rebuild: `scripts/build_sector_map.py`.
- Guard: `tests/test_sector_loss_gate.py`.

Ledger provenance + integrity (2026-07-06, after a false TS/USO block): the
ledger is a FULL BACKTEST REBUILD, not a fill record -- marginal limit fills
flicker between vintages as yfinance revises recent bars, and the gate's -2.0R
threshold is a knife edge (the false block was -2.008R from a since-rebooked
trade in a weekend vintage of unknown origin). Mitigations, all in place:
- `build_trade_ledger.py` embeds provenance in the parquet schema metadata
  (`ledger_build_utc`, `ledger_source` = gha:<run_id> | local:<host>,
  `ledger_git_sha`, `ledger_rows`) and prints a vintage diff vs the prior
  ledger (new/gone/rebooked trades touching the last 15td) on every build.
- R2 upload of the prod ledger key is gated behind `--upload`; only
  `deploy_site.yml` passes it. Local runs (`refresh_view.py`) build but never
  overwrite the key that gates live orders.
- `daily_scan.py` prints the ledger's provenance at gate load and warns when
  the vintage is >4 days old or was built outside GHA (still fail-open).
- Blocked-signal notes name every contributing exit (ticker, date, R) so a
  block stays auditable after its vintage is overwritten.

## Stop-Arming Convention (book-wide, 2026-06-09)

Stop legs ARM AT THE NEXT SESSION, not at the fill. Decided after measuring
81 entry-day-stop episodes over 24y: booking -1R each vs arming on day 2 cost
-33R book-wide (dip-buy limit entries get stopped at max fear; a third of
MonFri's day-1 stop-outs went on to hit +2R targets).

Aligned across both sides -- change one, change both:
- `pages/strat_backtester.py` (`process_signals_fast`): entry-day stop check
  gated on `execution['stop_active_entry_day']`, **default False** (= day-2
  arming). Set True on a strategy to model a day-1-armed stop.
- `eq_order_entry.py` (in `C:\Users\mckin\OneDrive\trading_ibkr\`): STP
  child submitted with `goodAfterTime = next_session_gat()` (next trading day
  09:30, BDay-aware; holiday dates harmlessly defer to the next real session).
  Still in the OCA group, so a TARGET/TIME fill cancels the inactive stop.

Related conventions: entry-day TARGETS are never credited in the backtest
(intraday timing vs fill is ambiguous); OVS has `use_stop_loss=False` entirely
(its day-one valve is the Friday-only EOD-DD, see section above).

## Stop-Fill Convention — gap-through + slippage (book-wide, 2026-06-27)

A stop the bar GAPS THROUGH fills at the OPEN, not the stop. The old engine
always booked the exit at exactly `stop_price`, which pinned every stop-out at
exactly -1R and understated the gap-down tail (the website showed OLV — and
every stop strategy — "never losing more than 1R"). The realized fill is now the
worse of the stop and that day's open, plus slippage.

`process_signals_fast` (`pages/strat_backtester.py`) — drives the full-history
ledger (`scripts/build_trade_ledger.py` -> site) AND `daily_portfolio_report.py`:
- `_stop_fill_price(direction, stop_price, day_open, gap_fill, slip_bps, gap_slip_bps)`
  is the single fill model. Long: `min(stop, open)`; Short: `max(stop, open)`.
- Slippage: `STOP_SLIP_BPS = 3.0` on EVERY stop fill, plus an ADDITIONAL
  `STOP_GAP_SLIP_BPS = 10.0` (so 13 bps total) when the bar gapped through.
  Always worsens the fill (long sells lower, short covers higher). Targets and
  time exits get NO slippage. OVS EOD-DD (close exit) is untouched.
- New kwargs `stop_gap_fill=True, stop_slip_bps=3.0, stop_gap_slip_bps=10.0`
  default to the prod behavior; pass `stop_gap_fill=False` to reproduce the
  legacy fill-at-stop for before/after measurement.
- Entry-day stop (off by default) gets slippage only — no gap-to-open, since the
  open precedes the intraday limit fill.
- Scale-invariant under the dividend-adjustment rule: the stop is relative and
  `Open` is on the same adjusted basis within a run, so both scale by the same
  factor (CLAUDE.md "Dividend-Adjustment Basis").

Impact (full book, 2003-2026, flat $750k, `scratch/stop_gap_slippage_impact.py`):
85 of 434 stop-outs (~20%) gapped through. Book TotR 605.9 -> 560.2 (-45.7R),
AvgR 0.525 -> 0.485, worst single trade -1.0R -> -4.56R, -$157.7k flat (~8% of
these strategies' PnL). OLV: 25/116 stops gapped, TotR 193.5 -> 182.9, worst
-1.0R -> -2.29R.

Live trading was already correct (IBKR STP -> market order fills at the gap
open); this only removes backtest/ledger/site optimism. `pages/backtester.py`
(interactive UI) is a separate engine: its persistent-limit path already does
`min(Open, stop)` (line ~2439); the simpler paths (~2312-2351) still fill at the
stop and would need the same treatment for full parity (deliberately separate
exploration surface, not yet aligned).

## OLV Entry-Order Live Window (T+3, 2026-06-24)

The OLV (Oversold Low Volume) persistent close-0.25 ATR limit is cancelled if
unfilled after **3 trading days** (T+1..T+3), not the full 10-day hold. A fill
inside the window is kept and unchanged (its hold is still reduced by wait time
off `hold_days`); a signal that hasn't filled by T+3 close is dropped.

Evidence (`scratch/olv_fill_window.py`, bucketing the full ledger by fill day):
89% of OLV fills land by T+3. The day 4-10 fills add ~0 total R (+211 -> +211 R
over 21y) while diluting per-trade edge: avgR +0.637 (T+3) vs +0.566 (T+10),
win 62.8% vs 60.6%, PF 2.90 vs 2.65. So total return is unchanged but
risk-adjusted quality improves, and capital isn't tied up in stale GTC orders
that mostly fill into names that kept bleeding for a week+.

Generic mechanism: `execution['fill_window_days']` caps the persistent-limit
fill search; **defaults to `hold_days` when absent**, so the other 5 persistent
strategies are untouched. Aligned sites (change together):
- `strategy_config.py` — OLV execution `fill_window_days: 3` (source of truth).
- `pages/strat_backtester.py` — `fill_window` bounds `search_end` in both
  persistent fill loops; the hold reduction still references `hold_days`.
  Drives the ledger + `daily_portfolio_report.py`.
- `daily_scan.py` — stamps `Fill_Window_Days` on every primary staging row.
- `order_staging.py` (in `OneDrive\trading_ibkr\`) — stamps `Entry_Expire_Time`
  = signal + `Fill_Window_Days` BDays = `Exit_Condition_Time` − (1 + hold − fill)
  BDays, into the execution CSV + `execution`/`execution_2` tabs. Defaults to
  `Exit_Condition_Time` (the 10-day-hold expiry) unless the row carries a valid
  `0 < Fill_Window_Days < Hold_Days`, so only OLV is affected.
- `eq_order_entry.py` + `pa_order_entry.py` (same dir) — the persistent GTC
  parent's `goodTillDate` reads `Entry_Expire_Time` (falls back to the time-exit
  `gat_time` when absent/blank). The TIME exit leg still uses `gat_time`, so a
  filled OLV position keeps its full reduced hold — only the unfilled entry order
  is cancelled early. The order is live T+1..T+3 (expires T+3 15:59).
- `pages/backtester.py` UI still uses `holding_days` as its fill window (an
  exploration surface, deliberately separate from the prod-locked rule).
- Regression coverage: `tests/test_olv_fill_window.py` (backtest engine);
  the live date math is validated by the entry-expire chain (daily_scan exit-date
  build ↔ order_staging back-computation, identical `CustomBusinessDay` calendar).

## Cloudflare R2 Cache + GHA Migration

As of 2026-04-30, the nightly pipeline runs entirely in GitHub Actions. The local Task Scheduler retains the radar tasks plus (as of 2026-05-13) two AM `workflow_dispatch` triggers that bypass GitHub's congested 8-9 UTC cron-queue lag. R2 is the persistence layer that lets cloud workflows share parquet caches.

### R2 secrets (in GHA repo settings)
- `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET=seasonals-cache`

### Bucket contents (key-value)
- `master_prices.parquet` — full ~2000 ticker × 25-yr OHLCV (~50-200 MB). Read by `daily_scan` (ALL scopes — **cache-first for every ticker**, incl. the liquid + 3x-ETF universes; yfinance is only a fallback for names the cache lacks, e.g. carets/delisted) and `daily_portfolio_report.py`. As of 2026-06-11 the 42 LEV3X names (DUST/JDST/TQQQ/…) were backfilled in so the liquid scan no longer depends on a live pre-market yfinance pull (that pull returned a stale bar on 2026-06-11 and silently zeroed the liquid tier). Written by `update_master_prices.yml` twice on weekdays (AM via local workflow_dispatch ~4:17 AM ET + PM via 20:30 UTC cron); its universe = whatever tickers already exist in the parquet, so backfilled names are auto-maintained. Pre-market runs pass `--exclude-today` so yfinance placeholder bars never enter the cache.
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
| `risk_report.yml` | Weekdays 21:15 UTC (5:15 PM ET) | Daily risk dashboard email (fragility dials + signals + forward returns). Writes `data/rd2_fragility.parquet` APPEND-ONLY (since 2026-07-02): history is frozen point-in-time, only new dates append (same-day rerun refreshes today's row). The full recompute still runs in memory for the report itself. Guard: `tests/test_fragility_append.py`. This series sizes live orders (`frag_risk_bands`, see "Fragility Risk Bands") — do not revert to full rewrites; recompute vintages drifted up to ~7 pts on the 63d dial. |
| `verify_fills.yml` | Weekdays 21:15 UTC | Post-close fill verification — updates Trade_Signals_Log. |
| `deploy_site.yml` | Reusable workflow (`workflow_call`), invoked by the `deploy-site` job at the tail of `daily_screener.yml` (`needs: run-scanner`) so it runs in the SAME run, right after the scan succeeds — 2x/trading day (after the ~4:47 AM ET dispatch scan and the PM bookend). Replaced the old best-effort `workflow_run` chain, which was silently not firing. A skipped (AM fallback) or failed scan skips the deploy and the prior deploy stays up. `workflow_dispatch` retained for manual rebuilds. | Builds + deploys the private analytics site to Cloudflare Pages (behind Cloudflare Access). Pipeline: R2 caches → `scripts/build_trade_ledger.py` (full-history ledger) → `scripts/build_signal_charts.py --all --upload --skip-existing` (renders only NEW per-trade charts to R2, best effort) → `daily_seasonal_ideas.py` (best effort) → `scripts/build_risk_json.py` (best effort) → `scripts/build_site.py` (JSON payloads + `site/` assets → `dist/`) → wrangler Pages deploy (config-driven via `wrangler.toml`; no positional dir, so the CHARTS R2 binding applies). Needs `CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID` secrets. One-time setup: `docs/private_site_setup.md`. Operational runbook (failure modes, decisions log,
trigger chain, out-of-repo file map): `docs/site_runbook.html`. |
| `weekly_rundown.yml` | Sundays 14:00 UTC (9 AM ET) | Tabloid PDF with all risk charts + radar digest body. |
| `trend_sleeve.yml` | Weekdays 21:35 UTC (no-ops except the month's last trading day) | Monthly trend-following ballast rebalance — writes MOO orders to the `Trend` Sheets tab + state to R2. See "Trend Sleeve" section. |
| `execution_report.yml` | Weekdays 20:30 AND 21:30 UTC — `daily_execution_report.py` gates on WHICH cron fired (`GHA_SCHEDULE` = `github.event.schedule`) + the date's DST regime, so exactly one sends at ~4:30 PM ET year-round even when GHA cron lag starts the run an hour+ late (the old hour==16 gate silently dropped the 2026-07-08/09 emails). Cron strings must match `EDT_CRON`/`EST_CRON` in the script; unknown cron fails open (sends). | Nightly email of LIVE primary-account positions (mirrors the site Execution tab). Pulls the execution-broker DO's `/book` snapshot (Bearer `STATUS_TOKEN`; URL via `EXEC_BROKER_URL` secret, defaults to the workers.dev URL), excludes OPT rows, and enriches each position from its own working exit legs: target = closing LMT `lmt`, stop = closing STP `aux` (NA if none), time stop = closing MKT leg's `goodAfterTime` date, strategy = 3rd pipe field of any leg's `orderRef` (requires `order_ref` in `book_snapshot.py` — added 2026-07-08 in `OneDrive\trading_ibkr`). No strategy-tagged legs → "Trend Sleeve" (symbol in `trend_sleeve_state.json` on R2) else "Discretionary". Recipients: repo variable `EXECUTION_REPORT_RECIPIENTS` (comma-separated; defaults to mckinleyslade@gmail.com). Guard: `tests/test_execution_report.py`. |

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

## Private Site (Cloudflare Pages)

Static, client-side analytics site deployed nightly by `deploy_site.yml` to
Cloudflare Pages project `seasonals-mslade`, locked behind Cloudflare Access
(email OTP, allowlist = mckinleyslade@gmail.com). One-time setup doc:
`docs/private_site_setup.md`.

- **Frontend** lives in `site/` (committed): `index.html` (portfolio app),
  `ideas.html`, `signals.html`, `charts.html` (per-trade chart gallery),
  `risk.html` + `assets/` (vanilla JS + Plotly CDN, no build step, no
  framework). `site/_headers` sets no-store on `/data/*`.
- **Payload contract** (written by `scripts/build_site.py` into `dist/data/`):
  `meta.json`, `trades.json` (columnar full ledger), `strategy_daily.json`
  (per `Strategy||Tier` daily MTM PnL on the FLAT $750k basis + book totals),
  `positions.json`, `exposure.json`, `correlation.json`, `charts.json`
  (per-trade chart manifest: stable image path + MAE/MFE), plus optional
  `ideas.json` / `signals.json` (Sheets snapshot) / `risk.json` / `fragility.json`
  (rd2 fragility dial series feeding the portfolio page's interactive sizing
  adjuster — per-trade what-ifs on dial/MA/threshold/floor/boost; forces the
  realized-at-exit curve basis while active) / `gate_lab.json` (sector-loss-gate
  counterfactual: blocked trades + gate-on/off realized curves, diffed from
  `data/backtest_trades_nogate.parquet` — a no-gate engine pass
  `build_trade_ledger.py` writes alongside the ledger; drives the portfolio
  page's gate-history section and its "All trades (+gate-blocked)" filter
  toggle, which also forces the realized-at-exit basis while on) /
  `ext_lab.json` (OVS hold-extension counterfactual — what-if lab, NOT a live
  rule: losing T+2 time exits rebooked to T+5 with the 2-ATR target live, a
  post-pass `build_trade_ledger.py` writes to
  `data/backtest_trades_ovsext.parquet`; drives the portfolio page's
  hold-extension section and its "OVS losers to T+5" filter toggle, which
  swaps the rebooked exits in by trade_id and forces the realized-at-exit
  basis while on. Evidence: scratch/ovs_hold_extension_*.py).
- **Trade charts** (the `charts.html` gallery): `scripts/build_signal_charts.py`
  renders a candlestick per trade (126 td before signal -> trade -> 63 td after
  exit; white/black candles, green/red volume, Signal/Entry/Exit verticals,
  dotted entry/stop/target, MAE/MFE stats box) and uploads to R2 under the
  `charts/` prefix. Keys are STABLE (`signals/<strategy>/<TICKER>_<YYYYMMDD>.png`,
  see `signal_chart_common.chart_relpath`) — not trade_id (reshuffles) or exit
  type (can flip). The site never bundles the PNGs (~360 MB); the
  `functions/chartimg/[[path]].js` Pages Function streams them from the `CHARTS`
  R2 binding on demand (route `/chartimg/*` -> R2 key `charts/*`; route differs
  from `/charts` so it doesn't shadow the gallery page). `deploy_site.yml`
  renders only NEW charts each run (`--all --upload --skip-existing`, best
  effort). Full backfill: `python scripts/build_signal_charts.py --all --upload`.
- **Sizing-basis rule**: client-side filtering recomputes everything on the
  flat $750k basis because per-trade dollars are additive. Strategy/tier/date
  filters get exact daily MTM curves (sum of per-strategy series);
  direction/ticker filters fall back to realized-PnL-at-exit step curves and
  the UI shows a badge. The compounded curve is shipped read-only — it cannot
  be decomposed per-filter (sizing depended on whole-book equity).
- **Local dev**: `python scripts/build_site.py --no-signals` then
  `python -m http.server 8123 --directory dist`. `--no-mtm` skips the slow
  payloads when iterating on frontend only.

## Google Sheets Integration

Tab layout in the `Trade_Signals_Log` workbook:
- `Order_Staging` — Liquid-tier signals (Limits, T+1 Open, Persistent GTC). Cleared + rewritten by every `daily_scan` run with `Scan_Source='Liquid'`.
- `Overflow` — Overflow-tier signals (same entry types, no MOC). Cleared + rewritten by `daily_scan --scope=overflow|all` with `Scan_Source='Overflow'`.
- Both staging tabs carry a `Manual_Limit` column (emitted empty by the scanner): type a price into it to pin that signal's entry — order_staging uses it verbatim as a LMT and anchors the bracket to it, skipping the gap clamp. Rows survive only until the next scan's clear+rewrite, so manual rows/pins must be added AFTER the ~4:47 AM ET scan and BEFORE order_staging runs (e.g. the 2026-07-06 TS/USO makeup rows via `scratch/stage_makeup_ts_uso.py`). Entry expiry is back-computed from `Time_Exit_Date` − (1 + `Hold_Days` − `Fill_Window_Days`) BDays, NOT from `Scan_Date`, so a makeup row can carry its true original schedule.
- `moc_orders` — MOC entries from liquid tier only (`save_moc_orders` skips overflow rows). Currently vestigial: the strategy book has no Signal Close entries, so this tab is never written. Reactivates automatically if any strategy is set to `entry_type='Signal Close'`.
- `Seasonal` — tradeable seasonal-ideas tickets (longs + non-equity shorts). Written by `seasonal_order_staging.py` from `data/daily_seasonal_ideas.json`, `Scan_Source='Seasonal'`. Separate pipeline from the systematic book. Entry type per instrument (validated geography rule): US single stocks + US-session equity ETFs → `REL_OPEN` limit (0.25 ATR, DAY); everything that gaps overnight (intl/commodity/bond/FX ETFs, GLD/TLT) → `MOO` (market-on-open, `TIF=OPG`). Sizing: 20 bps/trade (13 bps in midterm years, `year%4==2`), 1% aggregate daily cap. order_staging must add `MOO` handling — see `docs/seasonal_order_staging_spec.md`.
- `sznl_nostage` — NOT auto-executed. Single-stock equity shorts (sized, tagged `[eq-short]`) + non-tradeable signals (futures/index/FX/crypto, `Quantity=0`, `Order_Type=NONE`, tagged `[need-proxy]` pending the proxy-ETF promotion). order_staging does not read this tab.
- `Trade_Signals_Log` (sheet1) — append-only signal history.
- `Portfolio` — open-positions snapshot from `daily_portfolio_report.py`.
- `execution`, `execution_2` — order_staging.py output for primary + small-account execution.

`daily_scan.py` writes both `Order_Staging` and `Overflow` via `save_staging_orders(..., tier_filter='Liquid'|'Overflow')`. The function clears+rewrites only the tier it's responsible for (so a `--scope=liquid` run never touches `Overflow`).

`order_staging.py` (in `C:\Users\mckin\OneDrive\trading_ibkr\`) reads BOTH tabs and concatenates with `Scan_Source` distinguishing tier. Applies the OVS 2-path gap-tier sizer + path-2 daily aggregate cap + global 2.5% daily risk cap before submitting to IBKR.

`verify_fills.py` updates Trade_Signals_Log with fill status post-close.

Auth: `gspread` with GCP service account from Streamlit secrets / `GCP_JSON` env var (GHA) / `credentials.json` (local).
