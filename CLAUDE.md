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

**daily_portfolio_report.py** imports backtesting logic from `strat_backtester.py`. Both must stay in sync with `daily_scan.py` for signal detection, sizing, and trade processing. `ACCOUNT_VALUE` from `strategy_config.py` is the single source of truth for portfolio sizing across all three. Runs in **GitHub Actions** (weekdays 21:30 UTC = 5:30 PM ET) — pulls `data/master_prices.parquet` and `data/earnings_calendar.parquet` from Cloudflare R2 before running. Reports cover both liquid (LIQUID_PLUS_COMMODITIES) and overflow (CSV_UNIVERSE − LIQUID_PLUS_COMMODITIES) universes — overflow-eligible strategies get a second deep-copied pass with `OVERFLOW_RISK_OVERRIDES` (only OLV 35→25 bps remains; OVS uses path-1 nominal 40 bps for both tiers). Workflow: `.github/workflows/portfolio_report.yml`.

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
| `OVERFLOW_ELIGIBLE_STRATEGIES` | `daily_scan.py` | 6 | OVS, OLV, LT Trend ST OS, St OS Sznl, 52wh Breakout, ATR Extended Gap Up (native 60 bps on overflow) |
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

## Cycle-Year Risk Tilt (OVS, 2026-06-10)

OVS runs at 0.75x risk in midterm years (year%4==2). Evidence: all six
midterm years 2006-2026 underperform (avgR +0.19 vs +0.49 non-midterm),
leave-one-year-out stable, damage concentrated in P1 decisive-gap entries
(+0.63 -> +0.23 avgR). ~1.5 sigma after episode clustering -> shrunk-Kelly
0.75x, not the full-conviction 0.4x. Validated by LOYO, NOT by re-running
the backtest with the rule on (in-sample rules flatter themselves).

Four aligned sites -- change together:
- `strategy_config.py` OVS execution `cycle_risk_mults: {2: 0.75}` (source of truth)
- `pages/strat_backtester.py` sizing step 3b2 (generic: any strategy with the field)
- `daily_scan.py` sizing step 2c2 (stamps the mult into Sizing notes)
- `order_staging.py` `OVS_CYCLE_MULTS` -- needed because the live OVS P1
  resize to a FIXED dollar target (OVS_PATH1_RISK_DOLLARS) clobbers the
  scanner's Risk_Amt, so the tilt must be applied to the target itself.

NOTE a live-vs-backtest divergence found during this work: order_staging
RETIRED the OVS mild-gap Path 2 (P1-only, fixed $3,000 target; mild gaps
dropped), while the backtest/ledger still models the 2-path scheme
(P2 = 407 trades, +0.20 avgR, +82R/24y). Engine has `ovs_p1_only` parameter
if the ledger should be aligned to live instead. Unresolved -- decide
whether to re-enable P2 live or flip the backtest to P1-only.

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
| `risk_report.yml` | Weekdays 21:15 UTC (5:15 PM ET) | Daily risk dashboard email (fragility dials + signals + forward returns). |
| `verify_fills.yml` | Weekdays 21:15 UTC | Post-close fill verification — updates Trade_Signals_Log. |
| `deploy_site.yml` | Reusable workflow (`workflow_call`), invoked by the `deploy-site` job at the tail of `daily_screener.yml` (`needs: run-scanner`) so it runs in the SAME run, right after the scan succeeds — 2x/trading day (after the ~4:47 AM ET dispatch scan and the PM bookend). Replaced the old best-effort `workflow_run` chain, which was silently not firing. A skipped (AM fallback) or failed scan skips the deploy and the prior deploy stays up. `workflow_dispatch` retained for manual rebuilds. | Builds + deploys the private analytics site to Cloudflare Pages (behind Cloudflare Access). Pipeline: R2 caches → `scripts/build_trade_ledger.py` (full-history ledger) → `scripts/build_signal_charts.py --all --upload --skip-existing` (renders only NEW per-trade charts to R2, best effort) → `daily_seasonal_ideas.py` (best effort) → `scripts/build_risk_json.py` (best effort) → `scripts/build_site.py` (JSON payloads + `site/` assets → `dist/`) → wrangler Pages deploy (config-driven via `wrangler.toml`; no positional dir, so the CHARTS R2 binding applies). Needs `CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID` secrets. One-time setup: `docs/private_site_setup.md`. Operational runbook (failure modes, decisions log,
trigger chain, out-of-repo file map): `docs/site_runbook.html`. |
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
  `ideas.json` / `signals.json` (Sheets snapshot) / `risk.json`.
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
