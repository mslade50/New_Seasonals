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
├── daily_scan.py                   # Production scanner + email + Google Sheets
├── daily_risk_report.py            # Daily risk email (fragility dials + signals + forward returns)
├── daily_portfolio_report.py       # Daily portfolio health report (imports from strat_backtester)
├── weekly_market_rundown.py        # Weekly PDF rundown (tabloid landscape, 11 chart pages)
├── radar_weekly_summary.py         # Weekly radar digest (reads daily briefs, Claude distills best-of)
├── verify_fills.py                 # Post-close fill verification (updates Google Sheets)
├── indicators.py                   # Shared indicator library
├── abs_return_dispersion.py        # S&P 500 dispersion metric (~505 tickers)
├── risk_dashboard_clean_sheet.md   # Risk Dashboard V2 design doc
├── pages/                          # Streamlit pages (FLAT — no subfolders)
│   ├── risk_dashboard_v2.py        # Multi-layer regime monitor (standalone)
│   ├── backtester.py               # Strategy backtesting UI
│   ├── strat_backtester.py         # Extended backtester
│   ├── heatmaps.py                 # Market heatmap inspector
│   ├── correlation_heatmaps.py     # Correlation analysis
│   ├── sector_trends.py            # Sector trend analysis
│   ├── seasonal_sigs.py            # Seasonal signals
│   └── user_input.py               # User input page
├── scripts/                        # Task Scheduler PowerShell wrappers
│   ├── run_radar_weekly.ps1        # Sunday 8:30 AM ET — runs radar digest, commits + pushes
│   └── setup_radar_weekly_task.ps1 # One-time admin setup for Task Scheduler
├── data/                           # Persistent cache (parquet files + radar digest)
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

**daily_portfolio_report.py** imports backtesting logic from `strat_backtester.py`. Both must stay in sync with `daily_scan.py` for signal detection, sizing, and trade processing. `ACCOUNT_VALUE` from `strategy_config.py` is the single source of truth for portfolio sizing across all three.

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
| `LIQUID_UNIVERSE` | `strategy_config.py` | 190 | Liquid stocks for strategies |
| `SECTOR_ETFS` | `risk_dashboard_v2.py` | 11 | SPDR sector ETFs |
| `VOL_TICKERS` | `risk_dashboard_v2.py` | 4 | SPY, ^VIX, ^VIX3M, ^VVIX |
| `CROSS_ASSET_TICKERS` | `risk_dashboard_v2.py` | 7 | LQD, HYG, IEF, UUP, ^MOVE, ^TNX, ^IRX |
| `TAIL_RISK_TICKERS` | `risk_dashboard_v2.py` | 1 | ^SKEW |
| `SIGNAL_CACHE_PATH` | `risk_dashboard_v2.py` | — | `data/risk_dashboard_signal_state.json` |

## Automated Reports & Email Pipeline

| Report | Script | Trigger | Format | Schedule |
|--------|--------|---------|--------|----------|
| Daily Scan | `daily_scan.py` | GitHub Actions | HTML email + Google Sheets | Weekdays 5x (9:13, 17:40, 18:45, 19:30, 20:13 UTC) |
| Risk Report | `daily_risk_report.py` | GitHub Actions | HTML email (inline images) | Weekdays 21:15 UTC (5:15 PM ET) |
| Portfolio Report | `daily_portfolio_report.py` | GitHub Actions | HTML email | Daily 21:00 UTC (5 PM ET) |
| Fill Verification | `verify_fills.py` | GitHub Actions | Google Sheets update | Weekdays 21:15 UTC |
| Radar Weekly Digest | `radar_weekly_summary.py` | Local Task Scheduler | Markdown → `data/radar_weekly_summary.md` | Sundays 8:30 AM ET |
| Weekly Rundown | `weekly_market_rundown.py` | GitHub Actions | PDF attachment + HTML body (radar digest) | Sundays 14:00 UTC (9 AM ET) |

### Sunday Pipeline (two-step)
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
- `daily_scan.py` stages orders to Google Sheets (MOC + Order_Staging + Trade_Signals_Log)
- `verify_fills.py` updates Trade_Signals_Log with fill status post-close
- Uses gspread with GCP service account (from Streamlit secrets) or local `credentials.json`
