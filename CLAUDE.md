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
├── indicators.py                   # Shared indicator library
├── abs_return_dispersion.py        # S&P 500 dispersion metric (~505 tickers)
├── risk_dashboard_clean_sheet.md   # Risk Dashboard V2 design doc
├── pages/                          # Streamlit pages (FLAT — no subfolders)
│   ├── risk_dashboard.py           # V1: dispersion-based risk dashboard
│   ├── risk_dashboard_v2.py        # V2: multi-layer regime monitor (standalone)
│   ├── screener.py                 # Daily strategy screener
│   ├── backtester.py               # Strategy backtesting UI
│   ├── strat_backtester.py         # Extended backtester
│   ├── heatmaps.py                 # Market heatmap inspector
│   ├── correlation_heatmaps.py     # Correlation analysis
│   ├── sector_trends.py            # Sector trend analysis
│   ├── seasonal_sigs.py            # Seasonal signals
│   └── user_input.py               # User input page
├── data/                           # Persistent cache (parquet files)
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

**Strategy modules** (`screener.py`, `strat_backtester.py`, `daily_scan.py`) all depend on `strategy_config.py` for `STRATEGY_BOOK`.

## Risk Dashboard V2 — Current State

**Phases 1 & 2 complete** (Layers 0–4). See `notes.md` for full details.

### Executive Summary (Layer 0)
One-screen briefing at the top of the page. Three components side-by-side:
- **Verdict banner** — compact single line: regime name, sizing ref, timestamp, point count
- **Risk dial** — Plotly gauge, continuous 0-100 fragility score (average of per-metric danger-zone depth, NOT a rescaled point count). Labels: Robust → Neutral → Fragile.
- **Narrative** — 2-3 sentence synthesis grouped by theme (vol complex, internals, plumbing). Context-aware, not a metric list.

Below the banner row: **Situation Board** — Plotly bullet chart, 17 metrics on a common 0-100 percentile x-axis. Each row has green/yellow/red background zones (respecting `invert` flag for metrics where low = bad) and a colored dot at the current percentile. Layer separators between Vol / Internals / Plumbing groups.

Point system underneath (collapsed expander): Alert = +1, Alarm = +2.
- 0 pts = Normal (1.00x) | 1-2 = Caution (0.75x) | 3-4 = Stress (0.50x) | 5+ = Crisis (0.25x)

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

### Layer 4: Tail Risk & Cost of Protection (auto-expands on Caution+)
- 4A: SKEW Index — time series with 120/140 bands. Disorderly stress detection: flags when SKEW falling (>3pts in 5d) while VIX rising (>3pts in 5d).
- 4B: Protection Cost Proxy — VIX3M × (SKEW/130), percentile-ranked over 5yr trailing window. Plotly gauge display (green/yellow/orange/red).
- 4C: Hedge Recommendation — decision tree based on regime × protection cost percentile. Outputs: sizing guidance, collar vs puts vs exposure reduction.

### Chart Defaults
- HAR-RV and VRP charts default to last 1 year. Double-click to zoom out to full history.
- Layer 3 charts use compact 200px height (vs 250px for Layers 1/2).

### Phase 3 TODO
- Bayesian composite (replace point system)
- Full S&P 500 breadth (replace sector proxy with ~500 constituents)
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
| `LEADERSHIP_ETFS` | `risk_dashboard.py` | 19 | Extended sector + industry ETFs |

## Google Sheets Integration
- `daily_scan.py` stages orders to Google Sheets (MOC + Order_Staging + Trade_Signals_Log)
- Uses gspread with GCP service account (from Streamlit secrets) or local `credentials.json`
