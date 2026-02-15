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

**Phase 1 complete** (Layers 0, 1, 2). See `notes.md` for full details.

### Layer 0: Composite Verdict
Rules-based point system. Alert = +1, Alarm = +2.
- 0 pts = Normal (1.00x) | 1-2 = Caution (0.75x) | 3-4 = Stress (0.50x) | 5+ = Crisis (0.25x)

### Layer 1: Volatility State
- 1A: HAR-RV (Yang-Zhang at 1d/5d/22d)
- 1B: VRP = (VIX/100)^2 - RV_22d^2
- 1C: VIX Term Structure (VIX/VIX3M)
- 1D: VVIX

### Layer 2: Equity Market Internals
- 2A: Breadth (sector ETF proxy — % above 200d/50d SMA)
- 2B: Absorption Ratio (PCA on 63d sector returns)
- 2C: Cross-sectional dispersion + avg pairwise correlation (2x2 grid)
- 2D: Hurst exponent (DFA, focus on 5d rate of change)

### Phase 2 TODO
- Layer 3: Credit spreads, yield curve, MOVE, dollar
- Layer 4: SKEW, protection cost, hedge recommendations
- Bayesian composite (replace point system)
- Full S&P 500 breadth (replace sector proxy)
- Historical regime backtesting

## Ticker Constants

| Variable | Location | Count | Description |
|----------|----------|-------|-------------|
| `SP500_TICKERS` | `abs_return_dispersion.py` | ~505 | Full S&P 500 constituents |
| `LIQUID_UNIVERSE` | `strategy_config.py` | 190 | Liquid stocks for strategies |
| `SECTOR_ETFS` | `risk_dashboard_v2.py` | 11 | SPDR sector ETFs |
| `VOL_TICKERS` | `risk_dashboard_v2.py` | 4 | SPY, ^VIX, ^VIX3M, ^VVIX |
| `LEADERSHIP_ETFS` | `risk_dashboard.py` | 19 | Extended sector + industry ETFs |

## Google Sheets Integration
- `daily_scan.py` stages orders to Google Sheets (MOC + Order_Staging + Trade_Signals_Log)
- Uses gspread with GCP service account (from Streamlit secrets) or local `credentials.json`
