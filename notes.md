# Development Notes

## Session: 2026-01-28 - Schema & Config Refactoring

### 1. Strategy Config Schema Update

**Files Changed:** `strategy_config.py`

Added two new blocks to each strategy for improved email/signal clarity:

```python
"setup": {
    "type": "MeanReversion",           # MeanReversion | Breakout | Seasonal | Momentum | Custom
    "timeframe": "Swing",              # Intraday | Overnight | Swing | Position
    "thesis": "Human-readable why",    # One-liner explaining the edge
    "key_filters": [                   # Bullet points of active filters
        "5D rank < 50th %ile",
        "Seasonal > 33",
        ...
    ]
},
"exit_summary": {
    "primary_exit": "5-day time stop",
    "stop_logic": "2.0 ATR below entry",
    "target_logic": "8.0 ATR above entry",
    "notes": "Dynamic sizing info or special conditions"
}
```

**Purpose:** `daily_scan.py` email can now pull `setup.thesis` and `exit_summary` instead of parsing raw settings.

---

### 2. Ticker Universe Deduplication

**Files Changed:** `strategy_config.py`

**Problem:** 7 strategies shared identical 190-ticker lists = ~1,330 duplicated entries.

**Solution:** Created shared universe constants at top of file:

| Variable | Count | Used By |
|----------|-------|---------|
| `INDEX_ETFS` | 5 | Weak Close Reversion, Index Seasonals |
| `SECTOR_INDEX_ETFS` | 26 | Weak Close Decent Sznls |
| `LIQUID_UNIVERSE` | 190 | 7 strategies |
| `LIQUID_NO_INDEX` | 188 | Liquid Seasonals (short term) |
| `LIQUID_PLUS_COMMODITIES` | 198 | Deep Oversold Reversion (5d) |

**Result:** File size reduced from 52KB to 43KB. Single source of truth for ticker lists.

**Note:** `LIQUID_NO_INDEX` is computed dynamically:
```python
LIQUID_NO_INDEX = [t for t in LIQUID_UNIVERSE if t not in ['^GSPC', '^NDX']]
```

---

### 3. Backtester Export Fixes

**Files Changed:** `backtester.py`

#### A. Universe Placeholder
Export now outputs:
```python
# MANUAL EDIT REQUIRED: Replace with universe variable name (no quotes)
# Options: INDEX_ETFS, SECTOR_INDEX_ETFS, LIQUID_UNIVERSE, LIQUID_NO_INDEX, LIQUID_PLUS_COMMODITIES
"universe_tickers": "CHANGE_ME",
```

**Workflow:** After paste, replace `"CHANGE_ME"` with the variable name (no quotes).

#### B. True/False Capitalization Fix
Changed from `json.dumps()` to `pprint.pformat()` for export.

- **Before:** `"use_sznl": true` (JSON syntax - breaks Python)
- **After:** `'use_sznl': True` (Python syntax - works)

Download button now saves `.py` file instead of `.json`.

---

### 4. Auto-Generation Helper Functions

**Files Changed:** `backtester.py`

Added four helper functions that auto-populate `setup` and `exit_summary` from strategy params:

- `_infer_strategy_type(params)` → Returns `(type, thesis)`
- `_generate_key_filters(params)` → Returns list of active filter descriptions
- `_generate_exit_summary(params)` → Returns exit logic dict
- `_infer_timeframe(params)` → Returns timeframe category

These run inside `build_strategy_dict()` so exports are pre-populated with reasonable defaults.

---

## Pending / Future Work

- [ ] **Update `daily_scan.py` email template** to use new `setup.thesis` and `exit_summary` fields
- [ ] **Decouple hardcoded strategy names** in `daily_scan.py` (e.g., "Overbot Vol Spike" risk multiplier logic should use a config flag instead of string matching)
- [ ] **Unified indicator library** - move `calculate_indicators()` to shared `utils.py` for guaranteed parity between backtester and scanner

---

## Session: 2026-02-15 - Risk Dashboard V2 (Phase 1)

### New File: `pages/risk_dashboard_v2.py` (1,004 lines)

**Purpose:** Standalone market risk monitor. Completely independent from trading strategies — no imports from `strategy_config.py`, `strat_backtester.py`, `daily_scan.py`, or `indicators.py`.

**Design doc:** `risk_dashboard_clean_sheet.md` (project root)

#### Architecture: 3-Layer System

**Layer 0 — The Verdict**
- Rules-based point system: each metric in alert range = +1, alarm range = +2
- Regime classification: 0 pts = Normal (1.00x), 1-2 = Caution (0.75x), 3-4 = Stress (0.50x), 5+ = Crisis (0.25x)
- Color-coded banner + plain-English summary + expandable point breakdown

**Layer 1 — Volatility State** (left column, 4 metrics)
| Metric | Method | Alert | Alarm |
|--------|--------|-------|-------|
| 1A. HAR-RV | Yang-Zhang estimator at 1d/5d/22d | RV_1d > 2x RV_22d | RV_22d > 75th pctile & rising |
| 1B. VRP | (VIX/100)² - RV_22d² | < 25th percentile | Negative |
| 1C. VIX Term Structure | VIX / VIX3M ratio | > 0.95 | > 1.0 (backwardation) |
| 1D. VVIX | ^VVIX level | > 100 | > 120 |

**Layer 2 — Equity Market Internals** (right column, 5 metrics)
| Metric | Method | Alert | Alarm |
|--------|--------|-------|-------|
| 2A. Breadth | % of 11 sector SPDRs > 200d/50d SMA | < 60% w/ SPY near high | < 40% |
| 2B. Absorption Ratio | PCA eigenvalue₁/Σeigenvalues on 63d sector returns. **"Low & rising" logic** — fragility builds when AR is low but rising fast (Minsky transition). Chart has triangle markers at historical alert dates for visual validation. | AR < 40th pctile AND 21d Δ > +0.03 | AR < 40th pctile AND 21d Δ > +0.05 AND accelerating (21d Δ > 42d Δ) |
| 2C. Dispersion | Cross-sectional σ of 21d sector returns + avg pairwise corr | High disp | High disp + high corr |
| 2D. Hurst | DFA on SPY returns, **126d rolling**, box sizes [8,16,32,48,63]. **Empirical percentile bands** (P20/P80 of own history, expanding, min 252 obs) replace hardcoded 0.4/0.6 thresholds. 5d ΔH remains primary signal. | H > 80th pctile | H > 95th pctile |
| 2E. Days Since Correction | Trading days since last 5% and 10% peak-to-trough drawdown in SPY. Both shown in one styled box. | 5% streak > 80th pctile | 5% streak > 95th pctile |

#### Data

- **Source:** yfinance only — no broker connections, no API keys
- **Caching:** `@st.cache_data(ttl=3600)` for downloads; `data/` dir for persistent parquet
- **Ticker groups:**
  - Vol tickers: `SPY, ^VIX, ^VIX3M, ^VVIX`
  - Sector ETFs: `XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY`
- **Start date:** 2010-01-01 (configurable via sidebar slider, 5-15 years)
- **yfinance MultiIndex:** Handled via `.xs(ticker, level='Ticker', axis=1)` then flatten + capitalize

#### Critical Implementation Details

1. **yfinance MultiIndex bug:** ALL multi-ticker downloads return `(Price, Ticker)` MultiIndex. Must extract with `.xs()` then flatten.
2. **Pages directory is FLAT** — no subfolders under `pages/`.
3. **Optional import:** `SP500_TICKERS` from `abs_return_dispersion.py` (for future full breadth). Currently uses sector ETF proxy.
4. **Graceful degradation:** Each metric wrapped in try/except. If ^VVIX or ^VIX3M unavailable, shows "Data unavailable" and excludes from composite.

#### Phase 2 TODOs (not yet built)

- [ ] Layer 3: Credit (LQD/HYG spread), Yield Curve (^TNX/^IRX), MOVE, Dollar (UUP)
- [ ] Layer 4: SKEW, Protection Cost Proxy, Hedge Recommendation Engine
- [ ] Full Bayesian composite (replace simple point system)
- [ ] Full S&P 500 breadth (~500 constituents instead of 11 sector ETFs)
- [ ] Historical regime validation / backtesting

---

## Quick Reference: File Locations

| File | Purpose |
|------|---------|
| `strategy_config.py` | Strategy definitions (the "brain") |
| `daily_scan.py` | Production scanner (the "hands") |
| `backtester.py` | Research/testing UI |
| `pages/risk_dashboard.py` | Risk dashboard V1 (dispersion-based) |
| `pages/risk_dashboard_v2.py` | Risk dashboard V2 (multi-layer regime monitor) |
| `abs_return_dispersion.py` | S&P 500 absolute return dispersion (Nomura method) |
| `risk_dashboard_clean_sheet.md` | V2 design doc (panel debate format) |
| `docs/backtesting_logic.md` | Backtester architecture notes |
| `docs/screener_criteria.md` | Scanner/config documentation |
| `docs/portfolio_logic.md` | Portfolio simulation notes |
