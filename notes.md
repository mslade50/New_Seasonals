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

### New File: `pages/risk_dashboard_v2.py` (1,004 → 1,757 → 2,228 → 2,414 lines)

**Purpose:** Standalone market risk monitor. Completely independent from trading strategies — no imports from `strategy_config.py`, `strat_backtester.py`, `daily_scan.py`, or `indicators.py`.

**Design doc:** `risk_dashboard_clean_sheet.md` (project root)

#### Architecture: 3-Layer System

**Executive Summary** (rebuilt — see Executive Summary Rebuild session below)
- Signal-based three-question framework with 8 causal signals + risk dial
- Legacy point system preserved in collapsed expander

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
| 2B. Absorption Ratio | PCA eigenvalue₁/Σeigenvalues on 63d sector returns. **Display-only** — removed from composite scoring (under review). Red reference line at 0.40. AR measures how much of sector variance is explained by one factor; low AR (<0.4) = sectors independent, historically followed by below-avg returns (Minsky: stability breeds instability). | *Not scored* | *Not scored* |
| 2C. Dispersion | Cross-sectional σ of 21d sector returns + avg pairwise corr | High disp | High disp + high corr |
| 2D. Hurst | DFA on SPY returns, **126d rolling**, box sizes [8,16,32,48,63]. **Smoothed**: 11d rolling median → 15d EMA (raw is too choppy). **Empirical percentile bands** (P20/P80 of smoothed history). 5d ΔH computed from smoothed series. | H > 80th pctile | H > 95th pctile |
| 2E. Days Since Correction | Trading days since last 5% and 10% peak-to-trough drawdown in SPY. Both shown in one styled box. | 5% streak > 80th pctile | 5% streak > 95th pctile |

**Chart defaults:** HAR-RV and VRP charts show last 1 year by default. Double-click to zoom out to full history.

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

#### Phase 2 TODOs → DONE

- [x] Layer 3: Credit (LQD/HYG spread), Yield Curve (^TNX/^IRX), MOVE, Dollar (UUP)
- [x] Layer 4: SKEW, Protection Cost Proxy, Hedge Recommendation Engine
- [x] Executive summary rebuild: signal-based three-question framework
- [ ] Signal event study: backtest each of the 8 signals to calibrate hit rates
- [x] Full S&P 500 breadth (~500 constituents instead of 11 sector ETFs) — done in `84175dd`
- [ ] Historical regime validation / backtesting
- [ ] FRED data source for MOVE (more reliable than yfinance)

---

## Session: 2026-02-15 - Risk Dashboard V2 (Phase 2)

### Updated File: `pages/risk_dashboard_v2.py` (1,139 → 1,757 lines, +648 lines)

**What changed:** Added Layers 3, 4, updated 2E complacency counters, wired all new metrics into Layer 0 composite.

#### Layer 2E Update: Complacency Counters

Previously showed days since 5% and 10% SPX drawdown. Now two primary counters with compound scoring:

| Counter | Method | Threshold |
|---------|--------|-----------|
| Days since 5% SPX drawdown | Trailing high drawdown on closing basis | Breach = drawdown ≤ -5% |
| Days since VIX > 28 | Simple level check | VIX close ≥ 28 |

- 10% drawdown counter kept for context display (not scored)
- **Compound scoring:** Either > 80th pctile = alert (+1). BOTH > 80th = alarm (+2).
- Sawtooth time series charts for each counter

#### Layer 3: Cross-Asset Plumbing (4-column layout)

| Metric | Data | Method | Alert | Alarm |
|--------|------|--------|-------|-------|
| 3A. Credit Spreads | LQD, HYG, IEF | -(ETF/IEF) ratio, 63d rolling z-score. Inverted so higher = wider spreads. | IG or HY z > 1.0 | Both > 1.5 |
| 3B. Yield Curve | ^TNX, ^IRX | 10Y - 3M spread. 21d change z-scored against 252d history. | Inverted OR z < -1.5 | Inverted AND z < -2.0 |
| 3C. MOVE | ^MOVE | Raw level, bands at 80/120/150. Graceful fallback if unavailable. | > 120 | > 150 |
| 3D. Dollar | UUP | 21d % change as DXY proxy. Bars colored by magnitude. | |chg| > 3% | |chg| > 5% |

**Data notes:**
- `^MOVE` may not be available on yfinance — shows placeholder text, never crashes
- `UUP` used as DXY proxy (DX-Y.NYB unreliable on yfinance)
- All downloads wrapped in try/except with graceful degradation

#### Layer 4: Tail Risk & Cost of Protection

Collapsed by default (`st.expander`), auto-expands when Layer 0 regime ≠ Normal.

| Metric | Data | Method |
|--------|------|--------|
| 4A. SKEW | ^SKEW | Raw level + disorderly stress detection (SKEW falling >3pts/5d while VIX rising >3pts/5d) |
| 4B. Protection Cost | VIX3M × (SKEW/130) | Percentile-ranked over 1260d trailing window. Plotly gauge (green < 20th, red > 85th). |
| 4C. Hedge Rec | Decision tree | regime × protection_percentile → sizing guidance, collar vs puts vs exposure reduction |

**Hedge recommendation logic:**
- Protection < 20th pctile → "Historically cheap, allocate 1-2% NAV to puts"
- Caution/Stress + protection < 60th → "Fairly priced, 0.5-1% NAV or reduce to 0.75x"
- Stress/Crisis + protection 60-85th → "Expensive, prefer collars or reduce to 0.50x"
- Protection > 85th → "Expensive, reduce exposure directly to 0.50x or lower"
- Otherwise → "No action needed"

#### Layer 0: Updated Composite Scoring

All new metrics now feed into `score_alerts()`. Full point system:

| Layer | Metric | Alert (+1) | Alarm (+2) |
|-------|--------|------------|------------|
| 1A | HAR-RV | RV_1d > 2x RV_22d | RV_22d > 75th & rising |
| 1B | VRP | < 25th pctile | Negative |
| 1C | VIX Term Str | > 0.95 | > 1.0 (backwardation) |
| 1D | VVIX | > 100 | > 120 |
| 2A | Breadth | < 60% w/ SPY near high | < 40% |
| 2B | AR | *Not scored* | *Not scored* |
| 2C | Dispersion | High dispersion | High disp + high corr |
| 2D | Hurst | > 80th pctile | > 95th pctile |
| 2E | Complacency | Either counter > 80th | Both counters > 80th |
| 3A | Credit | IG or HY z > 1.0 | Both > 1.5 |
| 3B | Yield Curve | Inverted OR z < -1.5 | Inverted AND z < -2.0 |
| 3C | MOVE | > 120 | > 150 |
| 3D | Dollar | |21d chg| > 3% | |21d chg| > 5% |

**New tickers downloaded:**
- `CROSS_ASSET_TICKERS`: LQD, HYG, IEF, UUP, ^MOVE, ^TNX, ^IRX
- `TAIL_RISK_TICKERS`: ^SKEW

#### New Functions Added

| Function | Purpose |
|----------|---------|
| `compute_days_since_vix_spike()` | Days since VIX closed above threshold |
| `compute_credit_spread_proxy()` | ETF price ratio z-scores for IG/HY spreads |
| `compute_yield_curve()` | 10Y-3M spread + 21d change z-score |
| `compute_dollar_momentum()` | UUP 21d rate of change |
| `compute_protection_cost()` | VIX3M × SKEW/130 with trailing percentile |
| `generate_hedge_recommendation()` | Decision tree → (rec, detail, color) |
| `chart_credit_spreads()` | Overlaid IG/HY z-score chart |
| `chart_yield_curve()` | Spread with inversion shading |
| `chart_move()` | MOVE with 80/120/150 bands |
| `chart_dollar()` | Bar chart with colored bars by magnitude |
| `chart_days_since_sawtooth()` | Generic sawtooth chart for days-since counters |
| `download_cross_asset_data()` | Cached download for Layer 3 tickers |
| `download_tail_risk_data()` | Cached download for Layer 4 tickers |

---

## Session: 2026-02-15 - Risk Dashboard V2 (Executive Summary Rebuild — Final Architecture)

### Updated File: `pages/risk_dashboard_v2.py` (2,228 → 2,414 lines, +602 / -416)

**What changed:** Complete replacement of the executive summary. Removed situation board, narrative engine, and old fragility score. Replaced with signal-based three-question framework organized around causal questions. All computation functions and Layer 1-4 detail charts unchanged.

#### Design Philosophy

The dashboard measures **susceptibility to regime change**, not prediction. Analogy: can't predict earthquakes but can measure soil liquefaction risk. Measures "forest dryness" — how vulnerable the market microstructure is to disruption of the user's mean reversion systems.

#### New Page Layout (top section)

```
┌──────────────────────────────────────────────────┐
│  PRICE CONTEXT BANNER                            │
│  SPY: $598.42  +18.2% 12mo | +6.1% vs 200d |   │
│  -1.2% from high    "Strong uptrend, moderately  │
│                       extended"                   │
├──────────────────────────────────────────────────┤
│  Since last session: 🔴 Vol Compression activated │
├───────────────────────────────────┬──────────────┤
│  THREE QUESTIONS                  │  RISK DIAL   │
│                                   │  [gauge 0-100]│
│  💧 Is liquidity real?   CLEAR   │  "2 of 8     │
│    🟢 Vol Suppression             │   signals    │
│    🟢 VRP Compression             │   active"    │
│                                   │              │
│  👥 Is everyone on the same side? │              │
│     WATCH                         │              │
│    🟢 Breadth Divergence          │              │
│    🔴 Extended Calm               │              │
│      Days since 5%: 187 (82nd)...│              │
│    🔴 Vol Compression             │              │
│      RV below median 73 days...  │              │
│                                   │              │
│  🔗 Are correlations stable?     │              │
│     CLEAR                         │              │
│    🟢 Credit-Equity Divergence    │              │
│    🟢 Rates-Equity Vol Gap        │              │
│    🟢 Vol Uncertainty             │              │
├───────────────────────────────────┴──────────────┤
│  IF CONDITIONS DETERIORATE: (only when 2+ active)│
│  [Vol Compression] [Calm Streak]  [Potential     │
│   73d below median  187d since 5%  Unwind ~5-9%] │
├──────────────────────────────────────────────────┤
│  ▸ Legacy scoring detail (collapsed expander)    │
├──────────────────────────────────────────────────┤
│  Layer 1 / Layer 2 detail charts (unchanged)     │
```

#### The 8 Causal Signals

| # | Signal | Question | Trigger | Why It Matters |
|---|--------|----------|---------|----------------|
| 1A | Vol Suppression | Liquidity | AR < 25th pctile AND RV_22d < 35th | Index vol artificially suppressed by systematic selling |
| 1B | VRP Compression | Liquidity | VRP negative OR VRP < 15th pctile | Market not compensating for risk |
| 2A | Breadth Divergence | Positioning | SPY near 52w high AND < 55% sectors > 200d | Index held up by few names |
| 2B | Extended Calm | Positioning | Both complacency counters > 70th OR either > 85th | Positioning accumulated in one direction |
| 2C | Vol Compression | Positioning | > 60 consecutive days RV_22d below expanding median | Participants adapted to low vol |
| 3A | Credit-Equity Divergence | Correlations | HY z > 0.75 AND SPY 21d return > -2% | Bond market repricing risk equity hasn't |
| 3B | Rates-Equity Vol Gap | Correlations | MOVE > 70th pctile AND VIX < 40th pctile | Rates vol not yet transmitted to equity |
| 3C | Vol Uncertainty | Correlations | VVIX/VIX ratio > 80th pctile (or > 7.5 absolute) | Market doesn't trust current vol level |

#### Risk Dial Formula

```
fragility = min(100, (active_signals / 8) × 80 × regime_multiplier)
```

**Regime multiplier** (0.6–1.8x) based on price context:
- 12mo return > 25% → +0.25, > 15% → +0.10, < -5% → -0.15
- Extension > 10% → +0.25, > 5% → +0.10, < -2% → -0.15
- Near highs (DD > -2%) → +0.10, deep drawdown (DD < -10%) → -0.20

#### Signal Persistence

Daily change tracking via `data/risk_dashboard_signal_state.json`. On each load:
1. Load previous state (only if from a different calendar date)
2. Compare current vs previous signals
3. Display activations/deactivations
4. Save current state for next session

#### Functions Removed

| Function | Replaced By |
|----------|-------------|
| `generate_narrative()` | Signal detail strings in `compute_condition_signals()` |
| `build_situation_board()` | `render_three_questions()` |
| Old `compute_fragility_score()` | New signal-count-based `compute_fragility_score()` |
| Old `build_risk_dial()` | New score-color-based `build_risk_dial()` |

#### Functions Added

| Function | Purpose |
|----------|---------|
| `compute_price_context()` | SPY price, 12mo return, 200d extension, drawdown, regime label |
| `compute_regime_multiplier()` | Price context → 0.6-1.8x amplifier for fragility score |
| `compute_condition_signals()` | 8 binary signals across 3 causal questions |
| `render_price_context()` | Price context banner HTML |
| `render_three_questions()` | Signal board with CLEAR/WATCH/WARNING badges |
| `compute_fragility_score()` | Signal count × regime multiplier → 0-100 |
| `build_risk_dial()` | Plotly gauge with score-based coloring |
| `load_previous_signal_state()` | Read JSON cache from previous session |
| `save_current_signal_state()` | Write JSON cache for next session |
| `compute_changes()` | Diff current vs previous signal states |

#### New Metric Computations in main()

| Metric | Variable | Purpose |
|--------|----------|---------|
| VIX current + percentile | `cur_vix`, `cur_vix_pctile` | Rates-Equity Vol Gap signal |
| SPY 21d return | `spy_21d_return` | Credit-Equity Divergence signal |
| VVIX/VIX ratio percentile | `cur_vvix_vix_ratio_pctile` | Vol Uncertainty signal |
| AR percentile | `cur_ar_pctile` | Vol Suppression signal |
| MOVE percentile | `cur_move_pctile` | Rates-Equity Vol Gap signal |

---

## Session: 2026-02-18 - Sync Daily Scan / Backtester / Portfolio Report + Cleanup

### Sync Fixes

**Problem:** `strat_backtester.py` and `daily_portfolio_report.py` had drifted from `daily_scan.py`, causing mismatched signals and position sizes.

**Changes to `pages/strat_backtester.py`:**
1. **Vol Spike Case 3:** Added missing 4th branch to ATH/52w overlay. daily_scan has 4 cases; backtester had 3. Missing: "52w high in L5 but not today → LOC only." Now matches daily_scan lines 1555-1578.
2. **Range ATR filter:** Added `use_range_atr_filter` support in `get_historical_mask()` (supports `>`, `<`, `Between` logic). Latent — no active strategy uses it yet.
3. **Gap filter column:** Changed from hardcoded `GapCount` to `GapCount_{lookback}` with fallback. Latent — no active strategy uses gap filter yet.
4. **Default equity:** UI default changed from $150k to `ACCOUNT_VALUE` (from strategy_config).

**Changes to `daily_portfolio_report.py`:**
1. **Portfolio value:** Replaced three hardcoded `$450,000` references with `ACCOUNT_VALUE` from strategy_config. Single source of truth for sizing across daily_scan, backtester, and health report.

### Deleted Files

| File | Reason |
|------|--------|
| `daily_scan_2.py` | Stale copy of daily_scan.py — GSheets commented out, missing 4-case Vol Spike, error tracking, bracket exits. Nothing imported it. |
| `pages/risk_dashboard.py` | V1 dispersion dashboard fully superseded by V2's multi-layer signal framework. Nothing imported from it. |
| `.github/workflows/daily_scan_2.yml` | Workflow for deleted daily_scan_2.py. |

**Net deletion:** ~3,086 lines of dead code removed.

---

## Quick Reference: File Locations

| File | Purpose |
|------|---------|
| `strategy_config.py` | Strategy definitions + ACCOUNT_VALUE (the "brain") |
| `daily_scan.py` | Production scanner (the "hands") |
| `daily_portfolio_report.py` | Daily portfolio health report (imports from strat_backtester) |
| `backtester.py` | Research/testing UI |
| `pages/strat_backtester.py` | Extended backtester (must stay in sync with daily_scan.py) |
| `pages/risk_dashboard_v2.py` | Risk dashboard (multi-layer regime monitor) |
| `abs_return_dispersion.py` | S&P 500 absolute return dispersion (Nomura method) |
| `risk_dashboard_clean_sheet.md` | V2 design doc (panel debate format) |
| `docs/backtesting_logic.md` | Backtester architecture notes |
| `docs/screener_criteria.md` | Scanner/config documentation |
| `docs/portfolio_logic.md` | Portfolio simulation notes |
