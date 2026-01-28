# Module: Portfolio Simulation & Efficiency

## 📄 File Reference
* **Primary Script:** `pages/strat_backtester.py`
* **Dependency:** Imports `_STRATEGY_BOOK_RAW` from `strategy_config.py`.

## 🎯 Core Purpose
While `backtester.py` tests a single idea, `strat_backtester.py` tests the **Ensemble**. It simulates how all active strategies interact over time, specifically focusing on capital contention, exposure overlap, and the compounding effects of dynamic position sizing.

---

## ⚙️ Technical Architecture

### 1. Real-Time Mark-to-Market (MTM) Sizing
Unlike basic backtesters that use fixed capital, this engine simulates a live compounding account.
* **Logic:** `current_equity = starting_equity + realized_pnl + unrealized_pnl`
* **Impact:** Position sizes (`risk_bps`) are calculated based on the *current* account value at the moment of the signal. This accurately models drawdowns (sizing down) and winning streaks (sizing up).

### 2. The "Price Matrix" Optimization
To handle portfolio-level calculations without iterating through thousands of dataframes repeatedly, the script builds a unified **Price Matrix** (`build_price_matrix`).
* **Structure:** A single DataFrame where Index = Date, Columns = Tickers, Values = Close Price.
* **Benefit:** Allows for vectorized MTM calculations across the entire portfolio in milliseconds.

### 3. Execution Logic
The engine aggregates signals from all strategies into a single timeline.
* **Priority:** Signals are processed chronologically.
* **Conflict:** If multiple signals occur on the same day, they are processed in order. (Note: The current version does not enforce a global "Max Portfolio Positions" limit, only per-strategy limits).

---

## 📊 Key Analytical Metrics

### 1. Capital Efficiency Ratio
Calculated as: `(% of Total PnL Contribution) / (% of Total Risk Contribution)`
* **> 1.0:** The strategy punches above its weight (generating more profit than the risk it consumes).
* **< 1.0:** The strategy is capital inefficient (taking up room/risk but delivering sub-par returns).
* **AI Use Case:** Use this metric to suggest `risk_bps` adjustments in `strategy_config.py`.

### 2. Signal Density (Clustering)
Analyzes performance based on how many signals triggered on the same day.
* **Isolated:** Signals occurring alone (low correlation day).
* **Clustered:** Signals occurring with >10 others (high correlation/market event).
* **Insight:** Helps determine if the portfolio is truly diversified or just "long beta" on busy days.

---

## ⚠️ Refactoring & Maintenance Notes

### 1. Data Integrity
* **Risk:** The script attempts to download *all* tickers in `strategy_config.py` at once.
* **Constraint:** If the universe grows too large (>500 tickers), the `download_historical_data` function may hit `yfinance` rate limits or timeouts. Future refactors may need batching or local caching (SQL/Parquet).

### 2. Strategy Config Dependency
* **Risk:** The script imports `_STRATEGY_BOOK_RAW` directly.
* **Constraint:** If `strategy_config.py` has a syntax error or a broken import, this dashboard will crash immediately.

### 3. "Greedy" Execution
* **Logic:** The loop processes signals as they arrive. It does not "optimize" allocation if cash is low; it simply takes the first trade available.
* **Reality Check:** This mimics a trader taking signals as they see them, but may differ from a system that ranks signals by conviction before execution.
