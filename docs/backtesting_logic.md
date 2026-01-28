# Module: Strategy Backtester Logic

## 📄 File Reference
* **Primary Script:** `pages/backtester.py`
* **Key Dependencies:** `yfinance`, `plotly`, `pandas`
* **External Data:** `seasonal_ranks.csv` (Required for `load_seasonal_map`)

## 🎯 Core Purpose
The `backtester.py` script is a **Streamlit-based simulation engine**. It allows the user to define a trading strategy via UI widgets, tests it against historical data, and exports a JSON configuration (`strategy_config`) that mirrors the settings for the live screener.

---

## ⚙️ Logic Architecture (For AI Refactoring)

### 1. Data Ingestion & Cleaning (`download_universe_data`)
* **Source:** Uses `yf.download`.
* **Normalization:** The function `clean_ticker_df` is critical. It handles `yfinance`'s MultiIndex return format and flattens it.
* **Formatting:** All column names are capitalized (`Open`, `High`, `Close`) to ensure consistency across the repo.
    * *Constraint:* Do not change column naming conventions (e.g., to lowercase), as downstream calculations rely on `df['Close']`.

### 2. Indicator Calculation (`calculate_indicators`)
* **Batch Processing:** Indicators are vectorized and calculated upfront for the entire dataframe before signal logic begins.
* **Look-Forward Prevention:** The script calculates `NextOpen` (`df['Open'].shift(-1)`) to allow for T+1 entry calculations without looking ahead in the signal generation phase.

### 3. Signal Engine (`run_engine`)
This is the core loop. It builds a list of boolean Series called `conditions`.
* **Dynamic Construction:** Signals are not hardcoded classes. The script checks the `params` dictionary (derived from UI inputs) and appends conditions dynamically.
    * *Example:* If `params['use_52w']` is True, it appends the 52-week high logic to `conditions`.
* **Consolidation:** All conditions are combined via `&` (AND logic) to create `final_signal`.

### 4. Trade Simulation (Iterative Loop)
Once signals are identified, the script iterates through them to simulate execution.
* **The "Index Gap":**
    * `sig_idx`: The row where the signal occurred (Time T).
    * `actual_entry_idx`: The row where execution occurs (Time T+1, or T+x for pullbacks).
* **Entry Logic:** Handles complex order types including `Limit (Open +/- ATR)`, `Pullback to MA`, and `Stop/Limit` orders.
* **Exit Logic:** Checks daily `High`/`Low` against `stop_price` and `tgt_price` to determine if a trade was closed intraday.

### 5. Portfolio Constraints ("Greedy" Allocation)
The script simulates a portfolio by tracking `active_pos`.
* **Sorting:** Trades are sorted by `EntryDate` then `Ticker`.
* **Constraint:** If `max_daily_entries` or `max_total_positions` is reached, subsequent trades are rejected.
* **Bias Warning:** Because of the sort order, if multiple tickers signal on the same day, the system favors tickers coming first alphabetically.

---

## ⚠️ Critical Refactoring Rules (The "Do Not Break" List)

### 1. The Strategy Dictionary (`build_strategy_dict`)
* **Risk:** This function generates the JSON output used by the live `strategy_config.py`.
* **Workflow:** **Manual Copy-Paste.** The user manually copies the JSON output from the Streamlit UI and pastes it into `strategy_config.py`.
* **Rule:** You generally **cannot rename keys** in this dictionary (e.g., `sznl_thresh`, `vol_rank_logic`) without also refactoring the live screener logic that consumes them. The structure must remain consistent to facilitate this manual transfer.

### 2. YFinance Adjustments
* **Risk:** The script uses `auto_adjust=True` in `yf.download`.
* **Rule:** All price data is dividend/split adjusted. Do not mix unadjusted data sources, or signals will fire incorrectly on historical splits.

### 3. Streamlit Caching
* **Risk:** The functions `load_seasonal_map` and `download_universe_data` use `@st.cache_resource` and `@st.cache_data`.
* **Rule:** If modifying the input arguments to these functions, ensure the cache logic doesn't return stale data (e.g., if the user changes the "Backtest Start Date").

---

## 🔮 Future Improvements (AI Tasks)
* [ ] **Remove Alphabetical Bias:** Refactor the portfolio loop to randomize trade selection when `max_daily_entries` is exceeded, rather than prioritizing "A" tickers.
* [ ] **Vectorized Backtesting:** The current `run_engine` uses an iterative loop for trade management. Converting this to a fully vectorized approach (using `vectorbt` or similar) would significantly speed up processing.
* [ ] **Slippage Modeling:** Currently uses fixed `bps`. Implementing variable slippage based on volatility would increase accuracy.
