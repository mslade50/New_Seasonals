# New Seasonals: Quantitative Research & Trading Engine

## 🤖 System Context for AI Assistants
**Repository Purpose:**
This repository is a private, internal-only quantitative trading system used by a family office prop trader. It integrates historical analysis, seasonal trend tracking, and automated daily signal generation.

**Operational Logic:**
1.  **Non-HFT:** This is not a high-frequency trading system. It operates on daily closing data.
2.  **Data Source:** Primary historical data comes from `yfinance`.
3.  **Execution Boundary:** This repository handles **Signal Generation Only**. It does **NOT** handle trade execution.

---

## 🛡️ Safety & Execution Architecture
**CRITICAL:** To ensure operational safety and prevent accidental algorithmic trading errors, **Order Execution logic is strictly excluded from this repository.**

* **Repository Scope:** Ends at **Signal Staging**. The code here identifies a trade and pushes it to a private Google Sheet.
* **Local Execution:** The actual Python scripts that connect to broker APIs and place live orders reside **only on the local machine**. They are never committed to version control.
* **The "Air Gap":** This separation ensures that no cloud-based automation (like GitHub Actions) or remote code change can inadvertently trigger a financial transaction.

---

## 🔄 Data Flow & Architecture

The system is designed as a unidirectional pipeline moving from **Research** $\to$ **Configuration** $\to$ **Live Scanning** $\to$ **Staging**.

### 1. Alpha Research (Signal Identification)
*Objective: Find and validate trading edges using historical data.*
* **`pages/backtester.py`**: [See Docs](docs/backtesting_logic.md) - The primary tool for testing single-ticker strategies.
    * *Input:* User-defined parameters (tickers, dates).
    * *Output:* Returns performance metrics and trade logs.
* **`pages/heatmaps.py`** & **`pages/correlation_heatmaps.py`**: Visualization tools to spot sector rotation and asset correlation changes.

### 2. Market Context (The "Environment")
*Objective: Filter signals based on broader market conditions.*
* **`pages/seasonal_sigs.py`**: Checks if a ticker is entering a historically strong/weak window.
    * *Dependency:* Relies on pre-computed static files (`seasonal_ranks.csv`).
* **`pages/sector_trends.py`**: Analyzes relative strength of sectors to avoid trading against the tide.

### 3. Portfolio Simulation
*Objective: Test how multiple signals perform together.*
* **`pages/strat_backtester.py`**: Aggregates distinct signals (validated in the Alpha phase) into a portfolio simulation to test for overlap, drawdown clusters, and total return.

### 4. Live Production (The Daily Loop)
*Objective: Generate actionable orders for tomorrow's open.*
* **`strategy_config.py`**: **CRITICAL.** This file contains the "Source of Truth" for all active trading rules. It defines the universe of tickers and the specific logic for entry/exit.
* **`daily_scan.py`**: The automation engine.
    * *Action:* Runs daily (via GitHub Actions or locally).
    * *Logic:* Imports rules from `strategy_config.py` $\to$ Downloads fresh data $\to$ Checks conditions.
    * *Output:* Valid orders are pushed via API to the **Private Google Sheet**.
* **`pages/screener.py`**: A front-end interface to manually verify the output of `daily_scan.py`.

---

## 🗃️ Static Data Files (Pre-Computed)
*To optimize speed, specific heavy-lift calculations are done annually and stored as CSVs.*

| File Name | Description | Update Cycle |
| :--- | :--- | :--- |
| `seasonal_ranks.csv` | Core database of seasonal strength/weakness windows. | **Annual (Jan 1)** |
| `sznl_ranks.csv` | Supplementary ranking data for specific setups. | **Annual (Jan 1)** |
| `market_dates.csv` | A master calendar of trading days to handle holidays/weekends. | **Annual** |

---

## 🛠️ Developer Notes (Maintenance)
* **Google Sheets Connection:** The `daily_scan.py` script requires valid `gspread` credentials. If orders fail to appear, check the JSON key expiration.
* **Streamlit Structure:** This is a multi-page app. The `pages/` directory must remain flat. Do not create sub-folders inside `pages/` as it breaks the sidebar navigation.
* **Documentation:** Detailed logic for complex modules can be found in the `docs/` directory:
    * [Backtester Logic](docs/backtesting_logic.md)
    * [Screener & Config Logic](docs/screener_criteria.md)
