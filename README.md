# New Seasonals: Quantitative Research & Trading Engine

## 🤖 System Context for AI Assistants
**Repository Purpose:**
This repository is a private, internal-only quantitative trading system used by a family office prop trader. It integrates historical analysis, seasonal trend tracking, and automated daily signal generation.

**Operational Logic:**
1.  **Non-HFT:** This is not a high-frequency trading system. It operates on daily closing data.
2.  **Data Source:** Primary historical data comes from `yfinance`.
3.  **Execution Boundary:** The repository generates/stages signals and also contains the authenticated private-site command UI, Pages command policy, and Cloudflare execution broker. The final IBKR adapter remains local and out of repo. This system is therefore **live-execution capable when every gate is armed**.

---

## 🛡️ Safety & Execution Architecture
**CRITICAL:** This is not an air-gapped, signal-only repository. Treat changes under `site/assets/execution.js`, `site/assets/options.js`, `functions/`, `execution-broker/`, and `.github/workflows/deploy_broker.yml` as live-money infrastructure.

* **Signal staging:** `daily_scan.py` identifies trades and writes the private staging sheets.
* **Private command path:** Authenticated browser commands cross a Pages Function and a standalone Cloudflare broker to the local execution agent.
* **Local broker adapter:** The Python code that connects to IBKR and transmits remains only on the trading machine, but the cloud path can request that action when armed.
* **Fail-closed gates:** Live delivery requires a fresh online book, explicit `dry_run:false`, a dedicated `COMMAND_SECRET`, matching Pages/broker type and account allowlists, and the agent's own live gates. Risk-increasing commands are currently rejected at both server layers until the broker has an atomic aggregate-risk reservation; the existing per-command cap is not sufficient. Deploying the broker workflow atomically resets the Worker live switches to off and also disarms Pages.
* **Operational rule:** Never infer dry-run from this README or a stale banner. Verify the current Pages, broker, and agent configuration before testing any command.

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
* **`pages/macro_seasonality.py`**: Macro seasonality dashboard — MA extension ranks plus per-ticker seasonal paths sorted by largest seasonal deviation across short windows.

### 3. Portfolio Simulation
*Objective: Test how multiple signals perform together.*
### 3. Portfolio Simulation
* **`pages/strat_backtester.py`**: [See Docs](docs/portfolio_logic.md) - Simulates the entire strategy book running simultaneously.
    * *Key Features:* Real-time MTM equity sizing, Capital Efficiency analysis, and Signal Density breakdowns.
      
### 4. Live Production (The Daily Loop)
*Objective: Generate actionable orders for tomorrow's open.*
* **`strategy_config.py`**: **CRITICAL.** This file contains the "Source of Truth" for all active trading rules. It defines the universe of tickers and the specific logic for entry/exit.
* **`daily_scan.py`**: The automation engine.
    * *Action:* Runs daily (via GitHub Actions or locally).
    * *Logic:* Imports rules from `strategy_config.py` $\to$ Downloads fresh data $\to$ Checks conditions.
    * *Output:* Valid orders are pushed via API to the **Private Google Sheet**.
* **Optional execution bridge:** The private Execution/Options pages can relay a validated preview or, only when all independent gates are armed, a live command to the local IBKR agent. See [the go-live runbook](docs/site_execution_golive.md).

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
* **Workspace Hygiene:** Use one Git worktree per task, put disposable output in `artifacts/`, and run the baseline checker described in [Workspace Hygiene](docs/workspace_hygiene.md).
* **Documentation:** Detailed logic for complex modules can be found in the `docs/` directory:
    * [Backtester Logic](docs/backtesting_logic.md)
    * [Scanner & Config Logic](docs/screener_criteria.md)
