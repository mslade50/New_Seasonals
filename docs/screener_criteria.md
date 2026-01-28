# Module: Live Screener & Configuration Logic

## 📄 File Reference
* **Configuration:** `strategy_config.py` (The "Brain")
* **Automation Engine:** `daily_scan.py` (The "Hands")
* **Output Destination:** Google Sheets (`Trade_Signals_Log`)

## 🎯 Core Purpose
This module handles the transition from research to production. It runs daily, consumes the trading rules defined in `strategy_config.py`, checks them against live market data, and stages actionable orders to a private Google Sheet.

---

## ⚙️ Architecture & Data Flow

### 1. Configuration (`strategy_config.py`)
* **Workflow:** Strategies are manually copy-pasted from the Backtester JSON export into the `_STRATEGY_BOOK_RAW` list.
* **Global Risk Control:**
    * `ACCOUNT_VALUE`: Hardcoded at the top of the file (currently `$750,000`).
    * **Logic:** The script automatically converts the `risk_bps` (basis points) defined in each strategy into raw dollar risk based on this account value.
    * *AI Note:* To increase position sizing globally, edit `ACCOUNT_VALUE`. To change a specific strategy, edit `risk_bps`.

### 2. The Daily Scan (`daily_scan.py`)
This script runs automatically (via GitHub Actions or Local Scheduler).
* **Timing:** Designed to run near market close (approx. 3:00 PM and 4:05 PM ET) to capture "Signal Close" entries.
* **Data Source:** Downloads fresh data via `yfinance`.
* **Indicator Consistency:** The `calculate_indicators()` function here **must** functionally match the one in `backtester.py`. If you add an indicator to the Backtester, you MUST add it here too.

### 3. Order Staging Logic (The "Air Gap")
The script separates orders into two different Google Sheet tabs based on their urgency:

| Order Type | Destination Tab | Logic |
| :--- | :--- | :--- |
| **MOC (Signal Close)** | `moc_orders` | These need immediate attention before the market closes. |
| **Limits / MOO** | `Order_Staging` | These are for "Tomorrow Open" or "GTC" limits. |

---

## ⚠️ Critical Hidden Dependencies (The "Trap" List)

An AI refactoring this code must be extremely careful with the following:

### 1. Hardcoded Strategy Names in `daily_scan.py`
**🚨 HIGH RISK:** The scanner script contains **hardcoded logic** tied to specific strategy names.
* **Location:** Inside the `run_daily_scan()` loop.
* **The Trap:**
    ```python
    if strat['name'] == "Overbot Vol Spike":
        # Applies dynamic risk multipliers (Momentum, Day of Week, Volume)
    elif strat['name'] == "Weak Close Decent Sznls":
        # Applies conviction multipliers based on Seasonal Rank
    ```
* **Consequence:** If you rename "Overbot Vol Spike" in `strategy_config.py` to "Vol Spike 2.0", **the dynamic risk sizing logic will silently fail** and revert to 1.0x sizing.
* **Fix:** When renaming strategies, you must grep `daily_scan.py` for the old name and update it.

### 2. Exit Date Calculation
* **Logic:** The script calculates `Exit_Date` at the moment of scanning using `pandas.tseries.offsets.CustomBusinessDay`.
* **Dependence:** It relies on `USFederalHolidayCalendar`. If the script runs on a holiday (or data is missing), this calculation may drift.

### 3. Google Sheets Schema
* **Constraint:** The `save_moc_orders` and `save_staging_orders` functions expect a specific column order.
* **Risk:** Changing the column order in the Python script without clearing the Google Sheet first may result in mismatched data (e.g., "Price" landing in the "Quantity" column).

---

## 🔮 Future Improvements (AI Tasks)
* **[ ] Decouple Name Logic:** Refactor `daily_scan.py` to look for a specific flag in the config (e.g., `"dynamic_sizing": "vol_momentum"`) rather than matching the raw string name `"Overbot Vol Spike"`.
* **[ ] Description Standardization:** The current strategy descriptions are free-text. Refactor them into structured tags (e.g., `Type: MeanReversion | Timeframe: Swing | Setup: WeakClose`) so the email summary is easier to scan.
* **[ ] Unified Indicator Library:** Move `calculate_indicators` to a shared `utils.py` file imported by both the Backtester and the Daily Scan to guarantee 1:1 logic parity.
