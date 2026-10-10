"""PM Weekly v1: paths, R2 boundary and claim vocabulary.

The PM Weekly is a read-only, market-only weekly brief with graded forecasts
(docs/claude_ref/pm_agent.md). It sits beside the blind Risk Agent and must
never become an input to it:

  * Everything the PM writes lives OUTSIDE the repo checkout, in PM_AGENT_HOME
    (default ~/.pm_agent), and in R2 under `pm_agent/`. The Risk Agent's R2
    allowlist denies `pm_agent/` by default-deny; tests/test_pm_agent_universe.py
    pins that, so a future allow rule there cannot quietly re-admit it.
  * v1 reads market data only. The systematic book (fills, exposure, ledger,
    sizing) is denied here as well; book reads are phase 2 and need the owner's
    wall decision first.
  * The PM may read the Risk Agent's PUBLISHED output (R2 risk_agent/today.json)
    but only in the publisher, after its own forecasts are validated, so its
    forecasts are never anchored on the Risk Agent's.

Agent-product module: the book and the Risk Agent must not import it.
"""
from __future__ import annotations

import os
from pathlib import Path

SCHEMA_VERSION = "pm_agent.v1"
STATE_SCHEMA = "pm_agent_state.v1"
R2_PREFIX = "pm_agent/"


def home() -> Path:
    """PM_AGENT_HOME, else ~/.pm_agent. Never inside the repo checkout."""
    raw = os.environ.get("PM_AGENT_HOME")
    return Path(raw).expanduser() if raw else Path.home() / ".pm_agent"


def cache_dir() -> Path:
    return home() / "cache"


def checks_root() -> Path:
    return home() / "checks"


def state_path() -> Path:
    return home() / "state.json"


def brief_path() -> Path:
    return home() / "brief.json"


def journal_path() -> Path:
    return home() / "journal.jsonl"


def scoreboard_path() -> Path:
    return home() / "scoreboard.json"


def today_path() -> Path:
    return home() / "today.json"


def receipt_dir() -> Path:
    return home() / "delivery_receipts"


# ---------------------------------------------------------------------------
# R2 boundary. Deny rules win; anything not allowed is denied.
# Guard: tests/test_pm_agent_universe.py
# ---------------------------------------------------------------------------
MARKET_KEYS: tuple[str, ...] = (
    "master_prices.parquet",
    "cboe_putcall.parquet",
    "earnings_calendar.parquet",
    "macro_release_history.parquet",
    "market_breadth.parquet",
    "options/iv_history.parquet",
    "shared/site_risk.json",          # the redacted shared dashboard export
)
ALLOWED_R2_PREFIXES: tuple[str, ...] = MARKET_KEYS + (
    "risk_agent/today.json",          # published readout, read only by the publisher
    R2_PREFIX,                        # our own journal / today / receipts
)
DENIED_R2_PREFIXES: tuple[str, ...] = (
    "live_fills", "backtest_", "dial_sleeve_paper", "event_sleeve", "exposure_state",
    "morning_orders", "pitch_", "posts_journal", "radar_recs", "seasonal_agent",
    "seasonal_ideas", "trade_console", "trend_sleeve", "ops/", "review_inbox",
    "discretionary_focus", "idea_check", "site/", "charts/", "bt_indicator_cache",
    "rd2_environment.json", "automation/", "migrations/", "operations/",
    "risk_agent/journal", "risk_agent/delivery_receipts",
)

# Names a check script may not mention: book objects (v1 is market-only) and the
# Risk Agent's local working files (its forecasts must not anchor ours).
FORBIDDEN_SOURCE_TOKENS: tuple[str, ...] = (
    "live_fills", "exposure_state", "morning_orders", "dial_sleeve_paper",
    "event_sleeve", "trend_sleeve", "backtest_trades", "backtest_daily_pnl",
    "strategy_config", "trade_console", "rd2_environment", "data/site_risk.json",
    "risk_agent_journal", "risk_agent_today", "risk_agent_decision",
    "risk_agent_state", "risk_agent_scoreboard", "risk_agent/today",
    "pitch_journal", "seasonal_agent", "radar_recs",
)


def r2_key_allowed(key: str) -> bool:
    if not isinstance(key, str):
        return False
    if any(key.startswith(d) for d in DENIED_R2_PREFIXES):
        return False
    return any(key.startswith(a) for a in ALLOWED_R2_PREFIXES)


# ---------------------------------------------------------------------------
# Claim vocabulary (v1). Both are required every week so N grows evenly.
# ---------------------------------------------------------------------------
CLAIMS: dict[str, dict] = {
    "spy_week_return": {"symbol": "SPY", "unit": "pct",
                        "label": "SPY close-to-close return, %",
                        "fields": ("p_up", "q10_pct", "q90_pct")},
    "vix_week_change": {"symbol": "^VIX", "unit": "points",
                        "label": "VIX close change, points",
                        "fields": ("p_up", "q10", "q90")},
}

# Symbols summarised in the weekly recap (all present in master_prices).
RECAP_SYMBOLS: tuple[str, ...] = (
    "SPY", "QQQ", "IWM", "DIA",
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
    "SMH", "KRE", "XBI", "ITB",
    "EFA", "EEM", "FXI", "EWJ",
    "TLT", "IEF", "HYG", "LQD", "UUP", "GLD", "SLV", "USO", "DBC",
    "BTC-USD",
)
VOL_SYMBOLS: tuple[str, ...] = ("^VIX", "^VIX3M", "^VVIX", "^SKEW", "^MOVE")
RATE_SYMBOLS: tuple[str, ...] = ("^IRX", "^FVX", "^TNX")
CLIMATOLOGY_YEARS = 10
