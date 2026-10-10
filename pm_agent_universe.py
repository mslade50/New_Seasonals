"""PM layer: paths, R2 boundary and claim vocabulary.

The PM layer is read-only: a weekly brief with graded forecasts plus a daily
code-only check-in on the systematic book (docs/claude_ref/pm_agent.md). It
sits beside the blind Risk Agent and must never become an input to it:

  * Everything the PM writes lives OUTSIDE the repo checkout, in PM_AGENT_HOME
    (default ~/.pm_agent), and in R2 under `pm_agent/`. The Risk Agent's R2
    allowlist denies `pm_agent/` by default-deny; tests/test_pm_agent_universe.py
    pins that, so a future allow rule there cannot quietly re-admit it.
  * It reads market data and the book's published read surface (fills, broker
    snapshots, sleeve state, job receipts, the ledger). It writes nothing but
    its own namespace and never touches order paths.
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
# Book read surface (phase 2, owner 2026-10-09: one Windows user, no wall).
# Read only; the PM never writes any of these.
BOOK_KEYS: tuple[str, ...] = (
    "live_fills.parquet",              # canonical broker executions (harvest_fills)
    "live_fills_status.json",
    "ops/sleeve_runtime_status.json",
    "ops/expected_exit_status.json",
    "ops/olv_capacity/latest.json",    # daily broker snapshot: NLV, positions, orders
    "exposure_state.json",
    "event_sleeve_state.json",
    "trend_sleeve_state.json",
    "dial_sleeve_paper.json",
)
BOOK_PREFIXES: tuple[str, ...] = (
    "ops/olv_capacity/2",              # dated snapshots ops/olv_capacity/YYYY-MM-DD.json
    "automation/receipts/v1/",         # job health (latest.json per job per day)
    "pitch_delivery_receipts/",
    "seasonal_agent_delivery_receipts/",
    "risk_agent/delivery_receipts/",   # delivery facts only, no forecasts
)
LEDGER_NAMES: tuple[str, ...] = ("backtest_trades_full.parquet", "backtest_daily_pnl.parquet")
LEDGER_PREFIX = "site/builds/"         # site/builds/<run>/<ledger name>, newest run wins

ALLOWED_R2_PREFIXES: tuple[str, ...] = MARKET_KEYS + BOOK_KEYS + BOOK_PREFIXES + (
    "risk_agent/today.json",          # published readout, read only by the publisher
    R2_PREFIX,                        # our own journal / today / receipts
)
DENIED_R2_PREFIXES: tuple[str, ...] = (
    "morning_orders", "pitch_journal", "pitch_today", "posts_journal", "radar_recs",
    "seasonal_agent_journal", "seasonal_ideas", "trade_console", "review_inbox",
    "discretionary_focus", "idea_check", "charts/", "bt_indicator_cache",
    "rd2_environment.json", "migrations/", "operations/",
    "ops/tagged_inventory_seed.json",  # reviewed seed, sensitive
    "risk_agent/journal",              # its forecasts must not anchor ours
)

# Names a check script may not mention: the Risk Agent's working files (its
# forecasts must not anchor ours) and objects outside the read surface.
FORBIDDEN_SOURCE_TOKENS: tuple[str, ...] = (
    "risk_agent_journal", "risk_agent_today", "risk_agent_decision",
    "risk_agent_state", "risk_agent_scoreboard", "risk_agent/today",
    "risk_agent/journal", "data/risk_agent/", "trade_console", "morning_orders",
    "order_staging", "eq_order_entry", "tagged_inventory_seed",
)


def r2_key_allowed(key: str) -> bool:
    if not isinstance(key, str) or ".." in key:
        return False
    if any(key.startswith(d) for d in DENIED_R2_PREFIXES):
        return False
    if key.startswith(LEDGER_PREFIX):
        parts = key.split("/")
        return len(parts) == 4 and parts[3] in LEDGER_NAMES
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
