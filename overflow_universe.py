"""Dynamic overflow trading universe — loader + screen configuration.

Single source of truth for:
  - which extended (non-liquid) tickers are eligible to trade this week
    (``data/overflow_universe.parquet`` — rebuilt weekly by
    ``scripts/build_overflow_universe.py``)
  - the screen thresholds (ADDV / ATR% / price / history / freshness / quality)
  - the per-strategy ADDV floors and the ADV participation cap

Design contract: import-safe everywhere.
  * Does NOT import ``strategy_config`` (avoids a circular import — strategy
    modules import this, not the other way around).
  * Degrades gracefully to a caller-supplied ``fallback`` when the parquet is
    absent, so existing scans keep working before the universe is bootstrapped.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import pandas as pd

_STALE_AFTER_SECONDS = 18 * 3600  # match data_provider / earnings refresh cadence


def _maybe_pull_from_r2(path: str, key: str) -> None:
    """Best-effort pull of `key` from R2 to `path` when missing/stale. Silent
    no-op without cache_io/creds. Lets a fresh machine (with R2 creds in .env)
    read the universe without rebuilding."""
    try:
        from cache_io import download_to_local
    except Exception:
        return
    need = (not os.path.exists(path)) or (
        time.time() - os.path.getmtime(path) > _STALE_AFTER_SECONDS
    )
    if need:
        try:
            download_to_local(key, path)
        except Exception:
            pass

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OVERFLOW_UNIVERSE_PATH = os.path.join(_THIS_DIR, "data", "overflow_universe.parquet")
OVERFLOW_UNIVERSE_R2_KEY = "overflow_universe.parquet"
SYMBOL_MASTER_PATH = os.path.join(_THIS_DIR, "data", "symbol_master.parquet")
SYMBOL_MASTER_R2_KEY = "symbol_master.parquet"

# Isolated staging price cache for the NEW overflow candidate tickers (those not
# already in production master_prices). Kept separate so the live daily price
# pipeline (update_master_prices / portfolio / earnings) is untouched while the
# universe is being built + backtested. Screened + backtested from here; merged
# into master_prices only at promote time.
OVERFLOW_PRICES_PATH = os.path.join(_THIS_DIR, "data", "overflow_prices.parquet")
OVERFLOW_PRICES_R2_KEY = "overflow_prices.parquet"

# --------------------------------------------------------------------------
# ACTIVATION GATE.
# The dynamic universe is built + uploaded to R2 (so it can be backtested /
# inspected) WITHOUT being traded live. Live consumers (daily_scan,
# daily_portfolio_report) only switch to it once OVERFLOW_UNIVERSE_ACTIVE is
# truthy in the environment. Until then they keep the legacy static tier even
# if the parquet is present. Backtests / build tooling read the parquet with
# respect_active=False to bypass the gate.
# --------------------------------------------------------------------------
OVERFLOW_UNIVERSE_ACTIVE_ENV = "OVERFLOW_UNIVERSE_ACTIVE"


def is_active() -> bool:
    """True only when OVERFLOW_UNIVERSE_ACTIVE is explicitly enabled (default OFF)."""
    return os.environ.get(OVERFLOW_UNIVERSE_ACTIVE_ENV, "0").strip().lower() in (
        "1", "true", "yes", "on",
    )

# --------------------------------------------------------------------------
# Screen thresholds (Layer C). All tunable; mirrored into the config JSON the
# build script writes alongside the parquet so the screen is auditable.
# --------------------------------------------------------------------------
MIN_ADDV_BASE = 3_000_000.0   # 63d avg dollar volume, USD — loosest liquidity gate
MIN_ATR_PCT = 0.5             # 63d avg ATR% floor (R-T4: low — cut only dead names)
MIN_PRICE = 3.0              # last close floor (penny-stock guard)
MIN_BARS = 252               # need ~1y for 252d ranks / 52w high
FRESHNESS_TD = 5             # last bar must be within N trading days of the build
MAX_NAN_FRAC = 0.02          # R-T7 quality gate: drop names with >2% NaN closes in window
ADDV_WINDOW = 63
ATR_WINDOW = 14

# --------------------------------------------------------------------------
# Per-strategy ADDV floors (R-T3 — capacity / shortability).
# OVS shorts thin small-caps (borrow + slippage) → require deep liquidity.
# 52wh Breakout chases the open → mid. Patient GTC-limit strategies (OLV,
# LT Trend ST OS, St OS Sznl) tolerate the base floor.
# --------------------------------------------------------------------------
PER_STRATEGY_MIN_ADDV = {
    "Overbot Vol Spike": 10_000_000.0,
    "52wh Breakout": 5_000_000.0,
    "Oversold Low Volume": 3_000_000.0,
    "LT Trend ST OS": 3_000_000.0,
    "St OS Sznl": 3_000_000.0,
}

# Max single-position notional as a fraction of 63d ADDV (R-T3 sizing guard).
ADV_PARTICIPATION_CAP = 0.02


def min_addv_for(strategy_name: str) -> float:
    """Per-strategy ADDV floor (falls back to the base floor)."""
    return PER_STRATEGY_MIN_ADDV.get(strategy_name, MIN_ADDV_BASE)


def _norm(t: str) -> str:
    return str(t).upper().strip().replace(".", "-")


def _read_universe_frame(path: str = OVERFLOW_UNIVERSE_PATH) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as e:  # noqa: BLE001 — never let a bad cache crash the scan
        print(f"[overflow_universe] failed to read {path}: {e}")
        return None
    if df is None or df.empty or "ticker" not in df.columns:
        return None
    return df


def load_overflow_universe(
    fallback: Optional[list] = None,
    path: str = OVERFLOW_UNIVERSE_PATH,
    respect_active: bool = True,
) -> list:
    """Return the active overflow ticker list.

    Reads ``overflow_universe.parquet`` (column ``ticker``). When the parquet is
    missing / empty / unreadable, returns ``fallback`` (or ``[]`` if None) so
    callers preserve their pre-bootstrap behavior. Tickers are normalized to
    uppercase with ``.`` → ``-`` (matching ``load_master_prices_dict``).

    When ``respect_active`` is True (default) and the activation gate is OFF, the
    parquet is ignored and ``fallback`` is returned — so the live scan keeps the
    legacy static tier until OVERFLOW_UNIVERSE_ACTIVE is enabled. Backtests /
    tooling pass ``respect_active=False`` to read the built universe regardless.
    """
    if respect_active and not is_active():
        return list(fallback) if fallback is not None else []
    if path == OVERFLOW_UNIVERSE_PATH:
        _maybe_pull_from_r2(path, OVERFLOW_UNIVERSE_R2_KEY)
    df = _read_universe_frame(path)
    if df is None:
        return list(fallback) if fallback is not None else []
    tickers = {_norm(t) for t in df["ticker"].tolist()}
    return sorted(t for t in tickers if t)


def load_overflow_meta(path: str = OVERFLOW_UNIVERSE_PATH, respect_active: bool = True) -> dict:
    """Return ``{TICKER: {addv_63d, atr_pct_63d, last_close, n_bars, last_bar_date}}``.

    Empty dict when the parquet is unavailable OR when the activation gate is OFF
    (with ``respect_active=True``) → callers treat all metadata-driven gates
    (per-strategy ADDV floor, ADV cap) as no-ops, preserving legacy behavior
    until OVERFLOW_UNIVERSE_ACTIVE is enabled.
    """
    if respect_active and not is_active():
        return {}
    if path == OVERFLOW_UNIVERSE_PATH:
        _maybe_pull_from_r2(path, OVERFLOW_UNIVERSE_R2_KEY)
    df = _read_universe_frame(path)
    if df is None:
        return {}
    df = df.copy()
    df["ticker"] = [_norm(t) for t in df["ticker"].tolist()]
    cols = [
        c
        for c in ("addv_63d", "atr_pct_63d", "last_close", "n_bars", "last_bar_date")
        if c in df.columns
    ]
    out: dict = {}
    for _, row in df.iterrows():
        out[row["ticker"]] = {c: row[c] for c in cols}
    return out


def filter_by_addv(tickers, strategy_name: str, meta: dict) -> list:
    """Drop tickers below the per-strategy ADDV floor.

    No-op when ``meta`` is empty (no parquet). A ticker present in ``meta`` with
    a NaN/None ADDV is kept (can't judge); a ticker missing from ``meta`` is kept
    (it came from the universe list, which already passed the base screen).
    """
    if not meta:
        return list(tickers)
    floor = min_addv_for(strategy_name)
    kept = []
    for t in tickers:
        info = meta.get(_norm(t))
        if info is None:
            kept.append(t)
            continue
        addv = info.get("addv_63d")
        if addv is None or pd.isna(addv) or float(addv) >= floor:
            kept.append(t)
    return kept


def adv_share_cap(addv_63d, entry_price, participation: float = ADV_PARTICIPATION_CAP):
    """Max shares so that notional ≤ ``participation`` × 63d ADDV.

    Returns ``None`` when ADDV/price are unusable (caller should not cap).
    """
    try:
        addv = float(addv_63d)
        px = float(entry_price)
    except (TypeError, ValueError):
        return None
    if not (addv > 0 and px > 0):
        return None
    return int((addv * participation) / px)
