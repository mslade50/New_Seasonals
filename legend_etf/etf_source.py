"""ETF-only Legend setups from raw IBKR RTH 15-minute bars.

The prior session's body/range and inclusive EMA-touch rule come from the
original SPY/QQQ research. Execution retains the runner's 09:31/10:30 clocks.
No futures symbols, contract probes, or Databento client are used here.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .calendar import previous_session, require_full_entry_session, rth_bar_starts
from .config import NY_TZ, RULES, STRATEGY_VERSION
from .core import normalize_minutes
from .storage import atomic_write_json, content_hash, finalize_plan

ETF_SYMBOLS = ("SPY", "QQQ")
ETF_DATASET = "IBKR"
ETF_SCHEMA = "TRADES-raw-RTH-15min"
HISTORY_DAYS = 20
WARMUP_BARS = 200


def signal_history(frame: pd.DataFrame, entry_date: str) -> pd.DataFrame:
    """Select one deterministic window and require its entire exchange grid."""
    require_full_entry_session(entry_date)
    setup = previous_session(entry_date).date().isoformat()
    require_full_entry_session(setup)
    start = (pd.Timestamp(entry_date) - pd.Timedelta(days=HISTORY_DAYS)).tz_localize(NY_TZ)
    end = pd.Timestamp(f"{setup} 16:00", tz=NY_TZ)
    data = normalize_minutes(frame)
    data.index = data.index.tz_convert(NY_TZ)
    data = data.loc[(data.index >= start) & (data.index < end)]
    expected = rth_bar_starts(start.date(), setup)
    if not data.index.equals(expected):
        raise ValueError(
            "ETF signal history must match the complete prior XNYS RTH grid "
            f"(missing={len(expected.difference(data.index))}, "
            f"unexpected={len(data.index.difference(expected))})"
        )
    if len(data) < WARMUP_BARS + 26:
        raise ValueError("ETF signal history lacks 200 pre-setup warmup bars")
    return data


def history_digest(bars: pd.DataFrame) -> str:
    # Volume does not affect the signal; broker volume revisions cannot change it.
    return content_hash(
        [
            [stamp.isoformat(), *map(float, row)]
            for stamp, row in zip(
                bars.index,
                bars[["open", "high", "low", "close"]].to_numpy(),
                strict=True,
            )
        ]
    )


def evaluate_etf_setup(frame: pd.DataFrame, *, entry_date: str) -> dict[str, Any]:
    bars = signal_history(frame, entry_date)
    setup_date = previous_session(entry_date).date().isoformat()
    ema = bars["close"].ewm(span=RULES.ema_span, adjust=False).mean()
    day = bars.loc[bars.index.date == pd.Timestamp(setup_date).date()]
    day_ema = ema.loc[day.index]
    opened, closed = float(day.iloc[0]["open"]), float(day.iloc[-1]["close"])
    width = float(day["high"].max() - day["low"].min())
    ratio = abs(closed - opened) / width if width > 0 else 0.0
    touched = bool(((day["low"] <= day_ema) & (day_ema <= day["high"])).any())
    reason = (
        "trend_ratio_below_threshold"
        if ratio < RULES.trend_ratio_min
        else "ema_touch"
        if touched
        else "qualified"
    )
    return {
        "qualifies": reason == "qualified",
        "reason": reason,
        "setup_date": setup_date,
        "entry_date": entry_date,
        "trend_direction": 1 if closed > opened else -1 if closed < opened else 0,
        "trend_ratio": ratio,
        "rth_bar_count": len(day),
        "initial_ema": float(ema.iloc[-1]),
        "history_start": bars.index[0].isoformat(),
        "history_end": bars.index[-1].isoformat(),
        "history_sha256": history_digest(bars),
    }


def prepare_etf_signal_plan(
    *,
    histories: dict[str, pd.DataFrame],
    entry_date: str,
    as_of: datetime | pd.Timestamp,
    output_path: Path | None = None,
) -> dict[str, Any]:
    if set(histories) != set(ETF_SYMBOLS):
        raise ValueError("ETF signal input must contain exactly SPY and QQQ")
    stamp = pd.Timestamp(as_of)
    if stamp.tz is None:
        raise ValueError("ETF signal as_of must be timezone-aware")
    setup = previous_session(entry_date).date().isoformat()
    closed = pd.Timestamp(f"{setup} 16:00", tz=NY_TZ)
    if stamp < closed or stamp >= pd.Timestamp(f"{entry_date} 09:30", tz=NY_TZ):
        raise ValueError(
            "ETF plan must be prepared after the prior close and before entry open"
        )
    markets = [
        {
            "root": symbol,
            "etf": symbol,
            **evaluate_etf_setup(histories[symbol], entry_date=entry_date),
        }
        for symbol in ETF_SYMBOLS
    ]
    plan = finalize_plan(
        {
            "strategy_version": STRATEGY_VERSION,
            "entry_date": entry_date,
            "setup_date": setup,
            "created_at": stamp.isoformat(),
            "data_as_of": closed.isoformat(),
            "dataset": ETF_DATASET,
            "schema": ETF_SCHEMA,
            "request_start": min(item["history_start"] for item in markets),
            "quoted_cost_usd": 0.0,
            "quoted_billable_bytes": 0,
            "markets": markets,
        }
    )
    validate_etf_plan(plan, entry_date=entry_date)
    if output_path is not None:
        atomic_write_json(output_path, plan)
    return plan


def validate_etf_plan(plan: dict[str, Any], *, entry_date: str) -> None:
    require_full_entry_session(entry_date)
    setup = previous_session(entry_date).date().isoformat()
    require_full_entry_session(setup)
    allowed = {
        "strategy_version",
        "entry_date",
        "setup_date",
        "created_at",
        "data_as_of",
        "dataset",
        "schema",
        "request_start",
        "quoted_cost_usd",
        "quoted_billable_bytes",
        "markets",
        "plan_hash",
    }
    if (
        set(plan) != allowed
        or plan["schema"] != ETF_SCHEMA
        or plan["dataset"] != ETF_DATASET
    ):
        raise ValueError("ETF signal plan schema mismatch")
    body = {key: value for key, value in plan.items() if key != "plan_hash"}
    if (
        plan["plan_hash"] != content_hash(body)
        or plan["strategy_version"] != STRATEGY_VERSION
    ):
        raise ValueError("ETF signal plan hash or version mismatch")
    if plan["entry_date"] != entry_date or plan["setup_date"] != setup:
        raise ValueError(
            "ETF signal plan does not use the immediately preceding session"
        )
    closed = pd.Timestamp(f"{setup} 16:00", tz=NY_TZ)
    created = pd.Timestamp(plan["created_at"])
    if (
        created.tz is None
        or created < closed
        or created >= pd.Timestamp(f"{entry_date} 09:30", tz=NY_TZ)
        or pd.Timestamp(plan["data_as_of"]) != closed
    ):
        raise ValueError("ETF plan timestamp is stale, future, or incomplete")
    if plan["quoted_cost_usd"] != 0 or plan["quoted_billable_bytes"] != 0:
        raise ValueError("ETF plan cannot contain Databento charges")
    markets = plan["markets"]
    if not isinstance(markets, list) or len(markets) != 2:
        raise ValueError("ETF signal plan must contain exactly SPY and QQQ")
    seen = set()
    fields = {
        "root",
        "etf",
        "qualifies",
        "reason",
        "setup_date",
        "entry_date",
        "trend_direction",
        "trend_ratio",
        "rth_bar_count",
        "initial_ema",
        "history_start",
        "history_end",
        "history_sha256",
    }
    expected = rth_bar_starts(
        (pd.Timestamp(entry_date) - pd.Timedelta(days=HISTORY_DAYS)).date(), setup
    )
    for item in markets:
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError("ETF signal market schema mismatch")
        symbol = item["etf"]
        if symbol not in ETF_SYMBOLS or symbol in seen or item["root"] != symbol:
            raise ValueError("ETF signal plan has invalid or duplicated symbols")
        seen.add(symbol)
        if item["entry_date"] != entry_date or item["setup_date"] != setup:
            raise ValueError("ETF market session mismatch")
        if (
            pd.Timestamp(item["history_start"]) != expected[0]
            or pd.Timestamp(item["history_end"]) != expected[-1]
        ):
            raise ValueError("ETF market history bounds mismatch")
        digest = item["history_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("ETF market history hash invalid")
        ratio, ema = item["trend_ratio"], item["initial_ema"]
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not math.isfinite(ratio)
            or not 0 <= ratio <= 1
            or isinstance(ema, bool)
            or not isinstance(ema, (int, float))
            or not math.isfinite(ema)
            or ema <= 0
            or type(item["rth_bar_count"]) is not int
            or item["rth_bar_count"] != 26
            or type(item["trend_direction"]) is not int
            or item["trend_direction"] not in (-1, 0, 1)
            or type(item["qualifies"]) is not bool
        ):
            raise ValueError("ETF market signal values invalid")
        reason = item["reason"]
        if (
            reason not in {"qualified", "ema_touch", "trend_ratio_below_threshold"}
            or item["qualifies"] != (reason == "qualified")
            or (ratio < RULES.trend_ratio_min)
            != (reason == "trend_ratio_below_threshold")
            or (item["qualifies"] and item["trend_direction"] == 0)
        ):
            raise ValueError("ETF market qualification inconsistent")
    if pd.Timestamp(plan["request_start"]) != expected[0]:
        raise ValueError("ETF plan request start mismatch")
