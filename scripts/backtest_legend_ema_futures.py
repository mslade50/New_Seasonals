"""Backtest the MrMilk/Linda Raschke "Legend EMA" futures setup.

The linked rules are intentionally kept close to their published form:

* The prior regular session is a trend day when ``abs(close-open)`` is at
  least 75% of the session range.
* During that session price never touches the 20-period EMA on 15-minute
  bars.
* At the next 09:30 America/New_York open, trade back toward the EMA.
* Rest a limit at the EMA of the last completed 15-minute bar, updating it
  after every completed bar, and flatten on the 15:59 bar if it never fills.

Databento continuous futures prices are unadjusted.  The default result is
therefore roll-safe: the EMA resets when ``instrument_id`` changes and a
signal is omitted when its setup and entry sessions span a contract change.
The script also emits an inclusive-roll sensitivity check so the impact is
visible rather than hidden.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "artifacts" / "databento" / "parquet"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "databento" / "legend_ema_backtest"
NY_TZ = "America/New_York"
RTH_OPEN = "09:30"
RTH_CLOSE = "16:00"
SOURCE_THREAD_URL = "https://x.com/MrMilkTrading/status/2094768621419471130"
LINDA_CORROBORATION_URL = "https://x.com/LindaRaschke/status/2008937185399963728"


@dataclass(frozen=True)
class InstrumentSpec:
    root: str
    symbol: str
    point_value: float
    tick_size: float
    trusted_start: str


INSTRUMENTS = {
    # The Databento archive is nominally older, but ES/NQ omit many entire
    # RTH sessions before 2016.  These starts mark the contiguous research
    # windows used in the headline table; older eligible observations remain
    # in the output as a clearly labelled sensitivity sample.
    "ES": InstrumentSpec("ES", "ES.v.0", 50.0, 0.25, "2016-01-01"),
    "NQ": InstrumentSpec("NQ", "NQ.v.0", 20.0, 0.25, "2016-01-01"),
    # RTY is present from July 2017, but its launch-half-year is materially
    # thinner.  Keep those trades in the observed sample and headline 2018+.
    "RTY": InstrumentSpec("RTY", "RTY.v.0", 50.0, 0.10, "2018-01-01"),
}


@dataclass(frozen=True)
class Variant:
    name: str
    ema_basis: str = "all"
    stop_atr_multiple: float | None = None
    exclude_roll_windows: bool = True
    trend_threshold: float = 0.75
    friction_ticks: float = 5.0
    entry_delay_minutes: int = 0
    ema_update_delay_minutes: int = 0
    round_orders_to_tick: bool = False
    fallback_exit: str = "bar_close"
    time_stop: str | None = None


DEFAULT_VARIANTS = (
    Variant(
        "published_replication",
        ema_basis="rth",
    ),
    Variant(
        "executable_1m_delay",
        ema_basis="rth",
        entry_delay_minutes=1,
        ema_update_delay_minutes=1,
        round_orders_to_tick=True,
        fallback_exit="last_minute_open",
    ),
    Variant(
        "executable_time_stop_1000",
        ema_basis="rth",
        entry_delay_minutes=1,
        ema_update_delay_minutes=1,
        round_orders_to_tick=True,
        fallback_exit="last_minute_open",
        time_stop="10:00",
    ),
    Variant(
        "executable_time_stop_1030",
        ema_basis="rth",
        entry_delay_minutes=1,
        ema_update_delay_minutes=1,
        round_orders_to_tick=True,
        fallback_exit="last_minute_open",
        time_stop="10:30",
    ),
    Variant(
        "executable_time_stop_1100",
        ema_basis="rth",
        entry_delay_minutes=1,
        ema_update_delay_minutes=1,
        round_orders_to_tick=True,
        fallback_exit="last_minute_open",
        time_stop="11:00",
    ),
    Variant(
        "executable_time_stop_1200",
        ema_basis="rth",
        entry_delay_minutes=1,
        ema_update_delay_minutes=1,
        round_orders_to_tick=True,
        fallback_exit="last_minute_open",
        time_stop="12:00",
    ),
    Variant(
        "executable_1m_delay_atr_stop",
        ema_basis="rth",
        stop_atr_multiple=0.5,
        entry_delay_minutes=1,
        ema_update_delay_minutes=1,
        round_orders_to_tick=True,
        fallback_exit="last_minute_open",
    ),
    Variant(
        "executable_threshold_70",
        ema_basis="rth",
        trend_threshold=0.70,
        entry_delay_minutes=1,
        ema_update_delay_minutes=1,
        round_orders_to_tick=True,
        fallback_exit="last_minute_open",
    ),
    Variant(
        "executable_threshold_80",
        ema_basis="rth",
        trend_threshold=0.80,
        entry_delay_minutes=1,
        ema_update_delay_minutes=1,
        round_orders_to_tick=True,
        fallback_exit="last_minute_open",
    ),
    Variant(
        "all_session_executable_1m_delay",
        entry_delay_minutes=1,
        ema_update_delay_minutes=1,
        round_orders_to_tick=True,
        fallback_exit="last_minute_open",
    ),
)
PUBLISHED_VARIANT = "published_replication"
EOD_VARIANT = "executable_1m_delay"
# Adopted across ES/NQ/RTY and both directions.  The original EOD version is
# retained above as a legacy comparison, not as the primary specification.
BASE_VARIANT = "executable_time_stop_1030"
HEADLINE_VARIANTS = {
    PUBLISHED_VARIANT,
    BASE_VARIANT,
    EOD_VARIANT,
}


def _local_session_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return timezone-naive local midnights for a NY-localized index."""

    return index.normalize().tz_localize(None)


def _in_rth(index: pd.DatetimeIndex) -> np.ndarray:
    minutes = index.hour * 60 + index.minute
    return (minutes >= 9 * 60 + 30) & (minutes < 16 * 60) & (index.dayofweek < 5)


def load_symbol_minutes(data_dir: Path, symbol: str) -> pd.DataFrame:
    """Load one continuous symbol from yearly Databento Parquet files."""

    paths = sorted(data_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No Parquet files found under {data_dir}")

    columns = ["instrument_id", "open", "high", "low", "close", "volume", "symbol"]
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_parquet(
            path,
            columns=columns,
            filters=[("symbol", "==", symbol)],
        )
        if not frame.empty:
            frames.append(frame.drop(columns="symbol"))

    if not frames:
        raise ValueError(f"Symbol {symbol!r} was not present in {data_dir}")

    data = pd.concat(frames).sort_index()
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("Expected Databento ts_event to be restored as the Parquet index")
    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
    else:
        data.index = data.index.tz_convert("UTC")
    if data.index.has_duplicates:
        data = data.loc[~data.index.duplicated(keep="last")]

    data.index = data.index.tz_convert(NY_TZ)
    data.index.name = "ts_event"
    return data


def _ema_reset_on_contract_change(bars: pd.DataFrame, span: int = 20) -> pd.Series:
    segment = bars["instrument_id"].ne(bars["instrument_id"].shift()).cumsum()
    return bars.groupby(segment, sort=False)["close"].transform(
        lambda values: values.ewm(span=span, adjust=False, min_periods=span).mean()
    )


def build_15_minute_bars(minutes: pd.DataFrame) -> pd.DataFrame:
    """Build aligned 15-minute bars and all-session/RTH EMA variants."""

    bars = minutes.resample("15min", label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        instrument_id=("instrument_id", "first"),
        instrument_count=("instrument_id", "nunique"),
        minute_count=("close", "count"),
    )
    bars = bars.loc[(bars["minute_count"] > 0) & bars["instrument_count"].eq(1)].copy()
    bars["instrument_id"] = bars["instrument_id"].astype("uint32")
    bars["ema20_all"] = _ema_reset_on_contract_change(bars)

    rth = bars.loc[_in_rth(bars.index)].copy()
    rth["ema20_rth"] = _ema_reset_on_contract_change(rth)
    bars["ema20_rth"] = rth["ema20_rth"]
    bars["session_date"] = _local_session_dates(bars.index)
    return bars


def build_daily_sessions(minutes: pd.DataFrame, bars15: pd.DataFrame) -> pd.DataFrame:
    """Aggregate RTH sessions and compute point-in-time ATR and setup fields."""

    rth = minutes.loc[_in_rth(minutes.index)].copy()
    rth["session_date"] = _local_session_dates(rth.index)
    grouped = rth.groupby("session_date", sort=True)
    daily = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        minute_count=("close", "count"),
        first_ts=("close", lambda values: values.index[0]),
        last_ts=("close", lambda values: values.index[-1]),
        instrument_id=("instrument_id", "first"),
        instrument_count=("instrument_id", "nunique"),
    )
    daily["instrument_id"] = daily["instrument_id"].astype("uint32")

    first_clock = daily["first_ts"].map(lambda value: value.strftime("%H:%M"))
    last_clock = daily["last_ts"].map(lambda value: value.strftime("%H:%M"))
    # Databento omits intervals with no trades.  Requiring all 390 timestamps
    # would wrongly discard legitimate no-trade minutes and exchange halts.
    # Require actual execution anchors here and all 26 real 15-minute bins below.
    daily["complete_rth"] = (
        first_clock.eq("09:30")
        & last_clock.eq("15:59")
        & daily["instrument_count"].eq(1)
    )

    day_range = daily["high"] - daily["low"]
    daily["trend_ratio"] = (daily["close"] - daily["open"]).abs().div(day_range)
    daily["trend_direction"] = np.sign(daily["close"] - daily["open"]).astype("int8")

    rth15 = bars15.loc[_in_rth(bars15.index)].copy()
    rth15["session_date"] = _local_session_dates(rth15.index)
    for basis in ("all", "rth"):
        ema_col = f"ema20_{basis}"
        valid = rth15[ema_col].notna()
        above = valid & (rth15["low"] > rth15[ema_col])
        below = valid & (rth15["high"] < rth15[ema_col])
        rth15[f"no_touch_{basis}"] = above | below
        rth15[f"above_{basis}"] = above
        rth15[f"below_{basis}"] = below

    bar_groups = rth15.groupby("session_date", sort=True)
    touch_stats = bar_groups.agg(
        rth_bar_count=("close", "count"),
        no_touch_all=("no_touch_all", "all"),
        all_above_all=("above_all", "all"),
        all_below_all=("below_all", "all"),
        no_touch_rth=("no_touch_rth", "all"),
        all_above_rth=("above_rth", "all"),
        all_below_rth=("below_rth", "all"),
        rth15_instrument_count=("instrument_id", "nunique"),
        min_minutes_per_rth_bar=("minute_count", "min"),
    )
    daily = daily.join(touch_stats)
    daily["complete_rth"] &= daily["rth_bar_count"].eq(26) & daily[
        "min_minutes_per_rth_bar"
    ].ge(1)

    # ATR is a prior-day execution input.  Keep raw roll gaps out by resetting
    # previous-close true range at each contract ID, while carrying the Wilder
    # estimate across the splice (equivalent to treating the roll gap as zero).
    # Shortened/partial sessions do not enter the estimator.
    contract_segment = daily["instrument_id"].ne(daily["instrument_id"].shift()).cumsum()
    previous_close = daily.groupby(contract_segment, sort=False)["close"].shift(1)
    true_range = pd.concat(
        [
            day_range,
            (daily["high"] - previous_close).abs(),
            (daily["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    true_range = true_range.where(daily["complete_rth"])
    daily["atr14"] = true_range.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
        ignore_na=True,
    ).mean()

    daily["setup_all"] = daily["no_touch_all"] & np.where(
        daily["trend_direction"] > 0,
        daily["all_above_all"],
        daily["all_below_all"],
    )
    daily["setup_rth"] = daily["no_touch_rth"] & np.where(
        daily["trend_direction"] > 0,
        daily["all_above_rth"],
        daily["all_below_rth"],
    )
    daily["next_session"] = pd.Series(daily.index, index=daily.index).shift(-1)
    return daily


def _active_ema_for_minutes(
    bars15: pd.DataFrame,
    minute_index: pd.DatetimeIndex,
    ema_basis: str,
    activation_delay_minutes: int = 0,
) -> pd.Series:
    ema = bars15[f"ema20_{ema_basis}"].dropna().copy()
    ema.index = ema.index + pd.Timedelta(minutes=15 + activation_delay_minutes)
    return ema.reindex(minute_index, method="ffill")


def round_order_price(price: float, tick_size: float, direction: int, order: str) -> float:
    """Round an exit order away from the current position, conservatively."""

    scaled = price / tick_size
    if order == "limit":
        ticks = math.ceil(scaled - 1e-10) if direction > 0 else math.floor(scaled + 1e-10)
    elif order == "stop":
        ticks = math.floor(scaled + 1e-10) if direction > 0 else math.ceil(scaled - 1e-10)
    else:
        raise ValueError("order must be 'limit' or 'stop'")
    return float(ticks * tick_size)


def _limit_fill(row: pd.Series, target: float, direction: int) -> float | None:
    if direction > 0:
        if row["open"] >= target:
            return float(row["open"])
        if row["high"] >= target:
            return float(target)
    else:
        if row["open"] <= target:
            return float(row["open"])
        if row["low"] <= target:
            return float(target)
    return None


def _stop_fill(row: pd.Series, stop: float, direction: int) -> float | None:
    if direction > 0:
        if row["open"] <= stop:
            return float(row["open"])
        if row["low"] <= stop:
            return float(stop)
    else:
        if row["open"] >= stop:
            return float(row["open"])
        if row["high"] >= stop:
            return float(stop)
    return None


def simulate_trade(
    session_minutes: pd.DataFrame,
    active_ema: pd.Series,
    direction: int,
    stop_distance: float | None = None,
    tick_size: float | None = None,
    round_orders_to_tick: bool = False,
    fallback_exit: str = "bar_close",
) -> dict[str, object]:
    """Execute one trade; same-minute stop/target ambiguity is stop-first."""

    if session_minutes.empty:
        raise ValueError("Cannot simulate an empty session")
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")

    entry_ts = session_minutes.index[0]
    entry_price = float(session_minutes.iloc[0]["open"])
    if fallback_exit not in {"bar_close", "last_minute_open"}:
        raise ValueError("Unsupported fallback_exit")

    stop_price = np.nan
    if stop_distance is not None and math.isfinite(stop_distance):
        stop_price = entry_price - direction * stop_distance
        if round_orders_to_tick:
            if tick_size is None:
                raise ValueError("tick_size is required when rounding orders")
            stop_price = round_order_price(stop_price, tick_size, direction, "stop")

    exit_ts = session_minutes.index[-1]
    exit_price = float(
        session_minutes.iloc[-1]["close" if fallback_exit == "bar_close" else "open"]
    )
    exit_reason = "close" if fallback_exit == "bar_close" else "last_minute_open"
    scan_minutes = (
        session_minutes
        if fallback_exit == "bar_close"
        else session_minutes.iloc[:-1]
    )

    for ts, row in scan_minutes.iterrows():
        if pd.notna(stop_price):
            fill = _stop_fill(row, stop_price, direction)
            if fill is not None:
                exit_ts = ts
                exit_price = fill
                exit_reason = "atr_stop"
                break

        target = active_ema.get(ts, np.nan)
        if pd.notna(target):
            fill = _limit_fill(row, float(target), direction)
            if fill is not None:
                exit_ts = ts
                exit_price = fill
                exit_reason = "ema_limit"
                break

    # Minute OHLC cannot reveal whether an excursion in the exit minute came
    # before or after the fill, so diagnostics deliberately stop one bar early.
    path = session_minutes.loc[session_minutes.index < exit_ts]
    if path.empty:
        mfe_points = np.nan
        mae_points = np.nan
    elif direction > 0:
        mfe_points = float(path["high"].max() - entry_price)
        mae_points = float(path["low"].min() - entry_price)
    else:
        mfe_points = float(entry_price - path["low"].min())
        mae_points = float(entry_price - path["high"].max())

    return {
        "entry_ts": entry_ts,
        "exit_ts": exit_ts,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "gross_points": direction * (exit_price - entry_price),
        "mfe_points_before_exit_minute": mfe_points,
        "mae_points_before_exit_minute": mae_points,
        "stop_price": stop_price,
    }


def _session_slice(minutes: pd.DataFrame, session_date: pd.Timestamp) -> pd.DataFrame:
    start = session_date.tz_localize(NY_TZ) + pd.Timedelta(hours=9, minutes=30)
    end = session_date.tz_localize(NY_TZ) + pd.Timedelta(hours=16)
    return minutes.loc[start : end - pd.Timedelta(nanoseconds=1)]


def _session_through_time_stop(
    session_minutes: pd.DataFrame,
    session_date: pd.Timestamp,
    time_stop: str,
) -> pd.DataFrame:
    """Include the first observed minute at/after a causal intraday cutoff.

    ``simulate_trade(..., fallback_exit="last_minute_open")`` deliberately
    does not inspect the last row's high/low.  Its opening print is therefore
    an executable market-exit proxy at the requested time without intraminute
    lookahead.
    """

    try:
        hour_text, minute_text = time_stop.split(":", maxsplit=1)
        hour = int(hour_text)
        minute = int(minute_text)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("time_stop must use HH:MM format") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError("time_stop must be a valid clock time")

    cutoff = session_date.tz_localize(NY_TZ) + pd.Timedelta(
        hours=hour,
        minutes=minute,
    )
    candidates = session_minutes.index[session_minutes.index >= cutoff]
    if len(candidates) == 0:
        return session_minutes.iloc[0:0]
    return session_minutes.loc[: candidates[0]]


def _spans_contract_change(
    minutes: pd.DataFrame,
    setup_date: pd.Timestamp,
    entry_date: pd.Timestamp,
) -> bool:
    start = setup_date.tz_localize(NY_TZ) + pd.Timedelta(hours=9, minutes=30)
    end = entry_date.tz_localize(NY_TZ) + pd.Timedelta(hours=16)
    window = minutes.loc[start : end - pd.Timedelta(nanoseconds=1)]
    return window["instrument_id"].nunique() != 1


def run_variant(
    minutes: pd.DataFrame,
    bars15: pd.DataFrame,
    daily: pd.DataFrame,
    spec: InstrumentSpec,
    variant: Variant,
) -> pd.DataFrame:
    setup_col = f"setup_{variant.ema_basis}"
    qualifying = daily.loc[
        daily["complete_rth"]
        & daily[setup_col].fillna(False)
        & daily["trend_ratio"].ge(variant.trend_threshold)
    ]
    friction_points = variant.friction_ticks * spec.tick_size
    rows: list[dict[str, object]] = []

    for setup_date, setup in qualifying.iterrows():
        entry_date = setup["next_session"]
        if pd.isna(entry_date) or entry_date not in daily.index:
            continue
        entry_day = daily.loc[entry_date]
        if not bool(entry_day["complete_rth"]):
            continue

        roll_window = _spans_contract_change(minutes, setup_date, entry_date)
        if variant.exclude_roll_windows and roll_window:
            continue

        session = _session_slice(minutes, entry_date)
        if session.empty:
            continue
        active_ema = _active_ema_for_minutes(
            bars15,
            session.index,
            variant.ema_basis,
            activation_delay_minutes=variant.ema_update_delay_minutes,
        )
        initial_ema = active_ema.iloc[0]
        if pd.isna(initial_ema):
            continue

        reference_open = float(session.iloc[0]["open"])
        if reference_open == float(initial_ema):
            continue
        direction = -1 if reference_open > float(initial_ema) else 1

        if variant.round_orders_to_tick:
            active_ema = active_ema.map(
                lambda value: (
                    round_order_price(float(value), spec.tick_size, direction, "limit")
                    if pd.notna(value)
                    else np.nan
                )
            )

        expected_entry_ts = session.index[0] + pd.Timedelta(
            minutes=variant.entry_delay_minutes
        )
        entry_candidates = session.index[session.index >= expected_entry_ts]
        if len(entry_candidates) == 0:
            continue
        actual_entry_ts = entry_candidates[0]

        # For the executable sensitivity, the opening minute is observed before
        # entry.  If it already reached the EMA, the opportunity has passed.
        pre_entry = session.loc[session.index < actual_entry_ts]
        opportunity_passed = False
        for ts, row in pre_entry.iterrows():
            target = active_ema.get(ts, np.nan)
            if pd.notna(target) and _limit_fill(row, float(target), direction) is not None:
                opportunity_passed = True
                break
        if opportunity_passed:
            continue

        execution_session = session.loc[actual_entry_ts:]
        if variant.time_stop is not None:
            execution_session = _session_through_time_stop(
                execution_session,
                entry_date,
                variant.time_stop,
            )
            if execution_session.empty:
                continue
        entry_price = float(execution_session.iloc[0]["open"])
        target_at_entry = active_ema.get(actual_entry_ts, np.nan)
        if pd.isna(target_at_entry):
            continue
        if (direction > 0 and entry_price >= target_at_entry) or (
            direction < 0 and entry_price <= target_at_entry
        ):
            continue

        atr14 = float(setup["atr14"]) if pd.notna(setup["atr14"]) else np.nan
        stop_distance = None
        if variant.stop_atr_multiple is not None:
            if not math.isfinite(atr14):
                continue
            stop_distance = variant.stop_atr_multiple * atr14

        trade = simulate_trade(
            execution_session,
            active_ema,
            direction,
            stop_distance,
            tick_size=spec.tick_size,
            round_orders_to_tick=variant.round_orders_to_tick,
            fallback_exit=variant.fallback_exit,
        )
        if variant.time_stop is not None and trade["exit_reason"] == "last_minute_open":
            trade["exit_reason"] = "time_stop"
        net_points = float(trade["gross_points"]) - friction_points
        rows.append(
            {
                "symbol": spec.root,
                "variant": variant.name,
                "setup_date": setup_date.date().isoformat(),
                "entry_date": entry_date.date().isoformat(),
                "direction": "long" if direction > 0 else "short",
                "trend_direction": "up" if setup["trend_direction"] > 0 else "down",
                "trend_ratio": float(setup["trend_ratio"]),
                "initial_ema": float(initial_ema),
                "reference_open_0930": reference_open,
                "target_at_entry": float(target_at_entry),
                "entry_delay_minutes": variant.entry_delay_minutes,
                "ema_update_delay_minutes": variant.ema_update_delay_minutes,
                "time_stop": variant.time_stop,
                "atr14": atr14,
                "roll_window": bool(roll_window),
                **trade,
                "friction_points": friction_points,
                "net_points": net_points,
                "pnl_dollars": net_points * spec.point_value,
                "return_bps": net_points / float(trade["entry_price"]) * 10_000,
                "pnl_atr": net_points / atr14 if math.isfinite(atr14) and atr14 else np.nan,
            }
        )

    return pd.DataFrame(rows)


def max_drawdown(pnl: Iterable[float]) -> float:
    values = np.asarray(list(pnl), dtype=float)
    if values.size == 0:
        return 0.0
    equity = np.cumsum(values)
    peaks = np.maximum.accumulate(np.r_[0.0, equity])
    drawdowns = np.r_[0.0, equity] - peaks
    return float(drawdowns.min())


def summarize_trades(trades: pd.DataFrame) -> dict[str, object]:
    if trades.empty:
        return {
            "trades": 0,
            "win_rate_pct": None,
            "profit_factor": None,
            "total_pnl_dollars": 0.0,
            "max_drawdown_dollars": 0.0,
        }

    ordered = trades.sort_values(["entry_ts", "symbol"])
    pnl = ordered["pnl_dollars"].astype(float)
    gains = float(pnl.loc[pnl > 0].sum())
    losses = float(-pnl.loc[pnl < 0].sum())
    returns = ordered["return_bps"].astype(float)
    return {
        "trades": int(len(ordered)),
        "first_entry": str(ordered["entry_date"].iloc[0]),
        "last_entry": str(ordered["entry_date"].iloc[-1]),
        "win_rate_pct": float((pnl > 0).mean() * 100),
        "profit_factor": gains / losses if losses else None,
        "avg_net_points": float(ordered["net_points"].mean()),
        "median_net_points": float(ordered["net_points"].median()),
        "avg_return_bps": float(returns.mean()),
        "median_return_bps": float(returns.median()),
        "naive_iid_return_bps_t_stat": (
            float(returns.mean() / (returns.std(ddof=1) / math.sqrt(len(returns))))
            if len(returns) > 1 and returns.std(ddof=1) > 0
            else None
        ),
        "total_pnl_dollars": float(pnl.sum()),
        "avg_pnl_dollars": float(pnl.mean()),
        "worst_trade_dollars": float(pnl.min()),
        "max_drawdown_dollars": max_drawdown(pnl),
        "ema_exit_pct": float((ordered["exit_reason"] == "ema_limit").mean() * 100),
        "roll_window_trades": int(ordered["roll_window"].sum()),
    }


def build_summary(trades: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {
        "trusted_by_symbol_variant": {},
        "observed_by_symbol_variant": {},
        "eras": {},
    }
    if trades.empty:
        return result

    for (symbol, variant), group in trades.groupby(["symbol", "variant"], sort=True):
        key = f"{symbol}|{variant}"
        result["observed_by_symbol_variant"][key] = summarize_trades(group)
        trusted_start = pd.Timestamp(INSTRUMENTS[symbol].trusted_start)
        entry_dates = pd.to_datetime(group["entry_date"])
        result["trusted_by_symbol_variant"][key] = summarize_trades(
            group.loc[entry_dates >= trusted_start]
        )

    baseline = trades.loc[trades["variant"] == BASE_VARIANT].copy()
    trusted_mask = pd.Series(False, index=baseline.index)
    for symbol, spec in INSTRUMENTS.items():
        trusted_mask |= baseline["symbol"].eq(symbol) & pd.to_datetime(
            baseline["entry_date"]
        ).ge(pd.Timestamp(spec.trusted_start))
    baseline = baseline.loc[trusted_mask]
    entry_year = pd.to_datetime(baseline["entry_date"]).dt.year
    era_masks = {
        "2016_2019": entry_year.between(2016, 2019),
        "2020_present": entry_year >= 2020,
        "2022_present": entry_year >= 2022,
    }
    for era, mask in era_masks.items():
        result["eras"][era] = {}
        for symbol, group in baseline.loc[mask].groupby("symbol", sort=True):
            result["eras"][era][symbol] = summarize_trades(group)
    return result


def build_yearly(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    data = trades.copy()
    data["year"] = pd.to_datetime(data["entry_date"]).dt.year
    rows = []
    for (symbol, variant, year), group in data.groupby(["symbol", "variant", "year"]):
        summary = summarize_trades(group)
        rows.append({"symbol": symbol, "variant": variant, "year": year, **summary})
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No trades."
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    trades: pd.DataFrame,
    summary: dict[str, object],
    yearly: pd.DataFrame,
    coverage: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trades.sort_values(["symbol", "variant", "entry_ts"]).to_csv(
        output_dir / "trades.csv", index=False
    )
    yearly.to_csv(output_dir / "yearly.csv", index=False)
    payload = {
        "sources": {
            "strategy_thread": SOURCE_THREAD_URL,
            "linda_corroboration": LINDA_CORROBORATION_URL,
        },
        "coverage": coverage,
        "variants": [asdict(v) for v in DEFAULT_VARIANTS],
        **summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = []
    sensitivity_rows = []
    for key, values in summary["trusted_by_symbol_variant"].items():
        symbol, variant = key.split("|", maxsplit=1)
        row = {
            "Symbol": symbol,
            "Variant": variant,
            "Trades": values["trades"],
            "Win %": f"{values['win_rate_pct']:.1f}",
            "PF": f"{values['profit_factor']:.2f}" if values["profit_factor"] else "n/a",
            "Avg bps": f"{values['avg_return_bps']:.2f}",
            "PnL $": f"{values['total_pnl_dollars']:.0f}",
            "Max DD $": f"{values['max_drawdown_dollars']:.0f}",
        }
        (rows if variant in HEADLINE_VARIANTS else sensitivity_rows).append(row)

    time_stop_rows = []
    time_stop_variants = {
        EOD_VARIANT,
        "executable_time_stop_1000",
        "executable_time_stop_1030",
        "executable_time_stop_1100",
        "executable_time_stop_1200",
    }
    time_stop_trades = trades.loc[trades["variant"].isin(time_stop_variants)].copy()
    for (symbol, variant, direction), group in time_stop_trades.groupby(
        ["symbol", "variant", "direction"],
        sort=True,
    ):
        trusted = group.loc[
            pd.to_datetime(group["entry_date"]).ge(
                pd.Timestamp(INSTRUMENTS[symbol].trusted_start)
            )
        ]
        values = summarize_trades(trusted)
        time_stop_rows.append(
            {
                "Symbol": symbol,
                "Cutoff": "EOD" if variant == EOD_VARIANT else variant.rsplit("_", 1)[-1],
                "Side": direction,
                "Trades": values["trades"],
                "PF": (
                    f"{values['profit_factor']:.2f}"
                    if values["profit_factor"] is not None
                    else "n/a"
                ),
                "PnL $": f"{values['total_pnl_dollars']:.0f}",
                "Max DD $": f"{values['max_drawdown_dollars']:.0f}",
            }
        )
    report = [
        "# Legend EMA futures backtest",
        "",
        f"Source: {SOURCE_THREAD_URL}",
        "",
        "One-contract results after five ticks of round-trip friction. Every RTH",
        "variant resets its EMA on contract changes and excludes setup-to-entry",
        "windows that cross a roll. `published_replication` follows the post's",
        "idealized fill at the known 09:30 opening print. `executable_1m_delay`",
        "observes that minute, enters at the next real minute only if the EMA was",
        "not reached, delays cancel/replace updates one minute, rounds orders to",
        "valid ticks, and flattens at the 15:59 opening print.",
        "`executable_time_stop_HHMM` uses those same executable rules, keeps",
        "the EMA limit active through the preceding minute, and otherwise",
        "flattens at the cutoff minute's opening print without inspecting that",
        "minute's high or low.",
        "`executable_time_stop_1030` is the adopted primary specification for",
        "every symbol and direction; `executable_1m_delay` remains only as the",
        "legacy EOD comparison.",
        "The headline table starts ES/NQ in 2016 and RTY in 2018;",
        "non-headline eligible observations remain in `summary.json`.",
        "",
        _markdown_table(pd.DataFrame(rows)),
        "",
        "## Sensitivities",
        "",
        _markdown_table(pd.DataFrame(sensitivity_rows)),
        "",
        "## Time-stop direction split",
        "",
        _markdown_table(pd.DataFrame(time_stop_rows)),
        "",
        "See `summary.json`, `yearly.csv`, and `trades.csv` for full detail.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--symbols",
        nargs="+",
        choices=sorted(INSTRUMENTS),
        default=sorted(INSTRUMENTS),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_trades: list[pd.DataFrame] = []
    coverage: dict[str, object] = {}

    for root in args.symbols:
        spec = INSTRUMENTS[root]
        print(f"Loading {spec.symbol} ...", flush=True)
        minutes = load_symbol_minutes(args.data_dir, spec.symbol)
        bars15 = build_15_minute_bars(minutes)
        daily = build_daily_sessions(minutes, bars15)
        sessions_by_year = {}
        for year, group in daily.groupby(daily.index.year):
            sessions_by_year[str(year)] = {
                "observed_rth_sessions": int(len(group)),
                "complete_rth_sessions": int(group["complete_rth"].sum()),
                "median_minute_rows": float(group["minute_count"].median()),
            }
        coverage[root] = {
            "first_minute": minutes.index[0].isoformat(),
            "last_minute": minutes.index[-1].isoformat(),
            "minute_rows": int(len(minutes)),
            "rth_sessions": int(daily["complete_rth"].sum()),
            "instrument_ids": int(minutes["instrument_id"].nunique()),
            "trusted_start": spec.trusted_start,
            "sessions_by_year": sessions_by_year,
        }
        for variant in DEFAULT_VARIANTS:
            print(f"  Running {variant.name} ...", flush=True)
            result = run_variant(minutes, bars15, daily, spec, variant)
            if not result.empty:
                all_trades.append(result)
        del minutes, bars15, daily
        gc.collect()

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    summary = build_summary(trades)
    yearly = build_yearly(trades)
    write_report(args.output_dir, trades, summary, yearly, coverage)
    print(f"Wrote results to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
