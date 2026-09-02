"""Pure, causal signal and ETF execution logic for Legend EMA.

Every clock comparison is performed in America/New_York, while stored inputs
may be UTC.  The functions in this module are deterministic and broker-free so
the live implementation can be replayed exactly against historical fixtures.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .calendar import session_labels
from .config import NY_TZ, RULES

OHLC = ("open", "high", "low", "close")


@dataclass(frozen=True)
class FuturesSetup:
    qualifies: bool
    reason: str
    setup_date: str
    entry_date: str
    instrument_id: int | None = None
    entry_instrument_id: int | None = None
    trend_direction: int | None = None
    trend_ratio: float | None = None
    rth_bar_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EntryDecision:
    eligible: bool
    reason: str
    direction: int | None = None
    side: str | None = None
    initial_target: float | None = None
    open_0930: float | None = None
    decision_price: float | None = None


def normalize_minutes(
    frame: pd.DataFrame,
    *,
    timestamp_col: str | None = None,
    naive_tz: str | None = None,
    require_instrument: bool = False,
) -> pd.DataFrame:
    """Return sorted UTC-indexed OHLCV minutes and reject ambiguous input.

    Duplicate minute labels are rejected instead of silently aggregating a
    second decision bar.  Naive timestamps require an explicit source zone.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("minutes must be a pandas DataFrame")
    result = frame.copy()
    if timestamp_col is not None:
        if timestamp_col not in result:
            raise ValueError(f"missing timestamp column {timestamp_col!r}")
        index = pd.DatetimeIndex(pd.to_datetime(result.pop(timestamp_col), errors="raise"))
    else:
        index = pd.DatetimeIndex(pd.to_datetime(result.index, errors="raise"))
    if index.tz is None:
        if naive_tz is None:
            raise ValueError("naive timestamps require naive_tz")
        index = index.tz_localize(naive_tz, ambiguous="raise", nonexistent="raise")
    index = index.tz_convert("UTC")
    if index.has_duplicates:
        duplicates = index[index.duplicated()].unique()
        raise ValueError(f"duplicate minute timestamp(s): {list(duplicates[:3])}")
    result.index = index
    result = result.sort_index()
    missing = [column for column in OHLC if column not in result]
    if missing:
        raise ValueError(f"missing OHLC columns: {', '.join(missing)}")
    for column in OHLC:
        result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
    ohlc_values = result[list(OHLC)].to_numpy(dtype=float)
    if not np.isfinite(ohlc_values).all() or (ohlc_values <= 0).any():
        raise ValueError("OHLC values must be positive and finite")
    invalid = (
        (result["high"] < result[["open", "close", "low"]].max(axis=1))
        | (result["low"] > result[["open", "close", "high"]].min(axis=1))
    )
    if invalid.any():
        raise ValueError("malformed OHLC relationship")
    if require_instrument:
        if "instrument_id" not in result:
            raise ValueError("futures minutes require instrument_id")
        result["instrument_id"] = pd.to_numeric(
            result["instrument_id"], errors="raise"
        ).astype("int64")
    if "volume" not in result:
        result["volume"] = 0.0
    result["volume"] = pd.to_numeric(result["volume"], errors="raise").astype(float)
    volume = result["volume"].to_numpy(dtype=float)
    if not np.isfinite(volume).all() or (volume < 0).any():
        raise ValueError("volume values must be non-negative and finite")
    return result


def _et_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.tz is None:
        raise ValueError("timestamps must be timezone-aware")
    return index.tz_convert(NY_TZ)


def _rth_mask(index: pd.DatetimeIndex) -> np.ndarray:
    local = _et_index(index)
    clock = local.hour * 60 + local.minute
    return (clock >= 570) & (clock < 960) & (local.dayofweek < 5)


def _session_dates(index: pd.DatetimeIndex) -> pd.Index:
    return pd.Index(_et_index(index).date, name="session_date")


def _ema_reset_on_contract_change(bars: pd.DataFrame, span: int = 20) -> pd.Series:
    segment = bars["instrument_id"].ne(bars["instrument_id"].shift()).cumsum()
    return bars.groupby(segment, sort=False)["close"].transform(
        lambda values: values.ewm(span=span, adjust=False, min_periods=span).mean()
    )


def build_futures_rth15(minutes: pd.DataFrame) -> pd.DataFrame:
    """Build the exact left-labelled RTH 15-minute EMA series."""

    data = normalize_minutes(minutes, require_instrument=True)
    local = data.copy()
    local.index = _et_index(local.index)
    local = local.loc[_rth_mask(local.index)]
    pieces: list[pd.DataFrame] = []
    for _, day in local.groupby(local.index.date, sort=True):
        bars = day.resample("15min", label="left", closed="left").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            instrument_id=("instrument_id", "first"),
            instrument_count=("instrument_id", "nunique"),
            minute_count=("close", "count"),
        )
        bars = bars.loc[
            bars["close"].notna()
            & bars["instrument_count"].eq(1)
            & bars["minute_count"].gt(0)
        ]
        pieces.append(bars)
    if not pieces:
        return pd.DataFrame()
    bars = pd.concat(pieces).sort_index()
    bars["instrument_id"] = bars["instrument_id"].astype("int64")
    bars["ema20"] = _ema_reset_on_contract_change(bars, RULES.ema_span)
    bars["session_date"] = _session_dates(bars.index)
    return bars


def latest_observed_futures_rth_session(
    minutes: pd.DataFrame,
    *,
    entry_date: date | str | pd.Timestamp,
    as_of: pd.Timestamp,
) -> date | None:
    """Latest observed pre-entry CME date with any 09:30-16:00 ET trade."""

    entry_day = pd.Timestamp(entry_date).date()
    as_of_stamp = pd.Timestamp(as_of)
    if as_of_stamp.tz is None:
        raise ValueError("as_of must be timezone-aware")
    data = normalize_minutes(minutes, require_instrument=True)
    data = data.loc[data.index <= as_of_stamp.tz_convert("UTC")]
    if data.empty:
        return None
    local = _et_index(data.index)
    eligible = (local.date < entry_day) & _rth_mask(data.index)
    observed = pd.Index(local[eligible].date)
    return None if observed.empty else max(observed)


def evaluate_futures_setup(
    minutes: pd.DataFrame,
    *,
    setup_date: date | str | pd.Timestamp,
    entry_date: date | str | pd.Timestamp,
    as_of: pd.Timestamp,
) -> FuturesSetup:
    """Evaluate one roll-safe prior-session setup using only data at ``as_of``."""

    setup_day = pd.Timestamp(setup_date).date()
    entry_day = pd.Timestamp(entry_date).date()
    as_of_stamp = pd.Timestamp(as_of)
    if as_of_stamp.tz is None:
        raise ValueError("as_of must be timezone-aware")
    data = normalize_minutes(minutes, require_instrument=True)
    data = data.loc[data.index <= as_of_stamp.tz_convert("UTC")]
    base = {"setup_date": str(setup_day), "entry_date": str(entry_day)}
    if data.empty:
        return FuturesSetup(False, "missing_futures_data", **base)

    latest_setup = latest_observed_futures_rth_session(
        data, entry_date=entry_day, as_of=as_of_stamp
    )
    if latest_setup is None:
        return FuturesSetup(False, "missing_setup_session", **base)
    if latest_setup != setup_day:
        return FuturesSetup(False, "intervening_futures_session", **base)

    local_index = _et_index(data.index)
    setup_minutes = data.loc[
        (local_index.date == setup_day) & _rth_mask(data.index)
    ]
    if setup_minutes.empty:
        return FuturesSetup(False, "missing_setup_session", **base)
    setup_local = _et_index(setup_minutes.index)
    if (
        setup_local[0].strftime("%H:%M") != "09:30"
        or setup_local[-1].strftime("%H:%M") != "15:59"
    ):
        return FuturesSetup(False, "incomplete_setup_session", **base)
    setup_ids = setup_minutes["instrument_id"].unique()
    if len(setup_ids) != 1:
        return FuturesSetup(False, "setup_contract_changed", **base)
    setup_instrument = int(setup_ids[0])
    bars = build_futures_rth15(data)
    if bars.empty:
        return FuturesSetup(False, "ema_not_ready", **base)
    day_bars = bars.loc[bars["session_date"].eq(setup_day)]
    expected = pd.date_range(
        pd.Timestamp(setup_day, tz=NY_TZ) + pd.Timedelta(hours=9, minutes=30),
        periods=26,
        freq="15min",
    )
    if len(day_bars) != 26 or not day_bars.index.equals(expected):
        return FuturesSetup(
            False,
            "incomplete_setup_bars",
            instrument_id=setup_instrument,
            rth_bar_count=len(day_bars),
            **base,
        )
    if day_bars["ema20"].isna().any():
        return FuturesSetup(
            False,
            "ema_not_ready",
            instrument_id=setup_instrument,
            rth_bar_count=26,
            **base,
        )

    day_open = float(setup_minutes.iloc[0]["open"])
    day_close = float(setup_minutes.iloc[-1]["close"])
    day_high = float(setup_minutes["high"].max())
    day_low = float(setup_minutes["low"].min())
    day_range = day_high - day_low
    if day_range <= 0:
        ratio = math.nan
        direction = 0
    else:
        ratio = abs(day_close - day_open) / day_range
        direction = 1 if day_close > day_open else -1 if day_close < day_open else 0
    common = {
        "instrument_id": setup_instrument,
        "trend_direction": direction,
        "trend_ratio": ratio,
        "rth_bar_count": 26,
        **base,
    }
    if direction == 0 or not math.isfinite(ratio):
        return FuturesSetup(False, "flat_setup_session", **common)
    if ratio < RULES.trend_ratio_min:
        return FuturesSetup(False, "trend_ratio_below_threshold", **common)

    if direction > 0:
        side_ok = bool((day_bars["low"] > day_bars["ema20"]).all())
    else:
        side_ok = bool((day_bars["high"] < day_bars["ema20"]).all())
    if not side_ok:
        return FuturesSetup(False, "ema_touch_or_wrong_side", **common)

    entry_local = _et_index(data.index)
    premarket = data.loc[
        (entry_local.date == entry_day)
        & (entry_local.time < RULES.rth_open)
    ]
    if premarket.empty:
        return FuturesSetup(False, "missing_entry_contract_probe", **common)
    latest_probe = premarket.index[-1]
    probe_age = as_of_stamp.tz_convert("UTC") - latest_probe
    if probe_age > pd.Timedelta(minutes=10):
        return FuturesSetup(False, "stale_entry_contract_probe", **common)
    entry_instrument = int(premarket.iloc[-1]["instrument_id"])
    if entry_instrument != setup_instrument:
        return FuturesSetup(
            False,
            "roll_crossing",
            entry_instrument_id=entry_instrument,
            **common,
        )
    return FuturesSetup(
        True,
        "qualified",
        entry_instrument_id=entry_instrument,
        **common,
    )


def round_limit(price: float, direction: int, tick: float = 0.01) -> float:
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    if not math.isfinite(price) or price <= 0 or tick <= 0:
        raise ValueError("price and tick must be positive and finite")
    scaled = price / tick
    ticks = (
        math.ceil(scaled - 1e-10)
        if direction > 0
        else math.floor(scaled + 1e-10)
    )
    return round(float(ticks * tick), 10)


def target_touched(row: pd.Series, target: float, direction: int) -> bool:
    return (
        float(row["high"]) >= target - 1e-10
        if direction > 0
        else float(row["low"]) <= target + 1e-10
    )


def limit_fill(row: pd.Series, target: float, direction: int) -> float | None:
    if direction > 0:
        if float(row["open"]) >= target - 1e-10:
            return float(row["open"])
        if float(row["high"]) >= target - 1e-10:
            return float(target)
    else:
        if float(row["open"]) <= target + 1e-10:
            return float(row["open"])
        if float(row["low"]) <= target + 1e-10:
            return float(target)
    return None


def entry_decision(
    row_0930: pd.Series,
    *,
    initial_ema: float,
    decision_price: float | None = None,
    ex_dividend: bool = False,
) -> EntryDecision:
    """Apply the ETF-native 09:30 geometry and opportunity gates."""

    if ex_dividend:
        return EntryDecision(False, "ex_dividend")
    if not math.isfinite(initial_ema) or initial_ema <= 0:
        return EntryDecision(False, "ema_not_ready")
    open_0930 = float(row_0930["open"])
    if math.isclose(open_0930, initial_ema, abs_tol=1e-7):
        return EntryDecision(
            False, "open_equals_ema", open_0930=open_0930
        )
    direction = -1 if open_0930 > initial_ema else 1
    side = "long" if direction > 0 else "short"
    target = round_limit(initial_ema, direction, RULES.penny)
    if target_touched(row_0930, target, direction):
        return EntryDecision(
            False,
            "opportunity_passed_in_0930_minute",
            direction,
            side,
            target,
            open_0930,
        )
    if decision_price is not None:
        through = (
            direction > 0 and decision_price >= target - 1e-10
        ) or (
            direction < 0 and decision_price <= target + 1e-10
        )
        if through:
            return EntryDecision(
                False,
                "decision_price_through_target",
                direction,
                side,
                target,
                open_0930,
                float(decision_price),
            )
    return EntryDecision(
        True,
        "eligible",
        direction,
        side,
        target,
        open_0930,
        None if decision_price is None else float(decision_price),
    )


def ema_from_seed(closes: pd.Series, initial_ema: float) -> pd.Series:
    alpha = 2.0 / (RULES.ema_span + 1.0)
    previous = float(initial_ema)
    values: list[float] = []
    for close in closes.astype(float):
        previous = alpha * float(close) + (1.0 - alpha) * previous
        values.append(previous)
    return pd.Series(values, index=closes.index, dtype=float)


def build_entry_day_revisions(
    minutes: pd.DataFrame,
    *,
    entry_date: date | str | pd.Timestamp,
    initial_ema: float,
) -> pd.DataFrame:
    """Return only complete 15-minute ETF bars and their causal activation."""

    data = normalize_minutes(minutes)
    data.index = _et_index(data.index)
    day = pd.Timestamp(entry_date).date()
    data = data.loc[(data.index.date == day) & _rth_mask(data.index)]
    bars = data.resample("15min", label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        minute_count=("close", "count"),
    )
    bars = bars.loc[bars["minute_count"].eq(15)].copy()
    bars["ema20"] = ema_from_seed(bars["close"], initial_ema)
    bars["activation_ts"] = bars.index + pd.Timedelta(minutes=16)
    return bars


def active_ema_series(
    minute_index: pd.DatetimeIndex,
    revisions: pd.DataFrame,
    initial_ema: float,
) -> pd.Series:
    result = pd.Series(float(initial_ema), index=minute_index, dtype=float)
    for _, row in revisions.iterrows():
        result.loc[result.index >= row["activation_ts"]] = float(row["ema20"])
    return result


def simulate_etf_trade(
    minutes: pd.DataFrame,
    *,
    entry_date: date | str | pd.Timestamp,
    initial_ema: float,
    ex_dividend: bool = False,
) -> dict[str, Any]:
    """Replay the production rule on a complete historical ETF entry day."""

    data = normalize_minutes(minutes)
    data.index = _et_index(data.index)
    day = pd.Timestamp(entry_date).date()
    start = pd.Timestamp(day, tz=NY_TZ) + pd.Timedelta(hours=9, minutes=30)
    end = pd.Timestamp(day, tz=NY_TZ) + pd.Timedelta(hours=16)
    session = data.loc[start : end - pd.Timedelta(minutes=1)]
    expected = pd.date_range(start, end - pd.Timedelta(minutes=1), freq="1min")
    if len(session) != 390 or not session.index.equals(expected):
        return {"traded": False, "skip_reason": "incomplete_etf_entry_session"}
    decision = entry_decision(
        session.loc[start], initial_ema=initial_ema, ex_dividend=ex_dividend
    )
    if not decision.eligible:
        return {"traded": False, "skip_reason": decision.reason, **asdict(decision)}
    entry_ts = start + pd.Timedelta(minutes=1)
    decision_price = float(session.loc[entry_ts, "open"])
    decision = entry_decision(
        session.loc[start],
        initial_ema=initial_ema,
        decision_price=decision_price,
        ex_dividend=ex_dividend,
    )
    if not decision.eligible:
        # Historical naming is retained for the golden parity fixture.
        reason = (
            "entry_open_through_target"
            if decision.reason == "decision_price_through_target"
            else decision.reason
        )
        return {"traded": False, "skip_reason": reason, **asdict(decision)}

    revisions = build_entry_day_revisions(
        session, entry_date=day, initial_ema=initial_ema
    )
    cutoff = start + pd.Timedelta(hours=1)
    scan = session.loc[entry_ts : cutoff - pd.Timedelta(minutes=1)]
    active = active_ema_series(scan.index, revisions, initial_ema)
    direction = int(decision.direction)
    exit_ts = cutoff
    exit_price = float(session.loc[cutoff, "open"])
    exit_reason = "time_stop"
    target_at_exit = math.nan
    for timestamp, row in scan.iterrows():
        target = round_limit(float(active.loc[timestamp]), direction, RULES.penny)
        fill = limit_fill(row, target, direction)
        if fill is not None:
            exit_ts = timestamp
            exit_price = fill
            exit_reason = "etf_ema_limit"
            target_at_exit = target
            break
    return {
        "traded": True,
        "skip_reason": "",
        "direction": direction,
        "side": decision.side,
        "initial_target": decision.initial_target,
        "entry_ts": entry_ts,
        "entry_price": decision_price,
        "exit_ts": exit_ts,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "target_at_exit": target_at_exit,
        "gross_return_bps": direction
        * (exit_price - decision_price)
        / decision_price
        * 10_000.0,
    }


def prior_wilder_atr14(
    daily: pd.DataFrame,
    *,
    as_of_date: date | str,
    expected_last_session: date | str | None = None,
) -> float:
    """Raw RTH daily Wilder ATR14 using only sessions before ``as_of_date``."""

    if not set(OHLC).issubset(daily.columns):
        raise ValueError("daily bars require open/high/low/close")
    data = daily.copy()
    index = pd.DatetimeIndex(pd.to_datetime(data.index))
    if index.tz is not None:
        index = index.tz_convert(NY_TZ).tz_localize(None)
    data.index = index.normalize()
    data = data.loc[data.index < pd.Timestamp(as_of_date).normalize()].sort_index()
    if data.index.has_duplicates:
        raise ValueError("daily bars contain duplicate sessions")
    numeric = data.loc[:, list(OHLC)].apply(pd.to_numeric, errors="coerce")
    if numeric.empty or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("daily OHLC contains missing or non-finite values")
    invalid = (
        (numeric["open"] <= 0)
        | (numeric["high"] <= 0)
        | (numeric["low"] <= 0)
        | (numeric["close"] <= 0)
        | (numeric["high"] < numeric[["open", "close"]].max(axis=1))
        | (numeric["low"] > numeric[["open", "close"]].min(axis=1))
        | (numeric["high"] < numeric["low"])
    )
    if invalid.any():
        raise ValueError("daily OHLC geometry is invalid")
    data.loc[:, list(OHLC)] = numeric
    if expected_last_session is not None:
        expected = pd.Timestamp(expected_last_session).normalize()
        actual = None if data.empty else data.index[-1]
        if actual != expected:
            raise ValueError(
                f"daily ETF history ends at {actual}, expected {expected}"
            )
        expected_index = session_labels(data.index[0], expected)
        missing = expected_index.difference(data.index)
        unexpected = data.index.difference(expected_index)
        if len(missing) or len(unexpected) or len(data) != len(expected_index):
            raise ValueError(
                "daily ETF history does not match the exact XNYS session grid "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )
    if len(data) < 252:
        raise ValueError("ATR14 requires at least 252 contiguous prior sessions")
    previous_close = data["close"].shift(1)
    true_range = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(
        alpha=1.0 / RULES.atr_period,
        adjust=False,
        min_periods=RULES.atr_period,
    ).mean()
    if atr.empty or not math.isfinite(float(atr.iloc[-1])):
        raise ValueError("ATR14 is not ready")
    return float(atr.iloc[-1])
