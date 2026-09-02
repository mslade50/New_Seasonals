"""Fresh completed-session daily-price enrichment for EP research.

The EP discovery feed is intentionally independent of the repository's price
cache.  This module fetches adjusted daily OHLCV directly from yfinance, then
computes the same strict prior-session metrics used by the historical census
and the read-only IBKR adapter.  It contains no broker or order surface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from trading_calendar import TRADING_DAY

from .schema import PremarketSnapshot, iso_utc


YFINANCE_DAILY_PRICE_BASIS = "YFINANCE_AUTO_ADJUST_TRUE_REPAIR_TRUE"
IBKR_DAILY_PRICE_BASIS = "IBKR_ADJUSTED_LAST"
VERIFIED_PRIOR_ATR_PRICE_BASES = frozenset(
    {YFINANCE_DAILY_PRICE_BASIS, IBKR_DAILY_PRICE_BASIS}
)


class DailyPriceError(ValueError):
    """Raised when completed-session daily data cannot pass the ATR checks."""

    _MESSAGES = {
        "FEWER_THAN_126_COMPLETED_BARS": "fewer than 126 completed daily bars",
        "DUPLICATE_COMPLETED_DAILY_BARS": "duplicate completed daily bars",
        "STALE_OR_INCOMPLETE_ATR_WINDOW": "stale or incomplete 15-bar ATR source window",
        "UNCLEAN_ATR_WINDOW": "unclean 15-bar ATR source window",
        "INVALID_ATR_WINDOW": "unclean 15-bar ATR source window",
        "MISSING_ATR_WINDOW_VALUES": "unclean 15-bar ATR source window",
        "NONPOSITIVE_ATR_WINDOW_VALUES": "unclean 15-bar ATR source window",
        "HIGH_BELOW_LOW_ATR_WINDOW": "unclean 15-bar ATR source window",
        "HALF_DOUBLE_ATR_WINDOW": "unclean 15-bar ATR source window",
        "EXTREME_ATR_WINDOW": "unclean 15-bar ATR source window",
        "PRIOR_ATR_UNRESOLVED": "unresolved prior-session ATR",
    }

    def __init__(self, code: str) -> None:
        self.code = str(code).split(":", 1)[0]
        super().__init__(self._MESSAGES.get(self.code, str(code)))


@dataclass(frozen=True)
class DailyEnrichmentError:
    symbol: str
    code: str
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        value = {"symbol": self.symbol, "code": self.code}
        if self.detail:
            value["detail"] = self.detail
        return value


@dataclass(frozen=True)
class DailyEnrichmentResult:
    snapshots: tuple[PremarketSnapshot, ...]
    errors: tuple[DailyEnrichmentError, ...]
    fetched_at: str
    requested_count: int
    verified_count: int


DownloadFunction = Callable[..., pd.DataFrame]


def _finite(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def yfinance_symbol(symbol: str) -> str:
    """Translate common US share-class notation without changing identity."""

    return str(symbol).strip().upper().replace(".", "-")


def _normalized_daily_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DailyPriceError("NO_DAILY_ROWS")
    work = frame.copy()
    if isinstance(work.columns, pd.MultiIndex):
        work.columns = work.columns.get_level_values(0)
    work.columns = [str(column).strip().lower() for column in work.columns]
    required = ("open", "high", "low", "close", "volume")
    missing = [column for column in required if column not in work.columns]
    if missing:
        raise DailyPriceError(f"MISSING_DAILY_COLUMNS:{','.join(missing)}")

    if "date" in work.columns:
        dates = pd.to_datetime(work.pop("date"), errors="coerce")
    else:
        dates = pd.to_datetime(work.index, errors="coerce")
    if isinstance(dates, pd.Series):
        date_values = dates.dt.date
    else:
        date_values = pd.Series(pd.DatetimeIndex(dates).date, index=work.index)
    normalized_values: dict[str, object] = {
        "date": list(date_values),
        **{
            column: pd.to_numeric(work[column], errors="coerce").to_numpy()
            for column in required
        },
    }
    repair_column = next(
        (column for column in work.columns if column.rstrip("?") == "repaired"),
        None,
    )
    if repair_column:
        normalized_values["repaired"] = work[repair_column].map(
            lambda value: (
                str(value).strip().lower() in {"true", "1", "yes"}
                if isinstance(value, str)
                else bool(value)
                if pd.notna(value)
                else False
            )
        ).to_numpy()
    else:
        normalized_values["repaired"] = [False] * len(work)
    normalized = pd.DataFrame(normalized_values)
    normalized = normalized.dropna(subset=["date"])
    return normalized


def calculate_prior_daily_metrics(
    frame: pd.DataFrame, session_date: date
) -> dict[str, float | int | None | str]:
    """Calculate strict point-in-time metrics through the prior NYSE session.

    ATR is the simple mean of the latest fourteen true ranges.  Fifteen
    consecutive completed bars are therefore required, and the event session
    can never enter the calculation.
    """

    work = _normalized_daily_frame(frame)
    work = (
        work[work["date"] < session_date]
        .sort_values("date")
        .reset_index(drop=True)
    )
    if len(work) < 126:
        raise DailyPriceError("FEWER_THAN_126_COMPLETED_BARS")
    if work["date"].duplicated().any():
        raise DailyPriceError("DUPLICATE_COMPLETED_DAILY_BARS")

    expected_previous_session = (pd.Timestamp(session_date) - TRADING_DAY).date()
    expected_atr_dates = list(
        pd.date_range(
            end=expected_previous_session,
            periods=15,
            freq=TRADING_DAY,
        ).date
    )
    if work.tail(15)["date"].tolist() != expected_atr_dates:
        raise DailyPriceError("STALE_OR_INCOMPLETE_ATR_WINDOW")

    previous_close = float(work.iloc[-1]["close"])
    missing_values = work[["open", "high", "low", "close", "volume"]].isna().any(
        axis=1
    )
    nonpositive = work[["open", "high", "low", "close"]].le(0).any(
        axis=1
    ) | work["volume"].le(0)
    high_below_low = work["high"].lt(work["low"])
    close_ratio = work["close"] / work["close"].shift(1)
    half_double = close_ratio.between(0.45, 0.55) | close_ratio.between(1.80, 2.20)
    gap_pct = 100.0 * (work["open"] / work["close"].shift(1) - 1.0)
    close_change_pct = 100.0 * (close_ratio - 1.0)
    extreme = gap_pct.abs().ge(50) | close_change_pct.abs().ge(50)
    if missing_values.tail(15).any():
        raise DailyPriceError("MISSING_ATR_WINDOW_VALUES")
    if nonpositive.tail(15).any():
        raise DailyPriceError("NONPOSITIVE_ATR_WINDOW_VALUES")
    if high_below_low.tail(15).any():
        raise DailyPriceError("HIGH_BELOW_LOW_ATR_WINDOW")
    if half_double.tail(15).any():
        raise DailyPriceError("HALF_DOUBLE_ATR_WINDOW")
    if extreme.tail(15).any():
        raise DailyPriceError("EXTREME_ATR_WINDOW")

    previous = work["close"].shift(1)
    true_range = pd.concat(
        [
            work["high"] - work["low"],
            (work["high"] - previous).abs(),
            (work["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=False)
    atr_14 = float(true_range.tail(14).mean())
    if not math.isfinite(atr_14) or atr_14 <= 0 or previous_close <= 0:
        raise DailyPriceError("PRIOR_ATR_UNRESOLVED")

    daily_change = 100.0 * (work["close"] / work["close"].shift(1) - 1.0)
    prior_avg_volume_100 = work["volume"].shift(1).rolling(100, min_periods=50).mean()
    prior_ep = (daily_change.abs() >= 8.0) & (
        work["volume"] >= 3.0 * prior_avg_volume_100
    )
    prior_ep_positions = work.index[prior_ep]
    sessions_since_prior_ep = (
        int((len(work) - 1) - prior_ep_positions[-1])
        if len(prior_ep_positions)
        else None
    )
    return {
        "previous_close": previous_close,
        "prior_two_day_low": float(work.tail(2)["low"].min()),
        "atr_14": atr_14,
        "avg_volume_20": float(work["volume"].tail(20).mean()),
        "addv_63": float((work["close"] * work["volume"]).tail(63).mean()),
        "prior_63d_return_pct": 100.0
        * (previous_close / float(work.iloc[-64]["close"]) - 1.0),
        "sessions_since_prior_ep": sessions_since_prior_ep,
        "daily_source_session": expected_previous_session.isoformat(),
        "daily_repaired_bar_count": int(work.tail(15)["repaired"].sum()),
    }


def extract_yfinance_symbol_frame(
    raw: pd.DataFrame, provider_symbol: str
) -> pd.DataFrame:
    """Extract one ticker from yfinance's mandatory multi-ticker shape."""

    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise DailyPriceError("NO_BATCH_DATA")
    frame = raw
    if isinstance(raw.columns, pd.MultiIndex):
        wanted = provider_symbol.upper()
        level: int | str | None = None
        if "Ticker" in raw.columns.names:
            level = "Ticker"
        else:
            for index in range(raw.columns.nlevels):
                values = {
                    str(value).upper()
                    for value in raw.columns.get_level_values(index).unique()
                }
                if wanted in values:
                    level = index
                    break
        if level is None:
            raise DailyPriceError("TICKER_LEVEL_NOT_FOUND")
        try:
            frame = raw.xs(provider_symbol, level=level, axis=1)
        except KeyError:
            matches = [
                value
                for value in raw.columns.get_level_values(level).unique()
                if str(value).upper() == wanted
            ]
            if len(matches) != 1:
                raise DailyPriceError("TICKER_NOT_FOUND_IN_BATCH")
            frame = raw.xs(matches[0], level=level, axis=1)
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.copy()
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.copy()
    frame.columns = [str(column).strip().title() for column in frame.columns]
    if frame.dropna(how="all").empty:
        raise DailyPriceError("NO_SYMBOL_DATA")
    return frame


def _download_kwargs(
    symbols: list[str], *, session_date: date, timeout_seconds: float
) -> dict[str, Any]:
    return {
        "tickers": symbols,
        "start": session_date - timedelta(days=550),
        # yfinance's end is exclusive.  This prevents a partial event-day bar
        # from entering the source frame; the metric function also filters it.
        "end": session_date,
        "interval": "1d",
        "auto_adjust": True,
        "actions": False,
        "prepost": False,
        "repair": True,
        "keepna": True,
        "group_by": "column",
        "multi_level_index": True,
        "threads": True,
        "progress": False,
        "timeout": timeout_seconds,
    }


def _safe_error_detail(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    return text[:240]


def enrich_snapshots_from_yfinance(
    snapshots: Iterable[PremarketSnapshot],
    *,
    session_date: date,
    download: DownloadFunction | None = None,
    batch_size: int = 50,
    batch_retries: int = 1,
    timeout_seconds: float = 15.0,
    fetched_at: str | datetime | None = None,
    provider_cache_dir: str | Path | None = None,
) -> DailyEnrichmentResult:
    """Fetch adjusted daily bars and retain unresolved rows for audit/email.

    Provider or symbol failures are fail-soft at the research layer: affected
    candidates keep their TradingView discovery fields, but their ATR remains
    unresolved and they cannot consume news budget or create a sizing preview.
    """

    rows = list(snapshots)
    if batch_size < 1 or batch_size > 200:
        raise ValueError("batch_size must be between 1 and 200")
    if batch_retries < 0 or batch_retries > 3:
        raise ValueError("batch_retries must be between 0 and 3")
    if timeout_seconds <= 0 or timeout_seconds > 60:
        raise ValueError("timeout_seconds must be in (0, 60]")
    symbols = [snapshot.symbol for snapshot in rows]
    if len(symbols) != len(set(symbols)):
        raise ValueError("yfinance enrichment requires unique snapshot symbols")

    provider_by_symbol = {symbol: yfinance_symbol(symbol) for symbol in symbols}
    inverse: dict[str, list[str]] = {}
    for symbol, provider_symbol in provider_by_symbol.items():
        inverse.setdefault(provider_symbol, []).append(symbol)
    collisions = {key: value for key, value in inverse.items() if len(value) > 1}
    if collisions:
        raise ValueError(f"ambiguous yfinance symbol mapping: {collisions}")

    if download is None:
        import yfinance as yf

        if provider_cache_dir is not None:
            cache_path = Path(provider_cache_dir).resolve()
            cache_path.mkdir(parents=True, exist_ok=True)
            # yfinance requires writable SQLite cookie/timezone metadata.  This
            # does not cache OHLCV and is never a substitute for the live pull.
            yf.set_tz_cache_location(str(cache_path))
        download = yf.download

    observed = iso_utc(fetched_at or datetime.now(timezone.utc))
    metrics_by_symbol: dict[str, dict[str, float | int | None | str]] = {}
    errors_by_symbol: dict[str, DailyEnrichmentError] = {}
    for start in range(0, len(symbols), batch_size):
        batch = symbols[start : start + batch_size]
        provider_symbols = [provider_by_symbol[symbol] for symbol in batch]
        raw: pd.DataFrame | None = None
        last_exc: BaseException | None = None
        for _attempt in range(batch_retries + 1):
            try:
                raw = download(
                    **_download_kwargs(
                        provider_symbols,
                        session_date=session_date,
                        timeout_seconds=timeout_seconds,
                    )
                )
                break
            except Exception as exc:  # noqa: BLE001 - isolate provider failures.
                last_exc = exc
        if raw is None:
            detail = _safe_error_detail(last_exc or RuntimeError("download failed"))
            for symbol in batch:
                errors_by_symbol[symbol] = DailyEnrichmentError(
                    symbol=symbol,
                    code="YFINANCE_DOWNLOAD_FAILED",
                    detail=detail,
                )
            continue

        for symbol in batch:
            try:
                frame = extract_yfinance_symbol_frame(
                    raw, provider_by_symbol[symbol]
                )
                metrics_by_symbol[symbol] = calculate_prior_daily_metrics(
                    frame, session_date
                )
            except DailyPriceError as exc:
                errors_by_symbol[symbol] = DailyEnrichmentError(
                    symbol=symbol,
                    code=exc.code,
                    detail=_safe_error_detail(exc),
                )
            except Exception as exc:  # noqa: BLE001 - preserve sibling symbols.
                errors_by_symbol[symbol] = DailyEnrichmentError(
                    symbol=symbol,
                    code="YFINANCE_SYMBOL_PROCESSING_FAILED",
                    detail=_safe_error_detail(exc),
                )

    # yfinance can omit one name from an otherwise valid large response.  Retry
    # only provider/shape misses in smaller chunks; bad or stale price history
    # is deterministic and must remain failed rather than being papered over.
    retriable_codes = {
        "YFINANCE_DOWNLOAD_FAILED",
        "NO_BATCH_DATA",
        "TICKER_LEVEL_NOT_FOUND",
        "TICKER_NOT_FOUND_IN_BATCH",
        "NO_SYMBOL_DATA",
        "YFINANCE_SYMBOL_PROCESSING_FAILED",
    }
    retry_symbols = [
        symbol
        for symbol in symbols
        if errors_by_symbol.get(symbol)
        and errors_by_symbol[symbol].code in retriable_codes
    ]
    retry_batch_size = min(10, batch_size)
    for start in range(0, len(retry_symbols), retry_batch_size):
        batch = retry_symbols[start : start + retry_batch_size]
        provider_symbols = [provider_by_symbol[symbol] for symbol in batch]
        try:
            raw = download(
                **_download_kwargs(
                    provider_symbols,
                    session_date=session_date,
                    timeout_seconds=timeout_seconds,
                )
            )
        except Exception as exc:  # noqa: BLE001 - retain initial row errors.
            detail = _safe_error_detail(exc)
            for symbol in batch:
                errors_by_symbol[symbol] = DailyEnrichmentError(
                    symbol=symbol,
                    code="YFINANCE_DOWNLOAD_FAILED",
                    detail=detail,
                )
            continue
        for symbol in batch:
            try:
                frame = extract_yfinance_symbol_frame(
                    raw, provider_by_symbol[symbol]
                )
                metrics_by_symbol[symbol] = calculate_prior_daily_metrics(
                    frame, session_date
                )
                errors_by_symbol.pop(symbol, None)
            except DailyPriceError as exc:
                errors_by_symbol[symbol] = DailyEnrichmentError(
                    symbol=symbol,
                    code=exc.code,
                    detail=_safe_error_detail(exc),
                )
            except Exception as exc:  # noqa: BLE001 - preserve sibling symbols.
                errors_by_symbol[symbol] = DailyEnrichmentError(
                    symbol=symbol,
                    code="YFINANCE_SYMBOL_PROCESSING_FAILED",
                    detail=_safe_error_detail(exc),
                )

    enriched: list[PremarketSnapshot] = []
    for snapshot in rows:
        metrics = metrics_by_symbol.get(snapshot.symbol)
        if metrics is None:
            error = errors_by_symbol.get(snapshot.symbol)
            enriched.append(
                replace(
                    snapshot,
                    atr_14=0.0,
                    atr_reference_close=None,
                    daily_price_basis="UNVERIFIED",
                    daily_data_status=(error.code if error else "YFINANCE_UNRESOLVED"),
                    daily_data_observed_at=observed,
                    daily_source_session="",
                    daily_source_symbol=provider_by_symbol[snapshot.symbol],
                )
            )
            continue
        enriched.append(
            replace(
                snapshot,
                atr_14=float(metrics["atr_14"]),
                atr_reference_close=float(metrics["previous_close"]),
                prior_two_day_low=float(metrics["prior_two_day_low"]),
                avg_volume_20=float(metrics["avg_volume_20"]),
                addv_63=float(metrics["addv_63"]),
                prior_63d_return_pct=float(metrics["prior_63d_return_pct"]),
                sessions_since_prior_ep=(
                    int(metrics["sessions_since_prior_ep"])
                    if metrics["sessions_since_prior_ep"] is not None
                    else None
                ),
                daily_price_basis=YFINANCE_DAILY_PRICE_BASIS,
                daily_data_status=(
                    "VERIFIED_WITH_YFINANCE_REPAIR"
                    if int(metrics["daily_repaired_bar_count"])
                    else "VERIFIED"
                ),
                daily_data_observed_at=observed,
                daily_source_session=str(metrics["daily_source_session"]),
                daily_source_symbol=provider_by_symbol[snapshot.symbol],
                daily_repaired_bar_count=int(metrics["daily_repaired_bar_count"]),
            )
        )

    return DailyEnrichmentResult(
        snapshots=tuple(enriched),
        errors=tuple(errors_by_symbol[symbol] for symbol in sorted(errors_by_symbol)),
        fetched_at=observed,
        requested_count=len(rows),
        verified_count=len(metrics_by_symbol),
    )
