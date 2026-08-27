"""Build the pre-market Discretionary Focus research shortlist.

This is a reproducible daily-bar approximation of the TradingView ``Armed``
screen over the maintained R2 common-stock universe.
It identifies high-volatility stocks compressing near a breakout, excludes the
standard lane around earnings, and then requires current fundamental or news
support before scarce attention is allocated.  It emits zero to two names.

The output is research priority only.  It contains no quantity, side, order,
allocation, or frozen executable price.  Intraday confirmation stays in
TradingView, where raw live prices and relative volume share one basis.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import exchange_calendars as xcals

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from discretionary_focus.contracts import validate_payload  # noqa: E402
from discretionary_focus.selector import select_focus  # noqa: E402
from earnings_filter import load_earnings_dates_map  # noqa: E402
from fundamental.metrics import calculate_ticker_metrics  # noqa: E402
from trading_calendar import NYSE_HOLIDAYS  # noqa: E402


ET = ZoneInfo("America/New_York")
SCHEMA_VERSION = "discretionary-focus.v1"

DEFAULT_PRICES = ROOT / "data" / "master_prices.parquet"
DEFAULT_OVERFLOW_PRICES = ROOT / "data" / "overflow_prices.parquet"
DEFAULT_EARNINGS = ROOT / "data" / "earnings_calendar.parquet"
DEFAULT_EARNINGS_OVERFLOW = ROOT / "data" / "earnings_calendar_overflow.parquet"
DEFAULT_SYMBOLS = ROOT / "data" / "symbol_master.parquet"
DEFAULT_FUNDAMENTALS = (
    ROOT / "data" / "fundamental" / "current" / "daily_report_latest.json"
)
DEFAULT_OUTPUT = ROOT / "data" / "discretionary_focus" / "current.json"
DEFAULT_SCREEN_MANIFEST = ROOT / "discretionary_focus" / "tradingview_screens.json"

MIN_PRICE = 1.0
MIN_MARKET_CAP = 150_000_000.0
MAX_MARKET_CAP = 25_000_000_000.0
MIN_AVG_VOLUME_60 = 800_000.0
MIN_LATEST_VOLUME = 500_000.0
MAX_FLOAT_SHARES = 200_000_000.0
MIN_ADR_14_PCT = 4.0
MIN_HISTORY = 70
MIN_PERFORMANCE_1M_PCT = 30.0
MIN_PERFORMANCE_3M_PCT = 30.0
MIN_PERFORMANCE_1W_PCT = -5.0
MAX_PERFORMANCE_1W_PCT = 5.0
MIN_VOLATILITY_1W_PCT = 4.0
MIN_RELATIVE_VOLUME = 1.0
EARNINGS_BLACKOUT_TD = 5
MAX_ENRICHMENT_CANDIDATES = 20
MAX_ENRICHMENT_WORKERS = 6
MAX_FUNDAMENTAL_AGE_DAYS = 45
MAX_NEWS_AGE_DAYS = 14
MAX_ANNUAL_STATEMENT_AGE_DAYS = 550
MAX_SEC_FILING_AGE_DAYS = 200
MIN_PRODUCTION_UNIVERSE = 500
MIN_SYMBOL_PRICE_OVERLAP = 0.85
MIN_PRODUCTION_EARNINGS_COVERAGE = 500
MIN_SETUP_QUALITY = 60.0
MIN_PIVOT_DISTANCE_PCT = -2.0
MAX_PIVOT_DISTANCE_PCT = 8.0
MAX_COMPRESSION_RATIO = 1.0
MIN_CLOSE_LOCATION_20 = 0.65
MAX_VOLUME_RATIO_5_20 = 1.35
ALLOWED_EXCHANGES = {"NASDAQ", "NYSE", "AMEX"}

FMP_BASE = "https://financialmodelingprep.com/stable"
FMP_DOCS = "https://site.financialmodelingprep.com/developer/docs"
TRADINGVIEW_ARMED_URL = "https://www.tradingview.com/screener/FzMHioHX/"
TRADINGVIEW_LIVE_URL = "https://www.tradingview.com/screener/60i0utaT/"
TRADINGVIEW_PERFORMANCE_DOC = (
    "https://www.tradingview.com/support/solutions/43000636536-how-is-performance-calculated-in-the-screener/"
)
TRADINGVIEW_VOLATILITY_DOC = (
    "https://www.tradingview.com/support/solutions/43000635876-how-is-volatility-calculated-in-the-screener/"
)
TRADINGVIEW_ADR_DOC = (
    "https://www.tradingview.com/support/solutions/43000734653-how-are-adr-and-atr-calculated/"
)
NEWS_ENDPOINTS = ("news/press-releases", "news/stock")
FINANCIAL_ENDPOINTS = (
    "income-statement",
    "balance-sheet-statement",
    "cash-flow-statement",
)

POSITIVE_NEWS_TERMS = (
    "raises guidance", "raised guidance", "increases guidance", "increased guidance",
    "raises outlook", "raised outlook", "increases outlook", "increased outlook",
    "beats and raises", "record revenue", "record backlog", "contract award",
    "awarded a contract", "wins contract", "won contract",
)
NEGATIVE_NEWS_TERMS = (
    "public offering", "registered direct", "at-the-market", "shelf registration",
    "dilution", "restatement", "investigation", "subpoena", "delisting",
    "bankruptcy", "chapter 11", "going concern", "reverse split", "misses estimates",
    "cuts guidance", "lowered guidance",
)

SPECIALIST_SECTOR_LANES = {
    "financial services": "financials_specialist",
    "financials": "financials_specialist",
    "real estate": "real_estate_specialist",
}
SPECIALIST_INDUSTRY_LANES = {
    "biotechnology": "biotech_pipeline_specialist",
    "shell companies": "special_situation",
    "blank checks": "special_situation",
}


class FocusBuildError(RuntimeError):
    """The run could not establish a fresh, valid focus result."""


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _clean_text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return fallback if not text or text.lower() in {"nan", "none", "null"} else text


def _pct(value: float | None, digits: int = 1) -> float | None:
    return None if value is None else round(float(value), digits)


def _safe_url(value: Any) -> str:
    text = str(value or "").strip()
    try:
        parsed = urlparse(text)
    except ValueError:
        return ""
    return text if parsed.scheme in {"http", "https"} and parsed.netloc else ""


def _research_lane(sector: Any, industry: Any) -> str:
    sector_text = _clean_text(sector).lower()
    if sector_text in SPECIALIST_SECTOR_LANES:
        return SPECIALIST_SECTOR_LANES[sector_text]
    industry_text = _clean_text(industry).lower()
    if sector_text == "healthcare" and industry_text in {"", "unknown"}:
        # Production symbol_master currently lacks industry.  Do not let an
        # unidentified biotech slip into the generic-company underwriter.
        return "healthcare_specialist"
    for term, lane in SPECIALIST_INDUSTRY_LANES.items():
        if term in industry_text:
            return lane
    return "standard_company"


def _causal_cluster(industry: Any, sector: Any) -> str:
    industry_text = _clean_text(industry)
    if industry_text and industry_text.lower() != "unknown":
        return industry_text
    return _clean_text(sector, "Unknown")


def load_screen_manifest(path: Path = DEFAULT_SCREEN_MANIFEST) -> dict[str, Any]:
    """Load and pin the saved-screen contract used by the cloud mirror."""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FocusBuildError(f"TradingView screen manifest is unreadable: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "tradingview-focus-screens.v1":
        raise FocusBuildError("TradingView screen manifest schema is invalid")
    armed = manifest.get("armed") if isinstance(manifest.get("armed"), dict) else {}
    live = manifest.get("live") if isinstance(manifest.get("live"), dict) else {}
    if _safe_url(armed.get("url")) != TRADINGVIEW_ARMED_URL:
        raise FocusBuildError("TradingView Armed URL drifted from the pinned screen")
    if _safe_url(live.get("url")) != TRADINGVIEW_LIVE_URL:
        raise FocusBuildError("TradingView Live URL drifted from the pinned screen")
    expected = {
        "price_min": MIN_PRICE,
        "market_cap_min": MIN_MARKET_CAP,
        "market_cap_max": MAX_MARKET_CAP,
        "performance_1m_min_pct": MIN_PERFORMANCE_1M_PCT,
        "performance_3m_min_pct": MIN_PERFORMANCE_3M_PCT,
        "performance_1w_min_pct": MIN_PERFORMANCE_1W_PCT,
        "performance_1w_max_pct": MAX_PERFORMANCE_1W_PCT,
        "average_volume_60_min": MIN_AVG_VOLUME_60,
        "latest_volume_min": MIN_LATEST_VOLUME,
        "float_shares_max": MAX_FLOAT_SHARES,
        "adr_14_min_pct": MIN_ADR_14_PCT,
        "volatility_1w_min_pct": MIN_VOLATILITY_1W_PCT,
        "relative_volume_min": MIN_RELATIVE_VOLUME,
    }
    filters = armed.get("filters") if isinstance(armed.get("filters"), dict) else {}
    mismatches = [
        key for key, expected_value in expected.items()
        if _number(filters.get(key)) != float(expected_value)
    ]
    if mismatches:
        raise FocusBuildError(
            "TradingView manifest and cloud mirror disagree on: "
            + ", ".join(sorted(mismatches))
        )
    for flag in ("price_above_sma20", "price_above_sma50"):
        if filters.get(flag) is not True:
            raise FocusBuildError(f"TradingView Armed manifest requires {flag}=true")
    if filters.get("security_type") != "common_stock" or filters.get("primary_listing_only") is not True:
        raise FocusBuildError("TradingView Armed manifest must remain primary common stock")
    if set(filters.get("exchanges") or []) != ALLOWED_EXCHANGES:
        raise FocusBuildError("TradingView Armed exchange set drifted")
    trim = armed.get("cloud_attention_trim")
    expected_trim = {
        "pivot_distance_min_pct": MIN_PIVOT_DISTANCE_PCT,
        "pivot_distance_max_pct": MAX_PIVOT_DISTANCE_PCT,
        "compression_ratio_max": MAX_COMPRESSION_RATIO,
        "close_location_20_min": MIN_CLOSE_LOCATION_20,
        "volume_ratio_5_20_max": MAX_VOLUME_RATIO_5_20,
        "setup_quality_min": MIN_SETUP_QUALITY,
    }
    if not isinstance(trim, dict) or any(
        _number(trim.get(key)) != float(expected_value)
        for key, expected_value in expected_trim.items()
    ):
        raise FocusBuildError("TradingView manifest cloud-attention trim drifted")
    live_filters = live.get("filters") if isinstance(live.get("filters"), dict) else {}
    expected_live = {
        "performance_1w_min_pct": -10.0,
        "performance_1w_max_pct": 10.0,
        "daily_change_min_pct": 2.0,
        "relative_volume_at_time_min": 2.0,
        "change_from_open_min_pct": 0.0,
    }
    if any(
        _number(live_filters.get(key)) != expected_value
        for key, expected_value in expected_live.items()
    ) or live_filters.get("new_high_window") != "1_month":
        raise FocusBuildError("TradingView Live manifest drifted")
    return manifest


def validate_production_input_coverage(
    prices: pd.DataFrame,
    symbols: dict[str, dict[str, Any]],
    *,
    required_session: dt.date,
) -> None:
    """Refuse a narrow, stale, or mismatched universe before false zero."""
    price_tickers = {
        str(value).upper().strip() for value in prices.get("ticker", []) if str(value).strip()
    }
    symbol_tickers = set(symbols)
    if len(price_tickers) < MIN_PRODUCTION_UNIVERSE:
        raise FocusBuildError(
            f"price universe has {len(price_tickers)} tickers; minimum is {MIN_PRODUCTION_UNIVERSE}"
        )
    if len(symbol_tickers) < MIN_PRODUCTION_UNIVERSE:
        raise FocusBuildError(
            f"symbol universe has {len(symbol_tickers)} tickers; minimum is {MIN_PRODUCTION_UNIVERSE}"
        )
    if "date" not in prices:
        raise FocusBuildError("price universe lacks date coverage")
    price_dates = pd.to_datetime(prices["date"], errors="coerce").dt.date
    current_price_tickers = {
        str(value).upper().strip()
        for value in prices.loc[price_dates.eq(required_session), "ticker"]
        if str(value).strip()
    }
    covered_symbols = current_price_tickers & symbol_tickers
    coverage = len(covered_symbols) / len(symbol_tickers)
    if coverage < MIN_SYMBOL_PRICE_OVERLAP:
        raise FocusBuildError(
            f"eligible symbol price coverage for {required_session} is "
            f"{coverage:.1%} ({len(covered_symbols)}/{len(symbol_tickers)}); "
            f"minimum is {MIN_SYMBOL_PRICE_OVERLAP:.0%}"
        )


def validate_future_earnings_coverage(counts: dict[str, int]) -> None:
    covered = int(counts.get("future_earnings_covered", 0))
    if covered < MIN_PRODUCTION_EARNINGS_COVERAGE:
        raise FocusBuildError(
            f"only {covered} current measurable tickers have a future earnings "
            f"event; minimum is {MIN_PRODUCTION_EARNINGS_COVERAGE}"
        )


def _date(value: Any) -> dt.date | None:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed.date()


def _json_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        default=str,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _next_session(after: dt.date) -> dt.date:
    holidays = np.asarray(pd.DatetimeIndex(NYSE_HOLIDAYS), dtype="datetime64[D]")
    value = np.busday_offset(
        np.datetime64(after, "D"), 1, roll="forward", holidays=holidays
    )
    return pd.Timestamp(value).date()


def session_for_run(price_as_of: dt.date, now: dt.datetime) -> dt.date:
    """Return today's ET session when possible, else the next NYSE session."""
    local = now.astimezone(ET)
    today = local.date()
    holidays = {stamp.date() for stamp in pd.DatetimeIndex(NYSE_HOLIDAYS)}
    is_session = today.weekday() < 5 and today not in holidays
    if is_session and price_as_of < today:
        return today
    return _next_session(price_as_of)


def _is_nyse_session(day: dt.date) -> bool:
    holidays = {stamp.date() for stamp in pd.DatetimeIndex(NYSE_HOLIDAYS)}
    return day.weekday() < 5 and day not in holidays


def expected_prior_session(valid_for: dt.date) -> dt.date:
    """Return the sole daily-bar cutoff acceptable for a pre-market list."""
    if not _is_nyse_session(valid_for):
        raise FocusBuildError(f"valid_for is not an NYSE session: {valid_for}")
    holidays = np.asarray(pd.DatetimeIndex(NYSE_HOLIDAYS), dtype="datetime64[D]")
    prior = np.busday_offset(
        np.datetime64(valid_for, "D"), -1, roll="raise", holidays=holidays
    )
    return pd.Timestamp(prior).date()


def session_close(day: dt.date) -> dt.datetime:
    """Return the exchange-published XNYS close, including half sessions."""
    try:
        close = xcals.get_calendar("XNYS").session_close(pd.Timestamp(day))
    except Exception as exc:  # noqa: BLE001 - a calendar gap must fail closed
        raise FocusBuildError(f"XNYS session close unavailable for {day}: {exc}") from exc
    return close.to_pydatetime().astimezone(ET)


def session_expiry(day: dt.date) -> dt.datetime:
    return session_close(day) + dt.timedelta(minutes=15)


def require_fresh_price_cutoff(price_as_of: dt.date, valid_for: dt.date) -> None:
    expected = expected_prior_session(valid_for)
    if price_as_of != expected:
        raise FocusBuildError(
            "price cache is not the immediately completed session required for "
            f"{valid_for}: expected {expected}, found {price_as_of}"
        )


def load_price_data(
    path: Path,
    as_of: str | None = None,
    *,
    minimum_session_tickers: int = 1,
) -> tuple[pd.DataFrame, dt.date]:
    if not path.is_file():
        raise FocusBuildError(f"price cache is missing: {path}")
    required = ["ticker", "date", "Open", "High", "Low", "Close", "Volume"]
    try:
        frame = pd.read_parquet(path, columns=required)
    except Exception as exc:  # noqa: BLE001
        raise FocusBuildError(f"price cache is unreadable: {exc}") from exc
    if frame.empty:
        raise FocusBuildError("price cache is empty")
    frame = frame.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.tz_localize(None)
    frame = frame.dropna(subset=["ticker", "date"])
    requested_cutoff = (
        pd.Timestamp(as_of).normalize() if as_of else frame["date"].max().normalize()
    )
    frame = frame[frame["date"].le(requested_cutoff)]
    if frame.empty:
        raise FocusBuildError(
            f"price cache has no rows on or before {requested_cutoff.date()}"
        )
    daily_coverage = frame.groupby("date")["ticker"].nunique()
    complete_dates = daily_coverage[daily_coverage.ge(minimum_session_tickers)]
    if complete_dates.empty:
        raise FocusBuildError(
            "price cache has no broadly covered session on or before "
            f"{requested_cutoff.date()}; minimum tickers is {minimum_session_tickers}"
        )
    actual_stamp = pd.Timestamp(complete_dates.index.max()).normalize()
    # A few foreign-index/FX/crypto bars can be stamped with today's date
    # before the US session. Never let that thin partial cohort redefine the
    # completed daily-bar cutoff for the common-stock screen.
    frame = frame[frame["date"].le(actual_stamp)]
    actual = actual_stamp.date()
    return frame.sort_values(["ticker", "date"]), actual


def combine_price_data(primary: pd.DataFrame, overflow: pd.DataFrame) -> pd.DataFrame:
    """Union canonical and isolated histories, with canonical rows winning."""
    combined = pd.concat([overflow, primary], ignore_index=True)
    return (
        combined.drop_duplicates(["ticker", "date"], keep="last")
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )


def load_symbols(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FocusBuildError(f"symbol master is missing: {path}")
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        raise FocusBuildError(f"symbol master is unreadable: {exc}") from exc
    required = {"ticker", "company_name", "exchange"}
    missing = required - set(frame.columns)
    if missing:
        raise FocusBuildError(f"symbol master lacks columns: {', '.join(sorted(missing))}")
    frame = frame.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    return {
        row["ticker"]: row.to_dict()
        for _, row in frame.drop_duplicates("ticker", keep="last").iterrows()
    }


def _performance_from_anchor_open(
    bars: pd.DataFrame,
    *,
    close: float,
    calendar_days: int,
    evaluation_date: dt.date | None = None,
) -> float | None:
    """Mirror TradingView performance from the dated anchor bar's open."""
    anchor_date = evaluation_date or bars["date"].iloc[-1].date()
    cutoff = pd.Timestamp(anchor_date) - pd.Timedelta(days=calendar_days)
    anchors = bars[bars["date"].le(cutoff)]
    if anchors.empty:
        return None
    anchor_open = _number(anchors["Open"].iloc[-1])
    if anchor_open is None or anchor_open == 0:
        return None
    return (close - anchor_open) / abs(anchor_open) * 100.0


def _wilder_rma(values: pd.Series, length: int) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(clean) < length:
        return None
    average = float(clean.iloc[:length].mean())
    for value in clean.iloc[length:]:
        average = (average * (length - 1) + float(value)) / length
    return average if math.isfinite(average) else None


def _ticker_metrics(
    group: pd.DataFrame,
    as_of: dt.date,
    *,
    evaluation_date: dt.date | None = None,
) -> dict[str, Any] | None:
    bars = group[group["date"].dt.date.le(as_of)].tail(260).copy()
    bars = bars.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    if len(bars) < MIN_HISTORY or bars["date"].iloc[-1].date() != as_of:
        return None
    for column in ("Open", "High", "Low", "Close", "Volume"):
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    bars = bars.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    if len(bars) < MIN_HISTORY:
        return None

    close = float(bars["Close"].iloc[-1])
    if close <= 0:
        return None
    previous = bars["Close"].shift(1)
    daily_range_pct = (
        (bars["High"] - bars["Low"]) / bars["Low"].abs().replace(0, np.nan) * 100.0
    )
    true_range = pd.concat(
        [
            bars["High"] - bars["Low"],
            (bars["High"] - previous).abs(),
            (bars["Low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # TradingView ADR% = (SMA(high, 14) - SMA(low, 14)) / current close.
    adr14 = _number(
        (bars["High"].tail(14).mean() - bars["Low"].tail(14).mean())
        / close
        * 100.0
    )
    atr14_value = _wilder_rma(true_range, 14)
    atr14_pct = _number(
        atr14_value / close * 100.0 if atr14_value is not None else None
    )
    adv20 = _number((bars["Close"] * bars["Volume"]).tail(20).mean())
    average_volume60 = _number(bars["Volume"].tail(60).mean())
    latest_volume = _number(bars["Volume"].iloc[-1])
    average_volume10 = _number(bars["Volume"].iloc[-11:-1].mean())
    relative_volume = (
        float(latest_volume) / float(average_volume10)
        if latest_volume is not None and average_volume10 and average_volume10 > 0
        else None
    )
    performance1w = _performance_from_anchor_open(
        bars, close=close, calendar_days=7, evaluation_date=evaluation_date
    )
    performance1m = _performance_from_anchor_open(
        bars, close=close, calendar_days=30, evaluation_date=evaluation_date
    )
    performance3m = _performance_from_anchor_open(
        bars, close=close, calendar_days=90, evaluation_date=evaluation_date
    )
    # TradingView weekly volatility is the mean daily (high-low)/abs(low)
    # over the bars in the trailing seven-calendar-day window.
    week_cutoff = pd.Timestamp(evaluation_date or as_of) - pd.Timedelta(days=7)
    volatility1w = _number(daily_range_pct[bars["date"].gt(week_cutoff)].mean())
    sma20 = _number(bars["Close"].tail(20).mean())
    sma50 = _number(bars["Close"].tail(50).mean())
    sma200 = _number(bars["Close"].tail(200).mean()) if len(bars) >= 200 else None

    previous_20 = bars.iloc[-21:-1]
    if previous_20.empty:
        return None
    pivot = _number(previous_20["High"].max())
    high_lookback_sessions = min(len(bars), 252)
    available_high = _number(bars["High"].tail(high_lookback_sessions).max())
    low20 = _number(bars["Low"].tail(20).min())
    high20 = _number(bars["High"].tail(20).max())
    if None in (
        adr14,
        atr14_pct,
        adv20,
        average_volume60,
        latest_volume,
        relative_volume,
        performance1w,
        performance1m,
        performance3m,
        volatility1w,
        sma20,
        sma50,
        pivot,
        available_high,
        low20,
        high20,
    ):
        return None

    pivot_distance = (float(pivot) - close) / close * 100.0
    high_distance = (
        (float(available_high) - close) / float(available_high) * 100.0
    )
    prior_range = _number(daily_range_pct.iloc[-20:-5].mean())
    recent_range = _number(daily_range_pct.tail(5).mean())
    compression = (
        float(recent_range) / float(prior_range)
        if prior_range and prior_range > 0 and recent_range is not None
        else None
    )
    width = float(high20) - float(low20)
    close_location = (close - float(low20)) / width if width > 0 else None
    volume_ratio = _number(bars["Volume"].tail(5).mean() / bars["Volume"].tail(20).mean())
    if None in (compression, close_location, volume_ratio):
        return None
    return {
        "price": close,
        "adr14_pct": adr14,
        "atr14_pct": atr14_pct,
        "avg_dollar_volume20": adv20,
        "avg_volume60": average_volume60,
        "latest_volume": latest_volume,
        "relative_volume": relative_volume,
        "performance1w_pct": performance1w,
        "performance1m_pct": performance1m,
        "performance3m_pct": performance3m,
        "volatility1w_pct": volatility1w,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "pivot_distance_pct": pivot_distance,
        "distance_available_high_pct": high_distance,
        "high_lookback_sessions": high_lookback_sessions,
        "compression_ratio": compression,
        "close_location20": close_location,
        "volume_ratio5_20": volume_ratio,
    }


def _next_earnings_event(
    as_of: dt.date, earnings_dates: np.ndarray
) -> tuple[int, dt.date] | None:
    """Return positive trading days to the next earnings event and its date."""
    if earnings_dates is None or len(earnings_dates) == 0:
        return None
    current = np.datetime64(as_of, "D")
    position = int(np.searchsorted(earnings_dates, current, side="left"))
    if position >= len(earnings_dates):
        return None
    event = np.datetime64(earnings_dates[position], "D")
    holidays = np.asarray(pd.DatetimeIndex(NYSE_HOLIDAYS), dtype="datetime64[D]")
    sessions = int(np.busday_count(current, event, holidays=holidays))
    return sessions, pd.Timestamp(event).date()


def technical_screen(
    prices: pd.DataFrame,
    symbols: dict[str, dict[str, Any]],
    earnings_path: Path,
    *,
    earnings_overflow_path: Path | None = None,
    as_of: dt.date,
    valid_for: dt.date | None = None,
    observed_at: dt.datetime | None = None,
    minimum_earnings_tickers: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    earnings = load_earnings_dates_map(path=str(earnings_path))
    if earnings_overflow_path is not None and earnings_overflow_path.is_file():
        overflow = load_earnings_dates_map(path=str(earnings_overflow_path))
        for ticker, dates in overflow.items():
            if ticker in earnings:
                earnings[ticker] = np.unique(
                    np.concatenate([earnings[ticker], dates])
                ).astype("datetime64[D]")
            else:
                earnings[ticker] = dates
    if not earnings:
        raise FocusBuildError("earnings calendar is empty; the five-session gate cannot run")
    if len(earnings) < minimum_earnings_tickers:
        raise FocusBuildError(
            f"earnings calendar covers {len(earnings)} tickers; minimum is "
            f"{minimum_earnings_tickers}"
        )

    measured: list[dict[str, Any]] = []
    for ticker, group in prices.groupby("ticker", sort=False):
        identity = symbols.get(str(ticker).upper())
        if identity is None:
            continue
        metrics = _ticker_metrics(
            group, as_of, evaluation_date=valid_for or as_of
        )
        if metrics is None:
            continue
        market_cap = _number(identity.get("market_cap"))
        metrics.update(
            {
                "ticker": str(ticker).upper(),
                "company_name": _clean_text(identity.get("company_name"), str(ticker)),
                "exchange": _clean_text(identity.get("exchange")),
                "sector": _clean_text(identity.get("sector"), "Unknown"),
                "industry": _clean_text(identity.get("industry"), "Unknown"),
                "research_lane": _research_lane(
                    identity.get("sector"), identity.get("industry")
                ),
                "market_cap": market_cap,
            }
        )
        measured.append(metrics)

    if not measured:
        raise FocusBuildError("no current common-stock price histories could be measured")
    counts = {
        "measured": len(measured),
        "future_earnings_covered": sum(
            1
            for row in measured
            if _next_earnings_event(
                valid_for or as_of, earnings.get(row["ticker"])
            )
            is not None
        ),
        "identity_liquidity_gate": 0,
        "armed_technical_gate": 0,
        "earnings_gate": 0,
        "setup_readiness_gate": 0,
    }
    passed: list[dict[str, Any]] = []
    for row in measured:
        identity_liquidity = (
            row["price"] >= MIN_PRICE
            and row["avg_volume60"] >= MIN_AVG_VOLUME_60
            and row["latest_volume"] >= MIN_LATEST_VOLUME
            and row["exchange"].upper() in ALLOWED_EXCHANGES
        )
        if not identity_liquidity:
            continue
        counts["identity_liquidity_gate"] += 1
        trend_ok = row["price"] > row["sma20"] and row["price"] > row["sma50"]
        armed_ok = (
            trend_ok
            and row["adr14_pct"] >= MIN_ADR_14_PCT
            and row["performance1m_pct"] >= MIN_PERFORMANCE_1M_PCT
            and row["performance3m_pct"] >= MIN_PERFORMANCE_3M_PCT
            and MIN_PERFORMANCE_1W_PCT
            <= row["performance1w_pct"]
            <= MAX_PERFORMANCE_1W_PCT
            and row["volatility1w_pct"] >= MIN_VOLATILITY_1W_PCT
            and row["relative_volume"] >= MIN_RELATIVE_VOLUME
        )
        if not armed_ok:
            continue
        counts["armed_technical_gate"] += 1

        ticker = row["ticker"]
        if ticker not in earnings:
            # Standard-company lane only in v1. Unknown earnings is not proof
            # of a safe window, so it fails closed rather than inheriting the
            # ETF-friendly NaN pass-through used by systematic strategies.
            continue
        next_event = _next_earnings_event(valid_for or as_of, earnings[ticker])
        if next_event is None:
            continue
        earnings_td, earnings_date = next_event
        if earnings_td <= EARNINGS_BLACKOUT_TD:
            continue
        counts["earnings_gate"] += 1

        pivot_proximity = max(0.0, 12.0 - abs(row["pivot_distance_pct"]) * 1.5)
        compression_quality = max(
            0.0, min(15.0, (1.10 - row["compression_ratio"]) * 30.0)
        )
        close_quality = max(
            0.0, min(8.0, (row["close_location20"] - 0.50) * 20.0)
        )
        # A 52-week-high feature is useful when a full year exists, but it must
        # not redefine the universe by excluding 3-12 month IPOs.  Give shorter
        # histories a neutral contribution and retain their available-high
        # distance for audit.
        high_quality = (
            max(0.0, 8.0 - row["distance_available_high_pct"] * 0.4)
            if row["high_lookback_sessions"] >= 252
            else 4.0
        )
        volume_quality = max(
            0.0, min(5.0, (MAX_VOLUME_RATIO_5_20 - row["volume_ratio5_20"]) * 10.0)
        )
        setup_quality = min(
            100.0,
            35.0
            + min(max(row["performance1m_pct"] - 30.0, 0.0), 12.0)
            + min(max(row["adr14_pct"] - 4.0, 0.0) * 2.5, 10.0)
            + pivot_proximity
            + compression_quality
            + close_quality
            + high_quality
            + volume_quality,
        )
        setup_ready = (
            MIN_PIVOT_DISTANCE_PCT
            <= row["pivot_distance_pct"]
            <= MAX_PIVOT_DISTANCE_PCT
            and row["compression_ratio"] <= MAX_COMPRESSION_RATIO
            and row["close_location20"] >= MIN_CLOSE_LOCATION_20
            and row["volume_ratio5_20"] <= MAX_VOLUME_RATIO_5_20
            and setup_quality >= MIN_SETUP_QUALITY
        )
        if not setup_ready:
            continue
        counts["setup_readiness_gate"] += 1
        technical = {
            "adr14_pct": _pct(row["adr14_pct"]),
            "atr14_pct": _pct(row["atr14_pct"]),
            "avg_dollar_volume20": round(row["avg_dollar_volume20"]),
            "avg_volume60": round(row["avg_volume60"]),
            "latest_volume": round(row["latest_volume"]),
            "relative_volume": _pct(row["relative_volume"], 2),
            "performance1w_pct": _pct(row["performance1w_pct"]),
            "performance1m_pct": _pct(row["performance1m_pct"]),
            "performance3m_pct": _pct(row["performance3m_pct"]),
            "volatility1w_pct": _pct(row["volatility1w_pct"]),
            "pivot_distance_pct": _pct(row["pivot_distance_pct"]),
            "distance_available_high_pct": _pct(
                row["distance_available_high_pct"]
            ),
            "high_lookback_sessions": int(row["high_lookback_sessions"]),
            "compression_ratio": _pct(row["compression_ratio"], 2),
            "close_location20": _pct(row["close_location20"], 2),
            "volume_ratio5_20": _pct(row["volume_ratio5_20"], 2),
        }
        passed.append(
            {
                "ticker": ticker,
                "company_name": row["company_name"],
                "sector": row["sector"],
                "industry": row["industry"],
                "causal_cluster": _causal_cluster(
                    row["industry"], row["sector"]
                ),
                "research_lane": row["research_lane"],
                "technical_state": "ARMED",
                "screen_price": row["price"],
                "technical_gate": "PASS",
                "liquidity_gate": "PASS",
                "setup_quality": round(setup_quality, 1),
                "observed_at": (
                    observed_at.astimezone(ET)
                    if observed_at is not None
                    else session_close(as_of)
                ).isoformat(),
                "earnings_td": earnings_td,
                "event_date": earnings_date.isoformat(),
                "technical": technical,
                "setup": (
                    f"The daily-bar Armed mirror is intact: +{technical['performance1m_pct']:.1f}% "
                    f"over 1 month and +{technical['performance3m_pct']:.1f}% over 3 months, "
                    f"but only {technical['performance1w_pct']:+.1f}% this week; 14-day ADR is "
                    f"{technical['adr14_pct']:.1f}%. Price is "
                    f"{technical['pivot_distance_pct']:+.1f}% from the 20-session pivot, "
                    f"with recent range compression at {technical['compression_ratio']:.2f}x."
                ),
                "trigger": (
                    "TradingView Live RVOL screen only: make a new one-month high while "
                    "above the session open, up at least 2%, and RVOL-at-Time is at least 2.0."
                ),
                "invalidation": (
                    "Pass if the new-high breakout loses the live pivot, falls back below the "
                    "session open, or RVOL-at-Time never confirms."
                ),
            }
        )

    passed.sort(
        key=lambda row: (
            row["setup_quality"],
            row["technical"]["performance1m_pct"],
            row["ticker"],
        ),
        reverse=True,
    )
    for screen_rank, row in enumerate(passed, start=1):
        row["screen_rank"] = screen_rank
    return passed, counts


def load_fundamental_research(path: Path, *, as_of: dt.date) -> tuple[dict[str, dict], str]:
    if not path.is_file():
        raise FocusBuildError(f"fundamental research snapshot is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FocusBuildError(f"fundamental research snapshot is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise FocusBuildError("fundamental research snapshot must be a JSON object")
    health = payload.get("health") if isinstance(payload.get("health"), dict) else {}
    snapshot_as_of = _date(health.get("as_of"))
    if snapshot_as_of is None:
        raise FocusBuildError("fundamental research snapshot has no valid health.as_of")
    age = (as_of - snapshot_as_of).days
    if age < 0 or age > MAX_FUNDAMENTAL_AGE_DAYS:
        raise FocusBuildError(
            f"fundamental research snapshot is {age} days from the price cutoff; max is "
            f"{MAX_FUNDAMENTAL_AGE_DAYS}"
        )
    rows = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
    mapped = {
        str(row.get("ticker") or "").upper(): row
        for row in rows
        if isinstance(row, dict) and str(row.get("ticker") or "").strip()
    }
    if not mapped:
        raise FocusBuildError("fundamental research snapshot has no candidates")
    return mapped, snapshot_as_of.isoformat()


def load_research_controls(path: Path) -> tuple[dict[str, dict], str]:
    """Load durable sleeve controls without reusing cached scores as evidence."""
    if not path.is_file():
        return {}, "NOT_AVAILABLE"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FocusBuildError(f"research control snapshot is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise FocusBuildError("research control snapshot must be a JSON object")
    health = payload.get("health") if isinstance(payload.get("health"), dict) else {}
    snapshot_as_of = _date(health.get("as_of"))
    rows = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
    mapped = {
        str(row.get("ticker") or "").upper(): row
        for row in rows
        if isinstance(row, dict) and str(row.get("ticker") or "").strip()
    }
    return mapped, snapshot_as_of.isoformat() if snapshot_as_of else "UNKNOWN"


class FMPNewsClient:
    def __init__(
        self,
        api_key: str,
        *,
        session=None,
        timeout: int = 12,
        retries: int = 2,
    ):
        if not api_key.strip():
            raise FocusBuildError("FMP_API_KEY is required for --fetch-news")
        self.api_key = api_key.strip()
        self.session = session or requests.Session()
        self.timeout = timeout
        self.retries = retries

    def _fetch_list(
        self,
        endpoint: str,
        *,
        params: dict[str, Any],
        label: str,
        allow_empty: bool = False,
    ) -> list[dict[str, Any]]:
        query = {**params, "apikey": self.api_key}
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self.session.get(
                    f"{FMP_BASE}/{endpoint}", params=query, timeout=self.timeout
                )
                if response.status_code == 429:
                    raise requests.HTTPError("rate limited (429)")
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, list):
                    raise ValueError(f"unexpected {type(payload).__name__} response")
                rows = [item for item in payload if isinstance(item, dict)]
                if not rows and not allow_empty:
                    raise ValueError("empty response")
                return rows
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(2**attempt)
        raise FocusBuildError(f"FMP {label} failed: {last_error}")

    def fetch(self, ticker: str) -> list[dict[str, Any]]:
        """Fetch both current news feeds; either endpoint failing aborts."""
        rows: list[dict[str, Any]] = []
        for endpoint in NEWS_ENDPOINTS:
            payload = self._fetch_list(
                endpoint,
                params={"symbols": ticker, "page": 0, "limit": 20},
                label=f"{endpoint} for {ticker}",
                allow_empty=True,
            )
            rows.extend({**item, "_endpoint": endpoint} for item in payload)
        return rows

    def fetch_share_structure(self, ticker: str) -> dict[str, float]:
        payload = self._fetch_list(
            "shares-float",
            params={"symbol": ticker},
            label=f"shares-float for {ticker}",
        )
        float_shares = _number(payload[0].get("floatShares"))
        outstanding_shares = _number(payload[0].get("outstandingShares"))
        if float_shares is None or float_shares <= 0:
            raise FocusBuildError(f"FMP shares-float for {ticker} lacks floatShares")
        if outstanding_shares is None or outstanding_shares <= 0:
            raise FocusBuildError(
                f"FMP shares-float for {ticker} lacks outstandingShares"
            )
        return {
            "float_shares": float_shares,
            "outstanding_shares": outstanding_shares,
        }

    def fetch_float_shares(self, ticker: str) -> float:
        """Compatibility helper for focused callers that need only float."""
        return self.fetch_share_structure(ticker)["float_shares"]

    def fetch_financial_evidence(
        self,
        ticker: str,
        *,
        market_cap: float,
        as_of: dt.date,
        company_name: str,
        sector: str,
        industry: str,
        research_lane: str,
    ) -> dict[str, Any]:
        """Build a current, source-linked standard-company evidence packet."""
        frames: list[pd.DataFrame] = []
        latest_statement_date: dt.date | None = None
        for endpoint in FINANCIAL_ENDPOINTS:
            payload = self._fetch_list(
                endpoint,
                params={"symbol": ticker, "period": "annual", "limit": 5},
                label=f"{endpoint} for {ticker}",
            )
            frame = pd.json_normalize(payload, sep="__")
            if len(frame) < 4:
                raise FocusBuildError(
                    f"FMP {endpoint} for {ticker} has {len(frame)} annual rows; four required"
                )
            frame.insert(0, "ticker", ticker.upper())
            frame.insert(1, "endpoint", endpoint)
            accepted = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
            for column in ("acceptedDate", "accepted_date", "filingDate", "filing_date"):
                if column in frame.columns:
                    accepted = accepted.fillna(
                        pd.to_datetime(frame[column], utc=True, errors="coerce")
                    )
            frame["accepted_at"] = accepted
            statement_dates = pd.to_datetime(frame.get("date"), errors="coerce").dropna()
            if statement_dates.empty:
                raise FocusBuildError(f"FMP {endpoint} for {ticker} lacks fiscal dates")
            endpoint_latest = statement_dates.max().date()
            if endpoint_latest > as_of:
                raise FocusBuildError(f"FMP {endpoint} for {ticker} contains future data")
            latest_statement_date = (
                endpoint_latest
                if latest_statement_date is None
                else max(latest_statement_date, endpoint_latest)
            )
            frames.append(frame)

        if latest_statement_date is None:
            raise FocusBuildError(f"FMP financial evidence for {ticker} is empty")
        statement_age = (as_of - latest_statement_date).days
        if statement_age < 0 or statement_age > MAX_ANNUAL_STATEMENT_AGE_DAYS:
            raise FocusBuildError(
                f"latest annual statement for {ticker} is {statement_age} days old"
            )
        bundle = pd.concat(frames, ignore_index=True, sort=False)
        metrics = calculate_ticker_metrics(
            bundle,
            ticker=ticker,
            market_cap=market_cap,
            company_name=company_name,
            sector=sector,
            industry=industry,
            research_lane=research_lane,
        )
        filing = self.fetch_latest_sec_filing(ticker, as_of=as_of)
        revenue_change = _number(metrics.get("revenue_growth_change"))
        fcf_change = _number(metrics.get("fcf_margin_change"))
        dilution = _number(metrics.get("share_count_cagr_3y"))
        leverage = _number(metrics.get("net_debt_to_ebitda"))
        coverage = sum(
            value is not None
            for value in (revenue_change, fcf_change, dilution, leverage)
        ) / 4.0 * 100.0
        latest_fcf_positive = metrics.get("latest_fcf_positive") is True
        research_score = (
            70.0
            + (10.0 if revenue_change is not None and revenue_change > 0 else 0.0)
            + (10.0 if fcf_change is not None and fcf_change > 0 else 0.0)
            + (10.0 if latest_fcf_positive else 0.0)
        )
        return {
            "ticker": ticker.upper(),
            "company_name": company_name,
            "sector": sector,
            "industry": industry,
            "research_lane": research_lane,
            "market_cap": market_cap,
            "statement_periods": int(metrics.get("statement_periods") or 0),
            "latest_fiscal_date": latest_statement_date.isoformat(),
            "latest_fcf_positive": latest_fcf_positive,
            "revenue_growth_change": revenue_change,
            "fcf_margin_change": fcf_change,
            "share_count_cagr_3y": dilution,
            "net_debt_to_ebitda": leverage,
            "score_coverage_pct": coverage,
            "research_score": research_score,
            "issuer_cik": filing["issuer_cik"],
            "latest_accepted_at": filing["as_of"],
            "sec_source": filing,
            "source_current": True,
            "hard_exclusion_reason": "",
            "research_control": "",
            "research_suppressed": False,
        }

    def fetch_latest_sec_filing(
        self, ticker: str, *, as_of: dt.date
    ) -> dict[str, Any]:
        start = as_of - dt.timedelta(days=MAX_ANNUAL_STATEMENT_AGE_DAYS)
        payload = self._fetch_list(
            "sec-filings-search/symbol",
            params={
                "symbol": ticker,
                "from": start.isoformat(),
                "to": as_of.isoformat(),
                "page": 0,
                "limit": 100,
            },
            label=f"SEC filing search for {ticker}",
        )
        usable: list[tuple[dt.date, dict[str, Any], str, str]] = []
        for row in payload:
            form = str(
                row.get("formType") or row.get("form") or row.get("type") or ""
            ).strip().upper()
            if form not in {"10-K", "10-Q", "10-K/A", "10-Q/A"}:
                continue
            filed = _date(
                row.get("acceptedDate")
                or row.get("filingDate")
                or row.get("date")
            )
            url = _safe_url(row.get("finalLink") or row.get("link") or row.get("url"))
            host = (urlparse(url).hostname or "").lower()
            if filed is None or filed > as_of or not (
                host == "sec.gov" or host.endswith(".sec.gov")
            ):
                continue
            usable.append((filed, row, form, url))
        if not usable:
            raise FocusBuildError(
                f"FMP SEC filing search for {ticker} has no direct current 10-Q/10-K link"
            )
        filed, row, form, url = max(usable, key=lambda item: item[0])
        age = (as_of - filed).days
        if age > MAX_SEC_FILING_AGE_DAYS:
            raise FocusBuildError(
                f"latest 10-Q/10-K for {ticker} is {age} days old; max is "
                f"{MAX_SEC_FILING_AGE_DAYS}"
            )
        cik_text = str(row.get("cik") or row.get("cikNumber") or "").strip()
        digits = "".join(character for character in cik_text if character.isdigit())
        accession = str(
            row.get("accessionNumber") or row.get("accession_number") or ""
        ).strip()
        identity = accession or hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        return {
            "source_id": f"sec-{ticker.lower()}-{identity}",
            "label": f"{ticker} {form} filed {filed.isoformat()}",
            "url": url,
            "as_of": filed.isoformat(),
            "primary": True,
            "issuer_cik": digits,
            "form": form,
        }


def _news_date(row: dict[str, Any]) -> dt.date | None:
    return _date(row.get("publishedDate") or row.get("date") or row.get("published_at"))


def normalize_news(
    rows: list[dict[str, Any]], *, ticker: str, as_of: dt.date
) -> dict[str, Any]:
    recent = []
    for row in rows:
        published = _news_date(row)
        title = str(row.get("title") or row.get("headline") or "").strip()
        if not published or not title:
            continue
        age = (as_of - published).days
        if age < 0 or age > MAX_NEWS_AGE_DAYS:
            continue
        symbol = str(row.get("symbol") or row.get("ticker") or ticker).upper()
        if symbol and symbol != ticker:
            continue
        url = _safe_url(row.get("url") or row.get("link"))
        body = f"{title} {row.get('text') or row.get('snippet') or ''}".lower()
        recent.append(
            {
                "date": published,
                "title": title,
                "url": url,
                "publisher": str(row.get("publisher") or row.get("site") or "FMP feed"),
                "positive": any(term in body for term in POSITIVE_NEWS_TERMS),
                "negative": any(term in body for term in NEGATIVE_NEWS_TERMS),
                "official": row.get("_endpoint") == "news/press-releases",
            }
        )
    recent.sort(
        key=lambda row: (row["negative"], not row["positive"], not row["official"], -row["date"].toordinal())
    )
    negative = [row for row in recent if row["negative"]]
    usable = [row for row in recent if not row["negative"]]
    chosen = usable[0] if usable else None
    return {"chosen": chosen, "negative": negative, "count": len(recent)}


def load_news_fixture(path: Path) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FocusBuildError("news fixture must map tickers to lists")
    return {
        str(ticker).upper(): rows
        for ticker, rows in payload.items()
        if isinstance(rows, list)
    }


def load_float_fixture(path: Path) -> dict[str, float | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FocusBuildError("float fixture must map tickers to share counts")
    return {
        str(ticker).upper(): _number(value)
        for ticker, value in payload.items()
        if str(ticker).strip()
    }


def load_market_cap_fixture(path: Path) -> dict[str, float | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FocusBuildError("market-cap fixture must map tickers to values")
    return {
        str(ticker).upper(): _number(value)
        for ticker, value in payload.items()
        if str(ticker).strip()
    }


def load_fundamental_fixture(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FocusBuildError("fundamental fixture must map tickers to evidence objects")
    return {
        str(ticker).upper(): row
        for ticker, row in payload.items()
        if str(ticker).strip() and isinstance(row, dict)
    }


def merge_research_controls(
    evidence: dict[str, dict[str, Any]],
    controls: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Apply only explicit research controls; cached scores never carry proof."""
    allowed = {
        "hard_exclusion_reason",
        "research_control",
        "research_suppressed",
        "research_priority",
    }
    merged: dict[str, dict[str, Any]] = {}
    for ticker, row in evidence.items():
        result = dict(row)
        control = controls.get(ticker)
        if isinstance(control, dict):
            for key in allowed:
                if key in control:
                    result[key] = control[key]
        merged[ticker] = result
    return merged


def enrich_live_candidates(
    client: FMPNewsClient,
    technical_rows: list[dict[str, Any]],
    *,
    as_of: dt.date,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, float | None],
    dict[str, float | None],
    dict[str, dict[str, Any]],
]:
    """Fetch a complete bounded evidence set or fail the whole run."""
    if len(technical_rows) > MAX_ENRICHMENT_CANDIDATES:
        raise FocusBuildError(
            f"technical cohort has {len(technical_rows)} names; bounded enrichment capacity "
            f"is {MAX_ENRICHMENT_CANDIDATES}. Treat as screen drift, not NO_SETUP."
        )
    if not technical_rows:
        return {}, {}, {}, {}

    def fetch_basic(row: dict[str, Any]):
        ticker = row["ticker"]
        return ticker, client.fetch(ticker), client.fetch_share_structure(ticker)

    basic_results: dict[str, tuple[list[dict[str, Any]], dict[str, float]]] = {}
    failures: list[str] = []
    workers = min(MAX_ENRICHMENT_WORKERS, len(technical_rows))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_basic, row): row["ticker"] for row in technical_rows}
        for future in concurrent.futures.as_completed(futures):
            ticker = futures[future]
            try:
                result_ticker, news, shares = future.result()
                basic_results[result_ticker] = (news, shares)
            except Exception as exc:  # noqa: BLE001 - aggregate and fail closed
                failures.append(f"{ticker}: {exc}")
    expected = {row["ticker"] for row in technical_rows}
    if failures or set(basic_results) != expected:
        details = "; ".join(failures[:5]) or "one or more candidates returned no result"
        raise FocusBuildError(f"candidate news/share enrichment incomplete: {details}")

    news_by_ticker: dict[str, list[dict[str, Any]]] = {}
    float_by_ticker: dict[str, float | None] = {}
    market_cap_by_ticker: dict[str, float | None] = {}
    eligible: list[dict[str, Any]] = []
    for row in technical_rows:
        ticker = row["ticker"]
        news, shares = basic_results[ticker]
        news_by_ticker[ticker] = news
        float_shares = shares["float_shares"]
        market_cap = shares["outstanding_shares"] * float(row["screen_price"])
        float_by_ticker[ticker] = float_shares
        market_cap_by_ticker[ticker] = market_cap
        if (
            float_shares <= MAX_FLOAT_SHARES
            and MIN_MARKET_CAP <= market_cap <= MAX_MARKET_CAP
            and row.get("research_lane") == "standard_company"
        ):
            eligible.append(row)

    def fetch_financial(row: dict[str, Any]):
        ticker = row["ticker"]
        evidence = client.fetch_financial_evidence(
            ticker,
            market_cap=float(market_cap_by_ticker[ticker]),
            as_of=as_of,
            company_name=row["company_name"],
            sector=str(row.get("sector") or "Unknown"),
            industry=str(row.get("industry") or "Unknown"),
            research_lane=str(row.get("research_lane") or "standard_company"),
        )
        return ticker, evidence

    fundamentals: dict[str, dict[str, Any]] = {}
    failures = []
    if eligible:
        workers = min(MAX_ENRICHMENT_WORKERS, len(eligible))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(fetch_financial, row): row["ticker"] for row in eligible}
            for future in concurrent.futures.as_completed(futures):
                ticker = futures[future]
                try:
                    result_ticker, evidence = future.result()
                    fundamentals[result_ticker] = evidence
                except Exception as exc:  # noqa: BLE001 - aggregate and fail closed
                    failures.append(f"{ticker}: {exc}")
    expected_financial = {row["ticker"] for row in eligible}
    if failures or set(fundamentals) != expected_financial:
        details = "; ".join(failures[:5]) or "one or more candidates returned no result"
        raise FocusBuildError(f"candidate financial enrichment incomplete: {details}")
    return news_by_ticker, float_by_ticker, market_cap_by_ticker, fundamentals


def require_complete_fixture_enrichment(
    technical_rows: list[dict[str, Any]],
    *,
    news_by_ticker: dict[str, list[dict[str, Any]]],
    float_by_ticker: dict[str, float | None],
    market_cap_by_ticker: dict[str, float | None],
    fundamentals: dict[str, dict[str, Any]],
) -> None:
    if len(technical_rows) > MAX_ENRICHMENT_CANDIDATES:
        raise FocusBuildError("fixture cohort exceeds bounded enrichment capacity")
    missing: list[str] = []
    for row in technical_rows:
        ticker = row["ticker"]
        if ticker not in news_by_ticker:
            missing.append(f"{ticker}:news")
        if ticker not in float_by_ticker or float_by_ticker[ticker] is None:
            missing.append(f"{ticker}:float")
        if ticker not in market_cap_by_ticker or market_cap_by_ticker[ticker] is None:
            missing.append(f"{ticker}:market_cap")
        float_value = _number(float_by_ticker.get(ticker))
        cap_value = _number(market_cap_by_ticker.get(ticker))
        needs_financial = (
            float_value is not None
            and cap_value is not None
            and float_value <= MAX_FLOAT_SHARES
            and MIN_MARKET_CAP <= cap_value <= MAX_MARKET_CAP
            and row.get("research_lane") == "standard_company"
        )
        if needs_financial and ticker not in fundamentals:
            missing.append(f"{ticker}:fundamental")
    if missing:
        raise FocusBuildError(
            "fixture enrichment is incomplete: " + ", ".join(missing[:10])
        )


def research_overlay(
    technical_rows: list[dict[str, Any]],
    fundamentals: dict[str, dict[str, Any]],
    *,
    fundamental_as_of: str,
    as_of: dt.date,
    news_by_ticker: dict[str, list[dict[str, Any]]],
    float_by_ticker: dict[str, float | None] | None = None,
    market_cap_by_ticker: dict[str, float | None] | None = None,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for technical in technical_rows:
        ticker = technical["ticker"]
        fundamental = fundamentals.get(ticker)
        if not isinstance(fundamental, dict):
            continue
        if technical.get("research_lane") != "standard_company":
            continue
        if float_by_ticker is not None:
            float_shares = _number(float_by_ticker.get(ticker))
            if float_shares is None or float_shares > MAX_FLOAT_SHARES:
                continue
        if market_cap_by_ticker is not None:
            market_cap = _number(market_cap_by_ticker.get(ticker))
            if (
                market_cap is None
                or market_cap < MIN_MARKET_CAP
                or market_cap > MAX_MARKET_CAP
            ):
                continue
        news = normalize_news(news_by_ticker.get(ticker, []), ticker=ticker, as_of=as_of)
        chosen = news["chosen"]
        supporting_news = (
            chosen
            if chosen
            and chosen["positive"]
            and chosen["official"]
            and chosen["url"]
            else None
        )
        score = _number(fundamental.get("research_score"))
        coverage = _number(fundamental.get("score_coverage_pct"))
        statement_periods = int(_number(fundamental.get("statement_periods")) or 0)
        latest_fiscal_date = _date(fundamental.get("latest_fiscal_date"))
        revenue_change = _number(fundamental.get("revenue_growth_change"))
        fcf_change = _number(fundamental.get("fcf_margin_change"))
        dilution = _number(fundamental.get("share_count_cagr_3y"))
        leverage = _number(fundamental.get("net_debt_to_ebitda"))
        latest_fcf_positive = fundamental.get("latest_fcf_positive") is True
        source_current = fundamental.get("source_current") is True
        sec_source = fundamental.get("sec_source")
        hard_rejection = str(fundamental.get("hard_exclusion_reason") or "").strip()
        control = str(fundamental.get("research_control") or "").upper()
        suppressed = bool(fundamental.get("research_suppressed")) or control == "PASS"
        negative_news = bool(news["negative"])
        fiscal_age = (
            (as_of - latest_fiscal_date).days
            if latest_fiscal_date is not None
            else MAX_ANNUAL_STATEMENT_AGE_DAYS + 1
        )
        sec_valid = False
        if isinstance(sec_source, dict):
            sec_url = _safe_url(sec_source.get("url"))
            sec_host = (urlparse(sec_url).hostname or "").lower()
            sec_date = _date(sec_source.get("as_of"))
            sec_valid = bool(
                sec_source.get("primary") is True
                and sec_date is not None
                and 0 <= (as_of - sec_date).days <= MAX_SEC_FILING_AGE_DAYS
                and (sec_host == "sec.gov" or sec_host.endswith(".sec.gov"))
            )
        evidence_complete = (
            source_current
            and sec_valid
            and statement_periods >= 4
            and coverage is not None
            and coverage >= 90.0
            and latest_fcf_positive
            and dilution is not None
            and leverage is not None
            and 0 <= fiscal_age <= MAX_ANNUAL_STATEMENT_AGE_DAYS
        )
        financing_flag = dilution is None or dilution > 0.08
        leverage_flag = leverage is None or leverage > 5.0
        operating_support = bool(
            (revenue_change is not None and revenue_change > 0)
            or (fcf_change is not None and fcf_change > 0)
        )
        news_support = bool(supporting_news)
        if (
            suppressed
            or hard_rejection
            or financing_flag
            or leverage_flag
            or negative_news
            or not evidence_complete
        ):
            continue
        if not (operating_support or news_support):
            continue

        supporting_bits: list[str] = []
        if revenue_change is not None and revenue_change > 0:
            supporting_bits.append(
                f"annual revenue growth accelerated {revenue_change * 100:+.1f} percentage points"
            )
        if fcf_change is not None and fcf_change > 0:
            supporting_bits.append(
                f"free-cash-flow margin improved {fcf_change * 100:+.1f} percentage points"
            )
        sources = [
            {
                "source_id": str(sec_source["source_id"]),
                "label": str(sec_source["label"]),
                "url": str(sec_source["url"]),
                "as_of": str(sec_source["as_of"]),
                "primary": True,
            }
        ]
        if supporting_news:
            sources.append(
                {
                    "source_id": f"issuer-release-{ticker}-{supporting_news['date'].isoformat()}",
                    "label": f"{supporting_news['publisher']}: {supporting_news['title']}",
                    "url": supporting_news["url"],
                    "as_of": supporting_news["date"].isoformat(),
                    "primary": True,
                }
            )
        sources.append(
            {
                "source_id": f"fmp-annual-statements-{ticker}-{latest_fiscal_date.isoformat()}",
                "label": (
                    f"Financial Modeling Prep standardized annual statements through "
                    f"{latest_fiscal_date.isoformat()}"
                ),
                "url": FMP_DOCS,
                "as_of": latest_fiscal_date.isoformat(),
                "primary": False,
            }
        )

        catalyst_parts = list(supporting_bits)
        if supporting_news:
            catalyst_parts.append(
                f"issuer release: {supporting_news['title']} ({supporting_news['date'].isoformat()})"
            )
        catalyst = "; ".join(catalyst_parts).capitalize() + "."

        tech = technical["technical"]
        why_now = (
            f"Technically armed: {tech['adr14_pct']:.1f}% 14-day ADR, "
            f"{tech['performance1m_pct']:.1f}% 1-month and {tech['performance3m_pct']:.1f}% "
            f"3-month momentum with a {tech['performance1w_pct']:+.1f}% weekly pause. "
            f"Current filing-backed research support: {catalyst}"
        )
        support_summary = "; ".join(supporting_bits) or "the issuer's operating update"
        variant_wedge = (
            f"Working wedge: {support_summary} may persist into the next filing strongly enough "
            f"to outrun the stock's {tech['performance3m_pct']:.1f}% three-month move. This is a "
            "testable research hypothesis, not a claim about consensus."
        )
        priced_in = (
            f"The tape already reflects a {tech['performance3m_pct']:.1f}% three-month move and sits "
            "above both the 20- and 50-day averages. The screen does not establish what consensus "
            f"expects; the open question is whether {support_summary} proves durable rather than "
            "being fully reflected in the move."
        )
        next_proof = (
            "The next issuer filing or earnings release must sustain the cited revenue/FCF path "
            "without a financing, dilution, or restatement flag; live TradingView confirmation "
            "only validates timing, not the business hypothesis."
        )
        kill_condition = (
            "Remove from Focus if the next filing reverses the cited operating acceleration, "
            "free cash flow turns negative, annual dilution exceeds 8%, net leverage exceeds 5x, "
            "or a financing/restatement issue appears."
        )
        output[ticker] = {
            "ticker": ticker,
            "research_gate": "PASS",
            "fundamental_score": score,
            "attention_rank": (
                0.0 if len(supporting_bits) >= 2 else (1.0 if supporting_news else 2.0)
            ),
            "catalyst_quality": (
                3 if supporting_news and supporting_bits else (2 if supporting_news else 1)
            ),
            "source_quality": sum(1 for source in sources if source["primary"]),
            "source_current": True,
            "catalyst_reaches_economics": True,
            "variant_wedge": variant_wedge,
            "priced_in": priced_in,
            "catalyst": catalyst,
            "why_now": why_now,
            "next_proof": next_proof,
            "sources": sources,
            "unresolved_financing_risk": False,
            "unresolved_dilution_risk": False,
            "unresolved_restatement_risk": False,
            "kill_condition": kill_condition,
            "causal_cluster": technical["causal_cluster"],
            "research_priority": str(fundamental.get("research_priority") or ""),
        }
    return output


def build_payload(
    *,
    technical_rows: list[dict[str, Any]],
    research_by_ticker: dict[str, dict[str, Any]],
    counts: dict[str, int],
    as_of: dt.date,
    valid_for: dt.date,
    phase: str,
    generated_at: dt.datetime,
    fundamental_as_of: str,
    news_status: str,
    screen_manifest: dict[str, Any] | None = None,
    control_snapshot_as_of: str = "NOT_AVAILABLE",
) -> dict[str, Any]:
    manifest = screen_manifest or load_screen_manifest()
    selected, selector_summary = select_focus(
        technical_rows, research_by_ticker, max_names=2
    )
    screen_digest = _json_digest({"rows": technical_rows, "counts": counts})
    research_digest = _json_digest(
        {
            "rows": research_by_ticker,
            "fundamental_as_of": fundamental_as_of,
            "news_status": news_status,
        }
    )
    expires_at = session_expiry(valid_for)
    screen_captured_at = session_close(as_of)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "research_only": True,
        "quick_review_created": False,
        "live_actions_enabled": False,
        "order_staging_enabled": False,
        "phase": phase,
        "status": "READY" if selected else "NO_QUALIFIED_SETUP",
        "as_of": as_of.isoformat(),
        "valid_for": valid_for.isoformat(),
        "generated_at": generated_at.astimezone(ET).isoformat(),
        "expires_at": expires_at.isoformat(),
        "focus": selected,
        "screen_summary": {
            "screen_id": manifest["armed"]["screen_id"],
            "filter_version": (
                f"focus-daily-bar-mirror.v3/{manifest['revision']}"
            ),
            **counts,
            **selector_summary,
            "selected_count": len(selected),
        },
        "provenance": {
            "screen_snapshot_id": (
                f"armed-mirror-{valid_for.isoformat()}-{screen_digest[:12]}"
            ),
            "screen_captured_at": screen_captured_at.isoformat(),
            "screen_digest": screen_digest,
            "research_snapshot_id": (
                f"focus-research-{valid_for.isoformat()}-{research_digest[:12]}"
            ),
            "research_as_of": generated_at.astimezone(ET).isoformat(),
            "research_digest": research_digest,
            "policy_version": "discretionary-focus-policy.v1",
            "price_source": "Cloudflare R2 master_prices.parquet",
            "price_basis": "ADJUSTED_RECOMPUTED_DAILY",
            "trigger_basis": "TEXT_ONLY_LIVE_TRADINGVIEW",
            "tradingview_armed_url": TRADINGVIEW_ARMED_URL,
            "tradingview_live_url": TRADINGVIEW_LIVE_URL,
            "tradingview_formula_sources": [
                TRADINGVIEW_PERFORMANCE_DOC,
                TRADINGVIEW_VOLATILITY_DOC,
                TRADINGVIEW_ADR_DOC,
            ],
            "tradingview_manifest_revision": manifest["revision"],
            "tradingview_manifest_digest": _json_digest(manifest),
            "fundamental_source": (
                "Current bounded Financial Modeling Prep annual statements plus "
                "direct SEC 10-Q/10-K filing links"
            ),
            "fundamental_as_of": fundamental_as_of,
            "research_control_snapshot_as_of": control_snapshot_as_of,
            "news_source": "Financial Modeling Prep stock news and press releases",
            "news_status": news_status,
            "method": (
                "The cloud daily-bar approximation allocates research attention only; the "
                "saved TradingView screens remain authoritative for exact filter state and "
                "intraday confirmation. Fundamentals/news may "
                "remove or rank technically valid names; they cannot rescue a failed chart."
            ),
        },
    }
    if not selected:
        payload["no_setup_reason"] = (
            "No candidate cleared every technical, earnings, primary-source, and research gate."
        )
    return validate_payload(payload, now=generated_at)


def write_payload(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    parser.add_argument(
        "--overflow-prices",
        type=Path,
        default=DEFAULT_OVERFLOW_PRICES,
        help="isolated price history for symbol-master names outside master_prices",
    )
    parser.add_argument("--earnings", type=Path, default=DEFAULT_EARNINGS)
    parser.add_argument(
        "--earnings-overflow",
        type=Path,
        default=DEFAULT_EARNINGS_OVERFLOW,
        help="optional earnings coverage for dynamic overflow-universe names",
    )
    parser.add_argument("--symbols", type=Path, default=DEFAULT_SYMBOLS)
    parser.add_argument(
        "--fundamentals",
        type=Path,
        default=DEFAULT_FUNDAMENTALS,
        help="optional sleeve snapshot used only for durable PASS/suppression controls",
    )
    parser.add_argument("--screen-manifest", type=Path, default=DEFAULT_SCREEN_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--valid-for", default=None)
    parser.add_argument("--phase", choices=("PROVISIONAL", "FINAL"), default="FINAL")
    news = parser.add_mutually_exclusive_group()
    news.add_argument("--fetch-news", action="store_true")
    news.add_argument("--news-json", type=Path)
    parser.add_argument(
        "--float-json",
        type=Path,
        help="fixture mapping tickers to current float shares; required with --news-json when candidates exist",
    )
    parser.add_argument(
        "--market-cap-json",
        type=Path,
        help="fixture mapping tickers to current market caps; required with --news-json when candidates exist",
    )
    parser.add_argument(
        "--fundamental-json",
        type=Path,
        help="fixture mapping tickers to complete current filing-backed evidence",
    )
    args = parser.parse_args()

    started_at = dt.datetime.now(dt.timezone.utc)
    try:
        screen_manifest = load_screen_manifest(args.screen_manifest)
        fixture_mode = bool(args.news_json)
        prices, price_as_of = load_price_data(
            args.prices,
            args.as_of,
            minimum_session_tickers=(
                1 if fixture_mode else MIN_PRODUCTION_UNIVERSE
            ),
        )
        if args.overflow_prices.is_file():
            overflow_prices, _ = load_price_data(
                args.overflow_prices, price_as_of.isoformat()
            )
            prices = combine_price_data(prices, overflow_prices)
        elif not fixture_mode:
            raise FocusBuildError(
                f"overflow price cache is missing: {args.overflow_prices}"
            )
        valid_for = (
            dt.date.fromisoformat(args.valid_for)
            if args.valid_for
            else session_for_run(price_as_of, started_at)
        )
        require_fresh_price_cutoff(price_as_of, valid_for)
        symbols = load_symbols(args.symbols)
        if not fixture_mode:
            validate_production_input_coverage(
                prices, symbols, required_session=price_as_of
            )
        technical_rows, counts = technical_screen(
            prices,
            symbols,
            args.earnings,
            earnings_overflow_path=args.earnings_overflow,
            as_of=price_as_of,
            valid_for=valid_for,
            observed_at=session_close(price_as_of),
            minimum_earnings_tickers=(
                0 if fixture_mode else MIN_PRODUCTION_EARNINGS_COVERAGE
            ),
        )
        if not fixture_mode and counts["measured"] < MIN_PRODUCTION_UNIVERSE:
            raise FocusBuildError(
                f"only {counts['measured']} tickers have current measurable history; "
                f"minimum is {MIN_PRODUCTION_UNIVERSE}"
            )
        if not fixture_mode:
            validate_future_earnings_coverage(counts)
        if len(technical_rows) > MAX_ENRICHMENT_CANDIDATES:
            raise FocusBuildError(
                f"technical cohort has {len(technical_rows)} names; capacity is "
                f"{MAX_ENRICHMENT_CANDIDATES}"
            )

        controls, control_snapshot_as_of = (
            load_research_controls(args.fundamentals)
            if technical_rows
            else ({}, "NOT_REQUIRED:NO_TECHNICAL_CANDIDATES")
        )

        news_by_ticker: dict[str, list[dict[str, Any]]] = {}
        float_by_ticker: dict[str, float | None] = {}
        market_cap_by_ticker: dict[str, float | None] = {}
        fundamentals: dict[str, dict[str, Any]] = {}
        news_status = "NOT_REQUESTED"
        if args.news_json:
            if not (args.float_json and args.market_cap_json and args.fundamental_json):
                raise FocusBuildError(
                    "--news-json requires --float-json, --market-cap-json, and "
                    "--fundamental-json for a complete fixture run"
                )
            news_by_ticker = load_news_fixture(args.news_json)
            float_by_ticker = load_float_fixture(args.float_json)
            market_cap_by_ticker = load_market_cap_fixture(args.market_cap_json)
            fundamentals = load_fundamental_fixture(args.fundamental_json)
            require_complete_fixture_enrichment(
                technical_rows,
                news_by_ticker=news_by_ticker,
                float_by_ticker=float_by_ticker,
                market_cap_by_ticker=market_cap_by_ticker,
                fundamentals=fundamentals,
            )
            news_status = f"FIXTURE:COMPLETE:{len(technical_rows)}"
        elif args.fetch_news and technical_rows:
            client = FMPNewsClient(os.environ.get("FMP_API_KEY", ""))
            (
                news_by_ticker,
                float_by_ticker,
                market_cap_by_ticker,
                fundamentals,
            ) = enrich_live_candidates(client, technical_rows, as_of=valid_for)
            news_status = (
                f"CURRENT:COMPLETE:{len(technical_rows)};"
                f"FINANCIAL_COMPLETE:{len(fundamentals)}"
            )
        elif args.fetch_news:
            news_status = "NOT_REQUIRED:NO_TECHNICAL_CANDIDATES"
        elif technical_rows:
            raise FocusBuildError(
                "complete current research enrichment is required; use --fetch-news or "
                "provide all four fixture files"
            )
        counts["float_gate"] = sum(
            1
            for value in float_by_ticker.values()
            if value is not None and value <= MAX_FLOAT_SHARES
        )
        counts["market_cap_gate"] = sum(
            1
            for value in market_cap_by_ticker.values()
            if value is not None and MIN_MARKET_CAP <= value <= MAX_MARKET_CAP
        )
        counts["standard_lane_gate"] = sum(
            1 for row in technical_rows if row.get("research_lane") == "standard_company"
        )
        counts["financial_evidence_gate"] = len(fundamentals)

        fundamentals = merge_research_controls(fundamentals, controls)
        fundamental_as_of = (
            valid_for.isoformat()
            if technical_rows
            else "NOT_REQUIRED:NO_TECHNICAL_CANDIDATES"
        )
        research = research_overlay(
            technical_rows,
            fundamentals,
            fundamental_as_of=fundamental_as_of,
            as_of=valid_for,
            news_by_ticker=news_by_ticker,
            float_by_ticker=float_by_ticker,
            market_cap_by_ticker=market_cap_by_ticker,
        )
        generated_at = dt.datetime.now(dt.timezone.utc)
        payload = build_payload(
            technical_rows=technical_rows,
            research_by_ticker=research,
            counts=counts,
            as_of=price_as_of,
            valid_for=valid_for,
            phase=args.phase,
            generated_at=generated_at,
            fundamental_as_of=fundamental_as_of,
            news_status=news_status,
            screen_manifest=screen_manifest,
            control_snapshot_as_of=control_snapshot_as_of,
        )
        output = write_payload(payload, args.output)
    except (FocusBuildError, OSError, ValueError) as exc:
        print(f"Discretionary Focus build failed: {exc}")
        return 1

    print(
        f"Focus {payload['phase']} for {payload['valid_for']}: "
        f"{len(payload['focus'])} name(s); status={payload['status']}"
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
