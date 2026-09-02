"""Fail-closed normalization of full TradingView extended-hours CSV exports.

TradingView is a discovery source only.  Normalized rows intentionally contain
no executable quote, contract, or broker state; a separate fresh IBKR snapshot
would be required before even a hypothetical sizing preview can clear.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from trading_calendar import TRADING_DAY

from .schema import PremarketSnapshot, iso_utc, parse_timestamp

_NY = ZoneInfo("America/New_York")
_MISSING = {"", "-", "--", "—", "n/a", "na", "none", "null"}
_SUFFIXES = {
    "k": 1_000.0,
    "m": 1_000_000.0,
    "b": 1_000_000_000.0,
    "t": 1_000_000_000_000.0,
}

RESULT_COUNT_EXACT = "EXACT_MATCH"
RESULT_COUNT_UNVERIFIED = "NOT_INDEPENDENTLY_OBSERVED"
_EXPECTED_SCREEN_IDS = {
    "premarket": "yftOvM3e",
    "after_hours": "Hqgnyp7Y",
}


class TradingViewImportError(ValueError):
    """Raised when an export cannot be proven complete and internally coherent."""


@dataclass(frozen=True)
class TradingViewImport:
    schema_version: int
    provider: str
    saved_screen_id: str
    session: str
    captured_at: str
    target_session_date: str
    source_file: str
    source_file_sha256: str
    reported_result_count: int | None
    post_download_result_count: int | None
    extracted_row_count: int
    result_count_verified: bool
    result_count_verification: str
    snapshots: tuple[PremarketSnapshot, ...]

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["snapshots"] = [item.to_dict() for item in self.snapshots]
        return data


def _normalized_header(value: str) -> str:
    text = str(value or "").replace("\ufeff", "").strip().lower()
    text = text.replace("%", " percent ").replace("$", " dollar ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _parse_number(value: object, *, field: str) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
    else:
        text = str(value or "").strip().lower()
        if text in _MISSING:
            raise TradingViewImportError(f"missing {field}")
        text = (
            text.replace("\u2212", "-")
            .replace("\u2013", "-")
            .replace("\u2014", "-")
            .replace("\u00a0", "")
            .replace(",", "")
            .replace("$", "")
            .replace("%", "")
            .replace("usd", "")
            .strip()
        )
        negative = text.startswith("(") and text.endswith(")")
        if negative:
            text = text[1:-1].strip()
        multiplier = 1.0
        if text and text[-1:] in _SUFFIXES:
            multiplier = _SUFFIXES[text[-1]]
            text = text[:-1].strip()
        try:
            number = float(text) * multiplier
        except ValueError as exc:
            raise TradingViewImportError(f"invalid {field}: {value!r}") from exc
        if negative:
            number = -number
    if not math.isfinite(number):
        raise TradingViewImportError(f"non-finite {field}: {value!r}")
    return number


def target_session_date(captured_at: str | datetime, *, session: str) -> date:
    """Map a capture to the regular NYSE session it is researching."""

    if session not in {"premarket", "after_hours"}:
        raise TradingViewImportError("session must be premarket or after_hours")
    local = parse_timestamp(captured_at).astimezone(_NY)
    if not (1990 <= local.year <= 2040):
        raise TradingViewImportError(
            "capture year is outside the verified NYSE calendar range"
        )
    day = pd.Timestamp(local.date())
    if not TRADING_DAY.is_on_offset(day):
        raise TradingViewImportError(
            "extended-hours capture date is not an NYSE trading day"
        )
    if session == "premarket" and not (time(4, 0) <= local.time() < time(9, 30)):
        raise TradingViewImportError(
            "premarket capture must be between 04:00 and 09:30 ET"
        )
    if session == "after_hours" and not (time(16, 0) <= local.time() < time(20, 0)):
        raise TradingViewImportError(
            "after-hours capture must be between 16:00 and 20:00 ET"
        )
    target = day if session == "premarket" else day + TRADING_DAY
    return target.date()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def result_count_verification(
    reported_result_count: int | None,
    extracted_row_count: int,
    *,
    post_download_result_count: int | None = None,
) -> str:
    """Classify an observed screen count against a live CSV export.

    TradingView's extended-hours universe changes while the export request is
    being generated. The CSV must match either the immediate pre-download or
    post-download observation, and those observations may differ by at most one
    row. Anything else is ambiguous or potentially partial and fails closed.
    """

    extracted = int(extracted_row_count)
    if extracted < 0:
        raise TradingViewImportError("extracted_row_count cannot be negative")
    if reported_result_count is None and post_download_result_count is None:
        return RESULT_COUNT_UNVERIFIED
    observed = [
        int(value)
        for value in (reported_result_count, post_download_result_count)
        if value is not None
    ]
    if any(value < 0 for value in observed):
        raise TradingViewImportError("observed result counts cannot be negative")
    if len(observed) == 2 and abs(observed[0] - observed[1]) > 1:
        raise TradingViewImportError(
            f"ambiguous live result-count swing: observations were {observed}"
        )
    if extracted in observed:
        return RESULT_COUNT_EXACT
    if extracted < max(observed):
        raise TradingViewImportError(
            f"incomplete export: TradingView reported as many as {max(observed)} row(s), "
            f"CSV contains {extracted}"
        )
    raise TradingViewImportError(
        f"export count matched neither browser observation: "
        f"TradingView observations were {observed}, CSV contains {extracted}"
    )


def result_counts_are_verified(
    *,
    reported_result_count: object,
    post_download_result_count: object = None,
    extracted_row_count: object,
    verification_status: object = None,
    require_both_observations: bool = False,
) -> bool:
    """Revalidate persisted count provenance, including legacy exact imports."""

    try:
        extracted = int(extracted_row_count)
    except (TypeError, ValueError):
        return False
    observed: list[int] = []
    for value in (reported_result_count, post_download_result_count):
        if value is None:
            continue
        try:
            observed.append(int(value))
        except (TypeError, ValueError):
            return False
    status = str(verification_status or "").strip().upper()
    if extracted < 0 or not observed or any(value < 0 for value in observed):
        return False
    if require_both_observations and (
        reported_result_count is None or post_download_result_count is None
    ):
        return False
    if len(observed) == 2 and abs(observed[0] - observed[1]) > 1:
        return False
    return extracted in observed and status in {"", RESULT_COUNT_EXACT}


def _first_column(headers: dict[str, str], aliases: Iterable[str]) -> str | None:
    for alias in aliases:
        found = headers.get(_normalized_header(alias))
        if found is not None:
            return found
    return None


def _column_map(fieldnames: list[str], *, session: str) -> dict[str, str | None]:
    headers = {
        _normalized_header(name): name for name in fieldnames if name is not None
    }
    prefix = "pre market" if session == "premarket" else "post market"
    alternate = "premarket" if session == "premarket" else "after hours"
    columns = {
        "symbol": _first_column(headers, ("symbol", "ticker", "ticker symbol")),
        "company_name": _first_column(
            headers, ("name", "company", "company name", "description")
        ),
        "exchange": _first_column(headers, ("exchange", "listing exchange", "market")),
        "security_type": _first_column(
            headers, ("type", "security type", "instrument type")
        ),
        "last": _first_column(
            headers,
            (
                f"{prefix} price",
                f"{alternate} price",
                "extended hours price",
                f"{prefix} close",
            ),
        ),
        "previous_close": _first_column(
            headers, ("previous close", "prev close", "prior close")
        ),
        "change_pct": _first_column(
            headers,
            (
                f"{prefix} change percent",
                f"{alternate} change percent",
                "extended hours change percent",
            ),
        ),
        "move_dollars": _first_column(
            headers,
            (
                f"{prefix} change",
                f"{alternate} change",
                f"{prefix} change dollar",
                "extended hours change",
            ),
        ),
        "volume": _first_column(
            headers,
            (
                f"{prefix} volume",
                f"{alternate} volume",
                "extended hours volume",
            ),
        ),
    }
    required = ("symbol", "last", "volume")
    missing = [name for name in required if not columns[name]]
    if (
        not columns["change_pct"]
        and not columns["move_dollars"]
        and not columns["previous_close"]
    ):
        missing.append("change_pct or move_dollars or previous_close")
    if missing:
        available = ", ".join(fieldnames)
        raise TradingViewImportError(
            f"missing required TradingView column(s): {', '.join(missing)}; available: {available}"
        )
    return columns


def _text(row: dict[str, str], column: str | None) -> str:
    return str(row.get(column, "") if column else "").strip()


def _symbol_and_exchange(raw_symbol: str, raw_exchange: str) -> tuple[str, str]:
    token = raw_symbol.strip().upper()
    exchange = raw_exchange.strip().upper()
    if ":" in token:
        prefix, token = token.rsplit(":", 1)
        exchange = exchange or prefix
    token = token.replace(" ", "")
    if not re.fullmatch(r"[A-Z0-9.\-]{1,15}", token):
        raise TradingViewImportError(f"invalid ticker identity: {raw_symbol!r}")
    return token, exchange


def _normalize_row(
    row: dict[str, str],
    *,
    row_number: int,
    columns: dict[str, str | None],
    captured_at: str,
    session: str,
    target_date: date,
    screen_id: str,
    source_hash: str,
    reported_count: int,
    extracted_count: int,
    freeze_premarket_move: bool,
) -> PremarketSnapshot:
    try:
        symbol, exchange = _symbol_and_exchange(
            _text(row, columns["symbol"]), _text(row, columns["exchange"])
        )
        last = _parse_number(_text(row, columns["last"]), field="session price")
        volume_float = _parse_number(
            _text(row, columns["volume"]), field="session volume"
        )
        if last <= 0 or volume_float < 0:
            raise TradingViewImportError(
                "price must be positive and volume non-negative"
            )
        volume = round(volume_float)
        change_pct = (
            _parse_number(_text(row, columns["change_pct"]), field="change percent")
            if columns["change_pct"]
            else None
        )
        move_dollars = (
            _parse_number(_text(row, columns["move_dollars"]), field="dollar move")
            if columns["move_dollars"]
            else None
        )
        previous_close = (
            _parse_number(_text(row, columns["previous_close"]), field="previous close")
            if columns["previous_close"]
            else None
        )
        if previous_close is None and move_dollars is not None:
            previous_close = last - move_dollars
        if previous_close is None and change_pct is not None:
            denominator = 1.0 + change_pct / 100.0
            if denominator <= 0:
                raise TradingViewImportError(
                    "change percent implies a non-positive prior close"
                )
            previous_close = last / denominator
        if previous_close is None or previous_close <= 0:
            raise TradingViewImportError("could not derive a positive previous close")
        if move_dollars is None:
            move_dollars = last - previous_close
        if change_pct is None:
            change_pct = 100.0 * move_dollars / previous_close

        implied_move = last - previous_close
        implied_pct = 100.0 * implied_move / previous_close
        dollar_tolerance = max(0.05, abs(last) * 0.0025)
        percent_tolerance = max(0.15, abs(change_pct) * 0.05)
        if abs(implied_move - move_dollars) > dollar_tolerance:
            raise TradingViewImportError(
                "session price and reported dollar move are inconsistent"
            )
        if abs(implied_pct - change_pct) > percent_tolerance:
            raise TradingViewImportError(
                "session price and reported percent move are inconsistent"
            )

        return PremarketSnapshot(
            symbol=symbol,
            observed_at=captured_at,
            previous_close=previous_close,
            last=last,
            bid=0.0,
            ask=0.0,
            premarket_volume=volume,
            premarket_open=last,
            premarket_high=last,
            premarket_low=last,
            premarket_vwap=0.0,
            prior_two_day_low=0.0,
            atr_14=0.0,
            avg_volume_20=0.0,
            addv_63=0.0,
            company_name=_text(row, columns["company_name"]),
            market_data_status="BROWSER_EXPORT",
            halted=False,
            halt_status="UNKNOWN",
            tradeable=False,
            source="TRADINGVIEW_BROWSER_EXPORT",
            price_basis="TRADINGVIEW_EXTENDED_HOURS_REPORTED",
            primary_exchange=exchange,
            contract_identity_status="UNRESOLVED",
            premarket_metrics_at=None,
            first_trigger_at=None,
            session=session,
            provider="TRADINGVIEW",
            saved_screen_id=screen_id,
            target_session_date=target_date.isoformat(),
            reported_result_count=reported_count,
            extracted_row_count=extracted_count,
            source_file_sha256=source_hash,
            screen_exchange=exchange,
            security_type=_text(row, columns["security_type"]) or "UNKNOWN",
            reported_change_pct=change_pct,
            reported_move_dollars=move_dollars,
            premarket_move_verification_status=(
                "VERIFIED" if freeze_premarket_move else "UNVERIFIED"
            ),
            premarket_move_verification_source=(
                "TRADINGVIEW_BROWSER_EXPORT" if freeze_premarket_move else ""
            ),
            premarket_move_verified_at=(captured_at if freeze_premarket_move else None),
        )
    except TradingViewImportError as exc:
        raise TradingViewImportError(f"row {row_number}: {exc}") from exc


def import_tradingview_csv(
    path: str | Path,
    *,
    session: str,
    captured_at: str | datetime,
    saved_screen_id: str,
    reported_result_count: int | None = None,
    post_download_result_count: int | None = None,
) -> TradingViewImport:
    source = Path(path).resolve()
    if not source.is_file():
        raise TradingViewImportError(f"CSV file does not exist: {source}")
    screen_id = str(saved_screen_id or "").strip()
    if not screen_id:
        raise TradingViewImportError("saved_screen_id is required")
    expected_screen_id = _EXPECTED_SCREEN_IDS.get(session)
    if expected_screen_id is None:
        raise TradingViewImportError("session must be premarket or after_hours")
    if screen_id != expected_screen_id:
        raise TradingViewImportError(
            f"saved screen does not match {session}: expected {expected_screen_id}"
        )
    captured = iso_utc(captured_at)
    target_date = target_session_date(captured, session=session)
    source_hash = _sha256(source)

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        if not reader.fieldnames:
            raise TradingViewImportError("CSV has no header row")
        columns = _column_map(list(reader.fieldnames), session=session)
        rows = [
            row
            for row in reader
            if any(str(value or "").strip() for value in row.values())
        ]

    extracted = len(rows)
    count_status = result_count_verification(
        reported_result_count,
        extracted,
        post_download_result_count=post_download_result_count,
    )
    reported = int(reported_result_count) if reported_result_count is not None else None
    freeze_premarket_move = session == "premarket" and result_counts_are_verified(
        reported_result_count=reported_result_count,
        post_download_result_count=post_download_result_count,
        extracted_row_count=extracted,
        verification_status=count_status,
        require_both_observations=True,
    )

    snapshots = tuple(
        _normalize_row(
            row,
            row_number=index,
            columns=columns,
            captured_at=captured,
            session=session,
            target_date=target_date,
            screen_id=screen_id,
            source_hash=source_hash,
            reported_count=reported,
            extracted_count=extracted,
            freeze_premarket_move=freeze_premarket_move,
        )
        for index, row in enumerate(rows, start=2)
    )
    identities: set[tuple[str, str]] = set()
    symbols: set[str] = set()
    for snapshot in snapshots:
        identity = (snapshot.symbol, snapshot.screen_exchange)
        if identity in identities or snapshot.symbol in symbols:
            raise TradingViewImportError(
                f"duplicate or ambiguous ticker in export: {snapshot.symbol}"
            )
        identities.add(identity)
        symbols.add(snapshot.symbol)

    return TradingViewImport(
        schema_version=1,
        provider="TRADINGVIEW",
        saved_screen_id=screen_id,
        session=session,
        captured_at=captured,
        target_session_date=target_date.isoformat(),
        source_file=str(source),
        source_file_sha256=source_hash,
        reported_result_count=reported,
        post_download_result_count=(
            int(post_download_result_count)
            if post_download_result_count is not None
            else None
        ),
        extracted_row_count=extracted,
        result_count_verified=(
            reported_result_count is not None or post_download_result_count is not None
        ),
        result_count_verification=count_status,
        snapshots=snapshots,
    )
