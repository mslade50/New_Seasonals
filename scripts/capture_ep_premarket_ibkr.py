"""Capture a read-only IBKR premarket snapshot for EP shadow research.

This adapter intentionally contains no order API.  It uses IBKR scanners to
narrow the request set, then gathers quotes plus extended-hours and daily bars
for deterministic replay by ``run_episodic_pivot_shadow.py``.

``ib_insync`` is an optional dependency in the local TWS environment and is
imported only when this script is run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from episodic_pivot.config import DEFAULT_POLICY
from episodic_pivot.daily_prices import calculate_prior_daily_metrics
from episodic_pivot.manifest import sha256_file
from episodic_pivot.premarket import nominate_candidates, premarket_move_is_verified
from episodic_pivot.schema import PremarketSnapshot, parse_timestamp
from episodic_pivot.tradingview import result_counts_are_verified

_NY = ZoneInfo("America/New_York")
_MARKET_DATA_STATUS = {
    1: "LIVE",
    2: "FROZEN",
    3: "DELAYED",
    4: "DELAYED_FROZEN",
}
_DAILY_WHAT_TO_SHOW = "ADJUSTED_LAST"
_DAILY_PRICE_BASIS = "IBKR_ADJUSTED_LAST"
_AUTO_PORTS = (7496, 4001, 7497, 4002)
_IBKR_RECORD_TYPE = "EP_IBKR_PREMARKET_CAPTURE_V1"
_TRADINGVIEW_SCREEN_BY_SESSION = {
    "premarket": "yftOvM3e",
    "after_hours": "Hqgnyp7Y",
}
_REFRESH_TARGET_RECORD_TYPE = "EP_RESEARCH_QUOTE_REFRESH_TARGETS_V1"


def _finite(value, default=0.0):  # type: ignore[no-untyped-def]
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _halt_status(value) -> tuple[str, float | None]:  # type: ignore[no-untyped-def]
    """Map IBKR output tick 49 without treating missing telemetry as clear."""

    try:
        raw = float(value)
    except (TypeError, ValueError):
        return "UNKNOWN", None
    if not math.isfinite(raw) or raw < 0:
        return "UNKNOWN", raw if math.isfinite(raw) else None
    if raw == 0:
        return "NOT_HALTED", raw
    if raw == 1:
        return "GENERAL_HALT", raw
    if raw == 2:
        return "VOLATILITY_HALT", raw
    return "UNKNOWN", raw


def _exchange_key(value: str) -> str:
    token = "".join(
        character for character in str(value).upper() if character.isalnum()
    )
    aliases = {
        "NYSEARCA": "ARCA",
        "NASDAQGS": "NASDAQ",
        "NASDAQGM": "NASDAQ",
        "NASDAQCM": "NASDAQ",
    }
    return aliases.get(token, token)


def _port_candidates(value: object) -> tuple[int, ...]:
    token = str(value).strip().lower()
    if token == "auto":
        return _AUTO_PORTS
    try:
        port = int(token)
    except ValueError as exc:
        raise ValueError("--port must be 'auto' or a TCP port") from exc
    if port < 1 or port > 65_535:
        raise ValueError("--port must be between 1 and 65535")
    return (port,)


def _connect_read_only(
    ib_factory,  # type: ignore[no-untyped-def]
    *,
    host: str,
    ports: tuple[int, ...],
    client_id: int,
    attempted_ports: list[int] | None = None,
):  # type: ignore[no-untyped-def]
    failures: list[tuple[int, str]] = []
    for port in ports:
        if attempted_ports is not None:
            attempted_ports.append(port)
        ib = ib_factory()
        try:
            ib.connect(
                host,
                port,
                clientId=client_id,
                readonly=True,
                timeout=10,
            )
            if not ib.isConnected():
                raise ConnectionError("IBKR client did not enter connected state")
            return ib, port
        except Exception as exc:  # noqa: BLE001 - bounded local endpoint fallback.
            failures.append((port, type(exc).__name__))
            if ib.isConnected():
                ib.disconnect()
    summary = ", ".join(f"{port}:{kind}" for port, kind in failures)
    raise ConnectionError(f"no read-only IBKR API endpoint connected ({summary})")


def _subscribe_market_data_batch(ib, items, errors):  # type: ignore[no-untyped-def]
    """Subscribe independently so one rejected line preserves sibling quotes."""

    ticker_by_conid = {}
    subscribed_contracts = []
    for item in items:
        contract = item["contract"]
        try:
            ticker = ib.reqMktData(
                contract,
                genericTickList="",
                snapshot=False,
                regulatorySnapshot=False,
            )
            subscribed_contracts.append(contract)
            # ib_insync initializes this field to LIVE (1) before any callback.
            # Reset it so only an explicit IBKR callback can satisfy the gate.
            ticker.marketDataType = 0
            ticker_by_conid[contract.conId] = ticker
        except Exception as exc:  # noqa: BLE001 - preserve partial batch.
            errors.append(
                {
                    "symbol": contract.symbol,
                    "error": f"MARKET_DATA_SUBSCRIPTION_FAILED:{type(exc).__name__}",
                }
            )
    return ticker_by_conid, subscribed_contracts


def _cancel_market_data_batch(
    ib,
    contracts,
    errors,  # type: ignore[no-untyped-def]
) -> None:
    """Cancel each successful subscription without masking usable siblings."""

    for contract in contracts:
        try:
            ib.cancelMktData(contract)
        except Exception as exc:  # noqa: BLE001 - preserve partial batch.
            errors.append(
                {
                    "symbol": contract.symbol,
                    "error": f"MARKET_DATA_CANCEL_FAILED:{type(exc).__name__}",
                }
            )


def _round_robin_keys(
    keys_by_scanner: dict[str, list[object]], scanner_codes: list[str], limit: int
) -> list[object]:
    """Interleave scanner ranks so the first scan cannot consume the whole cap."""

    selected: list[object] = []
    seen: set[object] = set()
    max_rows = max(
        (len(keys_by_scanner.get(code, [])) for code in scanner_codes), default=0
    )
    for rank in range(max_rows):
        for code in scanner_codes:
            rows = keys_by_scanner.get(code, [])
            if rank >= len(rows) or rows[rank] in seen:
                continue
            seen.add(rows[rank])
            selected.append(rows[rank])
            if len(selected) >= limit:
                return selected
    return selected


def _as_ny_index(values) -> pd.DatetimeIndex:  # type: ignore[no-untyped-def]
    index = pd.DatetimeIndex(values)
    if index.tz is None:
        return index.tz_localize(_NY)
    return index.tz_convert(_NY)


def _daily_metrics(bars, session_date):  # type: ignore[no-untyped-def]
    frame = pd.DataFrame(
        {
            "date": [bar.date for bar in bars],
            "open": [_finite(bar.open) for bar in bars],
            "high": [_finite(bar.high) for bar in bars],
            "low": [_finite(bar.low) for bar in bars],
            "close": [_finite(bar.close) for bar in bars],
            "volume": [_finite(bar.volume) for bar in bars],
        }
    )
    return calculate_prior_daily_metrics(frame, session_date)


def _premarket_metrics(bars, session_date, previous_close: float | None = None):  # type: ignore[no-untyped-def]
    if not bars:
        raise ValueError("no extended-hours bars")
    frame = pd.DataFrame(
        {
            "date": [bar.date for bar in bars],
            "open": [_finite(bar.open) for bar in bars],
            "high": [_finite(bar.high) for bar in bars],
            "low": [_finite(bar.low) for bar in bars],
            "close": [_finite(bar.close) for bar in bars],
            "volume": [_finite(bar.volume) for bar in bars],
            "bar_count": [_finite(getattr(bar, "barCount", 0)) for bar in bars],
        }
    )
    frame.index = _as_ny_index(frame.pop("date"))
    same_day = frame.index.date == session_date
    in_hours = (frame.index.time >= time(4, 0)) & (frame.index.time < time(9, 30))
    frame = frame[same_day & in_hours]
    frame = frame[(frame["close"] > 0) & (frame["volume"] > 0)]
    if frame.empty:
        raise ValueError("no valid 04:00-09:30 ET bars")
    volume = float(frame["volume"].sum())
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    vwap = float((typical * frame["volume"]).sum() / volume)
    first_trigger_at = None
    if previous_close and previous_close > 0:
        cumulative_volume = frame["volume"].cumsum()
        gap_pct = 100.0 * (frame["close"] / previous_close - 1.0)
        move_dollars = frame["close"] - previous_close
        triggered = (cumulative_volume >= 100_000) & (
            (gap_pct.abs() >= 2.0) | (move_dollars.abs() >= 0.90)
        )
        if triggered.any():
            first_trigger_at = (
                frame.index[triggered][0]
                .tz_convert(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )
    return {
        "premarket_open": float(frame.iloc[0]["open"]),
        "premarket_high": float(frame["high"].max()),
        "premarket_low": float(frame["low"].min()),
        "premarket_vwap": vwap,
        "premarket_volume": int(volume),
        "premarket_last": float(frame.iloc[-1]["close"]),
        "first_trigger_at": first_trigger_at,
        # This is the newest observed bar timestamp, not the local fetch clock.
        # A stalled IB series must remain visibly stale.
        "premarket_metrics_at": frame.index.max()
        .tz_convert(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }


def _load_target_rows(path: Path) -> tuple[list[dict], str]:
    return _load_target_rows_unfiltered(path)


def _load_refresh_source_manifests(paths: list[Path]) -> dict[str, dict]:
    manifests: dict[str, dict] = {}
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("refresh source manifest must be an object")
        run_id = str(raw.get("run_id", "")).strip()
        search_provider = str(raw.get("search_provider", "")).strip()
        safety = raw.get("safety")
        if (
            raw.get("schema_version") != 2
            or not run_id
            or path.resolve().parent.name != run_id
            or not search_provider
            or search_provider.upper().startswith("OFFLINE")
            or not isinstance(safety, dict)
            or safety.get("research_only") is not True
            or safety.get("live_actions_enabled") is not False
            or str(safety.get("broker_route", "")).upper() != "NONE"
            or safety.get("order_submission_allowed") is not False
        ):
            raise ValueError(
                "refresh source manifest identity, network provenance, or safety is invalid"
            )
        if run_id in manifests:
            raise ValueError(f"duplicate refresh source manifest: {run_id}")
        manifests[run_id] = {"path": str(path.resolve()), "payload": raw}
    return manifests


def _verify_refresh_target_manifest(
    *,
    target_path: Path,
    target_session_date: str,
    source_run_id: str,
    source_manifests: dict[str, dict],
) -> None:
    record = source_manifests.get(source_run_id)
    if not record:
        raise ValueError("refresh target requires its source run manifest")
    manifest = record["payload"]
    artifact = (manifest.get("artifacts") or {}).get("refresh_targets.json")
    if not isinstance(artifact, dict):
        raise TypeError("source manifest is missing refresh_targets.json")
    if (
        artifact.get("sha256") != sha256_file(target_path)
        or int(artifact.get("size_bytes", -1)) != target_path.stat().st_size
    ):
        raise ValueError("refresh target digest does not match its source manifest")
    if not source_run_id.startswith(f"EP-RUN-{target_session_date}-"):
        raise ValueError("refresh target session date does not match source run")


def _validated_target_wrapper(
    path: Path,
    raw: object,
    *,
    source_manifests: dict[str, dict] | None = None,
) -> tuple[list[dict], str, dict[str, str]]:
    """Accept only immutable TradingView imports or safe research refresh lists."""

    if not isinstance(raw, dict) or not isinstance(raw.get("snapshots"), list):
        raise TypeError("target input must be a normalized snapshot object")
    rows = raw["snapshots"]
    provider = str(raw.get("provider", "")).strip().upper()
    record_type = str(raw.get("record_type", "")).strip().upper()
    if provider == "TRADINGVIEW":
        wrapper_session = str(raw.get("session", "")).strip().lower()
        wrapper_screen = str(raw.get("saved_screen_id", "")).strip()
        extracted = raw.get("extracted_row_count")
        if (
            _TRADINGVIEW_SCREEN_BY_SESSION.get(wrapper_session) != wrapper_screen
            or raw.get("result_count_verified") is not True
            or not result_counts_are_verified(
                reported_result_count=raw.get("reported_result_count"),
                post_download_result_count=raw.get("post_download_result_count"),
                extracted_row_count=extracted,
                verification_status=raw.get("result_count_verification"),
                require_both_observations=True,
            )
            or extracted != len(rows)
        ):
            raise ValueError("TradingView target count/provenance is not verified")
        for row in rows:
            if not isinstance(row, dict):
                raise TypeError("TradingView target rows must be objects")
            if (
                str(row.get("provider", "")).strip().upper() != "TRADINGVIEW"
                or str(row.get("source", "")).strip().upper()
                != "TRADINGVIEW_BROWSER_EXPORT"
                or str(row.get("session", "")).strip().lower() != wrapper_session
                or str(row.get("saved_screen_id", "")).strip() != wrapper_screen
            ):
                raise ValueError("TradingView target row identity differs from wrapper")
        input_type = "TRADINGVIEW_NORMALIZED_IMPORT"
    elif record_type == _REFRESH_TARGET_RECORD_TYPE:
        if raw.get("schema_version") != 1:
            raise ValueError("refresh target schema_version must be 1")
        if (
            raw.get("research_only") is not True
            or str(raw.get("broker_route", "")).strip().upper() != "NONE"
            or raw.get("order_submission_allowed") is not False
        ):
            raise ValueError("refresh target research-only safety sentinels failed")
        if not str(raw.get("source_run_id", "")).strip():
            raise ValueError("refresh target is missing source_run_id")
        generated_at = str(raw.get("generated_at", "")).strip()
        if not generated_at:
            raise ValueError("refresh target is missing generated_at")
        parse_timestamp(generated_at)
        input_type = _REFRESH_TARGET_RECORD_TYPE
    else:
        raise ValueError(
            "target input must be a validated TradingView import or "
            f"{_REFRESH_TARGET_RECORD_TYPE}"
        )

    wrapper_date = str(raw.get("target_session_date") or "").strip()
    if not wrapper_date and rows:
        raise ValueError("target input is missing target_session_date")
    if record_type == _REFRESH_TARGET_RECORD_TYPE:
        _verify_refresh_target_manifest(
            target_path=path,
            target_session_date=wrapper_date,
            source_run_id=str(raw["source_run_id"]).strip(),
            source_manifests=source_manifests or {},
        )
    input_record = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "record_type": input_type,
    }
    return rows, wrapper_date, input_record


def _load_target_rows_many_with_provenance(
    paths: list[Path],
    *,
    source_manifests: dict[str, dict] | None = None,
) -> tuple[list[dict], str, int, list[dict[str, str]]]:
    """Merge discovery files and retain only broad EP nominations.

    TradingView intentionally has no percentage-move filter so it cannot miss
    high-dollar movers.  The local broad move rule therefore has to run before
    the bounded IBKR request set is counted.  Repeated files are merged by the
    newest observation for each symbol; conflicting exchange identities fail
    closed.
    """

    snapshots: list[PremarketSnapshot] = []
    target_dates: set[str] = set()
    raw_count = 0
    exchanges: dict[str, set[str]] = {}
    screen_ids: dict[str, set[str]] = {}
    input_records: list[dict[str, str]] = []
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows, wrapper_date, input_record = _validated_target_wrapper(
            path,
            raw,
            source_manifests=source_manifests,
        )
        if wrapper_date:
            target_dates.add(wrapper_date)
        input_records.append(input_record)
        raw_count += len(rows)
        for row in rows:
            if not isinstance(row, dict):
                raise TypeError("target snapshot rows must be objects")
            snapshot = PremarketSnapshot.from_dict(row)
            if snapshot.target_session_date != wrapper_date:
                raise ValueError("row target_session_date differs from its wrapper")
            if (
                input_record["record_type"] == "TRADINGVIEW_NORMALIZED_IMPORT"
                and snapshot.provider.strip().upper() != "TRADINGVIEW"
            ):
                raise ValueError(
                    "TradingView target wrapper contains a non-TradingView row"
                )
            snapshots.append(snapshot)
            if snapshot.target_session_date:
                target_dates.add(snapshot.target_session_date)
            exchange = _exchange_key(
                snapshot.screen_exchange or snapshot.primary_exchange or ""
            )
            if exchange:
                exchanges.setdefault(snapshot.symbol, set()).add(exchange)
            if snapshot.saved_screen_id:
                screen_ids.setdefault(snapshot.symbol, set()).add(
                    snapshot.saved_screen_id
                )

    if len(target_dates) > 1:
        raise ValueError("target snapshots contain multiple session dates")
    exchange_conflicts = {
        symbol: sorted(values)
        for symbol, values in exchanges.items()
        if len(values) > 1
    }
    if exchange_conflicts:
        raise ValueError(
            f"conflicting target exchanges: {json.dumps(exchange_conflicts, sort_keys=True)}"
        )
    if not snapshots:
        return [], next(iter(target_dates), ""), raw_count, input_records

    as_of = max(parse_timestamp(item.observed_at) for item in snapshots)
    candidates = nominate_candidates(
        snapshots,
        as_of=as_of,
        policy=DEFAULT_POLICY,
        apply_candidate_limit=False,
    )
    cleaned = [
        {
            "symbol": candidate.snapshot.symbol,
            "expected_primary_exchange": str(
                candidate.snapshot.screen_exchange
                or candidate.snapshot.primary_exchange
                or ""
            )
            .strip()
            .upper(),
            "source_screen_id": "|".join(
                sorted(screen_ids.get(candidate.snapshot.symbol, set()))
            ),
        }
        for candidate in candidates
    ]
    return cleaned, next(iter(target_dates), ""), raw_count, input_records


def _load_target_rows_many(
    paths: list[Path], *, source_manifests: dict[str, dict] | None = None
) -> tuple[list[dict], str, int]:
    rows, session_date, raw_count, _ = _load_target_rows_many_with_provenance(
        paths,
        source_manifests=source_manifests,
    )
    return rows, session_date, raw_count


def _target_coverage_counts(
    rows: list[dict], *, requested_count: int, captured_at: str
) -> dict[str, int | bool]:
    """Count usable current quotes separately from merely serialized rows."""

    verified = 0
    for row in rows:
        try:
            snapshot = PremarketSnapshot.from_dict(row)
        except (KeyError, TypeError, ValueError):
            continue
        if premarket_move_is_verified(
            snapshot,
            as_of=captured_at,
            max_age_seconds=DEFAULT_POLICY.discovery.premarket_metrics_max_age_seconds,
            future_tolerance_seconds=(
                DEFAULT_POLICY.discovery.future_timestamp_tolerance_seconds
            ),
            require_fresh_at_as_of=True,
        ):
            verified += 1
    captured = len(rows)
    return {
        "captured_snapshot_count": captured,
        "verified_current_premarket_count": verified,
        "unverified_snapshot_count": captured - verified,
        "unresolved_target_count": max(0, requested_count - verified),
        "input_candidate_complete": verified == requested_count,
    }


def _stamp_verified_premarket_rows(rows: list[dict], *, captured_at: str) -> None:
    """Freeze only rows that remain live/current when the artifact completes."""

    for row in rows:
        try:
            snapshot = PremarketSnapshot.from_dict(row)
        except (KeyError, TypeError, ValueError):
            continue
        if not premarket_move_is_verified(
            snapshot,
            as_of=captured_at,
            max_age_seconds=DEFAULT_POLICY.discovery.premarket_metrics_max_age_seconds,
            future_tolerance_seconds=(
                DEFAULT_POLICY.discovery.future_timestamp_tolerance_seconds
            ),
            require_fresh_at_as_of=True,
        ):
            continue
        row.update(
            premarket_move_verification_status="VERIFIED",
            premarket_move_verification_source="IBKR_TARGETED_READ_ONLY",
            premarket_move_verified_at=captured_at,
        )


def _load_target_rows_unfiltered(path: Path) -> tuple[list[dict], str]:
    """Legacy parsing helper retained only for narrow unit fixtures."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("snapshots", []) if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise TypeError("target snapshot must contain a snapshots list")
    target_dates = {
        str(row.get("target_session_date", "")).strip()
        for row in rows
        if isinstance(row, dict) and row.get("target_session_date")
    }
    if len(target_dates) > 1:
        raise ValueError("target snapshot contains multiple session dates")
    cleaned: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("target snapshot rows must be objects")
        symbol = str(row.get("symbol", "")).strip().upper()
        if not symbol or symbol in seen:
            raise ValueError(f"missing or duplicate target symbol: {symbol!r}")
        seen.add(symbol)
        cleaned.append(
            {
                "symbol": symbol,
                "expected_primary_exchange": str(
                    row.get("screen_exchange") or row.get("primary_exchange") or ""
                )
                .strip()
                .upper(),
                "source_screen_id": str(row.get("saved_screen_id", "")).strip(),
            }
        )
    return cleaned, next(iter(target_dates), "")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only IBKR EP premarket capture")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port",
        default="auto",
        help="IBKR API port or 'auto' (tries 7496, 4001, 7497, then 4002)",
    )
    parser.add_argument("--client-id", type=int, default=91)
    parser.add_argument(
        "--max-captured",
        "--max-candidates",
        dest="max_captured",
        type=int,
        default=25,
        help="scanner-sample contracts to enrich; final policy ranking is separate",
    )
    parser.add_argument("--request-delay", type=float, default=0.25)
    parser.add_argument("--quote-wait-seconds", type=float, default=4.0)
    parser.add_argument(
        "--quote-batch-size",
        type=int,
        default=40,
        help="maximum concurrent streaming quote subscriptions",
    )
    parser.add_argument(
        "--scanner-code",
        action="append",
        choices=("TOP_PERC_GAIN", "HOT_BY_VOLUME", "MOST_ACTIVE"),
        dest="scanner_codes",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--symbols-from",
        type=Path,
        action="append",
        help=(
            "normalized discovery/refresh snapshot; repeat to merge after-hours "
            "and premarket files before applying the broad move rule"
        ),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        action="append",
        help=(
            "manifest.json for each EP_RESEARCH_QUOTE_REFRESH_TARGETS_V1 input; "
            "required to bind final targets to their network research run"
        ),
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help="connect read-only and write a local snapshot; default is a no-network dry run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_captured < 1 or args.max_captured > 150:
        raise SystemExit("--max-captured must be between 1 and 150")
    if args.quote_wait_seconds <= 0 or args.quote_wait_seconds > 15:
        raise SystemExit("--quote-wait-seconds must be in (0, 15]")
    if args.quote_batch_size < 1 or args.quote_batch_size > 75:
        raise SystemExit("--quote-batch-size must be between 1 and 75")
    try:
        port_candidates = _port_candidates(args.port)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    target_mode = bool(args.symbols_from)
    if target_mode and args.scanner_codes:
        raise SystemExit("--symbols-from cannot be combined with --scanner-code")
    target_rows: list[dict] = []
    target_session = ""
    target_input_rows = 0
    target_inputs: list[dict[str, str]] = []
    try:
        source_manifests = _load_refresh_source_manifests(
            [path.resolve() for path in (args.source_manifest or [])]
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid --source-manifest: {exc}") from exc
    if target_mode:
        try:
            (
                target_rows,
                target_session,
                target_input_rows,
                target_inputs,
            ) = _load_target_rows_many_with_provenance(
                [path.resolve() for path in args.symbols_from],
                source_manifests=source_manifests,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid --symbols-from snapshot: {exc}") from exc
        if len(target_rows) > args.max_captured:
            raise SystemExit(
                f"target snapshot has {len(target_rows)} rows; raise --max-captured explicitly"
            )
    if not args.capture:
        source = (
            f"{len(target_rows)} broad nomination(s) from {target_input_rows} discovery row(s)"
            if target_mode
            else "the configured rank-limited IBKR scanner union"
        )
        print(f"Dry run: would enrich {source} using a read-only IBKR connection.")
        print(
            "No broker connection or file write was performed. Add --capture to proceed."
        )
        return 0
    try:
        from ib_insync import IB, ScannerSubscription, Stock
    except ImportError as exc:
        raise SystemExit(
            "ib_insync is required only for capture; install it in the local TWS environment"
        ) from exc

    now = datetime.now(timezone.utc)
    now_ny = now.astimezone(_NY)
    if not (time(4, 0) <= now_ny.time().replace(tzinfo=None) < time(9, 25)):
        raise SystemExit(
            "IBKR EP capture is restricted to 04:00-09:25 America/New_York"
        )
    session_date = now_ny.date()
    if target_session and target_session != session_date.isoformat():
        raise SystemExit(
            f"target snapshot is for {target_session}, not today's {session_date.isoformat()} session"
        )
    output = args.output or (
        ROOT
        / "artifacts"
        / "episodic_pivot"
        / f"ibkr_snapshot_{now.strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    output = output.resolve()
    artifact_root = (ROOT / "artifacts").resolve()
    if artifact_root not in output.parents:
        raise SystemExit("--output must stay under this worktree's artifacts directory")
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing capture: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    ib = None
    connected_port = None
    attempted_ports: list[int] = []
    rows: list[dict] = []
    prepared: list[dict] = []
    errors: list[dict] = []
    scanner_unique_count = 0
    selected_contract_count = 0
    scanner_counts: dict[str, int] = {}
    successful_scans = 0
    scanner_codes = (
        []
        if target_mode
        else args.scanner_codes or ["TOP_PERC_GAIN", "HOT_BY_VOLUME", "MOST_ACTIVE"]
    )
    try:
        ib, connected_port = _connect_read_only(
            IB,
            host=args.host,
            ports=port_candidates,
            client_id=args.client_id,
            attempted_ports=attempted_ports,
        )
        if target_mode:
            selected_records = [
                {
                    "contract": Stock(
                        row["symbol"],
                        "SMART",
                        "USD",
                        primaryExchange=row["expected_primary_exchange"],
                    ),
                    "scanner_ranks": {},
                    "expected_primary_exchange": row["expected_primary_exchange"],
                    "source_screen_id": row["source_screen_id"],
                    "selection_origin": "TRADINGVIEW_TARGETED",
                }
                for row in target_rows
            ]
            scanner_unique_count = len(selected_records)
            selected_contract_count = len(selected_records)
            successful_scans = 1
        else:
            contracts: dict[object, dict] = {}
            keys_by_scanner: dict[str, list[object]] = {
                code: [] for code in scanner_codes
            }
            for code in scanner_codes:
                try:
                    subscription = ScannerSubscription(
                        numberOfRows=50,
                        instrument="STK",
                        locationCode="STK.US.MAJOR",
                        scanCode=code,
                        abovePrice=1.0,
                        aboveVolume=50_000,
                    )
                    scan_rows = ib.reqScannerData(subscription)
                    scanner_counts[code] = len(scan_rows)
                    successful_scans += 1
                    for rank, item in enumerate(scan_rows, start=1):
                        contract = item.contractDetails.contract
                        if contract.secType == "STK" and contract.currency == "USD":
                            key = contract.conId or contract.symbol
                            record = contracts.setdefault(
                                key,
                                {
                                    "contract": contract,
                                    "scanner_ranks": {},
                                    "selection_origin": "IBKR_SCANNER_SAMPLE",
                                },
                            )
                            record["scanner_ranks"][code] = rank
                            keys_by_scanner[code].append(key)
                except Exception as exc:  # noqa: BLE001 - isolate scanner failures.
                    scanner_counts[code] = 0
                    errors.append(
                        {
                            "scanner_code": code,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

            scanner_unique_count = len(contracts)
            selected_keys = _round_robin_keys(
                keys_by_scanner, scanner_codes, args.max_captured
            )
            selected_records = [contracts[key] for key in selected_keys]
            selected_contract_count = len(selected_records)
        for scanner_record in selected_records:
            contract = scanner_record["contract"]
            if datetime.now(timezone.utc).astimezone(_NY).time().replace(
                tzinfo=None
            ) >= time(9, 25):
                errors.append({"error": "CAPTURE_WINDOW_CLOSED_BEFORE_NEXT_SYMBOL"})
                break
            try:
                scanner_con_id = int(contract.conId or 0)
                scanner_symbol = str(contract.symbol).upper()
                qualified = ib.qualifyContracts(contract)
                if len(qualified) != 1:
                    raise ValueError(
                        f"contract qualification returned {len(qualified)} matches"
                    )
                contract = qualified[0]
                if scanner_con_id and int(contract.conId or 0) != scanner_con_id:
                    raise ValueError("qualified conId differs from scanner conId")
                if str(contract.symbol).upper() != scanner_symbol:
                    raise ValueError("qualified symbol differs from scanner symbol")
                details = ib.reqContractDetails(contract)
                if len(details) != 1:
                    raise ValueError(
                        f"contract details returned {len(details)} matches"
                    )
                detail = details[0]
                detail_contract = detail.contract
                if int(detail_contract.conId or 0) != int(contract.conId or 0):
                    raise ValueError("detail conId differs from qualified conId")
                if str(detail_contract.symbol).upper() != scanner_symbol:
                    raise ValueError("detail symbol differs from scanner symbol")
                company_name = detail.longName or contract.symbol
                primary_exchange = (contract.primaryExchange or "").strip()
                expected_primary_exchange = scanner_record.get(
                    "expected_primary_exchange", ""
                )
                if expected_primary_exchange and _exchange_key(
                    primary_exchange
                ) != _exchange_key(expected_primary_exchange):
                    raise ValueError(
                        "IBKR primary exchange does not match TradingView identity"
                    )
                valid_exchanges = str(getattr(detail, "validExchanges", "") or "")
                allowed_order_types = str(getattr(detail, "orderTypes", "") or "")
                valid_exchange_tokens = {
                    value.strip().upper()
                    for value in valid_exchanges.split(",")
                    if value.strip()
                }
                identity_valid = bool(
                    contract.conId
                    and contract.secType == "STK"
                    and contract.currency == "USD"
                    and primary_exchange
                    and primary_exchange.upper() != "SMART"
                    and "SMART" in valid_exchange_tokens
                )
                identity_status = (
                    "UNIQUE_IBKR_MATCH"
                    if identity_valid
                    else "INCOMPLETE_IBKR_IDENTITY"
                )
                daily_bars = ib.reqHistoricalData(
                    contract,
                    endDateTime="",
                    durationStr="1 Y",
                    barSizeSetting="1 day",
                    # IBKR TRADES adjusts splits but not dividends.  The
                    # historical study uses adjusted OHLCV, so ADJUSTED_LAST
                    # is required for a comparable prior-close/ATR basis.
                    whatToShow=_DAILY_WHAT_TO_SHOW,
                    useRTH=True,
                    formatDate=1,
                )
                extended_bars = ib.reqHistoricalData(
                    contract,
                    endDateTime="",
                    durationStr="2 D",
                    barSizeSetting="5 mins",
                    whatToShow="TRADES",
                    useRTH=False,
                    formatDate=2,
                )
                daily = _daily_metrics(daily_bars, session_date)
                premarket = _premarket_metrics(
                    extended_bars, session_date, daily["previous_close"]
                )
                prepared.append(
                    {
                        "contract": contract,
                        "company_name": company_name,
                        "tradeable": identity_valid,
                        "contract_identity_status": identity_status,
                        "resolved_symbol": str(contract.symbol).upper(),
                        "contract_sec_type": str(contract.secType),
                        "contract_currency": str(contract.currency),
                        "primary_exchange": primary_exchange,
                        "valid_exchanges": valid_exchanges,
                        "allowed_order_types": allowed_order_types,
                        "scanner_ranks": scanner_record["scanner_ranks"],
                        "selection_origin": scanner_record.get(
                            "selection_origin", "IBKR_SCANNER_SAMPLE"
                        ),
                        "source_screen_id": scanner_record.get("source_screen_id", ""),
                        "daily": daily,
                        "premarket": premarket,
                    }
                )
                ib.sleep(args.request_delay)
            except Exception as exc:  # noqa: BLE001 - isolate per-symbol failures.
                errors.append(
                    {"symbol": contract.symbol, "error": f"{type(exc).__name__}: {exc}"}
                )

        for batch_start in range(0, len(prepared), args.quote_batch_size):
            quote_batch = prepared[batch_start : batch_start + args.quote_batch_size]
            if datetime.now(timezone.utc).astimezone(_NY).time().replace(
                tzinfo=None
            ) >= time(9, 25):
                errors.append(
                    {"error": "CAPTURE_WINDOW_CLOSED_BEFORE_BATCH_QUOTE_REFRESH"}
                )
                break
            # Streaming watchlist requests are deliberate: IBKR documents
            # output tick 49 (halted) as available only for watchlist data.
            # Batching stays below account market-data line limits.
            ticker_by_conid, subscribed_contracts = _subscribe_market_data_batch(
                ib, quote_batch, errors
            )
            try:
                deadline = datetime.now(timezone.utc) + timedelta(
                    seconds=args.quote_wait_seconds
                )
                while datetime.now(timezone.utc) < deadline:
                    statuses = [
                        _halt_status(getattr(ticker, "halted", None))[0]
                        for ticker in ticker_by_conid.values()
                    ]
                    quotes_ready = all(
                        _finite(getattr(ticker, "bid", 0)) > 0
                        and _finite(getattr(ticker, "ask", 0)) > 0
                        for ticker in ticker_by_conid.values()
                    )
                    data_types_ready = all(
                        int(_finite(getattr(ticker, "marketDataType", 0), 0))
                        in _MARKET_DATA_STATUS
                        for ticker in ticker_by_conid.values()
                    )
                    if (
                        quotes_ready
                        and data_types_ready
                        and all(status != "UNKNOWN" for status in statuses)
                    ):
                        break
                    ib.sleep(0.1)
            except Exception as exc:  # noqa: BLE001 - retain any sibling ticker state.
                errors.append(
                    {
                        "error": f"BATCH_QUOTE_WAIT_FAILED:{type(exc).__name__}",
                        "batch_start": batch_start,
                    }
                )
            finally:
                # Values remain on the Ticker objects after cancellation.
                _cancel_market_data_batch(ib, subscribed_contracts, errors)
            batch_finished = datetime.now(timezone.utc)
            if batch_finished.astimezone(_NY).time().replace(tzinfo=None) >= time(
                9, 29
            ):
                errors.append(
                    {"error": "BATCH_QUOTE_REFRESH_FINISHED_TOO_LATE; rows discarded"}
                )
                break
            for item in quote_batch:
                contract = item["contract"]
                ticker = ticker_by_conid.get(contract.conId)
                if ticker is None:
                    errors.append(
                        {
                            "symbol": contract.symbol,
                            "error": "MISSING_BATCH_QUOTE",
                        }
                    )
                    continue
                premarket = item["premarket"]
                halt_status, halt_raw = _halt_status(getattr(ticker, "halted", None))
                quote_time = getattr(ticker, "time", None)
                if isinstance(quote_time, datetime):
                    if quote_time.tzinfo is None:
                        quote_time = quote_time.replace(tzinfo=timezone.utc)
                    observed_at = quote_time.astimezone(timezone.utc)
                    quote_timestamp_source = "IBKR_TICKER_TIME"
                else:
                    observed_at = datetime.fromisoformat(
                        premarket["premarket_metrics_at"].replace("Z", "+00:00")
                    )
                    quote_timestamp_source = "PREMARKET_BAR_FALLBACK"
                market_data_status = _MARKET_DATA_STATUS.get(
                    int(_finite(ticker.marketDataType, 0)), "UNKNOWN"
                )
                if quote_timestamp_source != "IBKR_TICKER_TIME":
                    market_data_status = "UNKNOWN_TIMESTAMP"
                rows.append(
                    {
                        "symbol": contract.symbol.upper(),
                        "company_name": item["company_name"],
                        "observed_at": observed_at.isoformat().replace("+00:00", "Z"),
                        "last": _finite(ticker.last, premarket["premarket_last"]),
                        "quote_previous_close": _finite(
                            getattr(ticker, "close", None), None
                        ),
                        "bid": _finite(ticker.bid),
                        "ask": _finite(ticker.ask),
                        "bid_size": int(_finite(ticker.bidSize)),
                        "ask_size": int(_finite(ticker.askSize)),
                        "market_data_status": market_data_status,
                        "quote_timestamp_source": quote_timestamp_source,
                        "halted": halt_status in {"GENERAL_HALT", "VOLATILITY_HALT"},
                        "halt_status": halt_status,
                        "halt_raw": halt_raw,
                        "tradeable": item["tradeable"],
                        "source": (
                            "IBKR_TARGETED_READ_ONLY"
                            if item["selection_origin"] == "TRADINGVIEW_TARGETED"
                            else "IBKR_SCANNER_SAMPLE_READ_ONLY"
                        ),
                        "provider": "IBKR",
                        "session": "premarket",
                        "target_session_date": session_date.isoformat(),
                        "saved_screen_id": item["source_screen_id"],
                        "scanner_sources": sorted(item["scanner_ranks"]),
                        "scanner_ranks": item["scanner_ranks"],
                        "price_basis": (
                            "IBKR_ADJUSTED_LAST_DAILY_WITH_LIVE_TRADES_QUOTE"
                        ),
                        "daily_price_basis": _DAILY_PRICE_BASIS,
                        "atr_reference_close": item["daily"]["previous_close"],
                        "daily_data_status": "VERIFIED",
                        "daily_data_observed_at": batch_finished.isoformat().replace(
                            "+00:00", "Z"
                        ),
                        "daily_source_symbol": contract.symbol.upper(),
                        "contract_con_id": contract.conId,
                        "primary_exchange": item["primary_exchange"],
                        "contract_identity_status": item["contract_identity_status"],
                        "resolved_symbol": item["resolved_symbol"],
                        "contract_sec_type": item["contract_sec_type"],
                        "contract_currency": item["contract_currency"],
                        "valid_exchanges": item["valid_exchanges"],
                        "allowed_order_types": item["allowed_order_types"],
                        **item["daily"],
                        **{k: v for k, v in premarket.items() if k != "premarket_last"},
                    }
                )
    finally:
        if ib is not None and ib.isConnected():
            ib.disconnect()

    captured_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    _stamp_verified_premarket_rows(rows, captured_at=captured_at)
    target_coverage = _target_coverage_counts(
        rows,
        requested_count=len(target_rows),
        captured_at=captured_at,
    )
    payload = {
        "schema_version": 1,
        "record_type": _IBKR_RECORD_TYPE,
        "provider": "IBKR",
        "captured_at": captured_at,
        "target_session_date": session_date.isoformat(),
        "mode": "IBKR_READ_ONLY_SHADOW",
        "connection": {
            "host": args.host,
            "port": connected_port,
            "attempted_ports": attempted_ports,
            "selected_port": connected_port,
            "readonly": True,
            "readonly_requested": True,
            "connected": connected_port is not None,
        },
        "inputs": target_inputs,
        "source_manifests": [
            {
                "run_id": run_id,
                "path": record["path"],
                "sha256": sha256_file(record["path"]),
            }
            for run_id, record in sorted(source_manifests.items())
        ],
        "scanner_codes": scanner_codes,
        "coverage": {
            "mode": (
                "TARGETED_TRADINGVIEW_CANDIDATES"
                if target_mode
                else "NON_EXHAUSTIVE_IBKR_SCANNER_SAMPLE"
            ),
            "exchange_complete": False,
            **(
                target_coverage
                if target_mode
                else {
                    **target_coverage,
                    "input_candidate_complete": False,
                }
            ),
            "input_discovery_row_count": target_input_rows,
            "requested_target_count": len(target_rows),
            "scanner_limit_per_code": 50,
            "scanner_counts": scanner_counts,
            "successful_scans": successful_scans,
            "unique_scanner_contracts": scanner_unique_count,
            "selected_for_detailed_capture": selected_contract_count,
            "omitted_before_detailed_capture": max(
                0, scanner_unique_count - selected_contract_count
            ),
            "selection_method": (
                "TRADINGVIEW_CANDIDATE_LIST"
                if target_mode
                else "ROUND_ROBIN_BY_SCANNER_RANK"
            ),
            "warning": (
                "Targeted mode covers only the validated TradingView input list; it does "
                "not prove that TradingView covered the full exchange."
                if target_mode
                else "IBKR API scanner results are rank-limited samples and do not prove "
                "coverage of every symbol meeting the EP move/volume rule."
            ),
        },
        "snapshots": rows,
        "errors": errors,
    }
    output.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Captured {len(rows)} snapshot(s); {len(errors)} error(s): {output}")
    if target_mode and not target_coverage["input_candidate_complete"]:
        print(
            "Warning: targeted IBKR coverage is partial; only verified current "
            "premarket rows may continue."
        )
    print("Safety: connected read-only and exposed no order-submission path.")
    # A completed targeted capture is a valid degraded artifact even when no
    # target produced a usable live row. The morning flow excludes those rows,
    # carries the coverage warning, and continues with verified TV candidates.
    return 0 if target_mode or rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
