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


_NY = ZoneInfo("America/New_York")
_MARKET_DATA_STATUS = {
    1: "LIVE",
    2: "FROZEN",
    3: "DELAYED",
    4: "DELAYED_FROZEN",
}


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
    token = "".join(character for character in str(value).upper() if character.isalnum())
    aliases = {
        "NYSEARCA": "ARCA",
        "NASDAQGS": "NASDAQ",
        "NASDAQGM": "NASDAQ",
        "NASDAQCM": "NASDAQ",
    }
    return aliases.get(token, token)


def _round_robin_keys(
    keys_by_scanner: dict[str, list[object]], scanner_codes: list[str], limit: int
) -> list[object]:
    """Interleave scanner ranks so the first scan cannot consume the whole cap."""

    selected: list[object] = []
    seen: set[object] = set()
    max_rows = max((len(keys_by_scanner.get(code, [])) for code in scanner_codes), default=0)
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
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    frame = frame[frame["date"] < session_date].sort_values("date").reset_index(drop=True)
    if len(frame) < 64:
        raise ValueError("fewer than 64 completed daily bars")
    previous_close = float(frame.iloc[-1]["close"])
    prior_two_day_low = float(frame.tail(2)["low"].min())
    previous = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous).abs(),
            (frame["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    daily_change = 100.0 * (frame["close"] / frame["close"].shift(1) - 1.0)
    prior_avg_volume_100 = frame["volume"].shift(1).rolling(100, min_periods=50).mean()
    prior_ep = (daily_change.abs() >= 8.0) & (frame["volume"] >= 3.0 * prior_avg_volume_100)
    prior_ep_positions = frame.index[prior_ep]
    sessions_since_prior_ep = (
        int((len(frame) - 1) - prior_ep_positions[-1])
        if len(prior_ep_positions)
        else None
    )
    return {
        "previous_close": previous_close,
        "prior_two_day_low": prior_two_day_low,
        "atr_14": float(true_range.tail(14).mean()),
        "avg_volume_20": float(frame["volume"].tail(20).mean()),
        "addv_63": float((frame["close"] * frame["volume"]).tail(63).mean()),
        "prior_63d_return_pct": 100.0 * (previous_close / float(frame.iloc[-64]["close"]) - 1.0),
        "sessions_since_prior_ep": sessions_since_prior_ep,
    }


def _premarket_metrics(
    bars, session_date, previous_close: float | None = None
):  # type: ignore[no-untyped-def]
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
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("snapshots", []) if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ValueError("target snapshot must contain a snapshots list")
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
            raise ValueError("target snapshot rows must be objects")
        symbol = str(row.get("symbol", "")).strip().upper()
        if not symbol or symbol in seen:
            raise ValueError(f"missing or duplicate target symbol: {symbol!r}")
        seen.add(symbol)
        cleaned.append(
            {
                "symbol": symbol,
                "expected_primary_exchange": str(
                    row.get("screen_exchange") or row.get("primary_exchange") or ""
                ).strip().upper(),
                "source_screen_id": str(row.get("saved_screen_id", "")).strip(),
            }
        )
    return cleaned, next(iter(target_dates), "")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only IBKR EP premarket capture")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7497, help="paper TWS default is 7497")
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
        "--scanner-code",
        action="append",
        choices=("TOP_PERC_GAIN", "HOT_BY_VOLUME", "MOST_ACTIVE"),
        dest="scanner_codes",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--symbols-from",
        type=Path,
        help="normalized TradingView snapshot; enrich only those symbols instead of using IBKR scanners",
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
    if args.symbols_from and args.scanner_codes:
        raise SystemExit("--symbols-from cannot be combined with --scanner-code")
    target_rows: list[dict] = []
    target_session = ""
    if args.symbols_from:
        try:
            target_rows, target_session = _load_target_rows(args.symbols_from.resolve())
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid --symbols-from snapshot: {exc}") from exc
        if len(target_rows) > args.max_captured:
            raise SystemExit(
                f"target snapshot has {len(target_rows)} rows; raise --max-captured explicitly"
            )
    if not args.capture:
        source = (
            f"{len(target_rows)} TradingView target(s)"
            if args.symbols_from
            else "the configured rank-limited IBKR scanner union"
        )
        print(f"Dry run: would enrich {source} using a read-only IBKR connection.")
        print("No broker connection or file write was performed. Add --capture to proceed.")
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
        raise SystemExit("IBKR EP capture is restricted to 04:00-09:25 America/New_York")
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

    ib = IB()
    rows: list[dict] = []
    prepared: list[dict] = []
    errors: list[dict] = []
    scanner_unique_count = 0
    selected_contract_count = 0
    scanner_counts: dict[str, int] = {}
    successful_scans = 0
    scanner_codes = (
        []
        if target_rows
        else args.scanner_codes
        or ["TOP_PERC_GAIN", "HOT_BY_VOLUME", "MOST_ACTIVE"]
    )
    try:
        ib.connect(args.host, args.port, clientId=args.client_id, readonly=True, timeout=10)
        if target_rows:
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
                except Exception as exc:
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
            if datetime.now(timezone.utc).astimezone(_NY).time().replace(tzinfo=None) >= time(9, 25):
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
                    whatToShow="TRADES",
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
                        "source_screen_id": scanner_record.get(
                            "source_screen_id", ""
                        ),
                        "daily": daily,
                        "premarket": premarket,
                    }
                )
                ib.sleep(args.request_delay)
            except Exception as exc:
                errors.append(
                    {"symbol": contract.symbol, "error": f"{type(exc).__name__}: {exc}"}
                )

        if prepared:
            if datetime.now(timezone.utc).astimezone(_NY).time().replace(tzinfo=None) >= time(9, 25):
                errors.append({"error": "CAPTURE_WINDOW_CLOSED_BEFORE_BATCH_QUOTE_REFRESH"})
            else:
                # Streaming watchlist requests are deliberate: IBKR documents
                # output tick 49 (halted) as available only for watchlist data.
                # A one-shot reqTickers snapshot can leave Ticker.halted as NaN.
                ticker_by_conid = {}
                subscribed_contracts = []
                try:
                    for item in prepared:
                        contract = item["contract"]
                        ticker_by_conid[contract.conId] = ib.reqMktData(
                            contract,
                            genericTickList="",
                            snapshot=False,
                            regulatorySnapshot=False,
                        )
                        # ib_insync initializes this field to LIVE (1) before
                        # any IBKR callback.  Reset it so only an explicit
                        # marketDataType callback can satisfy the live-data gate.
                        ticker_by_conid[contract.conId].marketDataType = 0
                        subscribed_contracts.append(contract)
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
                finally:
                    # Values remain on the Ticker objects after cancellation.
                    for contract in subscribed_contracts:
                        ib.cancelMktData(contract)
                batch_finished = datetime.now(timezone.utc)
                if batch_finished.astimezone(_NY).time().replace(tzinfo=None) >= time(9, 29):
                    errors.append({"error": "BATCH_QUOTE_REFRESH_FINISHED_TOO_LATE; rows discarded"})
                else:
                    for item in prepared:
                        contract = item["contract"]
                        ticker = ticker_by_conid.get(contract.conId)
                        if ticker is None:
                            errors.append({"symbol": contract.symbol, "error": "MISSING_BATCH_QUOTE"})
                            continue
                        premarket = item["premarket"]
                        halt_status, halt_raw = _halt_status(
                            getattr(ticker, "halted", None)
                        )
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
                                    if item["selection_origin"]
                                    == "TRADINGVIEW_TARGETED"
                                    else "IBKR_SCANNER_SAMPLE_READ_ONLY"
                                ),
                                "provider": "IBKR",
                                "session": "premarket",
                                "target_session_date": session_date.isoformat(),
                                "saved_screen_id": item["source_screen_id"],
                                "scanner_sources": sorted(item["scanner_ranks"]),
                                "scanner_ranks": item["scanner_ranks"],
                                "price_basis": "IBKR_TRADES",
                                "contract_con_id": contract.conId,
                                "primary_exchange": item["primary_exchange"],
                                "contract_identity_status": item[
                                    "contract_identity_status"
                                ],
                                "resolved_symbol": item["resolved_symbol"],
                                "contract_sec_type": item["contract_sec_type"],
                                "contract_currency": item["contract_currency"],
                                "valid_exchanges": item["valid_exchanges"],
                                "allowed_order_types": item[
                                    "allowed_order_types"
                                ],
                                **item["daily"],
                                **{k: v for k, v in premarket.items() if k != "premarket_last"},
                            }
                        )
    finally:
        if ib.isConnected():
            ib.disconnect()

    payload = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "mode": "IBKR_READ_ONLY_SHADOW",
        "scanner_codes": scanner_codes,
        "coverage": {
            "mode": (
                "TARGETED_TRADINGVIEW_CANDIDATES"
                if target_rows
                else "NON_EXHAUSTIVE_IBKR_SCANNER_SAMPLE"
            ),
            "exchange_complete": False,
            "input_candidate_complete": (
                len(rows) == len(target_rows) if target_rows else False
            ),
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
                if target_rows
                else "ROUND_ROBIN_BY_SCANNER_RANK"
            ),
            "warning": (
                "Targeted mode covers only the validated TradingView input list; it does "
                "not prove that TradingView covered the full exchange."
                if target_rows
                else "IBKR API scanner results are rank-limited samples and do not prove "
                "coverage of every symbol meeting the EP move/volume rule."
            ),
        },
        "snapshots": rows,
        "errors": errors,
    }
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Captured {len(rows)} snapshot(s); {len(errors)} error(s): {output}")
    print("Safety: connected read-only and exposed no order-submission path.")
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
