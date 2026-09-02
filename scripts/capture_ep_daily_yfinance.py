"""Fetch fresh yfinance daily bars for premarket-verified EP nominations.

This is a research-only, dry-by-default adapter.  It never reads the local
master-price cache and contains no broker, order, staging, or publishing path.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from episodic_pivot.config import DEFAULT_POLICY
from episodic_pivot.daily_prices import (
    YFINANCE_DAILY_PRICE_BASIS,
    enrich_snapshots_from_yfinance,
)
from episodic_pivot.manifest import sha256_file
from episodic_pivot.premarket import (
    nominate_candidates,
    premarket_move_is_verified,
)
from episodic_pivot.schema import PremarketSnapshot, parse_timestamp
from episodic_pivot.tradingview import result_counts_are_verified

_IBKR_RECORD_TYPE = "EP_IBKR_PREMARKET_CAPTURE_V1"
_TRADINGVIEW_SCREEN_BY_SESSION = {
    "premarket": "yftOvM3e",
    "after_hours": "Hqgnyp7Y",
}


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


def _load_discovery_inputs(
    paths: list[Path],
) -> tuple[
    list[PremarketSnapshot],
    str,
    int,
    list[dict[str, str]],
    tuple[str, ...],
]:
    snapshots: list[PremarketSnapshot] = []
    target_dates: set[str] = set()
    raw_count = 0
    exchanges: dict[str, set[str]] = {}
    input_records: list[dict[str, str]] = []
    saw_ibkr = False
    saw_premarket_tradingview = False
    warnings: set[str] = set()

    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("snapshots"), list):
            raise TypeError("each discovery input must be a normalized snapshot object")
        wrapper_provider = str(raw.get("provider", "")).upper()
        wrapper_mode = str(raw.get("mode", "")).upper()
        wrapper_record_type = str(raw.get("record_type", "")).upper()
        is_tradingview = wrapper_provider == "TRADINGVIEW"
        is_ibkr = (
            wrapper_provider == "IBKR"
            or wrapper_mode == "IBKR_READ_ONLY_SHADOW"
            or wrapper_record_type == _IBKR_RECORD_TYPE
        )
        if not (is_tradingview or is_ibkr):
            raise ValueError(
                "daily yfinance capture accepts validated TradingView or read-only IBKR inputs only"
            )
        rows = raw["snapshots"]
        if is_tradingview:
            wrapper_session = str(raw.get("session", "")).strip().lower()
            wrapper_screen = str(raw.get("saved_screen_id", "")).strip()
            if wrapper_session != "premarket":
                raise ValueError(
                    "daily yfinance capture rejects unrefreshed after-hours rows"
                )
            reported = raw.get("reported_result_count")
            extracted = raw.get("extracted_row_count")
            if (
                _TRADINGVIEW_SCREEN_BY_SESSION.get(wrapper_session) != wrapper_screen
                or raw.get("result_count_verified") is not True
                or not result_counts_are_verified(
                    reported_result_count=reported,
                    post_download_result_count=raw.get("post_download_result_count"),
                    extracted_row_count=extracted,
                    verification_status=raw.get("result_count_verification"),
                    require_both_observations=True,
                )
                or extracted != len(rows)
            ):
                raise ValueError("TradingView input count/provenance is not verified")
            saw_premarket_tradingview = True
        else:
            saw_ibkr = True
            coverage = raw.get("coverage")
            connection = raw.get("connection")
            if (
                wrapper_record_type != _IBKR_RECORD_TYPE
                or wrapper_provider != "IBKR"
                or wrapper_mode != "IBKR_READ_ONLY_SHADOW"
                or not isinstance(coverage, dict)
                or coverage.get("mode") != "TARGETED_TRADINGVIEW_CANDIDATES"
                or not isinstance(connection, dict)
                or connection.get("readonly") is not True
            ):
                raise ValueError("IBKR input is not a targeted read-only refresh")
            if coverage.get("input_candidate_complete") is not True:
                warnings.add("IBKR_PARTIAL_CARRYOVER_COVERAGE")
        wrapper_date = str(raw.get("target_session_date", "")).strip()
        if not wrapper_date and is_ibkr:
            row_dates = {
                str(row.get("target_session_date", "")).strip()
                for row in rows
                if isinstance(row, dict) and row.get("target_session_date")
            }
            if len(row_dates) == 1:
                wrapper_date = next(iter(row_dates))
        if not wrapper_date:
            raise ValueError("discovery input is missing target_session_date")
        target_dates.add(wrapper_date)
        captured_at = str(raw.get("captured_at", "")).strip()
        if not captured_at:
            raise ValueError("discovery input is missing captured_at")
        wrapper_captured_at = parse_timestamp(captured_at)
        raw_count += len(rows)
        input_records.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
        for row in rows:
            if not isinstance(row, dict):
                raise TypeError("TradingView snapshot rows must be objects")
            snapshot = PremarketSnapshot.from_dict(row)
            if snapshot.target_session_date != wrapper_date:
                raise ValueError("row target_session_date differs from its wrapper")
            if is_tradingview and snapshot.provider.upper() != "TRADINGVIEW":
                raise ValueError("TradingView wrapper contains a non-TradingView row")
            if is_tradingview and (
                snapshot.source != "TRADINGVIEW_BROWSER_EXPORT"
                or snapshot.session != wrapper_session
                or snapshot.saved_screen_id != wrapper_screen
            ):
                raise ValueError("TradingView row identity differs from its wrapper")
            if is_ibkr and snapshot.provider.upper() != "IBKR":
                raise ValueError("IBKR wrapper contains a non-IBKR row")
            if (
                is_tradingview
                and wrapper_session == "premarket"
                and not premarket_move_is_verified(
                    snapshot,
                    as_of=wrapper_captured_at,
                    max_age_seconds=DEFAULT_POLICY.discovery.premarket_metrics_max_age_seconds,
                    future_tolerance_seconds=(
                        DEFAULT_POLICY.discovery.future_timestamp_tolerance_seconds
                    ),
                    require_fresh_at_as_of=True,
                )
            ):
                raise ValueError(
                    "TradingView row was not current when its wrapper completed capture"
                )
            if is_ibkr and not premarket_move_is_verified(
                snapshot,
                as_of=wrapper_captured_at,
                max_age_seconds=DEFAULT_POLICY.discovery.premarket_metrics_max_age_seconds,
                future_tolerance_seconds=(
                    DEFAULT_POLICY.discovery.future_timestamp_tolerance_seconds
                ),
                require_fresh_at_as_of=True,
            ):
                # Partial/unknown quote coverage is expected. Such rows remain
                # in the immutable IBKR artifact but cannot enter ATR or news.
                continue
            snapshots.append(snapshot)
            exchange = _exchange_key(
                snapshot.screen_exchange or snapshot.primary_exchange or ""
            )
            if exchange:
                exchanges.setdefault(snapshot.symbol, set()).add(exchange)

    if len(target_dates) != 1:
        raise ValueError("discovery inputs must target exactly one NYSE session")
    exchange_conflicts = {
        symbol: sorted(values)
        for symbol, values in exchanges.items()
        if len(values) > 1
    }
    if exchange_conflicts:
        raise ValueError(
            f"conflicting discovery exchanges: {json.dumps(exchange_conflicts, sort_keys=True)}"
        )

    if snapshots:
        as_of = max(parse_timestamp(snapshot.observed_at) for snapshot in snapshots)
        candidates = nominate_candidates(
            snapshots,
            as_of=as_of,
            policy=DEFAULT_POLICY,
            apply_candidate_limit=False,
            require_verified_premarket_move=True,
        )
        selected = [candidate.snapshot for candidate in candidates]
    else:
        selected = []
    if not saw_ibkr:
        warnings.add("IBKR_CARRYOVER_NOT_INCLUDED")
    if not saw_premarket_tradingview:
        warnings.add("TRADINGVIEW_PREMARKET_NOT_INCLUDED")
    return (
        selected,
        next(iter(target_dates)),
        raw_count,
        input_records,
        tuple(sorted(warnings)),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fresh yfinance prior-ATR capture for EP shadow research"
    )
    parser.add_argument(
        "--snapshot",
        required=True,
        action="append",
        type=Path,
        help=(
            "validated current-premarket TradingView or targeted read-only IBKR "
            "snapshot; repeat to merge sources"
        ),
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help="perform read-only yfinance requests and write a local artifact",
    )
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--batch-retries", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "artifacts" / "episodic_pivot" / "yfinance_cache",
        help="writable yfinance cookie/timezone metadata cache; never stores OHLCV",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        (
            snapshots,
            target_date,
            raw_count,
            input_records,
            coverage_warnings,
        ) = _load_discovery_inputs([path.resolve() for path in args.snapshot])
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid EP discovery input: {exc}") from exc

    if not args.capture:
        print(
            f"Dry run: would fetch fresh adjusted daily bars for {len(snapshots)} "
            f"broad nomination(s) from {raw_count} discovery row(s)."
        )
        print(
            "No network request or file write was performed. Add --capture to proceed."
        )
        return 0

    try:
        session_date = datetime.fromisoformat(target_date).date()
        cache_dir = args.cache_dir.resolve()
        allowed_root = (ROOT / "artifacts").resolve()
        if cache_dir == allowed_root or allowed_root not in cache_dir.parents:
            raise ValueError("--cache-dir must stay under this worktree's artifacts")
        result = enrich_snapshots_from_yfinance(
            snapshots,
            session_date=session_date,
            batch_size=args.batch_size,
            batch_retries=args.batch_retries,
            timeout_seconds=args.timeout_seconds,
            provider_cache_dir=cache_dir,
        )
    except (ImportError, TypeError, ValueError) as exc:
        raise SystemExit(f"yfinance daily capture could not start: {exc}") from exc

    output = args.output or (
        ROOT
        / "artifacts"
        / "episodic_pivot"
        / f"yfinance_snapshot_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    output = output.resolve()
    allowed_root = (ROOT / "artifacts").resolve()
    if output == allowed_root or allowed_root not in output.parents:
        raise SystemExit("--output must stay under this worktree's artifacts directory")
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        import yfinance as yf

        version = str(getattr(yf, "__version__", "unknown"))
    except ImportError:
        version = "unknown"
    payload = {
        "schema_version": 1,
        "record_type": "EP_YFINANCE_DAILY_ENRICHMENT_V1",
        "mode": "YFINANCE_DAILY_RESEARCH_ONLY",
        "provider": "YFINANCE",
        "provider_version": version,
        "captured_at": result.fetched_at,
        "target_session_date": target_date,
        "daily_price_basis": YFINANCE_DAILY_PRICE_BASIS,
        "inputs": input_records,
        "warnings": list(coverage_warnings),
        "request": {
            "interval": "1d",
            "auto_adjust": True,
            "repair": True,
            "event_session_excluded": True,
            "batch_size": args.batch_size,
            "batch_retries": args.batch_retries,
            "timeout_seconds": args.timeout_seconds,
            "local_price_cache_used": False,
            "provider_metadata_cache": str(cache_dir),
        },
        "coverage": {
            "input_discovery_row_count": raw_count,
            "broad_nomination_count": result.requested_count,
            "daily_metrics_verified_count": result.verified_count,
            "daily_metrics_unresolved_count": (
                result.requested_count - result.verified_count
            ),
            "complete": result.requested_count == result.verified_count,
        },
        "snapshots": [snapshot.to_dict() for snapshot in result.snapshots],
        "errors": [error.to_dict() for error in result.errors],
        "safety": {
            "research_only": True,
            "local_price_cache_used": False,
            "broker_contacted": False,
            "broker_route": "NONE",
            "order_submission_allowed": False,
            "order_staging_performed": False,
            "production_deployed": False,
        },
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Captured prior daily metrics for {result.verified_count}/"
        f"{result.requested_count} broad nomination(s): {output}"
    )
    if result.errors:
        print(
            f"{len(result.errors)} symbol(s) remain ATR-unresolved and will be "
            "retained for audit without consuming news budget."
        )
    print("Safety: no broker, order, staging, cache, publishing, or production write.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
