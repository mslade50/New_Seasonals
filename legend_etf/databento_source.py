"""Cost-gated Databento adapter for immutable prior-session futures signals."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .calendar import require_full_entry_session
from .config import (
    DATABENTO_DATASET,
    DATABENTO_SCHEMA,
    MARKETS,
    NY_TZ,
    STRATEGY_VERSION,
)
from .core import evaluate_futures_setup, latest_observed_futures_rth_session
from .storage import (
    atomic_write_json,
    canonical_json,
    content_hash,
    exclusive_file_lock,
    finalize_plan,
    read_json,
)

DATASET = DATABENTO_DATASET
SCHEMA = DATABENTO_SCHEMA
KEYRING_SERVICE = "New_Seasonals.Databento"
KEYRING_USERNAME = "prod-001"
PAID_CONFIRMATION = "I_APPROVE_DATABENTO_CHARGE"
CACHE_INTEGRITY_PROTOCOL = "legend-futures-cache-integrity-v1"
CACHE_REFRESH_OVERLAP = timedelta(days=30)


@dataclass(frozen=True)
class Quote:
    root: str
    cost_usd: float
    billable_bytes: int


def available_end(client: Any, desired_end: datetime) -> datetime:
    """Clamp requests to the historical dataset's current available end."""

    details = client.metadata.get_dataset_range(DATASET)
    raw = details.get("end")
    if not raw:
        raise RuntimeError("Databento did not return the dataset available end")
    boundary = pd.Timestamp(raw)
    if boundary.tz is None:
        boundary = boundary.tz_localize("UTC")
    # The API end is exclusive. Stay one second inside the advertised range.
    safe = boundary.tz_convert("UTC") - pd.Timedelta(seconds=1)
    desired = pd.Timestamp(desired_end)
    if desired.tz is None:
        raise ValueError("desired Databento end must be timezone-aware")
    return min(desired.tz_convert("UTC"), safe).to_pydatetime()


def load_api_key() -> str:
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    except ImportError:
        pass
    key = os.environ.get("DATABENTO_API_KEY", "").strip()
    if key:
        return key
    try:
        import keyring
    except ImportError as exc:  # pragma: no cover - production dependency guard
        raise RuntimeError("keyring is required when DATABENTO_API_KEY is unset") from exc
    key = (keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME) or "").strip()
    if not key:
        raise RuntimeError(
            "Databento key not found in DATABENTO_API_KEY or the operating-system vault"
        )
    return key


def make_client() -> Any:
    try:
        import databento as db
    except ImportError as exc:  # pragma: no cover - production dependency guard
        raise RuntimeError("install requirements-legend-etf.txt") from exc
    return db.Historical(load_api_key())


def request_kwargs(
    symbol: str, start: datetime, end: datetime, *, for_timeseries: bool = False
) -> dict[str, Any]:
    kwargs = {
        "dataset": DATASET,
        "symbols": [symbol],
        "schema": SCHEMA,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "stype_in": "continuous",
    }
    if for_timeseries:
        kwargs["stype_out"] = "instrument_id"
    return kwargs


def quote_requests(client: Any, start: datetime, end: datetime) -> list[Quote]:
    quotes: list[Quote] = []
    for market in MARKETS:
        kwargs = request_kwargs(market.continuous_symbol, start, end)
        cost = float(client.metadata.get_cost(**kwargs))
        billable = int(client.metadata.get_billable_size(**kwargs))
        if not math.isfinite(cost) or cost < 0 or billable < 0:
            raise RuntimeError(f"invalid Databento quote for {market.root}")
        quotes.append(Quote(market.root, cost, billable))
    return quotes


def enforce_cost_gate(
    quotes: list[Quote],
    *,
    max_cost_usd: float,
    paid_confirmation: str | None,
    prior_cost_usd: float = 0.0,
) -> None:
    if not math.isfinite(max_cost_usd) or max_cost_usd < 0:
        raise ValueError("Databento run cap must be finite and non-negative")
    if any(
        not math.isfinite(quote.cost_usd)
        or quote.cost_usd < 0
        or quote.billable_bytes < 0
        for quote in quotes
    ):
        raise RuntimeError("Databento returned an invalid cost quote")
    if not math.isfinite(prior_cost_usd) or prior_cost_usd < 0:
        raise RuntimeError("Databento prior daily cost is invalid")
    total = sum(quote.cost_usd for quote in quotes)
    if not math.isfinite(total):
        raise RuntimeError("Databento aggregate quote is not finite")
    if prior_cost_usd + total > max_cost_usd + 1e-9:
        raise RuntimeError(
            f"Databento daily authorized ${prior_cost_usd:.4f} plus new quote "
            f"${total:.4f} exceeds the daily cap ${max_cost_usd:.4f}; no data "
            "request was sent"
        )
    if total > 0 and paid_confirmation != PAID_CONFIRMATION:
        raise RuntimeError(
            "Databento quoted a non-zero charge. Re-run only after explicit financial "
            f"approval with --paid-confirmation {PAID_CONFIRMATION}"
        )


def _store_to_frame(store: Any) -> pd.DataFrame:
    frame = store.to_df()
    if not isinstance(frame.index, pd.DatetimeIndex):
        if "ts_event" not in frame:
            raise RuntimeError("Databento response has no ts_event timestamp")
        frame = frame.set_index("ts_event")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
    columns = {str(column).lower(): column for column in frame.columns}
    required = ("open", "high", "low", "close", "instrument_id")
    missing = [name for name in required if name not in columns]
    if missing:
        raise RuntimeError(f"Databento response missing {missing}")
    output = frame.rename(columns={value: key for key, value in columns.items()})
    return output[[*required, *( ["volume"] if "volume" in output else [])]].copy()


def fetch_market_minutes(
    client: Any, symbol: str, start: datetime, end: datetime
) -> pd.DataFrame:
    store = client.timeseries.get_range(
        **request_kwargs(symbol, start, end, for_timeseries=True)
    )
    return _store_to_frame(store)


def _atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".parquet", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_parquet(temporary, index=True)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_integrity_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.integrity.json")


def _utc_iso(value: object) -> str:
    stamp = pd.Timestamp(value)
    if stamp.tz is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC").isoformat()


def _write_verified_cache(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError("refusing to persist an empty futures cache")
    _atomic_write_parquet(path, frame)
    atomic_write_json(
        _cache_integrity_path(path),
        {
            "protocol": CACHE_INTEGRITY_PROTOCOL,
            "parquet_name": path.name,
            "sha256": _file_sha256(path),
            "rows": len(frame),
            "columns": [str(column) for column in frame.columns],
            "first_timestamp": _utc_iso(frame.index[0]),
            "last_timestamp": _utc_iso(frame.index[-1]),
        },
    )


def _read_cache(path: Path) -> pd.DataFrame:
    integrity_path = _cache_integrity_path(path)
    if not path.exists() and not integrity_path.exists():
        return pd.DataFrame()
    if not path.exists() or not integrity_path.exists():
        raise RuntimeError(
            f"futures cache/integrity sidecar pair is incomplete for {path.name}"
        )
    manifest = read_json(integrity_path)
    if not isinstance(manifest, dict):
        raise RuntimeError(  # noqa: TRY004 - persisted operational state is corrupt
            f"invalid futures cache integrity sidecar for {path.name}"
        )
    if (
        manifest.get("protocol") != CACHE_INTEGRITY_PROTOCOL
        or manifest.get("parquet_name") != path.name
        or manifest.get("sha256") != _file_sha256(path)
    ):
        raise RuntimeError(f"futures cache integrity mismatch for {path.name}")
    frame = pd.read_parquet(path)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
    frame = frame.sort_index()
    if frame.empty or frame.index.has_duplicates:
        raise RuntimeError(f"invalid futures cache rows for {path.name}")
    expected_columns = [str(column) for column in frame.columns]
    try:
        sidecar_rows = int(manifest["rows"])
        sidecar_first = _utc_iso(manifest["first_timestamp"])
        sidecar_last = _utc_iso(manifest["last_timestamp"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"invalid futures cache integrity metadata for {path.name}"
        ) from exc
    if (
        sidecar_rows != len(frame)
        or manifest.get("columns") != expected_columns
        or sidecar_first != _utc_iso(frame.index[0])
        or sidecar_last != _utc_iso(frame.index[-1])
    ):
        raise RuntimeError(f"futures cache metadata mismatch for {path.name}")
    return frame


def _cache_request_start(
    frame: pd.DataFrame, *, start_fallback: datetime, end: datetime
) -> datetime:
    fallback = pd.Timestamp(start_fallback)
    boundary = pd.Timestamp(end)
    if fallback.tz is None or boundary.tz is None:
        raise ValueError("Databento cache request bounds must be timezone-aware")
    fallback = fallback.tz_convert("UTC")
    boundary = boundary.tz_convert("UTC")
    if fallback >= boundary:
        raise ValueError("Databento cache request start must precede end")
    if frame.empty:
        return fallback.to_pydatetime()
    next_missing = frame.index.max() + pd.Timedelta(minutes=1)
    overlap_start = boundary - CACHE_REFRESH_OVERLAP
    return max(fallback, min(next_missing, overlap_start)).to_pydatetime()


def _merge_cache_refresh(
    cached: pd.DataFrame,
    fresh: pd.DataFrame,
    *,
    request_start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Merge a fresh-wins overlap while rejecting a truncated refresh."""

    start_stamp = pd.Timestamp(request_start).tz_convert("UTC")
    end_stamp = pd.Timestamp(end).tz_convert("UTC")
    if fresh.empty:
        if cached.empty:
            return cached.copy()
        if not cached.loc[
            (cached.index >= start_stamp) & (cached.index < end_stamp)
        ].empty:
            raise RuntimeError("fresh Databento overlap omitted every cached row")
        return cached.copy()
    fresh = fresh.copy()
    fresh.index = pd.DatetimeIndex(pd.to_datetime(fresh.index, utc=True))
    if fresh.index.has_duplicates:
        raise RuntimeError("fresh Databento response contains duplicate timestamps")
    if cached.empty:
        return fresh.sort_index()
    cached_overlap = cached.loc[
        (cached.index >= start_stamp) & (cached.index < end_stamp)
    ]
    missing = cached_overlap.index.difference(fresh.index)
    if len(missing):
        raise RuntimeError(
            "fresh Databento overlap omitted cached timestamp(s); cache was not "
            f"modified (missing={len(missing)}, first={missing[0].isoformat()})"
        )
    return pd.concat([cached, fresh]).sort_index().loc[
        lambda value: ~value.index.duplicated(keep="last")
    ]


def _charge_request_id(root: str, start: datetime, end: datetime) -> str:
    return content_hash(
        {
            "dataset": DATASET,
            "schema": SCHEMA,
            "root": root,
            "start": start.isoformat(),
            "end": end.isoformat(),
        }
    )[:24]


def _append_charge_ledger(
    path: Path,
    *,
    root: str,
    start: datetime,
    end: datetime,
    quote: Quote,
    request_id: str,
    result: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    recorded_at = datetime.now(timezone.utc)
    line = canonical_json(
        {
            "recorded_at": recorded_at.isoformat(),
            "authorization_day_et": pd.Timestamp(recorded_at)
            .tz_convert(NY_TZ)
            .date()
            .isoformat(),
            "root": root,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "quoted_cost_usd": quote.cost_usd,
            "quoted_billable_bytes": quote.billable_bytes,
            "request_id": request_id,
            "result": result,
        }
    )
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _daily_charge_state(
    path: Path, *, now: datetime | None = None
) -> tuple[float, set[str], set[str]]:
    """Return today's cost plus ambiguous and completed request IDs."""

    if not path.exists():
        return 0.0, set(), set()
    wall_clock = pd.Timestamp(now or datetime.now(timezone.utc))
    if wall_clock.tz is None:
        raise ValueError("Databento charge clock must be timezone-aware")
    target_day = wall_clock.tz_convert(NY_TZ).date()
    authorized: dict[str, tuple[float, object]] = {}
    completed: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
            recorded_at = pd.Timestamp(record["recorded_at"])
            if recorded_at.tz is None:
                raise ValueError("naive recorded_at")
            record_day = pd.Timestamp(
                record.get(
                    "authorization_day_et",
                    recorded_at.tz_convert(NY_TZ).date().isoformat(),
                )
            ).date()
            request_id = str(record["request_id"])
            result = str(record["result"])
            cost = float(record["quoted_cost_usd"])
            if not request_id or not math.isfinite(cost) or cost < 0:
                raise ValueError("invalid charge record")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"invalid Databento charge ledger line {line_number}"
            ) from exc
        if result == "request_authorized":
            prior = authorized.get(request_id)
            if prior is not None and not math.isclose(
                prior[0], cost, rel_tol=0.0, abs_tol=1e-12
            ):
                raise RuntimeError(
                    f"Databento request {request_id} has conflicting authorizations"
                )
            authorized[request_id] = (cost, record_day)
        elif result == "download_persisted":
            completed.add(request_id)
        else:
            raise RuntimeError(
                f"invalid Databento charge ledger result on line {line_number}"
            )
    today_cost = sum(
        cost for cost, authorization_day in authorized.values()
        if authorization_day == target_day
    )
    orphaned_completions = completed.difference(authorized)
    if orphaned_completions:
        raise RuntimeError(
            "Databento charge ledger has completion(s) without authorization: "
            + ", ".join(sorted(orphaned_completions))
        )
    return today_cost, set(authorized).difference(completed), completed


def bootstrap_from_archive(
    archive_dir: Path, *, cutoff: datetime, days: int = 150
) -> dict[str, pd.DataFrame]:
    """Seed the small rolling cache from the already-purchased archive."""

    files = sorted(Path(archive_dir).glob("*.parquet"))
    if not files:
        return {}
    start = pd.Timestamp(cutoff).tz_convert("UTC") - pd.Timedelta(days=days)
    collected: list[pd.DataFrame] = []
    for path in reversed(files):
        frame = pd.read_parquet(path)
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
        collected.append(frame.loc[frame.index >= start])
        if frame.index.min() <= start:
            break
    if not collected:
        return {}
    combined = pd.concat(reversed(collected)).sort_index()
    result: dict[str, pd.DataFrame] = {}
    for market in MARKETS:
        if "symbol" not in combined:
            break
        rows = combined.loc[combined["symbol"].eq(market.continuous_symbol)].copy()
        if not rows.empty:
            result[market.root] = rows[
                ["open", "high", "low", "close", "volume", "instrument_id"]
            ]
    return result


def update_rolling_cache(
    *,
    client: Any,
    start_fallback: datetime,
    end: datetime,
    cache_dir: Path,
    archive_dir: Path | None,
    max_cost_usd: float,
    paid_confirmation: str | None,
) -> tuple[dict[str, pd.DataFrame], list[Quote], datetime]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = {
        market.root: _read_cache(cache_dir / f"{market.root}.parquet")
        for market in MARKETS
    }
    if archive_dir is not None and any(frame.empty for frame in cached.values()):
        seeded = bootstrap_from_archive(archive_dir, cutoff=end)
        for root, frame in seeded.items():
            if cached[root].empty:
                cached[root] = frame
                # Persist the already-paid archive seed before any new
                # request so an interruption cannot make it disappear.
                _write_verified_cache(cache_dir / f"{root}.parquet", frame)

    starts: dict[str, datetime] = {}
    for market in MARKETS:
        frame = cached[market.root]
        if frame.empty:
            request_start = start_fallback
        else:
            # Refresh a source-overlap as well as any append gap. The immutable
            # sidecar detects older local damage; this overlap heals recent
            # additions/corrections without assuming every minute traded.
            request_start = _cache_request_start(
                frame,
                start_fallback=start_fallback,
                end=end,
            )
        starts[market.root] = request_start
    ledger_path = cache_dir / "databento_charge_ledger.jsonl"
    prior_cost, ambiguous_requests, completed_requests = _daily_charge_state(
        ledger_path
    )
    if ambiguous_requests:
        raise RuntimeError(
            "prior paid Databento request has no persisted completion; inspect "
            "the cache/charge ledger before authorizing another request: "
            + ", ".join(sorted(ambiguous_requests))
        )

    quotes: list[Quote] = []
    request_ids: dict[str, str | None] = {}
    for market in MARKETS:
        request_start = starts[market.root]
        if request_start < end:
            request_id = _charge_request_id(market.root, request_start, end)
            request_ids[market.root] = request_id
            if request_id in completed_requests:
                if cached[market.root].empty:
                    raise RuntimeError(
                        f"completed Databento request has no cache for {market.root}"
                    )
                quotes.append(Quote(market.root, 0.0, 0))
                continue
            kwargs = request_kwargs(market.continuous_symbol, request_start, end)
            cost = float(client.metadata.get_cost(**kwargs))
            billable = int(client.metadata.get_billable_size(**kwargs))
            quotes.append(Quote(market.root, cost, billable))
        else:
            request_ids[market.root] = None
            quotes.append(Quote(market.root, 0.0, 0))
    enforce_cost_gate(
        quotes,
        max_cost_usd=max_cost_usd,
        paid_confirmation=paid_confirmation,
        prior_cost_usd=prior_cost,
    )

    quote_by_root = {quote.root: quote for quote in quotes}
    cutoff = pd.Timestamp(end).tz_convert("UTC") - pd.Timedelta(days=150)
    combined: dict[str, pd.DataFrame] = {}
    for market in MARKETS:
        request_start = starts[market.root]
        fresh = pd.DataFrame()
        request_id = request_ids[market.root]
        already_completed = request_id is not None and request_id in completed_requests
        if request_id is not None and not already_completed:
            _append_charge_ledger(
                ledger_path,
                root=market.root,
                start=request_start,
                end=end,
                quote=quote_by_root[market.root],
                request_id=request_id,
                result="request_authorized",
            )
            fresh = fetch_market_minutes(
                client, market.continuous_symbol, request_start, end
            )
        if cached[market.root].empty and fresh.empty:
            raise RuntimeError(f"no futures data available for {market.root}")
        if already_completed:
            frame = cached[market.root]
        else:
            frame = _merge_cache_refresh(
                cached[market.root],
                fresh,
                request_start=request_start,
                end=end,
            )
        frame = frame.loc[frame.index >= cutoff]
        # Persist each billed root immediately. If a later root fails, the
        # retry resumes this root after its last stored minute instead of
        # purchasing the same completed range again.
        _write_verified_cache(cache_dir / f"{market.root}.parquet", frame)
        if request_id is not None and not already_completed:
            _append_charge_ledger(
                ledger_path,
                root=market.root,
                start=request_start,
                end=end,
                quote=quote_by_root[market.root],
                request_id=request_id,
                result="download_persisted",
            )
        combined[market.root] = frame
    return combined, quotes, min(starts.values())


def prepare_signal_plan(
    *,
    client: Any,
    entry_date: str,
    as_of: datetime,
    output_path: Path,
    lookback_days: int = 150,
    max_cost_usd: float = 0.0,
    paid_confirmation: str | None = None,
    cache_dir: Path | None = None,
    archive_dir: Path | None = None,
) -> dict[str, Any]:
    """Quote, fetch, evaluate, and atomically persist today's signal plan."""

    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    if not isinstance(lookback_days, int) or lookback_days <= 0:
        raise ValueError("lookback_days must be a positive whole number")
    if not math.isfinite(max_cost_usd) or max_cost_usd < 0:
        raise ValueError("max_cost_usd must be finite and non-negative")
    require_full_entry_session(entry_date)
    end = available_end(client, as_of.astimezone(timezone.utc))
    start = end - timedelta(days=lookback_days)
    resolved_cache = cache_dir or output_path.parent / "futures_cache"
    with exclusive_file_lock(resolved_cache / "update.lock"):
        frames, quotes, request_start = update_rolling_cache(
            client=client,
            start_fallback=start,
            end=end,
            cache_dir=resolved_cache,
            archive_dir=archive_dir,
            max_cost_usd=max_cost_usd,
            paid_confirmation=paid_confirmation,
        )

    markets: list[dict[str, Any]] = []
    setup_dates: list[str] = []
    for market in MARKETS:
        observed_setup = latest_observed_futures_rth_session(
            frames[market.root], entry_date=entry_date, as_of=pd.Timestamp(as_of)
        )
        setup_date = None if observed_setup is None else observed_setup.isoformat()
        if setup_date is None:
            result = {
                "qualifies": False,
                "reason": "missing_setup_session",
                "setup_date": None,
                "entry_date": entry_date,
            }
        elif pd.Timestamp(setup_date) < pd.Timestamp(market.trusted_from):
            result = {
                "qualifies": False,
                "reason": "before_trusted_window",
                "setup_date": setup_date,
                "entry_date": entry_date,
            }
        else:
            result = evaluate_futures_setup(
                frames[market.root],
                setup_date=setup_date,
                entry_date=entry_date,
                as_of=pd.Timestamp(as_of),
            ).to_dict()
        if setup_date is not None:
            setup_dates.append(setup_date)
        markets.append(
            {
                "root": market.root,
                "futures_symbol": market.continuous_symbol,
                "etf": market.etf,
                **result,
            }
        )

    payload = finalize_plan(
        {
            "strategy_version": STRATEGY_VERSION,
            "entry_date": entry_date,
            "setup_date": (
                setup_dates[0]
                if setup_dates and len(set(setup_dates)) == 1
                else None
            ),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "data_as_of": end.isoformat(),
            "dataset": DATASET,
            "schema": SCHEMA,
            "request_start": request_start.isoformat(),
            "cache_dir": str(resolved_cache),
            "quoted_cost_usd": sum(quote.cost_usd for quote in quotes),
            "quoted_billable_bytes": sum(quote.billable_bytes for quote in quotes),
            "markets": markets,
        }
    )
    atomic_write_json(output_path, payload)
    return payload
