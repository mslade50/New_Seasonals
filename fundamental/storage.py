"""Immutable raw archives and append-only point-in-time snapshot helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import CURRENT_ROOT, RAW_ROOT, RUN_ROOT, SNAPSHOT_ROOT


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(value: datetime | None = None) -> str:
    return (value or utc_now()).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def payload_digest(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def archive_json(payload: Any, provider: str, dataset: str, identity: str) -> tuple[Path, str]:
    """Write content-addressed JSON once and return ``(path, digest)``."""
    digest = payload_digest(payload)
    safe_identity = "".join(c for c in identity.upper() if c.isalnum() or c in "-_")
    path = RAW_ROOT / provider / dataset / safe_identity / f"{digest}.json"
    if not path.exists():
        _atomic_write_bytes(path, canonical_json_bytes(payload))
    return path, digest


def snapshot_part_path(kind: str, as_of: str | date, ticker: str, dataset: str) -> Path:
    day = str(as_of)[:10]
    safe_ticker = "".join(c for c in ticker.upper() if c.isalnum() or c in "-_")
    safe_dataset = "".join(c for c in dataset.lower() if c.isalnum() or c in "-_")
    return SNAPSHOT_ROOT / kind / f"as_of={day}" / f"ticker={safe_ticker}" / f"{safe_dataset}.parquet"


def write_immutable_parquet(frame: pd.DataFrame, path: Path) -> Path:
    """Write a snapshot part once; reject attempts to alter an existing part."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        prior = pd.read_parquet(path)
        # Retrieval time is operational metadata, not snapshot content.  A
        # same-day retry of an identical content-addressed payload must be
        # idempotent even though the network fetch occurred a few seconds later.
        compare_columns = sorted(
            (set(prior.columns) | set(frame.columns)) - {"fetched_at", "raw_payload_digest"}
        )
        try:
            pd.testing.assert_frame_equal(
                prior.reindex(columns=compare_columns).reset_index(drop=True),
                frame.reindex(columns=compare_columns).reset_index(drop=True),
                check_dtype=False,
                check_like=True,
            )
        except AssertionError as exc:
            raise FileExistsError(f"immutable snapshot differs: {path}") from exc
        return path

    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return path


def write_current_parquet(frame: pd.DataFrame, name: str) -> Path:
    """Atomically replace a derived current-state view (raw history is untouched)."""
    CURRENT_ROOT.mkdir(parents=True, exist_ok=True)
    path = CURRENT_ROOT / name
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return path


def write_run_manifest(payload: dict, run_id: str | None = None) -> Path:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    rid = run_id or utc_now().strftime("%Y%m%dT%H%M%SZ")
    path = RUN_ROOT / f"{rid}.json"
    if path.exists():
        raise FileExistsError(f"run manifest already exists: {path}")
    _atomic_write_bytes(path, json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8"))
    return path


def load_snapshot_parts(kind: str, as_of: str | date, tickers: list[str] | None = None) -> pd.DataFrame:
    root = SNAPSHOT_ROOT / kind / f"as_of={str(as_of)[:10]}"
    if not root.exists():
        return pd.DataFrame()
    allowed = {t.upper() for t in tickers} if tickers else None
    parts = []
    for path in root.glob("ticker=*/*.parquet"):
        ticker = path.parent.name.split("=", 1)[-1].upper()
        if allowed is not None and ticker not in allowed:
            continue
        parts.append(pd.read_parquet(path))
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def available_snapshot_dates(kind: str) -> list[str]:
    root = SNAPSHOT_ROOT / kind
    if not root.exists():
        return []
    return sorted(p.name.split("=", 1)[-1] for p in root.glob("as_of=*") if p.is_dir())


def latest_snapshot_date(kind: str, on_or_before: str | date | None = None) -> str | None:
    dates = available_snapshot_dates(kind)
    if on_or_before is not None:
        cutoff = str(on_or_before)[:10]
        dates = [d for d in dates if d <= cutoff]
    return dates[-1] if dates else None


def snapshot_coverage(kind: str, on_or_before: str | date | None = None) -> pd.DataFrame:
    """Return the latest stored date for every ``(ticker, dataset)`` pair.

    This reads path metadata only, so broad-universe queue selection does not
    need to load years of parquet history.
    """
    rows: list[dict] = []
    cutoff = str(on_or_before)[:10] if on_or_before is not None else None
    root = SNAPSHOT_ROOT / kind
    if not root.exists():
        return pd.DataFrame(columns=["ticker", "dataset", "snapshot_as_of"])
    for dated_root in root.glob("as_of=*"):
        snapshot_day = dated_root.name.split("=", 1)[-1]
        if cutoff is not None and snapshot_day > cutoff:
            continue
        for path in dated_root.glob("ticker=*/*.parquet"):
            rows.append({
                "ticker": path.parent.name.split("=", 1)[-1].upper(),
                "dataset": path.stem.lower(),
                "snapshot_as_of": snapshot_day,
            })
    if not rows:
        return pd.DataFrame(columns=["ticker", "dataset", "snapshot_as_of"])
    return (
        pd.DataFrame(rows)
        .sort_values("snapshot_as_of")
        .drop_duplicates(["ticker", "dataset"], keep="last")
        .reset_index(drop=True)
    )


def load_latest_snapshot_parts(
    kind: str,
    on_or_before: str | date | None = None,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Load each ticker/dataset's most recent immutable part by a cutoff.

    Incremental enrichment batches can span many run dates.  A current research
    view therefore needs the latest part per dataset, rather than only the most
    recent global partition date.
    """
    coverage = snapshot_coverage(kind, on_or_before)
    if coverage.empty:
        return pd.DataFrame()
    if tickers:
        allowed = {str(t).upper() for t in tickers}
        coverage = coverage[coverage["ticker"].isin(allowed)]
    parts: list[pd.DataFrame] = []
    for row in coverage.itertuples(index=False):
        path = snapshot_part_path(kind, row.snapshot_as_of, row.ticker, row.dataset)
        if path.exists():
            parts.append(pd.read_parquet(path))
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def point_in_time_latest(
    frame: pd.DataFrame,
    decision_at: str | datetime,
    *,
    entity_columns: tuple[str, ...] = ("ticker", "endpoint", "date"),
    accepted_column: str = "accepted_at",
) -> pd.DataFrame:
    """Return the last observation actually available by ``decision_at``.

    Rows without an acceptance timestamp are excluded.  This intentionally
    fails closed for historical research; provider period dates are not a
    substitute for when the market could observe a filing.
    """
    if frame.empty:
        return frame.copy()
    if accepted_column not in frame.columns:
        return frame.iloc[0:0].copy()
    out = frame.copy()
    out[accepted_column] = pd.to_datetime(out[accepted_column], utc=True, errors="coerce")
    cutoff = pd.Timestamp(decision_at)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    out = out[out[accepted_column].notna() & (out[accepted_column] <= cutoff)]
    if out.empty:
        return out
    keys = [c for c in entity_columns if c in out.columns]
    if not keys:
        raise ValueError("no entity columns present for point-in-time selection")
    return (
        out.sort_values(accepted_column)
        .drop_duplicates(keys, keep="last")
        .reset_index(drop=True)
    )
