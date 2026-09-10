"""Copy the execution-broker's /fills ring into a durable R2-canonical parquet.

The broker accumulates account executions in a bounded rolling history. The
canonical parquet is the durable copy; local parquet is only a working artifact.
Publishing requires a verified canonical read and an ETag compare-and-swap.
Both old and new bytes are retained as immutable content-addressed generations.
Read errors never authorize replacing the canonical history with a local/empty
fallback; first initialization requires an explicit flag and confirmed absence.

Effective fills are keyed by account plus IB execution family. A higher revision
supersedes the original quantity rather than adding to it. Repeated identical
executions retain stored commission/PnL enrichment when the newer row omits it.

Coverage is explicit: fresh successful Primary execution-request receipts,
no truncation/merge errors, calendar-day retention checks, and scheduled
--assert-no-gap. PA-only source failures do not veto Primary harvests. Missing
historical coverage is reported; an empty ring does not prove the account flat.
A snapshot/file digest binds the completeness status to the canonical generation.

`order_ref` carries the book's `SYMBOL|ACTION|Strategy|Date` contract, so the
strategy, side and signal date are parsed into their own columns here -- that
is what makes the store joinable to `data/backtest_trades_full.parquet` for
live-vs-ledger measurement.

CLI:
    python scripts/harvest_fills.py                     # harvest + upload
    python scripts/harvest_fills.py --dry-run           # fetch + merge, no writes
    python scripts/harvest_fills.py --assert-no-gap     # non-zero exit on a hole
    python scripts/harvest_fills.py --summary-json data/live_fills_status.json
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from trading_calendar import TRADING_DAY  # noqa: E402

DEFAULT_BROKER_URL = "https://execution-broker.mckinleyslade.workers.dev"
LOCAL_PATH = _ROOT / "data" / "live_fills.parquet"
R2_KEY = "live_fills.parquet"
STATUS_R2_KEY = "live_fills_status.json"
ET = "America/New_York"
# The broker DO's documented retention. Used only when the payload does not
# carry `retention_days`; the empty-ring gap test needs SOME window.
DEFAULT_RETENTION_DAYS = 14

# Frozen schema. New columns from the broker are DROPPED rather than silently
# widening the store (the fragility-parquet convention); add them here first.
COLUMNS: tuple[str, ...] = (
    "exec_id",
    "time_utc",
    "session_date",
    "account",
    "account_key",
    "account_label",
    "symbol",
    "sec_type",
    "currency",
    "exchange",
    "side",
    "qty",
    "price",
    "avg_price",
    "cum_qty",
    "order_id",
    "perm_id",
    "client_id",
    "order_ref",
    "ref_symbol",
    "ref_action",
    "strategy",
    "ref_date",
    "commission",
    "realized_pnl",
    "expiry",
    "expiry_full",
    "con_id",
    "ingested_at",
    "harvested_at_utc",
)

# Columns the broker enriches AFTER the first sighting. A later fetch that has
# them null must never overwrite a stored value.
ENRICHMENT_COLUMNS: tuple[str, ...] = ("commission", "realized_pnl")

_STR_COLS = (
    "exec_id", "account", "account_key", "account_label", "symbol", "sec_type",
    "currency", "exchange", "side", "order_ref", "ref_symbol", "ref_action",
    "strategy", "expiry", "expiry_full",
)
_FLOAT_COLS = ("qty", "price", "avg_price", "cum_qty", "commission", "realized_pnl")
_INT_COLS = ("order_id", "perm_id", "client_id", "con_id", "ingested_at")


def parse_order_ref(ref: Any) -> tuple[str, str, str, str]:
    """Split the book's `SYMBOL|ACTION|Strategy|Date` orderRef contract.

    Returns ``(symbol, action, strategy, date)`` with empty strings for any
    field a ref does not carry. Discretionary and hand-placed orders have no
    ref at all, and untagged legs predate the 2026-07 tagging change; both
    yield four empties rather than a guess.
    """
    if not isinstance(ref, str) or not ref.strip():
        return ("", "", "", "")
    parts = [p.strip() for p in ref.split("|")]
    parts += [""] * (4 - len(parts))
    return (parts[0], parts[1], parts[2], parts[3])


def normalize(rows: list[dict]) -> pd.DataFrame:
    """Broker JSON rows -> the frozen schema, typed and ET-dated."""
    if not rows:
        return empty_frame()
    df = pd.DataFrame(rows)
    for col in ("exec_id", "time"):
        if col not in df.columns:
            raise ValueError(f"broker fills missing required column {col!r}")

    out = pd.DataFrame(index=df.index)
    out["exec_id"] = df["exec_id"].astype("string")
    ts = pd.to_datetime(df["time"], utc=True, errors="coerce")
    out["time_utc"] = ts
    # The session a fill belongs to is its EASTERN date: 19:59 UTC is 15:59 ET
    # the same day, but 00:30 UTC belongs to the previous ET session.
    out["session_date"] = ts.dt.tz_convert(ET).dt.date.astype("string")

    for col in ("account", "account_key", "account_label", "symbol", "sec_type",
                "currency", "exchange", "side", "order_ref", "expiry", "expiry_full"):
        out[col] = df[col].astype("string") if col in df.columns else pd.Series(pd.NA, index=df.index, dtype="string")

    refs = out["order_ref"].map(parse_order_ref)
    out["ref_symbol"] = refs.map(lambda r: r[0]).astype("string")
    out["ref_action"] = refs.map(lambda r: r[1]).astype("string")
    out["strategy"] = refs.map(lambda r: r[2]).astype("string")
    out["ref_date"] = refs.map(lambda r: r[3]).astype("string")

    for col in _FLOAT_COLS:
        out[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    for col in _INT_COLS:
        # A column the broker omits entirely must become an all-NA Series, not
        # a scalar: pd.to_numeric(pd.NA) is a scalar and cannot .astype("Int64").
        raw = df[col] if col in df.columns else pd.Series(pd.NA, index=df.index, dtype="object")
        out[col] = pd.to_numeric(raw, errors="coerce").astype("Int64")

    out["harvested_at_utc"] = pd.Timestamp.now(tz="UTC")
    out = out.dropna(subset=["exec_id"])
    return out[list(COLUMNS)].reset_index(drop=True)


def empty_frame() -> pd.DataFrame:
    """An empty frame carrying the frozen schema and its dtypes."""
    data: dict[str, pd.Series] = {}
    for col in COLUMNS:
        if col in _STR_COLS or col in ("session_date", "ref_date"):
            data[col] = pd.Series(dtype="string")
        elif col in _FLOAT_COLS:
            data[col] = pd.Series(dtype="float64")
        elif col in _INT_COLS:
            data[col] = pd.Series(dtype="Int64")
        else:
            data[col] = pd.Series(dtype="datetime64[ns, UTC]")
    return pd.DataFrame(data)


def merge_fills(existing: pd.DataFrame, incoming: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Upsert by account/execution family, never losing exact-id enrichment.

    The broker is upstream truth, so a re-fetched row wins field for field --
    except on ENRICHMENT_COLUMNS, where a null incoming value keeps whatever we
    already stored. That is the commission-lag case: the same execution comes
    back later with the commission attached, and could in principle come back
    again without it.

    Raises if any stored family disappears. A correction may supersede its
    previous revision; immutable canonical generations retain the old bytes.
    """
    existing = empty_frame() if existing is None or existing.empty else existing.copy()
    incoming = empty_frame() if incoming is None or incoming.empty else incoming.copy()
    for frame in (existing, incoming):
        frame.attrs = {}
        for col in COLUMNS:
            if col not in frame.columns:
                frame[col] = pd.NA

    def identity(frame):
        return set(zip(frame["account"].fillna("").astype(str), frame["exec_id"].fillna("").astype(str)))
    before = identity(existing)
    arriving = identity(incoming)
    new_ids = arriving - before
    seen_again = arriving & before

    # Carry stored enrichment onto re-fetched rows that arrive without it.
    if seen_again and not existing.empty:
        stored = existing.drop_duplicates(["account", "exec_id"], keep="last").set_index(["account", "exec_id"])
        idx = pd.Series(list(zip(incoming["account"], incoming["exec_id"])), index=incoming.index)
        for col in ENRICHMENT_COLUMNS:
            prior = idx.map(stored[col]) if col in stored.columns else pd.Series(pd.NA, index=incoming.index)
            incoming[col] = pd.to_numeric(incoming[col], errors="coerce").fillna(
                pd.to_numeric(prior, errors="coerce")
            )

    # Concat only the non-empty frames: an all-NA frame would coerce dtypes.
    parts = [f[list(COLUMNS)] for f in (existing, incoming) if not f.empty]
    combined = pd.concat(parts, ignore_index=True) if parts else empty_frame()
    # Incoming rows sit last, so keeping the last duplicate makes the broker win.
    combined = combined.drop_duplicates(subset=["account", "exec_id"], keep="last")
    # Broker corrections replace one execution family; they are not extra fills.
    # The immutable canonical generations preserve superseded raw observations.
    combined["_family"] = combined["exec_id"].map(execution_family)
    combined["_revision"] = combined["exec_id"].map(execution_revision)
    combined = combined.sort_values("_revision", kind="stable").drop_duplicates(
        ["account", "_family"], keep="last")
    combined = combined.drop(columns=["_family", "_revision"])
    combined = combined.sort_values(["time_utc", "exec_id"], kind="stable").reset_index(drop=True)

    # The invariant is set containment, not row count: every execution we held
    # must survive the merge. A count check would miss a row dropped while new
    # ones arrive, and would false-alarm on the harmless dedup of a store that
    # somehow holds the same exec_id twice.
    kept = identity(combined)
    kept_families = {(account, execution_family(exec_id)) for account, exec_id in kept}
    lost = {(account, exec_id) for account, exec_id in before
            if (account, execution_family(exec_id)) not in kept_families}
    if lost:
        raise ValueError(
            f"merge would drop {len(lost)} stored execution(s) "
            f"(e.g. {sorted(lost)[:3]}); refusing to write"
        )
    if len(combined) != len(kept):
        raise ValueError(
            f"merged store is not keyed by exec_id ({len(combined)} rows, {len(kept)} ids)"
        )
    stats = {
        "rows_before": int(len(existing)),
        "rows_after": int(len(combined)),
        "rows_new": int(len(new_ids)),
        "rows_updated": int(len(seen_again)),
    }
    return combined, stats


def execution_family(exec_id: str) -> str:
    value = str(exec_id)
    match = re.fullmatch(r"(.+)\.(\d+)", value)
    return match.group(1) if match else value


def execution_revision(exec_id: str) -> int:
    match = re.fullmatch(r"(.+)\.(\d+)", str(exec_id))
    return int(match.group(2)) if match else 0


def _today_eastern() -> pd.Timestamp:
    return pd.Timestamp.now(tz=ET).tz_localize(None).normalize()


def detect_gap(existing: pd.DataFrame, incoming: pd.DataFrame,
               retention_days: int | None = None,
               today: pd.Timestamp | str | None = None) -> dict:
    """Compare the ring's oldest session to our newest: a hole means lost rows.

    The ring keeps ~14 calendar days. If nothing harvested for longer than
    that, executions expired unseen and only an IBKR Flex pull can recover
    them, so this has to be loud rather than a green no-op.

    An EMPTY ring is the ambiguous case (2026-09-04): "nothing traded for a
    fortnight" and "everything aged out unseen" both come back as zero rows.
    They are told apart by the store: if our newest session is still inside
    the ring's window (today minus ``retention_days - 1`` trading sessions),
    any later fill would still be in the ring, so an empty ring is a genuine
    no-trade stretch. If our newest session is OLDER than that window, the
    ring has already dropped whatever happened after it, and that is a gap.
    Session arithmetic uses the NYSE trading calendar, not calendar days.
    """
    info = {"gap": False, "ring_oldest": None, "stored_newest": None,
            "missing_business_days": 0, "reason": None,
            "retention_days": int(retention_days) if retention_days else DEFAULT_RETENTION_DAYS}
    store_empty = existing is None or existing.empty
    stored_newest = None if store_empty else str(existing["session_date"].dropna().max())

    if incoming is None or incoming.empty:
        if store_empty or not stored_newest or stored_newest == "nan":
            info["reason"] = "first run: nothing stored and nothing in the ring"
            return info
        info["stored_newest"] = stored_newest
        today_ts = pd.Timestamp(today) if today is not None else _today_eastern()
        today_ts = today_ts.normalize()
        window = max(int(info["retention_days"]) - 1, 0)
        threshold = today_ts - pd.Timedelta(days=window)
        newest_ts = pd.Timestamp(stored_newest)
        if newest_ts < threshold:
            # Sessions strictly after our newest stored session, through today.
            span = pd.date_range(newest_ts + pd.Timedelta(days=1), today_ts, freq=TRADING_DAY)
            info["missing_business_days"] = int(len(span))
            info["gap"] = True
            info["reason"] = (
                f"ring is EMPTY and our newest stored session {stored_newest} is older "
                f"than the ring window (today {today_ts.date()} minus {window} calendar days "
                f"= {threshold.date()}); fills after {stored_newest} aged out unseen"
            )
        else:
            info["reason"] = (
                f"ring is empty but our newest stored session {stored_newest} is inside "
                f"the ring window (from {threshold.date()}): a genuine no-trade stretch"
            )
        return info

    ring_oldest = str(incoming["session_date"].dropna().min())
    info["ring_oldest"] = ring_oldest
    if store_empty:
        return info
    info["stored_newest"] = stored_newest
    if not stored_newest or not ring_oldest:
        return info
    # Business days strictly between the newest stored session and the oldest
    # session still in the ring. Zero or one means the windows touch.
    span = pd.date_range(
        pd.Timestamp(stored_newest) + pd.Timedelta(days=1),
        pd.Timestamp(ring_oldest) - pd.Timedelta(days=1), freq=TRADING_DAY,
    )
    info["missing_business_days"] = int(len(span))
    info["gap"] = len(span) > 0
    if info["gap"]:
        info["reason"] = (
            f"ring starts {ring_oldest} but our newest session is {stored_newest}"
        )
    return info


def fetch_fills(base_url: str, token: str, timeout: int = 45) -> dict:
    r = requests.get(
        f"{base_url.rstrip('/')}/fills",
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def load_existing(pull_r2: bool = True, *, allow_initialize: bool = False) -> pd.DataFrame:
    """Read canonical bytes and their exact CAS version, never downgrade errors."""
    if pull_r2:
        from cache_io import _client, _r2_creds
        client, creds = _client(), _r2_creds()
        if client is None or creds is None:
            raise RuntimeError("canonical fill storage is not configured")
        try:
            response = client.get_object(Bucket=creds["R2_BUCKET"], Key=R2_KEY)
            body = response["Body"].read()
            etag = response.get("ETag")
            if not etag:
                raise RuntimeError("canonical fill object has no version identity")
            frame = pd.read_parquet(io.BytesIO(body))
        except Exception as exc:
            code = str((getattr(exc, "response", {}) or {}).get("Error", {}).get("Code", ""))
            if code not in {"NoSuchKey", "404", "NotFound"} or not allow_initialize:
                raise RuntimeError("canonical fill read failed; prior history was preserved") from exc
            frame, body, etag = empty_frame(), b"", None
        frame.attrs.update(canonical_loaded=True, canonical_etag=etag, canonical_bytes=body)
        # Old or unavailable status may not attest history, but it must not
        # prevent preserving the existing execution rows in the next merge.
        try:
            status = json.loads(client.get_object(Bucket=creds['R2_BUCKET'], Key=STATUS_R2_KEY)['Body'].read())
            if status.get('canonical_sha256') == hashlib.sha256(body).hexdigest():
                frame.attrs['canonical_status'] = status
        except Exception:
            pass
        return frame
    if not LOCAL_PATH.exists():
        return empty_frame()
    try:
        return pd.read_parquet(LOCAL_PATH)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"FAIL: {LOCAL_PATH} exists but is unreadable ({e}); refusing to overwrite it")


def validate_source_completeness(payload: dict, *, now=None, required_account="primary") -> None:
    """Broker source coverage must be explicit; a quiet ring is not evidence."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    coverage = payload.get("completeness") or {}
    account = (coverage.get("accounts") or {}).get(required_account) or {}
    if coverage.get("truncated") or coverage.get("incomplete_days") or coverage.get("merge_error"):
        raise RuntimeError("broker fill history is incomplete or lacks coverage evidence")
    if account.get("complete") is not True or account.get("error"):
        raise RuntimeError("Primary fill source did not complete")
    stamp = pd.Timestamp(account.get("source_at") or coverage.get("complete_through"))
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise RuntimeError("broker fill coverage timestamp is invalid")
    received = pd.Timestamp(account.get("received_at"))
    if pd.isna(received) or received.tzinfo is None:
        raise RuntimeError("Primary fill receipt timestamp is invalid")
    if any(not 0 <= (now - value).total_seconds() <= 300 for value in (stamp, received)):
        raise RuntimeError("broker fill source is stale or future-dated")


def publish_canonical(frame: pd.DataFrame, original: pd.DataFrame) -> None:
    """Preserve immutable generations, then CAS the canonical pointer/object."""
    from cache_io import _client, _r2_creds
    if original.attrs.get("canonical_loaded") is not True:
        raise RuntimeError("cannot publish without a verified canonical read")
    client, creds = _client(), _r2_creds()
    if client is None or creds is None:
        raise RuntimeError("canonical fill storage is not configured")
    stream = io.BytesIO()
    serializable = frame.copy()
    serializable.attrs = {}
    serializable.to_parquet(stream, index=False)
    body = stream.getvalue()
    for content in (original.attrs.get("canonical_bytes", b""), body):
        if not content:
            continue
        digest = hashlib.sha256(content).hexdigest()
        key = f"live_fills/generations/{digest}.parquet"
        try:
            client.put_object(Bucket=creds["R2_BUCKET"], Key=key, Body=content, IfNoneMatch="*")
        except Exception as exc:
            code = str((getattr(exc, "response", {}) or {}).get("Error", {}).get("Code", ""))
            if code not in {"PreconditionFailed", "412", "ConditionalRequestConflict"}:
                raise
            existing = client.get_object(Bucket=creds["R2_BUCKET"], Key=key)["Body"].read()
            if existing != content:
                raise RuntimeError("immutable fill generation differs from its digest") from exc
    condition = {"IfMatch": original.attrs["canonical_etag"]} if original.attrs.get("canonical_etag") else {"IfNoneMatch": "*"}
    client.put_object(Bucket=creds["R2_BUCKET"], Key=R2_KEY, Body=body, **condition)


def extend_canonical_coverage(current: dict, original: pd.DataFrame) -> dict:
    """Retain proven coverage after the live ring rolls past the seed date.

    A hash-matched prior status and overlapping intervals are both required.
    A gap starts a new interval; saved rows alone never prove continuity.
    """
    coverage = json.loads(json.dumps(current))
    previous = original.attrs.get('canonical_status') or {}
    prior = previous.get('completeness') or {}
    if (previous.get('complete') is not True or previous.get('gap', {}).get('gap')
            or any(prior.get(k) for k in ('truncated', 'merge_error', 'incomplete_days'))):
        return coverage
    for key, value in coverage.get('accounts', {}).items():
        old = prior.get('accounts', {}).get(key) or {}
        if (value.get('complete') is not True or old.get('complete') is not True
                or not value.get('broker_account') or value['broker_account'] != old.get('broker_account')):
            continue
        try:
            start, end, new_start, new_end = [pd.Timestamp(v) for v in (
                old['continuous_from'], old['complete_through'],
                value['continuous_from'], value['complete_through'])]
            if any(pd.isna(t) or t.tzinfo is None for t in (start, end, new_start, new_end)):
                continue
            if start <= new_start <= end <= new_end:
                value['continuous_from'] = old['continuous_from']
            if ((value.get('olv_coverage') or {}).get('scope')=='OLV_US_STK_NON_OVERNIGHT'
                    and (old.get('olv_coverage') or {}).get('scope')=='OLV_US_STK_NON_OVERNIGHT'):
                old_start=pd.Timestamp(old.get('olv_continuous_from'))
                current_start=pd.Timestamp(value.get('olv_continuous_from'))
                if (not pd.isna(old_start) and not pd.isna(current_start)
                        and old_start.tzinfo is not None and current_start.tzinfo is not None
                        and old_start<=current_start<=end<=new_end):
                    value['olv_continuous_from']=old['olv_continuous_from']
        except (KeyError, ValueError, TypeError):
            continue
    return coverage


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"rows": 0, "accounts": {}, "first_session": None, "last_session": None,
                "sessions": 0, "tagged_pct": 0.0}
    sessions = df["session_date"].dropna().astype(str)
    tagged = int(df["strategy"].fillna("").astype(str).str.len().gt(0).sum())
    return {
        "rows": int(len(df)),
        "accounts": {str(k): int(v) for k, v in df["account_key"].fillna("?").value_counts().items()},
        "first_session": str(sessions.min()) if len(sessions) else None,
        "last_session": str(sessions.max()) if len(sessions) else None,
        "sessions": int(sessions.nunique()),
        "tagged_pct": round(100.0 * tagged / len(df), 1),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="fetch and merge, write nothing")
    ap.add_argument("--no-upload", action="store_true", help="write locally, skip the R2 push")
    ap.add_argument("--assert-no-gap", action="store_true",
                    help="exit non-zero when the ring starts after our newest stored session")
    ap.add_argument("--summary-json", help="write a small status JSON here")
    ap.add_argument("--initialize-empty-canonical", action="store_true",
                    help="allow initialization only after a confirmed missing canonical object")
    args = ap.parse_args(argv)

    try:
        from dotenv import load_dotenv
        load_dotenv(_ROOT / ".env", override=False)
    except ImportError:
        pass

    base_url = os.environ.get("EXEC_BROKER_URL", DEFAULT_BROKER_URL)
    token = os.environ.get("STATUS_TOKEN", "")
    if not token:
        print("FAIL: STATUS_TOKEN not set - cannot read the broker's fills ring.")
        print("      It lives in the trading credentials env the automation supervisor loads.")
        return 2

    print(f"Harvesting fills from {base_url}")
    try:
        payload = fetch_fills(base_url, token)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: broker /fills unreachable ({e})")
        return 2

    rows = payload.get("fills") or []
    retention = payload.get("retention_days")
    print(f"  ring: {len(rows)} rows, retention_days={retention}")

    try:
        validate_source_completeness(payload)
        existing = load_existing(pull_r2=not args.no_upload,
                                 allow_initialize=args.initialize_empty_canonical)
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 2
    incoming = normalize(rows)
    print(f"  stored: {len(existing)} rows")

    gap = detect_gap(existing, incoming, retention_days=retention)
    if gap["gap"] and gap["ring_oldest"] is None:
        print(f"  GAP: {gap['reason']} - up to {gap['missing_business_days']} session(s) "
              f"unaccounted for. Only an IBKR Flex/activity pull can recover them.")
    elif gap["gap"]:
        print(f"  GAP: ring starts {gap['ring_oldest']} but our newest session is "
              f"{gap['stored_newest']} - {gap['missing_business_days']} business day(s) "
              f"aged out unseen. Only an IBKR Flex/activity pull can recover them.")
    elif gap["reason"]:
        print(f"  ring: {gap['reason']}")
    merged, stats = merge_fills(existing, incoming)
    print(f"  merged: +{stats['rows_new']} new, {stats['rows_updated']} re-seen, "
          f"{stats['rows_after']} total")

    summary = summarize(merged)
    summary.update(stats)
    summary["gap"] = gap
    summary["retention_days"] = retention
    summary["completeness"] = extend_canonical_coverage(payload["completeness"], existing)
    summary["complete"] = not gap["gap"]
    summary["asof_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    print(f"  coverage: {summary['first_session']} -> {summary['last_session']} "
          f"({summary['sessions']} sessions, {summary['tagged_pct']}% strategy-tagged)")
    print(f"  by account: {summary['accounts']}")

    if args.dry_run:
        print("  dry run - nothing written")
        if args.summary_json:
            print(f"  dry run - status not written to {args.summary_json}")
        return _gap_exit(gap, args)

    LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(LOCAL_PATH, index=False)
    summary["canonical_sha256"] = hashlib.sha256(LOCAL_PATH.read_bytes()).hexdigest()
    print(f"  wrote {LOCAL_PATH} ({LOCAL_PATH.stat().st_size:,} bytes)")

    status_path = Path(args.summary_json) if args.summary_json else None
    if status_path:
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"  status -> {status_path}")

    if not args.no_upload:
        # Both files are declared producer outputs, and the supervisor VERIFIES
        # R2 rather than uploading for us: anything skipped here fails the job.
        try:
            from cache_io import upload_from_local
            publish_canonical(merged, existing)
            if status_path and not upload_from_local(str(status_path), STATUS_R2_KEY):
                raise RuntimeError("fill status upload was not confirmed")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL: R2 upload failed ({e})")
            return 2

    return _gap_exit(gap, args)


def _gap_exit(gap: dict, args: argparse.Namespace) -> int:
    if gap["gap"] and args.assert_no_gap:
        print("FAIL: --assert-no-gap and the ring has a hole")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

