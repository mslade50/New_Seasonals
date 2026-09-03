"""Copy the execution-broker's /fills ring into a durable R2-canonical parquet.

The broker Durable Object is the only place actual IBKR executions accumulate:
`book_snapshot.py` pushes each account's fills with the book, the DO folds them
into per-day keys, and it drops them after `retention_days` (14). IBKR's API
only ever serves the CURRENT session's executions, so a row that ages out of
the ring is gone from every machine-readable record we control -- only IBKR's
own Flex/activity statements still have it.

This harvest is the durable copy. Store: `data/live_fills.parquet`, R2 key
`live_fills.parquet` (gitignored like every other cache; R2 is canonical).

Two properties matter more than speed:

1. **Upsert, never append.** The DO upserts by `exec_id` because commission
   reports lag the fill by a beat, and an MOC fill can miss the day's last book
   push entirely and only appear in tomorrow's ring. So the same `exec_id` is
   re-fetched with better data; we merge rather than duplicate, and we never
   let a later fetch NULL out a commission we already stored.
2. **Loud on gaps.** If the machine is off for longer than the retention
   window, rows are lost silently. The oldest row in the ring is compared to
   the newest row we hold: a hole raises a GAP warning (and exits non-zero
   under --assert-no-gap) instead of a green run over missing history.

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
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEFAULT_BROKER_URL = "https://execution-broker.mckinleyslade.workers.dev"
LOCAL_PATH = _ROOT / "data" / "live_fills.parquet"
R2_KEY = "live_fills.parquet"
STATUS_R2_KEY = "live_fills_status.json"
ET = "America/New_York"

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
        raw = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.NA
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
    """Upsert `incoming` into `existing` by exec_id, never losing enrichment.

    The broker is upstream truth, so a re-fetched row wins field for field --
    except on ENRICHMENT_COLUMNS, where a null incoming value keeps whatever we
    already stored. That is the commission-lag case: the same execution comes
    back later with the commission attached, and could in principle come back
    again without it.

    Raises if any execution we already held would not survive the merge. This
    store is the only durable copy, so losing a row has to stop the run rather
    than quietly write a smaller file.
    """
    existing = empty_frame() if existing is None or existing.empty else existing.copy()
    incoming = empty_frame() if incoming is None or incoming.empty else incoming.copy()
    for frame in (existing, incoming):
        for col in COLUMNS:
            if col not in frame.columns:
                frame[col] = pd.NA

    before = set(existing["exec_id"].dropna().astype(str))
    arriving = set(incoming["exec_id"].dropna().astype(str))
    new_ids = arriving - before
    seen_again = arriving & before

    # Carry stored enrichment onto re-fetched rows that arrive without it.
    if seen_again and not existing.empty:
        stored = existing.set_index(existing["exec_id"].astype(str))
        idx = incoming["exec_id"].astype(str)
        for col in ENRICHMENT_COLUMNS:
            prior = idx.map(stored[col]) if col in stored.columns else pd.Series(pd.NA, index=incoming.index)
            incoming[col] = pd.to_numeric(incoming[col], errors="coerce").fillna(
                pd.to_numeric(prior, errors="coerce")
            )

    # Concat only the non-empty frames: an all-NA frame would coerce dtypes.
    parts = [f[list(COLUMNS)] for f in (existing, incoming) if not f.empty]
    combined = pd.concat(parts, ignore_index=True) if parts else empty_frame()
    # Incoming rows sit last, so keeping the last duplicate makes the broker win.
    combined = combined.drop_duplicates(subset=["exec_id"], keep="last")
    combined = combined.sort_values(["time_utc", "exec_id"], kind="stable").reset_index(drop=True)

    # The invariant is set containment, not row count: every execution we held
    # must survive the merge. A count check would miss a row dropped while new
    # ones arrive, and would false-alarm on the harmless dedup of a store that
    # somehow holds the same exec_id twice.
    kept = set(combined["exec_id"].dropna().astype(str))
    lost = before - kept
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


def detect_gap(existing: pd.DataFrame, incoming: pd.DataFrame) -> dict:
    """Compare the ring's oldest session to our newest: a hole means lost rows.

    The ring keeps ~14 calendar days. If nothing harvested for longer than
    that, executions expired unseen and only an IBKR Flex pull can recover
    them, so this has to be loud rather than a green no-op.
    """
    info = {"gap": False, "ring_oldest": None, "stored_newest": None, "missing_business_days": 0}
    if incoming is None or incoming.empty:
        return info
    ring_oldest = str(incoming["session_date"].dropna().min())
    info["ring_oldest"] = ring_oldest
    if existing is None or existing.empty:
        return info
    stored_newest = str(existing["session_date"].dropna().max())
    info["stored_newest"] = stored_newest
    if not stored_newest or not ring_oldest:
        return info
    # Business days strictly between the newest stored session and the oldest
    # session still in the ring. Zero or one means the windows touch.
    span = pd.bdate_range(
        pd.Timestamp(stored_newest) + pd.Timedelta(days=1),
        pd.Timestamp(ring_oldest) - pd.Timedelta(days=1),
    )
    info["missing_business_days"] = int(len(span))
    info["gap"] = len(span) > 0
    return info


def fetch_fills(base_url: str, token: str, timeout: int = 45) -> dict:
    r = requests.get(
        f"{base_url.rstrip('/')}/fills",
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def load_existing(pull_r2: bool = True) -> pd.DataFrame:
    """Canonical store from R2, falling back to whatever is on disk."""
    if pull_r2:
        try:
            from cache_io import download_to_local, is_configured
            if is_configured():
                LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
                if download_to_local(R2_KEY, str(LOCAL_PATH)):
                    print(f"  pulled canonical {R2_KEY} from R2")
                else:
                    print(f"  no {R2_KEY} in R2 yet (first run)")
            else:
                print("  R2 not configured - using the local copy only")
        except Exception as e:  # noqa: BLE001
            print(f"  WARNING: R2 pull failed ({e}); merging into the local copy")
    if not LOCAL_PATH.exists():
        return empty_frame()
    try:
        return pd.read_parquet(LOCAL_PATH)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"FAIL: {LOCAL_PATH} exists but is unreadable ({e}); refusing to overwrite it")


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

    incoming = normalize(rows)
    existing = load_existing(pull_r2=True)
    print(f"  stored: {len(existing)} rows")

    gap = detect_gap(existing, incoming)
    if gap["gap"]:
        print(f"  GAP: ring starts {gap['ring_oldest']} but our newest session is "
              f"{gap['stored_newest']} - {gap['missing_business_days']} business day(s) "
              f"aged out unseen. Only an IBKR Flex/activity pull can recover them.")
    merged, stats = merge_fills(existing, incoming)
    print(f"  merged: +{stats['rows_new']} new, {stats['rows_updated']} re-seen, "
          f"{stats['rows_after']} total")

    summary = summarize(merged)
    summary.update(stats)
    summary["gap"] = gap
    summary["retention_days"] = retention
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
            from cache_io import upload_from_local, is_configured
            if is_configured():
                upload_from_local(str(LOCAL_PATH), R2_KEY)
                print(f"  uploaded to R2 as {R2_KEY}")
                if status_path:
                    upload_from_local(str(status_path), STATUS_R2_KEY)
                    print(f"  uploaded to R2 as {STATUS_R2_KEY}")
            else:
                print("  R2 not configured - local write only")
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

