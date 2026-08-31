"""Crash-safe delivery receipts for the Daily Pitch email.

SMTP does not offer an idempotency key.  The safest unattended contract is
therefore at-most-once automatic sending: persist a ``sending`` receipt before
the SMTP call, promote it to ``sent`` only after the call succeeds, and treat
every uncertain outcome as ``ambiguous``.  A rerun may reconcile Sheets and
the journal after a confirmed send, but it must never send the same verdict a
second time.

The production receipt is mirrored to R2 before SMTP is attempted.  Tests and
development runs can pass an explicit receipt path, which is deliberately
local-only so they cannot mutate production evidence.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import uuid
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
RECEIPT_DIR = ROOT / "data" / "pitch_delivery_receipts"
R2_RECEIPT_PREFIX = "pitch_delivery_receipts"
R2_DOWNLOAD_DIR = ROOT / "artifacts" / "pitch_delivery_receipts"
R2_JOURNAL_DOWNLOAD_DIR = ROOT / "artifacts" / "pitch_delivery_journal"
SCHEMA = "daily-pitch-delivery.v1"
STATUSES = {"sending", "sent", "ambiguous"}
VERDICT_KINDS = {"idea", "killed", "stand_down", "short_slate"}


class DeliveryReceiptError(RuntimeError):
    """The receipt state is unsafe or inconsistent with the planned send."""


def default_receipt_path(asof: str | dt.date) -> Path:
    return RECEIPT_DIR / f"{asof}.json"


def r2_key(asof: str | dt.date) -> str:
    return f"{R2_RECEIPT_PREFIX}/{asof}.json"


def _canonical_record(record: dict) -> dict:
    """Remove journal-write metadata that is not part of the verdict."""
    return {key: value for key, value in record.items()
            if key not in {"written_at"}}


def _persistable_receipt(receipt: dict) -> dict:
    return {key: value for key, value in receipt.items()
            if not str(key).startswith("_r2_")}


@contextmanager
def _receipt_lock(path: Path):
    """Serialize same-machine claims without deleting a lock file."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def verdict_records(records: Iterable[dict], asof: str | None = None
                    ) -> list[dict]:
    selected = []
    for record in records:
        if record.get("kind") not in VERDICT_KINDS:
            continue
        if asof is not None and str(record.get("date")) != str(asof):
            continue
        selected.append(_canonical_record(record))
    return selected


def _canonical_lines(records: Iterable[dict]) -> list[str]:
    return sorted(json.dumps(_canonical_record(record), sort_keys=True,
                             separators=(",", ":"), ensure_ascii=True)
                  for record in records)


def verdict_digest(records: Iterable[dict], asof: str | None = None) -> str:
    canonical = _canonical_lines(verdict_records(records, asof))
    return _canonical_digest(canonical)


def _canonical_digest(canonical: Iterable[str]) -> str:
    canonical = sorted(canonical)
    payload = json.dumps(canonical, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def message_digest(subject: str, html: str, recipients: Iterable[str]) -> str:
    payload = json.dumps({
        "subject": subject,
        "html": html,
        "recipients": sorted(str(item).strip().lower()
                             for item in recipients if str(item).strip()),
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _read(path: Path) -> dict:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeliveryReceiptError(f"receipt is unreadable at {path}: {exc}") from exc
    if receipt.get("schema") != SCHEMA:
        raise DeliveryReceiptError(
            f"receipt at {path} has unsupported schema {receipt.get('schema')!r}")
    if receipt.get("status") not in STATUSES:
        raise DeliveryReceiptError(
            f"receipt at {path} has invalid status {receipt.get('status')!r}")
    required = {"date", "delivery_id", "verdict_digest", "verdict_count",
                "message_digest", "subject", "recipients"}
    missing = sorted(required - receipt.keys())
    if missing:
        raise DeliveryReceiptError(
            f"receipt at {path} is missing required field(s): {missing}")
    return receipt


def _atomic_write(path: Path, receipt: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _is_not_found(message: str) -> bool:
    lowered = message.lower()
    return any(token in lowered for token in ("404", "nosuchkey", "not found"))


def _load_remote(asof: str, *, required: bool) -> dict | None:
    try:
        import cache_io
    except Exception as exc:  # noqa: BLE001
        raise DeliveryReceiptError(f"cannot import R2 receipt support: {exc}") from exc

    if not cache_io.is_configured():
        raise DeliveryReceiptError(
            "R2 is not configured; production email delivery is blocked")

    key = r2_key(asof)
    before = cache_io.head(key)
    target = R2_DOWNLOAD_DIR / f"{asof}.json"
    if cache_io.download_to_local(key, str(target)):
        after = cache_io.head(key)
        before_etag = (before or {}).get("ETag")
        after_etag = (after or {}).get("ETag")
        if not before_etag or before_etag != after_etag:
            raise DeliveryReceiptError(
                f"R2 delivery receipt changed while it was read for {asof}")
        receipt = _read(target)
        receipt["_r2_etag"] = after_etag
        return receipt

    detail = str(cache_io.last_download_error() or "unknown R2 download error")
    if not required and _is_not_found(detail):
        return None
    raise DeliveryReceiptError(
        f"R2 delivery receipt could not be loaded for {asof}: {detail}")


def load_receipt(path: Path, asof: str, *, use_r2: bool,
                 require_remote: bool = False) -> dict | None:
    """Load one receipt, rejecting local/R2 divergence.

    R2 is authoritative in production, but a local-only receipt is never
    silently ignored: that state can mean SMTP succeeded just before an R2
    update failed.  Requiring an operator to resolve it is safer than sending
    a duplicate.
    """
    local = _read(path) if path.exists() else None
    if not use_r2:
        return local

    remote = _load_remote(asof, required=require_remote or local is not None)
    if remote is None:
        return None
    remote_on_disk = _persistable_receipt(remote)
    if local is not None and local != remote_on_disk:
        raise DeliveryReceiptError(
            f"local and R2 delivery receipts disagree for {asof}; "
            "automatic sending is blocked")
    if local is None:
        _atomic_write(path, remote_on_disk)
    return remote


def _persist(receipt: dict, path: Path, *, use_r2: bool,
             create_only: bool = False,
             expected_etag: str | None = None) -> str | None:
    on_disk = _persistable_receipt(receipt)
    _atomic_write(path, on_disk)
    if not use_r2:
        return None
    try:
        import cache_io
    except Exception as exc:  # noqa: BLE001
        raise DeliveryReceiptError(f"cannot import R2 receipt support: {exc}") from exc
    if not cache_io.is_configured():
        raise DeliveryReceiptError(
            "R2 is not configured; production email delivery is blocked")
    if not create_only and expected_etag is None:
        raise DeliveryReceiptError(
            "R2 delivery receipt update is missing its compare-and-swap ETag")
    result, etag = cache_io.conditional_upload_from_local(
        str(path), r2_key(on_disk["date"]), create_only=create_only,
        expected_etag=expected_etag)
    if result == "precondition_failed":
        raise DeliveryReceiptError(
            f"R2 delivery receipt changed concurrently for {on_disk['date']}")
    if result != "uploaded" or not etag:
        raise DeliveryReceiptError(
            f"could not persist delivery receipt to R2 for {on_disk['date']}")
    return etag


def reserve_delivery(*, asof: str, records: list[dict], subject: str,
                     html: str, recipients: list[str], path: Path,
                     use_r2: bool, prior_records: list[dict] | None = None
                     ) -> tuple[dict, bool]:
    """Reserve one send. Return ``(receipt, should_send)``.

    A matching ``sent`` receipt returns ``False`` so the caller can reconcile
    Sheets/journal without another email.  Every other existing state blocks.
    """
    planned = verdict_records(records, asof)
    digest = verdict_digest(planned)
    prior = verdict_records(prior_records or [], asof)

    with _receipt_lock(path):
        existing = load_receipt(path, asof, use_r2=use_r2)
        if existing is not None:
            if existing.get("status") != "sent":
                raise DeliveryReceiptError(
                    f"delivery receipt for {asof} is "
                    f"{existing.get('status')}; the SMTP outcome must be "
                    "resolved before any rerun")

            last_digest = existing.get(
                "delivery_digest", existing.get("verdict_digest"))
            if last_digest == digest:
                return existing, False

            planned_ideas = [record for record in planned
                             if record.get("kind") == "idea"]
            prior_ideas = [record for record in prior
                           if record.get("kind") == "idea"]
            prior_stand_down = [record for record in prior
                                if record.get("kind") == "stand_down"]
            directed_amendment = (
                bool(planned_ideas)
                and all(str(record.get("directed_by", "")).strip()
                        for record in planned_ideas)
                and not any(record.get("kind") in {"stand_down", "short_slate"}
                            for record in planned)
                and len(prior_stand_down) == 1
                and not prior_ideas
            )
            if not directed_amendment:
                raise DeliveryReceiptError(
                    f"a delivery receipt already exists for {asof}, but its "
                    "verdict digest differs from this payload")
            if (existing.get("verdict_digest") != verdict_digest(prior)
                    or int(existing.get("verdict_count", -1)) != len(prior)):
                raise DeliveryReceiptError(
                    f"the sent receipt and journal disagree before the "
                    f"directed amendment for {asof}")
            if Counter(_canonical_lines(prior)) & Counter(
                    _canonical_lines(planned)):
                raise DeliveryReceiptError(
                    f"the directed amendment duplicates an existing verdict "
                    f"record for {asof}")

            now = dt.datetime.now(dt.timezone.utc).isoformat(
                timespec="seconds")
            history = list(existing.get("delivery_history") or [])
            history.append({key: existing.get(key) for key in (
                "delivery_id", "delivery_digest", "delivery_count",
                "message_digest", "subject", "recipients", "sent_at")})
            combined = prior + planned
            receipt = {
                "schema": SCHEMA,
                "status": "sending",
                "date": asof,
                "delivery_id": str(uuid.uuid4()),
                "delivery_digest": digest,
                "delivery_count": len(planned),
                "verdict_digest": verdict_digest(combined),
                "verdict_count": len(combined),
                "message_digest": message_digest(subject, html, recipients),
                "subject": subject,
                "recipients": sorted(recipients),
                "created_at": existing.get("created_at", now),
                "updated_at": now,
                "smtp_attempted_at": now,
                "delivery_history": history,
                "_r2_etag": existing.get("_r2_etag"),
            }
            new_etag = _persist(
                receipt, path, use_r2=use_r2,
                expected_etag=existing.get("_r2_etag"))
            receipt["_r2_etag"] = new_etag
            return receipt, True

        now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        receipt = {
            "schema": SCHEMA,
            "status": "sending",
            "date": asof,
            "delivery_id": str(uuid.uuid4()),
            "delivery_digest": digest,
            "delivery_count": len(planned),
            "verdict_digest": digest,
            "verdict_count": len(planned),
            "message_digest": message_digest(subject, html, recipients),
            "subject": subject,
            "recipients": sorted(recipients),
            "created_at": now,
            "updated_at": now,
            "smtp_attempted_at": now,
        }
        new_etag = _persist(receipt, path, use_r2=use_r2, create_only=True)
        receipt["_r2_etag"] = new_etag
        return receipt, True


def complete_delivery(receipt: dict, path: Path, *, use_r2: bool,
                      sent: bool, reason: str | None = None) -> dict:
    """Record the SMTP result, conservatively marking failures ambiguous."""
    with _receipt_lock(path):
        current = load_receipt(
            path, str(receipt["date"]), use_r2=use_r2,
            require_remote=use_r2)
        if current is None:
            raise DeliveryReceiptError("reserved delivery receipt disappeared")
        if (current.get("delivery_id") != receipt.get("delivery_id")
                or current.get("status") != "sending"):
            raise DeliveryReceiptError(
                "reserved delivery receipt changed before SMTP completion")

        updated = dict(current)
        updated["status"] = "sent" if sent else "ambiguous"
        updated["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="seconds")
        if sent:
            updated["sent_at"] = updated["updated_at"]
            updated.pop("ambiguity_reason", None)
        else:
            updated["ambiguity_reason"] = (
                reason or "SMTP did not confirm delivery")

        expected_etag = current.get("_r2_etag")
        try:
            new_etag = _persist(
                updated, path, use_r2=use_r2,
                expected_etag=expected_etag)
            updated["_r2_etag"] = new_etag
        except DeliveryReceiptError as exc:
            # The SMTP call may already have succeeded. Never leave the local
            # receipt looking safely sent when durable R2 state is uncertain.
            updated["status"] = "ambiguous"
            updated["updated_at"] = dt.datetime.now(
                dt.timezone.utc).isoformat(timespec="seconds")
            updated["ambiguity_reason"] = (
                "receipt persistence failed after SMTP outcome: " + str(exc))
            _atomic_write(path, _persistable_receipt(updated))
            if use_r2 and expected_etag:
                try:
                    _persist(updated, path, use_r2=True,
                             expected_etag=expected_etag)
                except DeliveryReceiptError:
                    pass
            raise
        return updated


def reconcile_journal(records: list[dict], journal_path: Path) -> int:
    """Append only missing verdict records and reject conflicting history."""
    with _receipt_lock(journal_path):
        return _reconcile_journal_locked(records, journal_path)


def _reconcile_journal_locked(records: list[dict], journal_path: Path) -> int:
    import pitch_journal

    if not records:
        return 0
    dates = {str(record.get("date")) for record in records}
    if len(dates) != 1:
        raise DeliveryReceiptError(
            f"planned verdict spans multiple dates: {sorted(dates)}")
    asof = next(iter(dates))
    planned = verdict_records(records, asof)
    planned_lines = _canonical_lines(planned)
    existing = verdict_records(
        pitch_journal.load(journal_path, pull=False), asof)
    existing_lines = _canonical_lines(existing)

    planned_counts = Counter(planned_lines)
    existing_counts = Counter(existing_lines)
    unexpected = existing_counts - planned_counts
    if unexpected:
        planned_ideas = [record for record in planned
                         if record.get("kind") == "idea"]
        stand_downs = [record for record in existing
                       if record.get("kind") == "stand_down"]
        unexpected_ideas = Counter(
            json.dumps(record, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True)
            for record in existing if record.get("kind") == "idea")
        unexpected_ideas -= planned_counts
        directed_amendment = (
            bool(planned_ideas)
            and all(str(record.get("directed_by", "")).strip()
                    for record in planned_ideas)
            and not any(record.get("kind") in {"stand_down", "short_slate"}
                        for record in planned)
            and len(stand_downs) == 1
            and not unexpected_ideas
        )
        if not directed_amendment:
            raise DeliveryReceiptError(
                f"journal already contains conflicting verdict records for "
                f"{asof}")
        target_counts = existing_counts | planned_counts
    else:
        target_counts = planned_counts

    missing_counts = planned_counts - existing_counts
    missing: list[dict] = []
    for record in planned:
        line = json.dumps(record, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True)
        if missing_counts[line] > 0:
            missing.append(record)
            missing_counts[line] -= 1

    written = pitch_journal.append(missing, journal_path)
    final = verdict_records(pitch_journal.load(journal_path, pull=False), asof)
    final_lines = _canonical_lines(final)
    if (Counter(final_lines) != target_counts
            or _canonical_digest(final_lines)
            != _canonical_digest(target_counts.elements())):
        raise DeliveryReceiptError(
            f"journal reconciliation digest mismatch for {asof}")

    if journal_path == pitch_journal.JOURNAL_PATH:
        try:
            import cache_io
        except Exception as exc:  # noqa: BLE001
            raise DeliveryReceiptError(
                f"cannot import R2 journal support: {exc}") from exc
        if not cache_io.is_configured():
            raise DeliveryReceiptError(
                "R2 is not configured; journal reconciliation is incomplete")
        if not cache_io.upload_from_local(
                str(journal_path), pitch_journal.JOURNAL_R2_KEY):
            raise DeliveryReceiptError(
                f"journal upload to R2 failed for {asof}")
        cloud = load_cloud_journal()
        cloud_today = verdict_records(cloud, asof)
        if (verdict_digest(cloud_today) != verdict_digest(final)
                or len(cloud_today) != len(final)):
            raise DeliveryReceiptError(
                f"R2 journal verification failed for {asof}")
    return written


def load_cloud_journal() -> list[dict]:
    """Download and parse the production journal from R2, fail closed."""
    import pitch_journal
    try:
        import cache_io
    except Exception as exc:  # noqa: BLE001
        raise DeliveryReceiptError(
            f"cannot import R2 journal support: {exc}") from exc
    if not cache_io.is_configured():
        raise DeliveryReceiptError(
            "R2 is not configured; cloud journal cannot be verified")
    target = R2_JOURNAL_DOWNLOAD_DIR / pitch_journal.JOURNAL_R2_KEY
    if not cache_io.download_to_local(
            pitch_journal.JOURNAL_R2_KEY, str(target)):
        detail = str(cache_io.last_download_error() or "unknown R2 error")
        raise DeliveryReceiptError(
            f"R2 journal download failed: {detail}")
    return pitch_journal.load(target, pull=False)


def verify_sent_receipt(receipt: dict, records: list[dict], asof: str) -> None:
    if receipt.get("date") != asof:
        raise DeliveryReceiptError(
            f"receipt date {receipt.get('date')!r} does not match {asof}")
    if receipt.get("status") != "sent":
        raise DeliveryReceiptError(
            f"delivery receipt for {asof} is {receipt.get('status')}, not sent")
    current = verdict_records(records, asof)
    digest = verdict_digest(current)
    if receipt.get("verdict_digest") != digest:
        raise DeliveryReceiptError(
            f"journal and sent receipt verdict digests differ for {asof}")
    try:
        recorded_count = int(receipt.get("verdict_count", -1))
    except (TypeError, ValueError) as exc:
        raise DeliveryReceiptError(
            f"sent receipt has invalid verdict_count "
            f"{receipt.get('verdict_count')!r}") from exc
    if recorded_count != len(current):
        raise DeliveryReceiptError(
            f"journal has {len(current)} verdict record(s), but the sent "
            f"receipt records {receipt.get('verdict_count')}")
