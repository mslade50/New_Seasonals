"""Atomic plans/state plus an append-only execution audit journal."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    DATABENTO_DATASET,
    DATABENTO_SCHEMA,
    MARKETS,
    RULES,
    STRATEGY_VERSION,
    TRANSIENT_STATES,
)


class LockBusyError(RuntimeError):
    """Raised only when a non-blocking process lock cannot be acquired."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def read_json(path: Path, *, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def exclusive_file_lock(lock_path: Path) -> Iterator[None]:
    """Take a non-blocking, crash-released one-byte OS file lock."""

    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    if lock_path.stat().st_size == 0:
        handle.write(b"\0")
        handle.flush()
    handle.seek(0)
    acquired = False
    try:
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise LockBusyError(f"lock is already held: {lock_path}") from exc
        else:  # pragma: no cover - production host is Windows
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise LockBusyError(f"lock is already held: {lock_path}") from exc
        acquired = True
        yield
    finally:
        try:
            if acquired:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:  # pragma: no cover - production host is Windows
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


@contextmanager
def exclusive_machine_mutex(name: str) -> Iterator[None]:
    """Hold one crash-released mutex across every clone on this Windows host."""

    if os.name != "nt":  # pragma: no cover - production host is Windows
        with exclusive_file_lock(
            Path(tempfile.gettempdir()) / f"{_mutex_filename(name)}.lock"
        ):
            yield
        return
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_mutex = kernel32.CreateMutexW
    create_mutex.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
    create_mutex.restype = ctypes.c_void_p
    wait_for = kernel32.WaitForSingleObject
    wait_for.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    wait_for.restype = ctypes.c_uint32
    release = kernel32.ReleaseMutex
    release.argtypes = [ctypes.c_void_p]
    close = kernel32.CloseHandle
    close.argtypes = [ctypes.c_void_p]
    handle = create_mutex(None, False, name)
    if not handle:
        raise OSError(ctypes.get_last_error(), "CreateMutexW failed")
    acquired = False
    try:
        wait_result = wait_for(handle, 0)
        if wait_result == 0x00000102:  # WAIT_TIMEOUT
            raise LockBusyError(f"machine mutex is already held: {name}")
        if wait_result not in {0x00000000, 0x00000080}:  # OBJECT_0 / ABANDONED
            raise OSError(f"WaitForSingleObject failed: {wait_result}")
        acquired = True
        yield
    finally:
        if acquired:
            release(handle)
        close(handle)


def _mutex_filename(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:24]


def finalize_plan(payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body.setdefault("strategy_version", STRATEGY_VERSION)
    body.pop("plan_hash", None)
    body["plan_hash"] = content_hash(body)
    return body


def validate_plan(plan: dict[str, Any], *, entry_date: str) -> None:
    if plan.get("strategy_version") != STRATEGY_VERSION:
        raise ValueError("signal plan strategy version mismatch")
    if plan.get("entry_date") != entry_date:
        raise ValueError("signal plan is not for today's entry session")
    expected = str(plan.get("plan_hash") or "")
    body = dict(plan)
    body.pop("plan_hash", None)
    if not expected or content_hash(body) != expected:
        raise ValueError("signal plan hash mismatch")
    allowed_top = {
        "strategy_version",
        "entry_date",
        "setup_date",
        "created_at",
        "data_as_of",
        "dataset",
        "schema",
        "request_start",
        "cache_dir",
        "quoted_cost_usd",
        "quoted_billable_bytes",
        "markets",
        "plan_hash",
    }
    unknown_top = set(plan).difference(allowed_top)
    if unknown_top:
        raise ValueError(f"signal plan has unknown fields: {sorted(unknown_top)}")
    if plan.get("dataset") != DATABENTO_DATASET or plan.get("schema") != DATABENTO_SCHEMA:
        raise ValueError("signal plan data source/schema mismatch")
    for field in ("created_at", "data_as_of", "request_start"):
        stamp = pd.Timestamp(plan.get(field))
        if stamp.tz is None:
            raise ValueError(f"signal plan {field} must be timezone-aware")
    try:
        quoted_cost = float(plan.get("quoted_cost_usd"))
        quoted_bytes = int(plan.get("quoted_billable_bytes"))
    except (TypeError, ValueError) as exc:
        raise ValueError("signal plan quote fields are invalid") from exc
    if not math.isfinite(quoted_cost) or quoted_cost < 0 or quoted_bytes < 0:
        raise ValueError("signal plan quote fields must be finite and non-negative")
    markets = plan.get("markets")
    if not isinstance(markets, list) or len(markets) != len(MARKETS):
        raise ValueError("signal plan must contain exactly the pinned market set")
    expected_markets = {market.root: market for market in MARKETS}
    seen: set[str] = set()
    allowed_market_fields = {
        "root",
        "futures_symbol",
        "etf",
        "qualifies",
        "reason",
        "setup_date",
        "entry_date",
        "instrument_id",
        "entry_instrument_id",
        "trend_direction",
        "trend_ratio",
        "rth_bar_count",
    }
    for item in markets:
        if not isinstance(item, dict):
            raise TypeError("signal plan market entries must be objects")
        unknown = set(item).difference(allowed_market_fields)
        if unknown:
            raise ValueError(f"signal plan market has unknown fields: {sorted(unknown)}")
        root = str(item.get("root") or "")
        if root not in expected_markets or root in seen:
            raise ValueError("signal plan roots are unknown or duplicated")
        seen.add(root)
        pinned = expected_markets[root]
        if (
            item.get("futures_symbol") != pinned.continuous_symbol
            or item.get("etf") != pinned.etf
            or item.get("entry_date") != entry_date
        ):
            raise ValueError(f"signal plan mapping mismatch for {root}")
        if type(item.get("qualifies")) is not bool:
            raise TypeError(f"signal plan {root} qualifies must be boolean")
        reason = str(item.get("reason") or "")
        if not reason:
            raise ValueError(f"signal plan {root} has no decision reason")
        setup = item.get("setup_date")
        if setup is not None:
            setup_day = pd.Timestamp(setup)
            if setup_day.tz is not None or setup_day.normalize() >= pd.Timestamp(entry_date):
                raise ValueError(f"signal plan {root} setup date is invalid")
        for field in ("instrument_id", "entry_instrument_id"):
            value = item.get(field)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"signal plan {root} {field} is invalid")
        direction = item.get("trend_direction")
        if direction not in {None, -1, 0, 1}:
            raise ValueError(f"signal plan {root} trend direction is invalid")
        ratio = item.get("trend_ratio")
        if ratio is not None:
            ratio = float(ratio)
            if not math.isfinite(ratio) or not 0 <= ratio <= 1:
                raise ValueError(f"signal plan {root} trend ratio is invalid")
        bar_count = item.get("rth_bar_count", 0)
        if isinstance(bar_count, bool) or not isinstance(bar_count, int) or not 0 <= bar_count <= 26:
            raise ValueError(f"signal plan {root} RTH bar count is invalid")
        if item["qualifies"] and (
            reason != "qualified"
            or setup is None
            or direction not in {-1, 1}
            or ratio is None
            or ratio < RULES.trend_ratio_min
            or bar_count != 26
            or item.get("instrument_id") is None
            or item.get("instrument_id") != item.get("entry_instrument_id")
        ):
            raise ValueError(f"signal plan {root} qualified fields are inconsistent")
    if seen != set(expected_markets):
        raise ValueError("signal plan does not contain every pinned root")


def signal_identity(
    *,
    account: str,
    root: str,
    etf: str,
    setup_date: str,
    entry_date: str,
    direction: int,
) -> str:
    raw = "|".join(
        [
            account,
            STRATEGY_VERSION,
            root,
            etf,
            setup_date,
            entry_date,
            str(direction),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class StateStore:
    """Account-scoped session state with explicit ambiguous-restart handling."""

    def __init__(
        self,
        state_path: Path,
        audit_path: Path,
        *,
        alert_path: Path | None = None,
        alert_popup: bool = False,
    ):
        self.state_path = Path(state_path)
        self.audit_path = Path(audit_path)
        self.alert_path = Path(alert_path or self.state_path.parent / "critical_alert.json")
        self.alert_popup = bool(alert_popup)
        self._alerted: set[tuple[str, str, str]] = set()

    def load(self) -> dict[str, Any]:
        state = read_json(
            self.state_path,
            default={
                "strategy_version": STRATEGY_VERSION,
                "updated_at": utc_now_iso(),
                "signals": {},
            },
        )
        if state.get("strategy_version") != STRATEGY_VERSION:
            raise ValueError("state strategy version mismatch")
        return state

    def recover_interrupted(self) -> dict[str, Any]:
        state = self.load()
        changed = False
        for signal_id, record in state.get("signals", {}).items():
            if record.get("state") in TRANSIENT_STATES:
                previous = record["state"]
                record["state"] = "unknown"
                record["unknown_reason"] = f"restart_during_{previous}"
                record["updated_at"] = utc_now_iso()
                self.append_audit(
                    "restart_marked_unknown",
                    signal_id=signal_id,
                    previous_state=previous,
                )
                changed = True
        if changed:
            state["updated_at"] = utc_now_iso()
            atomic_write_json(self.state_path, state)
        return state

    def get(self, signal_id: str) -> dict[str, Any] | None:
        return self.load().get("signals", {}).get(signal_id)

    def put(self, signal_id: str, record: dict[str, Any]) -> dict[str, Any]:
        state = self.load()
        value = dict(record)
        value["updated_at"] = utc_now_iso()
        state.setdefault("signals", {})[signal_id] = value
        state["updated_at"] = utc_now_iso()
        atomic_write_json(self.state_path, state)
        self.append_audit(
            "state_write",
            signal_id=signal_id,
            state=value.get("state"),
            record=value,
        )
        state_name = str(value.get("state") or "")
        if state_name.startswith("critical_") or value.get("degraded_execution"):
            self._emit_critical_alert(signal_id, value)
        return value

    def put_many(
        self, records: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Persist a related state transition with one atomic file sync."""

        if not records:
            return {}
        state = self.load()
        now = utc_now_iso()
        values: dict[str, dict[str, Any]] = {}
        for signal_id, record in records.items():
            value = dict(record)
            value["updated_at"] = now
            state.setdefault("signals", {})[signal_id] = value
            values[signal_id] = value
        state["updated_at"] = now
        atomic_write_json(self.state_path, state)
        self.append_audit(
            "state_batch_write",
            signals={
                signal_id: value.get("state")
                for signal_id, value in values.items()
            },
        )
        for signal_id, value in values.items():
            state_name = str(value.get("state") or "")
            if state_name.startswith("critical_") or value.get(
                "degraded_execution"
            ):
                self._emit_critical_alert(signal_id, value)
        return values

    def _emit_critical_alert(self, signal_id: str, record: dict[str, Any]) -> None:
        reason = str(
            record.get("critical_reason")
            or "; ".join(map(str, record.get("degraded_reasons") or []))
            or "Legend ETF execution degraded"
        )
        key = (signal_id, str(record.get("state") or ""), reason)
        if key in self._alerted:
            return
        self._alerted.add(key)
        payload = {
            "at": utc_now_iso(),
            "strategy_version": STRATEGY_VERSION,
            "signal_id": signal_id,
            "state": record.get("state"),
            "account": record.get("account"),
            "root": record.get("root"),
            "etf": record.get("etf"),
            "reason": reason,
        }
        atomic_write_json(self.alert_path, payload)
        self.append_audit("critical_alert_emitted", **payload)
        if self.alert_popup and os.name == "nt":
            message = (
                f"LEGEND ETF CRITICAL: {payload.get('account')}/"
                f"{payload.get('etf')} {reason}. Inspect {self.alert_path} now."
            )
            try:
                subprocess.Popen(
                    ["msg.exe", "*", "/TIME:300", message],
                    creationflags=0x08000000,
                    close_fds=True,
                )
            except OSError as exc:
                self.append_audit(
                    "critical_popup_failed", error_type=type(exc).__name__
                )

    def append_audit(self, event: str, **payload: Any) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        line = canonical_json(
            {"at": utc_now_iso(), "event": event, "strategy_version": STRATEGY_VERSION, **payload}
        )
        with self.audit_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    @contextmanager
    def exclusive_session(self) -> Iterator[None]:
        """Hold an OS-released lock for one complete runner invocation.

        Live, dry, check, and shadow state files share one lock directory, so
        two scheduled/manual processes cannot race through broker preflight.
        The operating system releases the lock if the process crashes.
        """

        # A named machine mutex is independent of checkout/worktree and state
        # paths. It protects the shared IBKR client/account namespace across
        # live, dry, check, and shadow invocations on this host.
        try:
            with exclusive_machine_mutex(
                r"Global\NewSeasonals.LegendETF.Runner.v1"
            ):
                yield
        except LockBusyError as exc:
            raise RuntimeError(
                "another Legend ETF runner already holds the session lock"
            ) from exc
