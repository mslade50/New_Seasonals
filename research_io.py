"""Durable, locally serialized research evidence; corruption is never skipped."""
from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path

_locks_guard = threading.Lock()
_locks: dict[str, threading.RLock] = {}
_held = threading.local()


def _invalid_constant(value: str):
    raise ValueError(f"invalid JSON numeric constant: {value}")


@contextmanager
def file_lock(path: Path):
    """Reentrant thread/process lock. Persistent lock files are intentional."""
    path = Path(path).resolve()
    key = str(path).casefold() if os.name == "nt" else str(path)
    with _locks_guard:
        lock = _locks.setdefault(key, threading.RLock())
    with lock:
        held = getattr(_held, "paths", set())
        if key in held:
            yield
            return
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
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            _held.paths = held | {key}
            try:
                yield
            finally:
                _held.paths = held
                handle.seek(0)
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with file_lock(path):
        records = []
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"corrupt journal encoding; evidence preserved: {path}") from exc
        for number, line in enumerate(content.splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line, parse_constant=_invalid_constant)
            except ValueError as exc:
                raise ValueError(f"corrupt journal at line {number}; evidence preserved: {path}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"invalid journal object at line {number}: {path}")
            records.append(record)
        return records


def append_jsonl(path: Path, records: list[dict]) -> None:
    # Serialize the entire batch first so encoding errors cannot write half a batch.
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("invalid journal record: every record must be an object")
    encoded = "".join(json.dumps(r, allow_nan=False) + "\n" for r in records).encode("utf-8")
    path = Path(path)
    with file_lock(path):
        read_jsonl(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell():
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    handle.write(b"\n")
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
