"""PM Weekly journal: append-only JSONL in PM_AGENT_HOME, mirrored to R2.

Record kinds (docs/claude_ref/pm_agent.md, "Journal"):
    brief / stand_down   one per anchor week, written by the publisher
    forecast             one per claim, written by the publisher with the brief
    resolution           written by the grader once the outcome is in

Records are never edited. A correction is a new record with a reason.
R2 mirroring is explicit (push=True / pull=True): tests and dev runs that
point PM_AGENT_HOME at a temp folder never touch R2 unless asked.

Agent-product module: the book and the Risk Agent must not import it.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

from research_io import append_jsonl, read_jsonl

import pm_agent_universe as U

R2_JOURNAL_KEY = U.R2_PREFIX + "journal.jsonl"
VERDICT_KINDS = ("brief", "stand_down")


def sync_down(path: Path) -> bool:
    """Pull the R2 mirror over a MISSING local copy only. Never raises."""
    if Path(path).exists():
        return False
    try:
        import cache_io
        if not cache_io.is_configured():
            return False
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return bool(cache_io.download_to_local(R2_JOURNAL_KEY, str(path)))
    except Exception as exc:  # noqa: BLE001 - best effort by design
        print(f"[pm_agent_journal] sync_down skipped: {exc}")
        return False


def sync_up(path: Path) -> bool:
    import cache_io
    if not cache_io.is_configured():
        raise RuntimeError("R2 is not configured; the PM journal was not mirrored")
    if not cache_io.upload_from_local(str(path), R2_JOURNAL_KEY):
        raise RuntimeError("PM journal upload failed; local copy preserved")
    return True


def load(path: Path | None = None, pull: bool = False) -> list[dict]:
    path = Path(path or U.journal_path())
    if pull:
        sync_down(path)
    return read_jsonl(path)


def append(records: list[dict], path: Path | None = None, push: bool = False) -> None:
    if not records:
        return
    path = Path(path or U.journal_path())
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    append_jsonl(path, [{**r, "ts": r.get("ts") or stamp} for r in records])
    if push:
        sync_up(path)


def verdicts_for(records: list[dict], week: str) -> list[dict]:
    return [r for r in records if r.get("kind") in VERDICT_KINDS and r.get("week") == week]
