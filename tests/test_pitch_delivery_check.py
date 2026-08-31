"""Guards for scripts/check_pitch_delivered.py.

This is the only thing standing between a quiet agent failure and a green
Task Scheduler entry on a morning with no email, so its accepting cases (one
to three ideas, or a stand-down) and its rejecting cases all get a test.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

import pitch_delivery

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "check_pitch_delivered.py"
ASOF = "2026-08-07"


def read_records(journal_path: Path) -> list[dict]:
    return [json.loads(line) for line in journal_path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def write_receipt(journal_path: Path, asof: str = ASOF,
                  status: str = "sent", digest: str | None = None) -> Path:
    path = journal_path.with_name(f"{journal_path.stem}.receipt.json")
    records = read_records(journal_path)
    receipt, should_send = pitch_delivery.reserve_delivery(
        asof=asof, records=records, subject="Daily Pitch", html="<p>x</p>",
        recipients=["test@example.com"], path=path, use_r2=False)
    assert should_send
    if status == "sent":
        pitch_delivery.complete_delivery(
            receipt, path, use_r2=False, sent=True)
    elif status == "ambiguous":
        pitch_delivery.complete_delivery(
            receipt, path, use_r2=False, sent=False)
    if digest is not None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["verdict_digest"] = digest
        path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def run_check(journal_path: Path, asof: str = ASOF, *, receipt: bool = True,
              status: str = "sent", digest: str | None = None):
    receipt_path = (write_receipt(journal_path, asof, status, digest)
                    if receipt else journal_path.with_name("missing.json"))
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--asof", asof,
         "--journal", str(journal_path),
         "--delivery-receipt", str(receipt_path)],
        capture_output=True, text=True)


def write_journal(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records),
                    encoding="utf-8")
    return path


def idea(rank: int, date: str = ASOF) -> dict:
    return {"kind": "idea", "date": date, "rank": rank,
            "idea_id": f"{date}-{rank}"}


def test_three_ideas_is_delivery(tmp_path):
    journal = write_journal(tmp_path / "j.jsonl", [idea(i) for i in (1, 2, 3)])
    result = run_check(journal)
    assert result.returncode == 0
    assert "OK: 3 ideas" in result.stdout


def test_stand_down_is_delivery(tmp_path):
    journal = write_journal(tmp_path / "j.jsonl", [
        {"kind": "stand_down", "date": ASOF, "candidates_considered": 24},
        {"kind": "killed", "date": ASOF, "title": "k", "reason": "r"},
    ])
    result = run_check(journal)
    assert result.returncode == 0
    assert "stand-down journaled" in result.stdout


def test_empty_journal_is_not_delivery(tmp_path):
    journal = write_journal(tmp_path / "j.jsonl", [])
    result = run_check(journal)
    assert result.returncode == 1
    assert "did not deliver" in result.stdout


def test_journal_without_a_sent_receipt_is_not_delivery(tmp_path):
    journal = write_journal(tmp_path / "j.jsonl", [idea(1)])
    result = run_check(journal, receipt=False)
    assert result.returncode == 1
    assert "no delivery receipt" in result.stdout


@pytest.mark.parametrize("status", ["sending", "ambiguous"])
def test_uncertain_receipt_is_not_delivery(tmp_path, status):
    journal = write_journal(tmp_path / "j.jsonl", [idea(1)])
    result = run_check(journal, status=status)
    assert result.returncode == 1
    assert "not sent" in result.stdout


def test_receipt_digest_must_match_the_journal(tmp_path):
    journal = write_journal(tmp_path / "j.jsonl", [idea(1)])
    result = run_check(journal, digest="0" * 64)
    assert result.returncode == 1
    assert "digests differ" in result.stdout


@pytest.mark.parametrize("n", [1, 2])
def test_a_short_slate_is_delivery(tmp_path, n):
    """Since 2026-08-10 a morning with one survivor ships it. The grammar makes
    that expensive; this check only asks whether anything was delivered."""
    journal = write_journal(tmp_path / f"j{n}.jsonl",
                            [idea(i) for i in range(1, n + 1)])
    result = run_check(journal)
    assert result.returncode == 0
    assert "short slate" in result.stdout


def test_a_partial_run_cannot_hide_behind_a_stand_down(tmp_path):
    """Ideas AND a stand-down means the run published twice or crashed
    mid-way. That is not a verdict, and it must stay loud."""
    journal = write_journal(tmp_path / "j.jsonl", [
        idea(1),
        {"kind": "stand_down", "date": ASOF, "candidates_considered": 24},
    ])
    result = run_check(journal)
    assert result.returncode == 1


def test_yesterdays_delivery_does_not_count_for_today(tmp_path):
    journal = write_journal(tmp_path / "j.jsonl",
                            [idea(i, "2026-08-06") for i in (1, 2, 3)])
    result = run_check(journal)
    assert result.returncode == 1


def test_yesterdays_stand_down_does_not_count_for_today(tmp_path):
    journal = write_journal(tmp_path / "j.jsonl", [
        {"kind": "stand_down", "date": "2026-08-06"}])
    result = run_check(journal)
    assert result.returncode == 1


def directed(rank: int, date: str = ASOF) -> dict:
    return {**idea(rank, date), "directed_by": "mckinley"}


def test_stand_down_amended_by_a_directed_idea_is_delivery(tmp_path):
    """2026-08-07's real sequence: the sweep stood down, McKinley overruled a
    kill, and that one idea shipped."""
    journal = write_journal(tmp_path / "j.jsonl", [
        {"kind": "stand_down", "date": ASOF, "candidates_considered": 33},
        {"kind": "killed", "date": ASOF, "title": "k", "reason": "r"},
        directed(1),
    ])
    result = run_check(journal)
    assert result.returncode == 0
    assert "amended by 1 directed idea" in result.stdout


def test_undirected_ideas_beside_a_stand_down_still_fail(tmp_path):
    journal = write_journal(tmp_path / "j.jsonl", [
        {"kind": "stand_down", "date": ASOF},
        idea(1),
    ])
    assert run_check(journal).returncode == 1


def test_a_mixed_batch_beside_a_stand_down_fails(tmp_path):
    """One directed and one not is a half-published run, not an amendment."""
    journal = write_journal(tmp_path / "j.jsonl", [
        {"kind": "stand_down", "date": ASOF},
        directed(1), idea(2),
    ])
    assert run_check(journal).returncode == 1
