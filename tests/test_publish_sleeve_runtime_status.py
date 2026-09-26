import importlib.util
import json
from pathlib import Path
import sqlite3

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("publish_sleeve_runtime_status", ROOT / "scripts/publish_sleeve_runtime_status.py")
pub = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pub)


def _journal(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _sqlite(path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    db.executemany("INSERT INTO meta VALUES (?,?)", [(k, json.dumps(v)) for k, v in meta.items()])
    db.commit()
    db.close()


REF = "SPY|BUY|Legend_EMA|2026-09-24"
JOURNAL = [
    {"kind": "decision", "date": "2026-09-24", "symbol": "SPY", "side": "BUY", "strategy": "Legend_EMA", "reason": "LONG"},
    {"kind": "entry", "date": "2026-09-24", "symbol": "SPY", "side": "BUY", "qty": 1, "status": "SENT", "order_ref": REF, "strategy": "Legend_EMA"},
    {"kind": "target_filled", "date": "2026-09-24", "symbol": "SPY", "order_ref": REF, "strategy": "Legend_EMA"},
]
RESULT = {"date": "2026-09-24", "generated": "2026-09-24T10:32:00", "error": "",
          "symbols": [{"Symbol": "SPY", "Side": "BUY", "Qty": 1, "Status": "SENT", "Note": ""},
                      {"Symbol": "QQQ", "Side": "-", "Qty": 0, "Status": "NO_SETUP", "Note": "daily body ratio 0.645 < 0.75"}]}


def test_legend_reads_last_session(tmp_path: Path) -> None:
    _journal(tmp_path / "legend_ema_journal.jsonl", JOURNAL)
    (tmp_path / "legend_ema_last_result.json").write_text(json.dumps(RESULT), encoding="utf-8")
    out = pub.read_legend(tmp_path)
    assert out["available"] and out["session_date"] == "2026-09-24"
    assert out["trades"] == [{"symbol": "SPY", "side": "BUY", "qty": 1, "status": "SENT", "outcome": "target hit"}]
    assert out["skips"] == [{"symbol": "QQQ", "status": "NO_SETUP", "note": "daily body ratio 0.645 < 0.75"}]
    assert out["notes"] == []


def test_legend_ignores_test_entries_and_nan(tmp_path: Path) -> None:
    rows = JOURNAL[:2] + [{"kind": "forced_entry", "date": "2026-09-25", "symbol": "SPY", "qty": 1, "strategy": "Legend_EMA_TEST"}]
    (tmp_path / "legend_ema_journal.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + '\n{"kind": "decision", "target": NaN, "date": "2026-09-24", "strategy": "Legend_EMA", "symbol": "QQQ", "side": null, "reason": "short disabled"}\nnot json\n',
        encoding="utf-8")
    out = pub.read_legend(tmp_path)
    assert out["session_date"] == "2026-09-24"
    assert out["trades"][0]["outcome"] == "no exit record yet"
    assert out["skips"] == [{"symbol": "QQQ", "status": "decision", "note": "short disabled"}]
    assert "last-result file missing" in out["notes"]
    json.dumps(out, allow_nan=False)


def test_legend_missing_everything_is_unavailable(tmp_path: Path) -> None:
    out = pub.read_legend(tmp_path)
    assert out == {"available": False, "reason": "last-result file missing; journal missing"}


def _meta(mode: str, day: str) -> dict:
    return {"pid": 42, "mode": mode, "session": day, "phase": "RUNNING_LIVE" if mode == "live" else "RUNNING_SHADOW",
            "heartbeat": {"at": f"{day}T14:00:00+00:00", "events": 7, "connected": True, "healthy": True},
            "live_ack": "U1234567 acknowledged", "port": 7496,
            "settings": {"capital_base": 750000.0, "risk_bps": {"NQ": 15.0}, "execution": {"NQ": "MNQ"}, "port": 7496},
            "last_error": "order rejected for account U1234567",
            "inputs": {"hash": "x", "prior_range": {"NQ": {"prior_range_status": "OK", "atr20": 1.0, "ratio": 1.4,
                                                            "skip_prior_range": True, "half_prior_range": False,
                                                            "prior_range_reason": "ratio >= 1.25"}}}}


def test_breakout_picks_newest_shadow_and_live(tmp_path: Path) -> None:
    _sqlite(tmp_path / "2026-09-25-shadow/runtime.sqlite", _meta("shadow", "2026-09-25"))
    _sqlite(tmp_path / "2026-09-28-shadow/runtime.sqlite", _meta("shadow", "2026-09-28"))
    _sqlite(tmp_path / "2026-09-28-live/runtime.sqlite", _meta("live", "2026-09-28"))
    _sqlite(tmp_path / "2026-09-28-live-2/runtime.sqlite", _meta("live", "2026-09-28"))
    (tmp_path / "config-20260928-live.json").write_text("{}", encoding="utf-8")
    out = pub.read_breakout(tmp_path)
    assert out["shadow"]["run_dir"] == "2026-09-28-shadow"
    live = out["live"]
    assert live["run_dir"] == "2026-09-28-live-2"
    assert live["phase"] == "RUNNING_LIVE" and live["heartbeat_at"] == "2026-09-28T14:00:00+00:00" and live["events"] == 7
    assert live["prior_range"]["NQ"]["skip"] is True
    assert live["settings"] == {"risk_bps": {"NQ": 15.0}, "execution": {"NQ": "MNQ"}}
    text = json.dumps(out)
    assert "U1234567" not in text and "live_ack" not in text and "7496" not in text
    assert live["last_error"] == "order rejected for account [account]"


def test_breakout_missing_and_unreadable(tmp_path: Path) -> None:
    assert pub.read_breakout(tmp_path / "nope") == {"available": False, "reason": "runs directory missing"}
    (tmp_path / "2026-09-28-live").mkdir()
    bad = tmp_path / "2026-09-28-shadow/runtime.sqlite"
    bad.parent.mkdir()
    bad.write_bytes(b"not a sqlite database at all, just junk bytes" * 10)
    out = pub.read_breakout(tmp_path)
    assert out["live"] == {"available": False, "reason": "runtime.sqlite missing", "run_dir": "2026-09-28-live"}
    assert out["shadow"]["available"] is False and out["shadow"]["reason"].startswith("sqlite unreadable")


def test_breakout_locked_sqlite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _sqlite(tmp_path / "2026-09-28-live/runtime.sqlite", _meta("live", "2026-09-28"))

    def locked(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(pub.sqlite3, "connect", locked)
    out = pub.read_breakout(tmp_path)
    assert out["live"]["reason"] == "sqlite locked"
    assert out["shadow"] is None


def test_readonly_open_does_not_create_database(tmp_path: Path) -> None:
    (tmp_path / "2026-09-28-live").mkdir()
    pub.read_breakout(tmp_path)
    assert not (tmp_path / "2026-09-28-live/runtime.sqlite").exists()


def test_task_query_names_only_live_tasks() -> None:
    assert "'IBKR Legend EMA'" in pub.TASK_QUERY and "'IBKR Legend EMA Verify'" in pub.TASK_QUERY
    assert "LegendETF" not in pub.TASK_QUERY


def test_clean_text_scrubs_account_ids_next_to_underscores() -> None:
    assert pub.clean_text("acct_U16584234 / DU1234567: rejected") == "acct_[account] / [account]: rejected"
    assert pub.clean_text("order 12345678 perm U12") == "order 12345678 perm U12"
