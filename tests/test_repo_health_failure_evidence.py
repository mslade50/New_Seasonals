"""A health failure must preserve its evidence and report the actual cause."""
import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import repo_health_check as health


def test_changed_history_preserves_checkpoint_and_fails_again(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    frame = pd.DataFrame({"63d": [10.0, 20.0]},
                         index=pd.to_datetime(["2026-09-01", "2026-09-02"]))
    checkpoint = {
        "frag_frozen_through": "2026-09-02",
        "frag_frozen_columns": ["63d"],
        "frag_frozen_sha": hashlib.sha256(frame.round(6).to_csv().encode()).hexdigest(),
    }
    state = tmp_path / "checkpoint.json"
    state.write_text(json.dumps(checkpoint), encoding="utf-8")
    before = state.read_bytes()
    frame.iloc[0, 0] = 99.0
    frame.to_parquet(tmp_path / "data/rd2_fragility.parquet")
    monkeypatch.setattr(health, "ROOT", tmp_path)
    monkeypatch.setattr(health, "STATE_PATH", state)
    monkeypatch.setattr(health, "RESULTS", [])

    for _ in range(2):
        health.RESULTS.clear()
        health.check_fragility_pit()
        assert health.RESULTS[0][0] == "FAIL"
        assert state.read_bytes() == before


def test_delivery_failure_reports_diagnostic_not_download_chatter(monkeypatch):
    monkeypatch.setattr(health.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=1,
        stdout="[cache_io] downloaded journal\nOK: journal matches\nFAILED: no delivery receipt exists\n",
        stderr=""))
    monkeypatch.setattr(health, "RESULTS", [])
    health.check_delivery()
    assert all("FAILED: no delivery receipt exists" in item[2] for item in health.RESULTS)


def test_unchanged_history_allows_appended_rows_and_new_columns(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    frame = pd.DataFrame({"63d": [10.0, 20.0]},
                         index=pd.to_datetime(["2026-09-01", "2026-09-02"]))
    prior = frame.iloc[:1]
    state = tmp_path / "checkpoint.json"
    state.write_text(json.dumps({
        "frag_frozen_through": "2026-09-01",
        "frag_frozen_columns": ["63d"],
        "frag_frozen_sha": hashlib.sha256(prior.round(6).to_csv().encode()).hexdigest(),
    }), encoding="utf-8")
    frame["main_score"] = [30.0, 40.0]
    frame.to_parquet(tmp_path / "data/rd2_fragility.parquet")
    monkeypatch.setattr(health, "ROOT", tmp_path)
    monkeypatch.setattr(health, "STATE_PATH", state)
    monkeypatch.setattr(health, "RESULTS", [])
    health.check_fragility_pit()
    assert health.RESULTS[0][0] == "OK"
    updated = json.loads(state.read_text(encoding="utf-8"))
    assert updated["frag_frozen_columns"] == ["63d", "main_score"]
    assert updated["frag_frozen_through"] == "2026-09-02"


def test_delivery_exception_reports_final_error(monkeypatch):
    monkeypatch.setattr(health.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=1, stdout="[cache_io] downloaded journal\n",
        stderr="Traceback (most recent call last):\nValueError: invalid receipt\n"))
    monkeypatch.setattr(health, "RESULTS", [])
    health.check_delivery()
    assert all("ValueError: invalid receipt" in item[2] for item in health.RESULTS)


@pytest.mark.parametrize("old_label,new_label", [("Date", None), (None, "Date")])
@pytest.mark.parametrize("changed", [False, True])
def test_legacy_index_label_change_only_is_not_a_rewrite(
    tmp_path, monkeypatch, old_label, new_label, changed
):
    (tmp_path / "data").mkdir()
    frame = pd.DataFrame({"63d": [10.0, 20.0]},
                         index=pd.to_datetime(["2026-09-01", "2026-09-02"]))
    frame.index.name = old_label
    state = tmp_path / "checkpoint.json"
    state.write_text(json.dumps({
        "frag_frozen_through": "2026-09-02", "frag_frozen_columns": ["63d"],
        "frag_frozen_sha": hashlib.sha256(frame.round(6).to_csv().encode()).hexdigest(),
    }), encoding="utf-8")
    before = state.read_bytes()
    frame.index.name = new_label
    if changed:
        frame.iloc[0, 0] += 1
    frame.to_parquet(tmp_path / "data/rd2_fragility.parquet")
    monkeypatch.setattr(health, "ROOT", tmp_path)
    monkeypatch.setattr(health, "STATE_PATH", state)
    monkeypatch.setattr(health, "RESULTS", [])
    health.check_fragility_pit()
    assert health.RESULTS[0][0] == ("FAIL" if changed else "OK")
    if changed:
        assert state.read_bytes() == before
    else:
        assert json.loads(state.read_text())["frag_frozen_hash_version"] == 2
