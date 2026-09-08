import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts.run_strategy_research import run


@pytest.mark.parametrize("failure", [None, "source_collection", "research_agent", "completion_check"])
def test_launcher_propagates_failure_and_requires_all_phases(tmp_path, failure):
    calls = []
    phases = ["source_collection", "research_agent", "completion_check"]

    def execute(command, **kwargs):
        assert kwargs["stdin"] == subprocess.DEVNULL
        if "--send" in command:
            calls.append("alert")
            return SimpleNamespace(returncode=0)
        phase = phases[len(calls)]
        calls.append(phase)
        return SimpleNamespace(returncode=7 if phase == failure else 0)

    assert run(tmp_path, execute=execute) == (7 if failure else 0)
    receipt = json.loads((tmp_path / "artifacts/strategy_research_agent/last_run.json").read_text())
    assert receipt["status"] == ("failure" if failure else "success")
    if failure:
        assert calls == phases[:phases.index(failure) + 1] + ["alert"]
    else:
        assert calls == phases


def test_launcher_never_opens_shared_legacy_log_and_keeps_run_history(tmp_path):
    artifacts = tmp_path / "artifacts/strategy_research_agent"
    artifacts.mkdir(parents=True)
    # A directory is an unopenable log target on every OS (Windows uses a
    # lingering exclusive handle in the actual incident).
    (artifacts / "last_run.log").mkdir()
    execute = lambda *a, **k: SimpleNamespace(returncode=0)
    assert run(tmp_path, execute=execute) == 0
    first = json.loads((artifacts / "last_run.json").read_text())
    assert run(tmp_path, execute=execute) == 0
    second = json.loads((artifacts / "last_run.json").read_text())
    assert first["log"] != second["log"]
    assert (artifacts / (first["run_id"] + ".json")).is_file()


@pytest.mark.parametrize("error", [FileNotFoundError("missing interpreter"), subprocess.TimeoutExpired("collector", 1200)])
def test_launch_error_and_alert_error_remain_failed(tmp_path, error):
    def execute(*a, **k):
        raise error
    assert run(tmp_path, execute=execute) == 1
    receipt = json.loads((tmp_path / "artifacts/strategy_research_agent/last_run.json").read_text())
    assert receipt["status"] == "failure"
    assert receipt["alert_exit_code"] == 1
