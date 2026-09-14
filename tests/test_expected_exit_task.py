import xml.etree.ElementTree as ET

import pytest

from scripts.expected_exit_task import NS, task_xml


def arguments(**changes):
    return dict(runtime="C:/Fixture/runtime", config="C:/Fixture/config", python="C:/Python/python.exe",
                exec_env="C:/Fixture/broker.env", sha="a" * 40, user="FixtureUser",
                start="2026-09-14T18:00:00-04:00", **changes)


def test_task_is_disabled_and_repeats_without_overlapping_or_broker_commands():
    root = ET.fromstring(task_xml(**arguments()))
    text = lambda path: root.find(path, {"t": NS}).text
    assert text("t:Triggers/t:TimeTrigger/t:Repetition/t:Interval") == "PT1M"
    assert text("t:Settings/t:Enabled") == "false"
    assert text("t:Settings/t:MultipleInstancesPolicy") == "IgnoreNew"
    assert text("t:Principals/t:Principal/t:RunLevel") == "LeastPrivilege"
    argv = text("t:Actions/t:Exec/t:Arguments")
    assert "-WindowStyle Hidden" in argv and "run_expected_exit_monitor.ps1" in argv
    assert "exec_agent.py" not in argv and "--send" not in argv


def test_task_escapes_xml_and_quotes_paths_with_spaces():
    values = arguments()
    values["runtime"] = "C:/Fixture & Test/runtime"
    root = ET.fromstring(task_xml(**values))
    argv = root.find("t:Actions/t:Exec/t:Arguments", {"t": NS}).text
    assert '"C:/Fixture & Test/runtime' in argv
    assert root.find("t:Actions/t:Exec/t:WorkingDirectory", {"t": NS}).text == values["runtime"]


@pytest.mark.parametrize("field,value", [("runtime", "relative"), ("sha", "main"),
    ("exec_env", 'C:/injected"argument'), ("start", "2026-09-14T18:00:00")])
def test_task_rejects_ambiguous_configuration(field, value):
    values = arguments()
    values[field] = value
    with pytest.raises(ValueError):
        task_xml(**values)
