"""Generate an inert Task Scheduler definition for the expected-exit publisher.

This command only writes a new XML artifact. Registering/enabling the task is a
separate deployment step. The task repeats every minute, ignores overlaps, and
never enables email, scans, inventory refresh, or the execution command agent.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

NS = "http://schemas.microsoft.com/windows/2004/02/mit/task"
ET.register_namespace("", NS)


def task_xml(*, runtime, config, python, exec_env, sha, user, start):
    if not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
        raise ValueError("runtime commit must be an exact SHA")
    for value in (runtime, config, python, exec_env):
        if not re.match(r"^[A-Za-z]:[/\\]", value) or any(c in value for c in '\r\n"'):
            raise ValueError("task paths must be absolute Windows paths")
    if not user or any(c in user for c in "\r\n"):
        raise ValueError("task owner is required")
    parsed = dt.datetime.fromisoformat(start)
    if parsed.tzinfo is None:
        raise ValueError("task start time requires a timezone")
    root = ET.Element(f"{{{NS}}}Task", {"version": "1.4"})
    def add(parent, name, value=None, **attrs):
        child = ET.SubElement(parent, f"{{{NS}}}{name}", attrs)
        child.text = value
        return child
    info = add(root, "RegistrationInfo")
    add(info, "Description", "Publish observed Primary expected-exit status; no orders or email")
    trigger = add(add(root, "Triggers"), "TimeTrigger")
    repetition = add(trigger, "Repetition")
    add(repetition, "Interval", "PT1M")
    add(repetition, "StopAtDurationEnd", "false")
    add(trigger, "StartBoundary", parsed.isoformat())
    add(trigger, "Enabled", "true")
    principal = add(add(root, "Principals"), "Principal", id="Author")
    add(principal, "UserId", user)
    add(principal, "LogonType", "InteractiveToken")
    add(principal, "RunLevel", "LeastPrivilege")
    settings = add(root, "Settings")
    for key, value in (("MultipleInstancesPolicy", "IgnoreNew"), ("DisallowStartIfOnBatteries", "false"),
                       ("StopIfGoingOnBatteries", "false"), ("StartWhenAvailable", "true"),
                       ("RunOnlyIfNetworkAvailable", "true"), ("Enabled", "false"),
                       ("Hidden", "true"), ("ExecutionTimeLimit", "PT3M")):
        add(settings, key, value)
    action = add(add(root, "Actions", Context="Author"), "Exec")
    add(action, "Command", r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe")
    args = ["-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-ExecutionPolicy", "Bypass",
            "-File", str(Path(runtime) / "scripts/run_expected_exit_monitor.ps1"),
            "-RuntimeRoot", runtime, "-ConfigRoot", config, "-Python", python,
            "-ExecEnv", exec_env, "-PinnedSha", sha]
    # PowerShell -File arguments are passed as argv, with Windows quoting for
    # spaces. No command text or credential values are embedded in the task.
    add(action, "Arguments", subprocess.list2cmdline(args))
    add(action, "WorkingDirectory", runtime)
    return ET.tostring(root, encoding="utf-16", xml_declaration=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "config", "python", "exec-env", "sha", "user", "start", "output"):
        parser.add_argument("--" + name, required=True)
    args = vars(parser.parse_args(argv))
    output = Path(args.pop("output"))
    body = task_xml(**args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(body)
    print("Disabled expected-exit task XML created; registration and enablement are separate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
