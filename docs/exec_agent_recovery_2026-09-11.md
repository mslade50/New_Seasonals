# Execution-agent recovery — 11 September 2026

The local execution bridge was restored at 06:19:17 Eastern. At 06:21:40,
the relay reported online, one socket, and a heartbeat 1.8 seconds old.
Exactly one `exec_agent.py` process remained active after the next recovery
trigger. The connection timestamp was unchanged.

## Failure evidence

- The agent launched at 05:00:03; its final heartbeat before the outage was
  05:49:57. The Python process was absent and `ExecAgent` was Ready, with task
  result zero, when checked at approximately 06:10.
- The original PowerShell wrapper did not propagate Python's exit status.
  A broker-free reproduction using that exact wrapper, with only its runtime
  and Python paths redirected into an artifact fixture, called `os._exit(23)`.
  Task-facing exit status was incorrectly zero.
- The original task retried only failures (one minute, three retries), with
  no repeat trigger after a successful result. This explains why an unexpected
  Python exit could leave the bridge offline until the next day or login.
- The initiating termination remains unproven: no final exception was logged,
  Task Scheduler operational history was disabled, and no relevant Windows
  crash or reboot event was found in the 05:40–06:00 interval. This repair does
  not claim to identify who or what terminated the process.

## Installed repair

`scripts/run_exec_agent.ps1` is the maintained launcher source. Its installed
copy is `C:/Users/McKinley Slade/OneDrive/trading_ibkr/run_exec_agent.ps1`.
It logs start/exit timestamps, wrapper PID and native exit code, returns failure
for any exit inside operating hours (including native zero), and skips launch
outside the existing configurable window. Native stderr is logged without
interrupting the wait for Python.

The existing `ExecAgent` task keeps its principal, interactive-session requirement,
network condition, 17-hour limit, single-instance `IgnoreNew` policy, and
one-minute failure retries. Its daily 05:00 trigger now repeats every five minutes
for 16 hours. Repetition does not terminate an active instance at its boundary;
the agent retains responsibility for its normal 21:00 shutdown. The logon trigger
is preserved. PowerShell launches hidden. The local `register_exec_agent.ps1`
and its maintained source under `scripts/` now preserve those settings on setup.

No executor, broker order logic, arming configuration or credentials were changed.
There was no local scheduled-option file, and all four local position-action
records were done before restart. No test or diagnostic submitted an order.

## Verification and rollback

Five broker-free native Windows PowerShell tests passed:
`python tests/test_exec_agent_launcher.py -v`. Coverage includes abrupt native
termination, unexpected zero exit, stderr handling, missing configuration, and
outside-window skip. The task XML was parsed by native Task Scheduler COM before
registration. The registration source was evaluated only through definition
construction and checked for the exact repeat window, hidden launch, non-terminating
repetition boundary and single-instance policy. Installed scripts matched the
reviewed source hashes. Live relay evidence confirms advancing heartbeats and
one process across the 06:20 trigger; live process failure was not deliberately
induced to test recovery.

Original launcher, registration script, task XML, candidate XML, the old-wrapper
regression fixture, and sanitized live verification are retained under
`C:/Users/McKinley Slade/dev/New_Seasonals/artifacts/exec-agent-recovery/`.
Rollback consists of restoring the two original script copies and registering
`ExecAgent.original.xml`; stopping/restarting the live execution service for a
rollback requires separate operational authorization. No backup was deleted.
