# 8 September morning scheduler failure

## Evidence before recovery

The enabled v8 runtime was pinned to `0c912f5844dfa291ac9a720baa05023efa2fceca`
and `automation-runtime-2026-09-03.1`. Source merged to main after that date was
not installed. No 05:45 local retry existed in Task Scheduler.

At 04:10 ET premarket started. CBOE, prices, risk, and the event sleeve completed.
At 04:16 the scanner aborted at its freshness gate before staging or email.
Log: outgoing runtime `artifacts/automation/logs/2026-09-08/premarket-b8fcbc5d.log`.
Its one-row trim left weekend/holiday bars from crypto and FX after the expected
4 September settled equity session. SPY and QQQ had valid 4 September data.
Main already removed every later bar; production v8 did not.

The 06:06 ET GitHub controller (run 34213644006) executed successfully as a
controller but refused recovery: the supervisor had conservatively marked the
entire scanner process as potentially side-effecting. Both site jobs were blocked.
This guard prevented duplicate order staging; it must not be weakened to retry
unknown writes. Operator review of the precise pre-write failure is required.

The separate 00:30 research task updated its run marker but neither launched the
collector/agent nor wrote a fresh decision. It returned zero despite failure.
The shared `last_run.log` remained from 7 September and opening it for writing
under the task owner's account on 8 September raised PermissionError. Batch
redirection prevented child launch. The prior manual run did not validate the
unattended task, and the earlier operational claim was premature.

## Repair and rollout contract

- Deploy the already-merged scanner correction in a new clean pinned v9 runtime.
- Advance the GitHub controller and local runtime to one immutable release tag,
  `automation-runtime-2026-09-08.1`; register the full eight-task set, including
  05:45 retry and stable cross-generation logging/locking.
- Replace the research batch orchestration with Python subprocess calls using
  argument arrays, unique logs and markers, per-run receipts, a process lock,
  explicit timeouts and nonzero exits on any launch, collection, agent or
  completion failure. Register the task with an absolute interpreter and S4U.
- Apply the existing NYSE calendar gate to AM/PM scans and execution reports
  before local execution or cloud dispatch. The 7 September PM scan expected a
  holiday close and the execution report failed against a disconnected broker;
  neither trading-session job should have run on Labor Day. The controller
  installs pandas for these shared calendar gates (including discretionary).
- Preserve v8 and its disabled tasks as rollback evidence. No cleanup or deletion.
- Recover only the scan after its receipt is reconciled and financial staging
  approval is confirmed. Never repeat the already-successful event job.
- Build production sites only in GitHub Actions from authoritative R2 inputs.

A new source commit, a green mocked test, and a registered task are not completion.
Record the installed SHA and task definitions, real run receipts, delivery/NO_EMAIL
decision, cloud build/freshness result, and production deployment SHA at handoff.
