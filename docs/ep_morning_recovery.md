# EP morning continuity and recovery

An active morning run ends only with a confirmed morning email receipt, a confirmed
operational failure receipt, an explicitly reported uncertain delivery, or an
explicit user pause. Passing tests, source capture, a research queue and an answer
to an older conversation message are not completion. On September 23 the task
lost the morning objective after compaction and answered a September 16 request
about research safeguards. Its data capture succeeded but no email was attempted.

## Worker: retain the current objective

The existing 08:20 heartbeat is the worker. Read this document first on every
morning start, recovery or context reset. Continue today's research and delivery;
do not treat old user messages, code audits, test runs or troubleshooting plans as
a replacement task. A new user correction steers the active run unless the user
explicitly cancels it. Do not make code/Git changes during scheduled research.

Run in the pinned `ep-production-runtime`, after its exact-commit guard:

```powershell
python scripts/check_ep_morning_completion.py
```

This returns compact operational state and saved artifact pointers. It never sends
email. `DELIVERED` and `FAILURE_REPORTED` are terminal; `DELIVERY_UNCERTAIN` requires
reconciliation, with no automatic resend or definitive failure email. `NOT_DUE`
skips weekends, exchange holidays, other dates and the period before 08:20 ET.
`PAUSED_BY_USER` stays paused. `RESUME` means today's report is still outstanding.
`DEADLINE_MISSED` means no confirmed outcome exists at/after 09:30 ET; candidate
delivery is then prohibited and the existing sanitized failure route applies.

On first start (`NOT_STARTED` progress), record discovery:

```powershell
python scripts/check_ep_morning_completion.py --stage DISCOVERY
```

Add `--track-morning` to both `run_episodic_pivot_shadow.py` commands that prepare
the Google queue and build the final agent-reviewed report. Queue preparation
automatically saves the exact snapshot and queue paths with hashes; final building
saves the report manifest and review packet. These checkpoints do not certify news
research. Existing queue binding, full source review and sender validation remain
mandatory. Never replace a frozen queue just to hide unfinished names.

Save incremental review notes after each issuer, then checkpoint their location:

```powershell
python scripts/check_ep_morning_completion.py --stage RESEARCH --file "notes=<absolute-notes-json>"
```

The same command accepts `short_queue=<absolute-short-run>/queue.json` and
`short_notes=<absolute-short-notes-json>`. After a restart, inspect saved files and
resume unfinished research. Changed hashes are disclosed, never silently trusted:
revalidate frozen market/queue inputs; notes may contain newly saved progress but
still require full review validation. Missing/corrupt checkpoints do not mean the
email was sent. Inspect retained current-day artifacts without changing them.

Avoid context exhaustion: never dump complete `prices.json`, `universe.json`,
`discovery.json`, full listing-bearing short `queue.json`, or whole historical
run logs into the conversation. Use the compact checker and extract only counts,
current candidates and relevant source excerpts. Output files retain full evidence.

Before finishing or after any email attempt, run the checker again. A chat statement
cannot substitute for a `SENT` receipt. Persist the next action if interrupted.
For an explicit user stop only, record `--stage PAUSED_BY_USER`; resume only when
the user requests it, preserving the saved inputs and restarting the proper stage.

## Send once per morning session

Add the following flag to both final report sender commands and to the morning
failure sender command, alongside all their existing required flags:

```powershell
--completion-root "C:\Users\McKinley Slade\dev\New_Seasonals\artifacts\ep-production-runtime\episodic_pivot"
```

The sender takes one shared session lock across report and failure attempts,
rechecks receipts and the report's age/session/deadline inside that lock, then
uses the existing SMTP sender. This prevents two resumed runs or a report/failure
race from sending twice, even when their output directories differ. `--resend`
is forbidden in guarded runs. The existing sender writes durable `SENDING` before
SMTP DATA, then `SENT` after acceptance. `SENDING`, `AMBIGUOUS`, malformed or
conflicting receipts stop all automatic second emails. SMTP acceptance is the
observable boundary; inbox placement is not independently confirmed.

## Independent completion guard

A separate heartbeat in the maintenance task checks weekday mornings at
08:00/08:20/08:40 and 09:00/09:20/09:40 ET. The first two checks are quiet no-ops;
recovery begins at 08:40. It uses the same pinned source and this receipt checker,
not the worker's final text or test results. No new app task or Git checkout is
created. The guard never does investment research itself.

For `RESUME`, inspect the worker task's current status. Leave running/waiting tasks
alone. If idle, claim a recovery slot with `--claim-resume` and send a fresh
continuation only when `resume_claimed` is true. Claims are atomically limited to
one per 20-minute slot, and a failed dispatch is not silently marked complete.
The continuation states today's date, pinned runtime, checker command and current
artifact pointers, and tells the worker to finish the existing morning run.
If it reaches a now-active or already-finished worker, it must not start parallel
research or duplicate a delivered report.

At 09:40, if `DEADLINE_MISSED` still holds, the guard sends the existing operational
failure email through the guarded sender. Recheck receipts under the shared lock
before submission. No stale candidate report, extra short-only email, backdated
source observation or partial research is allowed. Delivery uncertainty, an
explicit pause or an unavailable checker must be surfaced for review rather than
treated as a missing receipt. Retain sanitized local error evidence.

This guard is quiet on unchanged/successful states. Report meaningful failures,
required decisions or failed recovery dispatches only. Both heartbeats depend on
the local Codex scheduler being available; this is not an external uptime service.
