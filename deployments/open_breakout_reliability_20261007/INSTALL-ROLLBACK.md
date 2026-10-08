Every installer command defaults to a read-only plan. This work did not execute
any live --apply command. Source publication to GitHub does not change the running
worker, broker client, Scheduler, request receipts, session journal or risk ledger.

Open Breakout source staging is separate from future-session activation:

    python installer.py preflight
    python installer.py stage
    python installer.py preflight --installed
    python installer.py activate --session YYYY-MM-DD
    python installer.py deactivate
    python installer.py rollback
    python installer.py recover

An operator must review the plan before adding --apply. The fixed 20-file
allowlist, 28 baseline hashes, package inventory, quiescent process gate and
outside-session window are enforced. Stage is disabled. Activation requires a
future complete exchange session and fresh exact-owner flat confirmation.
Default activation selects price pause only. Legend coordination needs a separate
--coordination choice; native automatic close remains false. Order reliability
needs --order-reliability, --qualification and --paper-evidence whose hashes,
source version and measured timing match. Offline fixtures cannot authorize live
reliability. Shadow retains its legacy simulator; it cannot qualify native paper.
No command starts or restarts a worker.

Execution reconnect has its own two-source-file installer and backup:

    python bridge_installer.py preflight
    python bridge_installer.py stage
    python bridge_installer.py rollback
    python bridge_installer.py recover

Future --apply requires the normal exited-agent window 21:05–04:55 New York,
ExecAgent Scheduler state Ready, and no other Python/pythonw process. The current
installer PID is excluded using process names and IDs only; no command lines are
inspected. Request/seen/receipt/schedule files and credentials are outside its
source allowlist. The agent backup and source transaction share an exclusive lock.
Recovery accepts only exact pinned stage/rollback transitions and rechecks all
paths/hashes after its gate. Symlinks and Windows reparse points are refused.

For an interruption before transaction.json is written, source mutation has not
started. The preparation lock remains fail closed. Preserve it and review the
unchanged source hashes and preparation files manually; automated recover refuses
a missing journal. Do not delete the lock or guess a preimage. An unknown hash,
unknown file, stale receipt, pending native order or active process blocks rollout.
Rollback restores source/config preimages only; it does not erase session state,
refill risk, repeat requests, change schedules or cancel orders.
