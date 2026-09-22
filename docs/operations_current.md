# Current operations entry point

Latest priority-3 inventory work: [Primary OLV cutover](olv_inventory_cutover_2026-09-09.md).
Its prepared source and reconciled opening candidate are not yet active.

Start with the [6 September audit repair release](audit_fix_release_2026-09-06.md) for current behavior, verification, unresolved migrations and rollout boundaries. Its branch is prepared source, not a declaration that production uses it.

Operational truth comes from exact runtime/source identities, dated producer receipts, strategy allocations and broker execution evidence. A scheduler flag, model target or website refresh is not a fill confirmation.

The long strategy descriptions in AGENTS.md and older notes are historical context where they conflict with current executable configuration. In particular, the old Layer 4C automated hedge recommendation is retired; current dial/hedge status must be read from dated implementation and runtime evidence. Read `strategy_config.py` and aligned execution contracts for current risk/ladder/path settings, not copied nominal examples in old notes.

Use the private-site build skill for every production site operation. Production inputs come from R2 through the cloud workflow. Retain historical scratch evidence; put machine-generated validation and candidate outputs under ignored artifacts.

### EP morning research runtime (restored 2026-09-22)

The active Codex heartbeat `ep-after-hours-shadow-queue` (EP Night and Morning
Shadow Process) uses the permanent worktree
`C:\Users\McKinley Slade\dev\New_Seasonals-worktrees\ep-production-runtime`,
branch `codex/ep-production-runtime`, pinned to
`9de30b6839ccbfcd03770c6a99bc8969897f3869`. This restores the source used by the
successful September 17 morning run. The worktree is locked with a reason naming
the active heartbeat; preserve both the worktree and branch during cleanup.
The heartbeat remains attached to its existing Codex task and runs weekdays at
08:20 and 19:20 America/New_York, with phase commands explicitly run in this runtime.

The runtime's ignored `artifacts/` is a directory junction to
`C:\Users\McKinley Slade\dev\New_Seasonals\artifacts\ep-production-runtime`.
This keeps generated files inside the task's writable workspace without changing
the pinned source or filesystem permissions. CLI defaults resolve through the
junction; browser downloads and agent notes use the resolved absolute path.
Preserve that output directory and junction as active runtime dependencies.

Restoration checks: exact-commit integrity guard passed, runtime Git status was
empty, all 352 EP tests passed, all seven CLI entry points loaded, email settings
resolved without sending, and artifact creation/readback passed under the task
sandbox. This verifies startup and local processing, not a completed live
premarket capture or email delivery. Evidence:
`artifacts/ep-runtime-repair-20260922/`.

The old `artifacts/worktrees/ep-yfinance-prod` location was removed during the
September 18 cleanup while the heartbeat still referenced it. Check Codex
automations as well as Windows scheduled tasks before retiring any runtime;
see [workspace_hygiene.md](workspace_hygiene.md).


### Preliminary afternoon breadth (2026-09-21)

The weekday 17:10 ET postclose pipeline uses `collect_market_breadth.py --source overview`
before scoring risk. This public Dow Jones overview returned the same NYSE and Nasdaq
counts as MarketWatch on September 21 (26/154 and 147/189), with an explicit
`4:15 PM EDT 9/21/26` timestamp. The detailed diary was still on the prior session
during the earlier 17:13–17:33 collection window. Arrival by 17:10 has not yet been
measured over multiple sessions; the collector retains its bounded 20-minute retry.

Overview observations are preliminary (`dow_jones_overview`). Import requires a real
trading date, a publication time of 16:15 ET or later, and completed-session/count checks.
A stale date is never relabeled. The 04:10 ET collection continues to read the detailed
WSJ Latest Close diary; that source takes priority regardless of later overview captures.
Both sources and all distinct revisions remain in SQLite. Nasdaq counts can differ by
source/universe; the live NYSE signal uses NYSE only. Historical workbook rows remain frozen.
The site continues to build in GitHub Actions using canonical R2 inputs.


The risk producer also consumes the canonical adjusted master-price snapshot using
the site's loader. Verification on September 21 caught an independent Yahoo refresh
ending SPY on September 18 while other risk rows extended to September 21, which had
silently written a missing main score. The risk input pull now requires master prices;
main-score assembly refuses a risk row beyond the available SPY history. This preserves
the signal formulas while aligning producer and dashboard price vintages. Runtime
release tag: `automation-runtime-2026-09-21.breadth-canonical`.
