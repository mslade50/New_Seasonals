# Execution report timestamp repair — 14 September 2026

The 16:30 ET execution report failed on September 8, 9, 10, 11, and 14 with
`Primary book is stale or has an invalid timestamp`. Failure-notice delivery
succeeded. The supervisor then suppressed automatic recovery because the report
process had crossed its email side-effect boundary.

The agent sends `time.time()` seconds. The broker preserves `msg.at`, using
`Date.now()` milliseconds only as a fallback. The report unconditionally divided
both formats by 1,000, interpreting current agent snapshots as January 1970.
Before repair, a live snapshot approximately six seconds old failed validation;
normalizing that same timestamp in memory passed.

The report now normalizes milliseconds only when the epoch value is at least
100 billion. It retains the 300-second age limit, 30-second future-clock tolerance,
finite-value check, and Primary-account completeness requirements. The incoming
snapshot is not mutated. No broker producer or order-execution code changes.

Verification: the new tests produced five failures against the old code, then
passed with the fix. The exact runtime release passed 70 targeted report,
execution-ops, strict-mode, and workflow-contract tests, including seconds and
milliseconds, fractional seconds, exact freshness boundaries, invalid timestamps,
and incomplete Primary snapshots.

The installed v9 runtime advanced from `c5d0f59f1a94ca66b741ffbfc181b14527459a52`
to `b5b8fa824025120408cec8953144b563b2c9e964`, tagged
`automation-runtime-2026-09-14.1`. Its only changed files are the report, its new
freshness tests, the fallback version pin, and the pin's existing contract test.
Promotion held the supervisor lock, preserved the previous marker, and passed
the Task Scheduler runner's `-ValidateOnly` guard and workspace hygiene check.
The source workflow in this change selects that same immutable release.

At 20:58:13 UTC on September 14, the installed runtime fetched a live snapshot
21.451 seconds old, passed validation, and rendered the full report with exit 0.
Verification explicitly disabled email; existing failure receipts were retained.
The next ordinary scheduled delivery remains the end-to-end email confirmation.

Local evidence is retained under
`artifacts/automation/runtime_promotions/20260914T205738Z-execution-report.json`
and the fix worktree's `artifacts/live-report-proof/verification.json`.
Rollback should be a reviewed reverting commit and new immutable release;
the previous runtime marker and commit are retained as evidence.
