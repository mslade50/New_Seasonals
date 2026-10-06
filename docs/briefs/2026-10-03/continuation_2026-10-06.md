# IBKR repairs and runtime release continuation, October 6, 2026

Prepared and verified before the morning tasks. Installation and runtime cutover
are still pending. The original weekend deadline passed; the handoff forbids
runtime cutovers from 04:00 through 09:35 ET. The prepared installers require the
10:00 to 16:00 ET slot and recheck live identities and task states.

## Completed preparation

The staging filter now catches date parsing, comparison and logging exceptions,
prints one CRITICAL line and returns the input rows unchanged. The existing
Scan_Date cutoff still runs first. A timezone-aware Scan_Date can still fail in
that older cutoff; this repair does not change its behavior.

The PA runner reads OLV_Exits_Primary through the Primary tranche contract. It
requires the exact entry orderRef, conId and time-exit date, sells only its own
matched leg and retains the held-share clamp. Missing or unreadable tabs and
invalid rows now produce CRITICAL and exit 1 before broker access. At or after
09:25 ET it leaves the auction exit pending. If cancellation crosses the
cutoff, it re-arms the original time exit and returns a failure status.

The actual wrapper probe confirmed PRIMARY then PA. A PA exception returns 1
after Primary has run. The same behavior now has regression coverage.

The reviewed source and test diffs are retained in
[patches](patches/). Full candidate files remain under
`artifacts/ibkr_patches_2026-10-01/patches/`. The guarded installer, guarded
runtime release script, exact hashes and test receipts are under
`artifacts/ibkr-runtime-release-20261006/`. Those local artifacts are retained
because installation is still owed.

For manual patch application after a Windows checkout, use
`git apply --ignore-space-change` to tolerate checkout line endings. The
guarded installer copies the verified candidate files directly.

## Verification

- Focused broker guards: 98 passed.
- Current broker baseline: 923 passed, 22 failed.
- Final candidate full suite: 980 passed, the same 22 failures, no new failures.
- Original code with revised tests: 53 failed, 45 passed.
- Pre-review candidates with review regressions: 13 failed, 1 passed.
- Parity probe: 15 normal scenarios byte-identical; three stale-row hazards
  differ as intended.
- Read-only candidate smoke: SMOKE OK; both staging tabs empty and zero due PA
  exit rows at the check.
- Main source preflight using the runtime venv: 100 passed, 1 skipped. This is
  preliminary validation, not a test of the released runtime tree. An initial
  sandbox run had a Git ignore-file permission warning interpreted as a dirty
  fixture; the rerun with normal host access passed.
- Current premarket and postclose ValidateOnly: exit 0 for both, each printing
  `Validated pinned local-automation runtime at a7f49f00865f.`

All broker tests use fakes with real IB connection methods and socket
connections disabled. The Sheets smoke enables Google reads while explicitly
disabling IB.connect and IB.connectAsync. No broker connection, orders,
production pipeline or test email was executed.

## SHA-256 identities

| File | Live before, still installed | Verified candidate |
| --- | --- | --- |
| order_staging.py | `4e6c6ac558cb575c8271254e6198a9aff8d9a587841d4de63864fd5d901313ca` | `d8106578fd049f7a8e392fb0a5a6bc6b7ee8ea00b16a36457a861d49d155e200` |
| olv_exit_pa_legacy.py | `16b7aa3cb9ef6df367b21768253366c259fd6e0b2d3b31bdae324fa15124d427` | `70e5fe4e22ce4de8191a5ece9bdd7c2fe4b0615cbe838fe132bbbf488eaeaeef` |
| test_olv_exits.py | `95adbadfcbea3b127f7ea23a1deaa235064eadf880fd5baf5dffe6b828cef82e` | `8ec0d461b6d1102189d1099ef9277761ff4b7dec985e1398acd57e4a7ddc3b7a` |
| test_staging_prior_day_rows.py | absent | `5004ed884201ba385dfd28366af21f6b4255fb7eaf9ae06cc0dd2dbb65caafe3` |

## Runtime release scope

The installed pin remains `a7f49f00865fdaf6ef598845b6a1504a9899478b`, tag
`automation-runtime-2026-10-01.earnings-alpha-only`. No new pin or tag exists.
The planned tag is `automation-runtime-2026-10-06.alerts-calendar-cache`.

Release sources, in order: `5c63a3a4` (lock-safe cache), `9344b8b0` (health and
stale tests), `7cb1560a` (alerts and AM calendar repair), and `beb3b58f` (context
delivery diagnostics). Runtime health invokes check_context_delivered.py, so
beb3b58f qualifies. Exclude `5187653f` and `71e44658`: their consumers run from
main. Preserve all unrelated newer main commits.

The curated runtime lacks three paths modified by those commits:
`fundamental/financing_opportunity_report.py`,
`tests/test_execution_runtime_audit.py`, and
`docs/claude_ref/daily_posts_and_context.md`. They are explicitly excluded;
release must not add missing unrelated subsystems. All four selected commit
patches pass read-only apply checks with those exclusions. The prepared release
script permits only those exact missing-file conflicts and stops on any others.

## Remaining cutover

1. Recheck all live hashes, task states and today's completed 09:31 chain.
   Run `artifacts/ibkr-runtime-release-20261006/install_broker.py --execute`
   with host access. It backs up the three existing files using the actual
   installation date, installs four candidates, runs guards in the live folder
   and runs the read-only smoke. Failed verification restores its own files.
2. Run `artifacts/ibkr-runtime-release-20261006/release.py --execute` with the
   runtime interpreter and host access. It holds the supervisor lock, verifies
   the old pin and clean source, runs baseline and candidate tests in the real
   runtime, commits the scoped cherry-picks, pushes only the lightweight tag,
   updates the CRLF/no-BOM marker and delivers the fallback pin to remote main.
3. Verify the new pin, remote tag, tag-tree fallback ref and remote main.
   Both ValidateOnly checks and post-pin workflow guards must pass. Run a safe,
   clearly labelled operator alert test after releasing the lock, if possible.
4. Change the prepared paragraphs in olv.md and sheets.md to installed status.
   Update automation_and_r2.md and operations_current.md, record the actual
   release evidence and finish source delivery and task-owned cleanup.

Do not run a real pipeline or connect to IBKR for verification. Do not reset
the runtime automatically. Preserve backups and unrelated operational data.
