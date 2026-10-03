# Brief: finish two trading_ibkr patches and the second runtime release (2026-10-03)

Owner: McKinley. Both jobs are approved. Do them this weekend, before the Monday
2026-10-05 04:10 ET premarket run. Read AGENTS.md and CLAUDE.md first. Background:
`docs/incidents/2026-10-01_earnings_fmp_scan_skip.md`,
`docs/earnings_alpha_only_release_2026-10-01.md`, `scripts/logs/health_check_2026-10-01.md`.

Paths:
- Dev checkout (main): `C:\Users\McKinley Slade\dev\New_Seasonals`
- Production runtime (Task Scheduler runs this): `C:\Users\McKinley Slade\dev\New_Seasonals-automation-runtime-v9`, branch `codex/local-primary-runtime-v9-20260912`, pin `a7f49f00`, tag `automation-runtime-2026-10-01.earnings-alpha-only`
- Live broker scripts (place real orders, not in git): `C:\Users\McKinley Slade\OneDrive\trading_ibkr`
- Patch work from 10/01: `artifacts\ibkr_patches_2026-10-01\` in the dev checkout (gitignored)

## Job 1: two trading_ibkr patches

Both are built and tested on a copy but NOT installed. The live files are still
the 2026-09-09 versions.

**Fix A, `order_staging.py`.** Stops the 09:31 chain re-placing a prior day's
morning staging rows. Today the stager keeps rows with `Scan_Date >= today - 1
NYSE day` and restamps `Staged_Date = today`, so if the evening scan on day D is
skipped and the morning scan on D+1 dies before staging, D's AM rows are placed
again on both accounts at stale levels. This nearly happened on 10/01 with a
FORM short. The patch drops a row only when `Scan_Date < today` AND `Signal_Date
< previous NYSE session`, and logs `[CRITICAL] Dropped N ... NOT re-placing`.
That keeps the designed fallback (D's PM rows stage on D+1 if D+1's AM scan
fails) and operator makeup rows. Normal-day output was byte-identical in 20
parity scenarios.

**Fix B, `olv_exit_pa_legacy.py`.** The PA account has had no OLV volume-confirmed
stop exits: its runner reads the dead `OLV_Exits` tab, while the scanner writes
only `OLV_Exits_Primary`. The patch reads `OLV_Exits_Primary` through
`olv_contract.validate_rows` and matches only the PA bracket with the exact
orderRef (first four fields), conId and time-exit date. There are no fallbacks,
because the legacy "nearest date" matcher could sell a different PA leg in the
same symbol. It then uses the existing cancel-bracket-then-OPG path for PA's own
full leg, clamped to held shares.

Files: `patches\fix_a\`, `patches\fix_b\` (full patched files plus tests), the
`.patch` diffs, `patches\smoke_check.py` (reads only, never connects to IBKR),
`orig\` (the pristine base), `results\`, and `review\` (the reviewer's probes).

An independent review leaned "install" on both, but found these to fix first:
1. Fix A, medium. The new filter (order_staging.py ~74-95) runs inside the try
   at ~878-914. A tz-aware or otherwise odd `Signal_Date` cell raises TypeError,
   which is reported as a Sheets connection error, exits 2 and skips entries on
   both accounts. Wrap the filter so any exception logs one `[CRITICAL]` line and
   returns the rows unchanged (fail open, same as the old code). Reproducer:
   `review\edge_inputs.py`.
2. Fix B, medium. Due rows now include rows carried forward from earlier days
   (`Execute_On <= today`). On a late or manual run after about 09:25, PA would
   place an intraday MKT DAY sell, while `olv_exit_primary.py` (~1267) refuses
   with CRITICAL_MISSED_AUCTION_PENDING. Mirror Primary's refusal past the cutoff.
3. Fix B, medium-low (~267-269). A failed Sheets read or contract failure prints
   a WARN or CRITICAL and still returns 0, and the WARN text still says
   "OLV_Exits". Return 1 with a CRITICAL line on either failure.
4. Not yet run: `review\wrapper_order_probe.py`. Run it to prove that a PA failure
   cannot stop Primary's exits inside `olv_exit_moo.py`. Primary runs first and
   PA exceptions appear to be caught, but this has only been read, not run.
Add tests for 1-3, rerun the full trading_ibkr suite (21 failures pre-exist in
close_resize, event_moo, execution_risk_policy, legend env and stop_limit; there
must be no new ones), and confirm the new tests fail against `orig\`.

Install rules:
- Window: after the Friday 09:31 chain and before Monday 09:10 ET. Only two tasks
  run these files: `\IBKR OLV Pre-Market Exits` (09:10) and `\IBKR Daily Order
  Chain` (09:31). `\IBKR OLV Book Cap (EOD)` imports order_staging but is Disabled.
- First check that the live `order_staging.py`, `olv_exit_pa_legacy.py` and
  `test_olv_exits.py` still hash-equal `orig\`. If not, stop and re-diff.
- Back up each file as `<file>.bak-2026-10-03`, copy the patched files in, run
  `smoke_check.py` and the guard tests from the trading_ibkr folder. Rollback is
  copying the backups back and deleting `test_staging_prior_day_rows.py`.
- Update `docs/claude_ref/olv.md` (PA reads OLV_Exits_Primary, strict identity
  match, due is `<= today`) and `docs/claude_ref/sheets.md` (new staging freshness
  rule; an operator makeup row must carry today's Scan_Date). Commit the docs on main.

## Job 2: second runtime release

On main but not in the runtime. Release all of them with the same procedure used
on 10/01. `artifacts\earnings-alpha-only-release\release.py` is the guarded
script; copy and adapt it: new SOURCES, TAG, PATHS and test list.
- `7cb1560a` supervisor failure emails (off switch `NEW_SEASONALS_ALERT_EMAIL=0`)
  plus the `scan_am` pre-step `scripts/ensure_earnings_calendar.py` and its
  `daily_screener.yml` step
- `9344b8b0` stale tests fixed and `scripts/repo_health_check.py` fixes
- `5c63a3a4` cache_io lock-safe download
- `5187653f`, `beb3b58f`, `71e44658` only if they touch runtime-executed files
  (check; Market Context and issuer review run from main)

Procedure (see memory note "local runtime release procedure" and
`docs/claude_ref/automation_and_r2.md`):
1. Diff each file between runtime HEAD and main first. Never merge main wholesale;
   cherry-pick only these commits.
2. Hold the supervisor lock (`GlobalFileLock` on
   `<dev>\artifacts\automation\automation_supervisor.lock`) and confirm no
   `New Seasonals Local v9*` task is running.
3. Run the touched suites with the runtime venv (`...-v9\.venv\Scripts\python.exe`,
   `-p no:cacheprovider`). The runtime venv lacks ib_insync; those failures are
   environmental.
4. Commit on the runtime branch and tag it
   `automation-runtime-2026-10-03.<name>` (lightweight, like the previous tags).
   Push only the tag, and confirm `AUTOMATION_RUNTIME_REF` in the tag's tree.
5. Update `.local/automation-runtime.json`: `pinned_sha`, `fallback_ref`, and a
   dated `<name>_release_at_utc` key. Keep its CRLF, no-BOM format.
6. Commit the `AUTOMATION_RUNTIME_REF` bump in
   `.github/workflows/local_automation_fallback.yml` on main and push.
7. Run `run_local_automation.ps1 -Pipeline premarket -ValidateOnly` and
   `-Pipeline postclose -ValidateOnly`, then do one test send of the alert email
   path if there is a safe way. Do not run a real pipeline on a weekend.
8. Add a short release note under `docs/`, in the shape of
   `docs/earnings_alpha_only_release_2026-10-01.md`.

Rollback: restore the old marker, run `git reset --keep a7f49f00` in the runtime,
and revert the main pin commit. Ask the owner before any reset.

## Hard rules
- Never run anything in trading_ibkr that can connect to IBKR. No broker orders.
- No `git reset --hard`, force-push or history rewrite.
- One runtime cutover per day, never 04:00-09:35 ET.
- Other sessions commit to main. Commit promptly, re-check `git log`, and never push blind.
- Docs style: plain prose, no em dashes.

## Also known, not part of this brief
- The Market Context brief missed 10/01 because the Claude account hit its monthly
  spend limit, not because of code. Next run is Sunday 18:30.
- The Codex earnings monitor prompt
  (`~/.codex/automations/compare-alpha-vantage-earnings-with-fmp/automation.toml`)
  still asks for an FMP reference and names the old pin. FMP returns 429. Switch it
  to `prepare_earnings_issuer_review.py --alpha-r2` and drop the FMP steps.
- The trend sleeve routes MKT DAY at about 09:31 because `trend_moo_enabled.flag`
  is absent. Owner decision before the 10/30 month end. Also, when both staging
  tabs are empty, `order_staging.py` returns before the Trend path (~875), so a
  zero-signal first session of a month skips the rebalance.

Report back with: hashes before and after, test counts, the new pin and tag, the
ValidateOnly output, and anything skipped.
