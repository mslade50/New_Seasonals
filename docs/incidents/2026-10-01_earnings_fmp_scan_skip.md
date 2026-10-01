# Incident: 2026-09-30 earnings refresh failure on FMP, evening scan skipped and morning scan late (local-primary runtime v9)

Written 2026-10-01 under cutover cadence rule 3 (`docs/local_automation_task_scheduler.md`, "Cutover cadence"): this record has to exist before the Alpha-only runtime release goes into v9. All times are US Eastern (EDT, UTC-4) unless a raw UTC value is quoted beside them.

## Sources

- Runtime logs, copied forward into the dev checkout: `artifacts/automation/logs/2026-09-30/postclose-274761df.log`, `artifacts/automation/logs/2026-10-01/premarket-67cf01e7.log` (04:10 run), `premarket-810bc4c2.log` (05:45 retry), `premarket-84c4193f.log` (06:07 operator-started retry).
- Runtime source at the v9 pin (`3ee156c3`, tag `automation-runtime-2026-09-30.earnings-retries-issuer-review.v2`): `scripts/refresh_earnings_calendar.py`, `scripts/automation_supervisor.py`, `daily_scan.py`, `earnings_filter.py`, `earnings_calendar_provider.py`. Read only.
- GitHub runs 36778522282 and 36798279276 (`build_earnings_calendar.yml`), 36780590846 (`build_macro_releases.yml`), 36847690283 (`deploy_site.yml`).
- Git on main: `be52b79f`, `8bfe68b9`, `e316fb6d`, `55017156`, `71e44658`, `9344b8b0`, `f65eb5db`, `7cb1560a`.
- Release preparation and receipts: `artifacts/earnings-alpha-only-release/` (`release.py`, `release.log`, `release.json`, `rollback-facts.json`, `marker-before.json`, `marker-after.json`, test logs). Gitignored.
- Release record: `docs/earnings_alpha_only_release_2026-10-01.md`.
- Background on the FMP expiry: `docs/earnings_monitor_promotion_2026-09-30.md`, `docs/fmp_cutover_2026-09-24.md`.

## 1. Summary

1. What failed: the `earnings_and_grades` job of the v9 `postclose` pipeline on 2026-09-30. `scripts/refresh_earnings_calendar.py` exited 1 at about 17:15 with "Earnings refresh stopped: Initial overflow history refresh failed; cutover refused". The calendar producer still made FMP requests for a history bootstrap after the FMP renewal had been cancelled, and one of those requests hard-failed.
2. Knock-on: `portfolio_report`, `scan_pm`, `private_site_pm` and `shared_site_pm` were skipped on unsatisfied dependencies. The next morning `scan_am` stopped at the scanner's own earnings freshness gate at about 04:16 and was left `indeterminate`, so nothing retried it automatically.
3. Impact on trading: the 9/30 evening scan and portfolio email did not run. The 10/01 AM scan ran at 06:12 instead of about 04:20, ahead of the 09:05 event auction, 09:10 OLV exits and 09:31 order chain. As far as the runtime logs show, no order was placed in error or missed. That statement comes from logs, not from broker records.
4. Recovery: the calendar was republished at 06:05 from a clean export of `be52b79f` (Alpha Vantage only, no FMP request), the `scan_am` receipt was resolved to `retryable_failure` at 06:06, and the scheduled task's own retry command ran the scan to success at 06:12. The runtime itself was not changed.
5. Production fix: `be52b79f`, `8bfe68b9` and `55017156` were released into v9 at 10:11 ET on 2026-10-01 (pin `a7f49f00865fdaf6ef598845b6a1504a9899478b`, tag `automation-runtime-2026-10-01.earnings-alpha-only`). The earnings job no longer makes any FMP request. The first scheduled proof is the 17:10 ET postclose run on 2026-10-01, which is still owed.

## 2. Timeline, 2026-09-30 17:10 to 2026-10-01 10:14 ET

| ET | Event | Source |
|---|---|---|
| 09-29 evening | Nightly bootstrap re-requests the same extra symbols. All return empty, the run publishes. R2 `earnings_calendar.parquet` gets `calendar_as_of` 2026-09-29, ETag `2770e2007c76d62b5648fe8faaa7edaa`. | operator notes; R2 object metadata |
| 09-30 17:10 | `postclose` starts on runtime v9, pin `3ee156c3`. | postclose log |
| ~17:15 | `start earnings_and_grades` (token `2026-09-30-earnings_and_grades-d4608603e3`). `refresh_earnings_calendar.py` prints "Earnings refresh stopped: Initial overflow history refresh failed; cutover refused" and exits 1. The run folder is `artifacts/earnings_provider/20260930T211535081472Z` (21:15:35Z). | postclose log lines 184-191 |
| ~17:15 | Supervisor: "local earnings_and_grades failed before side effects", immediate GitHub fallback dispatches `build_earnings_calendar.yml`. | postclose log lines 191-193 |
| ~17:20 | GitHub fallback run 36778522282 fails the same way. `skip portfolio_report: unsatisfied dependencies: earnings_and_grades`. | postclose log lines 210-211 |
| 17:10 window | `trend_sleeve` goes indeterminate after side effects (`trend_sleeve.py:456`, "Trend legacy inventory requires a reviewed fill-history bootstrap"). Separate failure, see section 6. | postclose log lines 234-266 |
| later in run | `skip scan_pm: unsatisfied dependencies: earnings_and_grades`. `macro_releases` fails on "official coverage gate failed" and its GitHub fallback 36780590846 fails too (separate failure). `skip private_site_pm` and `skip shared_site_pm: unsatisfied dependencies: scan_pm`. | postclose log lines 1074-1099 |
| 20:51 | Hourly fallback controller re-dispatches the earnings build, run 36798279276. It fails the same way. | GitHub run 36798279276 |
| ~22:15 to 22:35 | Alpha Vantage only mode written and committed on main as `be52b79f` (commit time 22:28:37). 79 focused earnings tests pass; independent review finds no blocker; a no-upload replay of that night's inputs matches the 9/29 calendar plus six new November rows. The cherry-pick applies cleanly in the v9 runtime line and passes there (72 earnings tests; the two automation test failures on that line predate the change). | `git log`; release test logs in `artifacts/earnings-alpha-only-release/` |
| overnight | Runtime release not run: the session's permission gate classed the release script as a production deploy and held it for owner approval. | session record |
| 10-01 04:10 | `premarket` starts. `cboe_am`, `breadth_am`, `master_prices_am`, `risk_am`, `event_sleeve_am` all `success (local)`. | `premarket-67cf01e7.log` lines 2-186 |
| ~04:16 | `start scan_am` (token `2026-10-01-scan_am-59074548ff`). Supervisor persists the side-effect boundary for "run unified scanner", then `daily_scan.py` raises `CalendarError: Earnings calendar missed the previous NYSE session; refresh required` from `earnings_calendar_provider.py:64` via `earnings_filter.py:130`, called at `daily_scan.py:3005`. Supervisor: "local scan_am is indeterminate after side effects ... automatic fallback suppressed". `skip private_site_am`, `skip shared_site_am`. | `premarket-67cf01e7.log` lines 187-240 (file mtime 04:16) |
| 05:45 | `premarket-retry` skips every earlier job on its success receipt and skips `scan_am`: "indeterminate receipt ... pre-existing, never re-run automatically". Sites skipped again. | `premarket-810bc4c2.log` lines 2-9 |
| ~05:58 | Owner approval for the runtime release arrives. This is inside the 04:00 to 09:35 no-cutover window (cadence rule 2), so the release is held and the calendar is repaired without touching the runtime. | session record |
| 06:02 | No-upload dry run from a clean `git archive` export of `be52b79f`. One Alpha request for the day, shared snapshot written to `provider_snapshots/alpha_earnings/2026-10-01.json`. Candidate: 147,801 rows, 1,484 tickers, 26 added rows (all Alpha `expected`, dated 2026-11-23/24), none removed, no duplicate (ticker, date), no date change inside 2026-09-17..2026-10-16. Seven 2026-09-30 events (CAG, CALM, FDS, JBL, MU, PRGS, SA) move from `expected` to `schedule_unverified`. `decision_differences` 0. | dry-run receipt |
| 06:05 | One-off publish from the same export with default arguments. Conditional upload on the prior ETag, readback verified. New ETag `71d2121f6b87b587f569e4fe22989350`, `calendar_as_of` 2026-10-01. No FMP request. | publish receipt; R2 object metadata |
| 06:06 | `resolve --pipeline premarket --job scan_am --date 2026-10-01 --disposition retryable_failure`. | operator command |
| 06:07 | Premarket retry started by hand with the scheduled task's own command (`run_local_automation.ps1 -Pipeline premarket-retry`), inside its 05:45 to 07:00 window, on the unchanged v9 runtime. `start scan_am` token `2026-10-01-scan_am-397cab4634`. | `premarket-84c4193f.log` line 7 |
| 06:12 | Scan completes: 1 signal (FORM, Overbot Vol Spike, Overflow tier), Signals Log synced (944 rows), `moc_orders` cleared, `Order_Staging` cleared (no Liquid rows), 1 instruction row staged, scan email sent (`SCAN_AUDIT_JSON generated_at 10:12:12Z`), `exposure_state.json` published and verified. `success scan_am (local)`. | `premarket-84c4193f.log` lines 8182-8201 |
| 06:12 onward | Site deploys dispatched: `deploy_site.yml` run 36847690283 for `private_site_am`, then `deploy_shared_seasonals.yml` run 36849767503 for `shared_site_am`. Both succeeded; the wrapper exited 0 at 06:34. | `premarket-84c4193f.log` line 8202 onward |
| 06:26 to 06:27 | `8bfe68b9` (guard hardening) and `e316fb6d` (keep the GitHub fallback pin on the live tag) committed on main. All three commits pushed. | `git log` |
| 06:56 | `55017156` committed on main: BEA GDP title match accepts the comma form, BEA links limited to `https://www.bea.gov/news/`, GDP and PCE parsed independently, ADP reference month may be the publication month or the month before. This is the fix for the separate 9/30 `macro_releases` failure. | `git log` |
| 07:11 | `71e44658`: `scripts/prepare_earnings_issuer_review.py --alpha-r2` reads the day's R2 Alpha snapshot read-only, so the issuer review no longer needs FMP. | `git log` |
| 07:13 | `9344b8b0`: stale automation tests fixed on main; `scripts/repo_health_check.py` blind spots fixed. Not in the runtime. | `git log` |
| 08:10 | Reviewed flat trend sleeve bootstrap state published to R2 `trend_sleeve_state.json`. The 9/30 `trend_sleeve` receipt was resolved `retryable_failure` the same day (section 6). | operator record |
| 10:11 | Runtime release in the clean slot: v9 pin `a7f49f00`, tag `automation-runtime-2026-10-01.earnings-alpha-only`, cherry-picks of `be52b79f`, `8bfe68b9` and `55017156` onto `3ee156c3`. Main pin commit `f65eb5db` moves `AUTOMATION_RUNTIME_REF` to the new tag. `released_at_utc` 14:11:31Z. | `release.log`, `release.json`, `marker-after.json` |
| 10:14 | `7cb1560a` committed on main: supervisor failure email and the `scan_am` earnings pre-step. Not in the runtime; needs the next release. | `git log` |

## 3. Root cause

### Known (evidence in hand)

- The raise is at the runtime's `scripts/refresh_earnings_calendar.py:270-272`: `fetch_fmp_rows(extra, ...)` returns `(bootstrap, failed, empty)`, and any non-empty `failed` raises `CalendarError("Initial overflow history refresh failed; cutover refused")`. The `extra` set is `universe - set(prior.ticker)` when the prior calendar is authoritative (line 268), so every night it is the symbol_master names that never get rows: about 68 symbols (ETFs, FX, indices), of which about 42 go to FMP.
- On 9/29 every one of those requests came back empty and the run published. On 9/30 at least one came back as a hard failure.
- The FMP renewal had been cancelled. `docs/earnings_monitor_promotion_2026-09-30.md` and `docs/fmp_cutover_2026-09-24.md` list removal of the FMP confirmation, bootstrap and fallback paths as open blockers before FMP expiry. The dependency was known and still live in the pinned runtime.
- `scan_pm` depends on `earnings_and_grades` (`scripts/automation_supervisor.py:933`, `depends_on=("master_prices_pm", "risk_pm", "earnings_and_grades")`), and so does `portfolio_report` (line 817). One failed earnings refresh removes the evening scan, the portfolio email and both PM site builds.
- The scanner has its own freshness gate (`earnings_calendar_provider.py:64`), reached at `daily_scan.py:3005`, after the supervisor has persisted the side-effect boundary. The morning scan therefore failed as `indeterminate` and automatic fallback and the 05:45 retry both stood down by design. The failure point is ahead of the Sheets staging writes (`daily_scan.py:3754` onward), so no staging rows or email went out at 04:16.
- The GitHub fallback (`build_earnings_calendar.yml`) runs the same script from the same tag, so it fails the same way on a provider-side error. It did, twice (36778522282, 36798279276).

### Unknown

- Which symbol hard-failed on 9/30 and with which HTTP status. The script does not log the `failed` list, and `failure.json` holds only the error string.

### Contributing factors

1. The nightly producer still hard-required FMP for a history bootstrap after the FMP renewal was cancelled.
2. The bootstrap re-requests symbols that can never return rows, and treats any single hard failure among them as fatal.
3. Two gates sit on one input: the supervisor dependency (`scan_pm` on `earnings_and_grades`) and the scanner's freshness check. A single failed refresh takes out both the evening scan and the next morning's scan.
4. The failing symbol and status are not logged, so the trigger cannot be diagnosed after the fact.
5. The GitHub fallback is the same code from the same tag, which gives no protection against a provider-side failure.

## 4. Response

### Code on main (pushed)

1. **`be52b79f`** (2026-09-30 22:28), "Cut earnings calendar over to Alpha Vantage only": config `provider alpha`, `alpha_fallback stop`, `confirmation_provider calendar`; no FMP request; events that vanish from Alpha are kept as `schedule_unverified`; the supervisor job requires `ALPHA_VANTAGE_API_KEY`. Validation as in the timeline (79 tests, independent review, no-upload replay matching 9/29 plus six November rows; clean cherry-pick and 72 earnings tests on the v9 line).
2. **`8bfe68b9`** (2026-10-01 06:26), "Harden Alpha-only earnings calendar guards": a re-dated event no longer publishes both dates; the coverage gate ignores re-inserted forward unverified rows on both sides; new forward shrink gate at 80%; the receipt counts only current and future unverified rows. Validated by replay over every saved day pair from 9/16 to 10/01 and reviewed independently.
3. **`e316fb6d`** (2026-10-01 06:27): keeps the GitHub fallback pin on the live runtime tag until the runtime release, so the controller and the local runtime stay on the same code.

4. **`55017156`** (2026-10-01 06:56): macro release parser fixes (timeline). Released with the earnings commits.
5. **`f65eb5db`** (2026-10-01 10:11): main pin commit, `AUTOMATION_RUNTIME_REF` moves to `automation-runtime-2026-10-01.earnings-alpha-only`.
6. **`7cb1560a`** (2026-10-01 10:14), on main only, needs the next release:
   - Failure email. After `run` or `run-pipeline`, one plain-text email goes out when any counted job ends failed, indeterminate or blocked. `health` emails its SUMMARY block on any FAIL or when the lock is held. Gmail SMTP through `EMAIL_USER` / `EMAIL_PASS`; recipients from `NEW_SEASONALS_ALERT_RECIPIENTS` (default: the operator address the scan email already uses); off switch `NEW_SEASONALS_ALERT_EMAIL=0`. Sent after the supervisor lock is released, secrets redacted, never changes an exit code, nothing on dry-run, plan, status or fallback-due.
   - `scan_am` pre-step `scripts/ensure_earnings_calendar.py`, before the side-effect boundary. It checks the canonical R2 calendar with `validate_freshness`; if stale, it runs the normal producer once; if R2 is unreadable it exits 1 without calling the producer. `.github/workflows/daily_screener.yml` gets the same step for the AM bookend. The `scan_am` local lease grows from 1800 s to 3000 s because the pre-step's 1200 s timeout now counts before the boundary.
   - Limits: if the local pre-step fails and the GitHub fallback also fails, `scan_am` still ends `indeterminate` and the 05:45 retry skips it. A morning repair spends that date's one Alpha request, so the 17:10 job reuses the morning snapshot.

### Runtime release (2026-10-01 10:11 ET)

v9 pin `a7f49f00865fdaf6ef598845b6a1504a9899478b`, tag `automation-runtime-2026-10-01.earnings-alpha-only`: cherry-picks of `be52b79f`, `8bfe68b9` and `55017156` onto the previous pin `3ee156c3` (tag `automation-runtime-2026-09-30.earnings-retries-issuer-review.v2`). The local marker gained `earnings_source_commit` `8bfe68b9…`, `earnings_alpha_only_release_at_utc` and `macro_releases_source_commit` `55017156…`. Main commit `f65eb5db` moved the GitHub fallback pin to the same tag, so the controller and the local runtime run the same code. Details: `docs/earnings_alpha_only_release_2026-10-01.md`.

### Why the runtime was not released overnight

The release was ready on the evening of 9/30. The session's permission gate blocked the release script as a production deploy, and owner approval arrived at about 05:58 on 10/01, inside the 04:00 to 09:35 window. Releasing then would have broken cadence rule 2, so the release was held. The calendar was repaired from a clean export instead, and the scan was rerun on the unchanged runtime with the scheduled task's own command.

## 5. Verification (2026-10-01)

- R2 `earnings_calendar.parquet`: ETag `71d2121f6b87b587f569e4fe22989350`, `calendar_as_of` 2026-10-01, readback verified after a conditional upload against the 9/29 ETag.
- Content check against the stale calendar: no date changed inside 2026-09-17..2026-10-16, so the blackout decisions the 9/30 calendar would have produced are the same as the 10/01 calendar's (`decision_differences` 0).
- `scan_am` 2026-10-01: `success scan_am (local)` at 06:12 (`premarket-84c4193f.log` line 8201). The run logged degraded coverage (`DX-Y-NYB` unavailable; prior-close OLV inventory `NoSuchKey`; the OLV-EXIT warning in section 6). These are not caused by this incident.
- Site deploys: `private_site_am` (run 36847690283) and `shared_site_am` (run 36849767503) both succeeded; the premarket retry finished with exit 0 at 06:34.
- Runtime release at 10:11: 174 earnings and macro tests passed in the runtime. The two automation test failures on the runtime line (`test_installer_defines_the_required_local_clock_schedule`, `test_only_guarded_controller_retains_a_cron_for_migrated_jobs`) were present before the release and are stale Discretionary-retirement tests, fixed on main in `9344b8b0` and not yet in the runtime. `run_local_automation.ps1 -ValidateOnly` passed for `premarket` and `postclose`.
- No-publish smoke runs from the released runtime: earnings with provider `alpha`, 147,801 rows, `decision_differences` 0; macro with 29 official series and `fmp_requests` 0.
- Still owed: the first scheduled proof of the released runtime, the 17:10 ET `postclose` run on 2026-10-01.

## 6. Still open

| Item | Detail | Needed by |
|---|---|---|
| Runtime release | DONE 2026-10-01 10:11 ET: pin `a7f49f00`, tag `automation-runtime-2026-10-01.earnings-alpha-only`, cherry-picks of `be52b79f`, `8bfe68b9`, `55017156`. First scheduled proof is the 17:10 ET postclose on 2026-10-01. | proof owed 2026-10-01 17:10 |
| Failure alerting and `scan_am` pre-step | Done on main in `7cb1560a`, not in the runtime. Waits for the next runtime release. | next release |
| Single point of failure | The earnings calendar still gates both scans: `scan_pm` through the supervisor dependency, `scan_am` through the scanner's freshness gate at `daily_scan.py:3005`. The pre-step in `7cb1560a` narrows the morning half once released. | open |
| Known gaps in `8bfe68b9` | A row already `schedule_unverified` from an earlier night is not superseded if Alpha re-lists the same period later. A single far-dated vanished event is dropped silently. A relabelled fiscal period publishes both dates. No actuals, EPS or surprise values are populated after the cutover. The earnings universe is still the symbol_master frozen 2026-06-05. | open |
| Earnings monitor | Still degraded. `scripts/prepare_earnings_issuer_review.py --alpha-r2` (`71e44658`) works without FMP. `scripts/compare_earnings_shadow.py` still requires an FMP baseline, so the Alpha versus FMP comparison is unavailable. The Codex monitor prompt (`~/.codex/automations/compare-alpha-vantage-earnings-with-fmp/automation.toml`) still names the old pin and the FMP steps; not edited. | open |
| Failure logging | Write the failed symbol list (and status) to `failure.json` for any future provider failure. | open |
| Stale docs | Being fixed in this pass for `docs/operations_current.md`, `docs/claude_ref/automation_and_r2.md`, `docs/fmp_retirement_inventory.md` and `docs/earnings_alpha_cutover.md`. `AGENTS.md` not touched in this pass. | in progress |
| `trend_sleeve` 2026-09-30 | Resolved. The run raised at `trend_sleeve.py:456` because the R2 state lacked `inventory_basis`; no side effect. Receipt resolved `retryable_failure`; reviewed flat bootstrap state published to R2 at 08:10 on 10/01. Next month-end run 2026-10-30. See `docs/claude_ref/strategies_3x_and_pilots.md`. | done |
| `macro_releases` 2026-09-30 | Fixed by `55017156` (BEA GDP comma-form title, BEA link checks, independent GDP/PCE parsing, ADP reference month) and released at 10:11. Official sources carry no consensus, so Market Context's P12 surprise lane has no cells for prints after 2026-09-23. | done |
| OLV exits | `[OLV-EXIT] WARNING: actual OLV inventory/exit metadata unverified (ValueError); prior staging preserved` in the AM scan on 9/29, 9/30 and 10/01, so `OLV_Exits_Primary` has not been rewritten for three mornings. Unrelated to this incident; needs its own look. | open |
| Automation tests | `test_only_guarded_controller_retains_a_cron_for_migrated_jobs` and `test_installer_defines_the_required_local_clock_schedule` are fixed on main in `9344b8b0` and still fail on the runtime line until the next release. | next release |
