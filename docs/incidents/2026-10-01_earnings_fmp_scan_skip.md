# Incident: 2026-09-30 earnings refresh failure on FMP, evening scan skipped and morning scan late (local-primary runtime v9)

Written 2026-10-01 under cutover cadence rule 3 (`docs/local_automation_task_scheduler.md`, "Cutover cadence"): this record has to exist before the Alpha-only runtime release goes into v9. All times are US Eastern (EDT, UTC-4) unless a raw UTC value is quoted beside them.

## Sources

- Runtime logs, copied forward into the dev checkout: `artifacts/automation/logs/2026-09-30/postclose-274761df.log`, `artifacts/automation/logs/2026-10-01/premarket-67cf01e7.log` (04:10 run), `premarket-810bc4c2.log` (05:45 retry), `premarket-84c4193f.log` (06:07 operator-started retry).
- Runtime source at the v9 pin (`3ee156c3`, tag `automation-runtime-2026-09-30.earnings-retries-issuer-review.v2`): `scripts/refresh_earnings_calendar.py`, `scripts/automation_supervisor.py`, `daily_scan.py`, `earnings_filter.py`, `earnings_calendar_provider.py`. Read only.
- GitHub runs 36778522282 and 36798279276 (`build_earnings_calendar.yml`), 36780590846 (`build_macro_releases.yml`), 36847690283 (`deploy_site.yml`).
- Git on main: `be52b79f`, `8bfe68b9`, `e316fb6d`.
- Release preparation: `artifacts/earnings-alpha-only-release/` (`release.py`, `rollback-facts.json`, `marker-before.json`, test logs).
- Background on the FMP expiry: `docs/earnings_monitor_promotion_2026-09-30.md`, `docs/fmp_cutover_2026-09-24.md`.

## 1. Summary

1. What failed: the `earnings_and_grades` job of the v9 `postclose` pipeline on 2026-09-30. `scripts/refresh_earnings_calendar.py` exited 1 at about 17:15 with "Earnings refresh stopped: Initial overflow history refresh failed; cutover refused". The calendar producer still made FMP requests for a history bootstrap after the FMP renewal had been cancelled, and one of those requests hard-failed.
2. Knock-on: `portfolio_report`, `scan_pm`, `private_site_pm` and `shared_site_pm` were skipped on unsatisfied dependencies. The next morning `scan_am` stopped at the scanner's own earnings freshness gate at about 04:16 and was left `indeterminate`, so nothing retried it automatically.
3. Impact on trading: the 9/30 evening scan and portfolio email did not run. The 10/01 AM scan ran at 06:12 instead of about 04:20, ahead of the 09:05 event auction, 09:10 OLV exits and 09:31 order chain. As far as the runtime logs show, no order was placed in error or missed. That statement comes from logs, not from broker records.
4. Recovery: the calendar was republished at 06:05 from a clean export of `be52b79f` (Alpha Vantage only, no FMP request), the `scan_am` receipt was resolved to `retryable_failure` at 06:06, and the scheduled task's own retry command ran the scan to success at 06:12. The runtime itself was not changed.
5. Not yet fixed in production: the v9 runtime still runs the FMP-dependent producer. Until `be52b79f` and `8bfe68b9` are released into v9, the 17:10 earnings job on 2026-10-01 will fail the same way.

## 2. Timeline, 2026-09-30 17:10 to 2026-10-01 06:12 ET

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

### Why the runtime was not released

The release was ready on the evening of 9/30. The session's permission gate blocked the release script as a production deploy, and owner approval arrived at about 05:58 on 10/01, inside the 04:00 to 09:35 window. Releasing then would have broken cadence rule 2, so the release was held. The calendar was repaired from a clean export instead, and the scan was rerun on the unchanged runtime with the scheduled task's own command.

## 5. Verification (2026-10-01)

- R2 `earnings_calendar.parquet`: ETag `71d2121f6b87b587f569e4fe22989350`, `calendar_as_of` 2026-10-01, readback verified after a conditional upload against the 9/29 ETag.
- Content check against the stale calendar: no date changed inside 2026-09-17..2026-10-16, so the blackout decisions the 9/30 calendar would have produced are the same as the 10/01 calendar's (`decision_differences` 0).
- `scan_am` 2026-10-01: `success scan_am (local)` at 06:12 (`premarket-84c4193f.log` line 8201). The run logged degraded coverage (`DX-Y-NYB` unavailable; prior-close OLV inventory `NoSuchKey`; the OLV-EXIT warning in section 6). These are not caused by this incident.
- Site deploys: `private_site_am` (run 36847690283) and `shared_site_am` (run 36849767503) both succeeded; the premarket retry finished with exit 0 at 06:34.

## 6. Still open

| Item | Detail | Needed by |
|---|---|---|
| Runtime release | Release `be52b79f` + `8bfe68b9` into v9 under tag `automation-runtime-2026-10-01.earnings-alpha-only`. Script and rollback facts in `artifacts/earnings-alpha-only-release/`. Must run in the clean slot after 10:00 ET and before the 17:10 postclose. Until then the 17:10 earnings job will fail again. | 2026-10-01 before 17:10 |
| Known gaps in `8bfe68b9` | A row already `schedule_unverified` from an earlier night is not superseded if Alpha re-lists the same period later. A single far-dated vanished event is dropped silently. A relabelled fiscal period publishes both dates. | open |
| Earnings monitor | `scripts/compare_earnings_shadow.py` and `scripts/prepare_earnings_issuer_review.py` still compare against an FMP reference and lose their baseline. The Codex monitor prompt (`~/.codex/automations/compare-alpha-vantage-earnings-with-fmp/automation.toml`) names the old SHA and tag. | open |
| Failure logging | Write the failed symbol list (and status) to `failure.json` for any future provider failure. | open |
| Stale docs | FMP descriptions in `docs/operations_current.md`, `docs/claude_ref/automation_and_r2.md`, `docs/fmp_retirement_inventory.md`, `AGENTS.md`. | open |
| `trend_sleeve` 2026-09-30 | Indeterminate after side effects (`trend_sleeve.py:456`). Needs an operator `resolve`. Not part of this incident. | open |
| `macro_releases` 2026-09-30 | "official coverage gate failed"; GitHub fallback 36780590846 failed too. Cause not investigated here. | open |
| OLV exits | `[OLV-EXIT] WARNING: actual OLV inventory/exit metadata unverified (ValueError); prior staging preserved` in the AM scan on 9/29, 9/30 and 10/01, so `OLV_Exits_Primary` has not been rewritten for three mornings. Unrelated to this incident; needs its own look. | open |
| Automation tests | `test_only_guarded_controller_retains_a_cron_for_migrated_jobs` and `test_installer_defines_the_required_local_clock_schedule` already fail on main and on the runtime line. | open |
