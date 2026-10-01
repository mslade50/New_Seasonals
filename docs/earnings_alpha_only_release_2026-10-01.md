# Earnings Alpha-only runtime release, October 1, 2026

The v9 runtime was released at 10:11 ET on October 1 in the clean slot after the
premarket window. New pin `a7f49f00865fdaf6ef598845b6a1504a9899478b`, tag
`automation-runtime-2026-10-01.earnings-alpha-only`: cherry-picks of main commits
`be52b79f`, `8bfe68b9` and `55017156` onto the previous pin `3ee156c3` (tag
`automation-runtime-2026-09-30.earnings-retries-issuer-review.v2`). Main commit
`f65eb5db` moved `AUTOMATION_RUNTIME_REF`, so the GitHub fallback uses the same tag.

Production earnings are Alpha Vantage only: provider `alpha`, `alpha_fallback`
`stop`, `confirmation_provider` `calendar` in `config/earnings_calendar.json`. No
FMP bootstrap, confirmation or fallback request remains. FMP has returned HTTP 429
since September 30 and the renewal was cancelled. Vanished near-term events are
kept as `schedule_unverified`; the `8bfe68b9` coverage and forward-shrink gates
(both 0.80) guard publication. One Alpha request per NY date is shared through the
R2 claim `provider_snapshots/alpha_earnings/<NY date>.json`. `55017156` fixes the
BEA GDP and ADP parsing that failed `macro_releases` on September 30.

Marker keys added or changed: `earnings_source_commit` `8bfe68b9…`,
`earnings_alpha_only_release_at_utc` 2026-10-01T14:11:31Z,
`macro_releases_source_commit` `55017156…`.

Runtime validation: 174 earnings and macro tests passed. Two automation tests
fail on the runtime line and failed identically before the release; they are
stale Discretionary-retirement tests fixed on main in `9344b8b0`, not yet
released. `run_local_automation.ps1 -ValidateOnly` passed for `premarket` and
`postclose`. No-publish smoke runs from the runtime passed: earnings with provider
`alpha`, 147,801 rows and `decision_differences` 0; macro with 29 official series
and `fmp_requests` 0. Receipts: `artifacts/earnings-alpha-only-release/` (gitignored).

Not in this release: `7cb1560a` (supervisor failure email, `scan_am` earnings
pre-step), `9344b8b0`, and `5c63a3a4` (cache_io lock-safe download). The Codex
monitor prompt still names the old pin and FMP steps and was not edited.

The first scheduled proof is the 17:10 ET postclose run on 2026-10-01. It is
still owed. Background: [9/30 incident](incidents/2026-10-01_earnings_fmp_scan_skip.md).
