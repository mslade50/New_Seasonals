# Weekly scan operations review

Owner request: September 12, 2026. Review each Friday at 22:00 America/New_York;
the inaugural review was run Saturday September 12 for September 7–11.

## Evidence collection

Use the installed runtime recorded by the enabled Task Scheduler actions and
its .local/automation-runtime.json; source on main alone is not deployment.
Read the stable state directory:
C:/Users/McKinley Slade/dev/New_Seasonals/artifacts/automation.

Run scripts/weekly_scan_audit.py with --state-root, --start and --end (ET dates)
and --output under ignored artifacts/weekly-scan-audit/YYYY-MM-DD/.
Pass current scan_coverage_all_am.json and scan_coverage_all_pm.json through
--coverage when they fall in the review window. Include downloaded GitHub
daily_screener.yml logs through --extra-log for cloud fallback runs.

The scanner archives the complete exception lists and the OLV number, multiplier,
budget and quantity behind each email in scan_audits/YYYY-MM-DD. It also prints
a sanitized SCAN_AUDIT_JSON record for cloud log retention. SMTP acceptance is
not proof of delivery to the recipient's inbox. If staging fails before the
email archive, supervisor logs and component receipts remain necessary.
Some GitHub secret masking can render JSON logs unreadable: report this evidence
gap and inspect individual warning lines; never guess a successful scan.

Inspect all trading sessions' AM/PM receipts and due jobs, including local/cloud
fallback decisions. Check prices, stale or malformed OHLCV, missing indicators,
earnings refresh failures, inventory attribution/coverage/NAV, unavailable
caps or exits, staging preservation/partial writes, email status, duplicate
dispatches, runtime pin drift and exception trends. Distinguish benign strategy
filter messages and library deprecations from operational failures.
Never rerun a scan whose side effects are uncertain.

## Ticker disposition

For each failing symbol, retry a bounded read and verify its identity. Confirm a
completed delisting/merger or permanent listing retirement using an issuer,
exchange or SEC source. Provider errors, stale bars, unchanged prices and an
announced acquisition alone do not establish retirement. Check symbol changes,
provider aliases, temporary halts and rate limits before proposing a removal.

The owner authorized removing confirmed delisted tickers from future scans.
Record each verified exclusion in config/live_scan_exclusions.json with source,
reason, verification date and a prospective effective_from date. The live scanner
applies this registry after expanding both liquid and overflow strategy books.
Keep historical universes, valid prices and trades intact to avoid survivorship
bias. Do not substitute an acquirer's or OTC symbol without a separate review.

Implement registry updates in an isolated worktree, run the exclusion/scan tests,
review the diff and deploy the tested source to the installed runtime and its
immutable GitHub fallback tag. A merged registry that is absent from the installed
runtime has not changed live scans. Preserve prior generations and user changes.
No broker orders, cancellations, inventory reallocations, historical deletion,
receipt clearing, or arbitrary strategy/risk changes are authorized by this
weekly housekeeping process. Escalate consequential repairs with the precise
missing evidence and proposed action.

## Result

Save a dated readable report with evidence coverage, ticker decisions and direct
sources, operational failures, changes actually deployed and unresolved actions.
Notify the owner of meaningful new findings, completed changes, failures or
required decisions. If nothing material changed, stay quiet. On an unavailable
machine or inaccessible evidence, report the audit as blocked/incomplete;
never present missing evidence as healthy production.
