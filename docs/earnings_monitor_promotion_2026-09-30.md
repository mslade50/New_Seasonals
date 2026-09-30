# Earnings monitor production promotion — September 30, 2026

The owner explicitly approved production promotion on September 30. The existing
v9 runtime is pinned to `3ee156c3cd2f0cbfd9680e887757ffea205f562c` with immutable fallback tag `automation-runtime-2026-09-30.earnings-retries-issuer-review.v2`. Source is
the earnings files from main commit `745141a6e1ea00ea0840d4689f383c8a9a79df83`; unrelated changes from that mixed
commit were excluded. The configured GitHub fallback uses the same tag.

Production and the scheduled morning monitor now share bounded Alpha retries:
initial request, 60-second wait, retry, 180-second wait, final retry. Only transient
network failures and HTTP 500/502/503/504 qualify. One daily claim coordinates all
callers. Quota/auth/invalid responses, failed coordination and uncertain writes
are terminal. Existing claims and ready snapshots are preserved.

Issuer website verification remains a morning monitoring task over tracked names
8–15 calendar days away, with follow-up for unresolved names outside that window.
The queue script and its tests are installed in the production runtime. Findings
are saved and alerted here; the publisher does not consume these review artifacts
or automatically gate or correct expected dates. Only reviewed explicit rules in
`config/earnings_calendar_overrides.json` change calendar dates. Existing
confirmation/history/publication guards and configured FMP fallback remain.

Main validation: 70 earnings tests passed. Promoted runtime validation is retained
at `artifacts/earnings-retry-promotion/runtime-tests.log`; promotion, tag, task,
marker and runner receipts are under the same artifact directory. Jobs were idle
and the supervisor lock held through activation. Retained data was hashed before
and after; no producer or scanner was executed during promotion. Runtime startup
validation confirms installation; the next scheduled receipt must demonstrate
actual publication with this runtime identity.

Known SA/WFC and APLD/FNB/GBCI/INFY date errors are separate unresolved data
corrections. This source promotion does not repair them or remove remaining FMP
confirmations/history/fallback dependencies. FMP renewal stays cancelled.

Activation time: 2026-09-30T19:54:43.845077+00:00.
