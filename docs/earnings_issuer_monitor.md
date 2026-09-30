# Earnings snapshot retries and issuer review

September 29, 2026: authorized by the owner. The monitor uses the reviewed MAIN
observer and queue script with the runtime Python environment. Production remains
on the September 24 pin until a separately approved promotion. No automatic
calendar corrections or strategy changes are part of this observer rollout.

## Shared snapshot retry policy

`alpha_calendar_snapshot.daily_alpha` owns one atomic daily claim and at most
three HTTP attempts: initial attempt, wait 60 seconds after a transient failure,
then wait 180 seconds after another transient failure. Only HTTP 500/502/503/504,
timeouts and connection failures qualify. Authentication, quota/429, malformed or
empty data, coordination conflicts and uncertain snapshot writes stop the run.
Another producer/observer cannot restart an existing claim. Abandoned claims
remain blocked; there is no lease takeover or daily-budget reset. Ready snapshots
are reused. Each attempt is persisted before HTTP, with safe error classification;
the successful raw payload retains its exact digest. Crossing a New York date
stops the old-day attempt sequence. This adds no paid plan or subscription.

The existing pinned producer still has its older single-attempt implementation.
It can reuse a ready snapshot created by the updated morning monitor. If it is
the first caller that day and fails, the monitor does not bypass its old claim.
Full producer retry coverage requires promoting the two changed source files
(`alpha_calendar_snapshot.py`, `scripts/compare_earnings_shadow.py`) together.

## Issuer verification at 8–15 calendar days

Run `scripts/prepare_earnings_issuer_review.py` from MAIN with the runtime Python:

```text
--snapshot-dir <today's authenticated observer artifact>
--symbol-master <downloaded canonical symbol-master artifact>
--fmp-reference <fresh independent FMP reference directory, when available>
--output-dir artifacts/earnings_issuer_review/<new timestamp>
```

This is a queue generator, not a website parser or verified-date feed. It checks
snapshot provenance/hash, counts unique Alpha symbols through day 14 inclusive,
and includes every tracked company where either Alpha or independent FMP has a
date 8–15 calendar days away, inclusive. The tracked universe is CSV_UNIVERSE plus
canonical symbol-master tickers. Liquid is a subset, not an additive population.
FMP-only candidates are included so Alpha omissions cannot escape review.

The weekday morning heartbeat opens each issuer's IR calendar and announcements,
following issuer-linked hosted calendars when necessary. Search snippets alone
do not establish verification. Save an evidence record for every queued company:
ticker, provider date/period, checked-at UTC, issuer URL, announced period/date,
results-publication time separately from webcast time, and a short sourced
paraphrase or permitted excerpt. Preserve page captures or retrieved-page evidence
under that timestamped artifact directory. Status must be one of:

- `confirmed_match`: explicit results release date agrees.
- `date_disagreement`: explicit issuer date differs from the provider.
- `call_date_only`: webcast/call date found; separate release timing unverified.
- `already_reported`: same fiscal period already reported, with filing/release proof.
- `unverified`: no sufficient announcement, inaccessible page or ambiguous period.

Never infer an earnings date from a dividend, last year's event, or an unrelated
filing. Absence of an announcement does not disprove a provider date. Do not use
an assumed fiscal-quarter end to override an issuer's different fiscal calendar.
No successful HTTP response or queue status alone establishes correct dates.

Review all new/changed candidates. Recheck unverified/call-only names each weekday
while in the window; revisit confirmed names when their provider date changes
and at day 8. Carry unresolved previously queued names into a follow-up checklist
if they disappear, jump outside the window or enter the final seven days. Report
new consequential discrepancies and unresolved names at that boundary once;
suppress unchanged alerts. Record any incomplete research, never silently count
it as verified. Findings produce proposed corrections, not canonical writes.

Keep the existing ten-NYSE-trading-day comparison separately. Neither 14 calendar
days for counts nor 8–15 calendar days for issuer review changes strategy blackout
arithmetic, the economic checks, schedule times, or November 11 endpoint.

## Validation and rollout boundary

Regression checks cover retry exhaustion, concurrent callers, quota/auth failures,
day rollover, hash integrity, calendar-day boundaries and FMP-only queue entries.
Initial live queue: `artifacts/earnings_issuer_review/20260929-initial/`.
Source changes require review before a production runtime promotion. The exposure
is that a recovered snapshot may let the Alpha trial publish where FMP fallback
would otherwise be selected; existing incorrect provider dates remain possible.
Rollback is to restore the September 24 runtime/fallback pin and the previous
monitor observer selection, preserving snapshot and historical evidence. A
separate canonical-data repair is required for already-published date errors.
