# FMP retirement progress — September 23, 2026

## Active retirements

The owner's September 23 instruction retires Discretionary Focus and analyst
grades. Historical files and existing research records are retained.

- Windows task `New Seasonals Local v9 - discretionary` is disabled, verified
  through Task Scheduler after administrator elevation.
- GitHub workflow `discretionary_focus.yml` is `disabled_manually`; its job also
  has an unconditional false guard. Its dedicated fallback cron was removed.
- The supervisor's legacy `discretionary` pipeline has no jobs, so an old task
  definition or manual dispatch cannot fetch, publish, or email Focus output.
- Future local installs omit the Focus task; health checks no longer demand
  Focus receipts or logs.
- The stable `earnings_and_grades` receipt ID now runs earnings only. No grade
  producer or grade-output validation remains in that job or its GitHub backup.
  The grade command-line entry point is inert even if invoked separately.
- Earnings provider selection is unchanged in production: FMP remains active.
  EP and the fundamental research sleeve are separate and remain unchanged.

Main commit: `2ef9fa21466857b9c4470ab3ed3a008f1f11a09c`.
Existing runtime v9 was advanced, without a new branch/worktree, to
`dd853d0892f6663feab3426be66c7511fe3d7de9`, published as
`automation-runtime-2026-09-23.research-retirement`. The local marker and GitHub
fallback controller use this tag. All other eight local tasks remain enabled.
The pinned-runner validation and 76 retirement/supervisor tests passed in v9.
This is runtime configuration verification; the next postclose production run
has not yet occurred. No producer, email, broker operation, or site build was
triggered during promotion.

Evidence: `artifacts/fmp_retirement/20260923/focus-task-state.json`,
`runtime-promotion.json`, and preserved pre-change files under `before/`.
Rollback would be an explicit forward change restoring the retired jobs and
enabling Focus; no histories or subscription records were deleted.

## Economic replacement validation

The candidate collector now retrieves **29 series** with release timestamps and
reported values from free sources. The previous 24-series implementation is
extended with:

- ISM manufacturing and services PMI: ISM-authored releases distributed on
  [PR Newswire](https://www.prnewswire.com/news/institute-for-supply-management/).
  The collector discovers the latest release, verifies its issuer, reference
  month and publication timestamp, and reads its explicit next-release notice.
- ADP employment change: [ADP's own press releases](https://mediacenter.adp.com/press-releases?l=100),
  including the 08:15 ET release timestamp, signed jobs change in thousands and
  explicit next-release notice.
- Retail sales excluding autos: the percentage-change table in the
  [Census release](https://www.census.gov/retail/marts/www/marts_current.pdf),
  with table-month checks to avoid extracting dollar levels or sampling errors.
- Claims freshness: the [DOL publication schedule](https://oui.doleta.gov/unemploy/archive.asp),
  applying its Thursday rule and explicit holiday exceptions. A PDF from an
  older due release is rejected; the Thanksgiving Wednesday exception is tested.
- JOLTS: BLS API values plus the [New York Fed's official economic calendar](https://www.newyorkfed.org/research/calendars/i-sep26.html).
  Its public monthly pages supply dates and Eastern times without the BLS
  schedule's HTTP 403 problem. Reference periods come from BLS observations;
  the collector requires a 20–45 day lag between reference month-end and release,
  a window shorter than any month, rejecting stale values and unusually delayed
  releases that need explicit period mapping. The calendar and value sources
  are retained separately in provenance.

The September 23 live run passed all 29 required series with no coverage gaps.
ISM August values were 54.6 manufacturing and 55.4 services; ADP August was +38K;
August retail excluding autos was +1.4%. BLS RSS returned 403, but the supported
BLS API supplied the 16 required BLS series, explicitly marked latest revised
vintage rather than originally announced prints.

`artifacts/fmp_retirement/20260923/macro_live02/` contains hashed raw responses,
observations, eight next-release notices, the candidate history and a
manifest. The candidate has 42,518 rows versus 42,489 baseline rows; all 37,712
previously populated rows retain their actual, consensus, previous, source and
vintage values. New observations carry no consensus or surprise. Forty-three focused
macro/retirement tests passed, including issuer identity, timestamp consistency,
signed values, changed HTML/table formats, DST and the holiday exception.

This candidate is **not yet the production macro writer**. All five previously
identified missing series now have working adapters. The wider FMP event catalog
is not represented by the 29-series coverage claim. Scheduled capture/revision
monitoring and production activation remain separate gates. FMP cancellation
has not been attempted. Workspace hygiene also reported three unrelated financing
research files created concurrently; they were left untouched and excluded from
these commits.
