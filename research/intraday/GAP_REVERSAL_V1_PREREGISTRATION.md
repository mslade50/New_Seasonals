# Gap-Reversal v1 Preregistration

## Status and purpose

This document freezes the first evaluation of the gap-reversal v1 experiment
before any full-sample result is computed or inspected. The experiment is an
isolated, research-only screen. It cannot stage an order, modify the strategy
book, write to R2, schedule a job, deploy anything, or promote itself.

The two co-primary arms are separate hypotheses. A combined long/short view is
diagnostic only and cannot rescue a failed arm.

## Frozen input and coverage boundary

The intended first run reuses the read-only local copy of the frozen
`intraday-r2-snapshot.v1` dated 2026-08-27 and the explicit liquid single-stock
request in `liquid_single_stock_pilot_2026-08-27.csv`. The CLI requires explicit
local paths and can enforce the snapshot-index, universe-file, and sector-map
SHA-256 values. Missing, excluded, stale, and malformed inputs remain visible.
No result may be described as an all-overflow or 1,025-name result.

Prices are raw/unadjusted regular-session 15-minute bars. Frozen dollar limits
are therefore minted and tested on the same raw basis.

## Co-primary arms

On every otherwise eligible full session, any strictly signed overnight gap
qualifies; there is deliberately no minimum gap in the literal primary rule.

1. `gap_down_open_minus_025atr_long_v1`
   - official 09:30 open is below the immediately prior scheduled 15:45 close;
   - buy limit = `09:30 open - 0.25 * ATR(14)`;
   - side = long.
2. `gap_up_open_plus_075atr_short_v1`
   - official 09:30 open is above the immediately prior scheduled 15:45 close;
   - sell-short limit = `09:30 open + 0.75 * ATR(14)`;
   - side = short.

`ATR(14)` is the simple mean of the last 14 valid raw daily true ranges. Daily
true range is the maximum of high-low, absolute high-prior-close, and absolute
low-prior-close. The series is shifted one completed session; session T can use
information only through T-1. No current-session range enters its own limit. A
raw split/discontinuity day is detected against the previous valid session
close and its true range is masked, leaving ATR unavailable for the following
14 sessions and restoring it on session +15. The current split-gap signal is
still filtered against the exact prior scheduled 15:45 close.

## Execution clock

- The 09:30 opening bar is excluded from the primary fill path because its
  within-bar sequence relative to learning the official open is unknowable.
- Orders activate at the scheduled 09:45 bar. If its open is already through
  the limit, the primary simulation still fills at the exact limit; no favorable
  price improvement is credited.
- Otherwise, the first scheduled bar from 09:45 through 15:30 whose low touches
  a long limit or whose high touches a short limit fills at the exact limit.
- Unfilled orders expire before the 15:45 bar. Filled positions exit at the
  scheduled 15:45 bar close.
- The current 09:30 source bar, every scheduled bar from activation through
  exit, the immediately prior 15:45 close, and the lagged ATR must be present
  and valid. Current required bars must have positive volume. Ranking is frozen
  before any later current-session tape outcome is used. If a selected name's
  later tape is missing or invalid, its slot is cash/rejected and a lower-ranked
  name cannot substitute. Any such selected tape failure blocks an advance
  label and is reported explicitly.
- An explicitly optimistic, non-primary sensitivity may count a threshold touch
  in the 09:30 bar and fills that touch at the exact limit.

Fifteen-minute bars cannot establish within-bar sequence, queue priority,
partial fills, spread, borrow, or halts. Short observations do not imply locate
availability.

## Point-in-time gates

Eligibility is frozen to the existing T-1 research proxy: prior price at least
$5, trailing 20-session median dollar volume at least $25 million, trailing bar
completeness at least 95%, and at least 10 completed sessions. The current day
cannot qualify itself. The existing raw-price discontinuity heuristic excludes
common split-factor and extreme open/prior-close ratios. Sector metadata is
required for concentration reporting, but no sector return is used in the rule.

## Primary endpoint and multiplicity

The small-account primary endpoint is evaluated separately for each arm at 10
bps round-trip cost using a fixed three-slot candidate-order portfolio:

1. At 09:30, rank that arm's candidates by descending `abs(gap / ATR)` and then
   ticker for a deterministic tie-break.
2. Reserve three equal notional slots and retain only the top three candidates.
3. A retained candidate that does not fill earns zero; a lower-ranked candidate
   cannot replace it after the fact.
4. Each fill contributes one third of its return. Round-trip cost is charged
   only to filled slots. Unused slots remain cash at zero return.

The inference denominator includes every exact canonical SPY full session on
which at least one loaded ticker is observable at signal time: T-1 eligibility,
an exact immediate prior 15:45 close, a valid lagged ATR(14), and a valid 09:30
open. Warmup and no-observable-universe sessions are excluded. Every included
session appears for both arms; an arm with no strictly signed candidate that
day contributes zero, as do unused and selected-but-unfilled slots.

The null is zero mean daily three-slot return. Report a two-sided day-cluster
t-test and deterministic day-block bootstrap confidence interval. Holm-adjust
the two 10-bps arm tests as one family. Daily equal-notional returns conditional
on all actual fills are secondary and cannot replace the slot-primary result.

## Prespecified robustness and diagnostics

- Round-trip costs: 5, 10, 15, 20, and 30 bps.
- Fixed candidate-order slot counts K = 1, 3, 5, and 10, always ranking before
  fills and leaving unused slots in cash.
- Literal any-signed-gap is the primary rule. Prespecified descriptive material-
  gap views retain `abs(gap / ATR) >= 0.25`, `0.50`, and `1.00`. The 0.50-ATR
  view is labeled the economically cleaner version, but none is a winner-pick
  and none can overwrite the literal primary result.
- Calendar-year, leave-one-year-out, and rolling five-calendar-year history to
  one-calendar-year test diagnostics for the fixed three-slot primary series.
- Ticker and sector concentration uses only actual fills among each arm/day's
  pre-fill-ranked top three candidates at 10 bps. Lower-ranked fills are
  excluded, and each included fill retains the fixed one-third slot weight.
  Fill rate and long/short arm economics are also reported.
- The opening-bar-touch sensitivity is optimistic, secondary, and cannot
  advance either arm.
- A combined long/short portfolio is diagnostic only.

## Research-priority interpretation

An arm may advance to deeper research only if its 10-bps three-slot mean is
positive with Holm-adjusted p-value below 0.05, remains positive at 20 bps,
shows broad chronological stability, and is not dominated by one year, ticker,
or sector. Failure of an arm is reported directly; the other arm or combined
portfolio cannot rescue it. Any next threshold, filter, or execution change is
a newly named experiment requiring new holdout evidence.

## Known limits

The frozen liquid universe is survivorship-biased and the sector map is static.
The data has no authoritative news, earnings, corporate-action, spread, queue,
borrow, or halt history. The split-factor filter is a conservative heuristic,
not a corporate-action source. Fixed slots model scarce candidate capacity,
not integer shares, commissions, margin, settlement, or broker permission.
