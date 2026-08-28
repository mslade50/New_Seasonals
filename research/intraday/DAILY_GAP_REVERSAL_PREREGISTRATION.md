# Daily OHLC asymmetric gap-reversal screen — preregistration

## Status and purpose

This document freezes a research-only, broad-universe screen before the full
price parquet is evaluated. It is a cheap prioritization test for a new v1
hypothesis. It does not modify the strategy book, stage or place an order,
write to production, schedule itself, or authorize automatic promotion.

Daily OHLC is intrinsically optimistic for this hypothesis. The open must be
observed before the limit can be computed and submitted, while a daily low or
high does not say whether the limit touch occurred before or after submission.
Consequently, a positive daily result can only justify a causal intraday rerun;
it cannot override or substitute for one.

## Frozen input contract and universe

- One explicit, local, long-format parquet with columns
  `ticker,date,Open,High,Low,Close,Volume` (case-insensitive).
- `--as-of` is required and means the last completed session admitted to the
  study. Rows after it are excluded before feature construction.
- `--evaluation-start` is optional. Earlier valid rows remain available for
  lagged-feature warmup, while candidates and returns begin only on that
  canonical NYSE session. This permits a recent same-adjustment-vintage
  diagnostic without rebuilding indicators on truncated history.
- The runner accepts either an explicit frozen universe file or an explicit
  `--all-tickers` choice. Universe files are deduped by normalized ticker.
- The current `CSV_UNIVERSE` eligibility exclusions are copied and frozen here
  rather than imported from production: exclude `CBZ`, `THS`, suffix `=F`,
  suffix `-USD`, and caret tickers other than `^GSPC` and `^NDX`.
- The input parquet, optional universe file, and sorted pre-filter,
  post-filter, and available universes are SHA-256 fingerprinted. Counts and
  the exact filter are recorded in the manifest.

The current universe and current membership create survivorship bias. No
claim of point-in-time membership is permitted.

## Point-in-time features and eligibility

For ticker/session T:

- prior close is T−1 close;
- daily true range is `max(High-Low, |High-prior Close|, |Low-prior Close|)`;
- ATR14 is the simple mean of 14 completed daily true ranges, shifted one full
  session, so the value used on T is known through T−1;
- prior 20-session median daily dollar volume is the median of `Close*Volume`
  over completed sessions, shifted one full session;
- require prior close at least $5, prior median dollar volume at least $25m,
  at least 14 prior sessions, and a finite positive lagged ATR14;
- current-session volume never affects current-session eligibility.
- session dates and predecessor adjacency use the repository's deterministic
  `trading_calendar.TRADING_DAY` NYSE sequence, including its frozen ad-hoc
  closure list. A row is ineligible unless its previous ticker row is exactly
  the previous canonical session. A malformed or missing predecessor taints
  that row, and ATR must rebuild a clean 14-session window rather than bridge
  the hole.

The completed-session cutoff is applied before duplicate auditing. Duplicate
ticker/date observations inside the admitted window fail the run before any
invalid row can be dropped; there is no discretionary duplicate winner.
Malformed rows are preserved in a rejection audit and excluded.

A conservative raw/adjusted discontinuity heuristic excludes an
open/prior-close ratio below 0.20 or above 5.0, or a ratio at least 20% from
one and within 3% of the frozen common split-factor list. This is not an
authoritative corporate-action source and may exclude real extreme gaps.

## Co-primary arms

Both arms are fixed before the broad run:

1. `gap_down_long_open_minus_0p25atr`
   - candidate when Open < prior Close;
   - buy limit = Open − 0.25 × lagged ATR14;
   - a daily range touch is recorded when Low <= limit;
   - exact-limit fill; same-day Close exit.
2. `gap_up_short_open_plus_0p75atr`
   - candidate when Open > prior Close;
   - short limit = Open + 0.75 × lagged ATR14;
   - a daily range touch is recorded when High >= limit;
   - exact-limit fill; same-day Close exit.

No stop, target, earnings filter, news filter, borrow model, or parameter
selection is introduced in this screen.

## Primary small-account endpoint

The two arms are tested separately and are co-primary.

- Before any fill information is used, rank each arm's candidates for each
  session by `abs(Open-prior Close) / lagged ATR14`, descending, then ticker
  ascending as the deterministic tie-breaker.
- Select at most three candidate orders per arm/session.
- Allocate exactly one-third of arm notional to each slot. Fewer than three
  candidates leave unused slots in cash. An unfilled selected order returns
  zero.
- A filled slot earns the signed exact-limit-to-close return minus 10 bps.
  Costs apply only to fills.
- The per-arm daily endpoint includes every study session from the first
  eligible candidate date through the completed-session cutoff; no-order days
  are zero.
- Report two-sided session-level t tests, deterministic session bootstrap 95%
  confidence intervals, and Newey-West/Bartlett HAC mean tests and confidence
  intervals. HAC p-values are the robust primary inference. Apply Holm
  adjustment across the two co-primary 10-bps arm tests.

The fixed 50/50 combination of the two arm endpoints is diagnostic only and
cannot rescue either primary arm.

## Prespecified diagnostics

- cost grid: 5, 10, 15, 20, and 30 bps, charged only to fills;
- all-filled conditional economics and fill rate (descriptive, not the
  small-account endpoint);
- material-gap candidate thresholds of 0.25, 0.50, and 1.00 ATR, with the
  zero-threshold primary shown alongside them; each view reranks at the open;
- annual, leave-one-year-out, and fixed-rule rolling five-calendar-year
  history / one-year test tables with no refit;
- ticker contribution/concentration for selected primary fills;
- coverage, eligibility, universe, and selected-order audits.

## Interpretation gate

- Non-positive 10-bps primary mean: reject the arm at this screen.
- Positive but non-significant after Holm, negative at 20 bps, unstable across
  years, or highly concentrated: watch only; do not promote.
- Positive, Holm-significant, cost-robust, and stable: advance only to a new
  causal intraday test with order latency and opening-bar ambiguity resolved.

No possible daily-OHLC outcome constitutes execution validation.

## Known unmodeled risks

The screen has no quote/spread data, opening-auction timing, queue priority,
partial fills, latency, halts, limit-up/down state, news or earnings state,
borrow/locate availability, integer-share sizing, market impact, or broker
account constraints. Adjusted OHLC can also conceal price-basis seams; the
discontinuity heuristic is intentionally only a conservative guard.
