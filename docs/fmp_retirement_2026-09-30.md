# FMP retirement — September 30, 2026

The FMP subscription is cancelled. This change removes every scheduled FMP call.
It is prepared source: production changes only when the pinned local-primary
runtime (and its GitHub fallback ref) is promoted to a release containing it.

## Earnings calendar (`scripts/refresh_earnings_calendar.py`)

- **Forward dates:** Alpha Vantage `EARNINGS_CALENDAR`, one shared daily request
  (unchanged).
- **History:** frozen. The ~146k FMP-era rows in the canonical calendar are never
  refetched or rewritten. The universe is frozen too; names without history only
  receive forward Alpha dates.
- **Confirmation of elapsed events:** SEC EDGAR 8-K Item 2.02 filings
  (`sec_earnings_dates.py`), dates only. The announcement date is the filing's
  `reportDate` when the 8-K carries only results items (2.02/7.01/9.01);
  otherwise the acceptance day, because `reportDate` is the *earliest* event in a
  combined filing (FDS 2026-09-30 furnished results with a 2026-09-24 bylaw
  change). Requires `SEC_USER_AGENT` or `FUNDAMENTAL_SEC_USER_AGENT`; if absent or
  SEC is down, the run still publishes with events left unverified.
- **No fallback provider.** Guards that previously switched to FMP now resolve
  conservatively (`build_candidate(unconfirmed="retain")`):
  - an elapsed Alpha expectation without SEC proof stays in history as
    `unverified_elapsed` (the post-earnings blackout side still sees it) and
    remains eligible for a later SEC confirmation, which moves it to its real date;
  - a release-day event or near-term Alpha period that disappears from Alpha
    keeps its prior date (`schedule_basis = retained_unconfirmed`).
  No blackout is removed without evidence.
- **Alpha outage:** nothing is published; the prior canonical object stays.
  Consumers accept a calendar up to `MAX_STALE_TRADING_DAYS = 2` NYSE sessions
  old (was 1) before `earnings_filter` refuses it.
- New receipt fields: `sec_confirmations`, `unconfirmed`,
  `unverified_elapsed_total`; `status` is `ok` or `ok_with_unverified`.
- Surprise/YoY columns exist only in frozen history. All `use_*_surp_filter`
  flags are off, so no live rule reads them.

Dry run against production inputs for 2026-09-30 (today's Alpha snapshot, live
SEC): Alpha selected, 147,781 rows, 6/7 elapsed expectations SEC-confirmed
(SA unmatched, not yet elapsed), **0 blackout/sizing decision differences**
versus the published calendar.

## Removed

- `scripts/build_earnings_calendar.py` (legacy FMP builder, bootstrap, fallback
  and `--reference-only` observer baseline) and its tests.
- `scripts/build_macro_releases.py` (legacy FMP macro builder; production uses
  `refresh_macro_releases.py`) and `scripts/compare_macro_shadow.py`.
- `FMP_API_KEY` from the earnings workflow and the supervisor job's required env
  (now `ALPHA_VANTAGE_API_KEY`); the Focus workflow's FMP earnings step.

## Still references FMP (not scheduled)

- Discretionary Focus (`build_discretionary_focus.py`, `discretionary_focus.yml`):
  retired 2026-09-23, workflow disabled with an unconditional false guard.

## Operator follow-ups

- The **Monitor earnings and economic cutover** heartbeat compares against an
  independent FMP reference (`--reference-only`, `compare_macro_shadow.py`).
  Both are gone; retire or rewrite that monitor before promoting this release.
- Promote the runtime pin and GitHub fallback ref together, as before.
- Rollback: restore the previous runtime pin; the canonical object format is
  unchanged, so no data rollback is needed.
