# FMP retirement inventory — September 22, 2026

## Status on 2026-10-01

FMP has returned HTTP 429 since 2026-09-30 and the renewal was cancelled. Any
code path below that still calls FMP now fails when run.

Retired from production:

- Earnings calendar: Alpha Vantage only since the 2026-10-01 10:11 ET runtime
  release (tag `automation-runtime-2026-10-01.earnings-alpha-only`). No FMP
  bootstrap, confirmation or fallback. See
  [the release](earnings_alpha_only_release_2026-10-01.md).
- Economic releases: `scripts/refresh_macro_releases.py` collects 29 official
  series with `fmp_requests` 0. The wider FMP event catalog is history only.
- Discretionary Focus and analyst grades: retired 2026-09-23
  ([progress note](fmp_retirement_2026-09-23.md)).
- Issuer review: `scripts/prepare_earnings_issuer_review.py --alpha-r2`
  (`71e44658`) reads the day's R2 Alpha snapshot and needs no FMP.

Still calls FMP, not scheduled, fails when run by hand:

- `scripts/build_symbol_master.py` through `rebuild_overflow_universe.yml`
  (manual dispatch, `FMP_API_KEY`, FMP company screener). The symbol universe
  stays frozen at 2026-06-05 until a replacement exists.
- `fundamental_sleeve_research.yml` (manual, research only) through
  `fundamental/fmp.py`.
- `scripts/compare_macro_shadow.py`, the read-only FMP versus official macro
  comparison.
- `scripts/compare_earnings_shadow.py`, which requires an FMP baseline, so the
  Alpha versus FMP earnings comparison is unavailable.
- The optional utilities (`enrich_episodic_pivot_history.py`, `diagnose_stale_tickers.py`,
  `pull_intraday_validator.py`) still reference FMP. Legacy
  `scripts/build_earnings_calendar.py` and `scripts/build_macro_releases.py`
  also still reference FMP and are no longer the production writers.

The table below is the September 22 inventory, kept as history. Its "Required
before cancellation" column was written before the cancellation.

## September 22 inventory

This is a source/dependency inventory, not a claim every manual tool is currently
scheduled. No subscription, API credential, research workflow or macro producer
was retired. The earnings cutover is prepared but not active; see
[cutover status and validation](earnings_alpha_cutover.md).

Same-day implementation follow-up: [replacement validation](fmp_replacement_validation_2026-09-22.md).
Official macro, Nasdaq/Yahoo, SEC statement and primary earnings-date adapters now
exist and have live candidate evidence. The 24-series macro gate and an Alpha +
SEC offline replay pass. Production consumers/fallbacks still require migration;
the linked report lists the remaining coverage and activation gates.

| Dependency | Current source surface | Required before cancellation |
| --- | --- | --- |
| Upcoming earnings | `scripts/build_earnings_calendar.py`; new `scripts/refresh_earnings_calendar.py` | Resolve near-term exceptions, activate/test Alpha and complete the independent comparison period. **Superseded 2026-10-01: Alpha only in production.** |
| Recent actual earnings dates/values and newly covered history | New Alpha path deliberately retains FMP confirmations, history bootstrap and failure fallback | Implement and validate a non-FMP confirmation/history path, including late reports, revised dates and zero actuals. An expected date moving into the past is not confirmation. **2026-10-01: FMP confirmations, bootstrap and fallback removed with no replacement. Vanished events are kept as `schedule_unverified`; no actuals, EPS or surprise values are populated.** |
| Independent earnings control | New artifact-only FMP reference refresh | Retire only after the replacement's monitored reliability is accepted. Preserve all evidence. **2026-10-01: unavailable, since `scripts/compare_earnings_shadow.py` needs an FMP baseline.** |
| Economic releases | `scripts/build_macro_releases.py`, Actions backup and local macro job | Integrate official release dates/times and reported values; preserve release vintages and revisions. Remove new consensus/surprise requirements and P12 consensus-based research. |
| Dynamic symbol universe | `scripts/build_symbol_master.py`, `rebuild_overflow_universe.yml` | Replace the FMP company screener and validate listing/status, share classes and universe coverage. Freezing today's symbols indefinitely is not a replacement. |
| Fundamental financials/metadata | `fundamental/fmp.py`, `fundamental/config.py`, fundamental research workflow | Replace or explicitly retire profiles, income/balance/cash-flow statements, metrics, ratios and analyst estimates. Analyst estimates are distinct from analyst grades. Any sleeve implementation must follow its mandatory research skill. |
| Discretionary news | `FMPNewsClient` in `scripts/build_discretionary_focus.py` and its runner | Replace/retire the news fetch and remove the required FMP key only after the replacement is verified. |
| Optional historical/enrichment utilities | `enrich_episodic_pivot_history.py`, `diagnose_stale_tickers.py`, `pull_intraday_validator.py` | Replace or mark unavailable the FMP enrichments, symbol-change lookup and intraday validation/backfill paths. Do not erase captured history. |
| Analyst grades | Old pinned earnings/grades job; collection removed in prepared source | Promote the prepared runner/workflow changes so the live job and output requirements stop expecting grades. Retain historical files. |

Ongoing intraday collection already uses `scripts/update_intraday_yfinance.py`;
its workflow explicitly requires no FMP key. No Alpha intraday subscription is
needed for this earnings migration. Existing FMP-funded historical bars remain
useful and should be preserved.

For economics, the existing [official-source assessment](economic_calendar_replacement.md)
maps BLS, BEA, Census, DOL and Federal Reserve releases. Source access checks are
not a complete unattended producer validation. BLS schedule access had a 403 and
needs a supported fallback. ISM requires its own access/automation decision.
The user's scope is release dates/times and reported values, not forecasts or
surprises. Keep observation period, release timestamp, units, adjustment basis
and initial/revised values separate.

Recommended sequence after earnings activation: economic release producer;
symbol-universe replacement; research/news dependency decisions; then a scheduled
run audit demonstrating no required FMP calls or fallback dependence. Only then
seek explicit approval to cancel the subscription. Cancellation and credential
removal are not authorized by this implementation task.
