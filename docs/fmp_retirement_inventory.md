# FMP retirement inventory — September 22, 2026

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
| Upcoming earnings | `scripts/build_earnings_calendar.py`; new `scripts/refresh_earnings_calendar.py` | Resolve near-term exceptions, activate/test Alpha and complete the independent comparison period. |
| Recent actual earnings dates/values and newly covered history | New Alpha path deliberately retains FMP confirmations, history bootstrap and failure fallback | Implement and validate a non-FMP confirmation/history path, including late reports, revised dates and zero actuals. An expected date moving into the past is not confirmation. |
| Independent earnings control | New artifact-only FMP reference refresh | Retire only after the replacement's monitored reliability is accepted. Preserve all evidence. |
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
