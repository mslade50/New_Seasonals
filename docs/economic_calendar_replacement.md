# Economic calendar replacement assessment — September 16, 2026

The user requires release dates/times and reported values, but not consensus
forecasts, surprise labels or analyst grades. Recommend official sources for
government releases rather than another paid calendar subscription.

## Current implementation

`scripts/build_macro_calendar.py` already builds `data/macro_events.csv` from
Federal Reserve and BLS schedules plus computed market-calendar events. The
local file spans 2000–2027. Its BLS path uses cached schedules and explicitly
handles historical shutdown revisions. It does not supply reported values.

`scripts/build_macro_releases.py` currently fetches FMP economic-calendar
history, normally from 45 days before today through today. It stores actual,
consensus, previous and surprise data in `macro_release_history.parquet` and
preserves already-captured reported values. It is not merely an upcoming
calendar. `scripts/build_context_state.py` consumes selected events for today's
release display and the P12 above/below-consensus follow-through research.

## Replacement mapping

| Data | Dates and times | Reported values |
| --- | --- | --- |
| CPI/core CPI, PPI/core PPI, payrolls, unemployment | BLS release calendar | BLS releases/API |
| PCE/core PCE and GDP | BEA release calendar | BEA releases/API |
| Retail sales; housing if retained | Census economic-indicator calendar | Census releases/API |
| Weekly initial claims | Department of Labor announcements | DOL weekly claims release; FRED as a secondary series source |
| FOMC decisions/minutes | Federal Reserve meeting calendar | Fed policy statements/rate releases |
| ISM manufacturing/services PMI | ISM's own release calendar | ISM's published headline reports; requires a separate access/automation check |

Official government data avoid another paid vendor bill. ISM is a private
publisher, not a government open-data API; do not assume a licensed historical
or redistribution feed is included simply because headline releases are public.

Sources checked:

- BLS calendar and subscription: https://www.bls.gov/schedule/
- BLS API: https://www.bls.gov/developers/
- BEA schedule: https://www.bea.gov/news/schedule
- BEA API: https://apps.bea.gov/api/signup/
- Census indicators: https://www.census.gov/economic-indicators/
- DOL claims: https://www.dol.gov/ui/data.pdf
- Federal Reserve meetings: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
- FRED release dates: https://fred.stlouisfed.org/docs/api/fred/releases_dates.html
- FRED observations: https://fred.stlouisfed.org/docs/api/fred/series_observations.html
- ISM calendar: https://www.ismworld.org/supply-management-news-and-reports/reports/rob-report-calendar/

## Access checks from this machine

The BEA iCalendar URL returned HTTP 200 and calendar content; the Census
indicators page returned HTTP 200. The BLS time-series API returned HTTP 200
with JSON. The direct BLS calendar ICS request returned HTTP 403. Therefore
BLS schedule refresh cannot be claimed unattended-ready: retain the existing
verified calendar and design a supported fallback before migration.

These were read-only probes, not an end-to-end collector validation. FRED can
provide a common secondary values interface, but availability can lag the
issuing agency and its dates do not replace precise release-time verification.

## Migration requirements

1. Keep the existing historical files. Store new official-source records in a
   separate comparison area first; preserve source and collection timestamps.
2. Match reference period, seasonal adjustment, units and transformation:
   levels, month-over-month changes, year-over-year changes and annualized GDP
   are different quantities. A monthly observation date is not its release date.
3. Capture the initial release and later revisions separately. Do not relabel
   revised API history as the number originally announced.
4. Remove consensus requirements from new observations and suppress P12's
   above/below-consensus studies. Do not replace unknown consensus with zero.
5. Validate the release clock, expected-event coverage and reported values
   alongside FMP before switching the macro producer and its health checks.
6. Remove analyst-grade collection from both the pinned local earnings job and
   GitHub fallback when retiring unused FMP dependencies; retain cached history.
   No active strategy-grade filter was found in the checked production config.

This task assesses economic replacements. It does not replace the live macro
producer, disable analyst-grade collection, or cancel FMP. The immediate live
change is limited to the separate earnings-comparison scope and its heartbeat.
