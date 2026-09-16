# Alpha Vantage earnings shadow comparison

This is a separate observer. Production continues to use FMP. Nothing in this
tool writes production data, uploads, stages orders, or switches providers.

Run from this worktree with the existing Python environment:

```powershell
python scripts/compare_earnings_shadow.py --config-root 'C:\Users\McKinley Slade\dev\New_Seasonals'
```

The key is read from `ALPHA_VANTAGE_API_KEY` in the environment or the config
root's `.env`. Never include it on the command line. The tool makes one bulk
Alpha Vantage `EARNINGS_CALENDAR` request using the three-month horizon. The
comparison window is 60 calendar days to avoid the feed's outer boundary.

Inputs are the existing main and, if present, overflow earnings parquets under
the config root. Date unions match `earnings_filter.load_earnings_dates_map`.
Conflicting EPS estimates for an identical ticker/date are excluded from EPS
comparison; they do not remove the date from the blackout comparison.

Each run creates a unique timestamped directory under
`artifacts/earnings_shadow/authenticated/` containing the raw Alpha CSV,
normalized Alpha calendar, FMP comparison snapshot, per-ticker comparison CSV,
JSON summary and Markdown report. Revisions are matched by symbol and fiscal
period against the previous successful snapshot. They include additions and
disappearances, which can reflect a rolling horizon rather than a provider error.
Failed runs retain a failure receipt and do not replace successful evidence.

Metrics distinguish the regular CSV universe, liquid subset, and additional
overflow tickers. A symbol without an upcoming date in either feed is unknown
coverage, not agreement. The blackout metric evaluates hypothetical signal
dates across the next 21 NYSE sessions using the existing +/-10 trading-day
rule. It shares past FMP dates between both paths and substitutes only future
dates, so it is NOT evidence that Alpha can replace historical earnings data.
EPS differences are diagnostic only: currency/accounting-basis equivalence
cannot be established from the existing FMP cache.

Offline replay is supported without a network request:

```powershell
python scripts/compare_earnings_shadow.py --config-root <root> --alpha-csv <saved-alpha_raw.csv> --as-of 2026-09-16
```

Offline and public demo evidence are stored separately from authenticated runs.
An offline replay uses the current FMP files supplied in the config root; for
historical reproduction supply that day's preserved FMP baseline as well.

## Trial and decision

Collect once per weekday through at least one earnings cycle (roughly 6-8 weeks).
An active Codex heartbeat, `compare-alpha-vantage-earnings-with-fmp`, follows up
in the original task weekdays at 06:30 America/New_York. It uses this isolated
worktree, reuses any successful authenticated run already captured that day,
and reports only meaningful changes, failures, required action, or completion.
Its prompt calls for a final assessment and pausing on or after November 11,
2026. Collection depends on the local Codex app being able to run the heartbeat.

Review regular-universe coverage, date revisions and all disagreements that
alter blackout decisions, with company investor-relations announcements as
the adjudicator. FMP is a comparator, not ground truth. There is deliberately
no automatic acceptance threshold or provider switch; perfect vendor agreement
could still conceal a shared incorrect estimate.

File age is recorded but is not proof of upstream freshness. Before concluding
that one provider is wrong, check successful producer receipts and any degraded
coverage. Preserve existing history throughout a future migration.

Matching forward dates alone does not replace EPS/revenue actuals, revenue
estimates, historical revisions, analyst grades, economic releases or research
enrichment. Any retirement of these dependencies is a separate reviewed change.

## First observation, 2026-09-16

Authenticated snapshot `20260916T105335879001Z`: 684 of 820 upcoming FMP events
in the regular universe matched exactly (83.4%). There were 105 tickers with
different date sets, 31 missing from Alpha, and two with Alpha-only events.
The full date union including additional overflow names matched 686 of 834.
This does not establish either provider's overall accuracy.

FedEx provided an initial primary-source check: its investor-relations calendar
listed October 28, 2026, matching FMP, while this Alpha snapshot omitted the
event. Source: https://investors.fedex.com/ (checked 2026-09-16).

Recent runtime evidence also contradicts the claim that FMP is only used for
earnings: the 2026-09-15 postclose log records successful earnings/analyst grades
and macro-release jobs. The discretionary log shows an attempted FMP-dependent
financial enrichment/news build. These should be accounted for before cancelling.

Verification: `python -m pytest tests/test_earnings_shadow.py -q -p no:cacheprovider`.
When using `--basetemp`, select a new path under `artifacts/` on every run so
pytest never removes an existing test directory.
