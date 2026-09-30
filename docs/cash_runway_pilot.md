# Free SEC cash-runway pilot

This is a local research screen, separate from the trading book and the long-only
fundamental decision workflow. It creates no recommendations, orders, portfolio
state, emails, uploads, schedules or production-site payloads. The deliberate
100-symbol sample in `config/cash_runway_pilot.json` includes capital-consuming
operating businesses, biotech/pre-revenue names and positive-cash-flow controls.
It is not a statistically representative, historical or tradability universe.

## Run and replay

Requires the existing Python environment (`requests`, `pandas`, `beautifulsoup4`,
`python-dotenv`) and the existing `FUNDAMENTAL_SEC_USER_AGENT` contact identity.
SEC APIs and the Nasdaq listing directory have no data subscription charge.
No financial-data vendor key is read by this pipeline.

Use a **new** output directory for every run; existing output is never cleared:

```powershell
python scripts/build_cash_runway_watchlist.py --output-dir artifacts/cash_runway/new-run --documents --max-filings 12
```

Use `--tickers VUZI FCEL PLUG` for a bounded subset. Without `--documents`, the
pipeline fetches financial facts only and visibly marks financing unreviewed.

Offline replay preserves the original research cutoff, checks SHA-256 capture
hashes, and never fetches a missing file from the network:

```powershell
python scripts/build_cash_runway_watchlist.py --source-dir artifacts/cash_runway/prior-run --output-dir artifacts/cash_runway/new-replay --documents --max-filings 12 --reviews artifacts/cash_runway/manual_source_checks_20260922.json
```

`--reviews` is optional. Reviews must match issuer, balance date and research
date, cite source documents, and specify expected USD amounts. They annotate
facts rather than replacing them. A numerical mismatch is visible and makes the
CLI return a failure status. The September 22 audit checks ten companies and six
values each, including missing capex; a $1,000 RXRX disclosure discrepancy has a
documented tolerance. Review notes must be refreshed for a later date or period.

Outputs: `cash_runway.html`, `watchlist.csv`, full-provenance `watchlist.json`,
`manifest.json`, `captures.json`, immutable captured bytes under `raw/`, and
`manual_reviews.json` when supplied. The standalone HTML has search, horizon
filters, filing links, period arithmetic, financing excerpts and audit notes.

## Calculation and source boundaries

- Reported liquidity is unrestricted cash plus mapped current investments, or a
  combined cash/short-term-investment fact counted once. Explicitly restricted
  cash is excluded. A missing investment fact produces a **cash-only lower
  bound**, not an assertion that investments are zero. Long-term securities are
  excluded from the automated total; material examples appear in audit notes.
- Operating monthly burn is `max(0, -OCF) / months`. Capex-inclusive burn is
  `max(0, PP&E/productive-asset cash capex - OCF) / months`. Runway is reported
  liquidity divided by positive monthly burn. Zero burn has no finite runway;
  missing capex never becomes zero. Other capitalized spending is not included.
- Both latest-three-month and latest-six-month intervals use actual fiscal
  start/end dates. YTD figures are differenced; adjacent quarters can be summed
  across a fiscal year boundary. Financials need not have any revenue history.
- Only US-GAAP USD facts with a known acceptance timestamp before the cutoff
  qualify. Later restatements and unknown acceptance times are excluded. This
  current sample is **not** a complete historical dissemination-time backtest.
  Latest filings are sourced from the SEC recent-submissions section; old history
  outside that coverage is a gap, never imputed.
- Financial/real-estate SICs and unmapped foreign taxonomies are unavailable.
  Listing absence or symbol changes remain coverage rows. Abnormal Nasdaq
  financial status is retained as a research warning because distressed issuers
  matter to the question. No liquidity, share-borrow or price filter is applied.
- The latest financial filing and up to twelve later relevant filings are read
  for financing clues. Documents may discuss prior transactions or unused
  capacity. Truncation/fetch failures stay visible. This is discovery, not an
  exhaustive newswire feed or fully reconciled current-cash bridge.
- Reported runway is **as of the balance date**, not today. Closed raises already
  inside that balance must not be added again. Conditional awards, warrants,
  shelf capacity and escrow are not available cash. Even verified later gross
  proceeds need costs, other cash movements and spending before a current-cash
  estimate could be defensible.

Initial horizon buckets (<3, 3–6, 6–12 and 12+ months) are uncalibrated research
flags. The first sample surfaced one-time operating receipts, restricted-cash
headline inflation, extra project/patent spending and conditional government
awards. These are recorded beside the relevant calculations.

## Verification

Run `python -m pytest tests/test_cash_runway.py -q` for unit/edge cases. The live
pilot must then be replayed against the saved manual checks, and the HTML viewed
in a local headless browser at desktop and narrow widths. Keep screenshots and a
report-digest attestation under the same artifact run. No local artifact is
production freshness evidence.

References: [SEC public APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces),
[SEC filing dissemination timing](https://www.sec.gov/submit-filings/filer-support-resources/how-do-i-guides/determine-status-my-filing),
[Nasdaq listing definitions](https://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs).
