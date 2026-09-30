# FMP replacement implementation and validation — 2026-09-22

Local candidate implementation; **production still uses FMP**. No subscription,
credential, pinned runtime, broker, Sheets, R2 object, or research recommendation
was changed. All captures and generated candidate datasets are under
`artifacts/fmp_retirement/20260922/`. These local artifacts are validation evidence,
not evidence of production freshness. Existing historical datasets are preserved.

## Implemented and exercised

| Replacement | Executable path | Observed result |
| --- | --- | --- |
| Government macro actuals | `scripts/build_official_macro_releases.py` | 24 distinct series from BLS API, BEA releases, Census retail and DOL claims; core coverage gate passes. |
| Primary earnings dates | `scripts/confirm_sec_earnings.py` | 14 explicitly announced dates and fiscal periods from 8-K Item 2.02, across ASTC/MSFT/NAVN/UEC; no parsing failures in available filings requested. |
| Alpha + SEC reconciliation | `scripts/refresh_earnings_calendar.py --confirmation-provider sec` | Offline Alpha replay succeeds with 147,734 rows and 8 strategy-policy differences versus the baseline; no FMP requests or publication. |
| Listing identity and eligibility | `scripts/validate_public_data_sources.py` | 13,257 listed securities; 5,061 pass conservative common-stock/listing checks before domicile and liquidity checks. |
| SEC annual financials | `fundamental/sec_statements.py` through the public-source validator | Eight issuers parsed into the existing metrics interface; six have five annual periods, JPM/NAVN have three in the accepted/mapped source coverage. Missing metrics remain explicit. |
| News, share structure and consensus | `public_market_sources.py` through the validator | Eight issuers returned metadata, float/outstanding shares, issuer-relevant news, and two annual EPS consensus rows. A separate bounded capture also returned two annual revenue consensus rows for all eight. |
| Secondary earnings history | Same validator | 153 past reported dates across eight issuers, excluding future/estimate-only rows; these are supporting evidence, not primary confirmation. |

Eight-company sample: AAPL, ASTC, BRK-B, JPM, MSFT, NAVN, O and UEC. Successful
fetches establish access and sample coverage, not universe-wide reliability or
complete accounting parity. Yahoo is an unofficial secondary source.

## Economic calendar behavior

The 24 series cover CPI/core CPI MoM/YoY, PPI/core PPI MoM/YoY, payrolls,
unemployment, hourly earnings MoM/YoY, weekly hours, private/government/manufacturing
payrolls, PCE/core PCE MoM/YoY, GDP second estimate, retail sales and initial/continuing
claims. GDP estimate vintages are separate event identities; GDP uses SAAR, payrolls
and claims use thousands, and inflation adjustment basis is explicit.

Every capture has a source URL, SHA-256 and observation/capture timestamps. The
BLS API is labelled **latest revised vintage**, never the originally announced
print. RSS is a separately labelled current headline snapshot. RSS worked during
the initial probe but was unavailable during the collector run; the supported API
fallback supplied all required BLS values. No original-release parity claim is made.

The candidate history has 42,513 rows versus 42,489 in the local preserved baseline.
All 37,712 previously populated rows retain their actual, consensus, previous,
source and vintage values unchanged. Newly captured official rows have no consensus
or surprise classification, so they cannot generate new consensus-based P12 events.
This does not remove historical P12 evidence.

The next release notices were parsed directly from the releases: GDP/PCE September
30 at 08:30 ET and retail October 15 at 08:30 ET. BLS observations must match a unique
official calendar row and cannot lag a newer due release. The existing official
calendar still supplies CPI/NFP/PPI and Fed event dates. Complete freshness checks
for the weekly claims holiday schedule and live schedule refresh remain to finish.

Unimplemented value coverage: ISM manufacturing/services, ADP, JOLTS, and retail
excluding autos. ISM's manufacturing request returned a CAPTCHA document and the
services request failed; those were not mistaken for data. JOLTS API observations
are accessible, but their official release-time mapping is not yet integrated.
The wider FMP event catalog has not been replaced; 24-series coverage is explicit.

## Earnings behavior

SEC confirmation requires an explicit statement that a press/news release **was
issued**, its announcement date, fiscal period ending date, accepted filing time,
SEC archive URL and content digest. Filing dates, fiscal period ends, scheduled
calls, future announcements, missing timestamps and ambiguous dates cannot become
confirmed earnings. Foreign 6-K and unfamiliar issuer prose remain gaps.

Date-only confirmation preserves previously captured EPS/revenue values and does
not calculate a surprise against potentially incomparable GAAP/adjusted estimates.
A matching fiscal period can replace an elapsed Alpha expectation or suppress an
already-announced period still appearing in the forward calendar. Zero reported EPS
is valid; an estimated date merely moving into the past is not confirmation.

This removes FMP calls **from the exercised offline candidate run only**. Scheduled
Alpha operation still defaults to FMP confirmation, overflow/new-name history
bootstrap, observer control and failure fallback until those paths are promoted
and their gates met. The prior near-term ASTC/BETA/NAVN/UEC policy differences still
require adjudication; eight changed flags are not eight approval-ready trades.

## Listing and research gaps

Of 855 symbols in the local existing master, 838 appear in the official listings
and 815 pass the initial common-stock/status checks. The 17 absent symbols are
SATS, OCTVV, ATAI, ESPR, FFIC, TOI, BCAR, XWIN, FONR, EQR, WBS, CWAN, TMHC, VSCO,
VRE, LEG and BBBY. A second Yahoo metadata check returned no current price/timestamp
for those symbols (eight returned quote-not-found). That is an exception list to
adjudicate, not proof that every company was delisted. Listing status alone does
not prove US domicile, liquidity, share-class identity or tradeability.

SEC mapping uses known acceptance times, USD/share units and actual annual start/end
dates. Unknown acceptance, quarterly/YTD facts, foreign-currency facts, incompatible
annual periods and conflicting observations are rejected. Each selected accounting
cell keeps accession/tag provenance. Capital expenditure outflows are negated for
the existing FCF calculation. Missing short-term debt is never assumed zero;
long-term debt alone is not relabelled total debt. SEC public float in dollars is
never treated as float shares.

All eight samples still have missing accounting metrics, particularly total debt,
EBITDA/leverage and issuer-specific concepts; bank/REIT/dual-share-class coverage
needs dedicated treatment. Neither the research coverage threshold nor REVIEW_READY
policy was weakened. Current/next fiscal-year Yahoo consensus is accessible but
does not replace FMP's full historical forecast surface or establish fiscal-date
alignment by itself. Analyst grades remain unnecessary and are removed only from
the previously prepared earnings runner; their live retirement awaits promotion.

## Reproduction

The macro PDF collector needs `pypdf==6.10.0` in addition to existing project data
dependencies (`requirements-official-data.txt`). Validation used an isolated copy
of the bundled library under artifacts; the pinned runtime was not modified.
Choose a new output directory on every run; existing captures are never cleared.

```powershell
python scripts/build_official_macro_releases.py --output-dir artifacts/fmp_retirement/new-macro-run --baseline data/macro_release_history.parquet
python scripts/validate_public_data_sources.py --output-dir artifacts/fmp_retirement/new-market-run --tickers AAPL MSFT JPM O --baseline-symbols data/symbol_master.parquet
python scripts/confirm_sec_earnings.py --output-dir artifacts/fmp_retirement/new-earnings-run --tickers MSFT NAVN ASTC UEC --filings-per-ticker 4
```

Use `--source-dir <previous-run>/raw` with the macro/market validators for offline
replay. Manifests verify hashes and retain captured timestamps. Early bootstrap
macro evidence without a manifest uses archived file timestamps; the validated
replay manifest records those explicitly as archived replays. SEC earnings proof
can be supplied to the existing Alpha offline replay with `--confirmations
<run>/confirmed_earnings.parquet --confirmation-provider sec`; this mode requires
`--no-upload --alpha-snapshot` and cannot silently switch a live producer.

Evidence runs: `macro_replay03`, `market_live01`, `market_replay02`,
`consensus_revenue01`, `listing_exceptions01`, `earnings_confirmations02`, and
`alpha_sec_replay01`. Source HTTP probes, intermediate failures and older candidate
runs are retained. The market validator now fetches revenue consensus too; older
eight-company replay inputs contain EPS consensus only, with the separately
archived revenue capture providing its live validation evidence.

## Gates remaining before cancellation

1. Finish the uncovered macro series and schedule/revision monitoring, including
   an unattended ISM source or an explicit decision to retire those two indicators.
2. Resolve listing exceptions and validate fresh metadata/liquidity across the
   full candidate universe, including new listings, not just the eight-name sample.
3. Complete the accounting mappings/estimate-period normalization and wire the
   new source clients into the scheduled research and discretionary consumers.
4. Complete earnings confirmation coverage, new-name historical bootstrap and a
   stop-on-error alternative to FMP fallback; resolve the near-term comparisons.
5. Promote reviewed producer/runner/workflow changes and observe clean scheduled
   runs with no required FMP calls. Only then approve cancellation separately.

Ongoing intraday collection already uses yfinance. Legacy FMP bars and grades are
retained; optional backfill/enrichment tools still need explicit retirement or
replacement. No billing or subscription action was attempted.

Source specifications: [BLS API](https://www.bls.gov/developers/api_signature_v2.htm),
[BLS core PPI series definitions](https://www.bls.gov/ppi/tables/final-demand-intermediate-demand-aggregation-index-seasonal-factors-2017-2021.htm),
[Nasdaq symbol-directory fields](https://nasdaqtrader.com/Trader.aspx?id=SymbolDirDefs),
[SEC developer resources](https://www.sec.gov/about/developer-resources),
[yfinance interfaces](https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.html).

Verification: 202 tests passed across the new adapters and existing earnings,
macro, fundamental and discretionary consumers. Git whitespace and workspace
hygiene checks passed. Two non-failing deprecation warnings remain (the existing
exchange-calendars minute alias and pandas concatenation of nullable earnings
rows). Live candidate captures, primary-date collection and offline replays above
were also checked; no production runtime execution was claimed.
