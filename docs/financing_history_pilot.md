# Historical financing research pilot

This is a local research experiment, separate from the daily watchlist, canonical fundamental sleeve, strategy book and execution system. It tests collection feasibility and the hypothesis that liquid rallies plus short reported cash runway anticipate standalone primary equity offerings. A prediction result would still not establish a profitable short strategy.

## First result: September 23, 2026

The frozen sample contains 90 issuers selected from 2,105 US-incorporated nonfinancial companies filing financial statements in 2022 Q4 with $50m–$5b reported assets. A deterministic CIK hash selects 30 healthcare, 30 technology and 30 other issuers before outcomes are inspected. This is a narrow accounting-size cohort, not a reconstructed Russell or small-cap market universe.

The run reconstructs 3,240 month-end observations in 2023–2025. Only 54 issuers have usable price histories under the identity/history checks. Missing companies are retained and never replaced. The original watchlist rules identify 20 strength-plus-short-runway stock-months from 12 issuers, with only one predefined match against a strong longer-runway control and three against short-runway names without strength. Seven target observations are cash-only lower bounds. After excluding those and documented intervening funding, five observations remain in the stricter sensitivity, still without complete negative coverage.

The event discovery captures 90 issuer searches and 1,566 documents. It is a candidate inventory, not a complete event ledger: unsearched 8-Ks, incomplete announcement timing and unaudited no-event periods remain explicit. A manually audited subset contains 21 deduplicated events, including provisional dates and boundary events. ELDN's November 11, 2025 offering is a documented future event after its October 31 signal. Several other flags already followed announced financing. Rates, significance and trading returns are not reported from incomplete labels.

Of 46 supplementary review-source URLs, 37 have archived raw bytes. Nine issuer/news-release pages were verified through the web tool but direct HTTP archival did not complete; `review_source_capture.json` preserves that distinction. The failed historical iShares download probe is retained as evidence but excluded from all study inputs. The SEC ZIP and sample manifest define the cohort.

Artifacts: `artifacts/cash_runway/history_20260923_v1/financing_history.html`, `analysis.json`, `protocol.json`, `cohort.csv`, `cohort_manifest.json`, `observations.csv`, `financial_vintages.json`, `event_candidates.csv`, `event_reviews.json`, `labeled_observations.csv`, source captures and QA evidence.

## Reproduce and extend

Prepare a new directory under `artifacts/`. Download the SEC 2022Q4 Financial Statement Data Set ZIP from `https://www.sec.gov/files/dera/data/financial-statement-data-sets/2022q4.zip` as `sec_2022q4.zip`, record its URL/retrieval time/SHA-256, and extract only its `sub.txt` member as `sec_2022q4_sub.tsv`. The script reads NUM directly from the ZIP. The first run's exact source capture is retained.

```powershell
python scripts/build_financing_history.py --output-dir artifacts/cash_runway/NEW_RUN --stage universe
python scripts/build_financing_history.py --output-dir artifacts/cash_runway/NEW_RUN --stage sec
python scripts/build_financing_history.py --output-dir artifacts/cash_runway/NEW_RUN --stage prices
python scripts/build_financing_history.py --output-dir artifacts/cash_runway/NEW_RUN --stage observations
python scripts/build_financing_history.py --output-dir artifacts/cash_runway/NEW_RUN --stage events
# Source-audit event_reviews.json before using any outcome labels.
python scripts/build_financing_history.py --output-dir artifacts/cash_runway/NEW_RUN --stage report
```

`FUNDAMENTAL_SEC_USER_AGENT` is required. Existing requests, pandas, yfinance, lxml, BeautifulSoup and exchange_calendars dependencies suffice; no paid API is used. Three event-capture workers each wait at least 0.4 seconds between requests; aggregate maximum is 7.5 requests/second. Do not run another SEC capture stage concurrently. URL-addressed captures retain source bytes and hashes. Failed derived attempts are preserved before retry, and completed downloads are reused. SHA checks prevent silently changing the frozen cohort or protocol.

`universe` is fixed to the documented 90-company 2023–2025 pilot. An expanded study needs a new explicit protocol/sample size and a new output directory; do not edit this cohort after observing results. The 2025 slice is a prespecified evaluation year, not independent external validation. Subsequent rule changes need a fresh evaluation design.

## Integrity rules

- SEC baseline assets are consolidated, USD, instantaneous and tied to the selected accession/period; conflicting values are rejected. Do not filter `prevrpt`, which incorporates subsequent amendment information. The SEC reprocessed this historical dataset in 2024; it is reconstructed as-filed data, not an archived 2022 database vintage.
- Baseline ticker, exchange and security title come from the original filing's matching XBRL context. Unresolved common-stock identities, changed symbols and missing price histories do not become zero-return histories. The current SEC symbol check is a conservative identity gate and creates explicit attrition; it is not a survivorship cure.
- Financial inputs require accession acceptance timestamps no later than the observation cutoff. Historical submission archives are included. Missing investments/capex remain unknown. A stale or incomplete financial group cannot become an ample-cash control.
- Yahoo OHLC and volume are split-adjusted even with `auto_adjust=False`. Future split actions restore the original nominal price for the $3 eligibility test. Returns, gaps and dollar turnover use consistent adjusted bases. Missing split actions fail acquisition.
- Full-text search uses **root forms**, without literal `8-K/A`: the latter was observed to silently zero the entire query. Include all 424B subtypes, preserve pagination counts and errors, inspect exhibit links, and retain remaining unsearched 8-K accessions. Keyword classification never validates an event or a negative window.
- An audited event requires CIK, event ID, first announcement date, type, sources and status. `verified` events can label positives; `provisional` dates cannot. Date-only announcements on the observation date remain ambiguous. Known same-day release times are compared with the exact signal cutoff.
- Primary outcome: standalone common/pre-funded equity cash-raise announcements. Partnership-linked equity remains a separate genuine funding record. ATM facilities/sales, shelf-only filings, resale-only registrations, debt, merger funding and ordinary exercises are not interchangeable outcomes. Deduplicate launch/pricing/closing and conditional tranche receipts under the original announcement.
- A reviewed negative requires the correct issuer and full window coverage, without overlapping gaps. By default every no-event window remains unknown. Rates and bootstrap intervals are withheld with incomplete labels, because positive-only ascertainment creates a biased denominator.
- Matching uses month, sector, reported assets and dollar volume, plus momentum for strong controls; it never uses outcome availability or outcomes. Resampling, when justified, clusters whole issuers, not overlapping stock-months.

## Verification and remaining gate

Run the three focused cash-runway/financing test files, audit saved financial vintages against their cutoffs, verify source hashes, and render the standalone report at desktop/mobile widths. Save visual QA under this artifact directory, not canonical fundamental state.

Before making an edge claim: expand the prespecified historical cohort, repair delisted/security-alias coverage, capture relevant remaining filings and issuer releases, reconcile equity proceeds with financing notes, finish negative-window audits, and obtain enough independent matched issuers/events. This pilot's sample and coverage are insufficient. No scheduling, upload, deployment, portfolio changes or trades occur in this workflow.
