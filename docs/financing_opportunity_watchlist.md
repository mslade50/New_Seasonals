# Financing opportunity research pilot

The local pilot joins trading strength with reported funding runway. It tests the hypothesis that companies with cash needs may use a liquid rally to issue equity. It does not estimate offering probability or provide a short-entry signal. A company with ample cash may also issue opportunistically; longer-runway setup matches remain visible.

## Run

Use a new directory under `artifacts/` for each observation date. The first stage freezes the information cutoff and latest completed NYSE session, allowing 30 minutes after the close. The exchange calendar handles holidays and early closes.

```powershell
python scripts/build_financing_watchlist.py --output-dir artifacts/cash_runway/opportunity_YYYYMMDD --stage prices
python scripts/build_financing_watchlist.py --output-dir artifacts/cash_runway/opportunity_YYYYMMDD --stage fundamentals --financial-limit 300 --documents
python scripts/build_financing_watchlist.py --output-dir artifacts/cash_runway/opportunity_YYYYMMDD --stage report --reviews artifacts/cash_runway/reviews_YYYYMMDD.json
```

Prices and SEC stages resume stored captures. They never modify canonical market caches, strategy configuration, portfolio state, or production research state. Use a new directory to retry unavailable captures or change the cohort after observations are frozen. Nothing is scheduled, uploaded or deployed. Public SEC, Nasdaq Trader and Yahoo data require no paid subscription; availability and revision risk remain.

## Initial rules

- SEC-mapped current Nasdaq/NYSE/AMEX common and ordinary shares. Excludes funds, depositary securities, units and unverified security types. This is a US listing universe, not a US domicile restriction.
- Close at least $3, mean 20-session dollar volume at least $5m, and 120 consecutive recent SPY sessions. Recent IPOs and sparse histories are excluded. Listing warnings are retained.
- Sustained strength: 20-day return at least 15%, 60-day return at least 25%, 60-day return at least 10 percentage points above SPY, and above the 50-day average.
- Fresh rally: five-day return at least 10% or a positive opening gap of at least 8% during the latest five sessions, plus a volume ratio of at least 1.5 on some session in that window. These conditions need not occur on the same day. Cohorts can overlap.
- Funding flag: six-month operating or PP&E-capex-inclusive runway at most 24 months, from a balance sheet at most 150 days old. The calculated runway starts at the balance date; it is not an estimate of cash remaining today.

The financial queue alternates the cohorts and deduplicates symbols, capped at 300 per run. All 265 setup matches received checks in the September 22 pilot. A check can return unavailable, partial or stale coverage. These states never become negative funding labels. The comparison group is selected trading setups, not a representative market control group.

## Data conventions

Momentum uses Yahoo adjusted close; liquidity uses Yahoo Close times Volume; opening gaps use Open divided by previous Close. Yahoo historical OHLC/volume are split-adjusted. RVOL divides the session's volume by the **preceding** 20-session mean, excluding the session itself. Both yfinance MultiIndex orientations are normalized. Prices are unofficial observations, saved with retrieval timestamps and SHA-256 hashes.

`fundamental/cash_runway.py` supplies point-in-time SEC calculations: filing acceptance timestamps, latest balance, fiscal cash-flow periods and YTD differences, unrestricted cash plus mapped current investments. Missing investments produce a cash-only lower bound. Missing capex remains unknown. Restricted cash, long-term investments, hypothetical warrant proceeds and undrawn financing are not automatically counted as spendable cash. Generic US-GAAP cash-flow coverage excludes financial SICs and does not resolve foreign taxonomies.

The optional filing search reads the latest financial report and up to 12 subsequent matching primary documents. It does not recursively read exhibits, establish shelf effectiveness, calculate legal issuance capacity, or prove that financing did not occur. Errors and truncation are retained. Manual checks review effective primary shelves versus resale registrations, dated ATM capacity, previous issuance, warrants/converts and post-balance funding. ATM and shelf amounts overlap and must not be added. Conditional merger financing is not completed cash.

## Output and verification

`financing_watchlist.html` contains filters, three source-checked examples and expandable evidence. `watchlist.csv` covers price setups; `coverage.csv` records the full listing universe. `watchlist.json`, capture manifests, raw sources and code/input hashes provide audit evidence. Review records must match ticker, CIK, cutoff and balance date; supplied financial figures must tie to calculated values. Same-day sources need a timestamp before the cutoff.

`observations.csv` freezes the setup cohort and funding status. Unknown funding flags stay blank. Outcome fields are pending, not assumed negatives. Rebuilding preserves entered outcomes and rejects changes to frozen baseline fields; use a fresh snapshot for a changed cohort. No automated follow-up is installed.

Subsequent research should label first public announcements of primary offerings separately from ATM sales, completed financings, shelf filings, resale registrations and merger financing. Preserve exact release timestamps and actual entry opportunity. Compare 30/60/90-day event incidence and subsequent returns across verified groups, accounting for overlapping observations, sector/size/momentum differences and listing changes. An issuance-prediction effect is not automatically a profitable short strategy; borrow, squeeze exposure and executable costs remain untested.

Validation: unit tests cover return/volume math, yfinance layout, source timing, stale/missing data, queue overlap and frozen observations. Render the report in a headless browser at desktop and mobile widths, inspect screenshots and test filters/anchors. Keep its visual QA attestation under this artifact directory, separate from fundamental-sleeve state.
