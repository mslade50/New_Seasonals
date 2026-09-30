# Financing research dashboard

The user shifted this work from formal signal validation to a highly filtered discretionary dashboard on September 24, 2026. No offering probability, trading rule, order, allocation, outcome study, or live strategy change is produced. Historical research artifacts remain preserved.

## Delivered snapshot

`artifacts/cash_runway/dashboard_20260924_v1/financing_dashboard.html`

- September 23 completed-session prices; September 24 SEC/issuer review.
- Refresh of 265 existing setup candidates from the September 22 screen of 5,348 listings. This is not a fresh whole-market discovery run; newly qualifying names outside that pool can be missed.
- 36 pass the strict price filters; all 36 received fresh SEC submissions/facts checks.
- One main watchlist row: IOVA. Its historical operating runway is 13.4 months, but management forecasts funding into H2 2028; that counterpoint is prominent. The dashboard does not imply imminent financing.
- Eleven need review: FTK and VSTM have post-balance financing changes; nine have incomplete/unavailable financial coverage. They are outside the main watchlist.
- One source-checked HAFN primary equity offering: September 23 term sheet targets approximately $300m, with final pricing/receipt unverified. Three related launch filings are one transaction.
- Fourteen selected SEC filing records from price-qualified names over 30 days, with a since-prior-close filter. This is a filing queue, not a complete overnight offering announcement feed. Shelf/prospectus/agreement filings are not automatically labeled offerings.

FTK's September 23 filing reports $75m funding at closing, partly used to refinance prior debt; $120m headline capacity is not cash received. VSTM's August disclosure describes conditional Oberland funding and a Secura milestone/pro forma balance that need reconciliation. These prevent treating the old balances as current funding pressure.

## Working filters

Price at least $5; 20-session average dollar turnover at least $10m and median at least $5m. At least 252 usable sessions. Above 50- and 200-session averages. No split in 60 sessions or current listing warning. Return over 20 sessions between +15% and +100%; over 60 sessions between +25% and +200%, at least 10 percentage points ahead of SPY. Within 25% of the 252-session high. These are conservative browsing defaults, not calibrated predictive thresholds.

Financial gate: positive six-month operating cash burn, at most 24 months reported operating runway, balance no more than 120 days old. Capex-only burn does not qualify. A company can reach the main list only with an exact-cutoff/CIK/balance-bound source review checking liquid investments, newer releases and intervening funding, and with no unresolved financing or balance change. Cash-only data is not treated as verified absence of investments. Restricted cash and undrawn credit are not added.

The user can tighten displayed turnover/runway, search, sort, expand evidence, and inspect coverage. The default view stays small; unknowns do not receive a funding score.

## Source handling

Free SEC identified/rate-limited requests, Nasdaq Trader listing identity and Yahoo daily bars. Captures, timestamps and SHA-256 hashes stay under the snapshot. The fresh Yahoo batch omitted a session for many names. 188 histories were completed from the prior hash-verified capture only after checking matching Open/Close/adjusted-close basis over 20 overlapping sessions. Original new and prior captures remain intact; `price_repairs.json` identifies reused sessions and hashes. Price fields are recomputed, never copied from failed fresh calculations.

Historical runway divides reported liquid assets by six-month average operating burn and starts at the balance date. It is not a current-cash forecast. Management forecasts, ATM capacity and completed financing are labeled separately. Undisclosed ATM sales remain possible even after a bounded public-source review.

## Rebuild or refresh

Offline rebuild of the reviewed snapshot:

```powershell
python scripts/build_financing_dashboard.py --output-dir artifacts/cash_runway/dashboard_20260924_v1
```

New captured snapshot of the same discovery cohort:

```powershell
python scripts/refresh_financing_dashboard.py --source-dir artifacts/cash_runway/opportunity_20260922_v1 --output-dir artifacts/cash_runway/dashboard_YYYYMMDD_v1
```

The refresh refuses existing output directories and does not copy manual reviews. Initially new names stay in Needs review until source-backed `reviews.json` entries match the new cutoff and financial balance. Then rebuild. No scheduler, paid service, email, site publication or production upload is installed. Expanding discovery or adding a complete announcement feed is separate work.

Source files: `fundamental/financing_dashboard.py` owns gates, `fundamental/financing_dashboard.html` owns the standalone interface, and the two dashboard scripts own capture/build. Existing runway and momentum calculations are reused without changing earlier experiments.

## Verification

Dashboard gate tests cover distressed rebounds, illiquidity, extreme moves, stale/missing prices, recent splits, unknown identity, stale reviews, missing investments, newer financials, financing changes, capex-only burn and source timing. Embedded JSON escapes HTML script terminators. Relevant existing cash-runway and momentum tests run with them.

Headless Edge checks the default list, evidence drawer, contradictory management guidance, FTK funding detail, all tabs, search, tighter runway/turnover, empty states, reset, prior-close filing filter, downloads and 390px mobile overflow. Desktop, expanded evidence and mobile screenshots are visually inspected. Exact-report visual attestation and verification hashes remain inside the artifact directory. The new reusable refresh command uses the same stages exercised on this snapshot; its CLI loads successfully. It has not been scheduled or deployed.
