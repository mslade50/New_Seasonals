# Frozen financing-window source audit — 2026-09-24

Research only. This audit does not change the original cohort, observation labels, strategy configuration, portfolio state, or production site. All data access used public SEC and issuer sources; no paid data service was added.

## Delivered result

`artifacts/cash_runway/window_audit_20260924_v1/financing_window_audit.html` reviews exactly the 20 original target observations and four predefined comparisons from `history_20260923_v1`, preserving the 60-calendar-day horizon and exact signal cutoffs.

| Outcome | Original signals | Comparisons | Total |
|---|---:|---:|---:|
| Reviewed, no standalone primary offering | 17 | 1 | 18 |
| Verified standalone offering | 1 | 0 | 1 |
| Mixed debt/equity boundary | 1 | 1 | 2 |
| Unclassified because of incomplete release chronology | 1 | 2 | 3 |

Eledon launched on November 11, 2025, 11 calendar days after its October 31 signal. Sarepta announced a separate $20m primary common-stock subscription with debt refinancing on August 21; that **one** mixed event intersects the July signal and June comparison. It is not two independent events. Inclusive cash-equity labels recognize the subscription, but strict standalone labels remain unknown because complete independent Sarepta release inventories were not certified.

CVI May 2025 and SXT July 2023/May 2025 remain unknown. Their SEC financing notes alone cannot establish absence of an offering announcement. Independent IR archives were unavailable; the bounded browser attempt also failed at Windows sandbox startup. Thus five windows lack complete negative-source coverage: these three unknowns plus the two mixed Sarepta windows.

## What the audit changes in our understanding

- Three original signals were fresh rallies with negative 60-session returns and prices below their 50-day averages: NTLA June 2024, SRPT July 2025, BBAI April 2025. SRPT was down about 74% over 60 sessions. These do not represent the intended healthy-strength setup. Preserve them in this experiment; specify a new strength gate before a new test rather than selecting winners after observing outcomes.
- FLGT November 2025 used $117.641m cash but omitted $258.162m current marketable securities. Combined: $375.803m. Keep $411.778m noncurrent investments separate. The filing also disclosed a later $67.9m tax-credit cash use. Neither subtotal is a full cash rollforward. The current-investment value was not in the standard Companyfacts tags used by the pilot; footnotes/custom facts need coverage.
- Earlier earnings releases can supersede a filed balance. SXT July 21, 2023 already reported June cash $36.546m and positive six-month FCF $6.565m. SRPT July 16, 2025 disclosed preliminary combined liquidity around $850m, including restricted cash/investments. KYMR January 2023 also had a newer preliminary balance release. Do not add those balances as financing receipts.
- GERN's newly recovered January 10, 2023 closing release confirms $227.8m gross and approximately $213.3m estimated net before the January signal. The prior ledger's unconfirmed-receipt status was a retrieval gap; preserve that ledger and record this correction here.
- Prior announced or completed raises cannot be advance predictions. NKTR's June 30 launch preceded the 16:30 ET signal; July pricing/closing are the same event. KYMR June and GERN March offerings were also already announced before the relevant signals.
- Debt capacity, partnership cash, ordinary warrant exercises and ATMs must remain separate. KALU and RRR had positive operating cash flow with capital expenditure and credit-capacity confounders. CVI subsidiary credit is not automatically fungible parent cash. Several issuers sold stock through ATMs even though they had no standalone offering in the window.

No predictive incidence comparison, significance test, trading P&L, or current-cash estimate is claimed. The sample is small, comparison coverage is poor, and input quality needs repair. A bounded reviewed negative means no qualifying announcement was found in the selected sources, not proof of universal absence or no financing of any kind.

## Evidence and gates

359 captured SEC primary documents/exhibits and 24 captured official issuer pages are SHA-256 checked by the builder. Inventory covers selected current, financial, prospectus, registration and related forms; 14 calendar days after each outcome endpoint provides a late-filing buffer. Covering financial statements and five initially omitted intermediate financials were added without changing the event horizon. Relevant filing bodies, exhibits and financing/equity notes were reviewed; keyword snippets were triage aids only.

A negative requires complete selected primary accession capture, all discovered linked exhibits captured, no capture error, explicit filing/exhibit/financial-note review, and an independently reviewed, archived issuer-release chronology. A verified offering can establish a positive despite incomplete negative search. An unresolved primary candidate blocks a negative. Unknown timing for a signal-day announcement excludes the observation even if a later positive is known. Event identities deduplicate launch, pricing and closing.

Source documents published after the signal are permitted to validate outcomes, but their facts cannot be inserted into the earlier predictor. Financing receipt completeness is explicitly **not asserted**. A known expected closing is not a confirmed receipt, gross is not net, and cumulative deal totals are not incremental proceeds.

## Reproduce and verify

Run from the existing checkout, with the retained local capture artifacts:

```powershell
python scripts/build_financing_window_audit.py
python -m pytest tests/test_cash_runway.py tests/test_financing_opportunity.py tests/test_financing_history.py tests/test_financing_ledger.py tests/test_financing_window_audit.py -q -p no:cacheprovider
python artifacts/cash_runway/window_audit_20260924_v1/qa_browser.py
```

The builder is offline and checks original input hashes and exact sample membership. Rebuilding replaces only derived audit outputs and resets manifest status to pending visual QA. Source reviews are human research inputs in `reviewed_windows.json`; they cannot be regenerated by a keyword rule. The renderer is deliberately scoped to this 24-window pilot, not a general dashboard.

Validation: 83 tests passed (one existing exchange_calendars deprecation warning). Two independent-review defects—omitted linked exhibits and ambiguous signal-day announcements overriding eligibility—were reproduced by failing regressions before fixes, then passed. Desktop/mobile screenshots and Eledon evidence disclosure were visually inspected; filter counts, search, empty state, six download links, contained mobile table scrolling and no page errors passed. The artifact manifest records exact hashes and visual attestation. No production deployment was performed.

## Next experiment

Repair cash/current-investment coverage and ingest releases available before filings. Reconcile prior announcements and confirmed financing without pretending to know exact current cash. Separate operating burn from growth capex and credit capacity. Define sustained strength and a damaged-price exclusion **before** collecting the next evaluation outcomes. Then freeze a larger, better-matched sample and validate offering prediction separately from any executable short strategy, costs and borrow availability. Keep ATM financing as a distinct outcome family if studying it; do not silently fold it into standalone offerings.
