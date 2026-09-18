# OVS model versus Primary execution

Reviewed September 8, 2026. Priority 3, first bounded review. Research and verification only; no trading rule, runtime, scheduler, scan, order, or email changed.

## Decision in plain language

Keep the current two-path OVS entry/sizing design. The old claim that live trading is decisive-gap-only with a fixed dollar size is obsolete. The installed code already takes mild-gap entries at reduced scanner sizing.

Owner clarification after the review: both exit differences are intentional. Keep same-day targets live, and keep them excluded from the daily-bar Portfolio model to avoid assuming a favorable entry/target sequence that the bars cannot establish. Retain Friday's timed live stop as the practical production implementation; the model's closing-price test remains an approximation. Neither difference is an implementation defect to fix. The remaining actionable findings are deterministic sizing fallback and cap/split rounding differences; they do not justify a strategy redesign.

## What was actually inspected

- The scheduled `IBKR Daily Order Chain` points to `OneDrive/trading_ibkr/run_order_staging.bat`, which runs the installed `order_staging.py` and `eq_order_entry.py`. Neither script was executed during this review.
- The active local postclose task points to `New_Seasonals-automation-runtime-v9`; its marker pins `bdf59bd1ac1eec132516794298d413b3498716ea`. Its strategy configuration, scanner, and strategy engine are byte-identical to the three reviewed development files.
- The installed September 8 morning log reports zero staging rows. It establishes the launch path ran, not an OVS fill or evidence that today's P2 logic was exercised.
- Inspected model call sites in `scripts/build_trade_ledger.py` and `daily_portfolio_report.py`; both use the two-path engine default. Pooled directional caps are removed in the inspected live and production model paths. Historical AGENTS.md claims to the contrary are superseded by executable code.
- Isolated fixtures execute the actual engine and extracted pure installed-code blocks. No broker module import, connection, or order submission occurs in these fixtures.

Exact file/data SHA-256 hashes, fixtures, and output tables are in [the evidence summary](../artifacts/ovs-parity-20260908/summary.json). The [reproduction script](../artifacts/ovs-parity-20260908/review.py) writes only into its own ignored artifact directory.

## Rules that agree

| Rule | Inspected behavior |
| --- | --- |
| No positive opening gap | Skip OVS. Missing live opening inputs also skip; they are not treated as a valid gap. |
| Gap above 0.25 ATR | P1: use scanner quantity. No fixed-dollar P1 override remains. |
| Positive gap at or below 0.25 ATR | P2: multiply by stamped path2/path1 risk, normally 8/40 = 20%. Equality at 0.25 ATR belongs to P2. |
| 2026 sizing | GRM 1.5 and midterm multiplier 0.75 produce 45 effective bps P1 and 9 bps P2 before caps/rounding. At configured $750,000 sizing capital, these are $3,375 and $675 of ATR sizing risk; OVS has no continuous protective stop, so these are not maximum-loss guarantees. |
| Daily caps | Stamped P2 aggregate cap is 1.125% ($8,437.50); total per-strategy staged risk is capped at 250 bps ($18,750). |
| Profit scale-out | Primary normally splits 40% at 1 ATR and the remainder at 2 ATR. |
| Other gates | Earnings blackout and same-symbol ATR Extended Gap Up precedence exist in both paths; the old engine-only P1-budget veto on P2 is removed. |

Six exact gap-boundary fixtures passed, including no gap, just positive, exactly 0.25 ATR, and just above. This establishes contract behavior with valid stamped inputs, not fill certainty or evidence that P2 is economically optimal.

## Differences ranked by usefulness

### 1. Same-day targets: intentional modeling boundary, confirmed by owner

Installed `eq_order_entry.py:655` creates a GTC target child without a next-session activation delay. The engine's OVS near-target loop at `pages/strat_backtester.py:2002` begins after the entry day; the broader engine also excludes entry-day targets. This is an explicit daily-bar convention, not a missing target in the live order.

The installed target construction was reproduced with an inert order object. The engine fixture enters short at 102.50, with a near target of 100.50; even when that day's close is 100.00, it waits until a subsequent session to book the target. IBKR documents `goodAfterTime` as the activation-time field and the parent/child bracket relationship: [Order reference](https://www.interactivebrokers.com/docs/tws-api/ref/order), [bracket orders](https://www.interactivebrokers.com/docs/general/order-types/complex-orders/bracket-orders).

The saved local ledger contains 1,271 distinct OVS entries (2,484 tranche rows), April 23, 2003–July 29, 2026. These are model trades under current rules, not a history of actual broker executions. Every joined opening price matched the ledger and every entry matched open + 0.75 ATR. The OVS sample ends July 29; no August/September coverage is claimed.

| Entry-day observation | Full saved OVS sample | 2026 subset |
| --- | ---: | ---: |
| Modeled entries | 1,271 | 111 |
| Low reached the 1-ATR target | 448 (35.2%) | 32 |
| Close was at/below the 1-ATR target | 84 (6.6%) | 4 |
| Low reached the 2-ATR target | 27 | 0 |
| Close was at/below the 2-ATR target | 9 | 0 |
| Friday entries | 247 | 18 |

A daily low can precede entry. Therefore 448 is a set requiring sequencing checks, not 448 missed profits. A close beyond the target establishes a later price crossing conditional on the modeled entry, but still does not prove a broker fill. Of the 84 near-target close-beyond cases, 82 later book the same target price and only two book a time exit. Most of that demonstrated difference is holding time/capital usage, not proven incremental profit. All four 2026 close-beyond cases later book the target.

The more uncertain cases are intraday target touches followed by reversals. Do not credit them all from daily OHLC, or claim the current convention is necessarily conservative: the eventual exit could be better or worse. A partial same-day profit followed by a Friday loss exit also needs tranche-level sequencing.

### 2. Friday loss exit: intentional production implementation, confirmed by owner

The installed executor creates a stop active Friday 15:58:00 through 15:59:58, anchored to the staged limit plus 0.25 ATR for a short. The engine checks whether the entry-day close exceeds entry + 0.25 ATR and exits at that close. The fixture and extracted native child construction reproduced both behaviors.

These differ when a stock breaches the threshold late and recovers before the close, breaches only at the closing print, or trades through the threshold at a worse price. The engine also uses a strict greater-than test; a stop's native trigger/fill is not equivalent to that final-close test. The installed comments calling the EOD child DAY or saying it expires at 15:59:30 are stale: executable fields are GTD and 15:59:58.

Example: entry 102.50, ATR 2, threshold 103.00, close 103.50. The engine records 103.50. The live code requests a timed stop at 103.00; its actual fill cannot be inferred without the late-session path and broker records. There were 18 Friday entries in the 2026 saved sample; daily highs alone cannot say whether the breach occurred during the active window.

### 3. Missing sizing stamps fall back to different rules

Installed `_ovs_path2_mult` defaults to 15% when path stamps are absent/invalid, while configured P2/P1 is 20%. A fixture with 200 initial shares produces 40 with normal stamps and 30 without: 25% less P2 quantity. The fallback daily P2 cap is 1.0%, versus configured/stamped 1.125% (11.1% lower).

The scanner currently supplies all three stamps, so this is a fallback-path defect, not evidence that normal live OVS orders are currently undersized. Recommended repair after approval: use the same versioned sizing contract for defaults and stamped values, with an explicit exception when the stamp is missing. Do not silently change or block ordinary valid trades as a side effect.

### 4. Small rounding/order-of-operations mismatch

Live floors the aggregate capped quantity before the 40/60 split. The model splits first and later rounds each tranche when applying its daily cap. A reproducible forced-cap fixture starts with 203 shares and scales by 0.625: live gives 50 near + 76 far = 126; the model gives 51 + 76 = 127. This fixture deliberately lowers the cap to trigger the branch; it is not a measured historical loss. Recommended repair: model live's cap-then-split order and compute P&L from final quantities. This is a small deterministic discrepancy; its historical economic impact has not been established.

## Owner decisions and revised next step

McKinley confirmed both exit decisions after reading the review:

1. Profit targets should remain active on entry day in live execution. Portfolio deliberately excludes entry-day target credit to avoid lookahead/within-bar sequencing bias. Preserve both behaviors.
2. Retain the Friday timed stop live; this is the feasible production method. Preserve the daily model's closing-price approximation and describe its limit honestly.

This supersedes the initial recommendation to prioritize intraday exit reconciliation or improve the model's entry-day target credit. No intraday retrieval or model exit change is required to close these two questions. The historical counts above remain evidence of the accepted model limitation, not a repair backlog.

The remaining proposed work is to align missing-stamp sizing defaults with the configured P2 contract and align deterministic cap/split rounding. The owner's exit clarification does not authorize a live sizing deployment. Keep current P1/P2 rules and bracket behavior intact.

For any separately requested future measurement of the accepted approximation, the 32 potential entry-day near-target cases and 18 Friday entries in the saved 2026 sample provide a bounded starting set. Existing local caches do not resolve that set: the repo has 15-minute bars for eight ETFs, while the broker's one-minute store contains only SPY on June 15–16, 2026. Distinguish modeled-order replay from actual fills, verify submission timing, and retain same-minute ambiguity. No such study is currently required or underway.

The exit questions are resolved. Priority 3 remains open for the remaining sizing findings; OLV is not reviewed or certified by this OVS report. No claim is made about broker fills, full production freshness, or the net profitability of a rule change.

## Verification

- `python -m pytest tests/test_ovs_scaleout.py tests/test_eod_dd.py tests/test_pooled_cap_sequential.py -q`: 9 tests passed (one existing dependency deprecation warning).
- `python tests/test_eod_dd.py`: all four standalone cases passed; these are not collected as pytest tests.
- `python artifacts/ovs-parity-20260908/review.py`: boundary assertions, missing-input fixtures, installed target/EOD child construction, cap-rounding demonstration, saved-history joins, basis checks, and sample counts passed.
- Only this review, the working plan, and ignored evidence artifacts were added/updated. Existing dirty work was preserved.
