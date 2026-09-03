# Live vs ledger from the primary account statement (2026-09-03)

Closes the measurement half of **gap 4** in the 2026-09-02 sizing due diligence
("no live R series exists"). Input: IBKR activity statement for the primary
account U16584234 (Denali Global Macro Fund, LLC), 2026-01-01 to 2026-09-02,
999 stock orders over 200 symbols. Matcher: `match_statement.py`.

No order reference is available (IBKR offers that field on single-day flex
queries only), so the match is keyless on symbol + session + side. Two
aggregations make the sides comparable: the ledger collapses to POSITION level
(OVS books two scale-out tranche rows per fill, exiting on different sessions),
and statement orders collapse to one weighted-average price per symbol/session/
side. 78 of 244 positions matched both legs, none ambiguous.

## 1. The book only went live in this account around June

Ledger position -> live entry match rate, against a statement that was busy
every month (99-187 stock orders/month):

| month | positions | matched | rate |
|---|---|---|---|
| 2026-01 | 34 | 1 | 3% |
| 2026-02 | 19 | 2 | 11% |
| 2026-03 | 37 | 0 | 0% |
| 2026-04 | 25 | 1 | 4% |
| 2026-05 | 19 | 8 | 42% |
| 2026-06 | 47 | 36 | 77% |
| 2026-07 | 44 | 42 | 95% |
| 2026-08 | 19 | 19 | 100% |

January to April activity in this account was therefore almost entirely
something other than the systematic book. **Any live-vs-ledger claim before
June is measuring two different things.** The 67 OVS positions with no live
counterpart are concentrated there and are not a 2-path gate divergence.

## 2. Execution is free

June onward, N=69 positions:

| metric | value |
|---|---|
| ledger avgR | +0.231 |
| live avgR | +0.226 |
| ratio | 0.982 |
| paired diff mean | -0.004 (CI95 -0.046 to +0.042) |
| entry slippage, median | -0.0 bps |
| exit slippage, median | +2.5 bps |
| live/ledger shares, median | 1.000 |

Per strategy (June onward): OVS n=29 diff +0.010, OLV n=21 diff -0.082,
LT Trend n=7 +0.068, 3x Bear Fade n=6 +0.124, Monday Dip n=3 -0.018,
MonFri n=2 +0.016, 3x ETF Fade n=1 -0.031.

**The `live R = ledger R x 0.60` working assumption is not supported as an
execution claim.** Orders fill where the model says they fill. Whatever
haircut is right is about edge decay and discretionary override, not fills.

Note this measures a different thing from the 2026-09-02 audit's ring-based
figure (18 legs, ratio 0.72). That one scored the position's realised outcome
including hand trims; this one scores whether the modelled orders filled at
the modelled price. Both are right. The gap between them IS the discretion.

## 3. Pre-August OLV live size was up to 2.9x what the ledger models

OLV share ratios split cleanly by date: every position entered 2026-07-29 or
later matches exactly (ratio 1.000), and every earlier one is 1.15x to 2.85x
larger live.

That is the ledger being a full re-simulation under CURRENT config, exactly as
documented. The OLV signal-recency ladder (0.5x / 0.7x / 1.0x) shipped
2026-07-30; before 2026-07-29 OLV ran with no ladder at all. So the ledger
halves June and July legs that live took at full size, and 1/0.5 = 2.0 is the
observed ratio on several.

R is per-share-normalised, so section 2 is unaffected. But it means **any
ledger-based exposure or notional analysis of OLV before August understates
live by up to ~2x**, which strengthens rather than weakens the due diligence's
warning about the June 2026 OLV stack and the disabled EOD book cap.

## Caveats

- Keyless matching cannot separate two positions sharing a ticker, session and
  side. None of the 78 matches were ambiguous, but that is partly because
  unmatched ambiguity shows up as a miss rather than a wrong match.
- Statement prices are order-level averages; the ledger models one price.
- The exit comparison uses the ledger's own exit sessions and tranche weights,
  so a hand exit on a different day is scored as a miss, not as slippage.
