# WP2 what-if runbook: Risk Navigator on the four peak books (one sitting)

Purpose: replace the stylised margin rates in `sizing_optimal_plan_2026-09-02.html`
section 3 / brief WP1 with numbers from IBKR's own engine, and settle the two
things public documentation does not: the leveraged-ETF short rate in THIS
account (TIMS 3x stress, i.e. 45% on a 3x sector fund, versus the rules-based
90%) and what triggers the 30% concentration minimum.

Everything you need to paste is in `whatif_books/`. Budget: ~60-90 minutes with
TWS open. Nothing here places an order; the what-if is a sandbox (red border).

## 0. Before you start

- TWS (Classic or Mosaic), logged in to the PRIMARY account (U16584234). The
  what-if inherits the account's margin type, so PM rates apply automatically.
  Confirm in the Risk Navigator footer: it prints the current Margin Type
  (should read a portfolio-margin type, e.g. `PMRGN`). If it does not, use
  Settings > Reference Margin Type > `PMRGN` (see step 3).
- Do it on a session day, ideally mid-morning: the Dashboard margin numbers
  are recomputed against live prices. The books are historical share counts at
  today's prices, so the DOLLAR requirement will not match 2013/2016/2019/2023
  dollars. What we want is the RATE STRUCTURE (requirement / gross by class,
  the 3x-short rate, the concentration behaviour) and the ratio to today's
  NLV. `requirement_results.json` holds our reconstructed dollars to compare
  ratios against.
- Have `whatif_results_template.csv` open; fill it as you go.

## 1. Open a blank what-if

Classic TWS: `Analytical Tools` menu > scroll to the Portfolio section > hover
`Risk Navigator` > `Open New What-If`.
Mosaic: `New Window` (top left) > scroll to `Portfolio Tools` (Other Tools
area) > hover `Risk Navigator` > `New What-If`.
Alternative from an open Risk Navigator: `Portfolio` menu > `New`.

A prompt asks whether to base the hypothetical portfolio on your existing
portfolio. Answer **No** (blank frame) for the four historical books. Answer
**Yes** once, at the end, for the live-book reading (step 6).

Source: ibkrguides.com/traderworkstation/what-if-portfolios.htm and
open-risk-navigator.htm; campus lesson "Risk Navigator - Alternative Margin
Calculator".

## 2. Load a book

Fastest: `Portfolio` menu > `Import` > pick
`whatif_books/<date>_rn_import_x1.00.csv`. The file carries the header IBKR
documents for the importer (`Action, Quantity, Symbol, SecType, Exchange,
Currency`; Buy = long, Sell = short). "The imported orders are opened in a new
What-If portfolio" (ibkrguides.com/traderworkstation/upload-a-portfolio.htm).
If the importer rejects a header, the documented valid titles are exactly:
Action, Quantity, Symbol, SecType, Exchange, Currency,
LastTradingDayOrContractMonth, Strike, Right, DivPrt, CUSIP, ISIN.

Manual fallback (10-41 rows per book): click the green `New` field, type the
symbol, choose Stock, then in the `Position` column type the signed share
count from `whatif_books/<date>_positions_x1.00.csv` (minus sign = short).
Positions can also be edited in place: double-click the position cell.

After loading, the Dashboard shows a blue swirl next to Maintenance Margin /
Initial Margin. **Click the swirl to refresh** or the numbers are stale.

Set cash so NLV is meaningful: `P&L` tab > expand `Cash` > click the USD
position cell and enter `605462` (live primary NLV on 2026-09-02; use today's
if you have it). The ratio we care about is requirement / NLV; the requirement
itself does not depend on cash.

Save each loaded book: `Portfolio` > `Save As` > `whatif_<date>_x1.00`.

## 3. Read the numbers (the four to record per book x multiple)

Dashboard (top strip of the Risk Navigator window): `Net Liquidation`,
`Maintenance Margin`, `Initial Margin`. Record:

1. **Maintenance Margin** ($) at PM.
2. **Initial Margin** ($) at PM (should be ~1.10 x maintenance; if the ratio is
   not 1.10 something else is binding, note it).
3. **Per-position rate**: `Margin Sensitivity` tab (far right of the tabset) >
   right-click the Underlying column > `Expand All Table`. The column
   `Nominal Margin Interval` is "IBKR's percentage rate applied to the
   underlying" and value x that rate = maintenance for the line (campus lesson
   "Margin Sensitivity Using IBKR Risk Navigator"). Read it for:
   - every 3x ETF line in the 2023-02-03 book (RETL, TECL, SOXL, TNA, DRN,
     LABU, ...). 45% (or 30% on TNA) = TIMS; 90% = rules-based "higher of".
   - the largest line in each book (XLF 2013, SPY 2016, DIA 2019, D 2026).
     30% on a 15%/8% name = the concentration minimum fired.
4. **Exposure fee projection**: `Report` menu > `Exposure Fee` adds an Exposure
   Fee tab that "shows a projection/estimation of any exposure fees, based on
   the current positions in the portfolio (both actual and what-if)" (TWS
   release notes 971). Record the projected daily fee ($/day) and any listed
   risk factor / average exposure. Zero is a result.

Optional but cheap: `Settings` > `Reference Margin Type` > `STKNOPT` (Reg T
reference) to see the rules-based number for the same book, then switch back
to `PMRGN`. The difference on 2023-02-03 is the direct answer to "45 vs 90".

## 4. Scale to 1.25x and 1.5x

Either import `<date>_rn_import_x1.25.csv` / `_x1.50.csv` as fresh what-ifs
(cleanest), or in the open what-if use `Margin Sensitivity` > Position
increment `Overall` and read the +/- change columns. The import route gives
the Dashboard totals directly; the sensitivity tab only gives deltas. Use
import.

Order of work (16 loads, ~3 min each once the first is done):
2023-02-03 x1.00 (settles the 3x rate first), then x1.25, x1.50;
2013-11-04 x1.00 / 1.25 / 1.50 (largest gross, concentration on XLF);
2016-06-14 and 2019-06-26 the same; 2026-09-01 x1.00 only.

## 5. Concentration probe (two extra loads, optional)

The 30% minimum's trigger is undisclosed. On the 2013-11-04 book, edit XLF
from 3,310 shares to ~1,000 and refresh; then to ~6,000. If the XLF nominal
margin interval jumps from 15% to 30% at some size, note the notional / NLV at
the jump. Do the same with SPY on 2016-06-14 (8% base). Two data points are
enough to write the WP1 rule.

## 6. Live book reading

Open a what-if with **Yes** (copy of current portfolio). Record the same four
numbers plus the Account window's `Current Maintenance Margin`,
`Projected Look Ahead Maintenance Margin`, `Excess Liquidity` for the primary
account (Account window > Margin Requirements section;
ibkrguides.com/traderworkstation/margin-monitoring.htm). This is the only
reading that includes the futures legs (MES/DX are SPAN, not TIMS) and the
options.

## 7. Where to paste

Fill `whatif_results_template.csv` (one row per book x multiple; columns are
labelled). Save it as `whatif_results_<YYYY-MM-DD>.csv` in this folder. Then:

- `requirement_recompute.py` prints our reconstruction for the same books;
  the ratio `RN maintenance / our pm maintenance` per book is the calibration
  factor WP1's rate table needs. If 2023-02-03 comes back near $748k (in
  today's dollars, scaled) the account is on the rules-based reading and WP1
  keeps 90%; near $351k it is TIMS and WP1 drops to 45%.
- WP1 (`order_staging.py` guard, OneDrive) rate table: replace the stylised
  rates with the observed nominal margin intervals.
- Plan section 3 / brief WP12: update the "m 1.60 base / 1.34 live" line
  with the RN ratios on live NLV.

## Known limits of the what-if

- Prices are today's; share counts are historical. Ratios by class transfer,
  dollars do not. (Alternative for exact dollars: none; IBKR does not back-date
  TIMS parameters in the client tool.)
- TIMS parameters are the OCC's nightly file; IBKR house add-ons (the 2020
  election uplift, the Jan-2021 short requirements) are not in the what-if
  unless currently in force. Treat the RN number as a floor in stressed tapes.
- A ticker delisted since the book date will not load; the reconstruction
  already aliases ^GSPC -> SPY and ^NDX -> QQQ at the same notional using the
  raw ETF close of the day.

## Appendix: the public-doc verification the session builds on (fetched 2026-09-02)

| Item | Plan value | What IBKR publishes | Where |
|---|---|---|---|
| Stock / sector ETF / narrow index | 15% | "stress parameter is plus or minus 15%" | interactivebrokers.co.uk/en/index.php?f=37749 (Portfolio Margin Mechanics); interactivebrokers.com.au/en/trading/marginRequirements/marginPortfolio.php |
| Broad-based index ETF | 8% | "plus 6%, minus 8%" (worst point 8% for a long, 6% for a short) | same |
| Small-cap index | 10% | "plus 10%, minus 10%" | same |
| 3x ETF under PM | 45% | not on any IBKR page; OCC/TIMS convention "market moves are multiplied by the ETF's stated leverage" (Wikipedia, Portfolio margin) | what-if question 1 |
| 3x ETF rules-based | 75 / 90% | "For Leveraged ETFs, Minimum(25% * Leverage Factor, 100%)" long; "Minimum(30% * Leverage Factor, 100%)" short | co.uk f=37749, Disclosures |
| Higher of rules and PM | unknown | "Interactive Brokers will calculate both methodologies and assess the higher of the two to your account." (sentence sits under the non-US-resident panel; US panel is JS-rendered, not captured) | co.uk f=37749; com.hk f=26658 |
| 30% concentration | add-on, trigger unknown | "Classes with large single concentrations will have a margin requirement of 30% applied to the concentrated position." Trigger unpublished. Separate rule: >1% of shares outstanding raises margin, 100% at >=9% (ETFs >=5%). | same |
| Initial 110% | 110% | "Initial margin will be 110% of Maintenance Margin." | same |
| Soft Edge Margin | 15:45 ET | "until 15 minutes before the close ... deficit to be within ... 10% [of NLV]. When SEM ends, the full maintenance requirement must be met." | com.au marginCalculationsSecurities.php |
| Short minimums | $5/sh < $16.67, 100% < $5 | Reg T short table (30% above $16.67, USD 5.00/sh between 5 and 16.67, 100% below 5) and "special margin requirements for ... low cap stocks that apply under Reg T, will still apply under Portfolio Margin" | co.uk f=37749 |
| No long/short offset across ETFs | as stated | "For stocks and Single Stock Futures offsets are only allowed within a class" | same |
| Exposure fee | report | Client Portal "Exposure Fee Calculation Report"; Risk Navigator Report menu > "Exposure Fee" tab projects it for actual and what-if portfolios | interactivebrokers.com/en/trading/margin-stocks.php; interactivebrokers.com/en/general/tws_notes_971.php |
