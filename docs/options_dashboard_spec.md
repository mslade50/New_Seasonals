<!-- Generated 2026-07-08 from the ultracode research workflow (5 lenses -> critic + feasibility -> synthesis). Phase-1 build plan lives in the session plan; this file is the durable product spec. -->

# Options Dashboard Spec — Delta-Edge Options Surface for New_Seasonals

## (a) Design Thesis

This dashboard has one job: translate an already-validated stock signal (known entry, stop, target, hold, win rate, avgR) into the cheapest well-shaped unit of directional exposure, sized in the same risk-bps currency as the rest of the book, entered under the same entry conditions that produced the backtest stats, and managed to the same exit script. It is a signal-to-structure pipeline, not an options platform. Vol analytics appear only where they change the structure decision (debit vs credit vs stock, which expiry, which side of the smile); everything else is workflow plumbing that keeps the options sleeve inside the existing systematic machinery (Risk_Amt sizing, daily caps, exposure accounting, verify-fills-style reconciliation, append-only parquet history). Two architectural facts shape everything: IBKR serves no historical per-contract IV or greeks, so all option history must be self-recorded from day one by the local agent (never GHA, which cannot reach TWS); and the stock edges are conditional on their entry mechanics (OLV's T+1..T+3 dip limit, OVS's gap tier), so any options expression that buys at the open unconditionally is trading a different, unvalidated system.

---

## (b) MUST-HAVE Features

### 1. Options-Viability Gate (universe screen)
- **Displays:** Per-ticker badge on the signals queue, three states: OPTIONS-OK (green), OPTIONS-MARGINAL (amber: monthlies only, or strike spacing coarse vs the trade geometry), NO-OPTIONS (grey, "Express" button disabled). Detail popover: weeklies yes/no, strike increment, strike increment as % of ATR, whether a strike exists within 0.25 ATR of the stock target.
- **Calculation:** Strike spacing test: nearest listed strike to the signal's target must be within max(0.5 × ATR, half the strike increment); expiry test: at least one expiry lands in [Time_Exit_Date, Time_Exit_Date + 15 td].
- **IBKR mechanism:** `reqSecDefOptParams` per underlying (no quotes, no market data lines), filtered to `tradingClass == symbol` to skip adjusted classes. Refreshed weekly by the local agent into a small parquet; static enough to cache.
- **Delta-edge role:** Kills structure math before it starts on the ~870 overflow names that mostly fail this. $2.50 strikes on a $23 stock cannot place a short strike at the target; discovering that at 9:30 against a staging deadline wastes the morning window.

### 2. Express-in-Options Handoff with Entry-Condition Parity
- **Displays:** An "Express in options" button on each staged signal row, opening the workbench pre-filled: underlying, direction, expiry (rule in feature 5), long strike nearest the stock entry level (fallback ~40Δ), short strike nearest the stock TARGET, plus the strategy's stats card (avgR, win rate, median hold, terminal move at exit). A "staged for open" tray holds tickets created at 4:47 AM for live pricing after the open. Each ticket shows its **entry condition**, mirrored from the stock side: e.g. "trigger only if stock trades ≤ $41.20 (entry limit), live T+1..T+3" for OLV, or "OVS: submit only on Path-1 gap (open > close + 0.25 ATR); skip on mild gap / gap down."
- **Calculation:** All parameters inherited from the staged row (Ticker, Direction, Entry, Stop, Target, ATR, Hold_Days, Risk_Amt, Time_Exit_Date, Fill_Window_Days). Nothing re-derived by hand.
- **IBKR mechanism:** The local agent monitors the underlying's stock line (already subscribed for the stock book) and submits the combo only when the stock-side entry condition triggers; alternatively an IBKR price condition on the underlying attached to the combo order. The OVS gap-tier check reuses the exact open-vs-close+0.25ATR logic order_staging already runs.
- **Delta-edge role:** This is the feature that keeps the backtest honest. The validated avgR is conditional on the limit filling; buying the vertical at the open unconditionally trades a different system and corrupts the counterfactual scorecard. Also enforces the opening-liquidity rule: no unconditional submit before 9:45 ET (opening rotation quotes are systematically wide); triggered orders naturally wait for the entry condition anyway.

### 3. Edge-vs-Priced Expected Move Comparator
- **Displays:** Bar pair per candidate expiry: (a) implied move over the hold, (b) the strategy's historical terminal move at exit, in % and ATR units. Edge Ratio = forecast ÷ implied, color-coded: > 1.2 green ("your edge exceeds what's priced, long premium is cheap"), 0.8–1.2 amber, < 0.8 red ("market prices a bigger move than your signal delivers, prefer credit structures or stock").
- **Calculation:** ONE convention used everywhere: **implied 1σ move = S × IV_ATM × √(DTE/365)**, scaled to the hold horizon by √(hold_days/DTE); the ATM straddle mid is shown beside it labeled "market price of the move" (a dollar cost, a different quantity, never blended into the ratio). Forecast side = the ledger's **terminal move at exit** (mean and median) from backtest_trades_full.parquet, never MFE (a running max, upward-biased; MFE is reserved for short-strike placement in feature 6).
- **IBKR mechanism:** IV_ATM and straddle mids from the chain snapshot already taken (2 contracts per expiry, one batched `reqTickers`); ledger side needs zero new data.
- **Delta-edge role:** The single most direct "am I getting paid for my edge" test. No retail platform can join implied EM against the trader's own conditional move distribution; he can, from day one.

### 4. IV Context Strip (IVR + IV Percentile + IV−RV)
- **Displays:** Header banner per underlying: IV Rank and IV Percentile side by side (Rank alone is poisoned for months after one spike; Percentile stays honest), 1y IV30 sparkline with current marker, and three IV−RV tiles: IV30 vs Yang-Zhang RV at 10/21/63d, spread in vol points, color-coded. Regime label: percentile < 30 "premium cheap, debit structures"; 30–50 "tiebreak on cost-of-delta"; > 50 "elevated, spreads/credit"; > 80 "rich, sell premium to buy delta". Badge "NO IV HISTORY" when the cache lacks the symbol.
- **Calculation:** IVR = (IV_now − 252d low)/(252d high − low) × 100; Percentile = share of last 252 days below IV_now; YZ RV from master_prices OHLC (drift-independent, gap-robust, standard k = 0.34/(1.34 + (n+1)/(n−1)), annualized √252).
- **IBKR mechanism:** `reqHistoricalData(whatToShow='OPTION_IMPLIED_VOLATILITY')` on the UNDERLYING stock contract (IB's 30d ATM-interpolated series, the one historical IV IBKR does serve). **Runs on the local agent, not GHA** (IBKR data only flows through local TWS); one-shot backfill is a paced job (60 requests/10 min, ~190 tickers ≈ 35–40 min), nightly incremental appends to an append-only parquet uploaded to R2, rd2_fragility discipline, fail-open with notice when the machine is off. Coverage is per-symbol (expect overflow gaps); probe and badge.
- **Delta-edge role:** The debit-vs-credit switch. His mean-reversion signals fire into vol spikes by construction, so without this gate he systematically overpays for delta on exactly the strategies most likely to use options.

### 5. Expiry/DTE Picker Driven by Hold + Earnings
- **Displays:** Actual listed expiries (replacing the hardcoded next-6-Fridays dropdown), each annotated: DTE, DTE remaining at the planned time exit ("you'll exit with 9 DTE left"), earnings marker if the ticker reports before that expiry, ATM IV per expiry with forward vol between adjacent expiries as step segments (rich front / event bump visible at a glance). Default selection highlighted.
- **Calculation:** ONE unified rule: **default = first listed expiry ≥ Time_Exit_Date + 5 trading days**, hard warning (not a block) when expiry < Time_Exit_Date or when the planned exit lands with < 7 DTE remaining (final-week theta/gamma noise while still in the trade). Forward vol via variance additivity: σ_fwd² = (σ₂²T₂ − σ₁²T₁)/(T₂ − T₁). The 21-DTE convention is NOT applied to debit verticals (it is short-premium research); their rail is remaining extrinsic vs remaining hold (feature 11).
- **IBKR mechanism:** Expiries from `reqSecDefOptParams`; ATM IVs from ~2–4 contracts × N expiries in one batched snapshot (~40 lines, ~5–8 s). Hold/exit dates from the staged row; earnings from data/earnings_calendar.parquet.
- **Delta-edge role:** Expiry choice for a known-hold trader is an optimization, not a preference: enough time to cover the hold plus buffer, no more (paying for 35 days of theta on a 10-day hold), never so little that the final week's gamma decides the trade.

### 6. Structure Shootout at Equal Risk (the core table)
- **Displays:** One table, 4–6 rows priced from one chain snapshot at the SAME risk dollars: stock-with-stop (baseline), target-anchored debit vertical (short strike AT the stock target), wider/ATM vertical, long call/put (~40Δ), credit vertical (short ~30Δ / long ~15Δ, shown only if credit ≥ 25% of width, a screen, since 1/3-width credits are rarely attainable outside high-IV regimes). Columns: net debit/credit at mid AND natural (gap in $ and % of debit), net delta in share-equivalents, premium per unit delta, breakeven move (% / ATR / as fraction of forecast move), max P/L, **P&L if target hit by time exit** (the money column), P&L if stopped, P&L if flat at time exit, **commissions round-trip**, margin, spread-tax traffic light, expected value. A one-line verdict row: which structure wins, with **"use stock"** as a first-class output whenever no structure beats the baseline after costs.
- **Calculation:** Legs re-priced at the planned exit date (median hold) via client-side BSM using each leg's own snapshot IV, **with dividend yield and rate inputs** (flat European BSM with q=0 systematically misprices ITM structures on his dividend-heavy universe), labeled "approximation". Default IV-shift assumption per strategy: a small crush (−3 to −5 pts) for signals that enter on vol spikes, user-adjustable. EV = probability-weighted P&L using the strategy's own ledger outcome histogram (win rate × P&L(target) + loser mix × P&L(stop/flat)), not expiry POP. Commissions at ~$0.65/contract per leg per side (a 25-lot vertical ≈ $130 round trip ≈ 3% of $4,500 risk, same order as the spread tax) appear in EV, never as a footnote.
- **IBKR mechanism:** All legs inside the band the existing option_quote.py pattern already fetches (`reqSecDefOptParams` → qualify → batched `reqTickers` + ~3s for modelGreeks; retry then local BS fallback when greeks stay None on illiquid legs). Margin per structure via **`whatIfOrder()`** (one cheap round-trip each, mandatory for any credit/undefined-risk row). OI/volume via generic ticks 100/101 on live lines only; on delayed lines degrade to bid-ask-width-based liquidity scoring (OI is yesterday's number intraday regardless).
- **Delta-edge role:** This IS the morning decision: stock or options, and which structure, judged at HIS target and HIS horizon at equal risk. Expiry payoff framing is banned as the headline; the T+hold number is the honest one.

### 7. Risk-bps Sizing Translator + Cap/Exposure Write-Back
- **Displays:** Panel on the ticket: target risk $ (the signal's already-overlay-adjusted Risk_Amt: GRM, fragility bands, ladder, derates all included), computed contracts, EFFECTIVE risk $ and bps after rounding, the quantization error with the floor/ceil pair shown when material ($3.40 debit: 13 lots = $4,420 vs 14 = $4,760, pick one), net position delta in share-equivalents beside what the stock trade would have staged, warning on leverage creep, warning when target risk < one contract's debit.
- **Calculation:** **Default R definition = debit at risk**: contracts = floor(Risk_Amt / (debit × 100 + round-trip commissions)). Stop-scenario sizing (contracts = Risk_Amt ÷ modeled loss at stop) is an explicit opt-in with a printed label that its R is model-dependent and gaps through the stop take the position toward full debit. Credit verticals: risk = (width − credit) × 100.
- **IBKR mechanism:** None new; arithmetic on the staged row + snapshot. The write-back is the real work: every submitted options trade posts its effective risk $ into the same accounting order_staging's **2.5% global daily risk cap** consumes, and its share-equivalent dollar-delta into `exposure_state.json`, so the stock dashboards and caps never go blind to the sleeve.
- **Delta-edge role:** The whole book runs on one bps-of-NAV convention with multiplicative overlays. An options sleeve that consumes a different risk number, or bypasses the daily cap, silently breaks portfolio-level risk. This keeps one currency.

### 8. Earnings/Event Panel, Fail-CLOSED for Options
- **Displays:** Three-state banner per (signal, expiry): CLEAR; EARNINGS INSIDE HOLD (date, td away); EARNINGS BETWEEN EXIT AND EXPIRY (mild, exit-day marks carry event vega). Plus a fourth state the stock book doesn't have: **"NO EARNINGS DATA"** rendered as an explicit amber warning, never silence. When flagged: estimated crush in vol points and in $ on the contemplated structure at current vega, and auto-suggestions (expiry before the event, vertical instead of long option, defer to post-print).
- **Calculation:** Date join against data/earnings_calendar.parquet over [today, Time_Exit_Date] and [today, expiry]. Crush estimate via the simple two-expiry event-variance extraction: event_var = σ_front²T_front − σ_fwd_baseline²T_front, labeled estimate. Ex-div dates inside the window flagged alongside (feeds feature 11).
- **IBKR mechanism:** Earnings need zero IBKR calls; per-expiry ATM IVs already fetched for feature 5.
- **Delta-edge role:** IV crush is the one mechanism by which a correct delta call still loses in options. The stock book's NaN-passes-through convention is tolerable for stock and dangerous for long premium; ~114 universe tickers have no earnings coverage and must say so out loud.

### 9. Combo Order Ticket with Auto-Walk and Fill-Quality Logging
- **Displays:** One visible form (no wizard): net bid/mid/natural for the spread plus per-leg quotes, limit defaulted to mid (mid − 1 tick for debits), walk controls behind progressive disclosure (rest 20–30 s at mid, step $0.01–0.05 toward natural, hard cap default 40% of the mid-to-natural gap, cancel-if-unfilled timeout). Pre-submit liquidity gate: flag any leg width > 10% of mid or > $0.15 absolute, per-leg volume/OI where available. Post-fill: fill vs mid-at-submit in cents and % of spread, logged. Confirmation screen always shows effective risk after contract rounding. Delayed-data mode forces an extra confirm ("do not price orders off these marks").
- **Calculation:** Slippage cap expressed as % of the structure's EV from feature 6, so "this walk can give up at most X% of the edge" is explicit.
- **IBKR mechanism:** `placeOrder` on a native BAG contract (leg conIds from qualification), LMT, SMART routing, through the existing gated agent execution path; SMART combos are atomic (never legged into a partial structure). Auto-walk = agent-side cancel/replace timer, steps rounded to the combo's order minTick (often 0.05 even when quotes display 0.01). The TWS-computed BAG quote is used as the anchoring number where available (snapshot mode unsupported on BAG, unreliable on delayed; fall back to leg-sum mid). Exits are managed by the alert engine + closing combo tickets, not resting bracket children (combo OCA support is limited).
- **Delta-edge role:** Sub-1R average edges cannot survive $0.05 of give-up per fill on a $1.50 spread being invisible. Codifying and measuring the walk keeps the sleeve's realized edge close to the stock sleeve's.

### 10. Options Positions Book with Signal Linkage and Mark Hygiene
- **Displays:** One row per spread (legs expandable): ticker, structure, **source strategy + signal date**, days held vs Hold_Days as a progress bar ("day 7 of 10"), DTE and DTE-vs-remaining-hold, net debit paid, current mid value, P&L in $ and % of max profit, **executable-close P&L** (long legs at bid, short at ask) with the mid-vs-executable gap as a liquidity-haircut column, net delta in share-equivalents, theta $/day, spot vs long strike / short strike / stock stop / stock target as a mini-ladder, alert badges. Header tiles: net dollar-delta by underlying (stacked with stock positions in the same name), beta-weighted SPY delta as a display number (betas from master_prices 252d regression; no alert thresholds at this book size), total theta $/day, vega bucketed by expiry. Global staleness governance: market-data mode banner (live/delayed/frozen), snapshot age per row (amber > 60 s, red > 5 min), one-click targeted re-quote of only on-screen conIds; all hard "act now" styling suppressed on stale/delayed inputs.
- **Calculation:** Headline P&L always from leg mids taken in the same snapshot cycle, never raw broker mark (raw marks on mid-cap verticals produce phantom ±40% intraday swings); %-of-max management rules run on the executable-close number. Badge one-sided/zero-bid legs and mute their intraday P&L styling. After-hours cycles freeze last regular-hours marks.
- **IBKR mechanism:** `ib.portfolio()` for OPT rows (book_snapshot.py pattern), one batched `reqTickers` on the leg conIds for bid/ask/modelGreeks + underlying spot (~2 contracts per vertical, line limits a non-issue); greeks retried then computed locally from mid + last known IV when None. modelGreeks on delayed lines arrive late or not at all on illiquid legs; the local-BS fallback is the plan, not a contingency.
- **Delta-edge role:** The option is a delta vehicle for a timed signal. Days-held vs the strategy's hold window, and spot vs the ORIGINAL stock levels, are what no off-the-shelf platform can show, because they require STRATEGY_BOOK metadata. A vertical gone deep ITM or far OTM has silently lost the delta the signal wanted; this row makes that visible.

### 11. Alert Engine + Expiry/Ex-Div Hygiene
- **Displays:** Action queue at the top of the book, each item deep-linking to the position with a pre-filled closing ticket. Rules: (1) executable value ≥ 50–75% of max profit on debit verticals ("remaining reward-to-risk has collapsed, take profit"); (2) **remaining extrinsic < expected remaining edge** on debit structures (the debit-vertical rail; the 21-DTE rule stays credit-structure-only); (3) underlying touches the short strike; (4) **underlying hits the stock signal's stop or target** ("the stock book would be exiting, why isn't the option?"); (5) **stock time-exit date reached**; (6) earnings drifted inside remaining DTE; (7) SHORT ITM leg with ex-div before expiry and extrinsic < dividend → red "close or roll by tomorrow's close", fired 2 td before ex-div; (8) the mirror: deep-ITM LONG call with extrinsic < dividend → "exercise or sell before ex-div or forfeit the dividend"; (9) weekly expiry checklist for anything expiring within 5 sessions, auto-classified (safely OTM → let expire; fully ITM vertical → close, never take double exercise; between strikes → red must-act), with the operational truth stated plainly: assignment can trigger IBKR auto-liquidation of un-marginable positions at bad prices, especially in the small account, so short ITM legs get closed before the event rather than reasoned about after.
- **Calculation:** All rules on executable-close marks. Loser rule tied to the signal, never "roll to get back to even" on a dead thesis.
- **IBKR mechanism:** Pure computation over the position snapshot + underlying prices + earnings parquet, on the agent's poll timer plus once at 21:15 UTC on the existing cron chain; delivery via the existing email pipeline; ex-div dates from generic tick 456 on live lines, FMP dividends as the delayed-mode source (cross-check both, IB's forward projections lag announcements). Intraday alerts only fire while the agent is online; the nightly pass is the guaranteed floor.
- **Delta-edge role:** The stock system's exits are mechanical; a discretionary "check the options book when I remember" step is where the edge leaks. Alerts 4 and 5 enforce the exact exit script the backtest validated. Expiry weekend is the only time this book can spawn surprise stock positions with no signal involved; the checklist makes it three known actions.

### 12. Options Data Spine: Snapshot Recorder + Journal + Fill Reconciliation
- **Displays:** Mostly invisible; surfaces as a journal table (filterable, per-strategy aggregates) and a nightly options section in the existing report email (EOD marks, P&L change, % of max, DTE countdown, alerts fired, fills/expiries reconciled).
- **Calculation/contents:** (a) **Daily position snapshot recorder**: append-only parquet, per-leg rows: date, conid, strike/expiry/right, bid/ask/mid, modelGreeks (delta/gamma/theta/vega/IV), underlying spot and IV30. Precise justification: IBKR serves no historical per-contract greeks/IV ever, and price bars only while a contract is listed (nothing for expired options), so attribution, entry-IV, and aging history exist only from the day recording starts. (b) **Per-scan vol-state appender**: tiny row per chain snapshot (ATM IV per expiry, RR25, straddle EM, term slope) to accrue the skew/term baselines no vendor will backfill. (c) **Journal**: one row per spread lifecycle, auto-populated at entry (source strategy + signal date, strikes/expiry, debit, contracts, target vs effective bps, entry IV + percentile, fill quality) and exit (reason, value, $ and R, days held, coarse delta-vs-extrinsic attribution split). (d) **Fill reconciliation, the options verify_fills**: nightly diff of agent-submitted combos against IBKR executions/positions, catching partial fills, unnoticed expiries, and assignment outcomes. This is a must, not a nicety: a partially filled combo or an expiry-weekend assignment is exactly the silent-failure class the stock pipeline's architecture exists to prevent.
- **IBKR mechanism:** book_snapshot.py subprocess pattern; post-close run on the 21:15/21:30 UTC chain uses frozen data (`reqMarketDataType(2)`) or the last regular-hours cycle; `ib.portfolio()` diffs for reconciliation (Flex queries later if fill-level detail is wanted). **All of it runs on the local agent and uploads to R2**; fail-open with notice when the machine is off.
- **Delta-edge role:** Every learning loop in this project (which strategies translate well, attribution, the kill/keep verdict) reads this file. Recording from day one is the difference between having answers in three months and never.

---

## (c) SHOULD-HAVE (next wave)

1. **Options-vs-stock counterfactual scorecard.** For every journaled trade, the same signal's stock P&L at equal risk from the existing ledger conventions; per-trade difference and cumulative by strategy and by IV-percentile-at-entry bucket. This is the number that decides whether the sleeve survives; it ships second only because it needs trades to accrue. Note the entry-parity caveat from feature 2: the comparison is only valid because both sides entered under the same condition.
2. **Structure recommender (rules-first verdict line).** Composes IV percentile, edge ratio, skew, earnings flag, and borrow into a one-sentence verdict with visible reasoning. Parameterized **per strategy from the measured IV-percentile-at-entry distribution in his own ledger** (computable once the IV cache exists), not from the textbook win-rate × IV matrix, which contradicts itself here: his mean-reversion fires at high IV by construction, so the "60–70Δ stock replacement at low IV" prescription rarely applies to it. "Use stock" is a first-class verdict.
3. **P&L attribution waterfall.** First-order Taylor split (delta·dS + ½gamma·dS² + vega·dIV + theta·dt + residual) from consecutive spine snapshots, per position and rolled up by strategy; flag days with |residual| > 25% of |total|. Forward-only by construction; the display ships once ~2 weeks of snapshots exist.
4. **Skew panel: RR25 + smile per expiry.** RR25 = IV(25Δ put) − IV(25Δ call), normalized by ATM IV; smile plot from the quoted band. Skew-aware defaults: steep put skew → put verticals over outright puts for shorts (his OVS/bear-fade shorts fire exactly when downside is most bid); inverted call skew → call verticals over outright calls for breakout longs. Percentile vs the name's own history is a **partial**: unranked at launch, usable ~3 months after the vol-state appender starts recording.
5. **T+0 payoff curve with date + IV sliders.** Client-side BSM re-pricing (same dividend/rate inputs as feature 6), date slider snapped by default to the strategy's median hold, IV slider −10 to +10 pts. The scenario grid carries the decision; this is the intuition-builder.
6. **VRP history per ticker.** IV30_t vs realized RV over the FOLLOWING 21 td from the IV cache + master_prices; rolling line, mean, and a hit-rate tile ("IV overpaid the next 21d in 74% of days, avg +3.1 pts"); last 21 days greyed (incomplete). Names with persistently positive VRP make credit structures the systematically better delta vehicle.
7. **Borrow/HTB input for short expressions.** Borrow rate via IBKR SLB data / `shortableShares` on the underlying. Two uses: locate-free short delta is one of the strongest reasons to express his shorts (3x ETF fades, OVS) in options at all; and borrow embeds into put prices via parity, so on HTB names the comparator will read long puts as overpriced without knowing why. Feed it to the recommender: HTB → put spread or call-side credit, never outright puts.
8. **Index-product routing + tax fields.** For index signals (^GSPC→SPY, ^NDX→QQQ aliases), prefer XSP/SPX/NDX: European exercise (no assignment/pin risk), cash-settled, §1256 60/40 treatment. Journal records enough per trade (dates, underlying, direction) to hand a CPA the wash-sale picture, since the scanner re-enters the same tickers within 30 days routinely.
9. **Position IV context sparkline.** Entry IV vs current per position (from the spine) over the underlying's IV history, with a vega-windfall alert (long premium whose IV rose > 20% relative: "consider harvesting, attribution says $X of this P&L is vol").

---

## (d) EXPLICITLY EXCLUDED

- **Per-contract historical IV/greeks features of any kind** — IBKR does not serve them under any subscription; the snapshot recorder (feature 12) is the alternative, forward-only.
- **Full option-chain grid browser** — he arrives with a ticker and a direction from the scanner; 4–8 strikes matter, never 90 rows.
- **Structure optimizer over thousands of combos (OptionStrat-style) / Probability Lab clone** — a fixed menu of 5 structures against his own ledger distribution answers the question; Probability Lab has no API surface anyway.
- **GBM Monte Carlo P50** — his ledger's actual exit distribution is strictly better; a second, worse probability model would only disagree with the first.
- **Probability cone chart overlay** — duplicates the expected-move bands already in features 3 and 5.
- **3D greeks-vs-time surface** — the 2D T+0 curve with a date slider covers the same decision at a fraction of the UI cost.
- **Beta-weighted delta alerting/thresholds** — premature machinery for a sleeve starting at a handful of verticals; the display number stays, the alarms don't.
- **Vol-surface fitting, dispersion analytics, calendars/diagonals as core structures, delta-hedged gamma books** — vol-arb tourism; he is buying delta, not trading the surface.
- **ORATS-style earnings implied-vs-historical report** — the book is earnings-averse by design; the lite crush quantifier in feature 8 covers the real need.
- **Streaming real-time chain / OPRA vendor feed** — on-demand snapshots with staleness badges fit the workflow and the budget; the mitigation is honesty about age, not more data.
- **Multi-step ticket wizards, per-leg-only pricing, expiry-payoff headline framing** — the named anti-patterns; the combo NBBO and the P&L-at-target-by-time-exit are the decision numbers.

---

## (e) Page Layout

**Existing Options tab becomes the Workbench + Today surface (options.html), one page, anchor links:**

Above the fold:
1. **Header status strip** — agent online/offline, live/delayed/frozen badge, last snapshot age, IV-cache freshness.
2. **TODAY panel** — this morning's staged signals with viability badges (feature 1) and Express buttons (feature 2); tickets staged-for-open awaiting 9:30 pricing with their entry conditions shown.
3. **OPEN OPTIONS BOOK summary** — compact version of the positions table with alert badges; fired alerts pinned to the top as the action queue.

Below the fold:
4. **WORKBENCH** — opened by an Express button or ad-hoc ticker entry: IV context strip (4) → expiry picker (5) → edge-vs-priced comparator (3) → structure shootout (6) → sizing panel (7) → earnings panel (8), flowing into the ticket. The current 40Δ/20Δ quoter is absorbed here as two preset rows of the shootout.

**Execution tab** keeps the order path: the combo ticket + auto-walk (feature 9) lives beside the existing stock/futures bracket flow, sharing the gated agent submission, the daily-cap accounting, and the exposure write-back. The workbench's "Submit" hands the built ticket here.

**New view: Options Book (options_book.html)** — the full management surface: positions table with mark hygiene and aging (10), action queue (11), expiry-week checklist, then collapsed REVIEW section: journal table, counterfactual scorecard, attribution waterfall. Alert emails deep-link `options_book.html#pos=XYZ` with the closing ticket pre-filled.

---

## (f) Open Questions

1. **Re-signal/ladder policy:** when a (ticker, strategy) re-signals while a vertical is open (OLV's 0.85/1.00/1.15 rungs), does the options sleeve add a second spread at new strikes, widen the existing one, or skip? This will hit in week one on any oil-cluster event.
2. **Which account trades the sleeve** (execution vs execution_2), and is it Reg-T or Portfolio Margin? This changes whatIfOrder results, which credit structures are capital-efficient, and how scary assignment-driven auto-liquidation is.
3. **Replace or coexist:** on a signal expressed in options, is the stock trade skipped, halved, or run alongside (doubling the delta)? The sizing write-back and the counterfactual both need this defined.
4. **Confirm debit-at-risk as the R definition** for the sleeve, with stop-based sizing as labeled opt-in only. If you want stop-based as default for any strategy, say which and accept the model-risk caveat.
5. **Pilot scope:** which 2–3 strategies go first? Suggested filter: liquid-universe only, options-viable tickers, holds ≥ 5 td, and at least one high-IV-entry mean-reversion strategy plus one breakout so the debit/credit branches both get exercised early.

---

## Appendix: IBKR Feasibility Assessment

### On-demand chain snapshot with per-strike greeks/IV (delta-based strike selection, smile plot, RR25 skew, prob-ITM columns)
- **Feasible:** yes
- **How:** Exactly the existing option_quote.py pattern: reqSecDefOptParams for expiries/strikes (filter tradingClass==symbol to avoid adjusted classes like 2TSLA), qualifyContracts on a strike band, reqTickers + ib.sleep(~3s) for modelGreeks (delta/gamma/theta/vega/impliedVol per contract, IBKR-computed via tickOptionComputation). Prob-ITM ~ |delta|, prob-touch ~ 2x|delta| computed client-side. RR25 needs only ~4 contracts per expiry, trivially inside the band already quoted.
- **Constraints:** ~3-5 s per expiry for a ~44-strike x calls+puts band; default ~100 simultaneous market data lines (option_quote.py's CAP=44 keeps 88 contracts under it — quoting 2+ expiries in one connection must be sequential, cancelling lines between batches). modelGreeks arrive asynchronously and can stay None on illiquid legs (retry or compute BS greeks locally from mid+spot). Works on delayed data (see delayed-greeks row).

### T+0 / intermediate-date payoff curve with date slider + IV-shift slider
- **Feasible:** yes
- **How:** Pure client-side Black-Scholes re-pricing in JS off the chain snapshot's per-leg modelGreeks.impliedVol, spot, and DTE. No additional IBKR calls. Date slider = re-price at t<T; IV slider = bump each leg's sigma. Rate input can be a constant or pulled once from IBKR (^IRX equivalent already in the stock pipeline).
- **Constraints:** Accuracy limited by snapshot staleness and by using each leg's own IV (skew-consistent); American-exercise/dividend early-exercise effects ignored by plain BSM — fine for the short-DTE OTM/ATM verticals in scope, less exact for deep-ITM stock-replacement calls on dividend payers.

### Expected move per expiration (ATM straddle x 0.85 / 1-sigma bands) vs the strategy's backtested move
- **Feasible:** yes
- **How:** Straddle mid from 2 contracts per expiry (ATM call+put) in the same snapshot; 1-sigma cross-check = S x IV_ATM x sqrt(DTE/365) from modelGreeks IV. Backtested side comes from backtest_trades_full.parquet (avg MFE, median move, hold) — zero new data. Multiple expiries = ~2-6 extra contracts each, one batched reqTickers.
- **Constraints:** Each additional expiry costs qualify+quote time (~1-2 s marginal); wide markets on overflow names make the straddle mid noisy — display mid AND natural. Delayed mode: quotes are 15-20 min old, label accordingly.

### IV Rank / IV percentile per underlying (252d) — the debit-vs-credit gate
- **Feasible:** yes
- **How:** reqHistoricalData on the UNDERLYING Stock contract with whatToShow='OPTION_IMPLIED_VOLATILITY', barSize='1 day' — this is the documented path (IBKR Quant Blog parts I/II; TWS API historical_bars.html lists it as a valid whatToShow). Returns IB's 30-day ATM-interpolated IV series. Duration strings up to multi-year ('1 Y', '2 Y') are accepted, so a one-shot backfill seeds the 252d window; then a nightly append job (update_master_prices.py pattern) maintains an append-only parquet in R2. Rank/percentile computed client-side.
- **Constraints:** Historical data generally requires a live market data subscription for that underlying in the funded account; ib_insync issue #458 documents 'No historical market data' errors for some symbols — coverage must be probed per ticker (expect gaps in overflow-tier names; fail-open with a 'no IV history' badge). Pacing: max 60 historical requests per 10 min, <=6 identical per 2 s — a ~190-ticker backfill takes ~35-40 min paced; nightly incremental is 1 request/ticker.

### IB historical volatility series (30d HV) as RV cross-check
- **Feasible:** yes
- **How:** Same reqHistoricalData call with whatToShow='HISTORICAL_VOLATILITY' (also exposed live as generic tick 104). But the better source is free: Yang-Zhang RV at 10/21/63d computed from master_prices.parquet OHLC — zero API cost, already-maintained data. Use IBKR HV only as a sanity cross-check.
- **Constraints:** Same subscription/pacing caveats as the IV series if pulled from IBKR; the master_prices route has none. Note master_prices is dividend-ADJUSTED — fine for vol estimation.

### Vol risk premium history (IV30 vs subsequent RV21) per ticker
- **Feasible:** yes
- **How:** Pure derivation from the IV-history cache (row above) + master_prices RV: VRP_t = IV30_t - realized RV over the following 21 td. No additional IBKR calls.
- **Constraints:** Last ~21 days incomplete by construction (grey out); quality inherits the per-symbol IV-history coverage gaps.

### Term structure curve + forward vol between expiries; DTE picker annotated with hold coverage
- **Feasible:** yes
- **How:** ATM IV per expiry from ~2-4 contracts x N expiries in one batched snapshot (qualify + reqTickers, modelGreeks.impliedVol); forward vol via variance additivity client-side. Expiry list from reqSecDefOptParams (replaces the hardcoded next-6-Fridays dropdown — secDefOptParams returns actual listed expirations including non-Friday weeklies). Hold_Days/Time_Exit_Date from the staged signal, earnings marker from earnings_calendar.parquet.
- **Constraints:** Quoting 8-10 expiries x 4 contracts = ~40 lines, one batch, ~5-8 s total. ATM IV from a single strike pair is noisier than a fitted ATM — acceptable for slope/kink detection.

### Skew/RR25 percentile vs the name's own history ('put skew 1.9x normal')
- **Feasible:** partial
- **How:** Current RR25 is trivially computable per snapshot (see chain row). The NORMALIZATION has no data source: IBKR serves no historical per-expiry IV or skew. Only path is self-accrual — append a tiny row (ticker, date, ATM IV per expiry, RR25, straddle EM, term slope) every time a chain snapshot is taken, rd2_fragility-style append-only parquet.
- **Constraints:** Unranked at launch; a usable 60d baseline exists only ~3 months after shipping the recorder. Ship the raw RR25 immediately, add the percentile when history accrues.

### Per-contract historical option bars / per-contract IV & greeks history
- **Feasible:** no
- **How:** Historical greeks/IV per option contract: not served by IBKR under any subscription. Historical price bars: available ONLY while a contract is listed (TRADES/MIDPOINT via reqHistoricalData) and explicitly NOT for expired options (TWS API historical_limitations.html: 'Expired options... are not available for historical data'). So no retroactive reconstruction of past positions or past skew is ever possible. Consequence: the daily snapshot recorder (next row) is a hard prerequisite for every attribution/aging/entry-IV feature.
- **Constraints:** Listed-contract bars are pacing-heavy (60 req/10 min) and need OPRA-level market data permission for live; usable for backfilling an open position's recent marks, useless for anything expired.

### Daily position snapshot recorder (marks, per-leg modelGreeks, IV, spot) — enabler for attribution/aging/entry-IV
- **Feasible:** yes
- **How:** Extend the book_snapshot.py subprocess pattern: ib.portfolio() for option positions (secType OPT rows already come through), then qualify their conIds and one batched reqTickers for bid/ask/modelGreeks per leg + the underlying spot; append rows to data/option_position_snapshots.parquet (append-only, R2). Run post-close on the existing 21:15/21:30 UTC cron chain plus each agent poll cycle intraday if wanted.
- **Constraints:** ~2 contracts per vertical — line limits are a non-issue for a realistic book (<50 legs). After-hours snapshots get frozen/stale quotes: use reqMarketDataType(2) (frozen) for the post-close run or record the last regular-hours cycle. Greeks can be None on illiquid legs — retry then fall back to local BS from mid.

### P&L decomposition (delta/gamma/theta/vega waterfall) per position and book
- **Feasible:** yes
- **How:** First-order Taylor attribution from consecutive rows of the snapshot parquet (recorder above): delta x dSpot + 0.5 gamma dSpot^2 + vega x dIV + theta x dt, residual = unexplained. No live IBKR dependency beyond the recorder.
- **Constraints:** Only exists FORWARD from the day the recorder ships — no history before that, ever (see per-contract history row). Attribution is approximate; flag days where |residual| > ~25% of |total| (wide-market marks).

### Option volume and open interest per contract (liquidity columns)
- **Feasible:** partial
- **How:** reqMktData with genericTickList='100,101' on the option contract: 100 returns call/put option volume (tick types 29/30), 101 returns call/put open interest (tick types 27/28) — verified against the TWS API tick-types page; ib_insync exposes them as ticker fields. Same-contract day volume also arrives as standard tick 8 (ticker.volume).
- **Constraints:** OI updates only once daily (OCC morning file), so it is yesterday's OI intraday. Generic ticks are NOT reliably served on delayed data lines (delayed mode delivers the delayed core ticks 66-76; treat OI/option-volume as live-subscription-only and degrade to bid/ask-width-based liquidity scoring on delayed). Adds ticks to already-open lines, no extra line cost.

### Mid vs natural pricing, spread-tax-on-edge readout, quote staleness badges + targeted re-quote
- **Feasible:** yes
- **How:** All from per-leg bid/ask in the snapshot (option_quote.py's _mid already does the mid; natural = buy-the-ask/sell-the-bid combination). Staleness = timestamp the snapshot in the agent JSON; re-quote = reqTickers on ONLY the on-screen conIds (2-8 contracts, ~1-2 s), not the whole band. ticker.marketDataType field distinguishes live vs delayed per ticker for honest badging.
- **Constraints:** None material; delayed mode must force the 'do not price orders off these marks' banner.

### Native combo (BAG) spread quote for the true executable net market
- **Feasible:** partial
- **How:** reqMktData on a Bag contract with ComboLegs (leg conIds from qualification) returns a TWS-computed combo bid/ask derived from leg markets; documented in TWS API 'Spreads' page. Useful as the order-anchoring number vs the leg-sum mid.
- **Constraints:** Quirks verified: snapshot=True is not supported for BAG, tick-by-tick explicitly rejects BAG, actively-trading combos can legitimately tick price=0 with positive size, and combo quote quality on DELAYED lines is unreliable — mark unverified-on-delayed. Fallback (already in hand): compute net from leg quotes. Combo display increment 0.01 but order minTick may be 0.05.

### Combo order execution: vertical as one native BAG limit order with auto-walk mid-to-natural and fill-quality logging
- **Feasible:** yes
- **How:** placeOrder on the same Bag contract, orderType='LMT', SMART routing, via the existing gated non-readonly execution path (the agent already places stock/futures brackets). Auto-walk = timer loop in the agent doing cancel/replace stepping the limit toward natural with a %-of-spread cap; fill-quality = fill price vs mid-at-submit, logged to the journal parquet. Guaranteed atomic — SMART combos never leg you into a partial structure at IBKR's account level (non-guaranteed only if explicitly flagged).
- **Constraints:** Requires options trading permissions on the account (assumed); combo LMT minTick rules (often 0.01 quote / 0.05 order on some classes — round the walk steps accordingly); OCA/bracket children on combos are more limited than stock brackets, so exits should be managed by the alert engine + closing combo tickets rather than resting bracket legs.

### Delayed-data fallback behavior for the whole options surface (greeks, quotes, badging)
- **Feasible:** yes
- **How:** reqMarketDataType(3) (already in option_quote.py). Delayed bid/ask/last arrive as tick types 66-76; delayed option COMPUTATIONS (bid/ask/last/model greeks + IV) exist as the delayed tickOptionComputation variants and ib_insync maps them into ticker.modelGreeks etc. (delayed-greeks support merged in ib_insync PR #53) — so strike-by-delta selection and BSM re-pricing keep working delayed. ticker.marketDataType tells the UI which mode each line is in.
- **Constraints:** 15-20 min old marks — fine for structure selection and analytics, NOT for anchoring limit orders (require extra confirm). Generic ticks (OI/option volume/106/104 live ticks) and some tick types don't populate delayed. Historical data requests are a separate permission axis: delayed historical (type 3/4 before reqHistoricalData) works for many stocks but is not guaranteed per symbol.

### Earnings flag/countdown per expiration + event-vol (crush) quantifier
- **Feasible:** yes
- **How:** Earnings dates: zero IBKR dependency — data/earnings_calendar.parquet (FMP, nightly, already drives the OVS blackout). Expiry-straddles-earnings flag is a date join. Crush quantifier: two-expiry event-variance extraction (front IV^2*T minus post-event forward baseline) from the same per-expiry ATM IVs the term-structure row already fetches; vega-$ impact from modelGreeks.
- **Constraints:** Earnings parquet covers ~946 tickers — same coverage gaps as OVS (indices/ETFs pass through as no-event). The decomposition is the simple 2-expiry version, not a fitted ORATS-style model — label as estimate.

### Dividend / early-assignment risk flags on short ITM legs
- **Feasible:** yes
- **How:** Next ex-div date + amount via generic tick 456 'IB Dividends' on the underlying stock line (returns past-12mo total, next-12mo expected, next ex-date, next amount) — verified on the tick-types page; FMP is an alternative source already in the pipeline. Extrinsic-vs-dividend test needs only the short leg's mid and intrinsic (chain snapshot) plus the same-strike put price (put-call-parity proxy) — 1 extra contract per flagged leg.
- **Constraints:** Tick 456 is a live-line generic tick — on delayed fallback use FMP dividends instead. IB's forward dividend projections can lag announcements; cross-check both sources for the 2-day-warning alert.

### Options positions book: per-position greeks table, book-level dollar-delta / beta-weighted SPY delta / theta / vega buckets, signal linkage, DTE-vs-hold aging
- **Feasible:** yes
- **How:** Positions from ib.portfolio() (book_snapshot.py already returns OPT rows with mark + unrealized PnL); one batched reqTickers on the option conIds for modelGreeks; betas regressed from master_prices.parquet vs SPY (no new data); strategy/signal linkage carried in the journal parquet at entry (the ticket knows its source signal); hold_days/Time_Exit_Date from STRATEGY_BOOK / staged rows. Aggregations are client-side arithmetic.
- **Constraints:** A realistic book (10-30 legs) is well under line limits; refresh is one agent subprocess cycle (~5-10 s). ib.portfolio()'s own marketPrice can be a stale/frozen mark — prefer the leg-mid recomputation for headline P&L (next row).

### Spread mark hygiene: leg-mid marking, executable-close estimate, one-sided-quote badges
- **Feasible:** yes
- **How:** All from the same batched leg quotes: spread mid = sum of leg mids from one snapshot cycle; executable close = long legs at bid / short legs at ask; badges from leg bid-ask width, zero-bid detection, and ticker.marketDataType. No additional IBKR mechanism needed.
- **Constraints:** None beyond snapshot cadence; after-hours cycles should freeze the last regular-hours marks rather than mark to one-sided books.

### Rule-based alert engine (% of max profit, 21 DTE, short-strike touch, stock stop/target/time-exit hit, earnings drift into DTE) + EOD reconciliation email
- **Feasible:** yes
- **How:** Pure computation over the position snapshot + underlying prices (master_prices / live spot from the stock line) + earnings parquet, evaluated on the agent's poll timer and once on the existing 21:15 UTC cron; delivery via the existing email pipeline. No new IBKR capability required. Expiry reconciliation mirrors verify_fills.py using ib.portfolio() diffs (or Flex queries if fill-level detail is wanted later).
- **Constraints:** Intraday alerts only fire while the local agent is online (TWS session dependency — no cloud path to IBKR); alert styling must downgrade on stale/delayed inputs.

### Structure shootout / cost-of-delta table, EV-under-ledger-distribution ranking, scenario P&L at planned exit, risk-bps-to-contracts sizing, options-vs-stock counterfactual scorecard
- **Feasible:** yes
- **How:** All derivation, no new data: one chain snapshot supplies every candidate structure's legs (long 30/40/60d, verticals, credit verticals, risk reversal — all strikes already inside the quoted band); probability weights and target/stop/hold scenarios from backtest_trades_full.parquet + STRATEGY_BOOK; sizing from the staged Risk_Amt (Sheets rows); counterfactual stock P&L from the existing ledger conventions. BSM re-pricing at exit date client-side.
- **Constraints:** Credit-structure margin numbers are estimates unless whatIfOrder() is called per structure (1 extra API round-trip each — cheap, worth doing for the risk-reversal row since its downside is undefined). EV ranking is only as honest as the ledger's per-strategy move distribution.

### P50 / managed-exit probability (probability of touching 50% of max profit within hold)
- **Feasible:** yes
- **How:** Client-side Monte Carlo (GBM at the structure's snapshot IV) or, better, replay of the strategy's own ledger exit distribution — no IBKR data beyond the snapshot. Note IBKR's own Probability Lab / Strategy Scanner is a TWS UI feature with no API surface, so this must be rebuilt, not fetched.
- **Constraints:** GBM-based P50 ignores skew/jumps — label as approximate; ledger-replay variant is limited to strategies with enough trades.

### Probability cone / expected-move cone overlaid on the existing signal candlestick charts
- **Feasible:** yes
- **How:** Cone = spot x exp(+/- z x IV_ATM x sqrt(t/365)) drawn client-side from the per-expiry ATM IVs already fetched for the term-structure row; candlesticks from master_prices. Purely presentational once those inputs exist.
- **Constraints:** Cone uses one IV per expiry (no smile adjustment) — fine for its visual strike-placement job.
