# Cell map — run 2026-09-14 (Monday)

asof session 2026-09-14 (Mon) -> next session 2026-09-15 (Tue). Midterm year, September.
prices_fresh = True (core bars 2026-09-14). Sweep: 1217 scanned, 110 fired (72 event / 38 price),
BH pass 4 at crit p 0.0023. Caps dropped ^FVX (P4), HYG/IEF/^FVX (P5), AUDJPY/^TNX/GBPJPY/JPY/IEF/^FVX (P5b):
all of those are the rates and yen complexes already represented, nothing unique lost.
Warnings: 6 tape tickers without a 09-14 bar (LBS=F, ^AXJO, ^HSI, ^KS11, ^N225, ^SKEW) — Asian cash
indices are price-lane only anyway; EEM's -2.73% cannot be cross-checked against ^HSI/^KS11 tonight.

Tape tonight: S&P -0.48%, 21d rank 7; QQQ -0.80%; IWM 21d -5.1%. VIX +7.95% to 17.10 (the Monday before
FOMC, last night's headline cell, printed). EEM -2.73%. IEF and LQD at 252d lows, TLT 0.19% above its low.
10y 4.961% / 5y 4.79% at 252d highs, ^IRX at a 252d high (rank 100 on 5/21/63d). Crude 101.89, z10 +2.51.
Yen crosses z10 -2.2 to -2.9 across six pairs. Coffee -7.78%, corn +4.65%, sugar +5.01%, gold -1.56%.

Repetition ledger from the journal (last ~8 briefs): pre-FOMC S&P drift (09-09), yen crosses x5 briefs
(ATR breadth, 5d-rank breadth, USDJPY/EEM, USDJPY/10y), CAC 200d break (09-09), bond three-way low (09-10),
IEF near low before PPI, IRX 63d rank with S&P near high, IWM 21d weak with SPY near high, VVIX spike
(09-10), crude +8% (09-10), VIX Monday-before-FOMC (09-13 headline), 5y surge into decision (09-13),
NG September Mondays (09-13), VIX 10% Friday drop (09-13).

## Calendar, next 5 sessions
| date | entry | verdict |
|---|---|---|
| Tue 09-15 | nothing scheduled | calendar line only |
| Wed 09-16 | FOMC decision 14:00 ET (top tier, k2 tonight) | DRILL (bond leg of the eve session; equity drift and 5y surge are repetition-blocked by content) |
| Wed 09-16 | VIX expiry, 09:30 settlement (k2 tonight) | DRILL (SPY/QQQ cell, split FOMC overlap vs not, Tuesday control) |
| Thu 09-17 | nothing scheduled | calendar line only |
| Fri 09-18 | monthly opex + quad witching (k4, outside the k1..3 lane) | SKIP(outside the event lane window; calendar line only, the post-opex week belongs to next week's briefs) |
| Mon 09-21 | nothing scheduled | calendar line only |

## Event lane
| trigger | subjects / cell | verdict |
|---|---|---|
| E:fomc_decision k2 | TLT +0.151% t 2.57 (solid hint), IEF +0.071% t 2.53 (solid hint), ^TNX -0.31% t -2.26 | DRILL: the eve-of-decision bond bid, conditioned on bonds entering at a 252d low / bottom-decile 5d. SWEPT cell (bh_pass false, not a famous pre-specified hypothesis for Treasuries), so [solid] is not available on the base cell. |
| E:fomc_decision k2 | SPY/QQQ/IWM/^GSPC +0.08% t < 1 | SKIP(the equity pre-FOMC drift is the pre-specified Lucca-Moench hypothesis, published 09-09 and last night's 5y item already carried the S&P decision-day record; nothing new at t 0.9) |
| E:fomc_decision k2 | EURUSD 103-77 up sign p 0.031, DX flat | SKIP(dollar index shows nothing, -0.03% t -0.89; EURUSD hit rate without a mean, t 1.55, fails BH) |
| E:fomc_decision k2 | SI=F, HG=F, NG=F, CL=F, GC=F, JPY, HYG, EEM, ^VIX | SKIP(all |t| < 1.3, no sign p under 0.09; VIX era-unstable and phantom-bar affected) |
| E:vix_expiry k2 | SPY +0.187% t 2.84 (solid hint, published 08-17 at 0.19), ^GSPC t 2.64, QQQ bh_pass (188-130, sign p 0.001), IWM t 2.49 | DRILL: overlap with FOMC eves inflates it? control against all Tuesdays (turnaround-Tuesday base), split by FOMC week. Re-telling needs new specificity after 08-17. |
| E:vix_expiry k2 | ^VIX -0.31% (147-173), ^TNX, CL, HG, JPY, GC, NG, TLT, SI, HYG, DX, EUR, IEF | SKIP(no |t| above 1.4; VIX leg has phantom bars and last night's footnote already dropped the decision/expiry VIX overlap) |
| E:weekday_month | Tuesdays in September: ^VIX +1.90% 68-44 (sign p 0.015), CL=F -0.56% t -2.24, HG=F 42-68 (sign p 0.015), SPY -0.15% t -1.33 | SKIP(bare weekday x month cells, 18 subjects swept, none passes BH at 0.0023; the VIX leg carries the known 2026 phantom-bar fault and a generic Tuesday effect; no mechanism for copper or crude on September Tuesdays) |
| E:seasonal_doy | HG=F midterm h1 6 of 6 down, sign p 0.0156, mean -0.62% | SKIP(one of 72 midterm DOY cells; ~2 such records expected by chance at that p; copper has live roll-seam faults (137x vol 09-10); no mechanism) |
| E:seasonal_doy | GC=F h5 18-7 up sign p 0.022; DX h5 18-8 sign p 0.038; EEM h5 16-7 sign p 0.047 | SKIP(all-years DOY h5 records fail BH; gold has a roll seam tonight (vol 14x 63d) and seasonal gold was dropped for seams twice this week) |
| E:seasonal_doy | NG=F | SKIP(repeat_blocked, published 09-08) |
| E:seasonal_doy | TLT (published 09-03), ^TNX midterm 5-1 down, ^VIX midterm 4-2, SPY/QQQ/IWM/^GSPC/IEF/HYG/SI/CL/EUR/JPY | SKIP(no record better than sign p 0.105; TLT re-telling adds nothing) |

## Price lane
| trigger | subject | verdict |
|---|---|---|
| P1 / P1b new 52w high (30+/90+) | ^IRX, n 19 / n 10, era-unstable | DRILL (folded into 02 as a conditioning read: does a bill yield at a fresh high into a decision keep rising through it?). Alone it is DEAD for publication: yield percent changes in the ZIRP era are degenerate and n=10. |
| P3 down 50bp after a 52w high | EWJ n 115, +0.19% t 1.62, sign p 0.096 | SKIP(generic reversal cell, t 1.62 fails BH, and the yen leg behind it is repetition-held) |
| P4 z10 stretched up | ^IRX n 241 | SKIP(percent-of-yield basis, edge -1.46% is ZIRP noise) |
| P4 z10 stretched up | CL=F n 175, h1 +0.01% | SKIP(null cell; crude published 09-10) |
| P4 z10 stretched down | CHFJPY, CADJPY, AUDJPY, EURJPY, NZDJPY, GBPJPY | SKIP(yen complex published in four consecutive briefs, held back 09-13; no single-pair cell above t 1.2; breadth versions already run: 4-of-6 at 2 ATR and 7-of-7 5d-rank) |
| P5 5d bottom 5% | EURJPY (bh_pass, 175-116 up, +0.10% t 1.49) | SKIP(BH passes on the hit rate but the mean is 10bp at t 1.49, and it is the same yen repetition) |
| P5 5d bottom 5% | CHFJPY, NZDJPY, AUDJPY | SKIP(same complex, |t| <= 1) |
| P5 5d bottom 5% | KC=F n 315, +0.24% t 1.89 | DRILL (01 tape verify: coffee printed roll seams 09-10 and 09-11; tonight's -7.78% must be checked for gap/volume before any use) |
| P5 5d top 5% | ^IRX, ^TNX | SKIP(percent-of-yield basis; TNX h1 -0.02% null; rates into FOMC is last night's item) |
| P5 5d top 5% | LE=F n 315 | SKIP(live cattle continuous contract has documented roll gaps; h1 +0.08% t 0.92) |
| P5b 21d top 5% | ^IRX, ZC=F (+0.18% t 2.02), ZS=F | DRILL for ZC=F in 01 (corn September expiry sits on 09-14, roll seam likely); ^IRX and ZS=F SKIP(null / yield basis) |
| P5b 21d bottom 5% | ^FCHI bh_pass (231-173 up, sign p 0.0023), era-unstable | SKIP(CAC published 09-09, held back 09-13, engine flags it era-unstable; overlapping-day hit rate, not declustered) |
| P5b 21d bottom 5% | NZDJPY, EURJPY, CHFJPY, CADJPY | SKIP(yen repetition) |
| P6 2 ATR down | KC=F n 54 | DRILL in 01 (roll-seam screen; almost certainly a contract change) |
| P6 2 ATR down | EURGBP=X n 15, +0.27% t 1.61 | SKIP(n 15, t 1.61, no follow-on, sterling cross has no link to tomorrow) |
| P6 2 ATR up | ZC=F n 183, h1 +0.10% hit 42% | DRILL in 01 (roll seam) |
| P7 5+ up closes | ^IRX (sign p 0.04, yield basis), ^MOVE (published 09-02, era-unstable), USDZAR (null) | SKIP(yield-basis cell; MOVE vol mean reversion already told via VVIX 09-10; USDZAR null) |
| P7 5+ up closes | USDTRY n 423 bh_pass | DEAD(degenerate: the lira devalues on ~70% of sessions by construction, the base rate is the whole "edge") |
| P7b 5+ down closes | IEF n 105, +0.08% t 1.93 | DRILL (folded into 02: IEF's streak and bottom-1% 5d rank are the conditioning state for the eve-of-decision bond cell) |
| P7b 5+ down closes | HYG n 124 null, era-unstable; NZDJPY; NZDUSD null | SKIP(null cells, yen repetition) |

## Tape extremes not covered by a trigger (dismissals and drills)
| state | verdict |
|---|---|
| ^VIX +7.95% on an S&P decline of 0.48%, two sessions before a decision | DRILL (04): direct follow-on to last night's headline cell, which printed. What does the eve session do after a pre-decision Monday lift of 5%+? And the general "VIX +7% on a sub-0.5% S&P decline" cell. NYSE calendar only (phantom bars). |
| EEM -2.73% against SPY -0.45% (a 2.3pp one-session gap, not a trigger) | DRILL (05): EM-specific underperformance days, declustered, forward EEM and EEM-minus-SPY. Note EWJ -0.99, EWZ -1.23, Asian cash bars missing tonight. |
| Crude 21d +23.7% at $101.89 into a decision | DRILL (06): FOMC decisions that arrive after a 20%+ crude month. Expect anecdote-scale N. |
| Gold -1.56% with volume 14x its 63d norm | DRILL in 01 (roll seam, gold seams flagged 09-10 and 09-13) |
| Sugar +5.01% at 52w high | DRILL in 01 (roll seam check; soft commodity, context value low even if clean) |
| Breadth 58.6% above 200d vs 66.7% 21d ago | SKIP(P11 did not fire; a slow drift, no threshold crossing) |
| HYG 5d bottom 5% (dropped by cap), LQD at 252d low | SKIP(rates-driven credit, HYG only 2.2% off its low; bond lows published 09-10) |
| SPY 21d rank 8, IWM 21d rank 6 within 2-6% of highs | SKIP(the IWM-weak/SPY-near-high cell published this month) |
| BTC +2.70%, ETH +3.25% on a risk-off equity day | SKIP(crypto weekend-to-Monday prints on a 24/7 calendar; not a trigger, no clean anchor) |

## Drill outcomes (written after stage C, before composing)
| drill | result | final verdict |
|---|---|---|
| 01 tape verify | Coffee 542x volume, gap -9.60% of a -7.78% session; silver 451x, gold 383x, copper 60x; soybeans 14x, cattle 11x. Sugar (1.2x) and corn (3.2x) traded. EEM's drop is EWY -6.62% and EWT -3.34%, FXI +1.01%. ^VIX phantom bar on 2026-09-07 confirmed. | KC=F P5/P6, GC/SI/HG cells DEAD (roll seams). Corn/sugar SKIP(soft commodities, low macro relevance). |
| 02 bonds eve of decision | Base eve TLT +0.151%, 109/192, vs Tuesdays +0.034%, era-stable. IEF 5d rank <= 10: TLT +0.36% 17/23 (top two 55%). IEF within 1% of 252d low: IEF up 3/15, TLT 5/15, -0.21%, 2/10 since 2018, same-state non-FOMC +0.06% (76 declustered); top two 72%. Tonight is in both; the 4 shared split 2-2. | PUBLISH [suggestive] as a split that reads both ways. Six conditioning cuts run; both published cuts are named. |
| 03/08 VIX expiry eve SPY | Base +0.187%/319 vs Tuesdays +0.065%. 2018+ not-FOMC-eve +0.256% (84, t 2.08); 2018+ also-FOMC-eve +0.075%, 9/19, median -0.11% (QQQ 12/19, +0.17%); pre-2018 double eves +0.48%/23. FOMC k2 with no expiry near: SPY +0.03%/170. | PUBLISH [suggestive]: new specificity over the 08-17 telling. Base cell swept; QQQ leg cleared BH, SPY did not. |
| 04/07 VIX pre-decision lift | k2, lift >= 7%, S&P > -1%: VIX lower by decision close 20/22, -5.73%, sign p 0.00006, 12/14 pre-2018, 8/8 since, top two 30%. Ladder: >=6% 24/29, >=8% 15/15, >=10% 8/8; band 5-7% only 7/19 lower. Controls: all k2 129/212 lower (-1.16%); same state non-FOMC 112/199 (-0.24%). Eve itself 12/22 lower. | PUBLISH [suggestive], headline. The 7% cut is quoted beside the 5-7% band that breaks it. Drill construction, not a swept p. Mechanism: event variance leaves the 30d window. |
| 05/08 EEM gap / Korea | EEM trails SPY 2pp+ with SPY down <1%: spread h5 +0.10%, 39/78 (null). EWY -5% with SPY down <1%: 32 since May 2000, nine in 2000, nine in 2026 incl. today, zero 2010-2025; EWY realized vol 56.6% vs 22.5% median. Forward h1 up 19/28, h5 lower 17/28. | PUBLISH [anecdote] as a rarity statement (n = nine 2026 sessions); the EEM gap cell SKIP(null). |
| 06 crude shock FOMC | Crude 21d >= 15%: 16 decisions, S&P decision close up 6/16, +0.32% t 0.60; >= 20%: 4 cases. | DEAD(N 4 at tonight's state, null at 15%). Footnote only. |
