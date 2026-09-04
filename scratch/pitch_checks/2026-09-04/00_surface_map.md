# Surface map — 2026-09-04

Run date 2026-09-04 (Friday). Prior session 2026-09-03. Freshest bar 2026-09-03.
Next session after today is **2026-09-08**: Monday 2026-09-07 is **Labor Day**, so
tonight is a three-calendar-day market closure. Cycle year **midterm** (2026 % 4 == 2).

Pipeline all green (7/7). One warning: 2 tape tickers stale (LEG, ^SKEW) — ^SKEW
matters below and its staleness is noted where used.

State readings that condition everything: fragility dial ma10-63d **87.8**
(raw 63d 85.7, raw 21d 63.1; the ma10 was 54.7 twenty-one sessions ago, so the dial
has risen 33 points in a month). Exposure leg **0.0x** (killed on raw-21d > 50),
trend sleeve **CASH**. P/C fear **off** at the 46th percentile. One fragility signal
on: **Dispersion**, composite 92nd percentile, component RV 30.1% against SPY RV 7.2%.
^VIX 14.32 at the **2.4th percentile of its own trailing year**, 21-day relative-range
percentile **3.35**.

---

## 1. Calendar events x asset classes

Eight events sit inside the [-5, +15] window. Nine live crossings plus one event the
state file does not carry.

| event | date | td | verdict by class |
|---|---|---|---|
| jackson_hole | 2026-08-28 | -5 | **DISMISS all classes.** Swept exhaustively 2026-08-28 across ten classes ("sweeping the conference close forward on ten asset classes": sign instability across the cycle plus family-wise multiplicity, 10 cells at abs(t)>=2 against an iid expectation of 10.5 on 210). The one clean member, long IEF one session out of the close, is watchlist 27 and is midterm-blocked. Anchor is five sessions gone. |
| **nfp** | **2026-09-04** | **0** | see below, the live anchor |
| ppi | 2026-09-10 | +3 | **DISMISS as an entry anchor.** The pre-print lane was swept in full yesterday across rates, energy, credit, gold, EM, vol and equities and every cell died. PPI's own runway is the registry's documented worst (median 2 td, 43.9% at <=1). It matters here only as a **runway boundary** for anything entered today: today's close has exactly 3 sessions of clear calendar. |
| cpi | 2026-09-11 | +4 | **DISMISS as an entry anchor, PARK as a watch.** Watchlist 2 (long SVXY overnight, MOC the eve to MOO on the print) arms at the 2026-09-10 close, five sessions out. Nothing to enter today. |
| fomc_decision | 2026-09-16 | +7 | **DISMISS.** Swept 2026-09-02 across fifteen asset classes (homogeneous family, Cochran Q 9.26 on 14 df at p 0.8138, I-squared 0.0%, no anchor at all); the midterm short at FOMC-10td died on the event sleeve's own rank gate; energy-into-a-midterm-FOMC died on the placebo ladder. |
| vix_expiry | 2026-09-16 | +7 | **DISMISS.** VIX-expiry-week drift swept and empty 2026-08-07 (within-month paired excess +0.065%, t 0.67, settle day itself the worst day). The FOMC/settle coincidence was killed 2026-09-02 as a calendar identity: 41 of 42 coincident decisions fall in Mar/Jun/Sep/Dec. |
| opex | 2026-09-18 | +9 | **CHECK, one form only.** The RUN INTO opex was swept for August 2026-08-07 and was worse than a random 10-day long, with the effect all in 2000-2004. September quad witching is the year's largest expiry and has never been separated from that August finding in this repo. Crossed with the small-cap laggard below → candidate C6. |
| quad_witching | 2026-09-18 | +9 | folded into C6. The POST-quad window is the event sleeve's T3 (short IWM, Sep opex MOC to Sep last MOC), so anything on the far side of the 18th is book territory and excluded by the no-re-run rule. |
| **labor_day** | **2026-09-07** | **+1 calendar anchor** | **NOT IN `macro_events.csv` AND NEVER OPENED IN THIS PRODUCT.** A three-calendar-day market closure starting at tonight's close. Every calendar object this registry has swept is a scheduled data release or an expiry; a market holiday is a different anchor with a different mechanism (calendar-day theta against session-count realized vol; pre-holiday inventory). → candidates C1 (vol) and C2 (equity direction). |

Cross with the ten classes. The payroll anchor is the only live one and it is the
anchor yesterday's run spent its whole morning on, from the other side:

| class | nfp x class verdict |
|---|---|
| us_large | **DISMISS.** Post-NFP equity DIRECTION swept and empty 2026-08-07 (NFP close to next CPI +0.129% on N=309 against an all-days control of +0.221%, sign flips between h=3 and h=5, the overbought conditioner NEGATIVE at -0.400%). The pre-print direction leg is watchlist 35, armed on the dial and out of range at 87.8. |
| us_small | **DISMISS on the print anchor**, carried into C6 on the expiry anchor instead. |
| rates | **CHECK, on a conditioner never applied.** Watchlist 0 (long TLT from the NFP close to +3td at the long-end floor) is midterm-dead and parks to 2027-01; the pre-print sibling died yesterday with the live k=-2 rung DEAD LAST of 17 placebo offsets. Neither has ever conditioned on the **labour data itself**. `data/macro_release_history.parquet` carries 163 NFP prints with actual, consensus and surprise from 2013, and this product has never once opened it. The last print was **-23k against +80k, a -103k miss**, the largest in the recent series. → candidate C3. |
| credit | **DISMISS.** Killed yesterday: the IG-at-a-low-with-HY-at-a-high conjunction crossed with a payrolls anchor has never occurred, and the conjunction is worth less than either leg alone at every vehicle and horizon. Watchlist 1 and 26 both PASS on episode counts today. |
| gold_miners | **DISMISS.** Killed yesterday: the joint metal-washout-with-miners-bid state crossed with a payrolls anchor has 2 days in 22 years, and its deep-drawdown half, which is where GLD sits at -17.28%, is 0-for-3. |
| metals | **DISMISS on the print anchor.** Silver's post-complex-break cell is watchlist 29 and the complex bounced. Metals are checked below on a positioning anchor instead (C4). |
| energy | **DISMISS.** Killed yesterday on the placebo ladder, which is now five-for-five: XLE's live k=-2 rung ranks 8 of 17 at h=3 against a best placebo of roughly three times it. Energy is checked below on a seasonal anchor instead (C10). |
| fx_dollar | **DISMISS.** The midterm split on the post-NFP dollar holds in BOTH vehicles (UUP -0.141% at a 37.5% hit, DX -0.184% at 41.7%) against non-midterm t-stats of 4.85 and 4.91; 2026 is midterm. Watchlist 23 parks the bare washout to a non-midterm year and DX is not washed out in any case (r21 33.1). |
| intl | **DISMISS.** EWZ into a payrolls print died yesterday on the reference class (3, 3, 5 and 2 of 10, never best) and "EWZ is EEM with a Brazil label" is now registry. FXI's break leg fails (r5 51.4). |
| volatility | **CHECK, and read the standing kill first.** The post-NFP vol cell is recorded as already dead as of 2026-08-07, and yesterday's entry was the paying one (the session BEFORE the print, +1.313% over 21 episodes). Today is one session late and that is normally fatal. What is genuinely new is the **runway/clear-calendar refinement discovered yesterday**, which post-dates the post-NFP vol sweep and was never applied to the far side of the print. Today's close carries a 3-session runway to PPI **plus a holiday**. That is the reopening argument and it is C1's job to destroy it. |

## 2. Tape extremes by class

| class | extremes |
|---|---|
| us_large | SPY -0.61% off its 52w high, r21 32.9, r63 24.2, ATR 0.83% of price. QQQ/^NDX r63 **9.5** and -3.71% off their highs. DIA r21 13.5. The index is at a high with its own 63-day rank in the bottom quartile. That exact shape ("the round-trip breakout") was tested and closed 2026-08-28: the low-63d-rank leg alone pays +0.530% at h=10 over 239 episodes at t 2.14 and the near-high clause inverts it. **DISMISS.** |
| us_small | IWM r5 19.8, r21 17.9, **r63 9.1**, -3.24% off its high, z10 -0.33. The persistent laggard. Carried into C6. |
| rates | ^TNX 4.76, **closed at or on its trailing-252 maximum for a fourth session**, r21 78.9, 252-session change +55.1bp. TLT +1.27% above its 252d low, r21 43.0; IEF +0.47%; LQD +0.27%. ^MOVE level percentile 62.7 with a 5-day return rank of 77.4. Four watchlist entries (13, 15, 18, 30) sit on this state and every one has a failing leg today. **DISMISS the price-state form**; the surprise-conditioned form is C3. |
| credit | HYG -0.35% off its 52w high, r5 20.2. LQD -4.12% off its high, +0.27% above its low. The IG/HY divergence is the tape's cleanest picture and it is the most-swept cell on this list (watchlist 1 and 26, killed 2026-08-10, 2026-08-26, 2026-08-27 and again yesterday). **DISMISS.** |
| gold_miners | GDX 21d **+21.28%** (r21 88.5) with GLD +5.28% and **-17.28% off its own 52w high**. The miner-metal divergence is closed in five separate forms and the deep-drawdown half is the wrong-signed one. **DISMISS as price state**; positioning form is C4. |
| metals | SLV 21d +7.99% but **-42.66% off its 52w high** and r63 31.7. Same complex, same closures. Folded into C4. |
| energy | The loudest thing in the tape. USO 21d **+23.69%**, DBC **at a fresh 52w high**, XLE -0.74% and XOP -0.43% off theirs, XLE r21 93.3, XOP r21 94.8, OIH r21 82.1. Every price-state form is closed (thrust bands, at-a-high, divergence-at-a-high, breadth counts; watchlist 4, 7, 19, 32 all PASS with failing legs, energy z10>=2 count is **0**). **DISMISS as price state**; the unopened form is seasonal → C10. |
| fx_dollar | DX-Y.NYB 98.99, r21 33.1, -2.58% off its high; UUP r21 33.7. Mid-range, no extreme, and midterm-blocked in every parked form. **DISMISS.** |
| intl | EWZ z10 **2.33** and r5 95.6, the tape's largest z-score; EWJ r5 77.4; EEM **r63 3.2**; FXI -13.79% off its high. EWZ was killed yesterday on the reference class and again on 2026-09-02 ("the live reading sits outside the historical support"). EEM's 63-day floor is the pooled laggard cell, watchlist 28, which has no holder of r21>=90 and r63<=10. **DISMISS.** |
| volatility | ^VIX 14.32 at the **2.4th percentile of its trailing year**, 21-day range percentile 3.2, relative-range 3.35 (the dead (0,5] half of yesterday's bimodal finding). ^VIX3M **at its 52-week low**. SVXY **at a fresh 52-week high**, z10 1.63. UVXY -51% from its 200d. ^SKEW 144.12 = the **49.2nd percentile of its trailing year** (the 99.6 in the tape is a 21-day RETURN rank; registry, yesterday). Every level form here is closed: the VIX3M floor (dose response backwards), SVXY at a high (placebo ladder), the skew forms (filter subtracts, midterm block), the bond-vol cross (mechanism inverted). What is NOT closed is the **cross-sectional** reading underneath the calm: Dispersion at the 92nd percentile with component RV 30.1% against SPY RV 7.2%. → C5. |
| sectors | XLF r63 **92.5**, XLV r63 88.5, XLP r63 77.8 against SMH r63 **1.2**, XLI r63 6.0, XLK r63 11.5. A wide defensive-and-financials over growth 63-day rotation with the index at its high. Every single-sector and sector-pair form of this is closed on the reference class (tech-minus-healthcare at a PIT floor, XLI triple floor, the twelve-name industrial count, the sector washout family, the pooled triple floor). Only **XLI** holds the triple floor today. **DISMISS.** |
| crypto | Not in the 218-name tape, not in `master_prices` for the vehicles this book trades. **Not examined**: no instrument. |

## 3. Seasonal and cycle cells

- **September, trading day 4, midterm.** "The index over the first week of September in a midterm year" was killed yesterday in the direction the trade needed: the month pays -0.042% at h=3 anchored at the first trading day, an excess of -0.080pp, and the midterm crossing is POSITIVE at +1.344%. The 48-cell month-by-cycle grid puts the pitched cell 43 of 48 with P(min-of-48) at 1.0000. **DISMISS.**
- **The Labor Day boundary.** The folk claim is that September weakness starts after Labor Day, not at the month turn. That is a different anchor from the one killed yesterday, because Labor Day floats between the 1st and the 7th, and it has never been measured here. → C2.
- **Post-Labor-Day driving season.** US gasoline demand steps down at the Labor Day boundary and refinery margins compress into the maintenance turnaround. This is the one energy anchor the registry has never opened, and energy is simultaneously the most extended thing on the tape. → C10.
- **Midterm as a conditioner, not an idea.** Applied to C2, C3 and C10 rather than pitched. The registry's own finding is that the midterm split is structural where it flips sign coherently across instruments, and it does exactly that on the post-NFP rates/utilities/dollar family.
- **November TLT** (watchlist 10) parks to trading days 4-12 of November. **December small-cap month-end overnight** (watchlist 31) parks to December.

## 4. Watchlist verdicts (36 active, every one answered)

Readings computed this morning in `00_watchlist_readings.py`.

| # | title (short) | verdict | today's number |
|---|---|---|---|
| 0 | TLT from the NFP close, long end at the floor | **PASS** | midterm again; today IS the print and the cell stays blocked. Parks to 2027-01. |
| 1 | LQD vs HYG at joint 52w extremes | **PASS** | episode count still 3 declustered against the 8 required. LQD +0.27% above its low, HYG -0.35% off its high. |
| 2 | SVXY overnight into the CPI print | **PASS** | arms at the 2026-09-10 close; CPI is 2026-09-11, four sessions out. |
| 3 | GLD on a miner-led thrust | **PASS** | GDX r5 31.1 against >=95; GLD -17.28% off its high against the within-10% leg. Both legs fail. |
| 4 | XLE on a crude thrust in the 5-6% band | **PASS** | USO 1-day +0.67%, nowhere near the band. |
| 5 | TLT with the IG complex pinned at 52w lows | **PASS** | TLT +1.27% above its low against the <=0.5% rung; IEF +0.47% and LQD +0.27% both clear. Same single failing leg for a fourth session. |
| 6 | SPY on a skew spike alone | **PASS** | ^SKEW r5 57.0 against >=95. Tape flags ^SKEW stale; the reading is from 2026-09-03 either way. |
| 7 | Fade a crude thrust out of a deep base | **PASS** | USO r63 55.8 against the <=20 deep-base leg. |
| 8 | IHI at a 21d rank of 100 | **PASS** | IHI r21 59.8. |
| 9 | FXI break while EEM holds | **PASS** | FXI r5 51.4 against <=20. |
| 10 | TLT on the November month position | **PASS** | parks to November. |
| 11 | Short SPY at a 52w high with TLT at a 52w low | **PASS** | SPY -0.61% off its high (clears <=0.5%? no, fails by 11bp); TLT +1.27% above its low against <=1%. Both legs fail, TLT more clearly. |
| 12 | SPY on a vol pop inside a calm tape | **PASS** | VIX fell -5.79% against a >=+5% pop; r21 29.9 against <=25. Wrong-signed on the defining leg. |
| 13 | Gold on an unconfirmed rate rise | **PASS** | DX r21 33.1 against <=15. Yield leg live (^TNX at its 252d max), dollar leg not. One-leg miss for a sixth session. |
| 14 | XLK vs XLV after a rotation gap | **PASS** | one-day XLV minus XLK gap **-1.11pp** against the >=+3.0pp rung, and wrong-signed today. |
| 15 | Short the dollar on an unconfirmed rate rise | **PASS** | ^TNX r21 78.9 clears >=65 for a fourth session; DX r21 33.1 fails <=20. |
| 16 | Short TLT after a thrust from the low zone | **PASS** | TLT 1-day +0.15% against the >=+1.5% thrust rung. |
| 17 | Short KRE against XLF on a breadth washout | **PASS** | ex-crisis cost threshold, no single session moves it. |
| 18 | IEF against 0.523 TLT at the yield extreme | **PASS** | 252-session yield change **+55.1bp** against the +78bp arm, down again as the trailing reference bar rolls. |
| 19 | Narrow energy thrust cluster, 2-3 names at z10>=2 | **PASS** | count is **0** (max VLO +1.45) against the [2,3] arm, a third straight session at zero with XLE, XOP, CVX and VLO all at or beside 52-week highs. |
| 20 | Survivorship-free new-high breadth | **PASS** | SPY -0.61% off its high against the >2.0% leg; raw-21d fragility 63.1 against <=50. Both fail. |
| 21 | The sector washout family at h=7 | **PASS** | XLI r5 10.4 clears the washout leg but XLI is -6.41% off its high against within-5%. |
| 22 | XLU washout with the long end hit alongside | **PASS** | XLU r21 39.4 against <=5; TLT r21 43.0 against <25. Rates leg wrong-signed a seventh straight session. |
| 23 | The bare dollar washout | **PASS** | parks to a non-midterm year; DX r21 33.1 is not a washout in any case. |
| 24 | HY at a fresh 52w high while the index is not | **PASS** | HYG -0.35% off its high against the <=0.05% touch; dial ma10 87.8 against <50. |
| 25 | SMH at a 63-day rank floor inside a top-decile year | **PASS** | SMH r63 **0.8**, floor live; r5 15.5 against the <15 still-falling arm, missing by half a point for a second session. Closest single-leg miss on the list. |
| 26 | Pure rates repricing with zero credit stress | **PASS** | HYG -0.35% off its high against the <=0.25% rung. |
| 27 | IEF out of the Jackson Hole close | **PASS** | midterm-blocked, anchor five sessions gone. |
| 28 | The pooled laggard that is still falling | **PASS** | no holder of r21>=90 and r63<=10 anywhere in the tape. |
| 29 | Short SLV after a complex break | **PASS** | complex bounced again (SLV +2.51%, GDX +3.95%, GLD +1.85%); depth arm needs -4.00% or worse. |
| 30 | TLT at a yield high with bond vol MID-RANGE | **PASS** | ^MOVE trailing-252 level percentile **62.7** against the [40,50) band. |
| 31 | December small-cap month-end overnight | **PASS** | parks to December. |
| 32 | XLE at a fresh high on a down-index session, h=21 | **PASS** | XLE -0.74% off its 252d max so the high leg lapsed, and SPY rose +1.05%. |
| 33 | SVXY into a print out of a VIX range in the (5,15] band | **PASS**, and it is the near-miss of the morning | rel-range **3.35**, in the dead (0,5] half for a second session. This entry is the direct parent of C1 and C1 is deliberately built on a DIFFERENT anchor rather than on this band. |
| 34 | The pooled sector triple floor with the index gate off | **PASS** | only **XLI** holds the triple floor today, down from nine names, so the pooled form has no breadth to trade. |
| 35 | SPY into a print out of a dead VIX range | **PASS** | dial ma10 87.8 against the arm's requirement, and the anchor was yesterday's session in any case. |

Nothing fired. Thirty-six entries, thirty-six passes, and the two closest (25 at half
a rank point, 33 in the wrong half of a bimodal band) are both blocked by the exact
number that was written into the arm, which is the point of writing arms as numbers.

## 5. Axis and scoreboard read

Five graded ideas lifetime is a handful, so the per-axis split is read and not
weighted: `event_fingerprint` 2 graded at +0.62R, `inversion` 1 at -0.62R,
`interaction_cell` 1 at +0.146R, `relative_value` 1 at +0.099R. Grade B is
3-for-3 at +0.448R and grade C is 2 at -0.237R. Too few to steer selection;
noted so the next map can compare.

## 6. Candidates selected

Twelve considered, seven sent to checkers, five dismissed above with reasons.

| id | candidate | axis | class | anchor |
|---|---|---|---|---|
| C1 | Long SVXY from the last close before a three-day market closure, clear calendar | event_fingerprint | volatility | event |
| C2 | Long the index into the Labor Day closure, tested for the arbitraged-fossil signature | event_fingerprint | us_large / us_small | event |
| C3 | Post-NFP duration conditioned on the PRIOR print's surprise, from `macro_release_history` | interaction_cell | rates | event |
| C4 | CFTC speculative net positioning in the metals complex as a live conditioner | flow_mechanics | metals | price-state / positioning |
| C5 | Cross-sectional dispersion at the 92nd percentile read as a SHORT-correlation signal rather than a fragility one | inversion | volatility | price-state |
| C6 | The small-cap laggard into September quad witching | interaction_cell | us_small | event |
| C10 | The post-Labor-Day driving-season boundary on crude and refiners | event_fingerprint | energy | event |

Asset classes touched by checked candidates: **volatility, us_large, us_small, rates,
metals, energy** (6). Axes: **event_fingerprint, interaction_cell, flow_mechanics,
inversion** (4), with instrument_translation and relative_value appearing only in
dismissals. Event-anchored: C1, C2, C3, C6, C10. Price-state-anchored: C4, C5.

Dismissed with reasons and no check spent: the round-trip breakout (closed
2026-08-28), the sector rotation pair (reference class, closed twice), the rates
level in every parked form (four watchlist entries, all one-leg misses), the
credit divergence (closed four times, episode counts), the dollar (midterm-blocked
in both vehicles).
