# Surface map, 2026-09-30 (Wednesday, Q3-end close, midterm year)

State: `data/pitch_state.json` built 05:1x ET. Freshest bar 2026-09-29 for 217 of 218
tape names (LEG stale, not a candidate). Pipeline 7/7 green. Dial ma10(63d) 81.6
(87.6 21 td ago), raw 21d 32.5, P/C fear off (22nd pctile), NYSE Net Highs signal ON
(5d EMA -348, SPY 1.51% off its 252 high). Exposure leg 0.0x. Book staged: one OVS
short (FORM). Event sleeve and trend sleeve flat.

## The tape in one paragraph

The sixth session of a rates shock that this pipeline has already swept on 09-21,
09-24, 09-25, 09-28 and 09-29 (all stand-downs) and 09-23 (one survivor, natgas).
^TNX 5.255% (5d rank 100, 21d 99.6, at its 252 high), ^MOVE 106.6 (+35.7% in 5d),
TLT/IEF/LQD all AT their 252 lows, HYG 0.65% off its low on 4.08x volume, DX 101.37
and UUP at 252 highs, GLD -4.3% / SLV -8.6% / GDX -9.0% in 5d (after the 09-28
complex break, a +1.3% bounce on 09-29), rate-sensitive equities at floors (XLU, XLRE,
IYR, ITB, HD, LOW), SPY -1.19% in 5d but only 1.51% off its high, QQQ 1.27% off with
semis strong (AMAT z10 2.14, SMH 1.92). USO -4.44% and UNG -4.08% on 09-29 (NG=F
+0.37% is a continuous-series roll seam: the October contract expired 09-28).
^VIX 16.0, VIX/VIX3M 0.887.

**What is new today and nowhere else in the last six sessions:** today's close IS the
quarter-end (QE-0 / ME-0) close, so every cell anchored ON the quarter-end close (as
opposed to run-ins INTO it, which 09-17/09-23/09-25/09-28 swept) can only be entered
today. NFP is now k=-2. MU and CAG print tonight, NKE tomorrow night.

## Scoreboard read

7 graded ideas lifetime (5 B, 2 C). B is 5-0 at +0.604R avg; C is 1-1 at -0.237R.
By axis: event_fingerprint 2-0 (+0.62R), interaction_cell 2-0, inversion 1-1,
relative_value 1-0. Seven is a handful; no axis gets penalised on it. Read noted:
grade C has bled, so a C needs a real mechanism to take the slot.

## 1. Calendar events x asset classes

Events in window: NFP 10-02 (k=-2 at today's close), CPI 10-14 (td 10), PPI 10-15
(td 11), opex 10-16 (td 12), VIX expiry 10-21 (td 15). Added anchors the calendar file
does not carry: **Q3-end / ME-0 close today**, Q4 turn 10-01, big-bank Q3 prints
10-13/10-14 (JPM, C, GS, WFC, then BAC, MS, STT), MU/CAG tonight, NKE 10-01, PEP 10-08,
China Golden Week 10-01..10-08, Brazil first round 10-04, FOMC 10-28 and the midterm
election 11-03 (both outside 15 td). Shutdown note: a lapse in appropriations on 10-01
would cancel the 10-02 print, as it did in October 2025; not verified either way this
morning, so any idea with NFP inside its hold carries it as a tail item.

### Quarter-end / ME-0 close (today) x class

| class | verdict |
|---|---|
| US large | **CHECK C1** (W55 SPDR winners-minus-losers reversal QE->QE+5, due today). Run-in forms dead (window dressing QE-9->QE, reg 4931; single-stock dressing reg 5114; SPY QE-5->QE+5 buyback blackout reg 5182). Turn-of-month equity leg dead post-2013 (reg 47, 1788). |
| US large, real estate | **CHECK C7** (XLRE residual after a rate-shock quarter, from the QE close). Utilities residual dead 10 ways (d12 09-28), XLRE duration forms dead (reg 2268, 4132); the QE-close anchor on REITs is unopened. |
| US small | Dismissed: IWM across the Q4 turn from a 63d floor killed 09-28 (a8, gate subtracts, IWM-SPY residual negative at every horizon). |
| rates | **CHECK C2** (short TLT from the ME-0 close, post-extension give-back). The registry's parked month-end TLT parent has an exit ladder falling from +0.540% at the ME close to -0.229% ten sessions later (reg 1126); nobody has tested the post-ME leg as its own trade. Crossed with today's price state (TNX at its 252 high). |
| credit | Dismissed: HYG's month-end bump lands on the ME close itself and the volume-spike month-turn cell subtracts (kC_c3 09-29); issuance-blackout LQD cell dead (reg 5142). No post-ME credit mechanism that is not the duration leg. |
| gold and miners | **CHECK C6** (long GLD from the QE close after a quarter-end-week metals flush). Gold-after-crash and gold-into-NFP both died 09-29, but neither was anchored on the quarter turn. |
| other metals | Dismissed: SLV/GLD pair forms dead 09-28 and 09-29 (c5, kA_c2); W28 first-break scoring is out-of-sample only. |
| energy | **CHECK C3** (short UNG across the October roll into winter contango). UNG long and short-through-CPI forms are dead (reg 96, 403) but the roll carry itself, in the month the curve is steepest, is unopened. |
| dollar and FX | **CHECK C5** (long DX from the QE close, dosed by the quarter's US-over-foreign equity return). The ME-0 London-fix session is dead (reg 2107) and the run-in is dead (reg 4940); the post-QE continuation (+0.228%, 62-44) is on file only as a parent and has never been checked as a trade. USDJPY book-close dead (09-29). |
| international | **CHECK C8** (short EEM vs beta-SPY from the QE close, the give-back of W62's QE-5->QE run-in). FXI Golden Week dead twice (09-18, 09-25), EWJ book close dead, EWZ vote dead (09-29, W71). |
| volatility | Dismissed: SVXY residual across the quarter turn dead 09-25 (1.2x cost, no curve softening). |

### NFP 10-02 (k=-2 at today's close) x class

| class | verdict |
|---|---|
| US large | Dismissed: pre-print SPY session out of a dead range is W34, dial-blocked (dial 81.6 vs < 50); post-NFP equity direction swept empty 08-07. |
| US small | Dismissed: no NFP small-cap cell with a mechanism; IWM rates-shock lag dead 09-24. |
| rates | Dismissed: TLT into payrolls with TNX at its 252 high killed 09-24 (every rung k=-10..-1 negative); TLT from the NFP close +3td is midterm-dead (W0, 2027-01); W36 fails (no CPI in the h=3 hold). C2 carries the NFP inside its window and must decompose it. |
| credit | Dismissed: no payroll-specific credit mechanism; the family has failed to show a credit residual six times (W43). |
| gold and miners | Dismissed: gold from a first complex break into the payrolls close killed 09-29 (kA_c7, -0.300% on 8-7). |
| other metals | Dismissed with gold, same kill; silver is gold beta here (kA_c2). |
| energy | Dismissed: crude into payrolls killed 09-29 (r63 gate) and the r21 form dies with a month-end inside the hold (W70; the 10-01 neighbour is for tomorrow's map). |
| dollar and FX | PASS on W68 (short DX k=-4 into NFP, TNX-high leg): out-of-sample episode 1 is live from the 09-28 close; score it 10-05. Long-DX k=-2 is the run-in the 09-28 kill found wrong-signed. |
| international | Dismissed: no payroll channel into EEM/EFA that is not the dollar leg. |
| volatility | Dismissed: SVXY into scheduled prints is W32, dial-blocked (needs ma10(63d) <= 68). |

### CPI 10-14, PPI 10-15, opex 10-16, VIX expiry 10-21 x all classes

Not examined for entry today, all ten classes, one reason: the anchors sit 10-15
sessions out, so an entry at today's close would be a 10+ td hold with the event at
its far edge, and every print/expiry cell in this repo lives in the k=-5..+3 band. These
rows are re-mapped from 10-07 onward. Specific parks that touch them: W41 (September
PPI-then-CPI, September only), W42 (TLT from the PPI release close, 10-15 anchor, owes a
correlated-ladder permutation), W2 (SVXY overnight into CPI), W57 (opex after a VIX
crush, non-midterm only), W44 (VIX expiry on an FOMC date, not this month).

### Earnings anchors

| anchor | verdict |
|---|---|
| Big-bank Q3 prints 10-13/10-14 | **CHECK C4** (long the money-centre banks into the kickoff after a 21d washout; entry today is k=-9). No bank-earnings-season cell exists in the registry. |
| MU tonight | Dismissed: pooled non-NVDA semi prints pay SMH +0.085% at h=3 over 465 prints (reg 2053); the turned-laggard MU form died 09-25. |
| CAG tonight, PEP 10-08 | Dismissed: 63d-winner-into-print gate subtracts (reg 4948); packaged-food complex is a family washout with no label effect (W48 reference class). |
| NKE 10-01 | Dismissed: short a 52w-low laggard into its print killed (reg 5108, NKE own record 1-2). |

## 2. Tape extremes by class

| class | extremes (5d/21d/63d rank, z10, 52w) | verdict |
|---|---|---|
| US large | QQQ/^NDX z10 1.53, 1.3% off high; META r21 98; AMAT z10 2.14, SMH 1.92; ADSK r21 1.2, PAYX r5 0.8 (-13.3%), MCD r5 0.8; insurers ALL/AON/BNY/GD/PAYX r21 0.4 | Semis thrust swept 09-23; insurance label carries nothing (reg 766); defense ITA retired (reg 3785); single-name washouts are family effects (W48). No candidate. |
| US large, rate-sensitive | XLRE r5 6.7 r21 5.2 r63 1.6; IYR, VNQ same; HD and LOW at 252 lows; XLU r21 5.2 | C7 (XLRE, QE-close anchor). Utilities retired. |
| US small | IWM r63 0.4, 8.3% off high | Dead three ways 09-24/09-28. |
| rates | ^TNX r5 100, TLT/IEF/LQD at 252 lows, TLT r21 0.4 | Directional duration at yield highs is dead in 8+ forms (W5, W16, W18, W29, W65, 09-24, 09-28 d10). Price state enters only as C2's gate. |
| credit | HYG r5 0.4 on 4.08x volume, 0.65% off its 252 low | Dead (kC_c3, W53 duration-driven form, W43). |
| gold and miners | GLD -22.8% off high, 8.0% under 200d; GDX r5 11.5 | C6 (quarter-turn anchor only). |
| other metals | SLV -47.5% off its high | Dead (09-28/09-29 pair forms). |
| energy | USO -4.44% day, r63 72; UNG -4.08% day, -38.8% off high; XOP/OXY/HAL z10 -2.0 to -2.3 | **CHECK C9** (price state: long USO after a >= 4% day inside a 63d uptrend). XLE-vs-USO dead (09-28 c9); energy effective N 1.4 (W31). C3 (UNG roll). |
| dollar and FX | UUP z10 2.02 at its 252 high, DX r21 89.7 | Dollar thrust long dead 09-24 (2018+ 2-9). C5 is calendar, not this state. |
| international | EWZ r5 7.5, FXI r63 93.3 r5 14.3, EEM r5 15.1 | C8 (calendar). FXI break-inside-thrust fails W9's r21 leg (18.3 vs 80). |
| volatility | ^MOVE r5 99.6; ^VIX r5 84.1 at 16.0; SVXY 0.6% off its high | Bond-vol extreme -> calmer equity vol confirmed four times and not tradeable (09-28 b3). W64 fails (SPY -0.18%, needs < -0.75%). |

## 3. Seasonal and cycle cells

The seasonal board in the state is stale (asof 2026-08-05) and carries no live tickets;
not used. Cycle: midterm year, October. Midterm Q4 long from a near-high index killed
09-29 (kC_c6: near-high midterm Octobers +0.05% on 4-4); midterm October long vol
killed 09-28 (d11). Midterm is a conditioner on C1-C9: each checker reports the
midterm split. Month-of-year: W10 (TLT November) and W58 (CL=F October tdom 13, 10-19)
are dated parks, not today.

## 4. Watchlist verdicts (72 active, 0 expired)

- W0 TLT NFP-close +3td, midterm-dead: PASS (2026 midterm; turns on 2027-01). Today's TLT at its 252 low is the state, the cycle blocks it.
- W1 LQD vs HYG joint extremes: PASS (HYG is 0.65% off its LOW, not near its high).
- W2 SVXY overnight into CPI: PASS (CPI 10-14; LOYO floor unchanged at 19.7 bps).
- W3 GLD on a miner-led thrust: PASS (GDX r5 11.5, GLD 22.8% off its high).
- W4 XLE on a 5-6% crude thrust: PASS (USO -4.44%).
- W5 TLT with the IG complex at 252 lows: PASS. State is live (TLT, IEF, LQD all at 0.00% from the low) but the arm is the kept-minus-deleted gap (+0.043pp vs >= 0.35pp), which one day does not move.
- W6 SPY on a skew spike: PASS (SPY 1.51% off high clears the depth leg; midterm fails).
- W7 crude thrust fade with a print in the hold: PASS (USO r5 42.9).
- W8 IHI thrust: PASS (IHI r21 19.4).
- W9 FXI break inside a thrust: PASS (FXI r5 14.3 clears, r21 18.3 fails >= 80).
- W10 TLT November: PASS (window opens ~11-05).
- W11 short SPY at a high with TLT at a low: PASS (TLT leg live, SPY 1.51% off vs within 0.5%).
- W12 SPY on a vol pop in calm tape: PASS (^VIX -0.19% on the day).
- W13 gold on an unconfirmed rate rise: PASS (DX r21 89.7 vs <= 15).
- W14 XLK vs XLV rotation: PASS (no >= 3pp one-day gap).
- W15 short dollar on an unconfirmed rate rise: PASS (DX confirms, r21 89.7).
- W16 short TLT after a big up day near the low: PASS (TLT -0.50%).
- W17 KRE vs XLF bank-breadth washout: PASS (statistical arm; C4 reads the same complex from the earnings side).
- W18 IEF vs 0.523 TLT curve, OOS only: PASS. ^TNX at its 252 max today; any new declustered episode is scored, not pitched.
- W19 narrow energy thrust count: PASS (0 energy names at z10 >= 2).
- W20 survivorship-free breadth: PASS (SPY -1.51% vs > 2.0% off; raw 21d 32.5 clears <= 50).
- W21 sector washout into a 52w high: PASS (no SPDR at r5 <= 5 within 5% of its high).
- W22 bare dollar washout: PASS (midterm).
- W23 HYG 52w high with SPY off its high: PASS (HYG at its low).
- W24 SMH family, OOS: PASS (SMH r63 5.2 but r5 39.7).
- W25 rates repricing with HYG at a high: PASS (HYG at its low).
- W26 IEF after Jackson Hole: PASS (2027).
- W27 still-falling laggard, 29 ETFs: PASS (no ETF at r21 >= 90 with r63 <= 10).
- W28 short silver on a first metals break: PASS (09-28 was OOS firing 1; its h=1 lag-1 exit is today's close, score 10-01; 09-29 not a break, metals up).
- W29 duration with MOVE mid-range: PASS (MOVE r5 99.6, far above the [40,50) band).
- W30 small-cap ME-0 overnight, December: PASS (September; mechanism closed).
- W31 energy at a 52w high on a down SPY day: PASS (XLE 6.1% off its high).
- W32 SVXY into a print from a dead range: PASS (dial 81.6 vs <= 68).
- W33 pooled sector triple floor: PASS (SPY 6.57% above its 200d; XLRE holds the floor but is not a nine-SPDR member).
- W34 SPY into a print from a dead range: PASS (dial 81.6 vs < 50).
- W35 closure risk premium: PASS (next closure 11-26).
- W36 post-NFP duration after a moderate miss: PASS (no CPI inside the h=3 hold).
- W37 SVXY after an extended closure: PASS (no closure).
- W38 SPY vs IWM in the 56-70 dial band: PASS (dial 81.6, falling; needs < 70).
- W39 HYG after an extended closure: PASS (no closure).
- W40 short IEF with commodities at a high and a print in the hold: PASS (DBC 5.0% off its high).
- W41 SPY across the September PPI-CPI pair: PASS (September only).
- W42 TLT from the PPI release close: PASS (anchor 10-15).
- W43 SPY with HYG and TNX at highs: PASS (HYG at its low).
- W44 SVXY on an expiry-FOMC collision: PASS (none in window).
- W45 NG=F September seasonal: PASS (September ends today; C3 opens the October roll instead).
- W46 FOMC-VIX collision run-in: PASS (2027-03-17).
- W47 r5<=2 / r63>=90 corner: PASS (no ETF in the corner).
- W48 XLV healthcare flush: PASS (XLV r5 48.8; expires 10-05).
- W49 hedged short SVXY after a VIX crush: PASS (no crush).
- W50 SVXY into an FOMC after a re-bid: PASS (next FOMC 10-28).
- W51 short a bank vs XLF after an intraday slide: PASS (largest bank move RF -1.36% = 0.64 ATR; expires 10-06).
- W52 TLT across the FOMC with TNX at a high: PASS (10-27 eve).
- W53 HYG after a spread-driven flush: PASS (HYG z10 -1.67 tape convention; IEF r5 0.4, duration-driven).
- W54 long dollar into the September QE: PASS (the window ends at today's close; the post-QE leg is C5).
- **W55 SPDR winners-minus-losers reversal from the QE close: CHECK (due today, anchor = today's close, h=5). Candidate C1.**
- W56 IWM quad witching: PASS (12-18).
- W57 SPY opex after a VIX crush: PASS (2027).
- W58 short CL=F from October tdom 13: PASS (10-19).
- W59 63d winner into its print: PASS (SPY above its 200d).
- W60 LQD vs IEF December QE-7: PASS (12-21).
- W61 utilities washout delay rule, OOS: PASS (XLU r21 5.2, TLT r21 0.4, the 09-18 cluster still running; no new first day).
- W62 EEM vs SPY December QE-5: PASS (December; C8 tests the post-QE give-back).
- W63 crude round-trip long: PASS (USO r5 42.9 vs <= 3; expires 10-14).
- W64 hedged SVXY after a MOVE spike: PASS (SPY -0.18% vs < -0.75%; MOVE +4.69%).
- W65 TLT after a high-volume down day into a low: PASS (TLT -0.50% vs <= -1.25%).
- W66 peso after a carry-unwind day: PASS (MXN=X is not in the price cache; no >= +1.50% USDMXN print can be confirmed, and the dollar index rose only +0.17%).
- W67 UNG re-fire, OOS: PASS (09-24 firing scored: entry 09-25 close 11.13, exit 09-29 close 10.35, -7.0%, a loss for the long; record 0-1).
- W68 short DX into NFP from k=-4, OOS: PASS (episode 1 running from the 09-28 close; DX +0.17% on 09-29 against it; score 10-05).
- W69 gold under its high with a dollar thrust, above-200d half: PASS (GLD 8.0% under its 200d).
- W70 crude into payrolls, month-end layout: PASS (month-end sits inside this NFP's hold; the 10-01 neighbour needs USO r21 >= 75 on today's close, 66.7 on 09-29).
- W71 EWZ vs EEM across the vote: PASS (no polling data in the repo; expires 10-20).

## 5. Candidates (9) selected from the map

| id | candidate | axis | class | anchor |
|---|---|---|---|---|
| C1 | W55: short the quarter's two best nine-SPDRs by 63d, long the two worst, from the QE close to QE+5 (live: short XLE, XLV; long XLU, XLI) | flow_mechanics | us_large | calendar (QE close) |
| C2 | Short TLT from the ME-0 close: the post-index-extension give-back, gated on ^TNX at its 252 high | flow_mechanics | rates | calendar x price state |
| C3 | Short UNG across the October roll, the month the winter contango is steepest | flow_mechanics | energy | calendar (futures roll) |
| C4 | Long the money-centre banks (XLF or JPM/C/WFC/GS) from k=-9 into the Q3 kickoff prints after a 21d washout (XLF r21 2.8) | event_fingerprint | us_large (financials) | calendar x price state |
| C5 | Long DX from the QE close, dosed by the quarter's US-over-foreign equity return (the FX hedge rebalance reversal); live: SPY 63d +2.59% vs EFA +0.64%, EEM -1.48% | interaction_cell | dollar_fx | calendar |
| C6 | Long GLD from the QE close after a quarter-end-week metals flush (GLD 5d -4.3%) | interaction_cell | gold_miners | calendar x price state |
| C7 | Long XLRE against beta-SPY from the QE close after a rate-shock quarter (XLRE r63 1.6, TNX r63 99.6) | relative_value | us_large (real estate) | calendar x price state |
| C8 | Short EEM against beta-SPY from the QE close to QE+5, the give-back of W62's QE-5->QE run-in | inversion | international | calendar |
| C9 | Long USO after a >= 4% one-day fall with USO's 63d rank >= 70 (the continuation parent inverting inside an uptrend) | inversion | energy | price state |

Coverage: 7 asset classes (us_large, rates, energy, dollar_fx, gold_miners,
international, plus real estate inside us_large), 5 axes (flow_mechanics,
event_fingerprint, interaction_cell, relative_value, inversion), calendar-anchored
(C1-C8) and price-state-anchored (C9, and the gates on C2/C4/C6/C7). Classes looked at
and dismissed with reasons above: US small, credit, other metals, volatility.

Registry collisions to answer, per candidate: C1 reg 4931/5114/5182; C2 reg 1126, 1765,
245, the 09-24 TLT-into-payrolls kill; C3 reg 96, 403, W45; C4 W17, W51, reg 255
(big-box earnings cluster); C5 reg 2107, 4940, 5492, 09-24 dollar-thrust kill; C6 09-29
kA_c1/kA_c7, 09-28 c5b; C7 reg 2179, 2268, 4132, 09-28 d12; C8 W62, 09-23 EEM QE short,
09-24 EEM-after-dollar-breakout; C9 09-28 c6, W63, 09-23 kC_b3.
