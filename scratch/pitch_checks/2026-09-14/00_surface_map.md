# Surface map, 2026-09-14 (Monday)

Inputs: `data/pitch_state.json` generated 05:11 ET; `data/pitch_tape.json` 218 names,
freshest bar 2026-09-11 (Friday, the prior session; only LEG stale at 08-27,
irrelevant). Pipeline green (4/4 receipts). Fragility dial ma10(63d) 86.3 as of
2026-09-11 (PIT append-only vintage), raw 21d 54.9, raw 5d 22.4; this is market
context for each idea's own risk, not a sizing instruction. Equity P/C fear OFF
(56.7th pctile, data 09-11). Signals on: VIX Range Compression only (21d range
13th pctile). Cycle: midterm year, September. Live probe numbers below come from
`00_live_arms.py` (output `00_live_arms_out.txt`). Entry for anything shipped
today is the 2026-09-14 MOO or MOC (lag=1 off the 09-11 close); holds 1-10 td
span FOMC + VIX settle (09-16, +2), opex/quad witching + S&P rebalance (09-18,
+4), and at h=10 end 09-28, two sessions before quarter-end.

Scoreboard read: 5 graded ideas lifetime (B 3 at +0.448R avg, C 2 at -0.237R;
event_fingerprint 2 at +0.62R, inversion 1 at -0.62R). A handful, not a signal;
noted and moved on. Eleven straight stand-downs (08-28 .. 09-11): several of
those kills were gate-attribution kills where the PARENT trade was left
unevaluated. Today's checker brief says so explicitly: a gate that does not
filter removes attribution, it does not by itself kill a live parent.

Portfolio/strategy overlap is NOT assessed (owner decision 2026-09-08).

## 1. Calendar events x asset classes

Events in window: NFP 09-04 (-5), PPI 09-10 (-2), CPI 09-11 (-1), FOMC 09-16 (+2),
VIX expiry 09-16 (+2), opex + quad witching 09-18 (+4), NFP 10-02 (+14). Not in
the event file but live: quarter-end 09-30 (+12), S&P quarterly rebalance at the
09-18 close.

| event | us_large | us_small | rates | credit | gold/miners | other metals | energy | dollar/fx | international | volatility |
|---|---|---|---|---|---|---|---|---|---|---|
| NFP 09-04 (-5) | dismiss: post-NFP equity direction swept empty (reg 08-07) and today is k+5, no anchor | dismiss, same | dismiss: TLT NFP cell midterm-dead (W0, parks 2027-01); prior-surprise form W37 fires only at next print | dismiss: no credit NFP cell survived (reg 08-07 PM cross-asset) | dismiss: k+5, GLD NFP cells swept 08-07 | dismiss | dismiss | dismiss: DX NFP cells swept 08-07 | dismiss | dismiss: post-NFP vol cells swept empty |
| PPI 09-10 (-2) | dismiss: Sep PPI/CPI pair cell killed 09-08 (W42; corrected perm 0.7354 per answer-quality review) | dismiss | dismiss: W43 TLT PPI-release anchor is the release close (09-10), not reachable today | dismiss | dismiss | dismiss | dismiss | dismiss | dismiss | dismiss |
| CPI 09-11 (-1) | dismiss: CPI-session SPY at TNX high killed 09-11 (gate loses to complement) | dismiss | dismiss: CPI x TNX-high killed 09-11; entry today is k+1 | dismiss | dismiss | dismiss | dismiss | dismiss | dismiss | dismiss: W33 second rung (SVXY MOC on 2nd print of pair) anchor was the 09-11 close, now past |
| FOMC 09-16 (+2) | dismiss: pre-FOMC window swept across 15 classes 09-01 (Cochran p 0.81, placebo 11 of 21); midterm run-in wrong-signed (W47) | dismiss: IWM run-in collision gate negative (reg 09-11) | dismiss: duration into FOMC at a TNX 252 max dead on calendar-leg attribution (reg 09-01); post-FOMC anchor not enterable today | dismiss: in the 15-class sweep | CHECK as C7: not the pre-FOMC drift but a yield-thrust WEEK with the decision inside the hold; gold is the class most exposed to a real-yield repricing and the 15-class sweep had no rate-state conditioner | dismiss: SLV in 15-class sweep; complex-break W29 lag-profile arm unmet | dismiss: crude into midterm FOMC dead (reg 09-01, placebo 8 of 12) | dismiss: DX in 15-class sweep; W15 rank form needs DX r21 <= 20, live 31.0 | dismiss: EFA/EEM/FXI in 15-class sweep | CHECK as C6: a >=10% one-day VIX crush inside the three sessions before an FOMC is a different object from the settle coincidence (dead) and from the compression band (W33, dial-blocked) |
| VIX expiry 09-16 (+2) | dismiss: collision is the FOMC's anchor (reg 09-01, 09-11) | dismiss | dismiss | dismiss | dismiss | dismiss | dismiss | dismiss | dismiss | dismiss: W45 settle-session SVXY needs 16 post-2018 collisions, today would be 13th and it is k=0 not reachable |
| opex/quad 09-18 (+4) | dismiss: Sep quad run-in is an FOMC anchor in costume (reg 09-04, 09-11) | dismiss: short IWM into Sep quad killed 09-11 (placebo 9 of 11); post-quad window starts 09-18 close, not enterable today | dismiss: no rates quad cell in registry and no mechanism for duration at an equity expiry | dismiss | dismiss | dismiss | dismiss | dismiss | dismiss | dismiss: V4 post-opex crush excludes September by prereg (September inverts); entry would be 09-18 |
| NFP 10-02 (+14) | dismiss: outside a 10 td hold from today | dismiss | dismiss | dismiss | dismiss | dismiss | dismiss | dismiss | dismiss | dismiss |
| quarter-end 09-30 (+12) | CHECK as C9 (paired): a stock-bond rebalancing flow is predicted by the 63d spread (SPY +3.86% vs TLT -4.85%), and a mid-month entry holding into the window has never been measured here (registry has 1 quarter-end hit, not this) | covered by C9's leg choice | CHECK as C9 | dismiss: credit is not a balanced-fund rebalancing leg | dismiss | dismiss | dismiss | dismiss | dismiss: no documented international rebalancing flow of this size | dismiss |
| midterm September (cycle) | CHECK as C11: mid-September-to-month-end in midterm years on SPY, the cycle-year conditioner applied to the calendar window itself | folded into C11 as a second vehicle | dismiss: TLT month-position cells park to November (W10) | dismiss | dismiss | dismiss | dismiss | dismiss: bare dollar washout W23 is midterm-inverted, not live | dismiss | dismiss |

## 2. Tape extremes by class (close 2026-09-11)

- **rates**: ^TNX 4.975 at its 252-day max for a 4th session, r5 98.4, r63 97.2, z10 +2.33; 252-session change +94.3 bp, 21-session +29.3 bp. TLT 0.11% above its 252 low, IEF and LQD exactly at theirs, IEF r5 1.6, z10 -2.04. -> W18 arm FIRED (below). W5 already killed on 09-11 (join re-anchors). CHECK C1.
- **credit**: HYG r5 5.6, z10 -1.92, -1.11% off its high, BUT the HYG/IEF ratio 5d rank is 88.9: high yield beat duration this week, so the HYG dip is a rate move, not a spread move. CHECK C4 on exactly that split. LQD at its low is duration (W1 note: LQD residual on IEF +0.000pp).
- **us_large / sectors**: SPY -1.75% off its high, r5 20.6, z10 -0.50, above 200d by 7.3%. Healthcare complex flush: XLV r5 0.8 (5d -4.56%, -3.41pp vs SPY, 2.0th pctile of the XLV/SPY 5d ratio over full history), IBB r5 1.2, IHI 2.0, XBI 3.2; AMGN -15.0%, SYK -10.5%, BMY -6.5%. CHECK C2. Sector breadth: only XLK (+2.06pp) and XLE (+1.95pp) beat SPY over 5d, 7 of 9 lag; 23.4% of the 218-name tape sits above a 21d rank of 50. CHECK C8. Defense: ITA r21 1.2, r63 6.0, LMT -13.1% 21d, GD r21 1.6. CHECK C10. Staples flush (KMB z10 -2.03, GIS -8.7% 5d, HRL r21 1.6): dismissed, the industrial-count object killed 09-11 as a relabelled sector oversold cell and a staples count is the same construction. Industrials MMM z10 -2.67, HON -2.17, XLI r21 3.6: dismissed, XLI count/washout killed 09-11 and W21 family arm unmet. Utilities PCG/EIX -20% 21d: dismissed, a single-state liability repricing, no class-level mechanism.
- **us_small**: IWM r21 6.7, r63 6.3, z10 -1.44, -5.31% off high while SPY is -1.75%. CHECK C3.
- **tech/semis**: INTC +12.3% 5d, AMD +13.2%, GLW +14.0%, HPQ at a 52w high +8.4% on the day, QCOM z10 +2.11; SMH r63 2.8 with r5 60.3. Dismissed: W25 needs SMH r5 < 15 (live 60.3); single-name momentum thrusts are the 52wh/OVS territory with no pitch-scale mechanism beyond that.
- **gold/miners**: GLD -2.79% 5d, r5 19.4, -19.6% off its high; GDX +24.9% 63d, r5 22.6. W3 needs GDX r5 >= 95 and GLD within 10% of high: both fail. CHECK C7 (the rates-state form).
- **other metals**: SLV -44.96% off its 52w high, r5 23.0. Dismissed: W29 is a lag-profile arm; the 09-11 SLV break-in-drawdown cell was killed on inverted gates.
- **energy**: USO z10 +2.45 (tape convention; pitch_lab.zscore +1.50), r21 87.3, +21.7% 21d, then -2.20% on 09-11, 2.2% off its high; VLO, XOP, CVX, COP at 52w highs. W19 count under the BINDING pitch_lab.zscore convention is 0 (USO 1.50, VLO 1.42, CVX 1.21) against the [2,3] arm, so PASS. CHECK C5 on the reversal day itself. UNG -39.8% off high: dismissed, W46 needs the NG=F front contract plus a mechanism that survives the month ladder.
- **dollar/fx**: DX-Y.NYB 99.12, r21 31.0, z10 -0.60, flat through a +29 bp yield month; UUP r5 55.6. Dismissed as a candidate: both rate-vs-dollar cells are parked on arms that fail today (W13 DX r21 <= 15; W15 DX r21 <= 20 and a cost floor) and re-deriving a looser dollar rung here would be the anti-rescue pattern.
- **international**: EWJ at a 52w high (+2.2% on the day), EFA r5 15.5 -1.94% off high, EEM r63 10.7, FXI -15.9% off high, EWZ r21 88.1. Dismissed: EWZ/EEM residual killed 09-11; W9 FXI arm unmet (r5 20.6, r21 43.7); no international extreme today carries a mechanism distinct from US beta.
- **volatility**: ^VIX 15.84 after -11.2% on 09-11 (5d +9.0%), ^VIX3M 18.60, VIX/VIX3M 0.852 contango; ^SKEW 154.5 at r21 98.4; ^MOVE 82.2 at the 91.7th level pctile. CHECK C6. SKEW: W6 requires a non-midterm year, dismissed.

## 3. Seasonal and cycle cells

- Seasonal board payload is stale (asof 2026-08-05, 0 A+B setups); not used as evidence.
- Midterm September: CHECK C11. Midterm is also the conditioner that blocks W0, W6, W23, W27, W47.
- Month-position TLT: November only (W10), not live.

## 4. Watchlist verdicts (49 active, 0 expired)

- W0 TLT NFP: PASS. Midterm year; parks to the first non-midterm NFP, 2027-01.
- W1 LQD/HYG credit divergence: PASS. HYG -1.11% off high against within 0.5%.
- W2 SVXY overnight into CPI: PASS. CPI printed 09-11; next CPI 10-14.
- W3 GLD on a miner-led thrust: PASS. GDX r5 22.6 against >= 95; GLD -19.6% off high against within 10%.
- W4 XLE on a crude one-day thrust 5-6%: PASS. USO 1d -2.20%.
- W5 TLT with IG complex at 52w lows: PASS. Construction arm (deleted-vs-kept gap >= 0.35pp); the state is still live (TLT +0.11% off low, IEF/LQD at lows) but the arm is not a state, and it was killed on 09-11.
- W6 SPY on a skew spike: PASS. Midterm year; ^SKEW r5 73.0 in any case.
- W7 fade crude thrust out of deep base: PASS. USO r63 71.8 against <= 20.
- W8 IHI 21d thrust: PASS, wrong-signed. IHI r21 4.8.
- W9 FXI break inside thrust: PASS. FXI r5 20.6, r21 43.7 against r21 >= 80.
- W10 TLT November: PASS. Parks to 2026-11-05.
- W11 short SPY at 52w high with TLT at low: PASS. SPY -1.75% off high against within 0.5% (TLT leg live at +0.11%).
- W12 SPY on a vol pop in calm tape: PASS. VIX -11.2% on the day.
- W13 gold on unconfirmed rate rise: PASS on the dollar leg. DX r21 31.0 against <= 15; yield leg live at +29.3 bp.
- W14 tech vs healthcare rotation gap: PASS. Arm needs XLV-minus-XLK one-day gap >= +3.0pp in calm tape; this week's gap runs the other way (XLV lagging), and C2 is the flush, not this rotation.
- W15 short dollar on unconfirmed rate rise: PASS. DX r21 31.0 against <= 20.
- W16 short TLT after a big up day near the low: PASS. TLT 1d +0.11% against >= +1.5%.
- W17 KRE vs XLF bank breadth: PASS. Cost arm; KRE r5 28.6, no washout.
- **W18 IEF vs 0.523 TLT curve at a TNX 252 max with a real thrust: CHECK, ARM FIRED.** 252-session change +94.3 bp against the >= +88 bp arm (clears the +78 bar by 16.3 bp); TNX at its 252 max. Debts carried: the 180-cell multiplicity charge (P 0.7097) the entry already failed, and today is the 4th consecutive session at the max (first touch 09-08), so whether today is an episode start under the cell's own declustering is the first question. -> C1.
- W19 narrow energy thrust count: PASS. Binding pitch_lab.zscore count 0 (USO 1.50, VLO 1.42, CVX 1.21); the tape convention would read 3, which is the convention debt the entry names.
- W20 survivorship-free new-high breadth: PASS. SPY -1.75% against more than 2.0% below high.
- W21 sector washout within 5% of high, family: PASS. XLV r5 0.8 is -5.87% off its high against within 5%, and the arm is heterogeneity in any case.
- W22 XLU washout with TLT hit: PASS. XLU r21 16.7 against <= 5; TLT r21 36.1 against < 25.
- W23 bare dollar washout: PASS. Midterm; DX r21 31.0.
- W24 HYG fresh high while index not: PASS. HYG -1.11% off high.
- W25 SMH 63d floor in a top-decile year: PASS. SMH r63 2.8 live, r5 60.3 against < 15.
- W26 IG lows with HY at a high: PASS. HYG -1.11% against within 0.25%.
- W27 IEF post-Jackson Hole: PASS. Midterm; parks to 2027-08-27.
- W28 laggard still falling, pooled: PASS. No name holds r21 >= 90 AND r63 <= 10.
- W29 short SLV after complex break: PASS. Lag-profile arm; no complex break on 09-11 (SLV +1.08%).
- W30 long duration at yield high with MOVE mid-range: PASS. MOVE level pctile 91.7 against [40,50).
- W31 IWM December month-end overnight: PASS. Parks to December.
- W32 XLE fresh high on a down-SPY session, h=21: PASS. SPY +0.85% on 09-11; the arm is a family permutation P in any case.
- W33 SVXY into a print out of a compression band: PASS. Dial 86.3 against <= 68.0; no print in the next session with a k=-2 anchor besides FOMC, and the alpha leg is unmet.
- W34 pooled sector triple floor: PASS on today's tape, and the entry's own arm text ("a reason to exist beside the book") is a portfolio-overlap criterion that the 2026-09-08 owner decision retires; recorded for the rewrite. No nine-SPDR name holds the 5/21/63 floor (nearest XLI r5 20.6 / r21 3.6 / r63 5.2).
- W35 SPY into a print out of a dead VIX range: PASS. Dial 86.3 against < 50.
- W36 closure risk premium: PASS. No market closure ahead.
- W37 post-NFP duration after a moderate prior miss: PASS. Next NFP 10-02.
- W38 SVXY first close after closure: PASS. No closure.
- W39 SPY vs IWM at dial 56-70: PASS. Dial 86.3. (C3 is the opposite leg on a price state and will read the dial split as a caveat only.)
- W40 HYG after closure: PASS. No closure.
- W41 short IEF with commodities at a 252 high and an inflation print in hold: PASS. DBC -1.37% off high and no CPI/PPI inside a 5 td hold (next CPI 10-14).
- W42 SPY across the Sep PPI-CPI pair: PASS. Pair printed; killed 09-08.
- W43 TLT from the PPI release close: PASS. Anchor 09-10 passed; arm is a correlated-family permutation.
- W44 SPY with HYG at high on a TNX-high session: PASS. HYG -1.11% off high.
- W45 SVXY settle session on an FOMC/VIX-expiry collision: PASS. 09-16 is a collision but the arm is 16 post-2018 collisions (12 today) and the settle session is not today.
- W46 NG=F September: PASS. Mechanism arm unmet (September ranks 5 of 12).
- W47 SPY run-in on an FOMC/VIX-expiry collision, non-midterm: PASS. Midterm year; parks to the next non-midterm collision.
- W48 deep 5d flush inside a top-decile 63d trend: PASS. IBB r5 1.2 with r63 81.3, IHI r5 2.0 with r63 76.2, both against r63 >= 90.

## 5. Candidates selected (11), with axis and source cell

| id | class | anchor mode | axis | candidate |
|---|---|---|---|---|
| C1 | rates | price-state (watchlist W18 armed) | relative_value | Long IEF / short 0.523 TLT (duration-neutral), TNX at a 252 max with a >= +88 bp 252-session change, h=8 |
| C2 | us_large (sectors) | price-state | interaction_cell | Healthcare complex flush: XLV at r5 <= 1 with >= 3 of XLV/IBB/XBI/IHI at r5 <= 5, long XLV outright and vs SPY, h=3..10 |
| C3 | us_small | price-state | relative_value | Small caps at a 21d AND 63d rank floor while SPY is within 2% of its high: long IWM / short SPY, h=5..10 |
| C4 | credit | price-state | interaction_cell | HYG 5d flush driven by duration rather than spread (HYG r5 <= 10 with HYG/IEF 5d rank >= 80): long HYG h=5 |
| C5 | energy | price-state | inversion | Crude reversal day: USO down >= 2% within 3% of its 252 high after a 21d top-decile thrust; forward on XLE and USO, h=1..5 |
| C6 | volatility | event (FOMC +2) | event_fingerprint | A >= 10% one-day VIX crush inside the 3 sessions before an FOMC: the event premium re-bid (short SVXY or long ^VIX proxy) into the decision, h=1..3 |
| C7 | gold/miners | event x price-state | interaction_cell | Gold after a yield-thrust week (TNX r5 >= 95 at a 252 max) with an FOMC inside the hold, h=3..5 |
| C8 | us_large | price-state | historical_analogue | Narrow leadership week: <= 2 of 9 SPDRs beat SPY over 5d while SPY is within 2% of its high; SPY forward h=5..10 |
| C9 | rates x us_large | calendar (quarter-end +12) | flow_mechanics | Quarter-end balanced-fund rebalancing: long TLT / short SPY from mid-September into quarter-end when the 63d SPY-minus-TLT spread is top-decile |
| C10 | us_large (industry) | price-state | interaction_cell | Defense complex washout: ITA at r21 <= 2 and r63 <= 10, long ITA h=5..10 |
| C11 | us_large / us_small | calendar (cycle) | inversion | Midterm-year second half of September: short SPY / IWM from trading-day-of-month 10 to month end |

Coverage: 7 asset classes (rates, credit, us_large, us_small, energy, volatility,
gold); event-anchored C6, C9, C11; price-state C1-C5, C8, C10; axes
relative_value, interaction_cell, inversion, event_fingerprint, flow_mechanics,
historical_analogue (6). International, dollar/fx and other metals were opened
and dismissed above with the numbers they turned on.
