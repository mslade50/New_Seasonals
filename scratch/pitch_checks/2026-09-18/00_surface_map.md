# Surface map, 2026-09-18 (Friday)

Inputs: `data/pitch_state.json` generated 2026-09-18 05:11 ET; `data/pitch_tape.json`
218 names, freshest bar 2026-09-17 (the prior session, fresh). One stale name (LEG), not
used. Pipeline receipts 4/4 green (prices bar, dial and P/C all dated 2026-09-17).
Fragility dial ma10(63d) 81.9 as of 2026-09-17 (raw 5d 20.3, 21d 45.1, 63d 74.7): context
for each idea's own risk only, not a sizing input. P/C fear OFF (equity P/C 10d-MA at the
48.0th pctile of its trailing year, data 2026-09-17). Risk signals on: Low Absorption
Ratio (AR 0.343, 9th pctile) and NYSE Net Highs (net highs -41 with SPY 1.96% below its
252-session closing high). Midterm year, September, day 18 (tdom 13). Live readings:
`01_watch_state.py`, `02_watch_extra.py` (pitch_lab conventions). Tape sort: `_tape_sort.py`.

Entry reality: signal close = 2026-09-17 (FOMC k=+1). MOC today lands on the 09-18 close,
which is September monthly opex, quad witching AND the S&P quarterly rebalance close. The
quarter's last session 09-30 is 8 sessions after today's close; NFP 10-02 is +10. Yom
Kippur is Monday 09-21 (+1). Mainland China's National Day (Golden Week) closure starts
10-01 (+9), with Stock Connect southbound trading suspended across it.

## The tape in one paragraph

A post-FOMC relief session. SPY +1.13%, QQQ +1.73%, SMH +2.76% (INTC +7.67%, AMD +6.36%,
MU +5.50%), IWM only +0.53%. ^VIX -12.82% to 15.44, VIX/VIX3M 0.832, ^MOVE -5.59%, SVXY
+1.91% to 0.31% off its 52w high, UVXY at its 52w low. Bonds rallied: TLT +1.11% (still
1.33% above its 252 low), IEF +0.57% (0.57% above), LQD 0.84% above; ^TNX 4.947 (-5.9 bp
on the day) against a 252 max of 5.006 and +92.1 bp over 252 sessions. The whole metals
complex rose together on a flat dollar: GLD +1.69%, SLV +3.37%, GDX +3.36% (DX-Y.NYB
-0.09%), with GLD 19.7% and SLV 44.2% below their highs. Crude slipped (USO -0.55%) but
holds a +18.9% 21d thrust, 40% above its 200d; VLO sits at a 252 high. The dollar holds a
5d rank of 92 (z10 1.36). Underneath: banks still at a breadth floor (10 of 11 at r5 <=
20, BNY 0.8, BAC 1.6, GS 3.2; median r63 34.1), utilities at 52w lows (XLU 0.96% above,
r21 3.6; CMS, PEG), defensive consumer at lows (MCD 0.00%, PEP 0.10%, LOW 0.05%, HD 3.2%,
XHB 2.48%), industrial gases flushed (APD z10 -2.38, LIN -2.23), XLI r63 0.8, ITA r63 0.4.
Small caps lag: IWM r21 7.9, r63 4.8, z10 -1.10. 11 tape names within 1% of a 52w low
against 8 within 1% of a high.

## 1. Calendar events x asset classes

Events in the state window: CPI 09-11 (-5), FOMC decision 09-16 (-2), VIX expiry 09-16
(-2), opex + quad witching 09-18 (0, today's entry close), NFP 10-02 (+10). Non-macro
anchors inside a 10 td hold, enumerated because the macro file does not carry them: S&P
quarterly rebalance 09-18 (0), Yom Kippur 09-21 (+1), quarter-end 09-30 (+8), China
Golden Week 10-01 (+9), Q4 turn-of-month 10-01 (+9).

| event | us_large | us_small | rates | credit | gold | metals | energy | dollar_fx | intl | vol |
|---|---|---|---|---|---|---|---|---|---|---|
| CPI 09-11 (past) | D1 | D1 | D1 | D1 | D1 | D1 | D1 | D1 | D1 | D1 |
| FOMC 09-16 (k=+2 at entry) | D2 | D2 | D3 | D3 | D3 | D3 | D4 | D5 | D6 | **c3 conditioner** |
| VIX expiry 09-16 | D7 | D7 | D8 | D8 | D8 | D8 | D8 | D8 | D8 | D7 |
| opex / quad 09-18 (today) | **CHECK c2** | **CHECK c1** | D9 | D9 | D9 | D9 | D9 | D9 | c1 rows | **CHECK c4** |
| S&P rebalance 09-18 | D10 | D10 | - | - | - | - | - | - | - | - |
| Yom Kippur 09-21 | D11 | D11 | D11 | D11 | D11 | D11 | D11 | D11 | D11 | D11 |
| quarter-end 09-30 (+8) | D12 | D12 | D12 | D12 | D12 | D12 | D12 | D12 | D12 | D12 |
| China Golden Week 10-01 (+9) | D13 | D13 | D13 | D13 | D13 | D13 | D13 | D13 | **CHECK c10** | D13 |
| NFP 10-02 (+10) | D14 | D14 | D15 | D14 | D14 | D14 | D14 | D14 | D14 | D14 |
| refinery turnaround season (mid-Sep to Oct) | - | - | - | - | - | - | **CHECK c9** | - | - | - |

Dismissals, each with its reason:
- **D1 CPI (past).** Post-release anchors needed the 09-11 close; the pair cells
  (watchlist 43, 44) are parked to the October prints.
- **D2 FOMC k=+2 x us_large/small.** 09-16 measured SPY from the decision close to the end
  of week zero at placebo 11 of 11; today is one rung later inside that dead window.
- **D3 FOMC k=+2 x rates/credit/gold/metals.** Registry 09-16: with the ten-year near its
  high the post-decision relief is wrong-signed (TLT h=3 -1.53% at the 252 max); gold
  closed on three rungs; the HYG decision-close flush filters the wrong way.
- **D4 FOMC x energy.** No policy channel into crude distinct from the dollar; crude's
  reversal-day cells closed 09-14 and 09-17.
- **D5 FOMC x dollar.** The 09-16 ship (long DX from the decision close) sits in the
  repetition window with nothing materially changed; quarter-end dollar closed 09-17.
- **D6 FOMC x international.** Reduces to dollar plus US beta (09-16 D5).
- **D7 VIX expiry x equity/vol.** The settle rung is era-dead (watchlist 46) and the run-in
  is midterm-wrong-signed (watchlist 48); both anchors sit behind the entry. The post-VIX
  -expiry vol cell is "swept and empty" in the registry. The crush that followed is c3.
- **D8 VIX expiry x non-equity classes.** No channel from a VIX settlement.
- **D9 quad x rates/credit/gold/metals/energy/FX.** An equity-derivative expiry with no
  delivery or roll channel into these ETFs; the 100-cell cross-asset post-opex grid is
  closed ("peaks at the horizon edge = exposure, not an impulse").
- **D10 S&P quarterly rebalance.** The mechanism lives in index adds and deletes; the
  repo holds no index-change history, so it cannot be measured (honesty rule).
- **D11 Yom Kippur.** Closed 09-17 on all four classes it could touch (live rung 9-17,
  sign p 0.962).
- **D12 quarter-end.** Closed 09-17 on window dressing, EWJ and the dollar; month-end
  closed on six forms; the post-QE reversal pair (watchlist 57) is date-parked to 09-30.
- **D13 Golden Week x non-China classes.** The Stock Connect suspension removes mainland
  buyers from Hong Kong only; no channel into US assets beyond FXI/EEM beta.
- **D14 NFP (+10) x everything but rates.** The only reachable anchor is the pre-print
  close at h=10, and post/pre-NFP equity direction is registry-swept; the pre-print
  dead-VIX-range cell (watchlist 36) is blocked on the dial (81.9 vs < 50).
- **D15 NFP x rates.** Midterm-dead (watchlist 1); into-print duration with the ten-year
  near its high is wrong-signed (09-03 kill).

## 2. Tape extremes by class

| class | extremes (2026-09-17) | verdict |
|---|---|---|
| us_large | SPY +1.13% to 1.96% off its high; QQQ +1.73%; SMH +2.76% out of a 63d floor (r63 2.8); banks 10 of 11 at r5 <= 20; XLU at 52w low (r21 3.6); MCD/PEP/LOW at lows; APD/LIN z10 -2.38/-2.23; META z10 2.78 (r21 99.6); CRM +24.0% 21d; TMO at a high; ITB/XHB 3.7%/2.5% above lows with TLT +1.11% | **CHECK c7** (homebuilders' next-session catch-up to a bond rally, a lead-lag construction the registry has never measured; the XLRE rates-gate kill owes it an exposure-ordering test). DISMISS: bank breadth (industry label carries no information, Cochran Q twice; broken-trend half), SMH floor family (killed 09-15), XLU/XLI floors (dead in the live regime 09-16), staples/food flush (09-08 both directions), industrial-gas pair (a two-name generic reversal; the 09-10 generic-reversal kill says labels lose to the generic), META/CRM thrusts (single-name momentum with no event or flow story; the 09-15 failed-thrust kill), COST pre-print (registry: "pre-print drift in a deeply lagging mega-cap", the lagging gate is monotone against the pitch and COST's r21 2.4 sits on the worst rung). |
| us_small | IWM r21 7.9, r63 4.8, z10 -1.10 into quad witching | **CHECK c1** (the post-quad window with small caps arriving washed out, the inversion of the event sleeve's T3 skip rule). The INTO-quad cells are closed (09-04, 09-07, 09-11). |
| rates | TLT +1.11% day, 1.33% above its 252 low; IEF 0.57%, LQD 0.84% above lows; ^TNX 1.18% off its 252 max | **CHECK c8** (watchlist 17's [1.0, 1.5) thrust band, recorded as significantly wrong-signed for the SHORT, read as a long; today sits in it). Outright duration at the low is dead four ways (IG floor 09-11, month-end gradient, post-FOMC 09-16, TNX high 09-09). |
| credit | HYG 0.96% off its high, z10 -0.93; LQD 0.84% above its low | DISMISS: watchlist 2/25/27/45 need HYG within 0.25-0.5% of its high; flush arm (55) needs z10 <= -2. |
| gold / miners | GLD +1.69% (19.7% off high, 4.3% below 200d), GDX +3.36% | Carried by c6 as rows. Miner/metal ratio both directions closed (08-27, 09-07, 09-17). |
| other metals | SLV +3.37% on a flat dollar, 44.2% off its high, 10.3% below 200d; GLD and GDX up together | **CHECK c6** (the complex-wide UP day, mirror of watchlist 30's complex-break short; the registry's continuation side "long SLV pays +0.755% on 32-21" makes the continuation direction pre-specified). |
| energy | USO +18.9% 21d, 40.1% above 200d, z10 1.21; VLO at a 252 high; DBC 1.87% off its high | **CHECK c9** (crude into the fall refinery-turnaround window after a 21d thrust, a demand-side mechanism no energy kill has named). Outright XLE/USO thrust, reversal and refiner cells are closed (ten entries). |
| dollar_fx | DX-Y.NYB r5 92.1, z10 1.36, r21 80.6; UUP 0.77% off its high | DISMISS: repetition window (09-16 ship); quarter-end dollar closed 09-17; watchlist 14/16/24 need DX r21 <= 15/20. |
| international | EEM +1.81% (r63 4.0); EWJ 0.66% off its high; FXI z10 -1.15, 8.2% above its low, -15.2% 252d; EWZ r21 85.7 | **CHECK c10** (FXI into the Golden Week Stock Connect suspension). EEM floor killed 09-16; EWJ quarter-end killed 09-17; EWZ residual killed 09-11. |
| volatility | ^VIX -12.82% to 15.44 (r5 13.1), VIX/VIX3M 0.832, SVXY 0.31% off its high, UVXY at its low; SKEW 145.7 (r5 36.9); ^MOVE -5.59% | **CHECK c3** (watchlist 51 live: a >= 12% crush, the dose bucket its arm names) and **CHECK c4** (the September post-opex month that V4 carves out). |

## 3. Seasonal and cycle cells

- **Midterm year** conditions everything; every checker reports a midterm split.
- **September second half.** Midterm second-half short killed 09-14 (2002 and 2022 carry
  120% of the total). The post-opex window is the event sleeve's T3 (short IWM, Sep opex
  MOC to the last September session MOC), whose prereg SKIPS the trade when IWM z10
  (lag-1) < -1, "washouts bounce"; today IWM z10 is -1.10 -> **c1** tests the inversion.
- **September post-opex volatility.** Registry book finding: V4 as specified pays
  September **-1.535% over 8 post-break anchors at a 0% hit** (long SVXY, opex MOC to +3),
  against +0.674% for the rest of V4. The registry routes that stress to T3 rather than to
  a short-vol trade; nobody has measured the short-vol side as a standalone trade with the
  mandatory SPY residual -> **c4**.
- **Fall refinery turnaround** (September-October maintenance cuts crude runs) -> **c9**.
- **China Golden Week** (10-01 to 10-07/08): Stock Connect southbound closes, removing
  mainland buyers from Hong Kong; the flow only exists since the November 2014 launch,
  which gives a pre-specified era prediction -> **c10**.
- **Seasonal board** (asof 2026-08-05, stale): zero A/B setups; its P/C complacency row is
  no longer live (equity P/C 48.0 pctile).
- **NG=F September** (watchlist 47): mechanism arm unmet, dismissed.

## 4. Watchlist verdicts (57 active, 0 expired)

1. TLT from the NFP close: PASS. Midterm-dead; parks to 2027-01.
2. LQD vs HYG at joint 52w extremes: PASS. HYG 0.96% off its high vs within 0.5% (LQD 0.84% clears).
3. SVXY overnight into CPI: PASS. Next CPI 10-14 (+18).
4. GLD on a miner-led thrust: PASS. GDX r5 44.0 vs >= 95.
5. XLE on a crude thrust in the 5-6% band: PASS. USO -0.55%.
6. TLT with the IG complex at 52w lows: PASS. State near-live (IEF 0.57%, LQD 0.84%, TLT 1.33% off lows) but the arm is a filter-vs-reanchor construction test one session cannot move.
7. SPY on a skew spike: PASS. SKEW r5 36.9 vs >= 95.
8. Fade a crude thrust out of a deep base: PASS. USO r63 74.6 vs <= 20.
9. IHI thrust: PASS. IHI r21 14.7 vs 100.
10. FXI break inside a thrust: PASS. FXI r21 40.5 vs >= 80.
11. TLT November month-position: PASS. Date-parked.
12. Short SPY at a 52w high with TLT at a low: PASS. SPY 1.96% off its high vs within 0.5%.
13. SPY on a vol pop in calm tape: PASS. VIX -12.82% (a crush, not a pop).
14. Gold on an unconfirmed rate rise: PASS. DX r21 80.6 vs <= 15.
15. Tech vs healthcare after a rotation gap: PASS. XLV minus XLK -1.63pp (XLK led) vs >= +3.0pp.
16. Short the dollar on an unconfirmed rate rise: PASS. DX r21 80.6 vs <= 20.
17. Short TLT after a big up day near the low: PASS on the short. TLT +1.11% vs >= +1.5%, and the arm needs the [1.0, 1.5) band to stop being significantly wrong-signed; today sits IN that band, whose recorded -0.241% for the short at h=2 (8-18) is the seed of **c8**.
18. Short KRE vs XLF on a bank-breadth washout: PASS. Breadth live (10 of 11 = 90.9% at r5 <= 20) but the median 63d rank is 34.1 vs intact, and the arm is an ex-crisis cost threshold.
19. IEF vs 0.523 TLT curve at a yield high: PASS. Out of sample only since 09-14.
20. Narrow energy thrust count: PASS. Count 0 (max VLO 1.39 under pitch_lab.zscore).
21. Survivorship-free new-high breadth: PASS. SPY 1.96% off its high vs > 2.0% (misses by 4 bp); raw-21d fragility 45.1 clears; zero SPDRs at a high.
22. Sector washout into a 52w high: PASS. No SPDR at r5 <= 5 (XLF 15.5 lowest).
23. Utilities washout with TLT hit: PASS. XLU r21 3.57 clears; TLT r21 68.65 vs < 25.
24. Bare dollar washout: PASS. Date-parked; DX r21 80.6.
25. HYG 52w high while the index is not: PASS. HYG 0.96% off.
26. SMH deep correction family: PASS. Out of sample only.
27. Rates repricing with zero credit stress: PASS. HYG 0.96% vs within 0.25%.
28. IEF out of Jackson Hole: PASS. Date-parked.
29. The laggard still falling, pooled: PASS. No ETF with r21 >= 90 and r63 <= 10.
30. Short silver after a complex break: PASS. The complex rose (SLV +3.37%); its mirror is **c6**.
31. Long duration at a yield high with MOVE mid-range: PASS. MOVE level pctile 71.8 vs [40,50).
32. IWM December month-end overnight: PASS. Date-parked.
33. Energy at a fresh 52w high on an index-down session: PASS. SPY +1.13%, XLE 2.20% off its high.
34. SVXY into a print from a (5,15] VIX range: PASS. No print at k=-2 (NFP +10); dial 81.9 vs <= 68.
35. Pooled sector triple floor: PASS. No SPDR at a 5/21/63 <= 10 floor today (XLU r5 16.7); arm is SPY below its 200d (+6.76% above).
36. SPY into a print out of a dead VIX range: PASS. Dial 81.9 vs < 50.
37. Risk premium across an extended closure: PASS. Next is Thanksgiving.
38. Post-NFP duration after a moderate miss: PASS. Next NFP 10-02.
39. SVXY after a closure: PASS. No closure.
40. SPY vs IWM with the dial in 56-70: PASS. Dial 81.9.
41. HYG after a closure: PASS. No closure.
42. Short IEF with commodities at a high and a print inside: PASS. DBC 1.87% off its high; no inflation print inside h=5.
43. SPY across the September PPI-CPI pair: PASS. Pair was 09-10/11.
44. TLT from the PPI release close: PASS. Next PPI 10-15.
45. SPY with HYG at a high as TNX prints one: PASS. HYG 0.96% off; TNX 1.18% off its max.
46. SVXY on a VIX expiry x FOMC settle session: PASS. The 09-16 collision is recorded, not traded.
47. NG=F September: PASS. Mechanism arm unmet.
48. SPY run-in on an FOMC x VIX expiry collision, non-midterm: PASS. Midterm; parked.
49. Deep 5-day flush inside a top-decile 63d trend: PASS. No ETF at r5 <= 2 with r63 >= 90.
50. XLV vs 0.71 SPY after a healthcare flush: PASS. XLV r5 77.8 vs <= 1.
51. Hedged short SVXY after a 10% VIX crush: **CHECK -> c3.** State live: ^VIX -12.82% on 09-17, inside the >= 12% bucket the arm names. The arm itself (hedged h=1 >= +0.60% AND the >= 12% bucket at least the >= 10% cell, recorded +0.097% at h=1 against +0.335%) is statistical, so the check re-runs it on today's tape and adds the one split never run: a crush on the session AFTER an FOMC decision that also leads into a monthly opex.
52. SVXY into FOMC after a re-bid, backwardated: PASS. Next anchors 10-26 and 12-07.
53. Short a large bank vs XLF after a 1.5 ATR intraday slide: PASS. Largest move GS +0.47 ATR (up); nothing slid.
54. Long TLT across the FOMC announcement session: PASS. Parked to the 10-27 eve close.
55. Long HYG after a spread-driven flush: PASS. HYG z10 -0.93 vs <= -2.
56. Long the dollar into the September quarter-end close: PASS. Killed 09-17 on the premium itself; nothing new.
57. Short the quarter's two best SPDRs vs its two worst from the QE close: PASS. Date-parked to the 09-30 close.

## 5. Scoreboard read

Five graded ideas lifetime (3 B at +0.448 avgR, 2 C at -0.237; event_fingerprint 2 at
+0.622, inversion 1 at -0.62, interaction_cell 1 at +0.146, relative_value 1 at +0.099).
Still a handful; no axis earns or loses a slot on it. Two of today's candidates are
inversions (c1, c4); the one graded inversion lost, which is noted and not weighted.

## 6. Candidates selected from this map (9)

| id | candidate | anchor | class | axis | registry / watchlist adjacency |
|---|---|---|---|---|---|
| c1 | Long IWM from the quad-witching close when small caps arrive washed out (IWM z10 <= -1 at lag-1), h=1..10, all four quarterly quads with September as a row; rows for SPY, QQQ, EEM, EFA | event (quad witching) | us_small | inversion | Inversion of the event sleeve T3 skip rule (prereg 2026-08-06). INTO-quad cells closed 09-04/07/11 ("the object is the post-opex window"); the post-opex IWM overnight/intraday decomposition and "opex gate is an INVERTER near a 52w high" (registry) are adjacent. Must separate from generic IWM z10 reversal on any day (gate attribution). |
| c2 | Short SPY from a monthly opex close when ^VIX fell >= 10% over the three sessions into expiry (the vanna/charm tailwind ending at expiry), h=1..5; September row | event x vol state | us_large | flow_mechanics | Registry: "post-opex closed both ways", the opex overnight decomposition, "dealer-hedging mechanism unfalsifiable in this repo" (option files thin). The vol-crush conditioner is the only new element and must beat the unconditioned post-opex anchor and a same-crush non-opex placebo. |
| c3 | Watchlist 51 live: SPY-hedged short SVXY entered MOC the session after a >= 10% ^VIX crush, 2018-03+, h=1..3, with the dose ladder and a new split: crush on FOMC k=+1 and/or inside the 3 sessions before a monthly opex | price state (live) | volatility | interaction_cell | W51 arm numbers; registry "FOMC-ahead gate subtracts", "SVXY as a pre-FOMC leg", W33 SPY-residual rule, the 2018-02-28 leverage-break rule. |
| c4 | SPY-hedged short SVXY from the SEPTEMBER opex close to +3 (the month the V4 sleeve carves out), with ^VIX spot as the long-history proxy; today arriving after a 12.8% crush and at VIX/VIX3M 0.832 | event (Sep opex) | volatility | inversion | Registry book finding: V4 September -1.535% over 8 post-break at 0% hit; "September post-opex vol crush INVERTS ... routed to T3, not a short-vol trade". Must survive the SPY residual (else it IS T3 in vol clothing) and the pre/post 2018-02-28 split. |
| c6 | Long SLV after a complex-wide metals UP day (GLD, SLV and GDX all >= +1.5% or a rank equivalent, dollar not up), h=1..5, inside a deep SLV drawdown as a row | price state | other metals (gold rows) | interaction_cell | Mirror of watchlist 30 (complex-break short). Registry owes: the LAG PROFILE (lag 0/1/2), the entry-day split, cluster_note's absolute-value netting, "long SLV pays +0.755% on 32-21" (continuation side), "SLV reversal after a break in a deep drawdown" and "long silver deep in a post-parabolic drawdown" kills. |
| c7 | Long ITB (homebuilders) the session after TLT rises >= 1% while ITB underreacts (ITB day return below its trailing TLT-beta times TLT's), ITB within 10% of its 52w low, h=1..5; rows XHB, XLRE, KRE, XLU | price state (cross-asset lead-lag) | us_large sectors | interaction_cell | Registry XLRE duration-rally kill: "check the exposure ordering" (gate paid best on sectors with the wrong duration sign). The catch-up must scale with each sector's TLT beta or the channel is false. |
| c8 | Long TLT after a +1.0% to +1.5% session from within 4% of its trailing-252 low, h=1..5 (watchlist 17's band, recorded 8-18 for the short) | price state | rates | inversion | A byproduct of a walked 125-cell grid, so it owes the search charge; the ladder is non-monotone (>= 1.5% flips sign). Registry: four dead TLT-at-the-low long cells. |
| c9 | Short crude (CL=F front, USO as the vehicle row) from mid-September into the October refinery-turnaround trough, conditioned on a 21d thrust (USO 21d rank >= 80), h=5..10 | seasonal x price state | energy | flow_mechanics | 09-04 "post-Labor-Day driving-season boundary: no mechanism"; USO wrapper vs CL=F roll (registry); crude thrust fades closed as price states. The turnaround demand mechanism must show as a month ladder with September-October distinct. |
| c10 | FXI into and across the China Golden Week closure (Stock Connect southbound suspended), from ~T-8 to the last pre-holiday session and across the closure, 2015+ vs 2003-2014 as the pre-specified era test | event (Golden Week) | international | event_fingerprint | Zero registry entries on the holiday. FXI break-in-thrust (watchlist 10) and EWZ/FXI pair kills are price states. Sep-2024 stimulus rally is one enormous episode (concentration test mandatory). |

Coverage: 9 candidates, 7 asset classes (us_small, us_large, volatility, other_metals,
rates, energy, international), 4 axes (inversion, flow_mechanics, interaction_cell,
event_fingerprint), 5 event/calendar-anchored (c1, c2, c4, c9, c10) and 4 price-state
(c3, c6, c7, c8).
