# Surface map, 2026-09-16 (Wednesday)

Inputs: `data/pitch_state.json` generated 2026-09-16 05:11 ET; `data/pitch_tape.json`
218 names, freshest bar 2026-09-15 (prior session, fresh). One stale name (LEG), not used.
Pipeline receipts 4/4 green. Fragility dial ma10(63d) 84.1 as of 2026-09-15 (context
for each idea's own risk only, not a sizing input). P/C fear OFF (equity P/C 10d-MA
at the 49.6th pctile of its trailing year, data 2026-09-15). Risk signals on:
Dispersion (composite 88th pctile), VIX Range Compression (21d range 13th pctile).
Midterm year, September, FOMC decision TODAY, VIX expiry TODAY, quad witching Fri 09-18.
Live readings for watchlist verdicts: `01_watch_state.py` (pitch_lab conventions).

Entry reality for today: signal close = 2026-09-15. MOC entry today lands on the
FOMC decision close (lag=1). MOO today is before the 2 pm decision.

## The tape in one paragraph

A rates-led, oil-led repricing under a soft index. ^TNX closed 4.996, exactly its
trailing-252 max, +30.0 bp in 21 sessions and +93.5 bp over 252. TLT, IEF and LQD
all closed AT their trailing-252 lows; IEF z10 -1.84 (5d rank 1.2); HYG z10 -2.27
(5d rank 2.0, the most stretched ETF z10 on the tape). ^MOVE +20.3% in 21d at a
92.9th level pctile. USO +27.85% in 21d, +3.32% on the day, at its 252 high with
z10 +2.73; DBC, XLE, XOP, CVX, COP, EOG, VLO all at 252 highs. SPY -2.44% in 21d
(21d rank 7.1, 2.63% off its high, 6.15% above its 200d); IWM r21 3.6 / r63 1.6.
Two SPDRs hold a 5/21/63-day rank floor simultaneously: XLU (1.2/1.2/0.8, 0.32%
above its 52w low) and XLI (4.4/1.2/0.4). XLY z10 -1.42, XLRE r63 2.0, ITA r21 0.4.
EEM r63 0.4 while +28.3% on the year. Dollar inert (DX r21 48.4). Gold inert
(GLD r5 31). VIX 17.20 vs VIX3M 19.36 (0.888, contango), VIX 21d rank 72.2.

## 1. Calendar events x asset classes

Events in window: PPI 09-10 (-4), CPI 09-11 (-3), FOMC decision 09-16 (0), VIX expiry
09-16 (0), opex + quad witching 09-18 (+2), NFP 10-02 (+12). Election 11-03 is outside.

| event | us_large | us_small | rates | credit | gold | metals | energy | dollar_fx | intl | vol |
|---|---|---|---|---|---|---|---|---|---|---|
| PPI 09-10 (past) | D1 | D1 | D1 | D1 | D1 | D1 | D1 | D1 | D1 | D1 |
| CPI 09-11 (past) | D1 | D1 | D1 | D1 | D1 | D1 | D1 | D1 | D1 | D1 |
| FOMC 09-16 (0) | **CHECK c3** | D2 | **CHECK c1** | **CHECK c4** | **CHECK c6** | D3 | D4 | **CHECK c5** | D5 | D6 |
| VIX expiry 09-16 | D7 | D7 | D8 | D8 | D8 | D8 | D8 | D8 | D8 | D7 |
| opex/quad 09-18 (+2) | D9 | D9 | D10 | D10 | D10 | D10 | D10 | D10 | D10 | D11 |
| NFP 10-02 (+12) | D12 | D12 | D13 | D12 | D12 | D12 | D12 | D12 | D12 | D12 |

Dismissals, each with its reason:
- **D1 PPI/CPI (past).** Both prints are behind the entry; a post-print anchor would have
  needed a 09-10 or 09-11 close. The back-to-back pair cells (watchlist SPY pair, TLT
  post-PPI) are anchored on sessions already gone. Nothing enterable today.
- **D2 FOMC x us_small.** IWM into/out of the September FOMC is IWM beta to the SPY
  cell (c3) plus the closed 09-11/09-14 IWM-vs-SPY pairs (both directions registry-dead).
  Folded into c3's reference class as a vehicle row, not a separate candidate.
- **D3 FOMC x silver/other metals.** No mechanism distinct from gold (c6); SLV's
  complex-break cell is a price-state entry that is not live (SLV +1.21%).
- **D4 FOMC x energy.** "Long crude and energy equity into a midterm-year FOMC" killed
  09-01; the post-decision leg of crude has no policy-transmission story that is not the
  rates story, and USO carries a registry-priced roll drag.
- **D5 FOMC x international.** EEM/EFA/FXI post-decision reduce to dollar plus US beta;
  the pre-FOMC 15-class family was homogeneous (Q p 0.81). International is covered on
  its own price state instead (c9).
- **D6 FOMC x volatility.** Registry: "post-NFP, post-FOMC and post-VIX-expiry vol cells,
  swept and empty". The settle-session SVXY collision (watchlist 44) died on era and
  needs 4 new post-2018 collisions. The pre-decision re-bid form (watchlist 52) arms on
  VIX/VIX3M >= 0.90 at k=-2; 09-14 read 0.887.
- **D7 VIX expiry x us_large/small/vol.** Today is the FOMC x VIX-expiry collision. The
  run-in rung is midterm-wrong-signed (watchlist 48, parks to 2027-03-17); the settle
  rung is era-dead (watchlist 44). Both anchors are the same session as c3's entry, so
  c3 will report the collision subset as a split rather than a new candidate.
- **D8 VIX expiry x rates/credit/gold/metals/energy/dollar/intl.** No mechanism links a
  VIX settlement to these classes; the collision's only measurable proxies (realised
  vol, VIX/VIX3M) refuted the pin story on 09-11.
- **D9 quad witching x us_large/small.** Long IWM into Sept quad (killed 09-07),
  small-cap laggard into Sept quad (09-04), short IWM into Sept quad midterm (09-11),
  second-half-September midterm short (09-14, above-200d 1-3). Closed four ways.
  c3's horizon scan will include h=2 (the quad witching close) as a row.
- **D10 quad witching x non-equity classes.** Quad witching is an equity/index
  derivative expiry; no delivery or roll mechanism in rates, credit, metals, energy
  ETFs, FX or international ETFs tied to it. Not examined for lack of a mechanism.
- **D11 quad x vol.** Registry: September INVERTS the post-opex crush; that stress is
  the event sleeve's T3 territory, and the post-opex entry is 09-18, not today.
- **D12 NFP x everything but rates.** +12 td puts any pre-print anchor at 10-01 or
  later; nothing enterable today inside a 1-10 td hold that has the print as its
  mechanism.
- **D13 NFP x rates.** Midterm-dead (watchlist 1, +0.071% N=12); parks to 2027-01.

## 2. Tape extremes by class

| class | extremes (2026-09-15) | verdict |
|---|---|---|
| us_large | SPY r21 7.1, QQQ r63 3.2, SMH r63 0.4 (-16.2% 63d), XLI and XLU at a 5/21/63 triple floor, XLY z10 -1.42, XLRE r63 2.0, ITA r21 0.4 | **CHECK c7** (triple floor family), **CHECK c8** (floor pair vs energy high). SMH family killed on firing 09-15; ITA and the 22-ETF washout family killed 09-14 (above-200d slice -0.26pp); XLY/XLE pair killed 09-10 as the generic reversal factor; QQQ laggard has no distinct cell. |
| us_small | IWM r21 3.6, r63 1.6, z10 -1.11 (pitch_lab -1.13) | DISMISS: IWM-vs-SPY floor pair closed both directions (09-11, 09-14); outright IWM floor is the washout family (above-200d wrong-signed). |
| rates | ^TNX at 252 max (4.996), +93.5 bp/252; TLT/IEF/LQD at 252 lows; IEF z10 -1.84 | Price-state forms: IG-complex floor killed on firing 09-11 (re-anchoring, not filtering); curve dose killed 09-14; yield-high x MOVE band not live (MOVE 92.9 pctile). IEF 5d flush: HYG-flush kill measured long IEF on those dates at -0.320%. Rates are CHECKED via the event anchor instead (c1). |
| credit | HYG z10 -2.27, 5d rank 2.0, 1.98% above its 252 low, 1.39% below its high | Price-state duration-flush form killed 09-14 (2022+2026 concentrated). **CHECK c4** as the post-decision anchor, a different object. |
| gold / miners | GLD r5 31.0, 20.5% off high; GDX r5 23.0 | Nothing extreme. Gold into a yield-thrust FOMC closed both directions (09-14, 09-15). **CHECK c6** only on the post-decision rung, which neither kill measured on its own. |
| other metals | SLV +1.21% day, 45.5% off high; XME r5 4.0 | DISMISS: silver drawdown/complex-break cells killed 09-01/09-02/09-11; XME 5d flush has no distinguishing mechanism from the washout family. |
| energy | USO 252 high, z10 +2.73 (tape) / +1.65 (pitch_lab), +27.85% 21d; DBC, XLE, XOP, CVX, COP, EOG, VLO at 252 highs; OIH/SLB/HAL/WMB 5d flushes | DISMISS as a standalone long/short: nine energy expressions killed 09-01..09-15 (thrust fade, at-high momentum, placebo ladder 16-for-16, at-high into print, crude reversal, oil services flush). Narrow thrust count reads 0 under the binding pitch_lab convention. Energy enters only as the short leg of c8. |
| dollar_fx | DX-Y.NYB r21 48.4, r5 84.9; UUP r5 81.3 | Price state inert. **CHECK c5** as the post-decision anchor on DX (futures vehicle; UUP registry-dead on drag). |
| international | EEM r63 0.4 with +28.3% 252d; EFA r21 7.9; FXI -1.26% day; EWZ r21 84.1 | **CHECK c9** (EEM at a 63d floor inside a strong year). Must be run against the 09-15 23-ETF "63d floor inside a top-decile year" family kill before it spends anything. EWZ residual vs EEM killed 09-11. |
| volatility | VIX 17.20, VIX/VIX3M 0.888, VIX 21d rank 72.2, 21d range 13th pctile; ^MOVE 92.9 level pctile; SKEW r5 30.6; SVXY r5 29.8 | DISMISS: compression-into-print cells blocked on the dial (84.1 vs <= 68 / < 50 arms) and the SPY residual; bond-vol-bid-while-equity-vol-dead killed 09-03 and 09-09; dispersion-as-short-correlation killed 09-04; post-FOMC vol swept empty. |

## 3. Seasonal and cycle cells

- **Midterm year** conditions everything: pre-FOMC drift inverts in midterm years
  (registry); c3 must report the midterm split of the POST-decision leg.
- **September**, day 16, tdom 11. Second-half midterm September short killed 09-14
  (above-200d entries 1-3). TLT's own tdom profile is positive mid-month (registry method
  trap), so every rates event cell here needs a tdom-matched control.
- **Seasonal board** (asof 2026-08-05, stale for today): zero A/B setups; its regime rows
  are book-sleeve context, not ideas. Its P/C complacency row is dated 08-04 and no
  longer live (equity P/C 49.6 pctile). Nothing to carry.
- **Natural gas September** (watchlist 47): the NG=F wedge is the roll's own September
  seasonality (+0.3601 pp/day, registry). DISMISS.
- **Quarter-end rebalancing** (09-30, +10 td): TLT vs SPY into quarter-end killed 09-14
  (no quarter-end premium; the window's return sat on the post-FOMC and post-quad
  sessions). That note is the reason c1 exists.

## 4. Watchlist verdicts (53 active, 0 expired)

1. Long TLT from the NFP close (nfp x rates): PASS. 2026 is midterm; next NFP 10-02 is midterm-dead (+0.071%, N=12); parks to 2027-01.
2. LQD vs HYG at joint 52w extremes: PASS. HYG 1.39% below its 252 high vs within 0.5%; episode-count arm unchanged at 4.
3. SVXY overnight into CPI: PASS. Next CPI 10-14 (+20 td); concentration arm not re-measurable until then.
4. GLD on a miner-led thrust: PASS. GDX r5 23.0 vs >= 95.
5. XLE on a crude one-day thrust in the 5-6% band: PASS. USO +3.32% on 09-15, below the [5,6)% band.
6. TLT with the IG complex pinned at 52w lows: PASS. The price state is live again (TLT/IEF/LQD all at 0.00% from their 252 lows) but the arm is a construction test (deleted-vs-kept parent anchors >= 0.35pp apart), unchanged at +0.043pp; re-running the price state re-runs the 09-11 kill.
7. SPY on a skew spike: PASS. SKEW r5 30.6 vs >= 95.
8. Fade a crude thrust out of a deep base with a print inside: PASS. USO r63 74.6 vs <= 20; no print inside a 5 td hold.
9. IHI medical-device thrust: PASS. IHI r21 13.1 vs 100.
10. FXI break inside a thrust: PASS. FXI r21 48.0 vs >= 80.
11. TLT November month-position: PASS. Date-parked to November tdom 4-12.
12. Short SPY at a 52w high while TLT at a 52w low: PASS. TLT leg live (0.00%), SPY 2.63% off its high vs within 0.5%.
13. SPY on a vol pop inside calm tape: PASS. VIX 21d rank 72.2 vs <= 25; VIX +0.58% vs >= 5%.
14. Gold on an unconfirmed rate rise: PASS. DX r21 48.4 vs <= 15.
15. Tech vs healthcare after a rotation gap: PASS. XLV minus XLK one-day gap +0.24pp vs >= 3.0pp; SPY 2.63% off high.
16. Short the dollar on an unconfirmed rate rise: PASS. TNX r21 94.4 clears; DX r21 48.4 vs <= 20.
17. Short TLT after a big up day near the 52w low: PASS. TLT -0.27% vs >= +1.5%.
18. Short KRE vs XLF on a bank-breadth washout: PASS. KRE r5 40.5, XLF r5 28.2; the bank complex is split (GS r5 3.6, MS 5.6, BAC 6.0 vs WFC 67.1, SCHW 55.2), short of 70% at r5 <= 20.
19. IEF vs 0.523 TLT curve: PASS. Out of sample only since the 09-14 kill.
20. Narrow energy thrust count: PASS. Count 0 under the binding pitch_lab.zscore (max USO 1.65); the tape convention reads 2 (USO 2.73, VLO 2.10), the convention debt recorded 09-11 is unpaid.
21. Survivorship-free new-high breadth: PASS. Of 11 SPDRs only XLE sits at a 252 high; SPY 2.63% off; breadth at the wrong extreme for the arm.
22. Sector washout into a 52w high family: PASS. The r5 <= 5 SPDRs are XLI (-9.47% off high) and XLU (-12.27%), both outside the within-5% clause.
23. Utilities washout with TLT hit alongside: PASS. XLU r21 1.19 clears <= 5; TLT r21 34.13 vs < 25 (further than 09-15's 25.79).
24. Bare dollar washout: PASS. Parks to a non-midterm date; DX r21 48.4 in any case.
25. HYG printing a 52w high while the index has not: PASS. HYG 1.39% below its high.
26. SMH deep correction family: PASS. Out of sample only since the 09-15 kill on firing.
27. Rates repricing with zero credit stress: PASS. HYG 1.39% below its high vs within 0.25%.
28. IEF out of Jackson Hole: PASS. Date-parked to 2027.
29. The laggard still falling, pooled 29 ETFs: PASS. No tape ETF with r21 >= 90 has r63 <= 10 (USO 74.6, DBC 77.4 are the ETF r21 leaders).
30. Short silver after a complex break: PASS. SLV +1.21%, GDX -0.01%.
31. Long duration at a yield high with MOVE mid-range: PASS. TNX at its 252 max clears; ^MOVE level pctile 92.9 vs [40,50).
32. IWM December month-end overnight: PASS. Date-parked to December.
33. Energy at a fresh 52w high on an index-down session, h=21: PASS, and the STATE is live (XLE at 0.00% from its 252 high, SPY -0.46%). The arm is two permutation numbers plus a standing blocker that "may not be waived": the dose is inverted, and today's -0.46% sits in exactly the mild interior bin [-0.5,-0.25)% the blocker names. One more observation cannot move the h=21 family P or the ex-2022/2026 excess.
34. SVXY into a scheduled print from a (5,15] VIX range: PASS. Today's FOMC k=-2 anchor was 09-14 and is gone; arm (iii) needs the dial <= 68.0 against 84.1.
35. **Pooled sector triple rank floor: CHECK.** Restated per the 09-14 note as a live-state arm (a nine-SPDR member at the 5/21/63 <= 10 floor): XLU 1.2/1.2/0.8 and XLI 4.4/1.2/0.4 both hold it. The portfolio-overlap arm is retired by the 2026-09-08 owner decision. Carries the 09-14 debt: the washout family pays -0.26pp above SPY's 200d in midterm years, and SPY is 6.15% above. -> candidate c7.
36. SPY into a scheduled print out of a dead VIX range: PASS. Dial 84.1 vs < 50; anchor was 09-15.
37. Risk premium across an extended closure: PASS. Next >= 3-day closure is Thanksgiving.
38. Post-NFP duration after a moderate prior miss: PASS. Next NFP 10-02.
39. SVXY at the first close after an extended closure: PASS. No closure.
40. SPY vs IWM with the dial in 56-70: PASS. Dial 84.1.
41. HYG at the first close after an extended closure: PASS. No closure.
42. Short IEF with commodities at a 252 high and a print inside: PASS. DBC at its 252 high clears; no inflation print inside h=5 (CPI 10-14); the placebo-rank arm is unchanged.
43. SPY across the September PPI-CPI pair: PASS. The pair was 09-10/09-11.
44. TLT from the PPI release close at a 252 yield high: PASS. Next PPI 10-15. Its post-release anchor is the structural cousin of c1.
45. SPY with HYG at a 252 high as TNX prints one: PASS. TNX clears, HYG 1.39% below its high.
46. SVXY on the settle session of a VIX expiry x FOMC collision: PASS, and the state is live today. The arm is +4 NEW post-2018 collisions on the -0.5x vehicle; today is one. Not entered on a single collision.
47. NG=F September seasonal: PASS. Mechanism arm unmet; the registry measured the NG=F September edge as the roll wedge itself.
48. SPY run-in on an FOMC x VIX-expiry collision, non-midterm: PASS. 2026 is midterm; parks to 2027-03-17.
49. Deep 5-day flush inside a top-decile 63d trend (r5 <= 2, r63 >= 90): PASS. No tape ETF meets it (XOP r63 89.7 at r5 67.9).
50. XLV vs 0.71 SPY after a healthcare flush: PASS. XLV r5 50.0 vs <= 1.
51. Hedged short SVXY after a 10% VIX crush: PASS. VIX +0.58%.
52. SVXY into an FOMC after a re-bid, backwardated only: PASS. The 09-14 k=-2 close read VIX/VIX3M 0.887 vs >= 0.90, and 09-15 was the entry; next anchors 10-26 and 12-07.
53. Short a large bank vs XLF after a non-earnings 1.5 ATR slide: PASS. Largest bank drop on 09-15 was BNY -1.90% = 0.92 ATR (STT 0.69).

## 5. Scoreboard read

Five graded ideas lifetime (3 B at +0.448 avgR, 2 C at -0.237; event_fingerprint 2 at
+0.622, inversion 1 at -0.62). That is a handful; no axis gets more or fewer slots on
it. The more telling number is outside the scoreboard: one published idea since
2026-08-20 against 11 stand-downs, which is a reason to apply the small-N doctrine and
the pre-specified-hypothesis rule exactly as written, not a reason to relax a kill.

## 6. Candidates selected from this map (9)

| id | candidate | anchor | class | axis | registry / watchlist adjacency |
|---|---|---|---|---|---|
| c1 | Long TLT MOC on the FOMC decision close, h=1..5, with ^TNX at a trailing-252 max on the eve | event (FOMC, post-decision) | rates | event_fingerprint | Prior is pre-specified (Hillenbrand 2021: the secular fall in long yields is concentrated in the FOMC window). Registry: tdom-matched control mandatory; "into the FOMC" duration closed twice (09-01, 09-15) at k=-1, not the decision close; QE-12 kill located the TLT-SPY return on the post-FOMC session (+39.8 bp). |
| c5 | Short the dollar (DX-Y.NYB, DX futures vehicle) MOC on the decision close with ^TNX at a 252 max | event | dollar_fx | event_fingerprint | Same mechanism as c1 (post-decision yield relief). UUP registry-dead on drag; DX futures passed cost before. |
| c6 | Long GLD MOC on the decision close with ^TNX at a 252 max | event | gold | event_fingerprint | Gold INTO a yield-thrust FOMC closed both directions; the decision-close rung alone was never isolated. |
| c3 | Long SPY MOC on the decision close to the end of FOMC week 0 (h=1..3, h=2 = quad witching close), midterm split | event | us_large | inversion | Midterm inverts the pre-decision drift (registry); does the post-decision leg invert too? Collision subset (VIX expiry) reported as a split. |
| c2 | Long XLU MOC on the decision close with XLU at a rank floor and ^TNX at a 252 max | event x price state | us_large (utilities) | interaction_cell | Utilities dead in eight expressions (watchlist 23 note); XLU+TLT-hit form not live. Needs the rate-sensitive family (XLRE, XHB) as reference class. |
| c4 | Long HYG MOC on the decision close after a five-day flush (z10 <= -2) | event x price state | credit | interaction_cell | 09-14 kill of the price-state duration-flush form (2022+2026). The decision-close anchor is the new object; must show it filters rather than re-labels that cell. |
| c7 | Long the nine-SPDR members at a 5/21/63 <= 10 triple floor (today XLU + XLI equal weight), h=10 | price state | us_large (sectors) | interaction_cell | Watchlist 35 bare pooled form +1.131% h=10 over 493 episodes, I2 0. 09-14: washout family above SPY's 200d is wrong-signed in midterm years. Near-high gate registry-negative (not used here). |
| c8 | Long XLU + XLI against short XLE: two SPDRs at the triple floor while energy prints a 252 high | price state | us_large x energy | relative_value | 09-10 XLY/XLE kill: the generic long-worst/short-best SPDR pair beat the labelled pair. Must beat that generic version on the same dates. |
| c9 | Long EEM at a 63-day rank floor (<= 2) inside a year still up > 20%, h=5..10 | price state | international | interaction_cell | 09-15 23-ETF "63d floor inside a top-decile year while still falling" family kill; check whether EEM is in it and whether today's state is that cell. |

Coverage: 9 candidates, 6 asset classes (rates, dollar_fx, gold, us_large, credit,
international; energy as a pair leg), 4 axes (event_fingerprint, inversion,
interaction_cell, relative_value), 6 event-anchored (c1-c6) and 3 price-state (c7-c9).
