# Surface map, 2026-09-21 (Monday)

Inputs: `data/pitch_state.json` generated 2026-09-21 05:10 ET; `data/pitch_tape.json`
218 names, freshest bar 2026-09-18 (the prior session, fresh). One stale name (LEG), not
used. Pipeline receipts 7/7 green (prices bar, dial and P/C all dated 2026-09-18).
Fragility dial ma10(63d) 84.2 in the state (80.8 on the parquet tail read by
`01_watch_state.py`; raw 5d 22.8, 21d 46.0, 63d 75.0): context for each idea's own risk
only, never a sizing input. P/C fear OFF (equity P/C 10d-MA at the 57.5th pctile, data
2026-09-18). Risk signals on: Low Absorption Ratio (AR 0.333, 6th pctile) and NYSE Net
Highs (5d EMA -149 with SPY 1.84% below its 252-session closing high). Midterm year,
September, day 21 (Monday, Yom Kippur; NYSE open). Live readings: `01_watch_state.py`,
`02_watch_extra.py` (pitch_lab conventions), tape sort `_tape_dump.py`.

Entry reality: signal close = 2026-09-18 (September opex + quad witching). MOC today lands
on the 09-21 close. From that close: GIS/PAYX print 09-23 BMO (+2), COST 09-24 (+3),
quarter-end 09-30 (+7, also Japan's fiscal half-year end), MU 09-30 and CAG 09-30 BMO
(+7), NKE 10-01 (+8), Q4 turn-of-month 10-01 (+8), NFP 10-02 (+9). Election 11-03 sits
+31 but is already inside the November VIX future. The 09-16 pitch (long DX, 5 td, not
approved) grades through the 09-23 close.

## The tape in one paragraph

A rate shock with an oil shock under a calm index. ^TNX 4.998 against a 252 max of 5.006
(+34.5 bp in 21 sessions, +92.2 bp over 252), ^IRX 3.978, so the 10y-3m spread is +102 bp
and steepening from the long end. IEF closed 0.08% above its 252 low, LQD 0.40%, TLT
0.67%; ^MOVE +5.80% to 80.6 (86.9th pctile of its year). The dollar holds a 5d rank of 91
(UUP 0.73% off its 52w high) and USDJPY is 156.1. Crude holds a +17.5% 21d thrust (USO
-0.96% Friday), VLO sits at a 252 high (+75.6% 63d). Equities: SPY +0.13% and 1.84% off
its high, QQQ +0.92% on the week, but 18 tape names sit within 1% of a 52w low against 4
within 1% of a high, and only 16.1% of the tape has a 21d rank above 50. The lows are the
rate-sensitive and defensive complex: XLU at 0.00% above its low (r5/r21/r63 7.1/1.6/1.2),
CMS PEG EXC SRE PCG, PEP MCD TAP, LOW WHR VMC, AON, NKE; VNQ/IYR z10 -2.1. Banks are at
a breadth floor again (11 of 11 at r5 <= 20: BAC 0.4, GS 1.6, BNY 2.0, MS/STT 3.2; median
r63 34.5) even as the curve bear-steepens. International lagged Friday on a flat dollar
(EFA -1.01%, EWJ -0.93% against SPY +0.13%, DX-Y.NYB 0.00%). ^VIX -4.08% to 14.81,
VIX/VIX3M 0.812, SVXY 0.44% off its 52w high, UVXY at its 52w low. Metals: GLD +0.71%,
SLV +1.63% on the week's yield rise; GDX -0.46%.

## 1. Calendar events x asset classes

Events in the state window: FOMC decision 09-16 (-3), VIX expiry 09-16 (-3), opex + quad
witching 09-18 (-1, the signal close), NFP 10-02 (+9). Non-macro anchors inside a 10 td
hold, enumerated because the macro file does not carry them: single-name prints (GIS,
PAYX, COST, MU, CAG, NKE), quarter-end 09-30 (+7) which is also Japan's fiscal
half-year end, Q4 turn-of-month 10-01 (+8), and the November election priced in the
VIX futures curve from here.

| event | us_large | us_small | rates | credit | gold | metals | energy | dollar_fx | intl | vol |
|---|---|---|---|---|---|---|---|---|---|---|
| FOMC 09-16 (k=+3 at entry) | D1 | D1 | D2 | D2 | D2 | D2 | D3 | D4 | D4 | D5 |
| VIX expiry 09-16 | D5 | D5 | D6 | D6 | D6 | D6 | D6 | D6 | D6 | D5 |
| opex / quad 09-18 (signal close) | D7 | D8 | D9 | D9 | D9 | D9 | D9 | D9 | **CHECK c6** | D10 |
| single-name prints 09-23..10-01 | **CHECK c3, c9** | - | - | - | - | - | - | - | - | - |
| quarter-end 09-30 (+7) | **CHECK c4** | D11 | D12 | D12 | D12 | D12 | D12 | **CHECK c8** (+W56 row) | D13 | D12 |
| Japan fiscal half-year end 09-30 | - | - | - | - | - | - | - | **c8** | D13 | - |
| Q4 turn-of-month 10-01 (+8) | D11 | D11 | D12 | D12 | D12 | D12 | D12 | D12 | D12 | D12 |
| NFP 10-02 (+9) | D14 | D14 | D15 | D14 | D14 | D14 | D14 | D16 | D14 | D14 |
| election 11-03 (in the Nov VIX future) | - | - | - | - | - | - | - | - | - | **CHECK c7** |

Dismissals, each with its reason:
- **D1 FOMC k=+3 x us_large/small.** 09-16 measured SPY from the decision close to the end
  of week zero at placebo rank 11 of 11; k=+3 is inside that dead window.
- **D2 FOMC x rates/credit/gold/metals.** Registry 09-16: TLT from the decision close at
  the 252 yield max -1.53% at h=3; three gold rungs closed 09-14/15/16; the HYG
  decision-close flush filters the wrong way.
- **D3 FOMC x energy.** No policy channel into crude distinct from the dollar.
- **D4 FOMC x dollar/intl.** The 09-16 ship (long DX, 5 td) is still grading through the
  09-23 close; international reduces to dollar plus US beta (09-16 D5).
- **D5 VIX expiry / FOMC x vol.** Settle rung era-dead (W46), run-in midterm
  wrong-signed (W48), post-crush hedged short SVXY in its losing dose bucket (09-18).
- **D6 VIX expiry x non-equity classes.** No channel from a VIX settlement.
- **D7 quad x us_large.** Post-opex SPY closed both ways; the vanna-unwind short killed
  09-18 (filter wrong-signed); the VIX-crush-into-opex long is cycle-parked (W59).
- **D8 quad x us_small.** Killed 09-18 by name: the September quad is the losing half of
  its own cell (ungated September post-quad 6-19, -2.03%); parked to December (W58).
- **D9 quad x rates/credit/gold/metals/energy/FX.** Equity-derivative expiry with no
  delivery or roll channel; the 100-cell cross-asset post-opex grid is closed.
- **D10 quad x vol.** The September post-opex hedged short SVXY was emptied by the SPY
  residual on 09-18; V4's September carve-out is T3's territory.
- **D11 quarter-end / turn-of-month x us_small and index.** Month-end closed on six
  forms; SPDR window dressing killed 09-17 (no quarter-end premium over month-ends);
  the post-QE SPDR reversal pair is date-parked to the 09-30 close (W57). The
  single-STOCK loser form has never been measured and is c4.
- **D12 quarter-end x rates/credit/gold/metals/energy/vol.** TLT into quarter-end
  (pension rebalancing) killed 09-14, "no quarter-end premium, gate has no gradient";
  no documented quarter-end channel in credit, metals, energy or VIX that is not the
  equity/bond rebalance already closed.
- **D13 quarter-end x intl.** EWJ into the Japanese half-year dividend-reinvestment
  window killed 09-17 (control months beat the chosen ones). The CURRENCY leg of the
  same fiscal calendar was never measured and is c8.
- **D14 NFP (+9) x everything but rates/dollar.** The only reachable anchor is the
  pre-print close at h=9; pre/post-NFP equity direction is registry-swept; the
  dead-VIX-range pre-print cell (W36) is blocked on the dial (80.8 vs < 50).
- **D15 NFP x rates.** Midterm-dead (W1); into-print duration with the ten-year near its
  high wrong-signed (09-03 kill). The round-number level (c2) is a price state, not an
  NFP anchor, and reports the NFP-in-hold split as a row.
- **D16 NFP x dollar.** Would be a second long-DX idea inside the 09-16 ship's grading
  window with nothing materially changed; dismissed on repetition.

## 2. Tape extremes by class

| class | extremes (2026-09-18) | verdict |
|---|---|---|
| us_large | XLU at its 52w low with the 5/21/63 floor; utilities CMS PEG EXC SRE PCG at lows; staples PEP MCD TAP at lows (PEP z10 -2.09); LOW WHR VMC AON NKE at lows; REITs z10 -2.1; banks 11 of 11 at r5 <= 20; META r21 98.4, AMD z10 2.32; IBM -3.45% Friday | **CHECK c1** (W23 armed: XLU r21 1.59 AND TLT r21 24.60), **CHECK c5** (bank flush under a bear steepener: the rates gate is new; bank breadth alone is a dead label), **CHECK c3/c9/c4** (the names at lows carry prints and a quarter-end). DISMISS: staples/food flush (closed 09-08 both directions), REIT floor vs XLF (09-10 anti-filter), homebuilders' bond catch-up (09-18), META/AMD thrusts (single-name momentum with no event or flow story; 09-15 failed-thrust kill), SPDR triple floor (W35: midterm with SPY above its 200d is the wrong-signed slice, -0.393pp), narrow leadership (09-14). |
| us_small | IWM r21 6.0, r63 0.8, z10 -1.35 (pitch_lab) | DISMISS: post-quad IWM killed 09-18 on the September row; IWM vs SPY at a TNX high killed 09-11 (cost). |
| rates | ^TNX 4.998, 0.16% below its 252 max and 0.2 bp below the 5.00% round number; IEF 0.08% above its low; MOVE 86.9 pctile | **CHECK c2** (a whole-percent yield level after a thrust: the level itself has never been tested, and the half/quarter levels give a built-in placebo). DISMISS: the four dead outright forms (IG floor W6, month-end gradient, post-FOMC, TNX high into prints), MOVE-mid-range (W31, pctile 86.9), curve pairs (W19 OOS only). |
| credit | HYG 1.20% off its high, z10 -1.52; LQD 0.40% above its low | DISMISS: W2/W25/W27/W45 need HYG within 0.25-0.5% of its high; the flush arm (W55) needs z10 <= -2 and the live move is duration (IEF r5 39.7, HYG spread flat), which is the 09-14 kill. |
| gold / miners | GLD +0.71% Friday on a +1 bp yield day, 19.1% off its high; GDX -0.46% | DISMISS: gold with the ten-year at a 252 high closed on three rungs (09-14/15/16); W14 needs DX r21 <= 15 (78.6); miner/metal ratio closed both ways. |
| other metals | SLV +1.63% Friday, +3.11% 5d, 43.3% off its high | DISMISS: complex up-day long SLV killed 09-18 (edge outside the tradeable window); Friday was not a complex day (GDX -0.46%). |
| energy | USO +17.5% 21d, -0.96% Friday; VLO at a 252 high; XOP r5 19.8 while DBC 2.23% off its high | DISMISS: ten closed crude/energy entries including the thrust fade, the reversal day, oil services on a flush under a crude thrust (09-15, crude clause does not filter), VLO vs XLE (09-17) and the turnaround short (09-18). The crack-spread seasonal (RVP switch) cannot be measured: no RB=F or HO=F series in the cache. W60 (CL=F October tdom-13) is date-parked to 10-19. |
| dollar_fx | DX-Y.NYB r5 91.3, r21 78.6, r63 12.3; UUP 0.73% off its high; USDJPY 156.1 inside a 147.1-163.9 year | **CHECK c8** (the yen leg of Japan's fiscal half-year end, never measured). DISMISS the long-DX forms: 09-16 ship still grading; DX at a 63d floor with a TNX high killed 09-10; QE-dollar killed 09-17. |
| international | EFA -1.01%, EWJ -0.93% on the quad Friday against SPY +0.13% with the dollar flat; EEM r63 0.8; FXI r21 24.6 | **CHECK c6** (an expiry-day international gap against the US index). DISMISS: EEM floor (09-16), EWJ quarter-end (09-17), EWJ vs EFA term premium (09-10), Golden Week FXI (09-18). |
| volatility | ^VIX 14.81 (-4.08%), within 10% of its 52w low; VIX/VIX3M 0.812; SVXY 0.44% off its high; UVXY at its low; SKEW r5 15.5 | **CHECK c7** (the election-year October roll into the election-bearing future). DISMISS: MOVE/VIX divergence on any leg (registry: forward ^VIX after a MOVE spike runs below its baseline; corr -0.05 over 5,876 days), sub-5th-pctile VIX range into a dense calendar (09-10, gate attribution), post-crush forms (dose bucket, 09-18). |

## 3. Seasonal and cycle cells

- **Midterm year** conditions everything; every checker reports a midterm split.
- **Late September.** Midterm second-half short killed 09-14; T3 (short IWM, Sep opex to
  the last September session) was skipped by its own washout rule and the inversion
  was killed 09-18. Nothing left in the index lane.
- **Quarter-end.** Every INDEX and SECTOR form is closed or parked; the single-stock
  loser form (c4) and the yen fiscal-half-year form (c8) are the two unmeasured objects.
- **October volatility in an election year.** The VX curve carries a kink at the
  November contract; SVXY rolls into it during October (c7).
- **Seasonal board** (asof 2026-08-05, stale): zero A/B setups; its P/C complacency row
  is not live (equity P/C 57.5 pctile).
- **NG=F September** (W47): mechanism arm unmet, dismissed.

## 4. Watchlist verdicts (60 active, 0 expired)

1. TLT from the NFP close: PASS. Midterm-dead; parks to 2027-01.
2. LQD vs HYG at joint 52w extremes: PASS. HYG 1.20% off its high vs within 0.5% (LQD 0.40% clears).
3. SVXY overnight into CPI: PASS. Next CPI 10-14 (+17).
4. GLD on a miner-led thrust: PASS. GDX r5 34.9 vs >= 95.
5. XLE on a crude thrust in the 5-6% band: PASS. USO -0.96%.
6. TLT with the IG complex at 52w lows: PASS. State live (IEF 0.08%, LQD 0.40%, TLT 0.67% off lows) but the arm is a filter-vs-reanchor construction test that one session cannot move.
7. SPY on a skew spike: PASS. SKEW r5 15.5 vs >= 95.
8. Fade a crude thrust out of a deep base: PASS. USO r63 73.8 vs <= 20.
9. IHI thrust: PASS. IHI r21 7.9 vs 100.
10. FXI break inside a thrust: PASS. FXI r21 24.6 vs >= 80.
11. TLT November month-position: PASS. Date-parked to November.
12. Short SPY at a 52w high with TLT at a low: PASS. SPY 1.84% off its high vs within 0.5% (TLT 0.67% clears).
13. SPY on a vol pop in calm tape: PASS. ^VIX -4.08% (a fall, not a pop).
14. Gold on an unconfirmed rate rise: PASS. DX r21 78.6 vs <= 15.
15. Tech vs healthcare after a rotation gap: PASS. XLV minus XLK -1.07pp vs >= +3.0pp.
16. Short the dollar on an unconfirmed rate rise: PASS. DX r21 78.6 vs <= 20.
17. Short TLT after a big up day near the low: PASS. TLT -0.65%.
18. Short KRE vs XLF on a bank-breadth washout: PASS. Breadth live (11 of 11 at r5 <= 20) but the median 63d rank is 34.5 against intact, and the arm is an ex-crisis cost threshold. The rates-gated long-XLF object is a different cell and is c5.
19. IEF vs 0.523 TLT curve at a yield high: PASS. Out of sample only since 09-14.
20. Narrow energy thrust count: PASS. Count 0 (max VLO 1.20 under pitch_lab.zscore).
21. Survivorship-free new-high breadth: PASS. SPY 1.84% off its high vs > 2.0% (misses by 16 bp).
22. Sector washout into a 52w high: PASS. No SPDR at r5 <= 5 (XLF and XLU 7.1 lowest).
23. Utilities washout with TLT hit: **ARMED -> CHECK c1.** XLU r21 1.59 clears <= 5 AND TLT r21 24.60 clears < 25, both on the 09-18 close. Its three debts go to the checker verbatim (pre-registration/forward status, the nine-sector reference class never run on this form, the bad utilities prior), plus W35's adverse fact that midterm with SPY above its 200d is the wrong-signed slice of the sector-floor family.
24. Bare dollar washout: PASS. Date-parked to 2027-06; DX r21 78.6.
25. HYG 52w high while the index is not: PASS. HYG 1.20% off.
26. SMH deep correction family: PASS. Out of sample only.
27. Rates repricing with zero credit stress: PASS. HYG 1.20% vs within 0.25% (IEF 0.08%, LQD 0.40% clear).
28. IEF out of Jackson Hole: PASS. Date-parked to 2027.
29. The laggard still falling, pooled: PASS. No index or industry ETF at r21 >= 90 with r63 <= 10.
30. Short silver after a complex break: PASS. SLV +1.63% (no break).
31. Long duration at a yield high with MOVE mid-range: PASS. MOVE level pctile 86.9 vs [40,50).
32. IWM December month-end overnight: PASS. Date-parked.
33. Energy at a fresh 52w high on an index-down session: PASS. SPY +0.13%, XLE 2.46% off its high.
34. SVXY into a print from a (5,15] VIX range: PASS. No print at k=-2 (NFP +9); dial 80.8 vs <= 68.
35. Pooled sector triple floor: PASS. XLU holds the 5/21/63 floor (7.1/1.6/1.2) but the arm is SPY below its 200d (+6.9% above). Its midterm-above-200d slice (-0.393pp) is carried into c1 as an adverse fact.
36. SPY into a print out of a dead VIX range: PASS. Dial 80.8 vs < 50.
37. Risk premium across an extended closure: PASS. Next is Thanksgiving.
38. Post-NFP duration after a moderate miss: PASS. Next NFP 10-02; checked on the print session.
39. SVXY after a closure: PASS. No closure.
40. SPY vs IWM with the dial in 56-70: PASS. Dial 80.8.
41. HYG after a closure: PASS. No closure.
42. Short IEF with commodities at a high and a print inside: PASS. DBC 2.23% off its high; no inflation print inside h=5.
43. SPY across the September PPI-CPI pair: PASS. The pair was 09-10/11.
44. TLT from the PPI release close: PASS. Next PPI 10-15.
45. SPY with HYG at a high as TNX prints one: PASS. HYG 1.20% off (TNX 0.16% off its max).
46. SVXY on a VIX expiry x FOMC settle session: PASS. The 09-16 collision is recorded, not traded.
47. NG=F September: PASS. Mechanism arm unmet.
48. SPY run-in on an FOMC x VIX expiry collision, non-midterm: PASS. Midterm; parked to 2027-04.
49. Deep 5-day flush inside a top-decile 63d trend: PASS. No ETF at r5 <= 2 with r63 >= 90 (XLF r5 7.1, r63 69.4).
50. XLV vs 0.71 SPY after a healthcare flush: PASS. XLV r5 77.0 vs <= 1. Expires 10-05.
51. Hedged short SVXY after a 10% VIX crush: PASS. ^VIX -4.08%. Expires 10-05.
52. SVXY into FOMC after a re-bid, backwardated: PASS. Next decision 10-28.
53. Short a large bank vs XLF after a 1.5 ATR intraday slide: PASS. Largest slide GS -0.32 ATR close-to-close (open-to-close -6.96 on a 29.42 ATR).
54. Long TLT across the FOMC announcement session: PASS. Parked to the 10-27 eve close.
55. Long HYG after a spread-driven flush: PASS. HYG z10 -1.52 vs <= -2.
56. Long the dollar into the September quarter-end close: PASS on the September form (it is the 09-16 post-FOMC ship by construction, the entry's own words). Its unmeasured arm (no-FOMC quarters vs ordinary month-ends, >= +0.25pp since 2008) is run as a ROW inside c8, which measures FX at quarter-ends anyway.
57. Short the quarter's two best SPDRs vs its two worst from the QE close: PASS. Date-parked to the 09-30 close.
58. IWM from a quad close after a washout, outside September: PASS. Next quad 12-18.
59. SPY from an opex close after a VIX crush: PASS. Cycle-parked to 2027-01-15.
60. Short CL=F from October tdom 13: PASS. Date-parked to the 10-19 close.

## 5. Scoreboard read

Five graded ideas lifetime (3 B at +0.448 avgR, 2 C at -0.237; event_fingerprint 2 at
+0.622, inversion 1 at -0.62, interaction_cell 1 at +0.146, relative_value 1 at +0.099).
Still a handful; no axis earns or loses a slot on it. One of today's candidates is an
inversion (c3); the one graded inversion lost, noted and not weighted. Context: 13 of the
last 14 mornings stood down, so the registry is dense around every rates, energy and
index state live today; candidates were chosen where the registry is empty (the yield
LEVEL, single-stock quarter-end and pre-print objects, the yen fiscal calendar, the VX
election roll, the expiry-day international gap) or where a pre-registered arm fired
(c1).

## 6. Candidates selected from this map (9)

| id | candidate | anchor | class | axis | registry / watchlist adjacency |
|---|---|---|---|---|---|
| c1 | W23 armed: long XLU when XLU r21 <= 5 AND TLT r21 < 25 on the same session, h=3/5; hedged row long XLU against beta-TLT | price state (armed arm) | us_large sector (rates-conditioned) | interaction_cell | W23 debts; W35 midterm-above-200d slice -0.393pp; utilities dead in eight expressions (2026-08-12 rank21 cell, 09-16 decision-close XLU); 09-10 XLRE rates-gate anti-filter. |
| c2 | Long TLT/IEF when ^TNX closes within 5 bp below (or first crosses) a whole-percent level after a 63d rise >= 50 bp and at a 252 max, h=5/10; placebo: the same rule at the .25/.50/.75 levels | price state | rates | flow_mechanics | Four dead outright duration forms at the low/high (W6 IG floor 09-11, month-end, post-FOMC, TNX high into prints); W31 MOVE mid-range. The round-number gate must beat the plain 252-max gate and the off-round placebo levels. |
| c3 | Short a liquid name at/near its 52w low (21d rank <= 15) from ~T-8 to its pre-print close, pooled; live on NKE (10-01) | event (earnings) x price state | us_large single names | inversion | Inverts the pre-print washout lane (killed) and the COST ladder ("the lagging name keeps FALLING into its print", -4.52pp at 9 td on one name). Earnings calendar usable only 1996+. Placebo offset ladder is 12-for-12 against earnings anchors: mandatory. |
| c9 | Long a liquid 63d winner (r63 >= 80) from T-k to its pre-print close, pooled; live on GIS/PAYX (09-23 BMO), CAG (09-30 BMO) | event (earnings) x price state | us_large single names | event_fingerprint | The COST gate ladder's >25 bucket +0.779%; SMH-into-NVDA falsified by print month; pre-print washout; ladder 12-for-12. Paired with c3 as the momentum split of one object. |
| c4 | Short liquid names within 2% of a 52w low from QE-7 to the quarter-end close (window dressing), control = the same names/state at ordinary month-ends; reversal row QE to QE+5 | event (quarter-end) | us_large single names | flow_mechanics | SPDR window dressing killed 09-17 (no QE premium over month-ends); W57 SPDR reversal parked to 09-30; survivorship bias in the cache flatters longs of losers, so a short-loser edge is measured conservatively. |
| c5 | Long XLF (rows KRE, big banks EW) when >= 80% of the 11-bank complex sits at r5 <= 20 AND the 10y-3m spread widened >= 25 bp over 21 sessions (bear steepener), h=3/5 | price state | us_large financials (rates-conditioned) | interaction_cell | Bank breadth alone a dead label (Cochran Q twice); W18 KRE vs XLF; 09-15 money-centre bank vs XLF. The curve gate is the only new element and must beat the ungated breadth floor. |
| c6 | Long EFA against beta-SPY after EFA lags SPY by >= 1.0pp on a quad-witching (or monthly opex) session with the dollar flat, h=1..5; control = same gap on non-expiry sessions | event (quad witching) | international | relative_value | 09-18 c1 rows (EFA from the quad close, unconditioned); EWJ/EFA/EEM kills are price states. Must separate from generic EFA-SPY gap reversal on any day. |
| c7 | Short SVXY (long vol) across October in election years (both presidential and midterm, 2012+), entered at the late-September close, gated on ^VIX within 15% of its 252 low; row: all Octobers; row: VIX spot | event (election in the VX curve) x seasonal | volatility | flow_mechanics | V4 September carve-out (T3 territory); 09-18 hedged post-opex short SVXY emptied by the SPY residual (so this must survive the same SPY residual); SVXY leverage break 2018-02-28 (-1x to -0.5x). |
| c8 | Long yen (short JPY=X) from QE-7 to the quarter-end close in March and September (Japan fiscal year and half-year ends), control June/December and ordinary month-ends; reversal row QE to QE+5; W56 row: DX at no-FOMC quarter-ends vs ordinary month-ends since 2008 | event (fiscal calendar) | dollar_fx | flow_mechanics | EWJ half-year dividend reinvestment killed 09-17 (a different leg: equities, not the currency); registry notes EWJ/yen correlation +0.020. Futures 6J are tradeable (manual card). |

Coverage: 9 candidates, 5 asset classes (us_large incl. sectors and single names, rates,
international, volatility, dollar_fx), 5 axes (interaction_cell, flow_mechanics,
inversion, event_fingerprint, relative_value), 6 event/calendar-anchored (c3, c9, c4, c6,
c7, c8) and 3 price-state (c1, c2, c5).

## 7. Second wave (added 05:35 ET after all nine first-wave candidates died)

Round-1/2 outcome of the first wave (checkers kA/kB/kC): nine KILLs, none reached round
3. Rather than stand down on the first pass, two cells from the section 1 grid that the
registry has never measured were opened. Both were dismissed or unexamined in the grid
above only because the first wave was sized to nine; neither is a rescue of a kill.

| id | candidate | anchor | class | axis | registry / watchlist adjacency |
|---|---|---|---|---|---|
| c10 | Long SPY from the late-September close into a MIDTERM election (entry 30 sessions before 11-03), h=5/10/21/to the pre-election close; controls: presidential years, odd years, the offset ladder, a post-election anchor row | event (election) | us_large index | event_fingerprint | Zero registry entries on the election run-in. 09-14 kill of the second-half-September midterm SHORT (2002/2022, both below the 200d, carried 120%): the long must be read episode by episode with SPY vs its 200d. |
| c11 | Long LQD against beta-IEF from QE-7 through early earnings season, every quarter (the IG issuance blackout); controls: same-length heavy-issuance windows, offset ladder, SPY-beta residual; HYG row | event (issuance calendar) | credit | flow_mechanics | Zero registry entries on issuance or blackout supply; HYG/LQD price-state cells are dead in many forms (W2, W25, W27, W45, W55) and this must not be one re-labelled. |

Coverage with the second wave: 11 candidates, 6 asset classes (adds credit), 5 axes,
8 event/calendar-anchored and 3 price-state.
