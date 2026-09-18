# Surface map, 2026-09-17 (Thursday)

Inputs: `data/pitch_state.json` generated 2026-09-17 05:10 ET; `data/pitch_tape.json`
218 names, freshest bar 2026-09-16 (the prior session, fresh). One stale name (LEG), not
used. Pipeline receipts 4/4 green (prices bar, dial and P/C all dated 2026-09-16).
Fragility dial ma10(63d) 82.9 as of 2026-09-16 (context for each idea's own risk only,
not a sizing input). P/C fear OFF (equity P/C 10d-MA at the 52.0th pctile of its trailing
year, data 2026-09-16). Risk signal on: VIX Range Compression only (21d range 13th pctile).
Midterm year, September, day 17 (tdom 12). Live readings: `01_watch_state.py`,
`02_watch_extra.py` (pitch_lab conventions). Tape sort: `_tape_sort.py`.

Entry reality: signal close = 2026-09-16 (the FOMC decision session). MOC today lands on
the 09-17 close, the session BEFORE September opex and quad witching (09-18). The
quarter's last session, 09-30, is exactly 9 sessions after today's close. NFP 10-02 is
+11. Yom Kippur 2026 falls on Monday 09-21 (Rosh Hashanah was Saturday 09-12), so
today's close sits inside the Rosh Hashanah to Yom Kippur window, two sessions before
the Yom Kippur close.

## The tape in one paragraph

A hawkish-looking decision session. ^TNX closed 5.006, a fresh trailing-252 max (+97.2 bp
over 252 sessions, +28.2 bp over 21), yet TLT rose +0.21% while IEF fell -0.10%: the
belly led, IEF at its 252 low with z10 -1.60 (5d rank 3.2) against TLT's 5d rank 21.8.
The dollar popped (DX-Y.NYB +0.66%, 5d rank 98.0; UUP within 0.7% of its high). Crude
reversed hard off a thrust (USO -3.52% on the day, still +19.9% in 21d, 3.52% below its
high; XLE -2.88%, SLB -3.51%, OIH -2.83%) while the refiner VLO printed a fresh 252 high
(+1.57%, z10 2.26, r63 100). Banks broke on a calm index (SPY -0.44%, DIA -1.15%): XLF
-1.62%, KRE -1.77%, GS -3.96% (-1.43 ATR), USB -4.00% (-2.35 ATR), PNC -3.86% (-2.14 ATR),
9 of 11 bank names at a 5d rank <= 20 with the median 63d rank at 29. Metals kept
leaking: GDX -6.71% over 5d against GLD -2.88%, SLV -6.04%. VIX 17.71 (+2.97%),
VIX/VIX3M 0.898, ^MOVE -3.56% on the day. Breadth is thin under a near-high index:
SPY 3.06% off its 52w high, 20.2% of the tape above its 21d median, 15 tape names within
1% of a 52w low (TJX, LOW, MCD, NKE, WHR, VFC, VMC, AON, CMS, PEG, PEP, XLU, IEF, TLT,
LQD) against 4 within 1% of a high (TMO, VLO, UUP, ^TNX). XLU sits at a 5/21/63 floor
(3.2/2.0/0.8); XLI r21 1.6 / r63 0.4; XLRE, VNQ, IYR r63 0.4.

## 1. Calendar events x asset classes

Events in window: PPI 09-10 (-5), CPI 09-11 (-4), FOMC decision 09-16 (-1), VIX expiry
09-16 (-1), opex + quad witching 09-18 (+1), NFP 10-02 (+11). Non-macro calendar anchors
inside a 10 td hold, enumerated because the macro file does not carry them: quarter-end
09-30 (+9), Yom Kippur 09-21 (+2), Japanese half-year book close 09-30 (+9).

| event | us_large | us_small | rates | credit | gold | metals | energy | dollar_fx | intl | vol |
|---|---|---|---|---|---|---|---|---|---|---|
| PPI 09-10 (past) | D1 | D1 | D1 | D1 | D1 | D1 | D1 | D1 | D1 | D1 |
| CPI 09-11 (past) | D1 | D1 | D1 | D1 | D1 | D1 | D1 | D1 | D1 | D1 |
| FOMC 09-16 (k=+1 today) | D2 | D2 | D3 | D3 | D3 | D3 | D4 | D5 | D6 | D7 |
| VIX expiry 09-16 | D8 | D8 | D9 | D9 | D9 | D9 | D9 | D9 | D9 | D8 |
| opex/quad 09-18 (+1) | D10 | D10 | D11 | D11 | D11 | D11 | D11 | D11 | D11 | D12 |
| NFP 10-02 (+11) | D13 | D13 | D14 | D13 | D13 | D13 | D13 | D13 | D13 | D13 |
| quarter-end 09-30 (+9) | **CHECK c2** | D15 | D16 | D17 | D18 | D18 | c2 leg | **CHECK c7** | **CHECK c4** | D19 |
| Yom Kippur 09-21 (+2) | **CHECK c1** | c1 row | c1 row | D20 | D20 | D20 | D20 | D20 | D20 | c1 row |

Dismissals, each with its reason:
- **D1 PPI/CPI (past).** Both prints sit behind the entry; post-release anchors needed
  09-10/09-11 closes. The pair cells (watchlist 43, 44) are parked to the next prints.
- **D2 FOMC k=+1 x us_large/small.** 09-16 measured SPY from the decision close to the end
  of week zero at placebo 11 of 11 (the cycle premium is the decision session itself) and
  midterm h=3 -0.470%. Today's entry is one rung later inside the same dead window, a
  neighbour of a killed cell with no new mechanism. The FOMC-cycle odd-week short
  (Cieslak-Morse-Vissing-Jorgensen week 1 = days 4-8) is not enterable today: its anchor
  is the 09-21 close. Not examined for timing, noted as a future anchor.
- **D3 FOMC k=+1 x rates/credit/gold/metals.** Registry 09-16: with the ten-year at its
  high the post-decision relief "does not arrive at all after the close" (TNX +4.2 bp by
  h=3, TLT -0.48% on 31 decisions); gold closed on all three rungs; HYG decision-close
  flush filters the wrong way. k=+1 inherits the wrong-signed base.
- **D4 FOMC x energy.** Crude's -3.52% decision-day reversal is the 09-14 "crude reversal
  day" kill in a new calendar slot; no policy-transmission channel distinct from the
  dollar leg.
- **D5 FOMC x dollar.** Yesterday's shipped idea (long DX from the decision close, h=5)
  is live in the fingerprint table; a k=+1 re-entry is a repeat without a changed fact.
- **D6 FOMC x international.** Reduces to dollar plus US beta (09-16 D5; pre-FOMC
  15-class family homogeneous, Q p 0.81).
- **D7 FOMC x vol.** "Post-NFP, post-FOMC and post-VIX-expiry vol cells, swept and empty";
  watchlist 46/52 arms unmet (below).
- **D8 VIX expiry x us_large/small/vol.** Settle rung era-dead (watchlist 46), run-in
  midterm-wrong-signed (watchlist 48). Both anchors are behind today's entry.
- **D9 VIX expiry x non-equity classes.** No mechanism links a VIX settlement to these
  classes; the collision's measurable proxies refuted the pin story on 09-11.
- **D10 quad witching x us_large/small.** Closed by name 09-04 ("an FOMC anchor in
  costume") and re-killed on IWM 09-11; long IWM into quad killed 09-07; second-half
  midterm September short killed 09-14 (above-200d 1-3).
- **D11 quad x non-equity classes.** An equity-derivative expiry with no delivery or roll
  channel into rates, credit, metals, energy ETFs, FX or foreign ETFs.
- **D12 quad x vol.** September inverts the post-opex crush (registry, event sleeve V4
  carve-out); the pre-expiry SVXY carry is registry-dead.
- **D13 NFP x everything but rates.** +11 td; no enterable pre-print anchor inside a
  1-10 td hold today.
- **D14 NFP x rates.** Midterm-dead (watchlist 1); parks to 2027-01.
- **D15 quarter-end x us_small.** IWM into month-end closed on the equity month-end anchor
  (registry: month-end closed on six forms); the small-cap quarter-end OVERNIGHT is
  December-parked (watchlist 32). c2 carries the cross-sectional quarter-end form.
- **D16 quarter-end x rates.** Month-end TLT is the strongest duration cell on file and
  it is dead for today twice: the index-extension last sessions stopped paying in
  2020-2026 (+3.99 bp, t 0.37), and the distance-to-low gradient puts TLT (0.21% off its
  low) in the -0.581% bucket. Quarter-end TLT vs SPY killed 09-14 (+0.218% against
  +0.233% at ordinary month-ends).
- **D17 quarter-end x credit.** Credit month-end sits inside the six closed month-end
  forms; no quarter-specific balance-sheet channel for HYG/LQD beyond the dollar funding
  story, which c7 tests where it is sharpest.
- **D18 quarter-end x gold/metals.** No quarter-end flow story for bullion ETFs; the
  metals month-end anchor is closed (XME h=5 Sidak p 0.2374).
- **D19 quarter-end x vol.** No mechanism; the quarter's last session is not an expiry.
- **D20 Yom Kippur x credit/gold/metals/energy/dollar/intl.** The holiday-participation
  story (Frieder and Subrahmanyam 2004) is about US equity desks; c1 carries SPY and
  rows for IWM, TLT and ^VIX so the null classes are visible, the rest are not examined
  for lack of a channel.

## 2. Tape extremes by class

| class | extremes (2026-09-16) | verdict |
|---|---|---|
| us_large | SPY 3.06% off its high with 15 tape names within 1% of a 52w low vs 4 at a high; 20.2% of the tape above its 21d median; XLU triple floor; XLI r63 0.4; XLRE/VNQ/IYR r63 0.4; banks 9 of 11 at r5 <= 20 (median r63 29); GS -1.43 ATR, USB -2.35, PNC -2.14 intraday-led on a -0.44% index; META z10 2.47, TMO at a high; DIA -1.15% vs SPY -0.44% | **CHECK c3** (new-LOW breadth under a near-high index, never measured: the registry holds new-HIGH breadth and narrow-leadership only). DISMISS the bank break: the bank-breadth washout's industry label carries no information (Cochran Q two samples) and today's form is the broken-trend half (median r63 29 vs >= 70); single-bank shocks closed 09-15 both directions. XLU/XLI floors dead in the live regime (09-16). DIA-SPY gap reduces to GS's single-name shock. META/TMO single-name momentum has no event or flow story. |
| us_small | IWM r21 2.4, r63 2.4, z10 -0.91 | DISMISS: IWM-vs-SPY floor pair closed both directions (09-11, 09-14); outright IWM floor is the washout family, wrong-signed above SPY's 200d in midterms. |
| rates | ^TNX 252 max 5.006; IEF at its 252 low, z10 -1.60, r5 3.2 while TLT r5 21.8 (belly-led five days); TLT 0.21%, LQD 0.16% off lows | **CHECK c8** (belly-led five-day selloff as a curve relative-value state, a construction the curve-dose kill never conditioned on). Outright duration at the low is dead four ways (IG-complex floor 09-11, month-end gradient, post-FOMC 09-16, TNX 252 high 09-09). |
| credit | HYG z10 -1.55 (both conventions), r5 7.5, 1.34% off its high | DISMISS: watchlist 55 arm needs z10 <= -2; duration-driven flush dead 09-14. |
| gold / miners | GDX -6.71% over 5d vs GLD -2.88% (miner underperformance -3.83pp), GLD r5 18.7, GDX r5 16.7 | **CHECK c5** (downside miner/metal ratio break; the registry's ratio cells are all the UPSIDE thrust). |
| other metals | SLV -6.04% 5d, -0.83% day, 46% off its high | DISMISS: complex-break arm needs an SLV break <= -4% on the day (watchlist 30); drawdown cells killed 09-01/02/11. |
| energy | USO -3.52% day after +19.9% 21d; XLE -2.88%, SLB -3.51%, OIH -2.83%; VLO +1.57% to a fresh 252 high (z10 2.26, r63 100) | **CHECK c6** (refiner at a high on a crude reversal day, the crack-spread relative value; energy's nine dead expressions are all outright). Outright crude/XLE dismissed (09-14 reversal-day kill, placebo 16-for-16). |
| dollar_fx | DX-Y.NYB r5 98.0 (+0.66% day), r21 60.3; UUP 0.7% off its high | Price state is yesterday's pitch (repetition). **CHECK c7** on the quarter-end anchor instead, which is a different object (funding demand, not policy). |
| international | EEM r5 8.3, r63 1.2; EWJ 1.57% off its high; FXI -1.40%, z10 -1.24; EWZ r21 82.9 | **CHECK c4** (Japan half-year-end dividend-reinvestment flow on EWJ, calendar-anchored). EEM floor killed 09-16; FXI break-in-thrust arm unmet (r21 29.0 vs >= 80); EWZ residual killed 09-11. |
| volatility | VIX 17.71 (+2.97%), VIX/VIX3M 0.898, VIX 21d rank 70.2, range compression ON; ^MOVE -3.56% day, 87.3 level pctile; SVXY 2.18% off its high | DISMISS: compression-into-print cells blocked on the dial (82.9 vs <= 68 / < 50); post-FOMC vol swept empty; September pre/post quad vol inverted. ^VIX appears as a c1 row. |

## 3. Seasonal and cycle cells

- **Midterm year** conditions everything; every checker reports a midterm split.
- **September, tdom 12, second half.** Midterm second-half short killed 09-14; the event
  sleeve's T3 (post-quad IWM short) owns the post-opex window and its entry is 09-18.
- **Quarter-end (09-30, +9).** Two flows with pre-specified literature and no registry
  entry: window dressing / loser selling into quarter-end (Lakonishok et al 1991; He, Ng
  and Wang 2004), sharper in September because US mutual funds close their tax year on
  10-31 -> **c2**; and Japanese half-year-end dividend-reinvestment demand -> **c4**.
  Quarter-end dollar funding demand (Du, Tepper and Verdelhan 2018 CIP spikes) -> **c7**.
- **Yom Kippur (09-21, +2).** "Sell Rosh Hashanah, buy Yom Kippur", a named adage with a
  published volume/return regularity (Frieder and Subrahmanyam 2004); zero registry
  entries -> **c1**.
- **Seasonal board** (asof 2026-08-05, stale): zero A/B setups; regime rows are book
  context; its P/C complacency row (08-04) is no longer live (equity P/C 52.0 pctile).
- **NG=F September** (watchlist 47): roll wedge, dismissed.

## 4. Watchlist verdicts (55 active, 0 expired)

1. TLT from the NFP close: PASS. Midterm-dead; next NFP 10-02; parks to 2027-01.
2. LQD vs HYG at joint 52w extremes: PASS. HYG 1.34% below its high vs within 0.5%.
3. SVXY overnight into CPI: PASS. Next CPI 10-14 (+19).
4. GLD on a miner-led thrust: PASS. GDX r5 16.7 vs >= 95 (the opposite extreme is c5).
5. XLE on a crude thrust in the 5-6% band: PASS. USO -3.52% on 09-16.
6. TLT with the IG complex at 52w lows: PASS. State live again (IEF 0.00%, LQD 0.16%, TLT 0.21% off lows) but the arm is a filter-vs-reanchor construction test that one session cannot move.
7. SPY on a skew spike: PASS. SKEW r5 23.4 vs >= 95.
8. Fade a crude thrust out of a deep base: PASS. USO r63 74.6 vs <= 20.
9. IHI thrust: PASS. IHI r21 17.5 vs 100.
10. FXI break inside a thrust: PASS. FXI r21 29.0 vs >= 80.
11. TLT November month-position: PASS. Date-parked.
12. Short SPY at a 52w high with TLT at a low: PASS. TLT 0.21% (live) but SPY 3.06% off its high vs within 0.5%.
13. SPY on a vol pop in calm tape: PASS. VIX +2.97% (< 5%) and 21d rank 70.2 vs <= 25.
14. Gold on an unconfirmed rate rise: PASS. DX r21 60.3 vs <= 15.
15. Tech vs healthcare after a rotation gap: PASS. XLV minus XLK -0.03pp vs >= 3.0pp.
16. Short the dollar on an unconfirmed rate rise: PASS. DX r21 60.3 vs <= 20.
17. Short TLT after a big up day near the low: PASS. TLT +0.21% vs >= +1.5%.
18. Short KRE vs XLF on a bank-breadth washout: PASS. Breadth live (9 of 11 = 81.8% at r5 <= 20) but median 63d rank 29.0 vs the intact >= 70, and the arm is an ex-crisis cost threshold.
19. IEF vs 0.523 TLT curve at a yield high: PASS. Out of sample only since 09-14. c8 is a different state (the five-day belly-led move, not the yield level) and owes a reconciliation with this kill.
20. Narrow energy thrust count: PASS. Count 0 (max VLO 1.20, USO 0.68 under pitch_lab.zscore).
21. Survivorship-free new-high breadth: PASS. Only XLE-adjacent names near highs; wrong extreme. c3 is the new-LOW mirror.
22. Sector washout into a 52w high: PASS. XLU is the only SPDR at r5 <= 5 and sits 12.27% off its high.
23. Utilities washout with TLT hit: PASS. XLU r21 1.98 clears; TLT r21 56.75 vs < 25.
24. Bare dollar washout: PASS. Date-parked; DX r21 60.3.
25. HYG 52w high while the index is not: PASS. HYG 1.34% off.
26. SMH deep correction family: PASS. Out of sample only.
27. Rates repricing with zero credit stress: PASS. HYG 1.34% vs within 0.25%.
28. IEF out of Jackson Hole: PASS. Date-parked.
29. The laggard still falling, pooled: PASS. No ETF with r21 >= 90 (USO 84.9 top).
30. Short silver after a complex break: PASS. SLV -0.83% on the day vs a <= -4% break; GDX -1.41%, GLD -0.61%.
31. Long duration at a yield high with MOVE mid-range: PASS. MOVE 87.3 level pctile vs [40,50).
32. IWM December month-end overnight: PASS. Date-parked.
33. Energy at a fresh 52w high on an index-down session: PASS. XLE 2.88% below its high.
34. SVXY into a print from a (5,15] VIX range: PASS. Next print anchor NFP 10-02; dial 82.9 vs <= 68.
35. Pooled sector triple floor: PASS. XLU holds the floor but the arm is SPY below its 200d; SPY +5.63% above.
36. SPY into a print out of a dead VIX range: PASS. Dial 82.9 vs < 50.
37. Risk premium across an extended closure: PASS. Next is Thanksgiving.
38. Post-NFP duration after a moderate miss: PASS. Next NFP 10-02.
39. SVXY after a closure: PASS. No closure.
40. SPY vs IWM with the dial in 56-70: PASS. Dial 82.9.
41. HYG after a closure: PASS. No closure.
42. Short IEF with commodities at a high and a print inside: PASS. DBC 1.54% off its high; no inflation print inside h=5.
43. SPY across the September PPI-CPI pair: PASS. Pair was 09-10/11.
44. TLT from the PPI release close: PASS. Next PPI 10-15.
45. SPY with HYG at a high as TNX prints one: PASS. TNX clears; HYG 1.34% off.
46. SVXY on a VIX expiry x FOMC settle session: PASS. The 09-16 collision is one of the four post-2018 collisions it needs; it is recorded, not traded.
47. NG=F September: PASS. Mechanism arm unmet.
48. SPY run-in on an FOMC x VIX expiry collision, non-midterm: PASS. Midterm; parked.
49. Deep 5-day flush inside a top-decile 63d trend: PASS. No ETF at r5 <= 2 with r63 >= 90.
50. XLV vs 0.71 SPY after a healthcare flush: PASS. XLV r5 56.7 vs <= 1.
51. Hedged short SVXY after a 10% VIX crush: PASS. VIX +2.97%.
52. SVXY into FOMC after a re-bid, backwardated: PASS. Next anchors 10-26 and 12-07.
53. Short a large bank vs XLF after a 1.5 ATR intraday slide: PASS. Of its eight names only GS moved (-1.43 ATR, below 1.5); USB (-2.35) and PNC (-2.14) are outside its universe, and the arm (banks' max-of-12 P <= 0.10 AND 2018+ >= +0.25%) is not moved by one session.
54. Long TLT across the FOMC announcement session: PASS. Parked to the 10-27 eve close.
55. Long HYG after a spread-driven flush: PASS. HYG z10 -1.55 vs <= -2.

## 5. Scoreboard read

Five graded ideas lifetime (3 B at +0.448 avgR, 2 C at -0.237; event_fingerprint 2 at
+0.622, inversion 1 at -0.62, interaction_cell 1 at +0.146, relative_value 1 at +0.099).
A handful; no axis earns or loses a slot on it. Two of today's candidates (c1, c2 via c4
and c7) sit on calendar anchors the registry has never opened, which is where the
event_fingerprint pair earned.

## 6. Candidates selected from this map (8)

| id | candidate | anchor | class | axis | registry / watchlist adjacency |
|---|---|---|---|---|---|
| c1 | Short SPY from the close two sessions before Yom Kippur to the Yom Kippur close (the tail of the Rosh Hashanah to Yom Kippur window), rows for the full window and for IWM / TLT / ^VIX | event (Hebrew calendar) | us_large | event_fingerprint | Pre-specified adage + Frieder and Subrahmanyam 2004; zero registry entries. Dates must be COMPUTED (Gauss Passover + 163 days), not typed. Must separate from September's own second-half drift (tdom-matched control) and from quad witching, which falls inside the window in several years. |
| c2 | Quarter-end window dressing: long the two best 63d SPDRs (XLE, XLV) against the two worst (XLU, XLY) from QE-9 to the quarter's last close; mechanism test on the liquid single-stock cross-section; September vs other quarter-ends | event (quarter-end) | us_large sectors (energy leg) | flow_mechanics | Registry: month-end anchor closed on six forms, all own-instrument close-to-close, none cross-sectional; 09-16 floor-vs-XLE pair was the generic REVERSAL pair and lost in the live regime. Must beat the same spread at non-quarter month-ends. |
| c3 | New-LOW breadth under a near-high index: share of a fixed liquid universe within 1% of a 52w low at a trailing-252 percentile extreme while SPY is within 5% of its high, SPY h=1..10 | price state | us_large | interaction_cell | Mirror of watchlist 21 (new-high breadth near-miss) and the 09-14 narrow-leadership kill. Survivorship: fixed today's-universe lists undercount historical lows; rank the share against its own trailing distribution, and repeat on a sector/industry ETF universe. |
| c4 | Long EWJ from QE-9 to the March and September quarter-end closes (Japanese half-year and year-end dividend-reinvestment demand), control June/December | event (quarter-end) | international | flow_mechanics | Registry Japan entries are the washout cell (price state). Currency must be separated (EWJ is USD; use DXJ if cached, else report yen-leg correlation). |
| c5 | Long GDX against beta-GLD after a DOWNSIDE miner/metal ratio break (GDX 5d minus GLD 5d at a trailing-252 low), h=1..10 | price state | gold / miners | relative_value | Registry ratio cells are the upside thrust (killed); watchlist 4 is upside. |
| c6 | Short VLO (refiners) against XLE when the refiner prints a 252 high on a session crude falls >= 3% | price state | energy | relative_value | Nine outright energy expressions dead; crack-spread relative never measured. Must survive the generic "short the day's best energy name vs XLE" placebo. |
| c7 | Long the dollar (DX-Y.NYB / DX futures) from QE-9 to the quarter's last close, quarter-ends vs ordinary month-ends, with the dollar already at a 5d rank >= 95 as a row | event (quarter-end) | dollar_fx | flow_mechanics | Registry "month-end on FX" closed; this must show quarter-end separates from month-end or it is that kill. Yesterday's long-DX pitch is live, so a same-direction ship needs changed_since. |
| c8 | Long IEF against 0.523 TLT after a belly-led five-day selloff (IEF 5d rank <= 5 while TLT 5d rank >= 20), h=1..10 | price state | rates | relative_value | Watchlist 19 (curve dose at the yield level) killed 09-14 out of sample only; this conditions on the five-day curve move, not the level, and must show it filters rather than re-anchors that parent. |

Coverage: 8 candidates, 6 asset classes (us_large, international, gold_miners, energy,
dollar_fx, rates), 4 axes (event_fingerprint, flow_mechanics, interaction_cell,
relative_value), 4 event-anchored (c1, c2, c4, c7) and 4 price-state (c3, c5, c6, c8).
