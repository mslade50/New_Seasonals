# Surface map, 2026-09-25 (Friday), tape through the 2026-09-24 close

State: pipeline 7/7 green, prices bar 2026-09-24, dial ma10-63d 80.9 (21d ago 88.9),
raw 21d 44.8, P/C fear OFF (36th pctile), signals on: NYSE Net Highs only (5d EMA -271,
raw -408, SPY 1.13% off its 252 close high). VIX Range Compression is OFF today
(needs a negative 5d VIX change, has +0.23). Midterm year, QE-3 (09-30 is three
sessions out), Q4 turn 10-01. Warnings: LEG stale; nothing else. Book: 3 staged
(OVS short META on Order_Staging; OVS short GME and LT Trend ST OS long FIVE on
Overflow). Event sleeve flat. Trend sleeve CASH (dial gate). Exposure leg 0.0x (Rule 3).
Seasonal board meta is stale (2026-08-05); it carries only regime context.

Scoreboard read: 6 graded ideas lifetime (B +0.644R on 4 at a 100% hit, C -0.237R on 2;
event_fingerprint +0.622R on 2, inversion +0.307R on 2, interaction_cell +0.146R on 1,
relative_value +0.099R on 1). Still a handful, so no axis is starved or promoted on it.
Grade C is the losing grade so far, noted for compose. The 09-23 UNG pitch (B) is in
its hold (exits at today's close) and ungraded.

Extra series outside the tape (copper, curve, FX, softs, indices) were pulled by
`_extra_surface.py`; output in `_extra_surface_out.txt`. Watchlist arm values by
`zz_watchlist_values.py` (output `zz_watchlist_values_out.txt`).

## What moved (the tape in one paragraph)

09-24 was a second leg of the rates shock on a flat index. ^TNX +4.8 bp to 5.162% (252
high; r21/r63 100; +104 bp over 252 sessions); ^FVX 5.02% and ^IRX 4.07% both at 252
highs, 10y-3m at 1.094 (99.6th pctile, steepest of the year) while 10y-5y sits at the
1.6th pctile (belly-led). TLT -1.29% on 1.94x volume, IEF -0.55%, LQD -0.71%, TIP
-0.49%: all four exactly at 252 lows. ^MOVE +9.57% to 104.58 (97.4th pctile of daily
moves; level at the 98.4th pctile of its trailing year) while ^VIX is 15.67 (19th pctile
of its year). Dollar: UUP at its 252 high (z10 +2.46), EURUSD z10 -2.71 (0.24% above its
low), GBP z10 -2.59, USDMXN +1.48% (r5 98.4), USDBRL +1.26%, AUDJPY r21 1.6: a carry-
unwind day. Metals: GC=F -7.64% 21d, SI=F -44.9% off its high, but HG=F copper sits
1.25% under its 252 high, so copper/gold is at the 96.8th level pctile (+10.3% in 21d).
COPX -10.45% 21d against copper +1.88%. Energy: CL=F +2.66% (94.61), USO +2.86%; NG=F
+9.06% and UNG +6.26% on 5.55x its 63d volume, the second 5%/3x thrust in three
sessions (09-22 +5.85% on 4.29x). Equities: SPY -0.08% (1.13% off its high), QQQ +3.48%
in 5d, SMH +7.12% in 5d, MU +10.54% in 5d into its 09-30 print, IWM r63 0.4, only 18.3%
of the tape above a median 21d rank. Utilities at 52w lows (XLU z10 -2.46; DTE -3.37,
CMS -3.27), IYT r63 0.4, XHB/ITB 2-3% above 52w lows. Softs: KC=F coffee -25.9% in 21d.

## 1. Calendar events x asset classes

Live events in [-5, +15]: opex/quad 09-18 (k=+5), NFP 10-02 (k=-5), CPI 10-14 (k=-13),
PPI 10-15 (k=-14), opex 10-16 (k=-15). Not in the events file but live: quarter-end
09-30 (QE-3), Q4 turn 10-01, US fiscal year-end 09-30, China Mid-Autumn (09-25) and
Golden Week (mainland closed 10-01..10-07, Stock Connect suspended), NG October
contract expiry 09-28, earnings MU and CAG 09-30, NKE 10-01, PEP 10-08. FOMC 10-28
(k=-23) and the election 11-03 are outside every horizon.

### Opex / quad witching 09-18 (k=+5)
- All ten classes DISMISSED. Post-opex ladders that paid lived at k=0..+2 (registry);
  the vanna short, the September post-opex SVXY short and the EFA expiry lag all closed
  09-18/09-21; W56 parks to December. k=+5 has no stated channel into rates, credit,
  gold, metals, energy, dollar or intl.
### NFP 10-02 (k=-5)
- rates: DISMISSED. The TLT run-in with ^TNX near its high was killed yesterday on the
  full k=-10..-1 ladder (-0.572% on 5-9, every gated rung negative, October prints
  -0.324%). W0 midterm-parked, W36 fails its CPI-in-hold leg.
- us_large / gold / dollar: DISMISSED. The seven-session run-in is closed on four
  vehicles (registry "The run into NFP entered seven sessions early": plateau ladders,
  September-midterm -0.676% on 3-3). k=-5 is inside that dead ladder.
- volatility: W32/W34 anchor at k=-2 (09-30), not today, and both fail on rel-range
  21.03 and dial 80.9 (see section 4).
- credit / metals / energy / intl / us_small: DISMISSED, no mechanism enters at k=-5.
  NFP sits inside the hold of every h>=5 candidate below and is carried as tail risk.
### CPI 10-14, PPI 10-15, opex 10-16 (k=-13..-15)
- all ten classes: DISMISSED, outside a 10-td hold entered today. W2 / W42 park to dates.
### Quarter-end 09-30 (QE-3) and the Q4 turn 10-01
- The month-end anchor is closed on equities, rates, FX and commodities (registry "The
  month turn on commodities and metals ... closed on equities, rates, FX and
  commodities"), and the September quarter specifically on the dollar (W54, arm off),
  sector window dressing (W55 enters only at the QE close), single stocks (09-21), LQD
  issuance (September is the losing quarter), pension TLT/SPY (09-14), yen (09-21), EEM
  funding (09-23), SPY quarter-turn straddle (09-23). us_large, us_small, rates, credit,
  gold, metals, energy, dollar, intl: DISMISSED on those entries.
- volatility: NOT in that list. The month/quarter turn has never been measured on SVXY
  or the VIX complex in this repo (zero registry hits for quarter-end x SVXY/VIX).
  -> CANDIDATE Q1.
### US fiscal year-end 09-30
- all classes DISMISSED: no shutdown-probability series in the repo (09-24 verdict stands).
### China Mid-Autumn (09-25) and Golden Week (10-01..10-07)
- intl: the 09-18 FXI SHORT died because its mechanism was falsified in-window: across
  the closure FXI rises 16-5 (+1.178%) and the run-in residual against EEM is +1.601%
  (6-5), the wrong sign for pre-holiday de-risking. The long flip has never been checked.
  -> CANDIDATE I1 (inversion, charged as a flip).
- all other classes: DISMISSED, no channel.
### NG October contract expiry 09-28
- energy: DISMISSED as a standalone anchor. UNG already holds November; the expiring
  front is not a vehicle, and the NG=F continuous series will carry a roll seam on
  09-29 (memory: futures continuous roll gaps). Carried as a data caveat on N1.
### Earnings: MU and CAG 09-30, NKE 10-01, PEP 10-08
- MU: semis V-turn (r63 1.98, r5 73, +10.5% in 5d) into the print. The registry's
  pre-print lanes are winner-into-print (gate subtracts, 09-21), laggard-into-print (it
  is the laggard reversal, 09-21) and the pre-print washout (closed). A 63d-floor name
  that has already turned (r63 <= 5 AND r5 >= 70) into its own print is none of those.
  -> CANDIDATE E1 (event_fingerprint).
- CAG (GIS form, W59 fails on SPY above its 200d), NKE (laggard lane, 09-21), PEP
  (laggard lane): DISMISSED on those numbers.

## 2. Tape extremes by class

- us_large: SPY -1.13% off its high, QQQ -0.85%, narrow tape (18.3% above a median r21).
  The breadth-divergence short family is dead three ways (new-low breadth 09-17, NYSE
  net highs 09-23, HYG-weak 09-24); IYT at a 63d floor under a near-high index is the
  same family, DISMISSED. META at its 52w high is the book's staged OVS short: a pitch
  long contradicts the book, a pitch short duplicates it, DISMISSED. TMO/AMGN single-name
  thrusts: no mechanism, DISMISSED. Equity content enters through V1 (the SPY leg) and E1.
- us_large sectors: utilities (XLU 52w low, W61 OOS-only, W33 needs SPY below its 200d),
  DTE/CMS z10 -3.3 (same-state regulatory news in no series here), staples and REITs
  (rate-sensitive family closed), banks (9 of 11 at r5 <= 20 but median r63 9.9, the
  broken form; bear-steepener XLF long killed 09-21), homebuilders (ITB long killed
  09-18), transports: all DISMISSED on those entries. HPQ and VLO sit in the W47 corner
  (r5 <= 2, r63 >= 90) but W47 is an ETF-family entry that owes a prereg: DISMISSED.
- us_small: IWM r63 0.4, z10 -0.74. Size spread closed (W38 needs dial 56-70; U1 killed
  09-24). Midterm-October short considered and DISMISSED: the registry's midterm
  September short above the 200d goes 1-3 at -0.73% with a month-by-cycle permutation
  of 0.843, and IWM is 2.56% above its 200d.
- rates: every price state here is closed or parked (W5 construction test, W11, W16,
  W18 OOS in progress, W29 band, W42, W52, W65 MOVE leg, whole-percent yield 09-21,
  belly-led IEF/TLT curve 09-17, TIP/IEF breakevens pair, month-end TLT decayed, TLT
  high-volume capitulation 09-24). NEW: copper/gold, the classic growth-inflation
  confirmation of a yield move, has never been used as a rates conditioner here (the
  copper entries are a thrust on FCX). Today it is at the 96.8th level pctile with ^TNX
  at its 252 high. -> CANDIDATE C1.
- credit: HYG 1.34% above its 52w low on 2.2x volume, z10 -1.25; its move matches a
  rates-plus-equity replication (IEF -0.55%, SPY -0.08%), so there is no residual
  dislocation to trade. W1/W23/W25/W43 need HYG at a HIGH; W53 needs a spread-driven
  flush (IEF r5 0.4, duration-driven again). DISMISSED.
- gold / miners: GLD -8.5% in 21d, -21% off its high, below its 200d. Short GLD on a
  confirmed rate rise killed 09-24; W13 is the unconfirmed pole; GDX/GLD pairs closed
  both directions. DISMISSED as a standalone; gold enters through C1's ratio.
- other metals: SLV -45% off its high (the crash is old: SI/GC ratio at the 41.7th
  pctile, no fresh state), W28 needs a first complex break (none today). Copper is the
  live metal: HG=F within 1.25% of its 252 high while COPX fell 10.45% in 21 sessions.
  Miner-vs-metal has only been opened on gold. -> CANDIDATE M1.
- energy: UNG +6.26% on 5.55x volume is the 09-23 pitch's own trigger firing again
  inside that pitch's hold. The rule's re-firings have never been separated from first
  firings. -> CANDIDATE N1 (changed_since owed). CL=F: W58 parks to 10-19, W63 state off,
  W4/W7/W19 fail, XLE-vs-USO divergence closed (b5), crude-through-dollar closed 09-24.
- dollar / FX: DX breakout pole closed 09-24 (D1), W22 parked, W54 off, yen half-year
  closed 09-21, USDJPY 158.3 printed 163.9 inside its own trailing year so a 160
  "intervention line" is falsified in-window. NEW: the carry side. USDMXN +1.48% (r5
  98.4) and USDBRL +1.26% on a MOVE spike is a carry-unwind session, and the carry
  currencies have never been opened here. -> CANDIDATE X1.
- international: EEM breakout short closed (09-24), EFA expiry lag closed, EWJ dividend
  window closed, FXI break (W9) fails. ^KS11 +6.85% in 5d rides the memory names (no
  EWY series). Golden Week long -> I1 above.
- volatility: ^MOVE at the 98.4th pctile of its year while ^VIX sits at the 19.0th. The
  registry's MOVE/VIX kill (b4_c11) died because its premise was false (MOVE was at the
  46th pctile; the ratio was high on a cheap denominator). Today the premise is true on
  both levels. -> CANDIDATE V1. The one-day MOVE spike itself is W64 (fails both legs).
- softs / livestock / grains: KC=F coffee -25.9% in 21d (z10 -2.06), HE=F 1.8% above its
  low, grains at 63d highs. No repo precedent, the moves are harvest and weather
  information rather than forced flow, and the continuous series carry roll seams.
  DISMISSED.

## 3. Seasonal and cycle cells

- Midterm year, late September into October. "Long SPY from late September into a
  midterm election" (09-21) and "Short SPY and IWM over the second half of a midterm
  September" are both closed; midterm conditions every candidate's era split below.
- Seasonal board is stale (2026-08-05, 0 A+B setups); no live board candidate.
- W58 (CL=F October tdom-13) parks to 10-19, W10 (November TLT) to November, W45
  (September NG front) closes 09-30 and its arm is a non-roll form it does not have.

## 4. Watchlist verdicts (66 active, 0 expired)

Tape through the 2026-09-24 close. Values from zz_watchlist_values_out.txt and
_extra_surface_out.txt. Near expiries: W55 10-01, W48 and W49 10-05, W51 10-06,
W53 10-07, W54 10-08, W63 10-14, W58 10-20.

- W0 NFP x TLT: PASS, midterm-parked to 2027-01; the 10-02 print is a midterm print.
- W1 LQD/HYG divergence: PASS, HYG -2.01% off its high vs <= 0.5% (LQD 0.00% above its
  low clears); the arm is an episode count, unchanged.
- W2 CPI overnight SVXY: PASS, CPI 10-14.
- W3 GLD on a miner thrust: PASS, GDX r5 25.8 vs >= 95.
- W4 XLE on a crude 5-6% day: PASS, USO +2.86% (0.83 ATR) vs [5,6)%.
- W5 TLT with IG pinned at lows: PASS. State live again (TLT, IEF, LQD 0.00% above
  their lows, 09-23 and 09-24) but the first-in-10 anchor is still 09-10; the arm is
  the construction test (+0.043pp vs >= 0.35pp), unchanged.
- W6 SPY on a skew spike: PASS, ^SKEW r5 48.0 vs >= 95.
- W7 fade a crude thrust from a deep base: PASS, USO r5 34.9 vs >= 90, r63 75.0 vs <= 20.
- W8 IHI thrust: PASS, IHI r21 11.1 vs 100.
- W9 FXI break inside a thrust: PASS, FXI r5 53.2 vs <= 20, r21 25.4 vs >= 80.
- W10 November TLT: PASS, parks to November tdom 4-12.
- W11 short SPY at a high with TLT at a low: PASS, TLT 0.00% above its low clears but
  SPY -1.13% off its high vs <= 0.5%; the arm is cost on the de-concentrated form.
- W12 SPY on a vol pop in calm tape: PASS, VIX +3.23% vs >= +5%, VIX r21 56.3 vs <= 25.
- W13 gold on an unconfirmed rate rise: PASS, DX r21 91.7 vs <= 15.
- W14 XLK vs XLV rotation gap: PASS, XLV-XLK +0.96pp vs >= +3.0pp.
- W15 short dollar on an unconfirmed rate rise: PASS, ^TNX r21 100 clears, DX r21 91.7
  fails <= 20.
- W16 short TLT after a big up day near the low: PASS, TLT -1.29% vs >= +1.5%.
- W17 KRE vs XLF on bank breadth: PASS, breadth 9 of 11 (82%) clears but median r63 9.9
  is the broken form; the arm is the ex-crisis cost bar, a statistic.
- W18 IEF vs TLT curve: PASS, OOS episode 2 (signal 09-23, entry 09-24, exit 10-06) is
  in its hold; 09-24 sits inside the gap-10 window; episode 1 realized -65.0 bps.
- W19 narrow energy thrust: PASS, zero of 11 names at z10 >= 2 (max VLO -0.10 tape).
- W20 new-high breadth: PASS, SPY -1.13% off its high vs > 2.0%.
- W21 sector washout near a high: PASS, the only SPDR at r5 <= 5 is XLU (1.6) at -16.43%
  off its high vs within 5%.
- W22 bare dollar washout: PASS, midterm-parked to 2027; DX r21 91.7 is the other pole.
- W23 HYG fresh 52w high: PASS, HYG -2.01% off its high.
- W24 semis 63d floor: PASS, OOS only; SMH r63 6.0 with r5 87.3 is the bounced side.
- W25 rates repricing with no credit stress: PASS, HYG -2.01% vs within 0.25% of a high.
- W26 post-Jackson Hole IEF: PASS, 2027-08.
- W27 laggard still falling: PASS, no holder of r21 >= 90 AND r63 <= 10 in the 29-ETF
  family or the tape.
- W28 short silver after a complex break: PASS, no break (GLD -0.30%, SLV -0.93%, GDX
  -1.29% vs each <= -2%); OOS first-break tally 0 of 10.
- W29 TLT with MOVE mid-range: PASS, ^MOVE level pctile 98.4 vs [40,50).
- W30 December small-cap overnight: PASS, 2027-12.
- W31 XLE 52w high on a down-SPY day: PASS, XLE -4.49% off its high.
- W32 SVXY into a print out of compression: PASS, rel-range 21.03 vs (5.0, 8.5]; dial
  80.9 vs <= 68.0; next anchor NFP k=-2 = 09-30.
- W33 pooled triple floor: PASS, SPY +7.18% above its 200d vs below.
- W34 SPY into a print out of a dead range: PASS, rel-range 21.03 vs <= 15, dial 80.9
  vs < 50.
- W35 closure risk premium: PASS, no >= 3-day closure before 11-26.
- W36 post-NFP duration after a moderate miss: PASS, the 10-02 h=3 hold ends 10-07,
  CPI 10-14 is outside.
- W37 SVXY after a closure: PASS, no closure before 11-26.
- W38 SPY vs IWM in the dial band: PASS, dial 80.9 vs [56,70).
- W39 HYG after a closure: PASS, no closure before 11-26.
- W40 short IEF with commodities at a high: PASS, DBC -1.48% off its high vs within
  0.25%; no print inside an h=5 hold.
- W41 September PPI-CPI pair: PASS, 2027.
- W42 TLT from the PPI close: PASS, PPI 10-15.
- W43 SPY with HYG at a high on a TNX high: PASS, HYG -2.01% and SPY -1.13% fail.
- W44 SVXY on a VIX-expiry x FOMC collision: PASS, 10-21 expiry has no FOMC.
- W45 NG=F September seasonal: PASS, window closes 09-30; NG=F +9.06% today but the arm
  is a non-roll form plus a mechanism, and the front contract expires 09-28 (seam).
- W46 FOMC x VIX-expiry run-in: PASS, 2027-03-17.
- W47 flush-inside-strength corner: PASS, XLU r5 1.6 at r63 0.4 is the wrong corner;
  HPQ (1.2, 90.9) and VLO (2.0, 94.8) are single names outside the ETF family.
- W48 XLV family flush: PASS, XLV r5 62.3 vs <= 1. Expires 10-05.
- W49 hedged short SVXY after a 10% crush: PASS, ^VIX +3.23%. Expires 10-05.
- W50 SVXY into FOMC after a re-bid: PASS, next k=-2 is 10-26; VIX/VIX3M 0.850 vs >= 0.90.
- W51 short a bank vs XLF after a slide: PASS, largest GS -0.48 ATR vs 1.5. Expires 10-06.
- W52 TLT across the FOMC announcement: PASS, anchor 10-27.
- W53 HYG spread-driven flush: PASS, HYG z10 -1.25 vs <= -2 and IEF r5 0.4 (flushed
  alongside) fails. Expires 10-07.
- W54 dollar into the September QE: PASS, arm measured off 09-21 (+0.197pp vs +0.25pp).
- W55 QE sector reversal pair: PASS today, enterable only at the 09-30 close, CHECK owed
  that morning. Preview on 63d: top two XLE +16.42%, XLV +9.57%; bottom two XLU -13.52%,
  XLI -8.06%. Expires 10-01.
- W56 IWM from a quad close after a washout: PASS, December.
- W57 SPY from opex after a VIX crush: PASS, 10-16 opex is midterm; 2027-01-15.
- W58 CL=F October tdom-13 short: PASS, 10-19 close.
- W59 GIS-form pre-print: PASS, SPY above its 200d.
- W60 LQD vs IEF December QE-7: PASS, 12-21.
- W61 utilities washout as a delay rule: PASS, joint state live (XLU r21 0.79, TLT r21
  1.59) but inside the 09-18 cluster at gap 21; OOS count 1 of 5.
- W62 EEM vs SPY December QE-5: PASS, 12-23.
- W63 crude round-trip long: PASS, USO state off (r5 34.9 vs <= 3), CL=F never entered.
- W64 SVXY after a sub-tail MOVE spike: PASS, ^MOVE +9.57% is the 97.4th pctile of daily
  moves vs the [90,97] band, and SPY -0.08% vs a fall of more than 0.75%.
- W65 TLT after a high-volume down day into a fresh low: PASS, TLT -1.29% on 1.94x at a
  fresh 252 low clears the first three legs, but ^MOVE +9.57% vs < +8.7% fails.

## 5. Candidates selected (8), with axis and the cell they came from

Signs are written here BEFORE any check runs. A result on the opposite sign is a post-hoc
flip and owes the registry's charge.

| id | candidate | class | axis | anchor | pre-specified sign |
|---|---|---|---|---|---|
| N1 | Long UNG h=2 from today's close: the 09-23 pitch's rule (>= +5% on >= 3x 63d volume) FIRING AGAIN two sessions after its first firing; hold Mon-Tue carries no Thursday EIA report | energy | flow_mechanics | price state (re-fire) | long (continuation) |
| V1 | Rate vol at a one-year extreme under calm equity vol: ^MOVE level pctile >= 95 AND ^VIX level pctile <= 30 (trailing 252), equity vol catches up; SVXY residual vs 1.48 SPY and the SPY leg, h=1..10 | volatility -> us_large | interaction_cell | price state | long equity vol (short SVXY residual), SPY down |
| Q1 | Long SVXY across the quarter turn, QE-3 close to the QE+1 close (exits before NFP), beta-charged against SPY | volatility | event_fingerprint | calendar (QE-3 today) | long SVXY residual |
| C1 | Copper/gold ratio at its 252 high zone (21d rank >= 90) with ^TNX at a 252 high: the yield move is confirmed by growth/inflation, short TLT h=3..10 continuation | rates (gold, other metals as conditioner) | interaction_cell | price state | short TLT (continuation) |
| M1 | Long COPX against beta-HG=F after copper miners lag the metal by >= 8pp over 21 sessions with copper within 3% of its 252 high (FCX as the long-history proxy) | other_metals | relative_value | price state | long miner residual (catch-up) |
| X1 | Long the peso (short USDMXN, 6M futures) after a carry-unwind session: USDMXN >= +1.25% with ^MOVE or ^VIX up; reference class = the other carry crosses (BRL, ZAR, AUDJPY, NZDJPY) | dollar_fx | flow_mechanics | price state (one-day) | long MXN (reversal, carry reasserts) |
| I1 | Long FXI from QE-3 across China's Golden Week to the first post-holiday close (the inversion of the 09-18 short), residual against beta-EEM reported | international | inversion | calendar (Golden Week) | long FXI |
| E1 | Long MU from k=-3 into its 09-30 print close: a 63d-floor name that has already turned (r63 <= 5 AND r5 >= 70) into its own print; reference class = liquid names in the same state, SMH-hedged | us_large (single name) | event_fingerprint | calendar (earnings k=-3) | long |

Coverage: seven asset classes through candidates (energy, volatility, us_large, rates,
other_metals, dollar_fx, international; gold enters as C1's denominator), five axes
(flow_mechanics, interaction_cell, event_fingerprint, relative_value, inversion), three
calendar-anchored (Q1, I1, E1) and five price-state (N1, V1, C1, M1, X1). V1 and Q1 are
both volatility and cannot ship as two ideas if they point the same way; V1 (long equity
vol) and Q1 (short equity vol) point OPPOSITE ways by pre-specification, so at most one
of them can be right. C1 and X1 are both expressions of the rates-dollar shock and the
red team must check whether they are one bet.

Checker assignment: kA = N1, V1, Q1 (energy and volatility); kB = C1, M1 (rates and
metals); kC = X1, I1, E1 (FX, international, single-name event).
