## 4. Watchlist verdicts (64 active, 0 expired)

Values from zz_watchlist_values.py (output in zz_watchlist_values_out.txt), tape through
the 2026-09-23 close. Dial ma10(63d) 81.1, raw-21d 45.8. Near expiries: W55 10-01,
W48 and W49 10-05, W51 10-06, W53 10-07, W54 10-08, W63 10-14, W58 10-20.

- W0 NFP x TLT: PASS, midterm-parked to 2027-01; the 10-02 NFP is a midterm print.
- W1 LQD/HYG credit divergence: PASS, HYG -1.74% off its high vs <= 0.5% (LQD 0.00%
  above its low clears <= 2%); the arm is an episode count (5 declustered, 1 year ex-2018
  vs >= 8 over >= 3 years), unchanged.
- W2 CPI overnight SVXY: PASS, CPI 10-14 (overnight entry is the 10-13 close); LOYO floor
  on file 19.7 bps vs the 40-50 bps arm.
- W3 GLD on a miner thrust: PASS, GDX r5 49.2 vs >= 95 (GDX -4.36% on the day); GLD
  -20.77% off its high also fails the within-10% leg.
- W4 XLE on a crude 5-6% day: PASS, USO +3.30% (0.93 ATR) vs the [5,6)% band.
- W5 TLT with IG pinned at lows: PASS. The price state is live again (TLT, IEF, LQD all
  0.00% above their 252 lows) but the last state day was 09-16, 5 td back, so 09-23 is not
  a first-in-10-td anchor; the arm is the construction test (kept-minus-deleted gap
  +0.043pp vs >= 0.35pp), unchanged.
- W6 SPY on a skew spike: PASS, ^SKEW r5 44.4 (fresh 09-23 bar) vs >= 95; SPY -1.05% off
  its high clears the > 1% leg by 5 bp, midterm leg fails.
- W7 fade crude thrust out of a deep base: PASS, USO r5 16.7 vs >= 90 and r63 75.0 vs <= 20.
- W8 IHI rank-100 thrust: PASS, IHI r21 13.1 vs 100.
- W9 FXI break inside thrust: PASS, FXI r5 69.4 vs <= 20 (r21 29.8 vs >= 80 also fails;
  EEM 5d +3.03% clears).
- W10 November TLT: PASS, parks to November tdom 4-12 (about 11-05 to 11-17).
- W11 short SPY at high with TLT at low: PASS, TLT 0.00% above its low clears <= 1% but
  SPY -1.05% off its high vs <= 0.5%; the arm is cost on the de-concentrated form anyway.
- W12 SPY on a vol pop in calm tape: PASS, two of three legs cleared (VIX +6.83% vs >= +5%;
  SPY -0.72% vs down less than 0.75%, by 3 bp) but VIX r21 45.2 fails the calm-tape
  <= 25 leg, so this is not an instance of the cell.
- W13 gold on an unconfirmed rate rise: PASS, DX r21 94.0 vs <= 15 (the yield leg clears,
  ^TNX +0.410pt over 21 sessions vs >= +0.20pt).
- W14 XLK vs XLV rotation gap: PASS, XLV-XLK 1d -0.17pp vs >= +3.0pp.
- W15 short dollar on unconfirmed rate rise: PASS, ^TNX r21 97.6 clears >= 65, DX r21
  94.0 fails <= 20.
- W16 short TLT after a big up day near the low: PASS, TLT 1d -1.58% vs >= +1.5%.
- W17 short KRE vs XLF on bank breadth washout: PASS, breadth 7 of 11 at r5 <= 20 (64%)
  vs >= 70%, median r63 13.9 vs the intact >= 70 form (C 1.6, GS 2.4, KEY 0.4); the arm
  is the ex-crisis cost bar (+0.102% vs +0.35% at h=3), a statistic.
- W18 IEF vs TLT curve: CHECK, the out-of-sample tally moved. OOS episode 1 (signal 09-09,
  entry 09-10, exit 09-22) realized -65.0 bps on the h=8 curve (IEF -0.02%, 0.523 TLT
  +1.20%) against the required mean of +22.1 bps. OOS episode 2 signalled at the 09-23
  close (^TNX 5.114 at its 252 max, 252-session change +97.1 bp, exactly 10 td after the
  09-09 kept day under filter-then-decluster gap 10): entry 09-24 MOC, exit 10-06 close.
  The arm needs >= 3 episodes, so it cannot fire yet; episodes 2 and 3 must now sum to
  >= +131.3 bps. No rerun (nothing parked to reproduce).
- W19 narrow energy thrust count: PASS, zero of 11 at z10 >= 2 under both conventions
  (pitch_lab.zscore max USO -0.39, tape max USO -0.08) vs the [2,3] arm.
- W20 new-high breadth survivorship-free: PASS, SPY -1.05% off its high vs > 2.0% fails;
  raw-21d 45.8 clears <= 50.
- W21 sector washout within 5% of a 52w high: PASS, no SPDR at r5 <= 5. Nine SPDRs
  (r5, off high): XLB 53.6 -5.89%, XLE 19.8 -4.84%, XLF 11.5 -6.54%, XLI 66.3 -8.55%,
  XLK 92.1 -1.22%, XLP 37.3 -7.39%, XLU 7.1 -15.60%, XLV 61.1 -3.55%, XLY 67.1 -10.96%.
  The only names inside 5% (XLK, XLV, XLE) are not washed; the lowest r5 (XLU 7.1) is
  15.6% off.
- W22 bare dollar washout: PASS, midterm-parked to 2027; DX r21 94.0 is the opposite pole
  of the <= 2 washout.
- W23 HYG fresh 52w high: PASS, HYG -1.74% off its high vs <= 0.05%; SPY -1.05% vs >= 2.0%
  off and dial 81.1 vs < 50 also fail.
- W24 semis 63d floor: PASS, out-of-sample only; SMH r63 7.1 but r5 96.0, the bounced side.
  The ex-SMH OOS tally was not recounted (the family has 86 episodes in all of history, so
  the >= 20 new episodes needed since 09-15 cannot have accrued in seven sessions).
- W25 rates repricing with no credit stress: PASS, HYG -1.74% vs within 0.25% of its high
  (IEF and LQD at 0.00% above their lows clear the 1.5% legs).
- W26 post-Jackson Hole IEF: PASS, date-parked to 2027-08-27.
- W27 laggard still falling, pooled: PASS, no holder of r21 >= 90 AND r63 <= 10 in the
  29-ETF family or the 218-name tape; the r63-floor names (IWM, XLU, XLRE, XLI, KRE, ITB)
  all sit at r21 <= 18, and SMH r63 7.1 is at r21 70.6.
- W28: adjudicated as candidate S1 (see section 5)
- W29 TLT with MOVE mid-range: PASS, ^MOVE 95.45 (+21.5%) puts its trailing-252 level
  percentile at 96.8 (from 80.2) vs the [40,50) band.
- W30 December small-cap overnight: PASS, date-parked to 2027-12-31.
- W31 XLE 52w high on a down-SPY day: PASS, the down-SPY leg clears (SPY -0.72%) but XLE is
  -4.84% off its 252 high vs a fresh high.
- W32 SVXY into a print out of compression: PASS, the entry's own relative-range percentile
  reads 21.83 vs (5.0, 8.5], so compression has lapsed under the entry definition (the
  site's abs-range/504d flag reads 14.31, which is the 14th pctile quoted yesterday); dial
  81.1 vs <= 68.0; next legal anchor NFP k=-2 = 09-30.
- W33 pooled triple floor: PASS, SPY +7.33% above its 200d vs below.
- W34 SPY into a print out of a dead range: PASS, rel-range 21.83 vs <= 15 and dial 81.1
  vs < 50; anchor 09-30.
- W35 closure risk premium: PASS, no >= 3-day closure before Thanksgiving 11-26.
- W36 post-NFP duration after a moderate miss: PASS, the 10-02 print's h=3 hold ends 10-07
  and CPI is 10-14, so the CPI-in-hold leg fails whatever the prior surprise.
- W37 SVXY after an extended closure: PASS, no closure before 11-26.
- W38 SPY vs IWM in the dial 56-70 band: PASS, dial 81.1 vs [56,70), 11.1 points above the
  band (89.5 21 sessions ago).
- W39 HYG after an extended closure: PASS, the state leg has flipped (HYG -1.74% off its
  high vs > 1%; 21d realized vol 3.90% vs 4.4%) but there is no closure before 11-26.
- W40 short IEF with commodities at a high and a print inside: PASS, DBC -2.38% off its
  252 high vs within 0.25%; CPI 10-14 and PPI 10-15 are outside an h=5 hold; the arm is
  the placebo rank (5 of 11 vs 1st or 2nd).
- W41 September PPI-CPI pair: PASS, next September pair is 2027.
- W42 TLT from the PPI close at a TNX high: PASS, PPI 10-15 (anchor is the release close);
  ^TNX at its 252 max today; the arm is the correlated-family permutation (0.2682 vs 0.05).
- W43 SPY with HYG at a high on a TNX high: PASS, ^TNX at its 252 max clears but HYG -1.74%
  vs within 0.5% and SPY -1.05% vs within 0.5% fail.
- W44 SVXY on a VIX-expiry x FOMC collision: PASS, the 10-21 expiry has no FOMC (10-28);
  2018+ tally 12 vs 16 collisions.
- W45 NG=F September seasonal: PASS, the September window closes 09-30; the arm is a
  non-roll form plus a mechanism that tops the month ladder (September 5 of 12), unchanged.
- W46 FOMC x VIX-expiry run-in, non-midterm: PASS, date-parked to 2027-03-17.
- W47 flush inside strength corner r5<=2 & r63>=90: PASS, no ETF in the corner in the
  20-family or the tape (XLU r5 7.1 sits at r63 0.4, the wrong corner); arm +1.577% vs
  +1.743%.
- W48 XLV family flush: PASS, XLV r5 61.1 vs <= 1 (IBB 71.8, XBI 48.4, IHI 54.4, none at
  <= 5). Expires 10-05.
- W49 hedged short SVXY after a 10% VIX crush: PASS, ^VIX +6.83% vs a >= 10% one-day fall;
  cost leg +0.326% vs +0.60%. Expires 10-05.
- W50 SVXY into FOMC after a re-bid: PASS, next k=-2 is 10-26; VIX/VIX3M 0.838 today vs
  >= 0.90.
- W51 short a bank vs XLF after a non-earnings slide: PASS, none of the eight names at
  >= 1.5 ATR on a -0.72% SPY (largest WFC -0.59 ATR, GS -0.47). Expires 10-06.
- W52 TLT across the FOMC announcement: PASS, anchor is the 10-27 eve close; ^TNX at its
  252 max today would clear the within-2% leg; reconciliation debt still owed.
- W53 HYG spread-driven flush: PASS, HYG z10 -1.44 on the tape convention (-1.90 on
  pitch_lab.zscore) vs <= -2, and IEF r5 17.5 vs > 20 also fails. Expires 10-07.
- W54 dollar into the September QE: PASS, arm measured and off on 09-21 (+0.197pp vs
  +0.25pp). Expires 10-08.
- W55 QE sector reversal pair: PASS today, enterable only at the 09-30 close; CHECK owed
  that morning. Preview on 63d: top two XLE +17.12%, XLV +10.50%; bottom two XLU -12.07%,
  XLI -5.36%. Expires 10-01.
- W56 IWM from a quad close after a washout: PASS, signal close 12-17 (December quad).
- W57 SPY from opex after a VIX crush: PASS, first chance 2027-01-15; the 10-16 opex is
  midterm.
- W58 CL=F October tdom-13 short: PASS, 10-19 close. Roll note: CL=F printed -2.57% on
  09-23 against USO +3.30%, on a -5.0% open gap, consistent with the October contract's
  09-22 expiry (roll seam in the continuous series).
- W59 GIS-form pre-print winner/laggard: PASS, SPY +7.33% above its 200d vs below.
- W60 LQD vs IEF December QE-7: PASS, date-parked to the 12-21 close.
- W61 utilities washout as a delay rule: PASS. The joint state is live again (XLU r21 1.19
  vs <= 5, TLT r21 18.65 vs < 25) but 3 td inside the 09-18 cluster at gap 21, so the OOS
  count stays 1 of the 5 needed (the 09-18 h=5 exits 09-28).
- W62 EEM vs SPY into the December QE: PASS, date-parked to the 12-23 QE-5 close.
- W63 crude round-trip long: PASS, the USO state fired 09-22 (r5 2.4, prior-10 r21 max
  90.9) and is off today (r5 16.7 vs <= 3); CL=F never entered it (r5 5.6) and its 09-23
  bar carries the roll seam noted at W58. No new firing to score. Expires 10-14.
