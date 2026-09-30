## 4. Watchlist verdicts (70 active)

Values are as of the 2026-09-28 close. Live readings come from `w_live_values.py`, `w_live_values2.py` and `w_live_values3.py` (outputs in the matching `*_out.txt`), all in this folder. The W28 re-run is `w_28_first_break.py` (the 09-24 entry script with its bar moved to 09-28), and the W64 pre-read is `w_64_band_damage.py`. Dial readings are from `data/rd2_fragility.parquet` (ma10 81.2, raw 21d 37.1). Ranks are `pitch_lab.pct_rank` (trailing-252 percentile of the n-day return). No ticker was stale.

- [0] NFP x TLT at the 52w floor, non-midterm -- DATE-PARKED. The arm is the first non-midterm NFP (2027-01). The state is live (TLT 0.00% above its 252 low after Monday's fresh low, NFP 10-02), but 2026 is a midterm year.
- [1] Long LQD / short HYG at joint extremes -- PASS. There are 5 declustered episodes with 1 distinct year ex-2018, against >= 8 spanning 3 years ex-2018. The joint state is off: HYG is 2.45% below its high against within 0.5%, while LQD at 0.00% above its low meets its 2% leg.
- [2] SVXY overnight into CPI -- PASS. There has been no print since the 09-28 re-run (LOYO floor 21.9 bps against 40-50 bps). The next CPI eve (10-13) is 10 sessions out.
- [3] GLD on a miner thrust the metal has not joined -- PASS. GDX r5 is 16.3 against >= 95 (GDX -5.36% Monday). GLD is 23.79% below its 252 high against within 10%, and GLD r63 is 41.3 against >= 50.
- [4] XLE on a 5-6% crude thrust -- PASS. USO 1d was +1.13% against the [5,6)% band. The statistical legs are unchanged (band residual sign p 0.314 against <= 0.10; the [4,5) bucket is -0.186pp against >= 0).
- [5] TLT with the IG complex pinned at lows -- PASS. The kept-minus-deleted gap is +0.043pp against >= 0.35pp, and decluster-then-filter is N=4 at p 0.3125 against <= 0.05. The price state is fully live: TLT, IEF and LQD are all at 0.00% above their 252 lows.
- [6] SPY on a SKEW spike alone -- PASS. SKEW r5 is 74.6 against >= 95. The SPY leg is now ON (1.33% below its high, more than 1%), but 2026 is midterm, which blocks the cell until 2027.
- [7] Fade a crude thrust out of a deep base with a print in the hold -- PASS. USO r5 is 52.4 against >= 90 and r63 is 74.6 against <= 20. There are 4 post-2020 episodes against >= 8.
- [8] IHI 21d-rank-100 thrust -- PASS. IHI r21 is 25.4 against 100 (19.1% below its high). The reference-class Cochran p is 0.544 against < 0.05.
- [9] FXI 5d break inside a thrust while EEM holds -- PASS. FXI r5 is 25.0 against <= 20 and r21 is 33.7 against >= 80. EEM's 5d is -2.37%, which fails the EEM-holds leg, and the residual is still -0.277% against > 0.
- [10] TLT November month-position -- DATE-PARKED. The entry window is tdom 4-12 of November, about 2026-11-05 to 11-17.
- [11] Short SPY at its high while TLT sits at its low -- PASS. The joint state is off: SPY is 1.33% below its high against within 0.5%, while TLT at 0.00% above its low meets the 1% leg. There has been no joint day since 08-14, so the de-concentrated mean is unchanged at +0.039% against 0.15% (5x cost).
- [12] SPY on a VIX pop in a calm tape -- PASS. Two of the three state legs are met: VIX rose +8.07% against >= +5%, and SPY's -0.744% is down less than 0.75% by 0.6 bp. But VIX r21 is 70.2 against <= 25, so there is no new instance, and the increment Welch t stays +1.09 against >= 2.0.
- [13] Gold on an unconfirmed rate rise -- PASS. The yield leg is ON (21-session ^TNX +0.568pt against +0.20pt), but DX r21 is 80.2 against <= 15.
- [14] XLK over XLV after a rotation gap -- PASS. The XLV-minus-XLK 1d gap is +1.21pp (XLV +0.33%, XLK -0.89%) against >= 3.0pp. The tape legs are met: SPY is 1.33% off its high against within 3%, and ATR is 0.87% of price against < 1.2%.
- [15] Short the dollar on an unconfirmed rate rise -- PASS. DX r21 is 80.2 against <= 20 (TNX r21 100.0 meets >= 65). The magnitude-floor form is 3.9 bps against 7.5 bps.
- [16] Short TLT after a big up day near its low -- PASS. TLT 1d was -0.88% against >= +1.5% (at its low, 0.00%). The [1.0,1.5) band is still wrong-signed at -0.241%.
- [17] Short KRE / long XLF on a bank-breadth washout -- PASS. The breadth leg is ON (10 of 11, 91%, at r5 <= 20 against >= 70%). But median r63 is 11.9, the broken form rather than the intact cell (the 09-17 and 09-21 notes), and the ex-crisis mean is +0.102% against +0.35%.
- [18] IEF against 0.523 TLT at a yield high with dose -- PASS. OOS episode 1 (09-09 signal) realized -65.0 bp at h=8. Episode 2 (09-23 signal) is marked at +34.8 bp from the 09-24 entry to the 09-28 close and stays open to the 10-06 close. ^TNX is still at its 252 max (252d change +106.8 bp). The arm needs >= 3 episodes averaging >= +22.1 bp.
- [19] Narrow energy thrust cluster -- PASS. The count at z10 >= 2 is 0 of 11 (best VLO -0.32) against 2 or 3.
- [20] Survivorship-free breadth with the index further off its high -- PASS. Leg (b) is ON (raw-21d fragility 37.1 against <= 50). Leg (a) fails: SPY is 1.33% below its high against more than 2.0%, 0.67pp short.
- [21] Sector washout near a 52w high, family form -- PASS. XLF (r5 4.8) and XLU (5.6) are washed out, but both are far from their highs (XLF -7.13%, XLU -16.66% against within 5%). Cochran p is 0.789 against < 0.10.
- [22] Bare dollar washout -- DATE-PARKED. The arm is the first trigger in a non-midterm year (2027 at the earliest). DX r21 is 80.2 against <= 2 anyway.
- [23] HYG at a fresh high while the index is not -- PASS. HYG is 2.45% below its high against within 0.05%, SPY is 1.33% off against >= 2.0%, and dial ma10 is 81.2 against < 50.
- [24] SMH / laggard family at a 63d floor -- PASS. No cell-C episode (r63 <= 5, 252d >= 40%, r5 < 15) has signalled after 09-15, against >= 20 needed. XBI is the only name at the floor (r63 2.4), and its r5 is 29.0 against < 15 (SMH r63 6.7, r5 44.4).
- [25] Rates repricing with zero credit stress -- PASS. HYG is 2.45% below its high against within 0.25%; IEF and LQD at 0.00% above their lows meet their 1.5% legs. The tight rung has 1 episode against >= 8.
- [26] Long IEF out of the Jackson Hole close -- DATE-PARKED. The next non-midterm anchor is 2027-08-27.
- [27] Pooled laggard still falling, 29 ETFs -- PASS. No member holds r21 >= 90 & r63 <= 10 today, so there is no candidate for the r5 < 15 print.
- [28] Short SLV after a first metals break -- CHECK. Monday was a faithful break: GLD -3.94%, SLV -5.49% and GDX -5.36%, each <= -2%. There was none in the prior 5 sessions (09-21 to 09-25; 09-23 missed on GLD -1.80%), so this is out-of-sample first break 1 of 10. It is scored, not pitched: short SLV from the 09-29 close to the 09-30 close (h=1 lag 1; SLV closed 54.95 on 09-28). The re-run reproduces the in-sample first-break record (+0.639% on 63-46, sign p 0.011).
- [29] Duration at a yield high with MOVE mid-range -- PASS. The MOVE trailing-252 level percentile is 97.6 against the [40,50) band (the ^TNX 252-high leg is ON). There are 7 episodes against the 25-30 needed.
- [30] Small-cap December month-end overnight -- DATE-PARKED. The first eligible date is 2027-12-31, and the month permutation has to clear first.
- [31] XLE at a fresh high on a down-SPY session, h=21 -- PASS. SPY fell -0.74%, but XLE is 5.25% below its 252 high against at a high. The family P is 0.051-0.062 against < 0.05.
- [32] SVXY into a print out of a (5,8.5] VIX range -- PASS. The rel-range percentile is 20.2 on the cell's 21d/252 definition against (5, 8.5]. Dial ma10 is 81.2 against <= 68.0, and alpha sign p is 0.32 against <= 0.10.
- [33] Pooled sector triple floor below the 200d -- PASS. SPY is +6.83% above its 200d SMA against below.
- [34] SPY into a print out of a dead VIX range -- PASS. Dial ma10 is 81.2 against < 50, and rel-range is 20.2 against <= 15. The next pre-print session is 10-01 (NFP 10-02).
- [35] Short the index / long vol across an extended closure -- DATE-PARKED. The next >= 3-day closures are the 2026-11-26 and 2026-12-25 boundaries.
- [36] Post-NFP TLT after a moderate prior miss with CPI in the hold -- PASS. The 10-02 print's h=3 hold ends 10-07, so CPI 10-14 falls outside it. The prior (09-04) surprise is not in macro_release_history (frozen 08-07), and drop-2021 is -0.043% against > 0.
- [37] SVXY at the first close after an extended closure -- DATE-PARKED. The next >= 3-day closure is the Christmas boundary (12-24 close to 12-28).
- [38] SPY over IWM with the dial in [56,70) -- PASS. Dial ma10 is 81.2 against [56, 70), which needs a fall of at least 11.2 points.
- [39] Long HYG at the first close back from an extended closure -- DATE-PARKED. The state leg is ON (HYG 2.45% below its 252 high), but the anchor is the next >= 4-day closure, 2026-12-28.
- [40] Short IEF with commodities at a high and a print in the hold -- PASS. DBC is 3.53% below its 252 high against at a high, and the placebo rank is 5 of 11 against 1-2.
- [41] SPY across the September PPI-then-CPI pair -- PASS. The permutation P is 0.1618 against < 0.05, and the pair gate is -0.115pp against > 0. The next pair is September 2027.
- [42] TLT from the PPI release close at a yield high -- PASS. The charged permutation is 0.2682 against < 0.05. The next PPI anchor is the 10-15 close, 12 sessions out; ^TNX is at its 252 max today.
- [43] SPY with HYG and ^TNX both at 252 highs -- PASS. ^TNX is ON (5.240 at its high), but HYG is 2.45% below its high and SPY 1.33% below, both against within 0.5%.
- [44] SVXY on a VIX-expiry x FOMC settle session -- PASS. The 2018+ mean stays negative after the 09-16 loser (-0.43%). The next collision is 2027-03-17.
- [45] NG=F September seasonal -- PASS. The mechanism leg is unmet (September ranks 5 of 12 at h=5, behind May and April); the NG=F vehicle leg is satisfiable. NG=F was -6.13% Monday.
- [46] SPY run-in to an FOMC x VIX-expiry collision, non-midterm -- DATE-PARKED. The next non-midterm collision is 2027-03-17.
- [47] Deep flush inside a top-decile trend, r5<=2 / r63>=90 corner -- PASS. No member of the 20 is at r5 <= 2 & r63 >= 90 (none even at r5 <= 5 & r63 >= 85). The excess is +1.577% against >= +1.743%, and the LOO floor is +0.436% against > +0.961%.
- [48] XLV against 0.71 SPY after a healthcare flush -- PASS. XLV r5 is 71.0 against <= 1 (IBB 65.9, XBI 29.0, IHI 58.3; 0 of 4 at r5 <= 5). The common excess is +0.294pp against >= +0.33pp.
- [49] Hedged short SVXY after a 10% VIX crush -- PASS. VIX rose +8.07% against a >= 10% crush.
- [50] SVXY into FOMC after a backwardated re-bid -- DATE-PARKED. The k=-2 anchor for the 10-28 FOMC is 10-26. VIX/VIX3M is 0.882 today (up from 0.829) against >= 0.90.
- [51] Short a large bank against XLF after an information shock -- PASS. No large bank fell >= 1.5 ATR Monday (largest WFC -1.08 and BAC -1.02 ATR close-to-close, gaps under 0.25 ATR, on a -0.74% SPY). The study legs are unmoved: banks' max-of-12 P is 0.324 against <= 0.10, and 2018+ is +0.034% against >= +0.25%. The entry expires 10-06.
- [52] TLT across the FOMC announcement at a yield high -- DATE-PARKED. The anchor is the 10-27 eve close for FOMC 10-28. ^TNX is at its 252 max, so the within-2% leg is ON today.
- [53] HYG after a spread-driven flush -- PASS. HYG z10 is -2.08 on pitch_lab (past -2) but -1.63 on the tape convention. On either convention IEF r5 is 0.4 against > 20, so Monday is the duration-driven form (the dead 09-14 cell), not the spread-driven one. The entry expires 10-07.
- [54] Long the dollar into the September quarter-end -- PASS. The arm was measured and found off on 09-21 (no-FOMC quarters +0.197pp against +0.25pp), and the QE-9 entry (09-18) has passed. DX r5 is 82.1, so the >= 95 blocker is off. The entry expires 10-08.
- [55] Short top-2 / long bottom-2 SPDRs from the quarter-end close -- DATE-PARKED. The anchor is the 09-30 close, and the 09-30 morning owns the decision. The pre-read cleared on 09-28: 2018+ +0.688% on 23-11 against +0.40%, and September 2018+ +2.545% on 7-1. It uses the fixed 09-16 QE-10 ranking: short XLE and XLV, long XLY and XLU. The entry expires 10-01.
- [56] IWM from a non-September quad close after a washout -- DATE-PARKED. The next quad is 2026-12-18 (signal 12-17).
- [57] SPY from an opex close after a VIX crush, non-midterm -- DATE-PARKED. The first eligible opex is 2027-01-15.
- [58] Short CL=F from October tdom 13 -- DATE-PARKED. The anchor is the 2026-10-19 close, 14 sessions out.
- [59] 63d winner lagging 21d into its print -- PASS. SPY is +6.83% above its 200d against below.
- [60] LQD against beta-IEF into the December quarter-end -- DATE-PARKED. The anchor is the 2026-12-21 close.
- [61] XLU washout with TLT hit, delay rule -- PASS. OOS episode 1 (09-18) closed at the 09-28 close: long XLU lost -3.47% at h=5 lag 1 (09-21 to 09-28). XLU's own r21 <= 5 washout episode (first day 09-14, gap 21) lost -1.19%, so the joint form trails by 2.28pp against the +0.25pp bar. That makes 1 of 5 closed and 0 beating. The joint state is still live (XLU r21 1.6, TLT r21 0.8), with 09-23 to 09-28 inside the 09-18 cluster.
- [62] EEM against beta-SPY into the December quarter-end -- DATE-PARKED. The anchor is the 2026-12-23 QE-5 close.
- [63] Long crude after a 21d thrust round-trips -- PASS. USO r5 is 52.4 against <= 3 (CL=F r5 27.4). The 09-15 r21 >= 90 print keeps the 10-session window open only through today's 09-29 close. The CL=F form has not cleared sign p 0.05.
- [64] Hedged SVXY after a sub-tail MOVE spike -- PASS, within a hair. The band leg is met on the entry script's own full-history quantiles: ^MOVE +6.06% is the 92.2nd percentile of daily moves (band +5.13% to +9.03%). On 2018-03+ quantiles it is the 89.99th (q90 +6.069%), and the PIT trailing-252 rank is 88.9. The damage leg fails: SPY fell -0.744% against more than 0.75%, 0.6 bp short, so Monday sits in the no-damage half. An uncharged pre-read of the damage-half band cell (`w_64_band_damage.py`) gives h=1 +0.291%, h=2 +0.213% and h=3 +0.394% against +0.35%, so even a clean firing would clear only at h=3, before the band-walk charge.
- [65] TLT after a high-volume down day into a fresh low -- PASS. TLT 1d was -0.88% against <= -1.25%, 0.37pp short. The volume (2.03x against 1.5x), at-the-low (0.00%) and MOVE (+6.06% against < +8.7%) legs are all met.
- [66] Long MXN after a 1.5% carry-unwind day -- PASS. USDMXN 09-28 was +0.056% (MOVE and VIX both up) against >= +1.50%.
- [67] UNG volume-thrust re-fire, no EIA in the hold -- PASS. UNG 09-28 was -3.05% on 1.61x volume against >= +5% on >= 3x. The 09-24 re-fire (entry at the 09-25 close) is marked -3.05% and exits at today's 09-29 close. OOS is 0 of 5 closed.
- [68] Short the dollar into payrolls from k=-4 with ^TNX at its 252 high -- PASS (OOS only). The 09-28 k=-4 close qualified: ^TNX 5.240 is at its 252 max (0.00% off, against within 1%). OOS episode 1 is therefore running: short DX from 101.200 (UUP 28.70) to the 10-02 NFP close, to be scored on the 10-05 morning. That is 0 of 3 closed; the arm needs 3 at >= 2-1 at h=4 plus a positive pooled k=-3 neighbour.
- [69] Long gold 15%+ under its high with the dollar thrusting, above-200d half -- PASS. GLD 377.91 is 9.24% under its 200d (416.39) against above; it was 5.53% under when the entry was written. The drawdown leg holds (23.79% under its 252 high), but the DX leg has lapsed (r21 80.2 against >= 85).

Tally: 1 CHECK, 52 PASS, 17 DATE-PARKED, 0 EXPIRED. No entry expires before 09-29. The nearest expiries are W55 (10-01), W48 and W49 (10-05), W51 (10-06), W53 (10-07), W54 (10-08) and W63 (10-14).

### Fired or near today

**W28 fires, and it is scored rather than pitched.** Monday was the first faithful complex-wide break since the rewrite: GLD -3.94%, SLV -5.49% and GDX -5.36%, with no faithful break on 09-21 to 09-25. It is out-of-sample first break 1 of 10 (the arm is 8-2 or better with a mean of at least +0.30%). The score is a short SLV from the 09-29 close to the 09-30 close, recorded on the 10-01 morning. `w_28_first_break.py` confirms the first-break flag on the entry's own mask and reproduces the in-sample record (+0.639% on 63-46, sign p 0.011 against SLV's 46.37% down-rate). In-sample 2026 first breaks ran 7-2 at lag 1.

**Out-of-sample episodes open or closed today:**
- **W61 closed episode 1 badly.** It lost -3.47% against -1.19% for XLU's own washout, trailing by 2.28pp.
- **W18** episode 2 is +34.8 bp and open to 10-06.
- **W67** re-fire is -3.05% and exits at today's close.
- **W68** episode 1 is running from the 09-28 close (DX 101.200) and will be scored on 10-05.

**Within a hair, not armed:**
- **W64:** the MOVE band leg is met on the script's convention (92.2nd percentile), but SPY missed the damage leg by 0.6 bp (-0.744%). The uncharged damage cell clears +0.35% only at h=3.
- **W12:** the VIX pop and SPY legs are met, the SPY leg by 0.6 bp. But the VIX r21 of 70.2 is far from <= 25, so this is not an instance.
- **W65:** TLT fell -0.88% against -1.25%. Every other leg is met (2.03x volume, at the low, MOVE under +8.7%).
- **W20:** SPY is 1.33% off its high against more than 2.0%. Raw-21d fragility (37.1) meets its leg.
- **W53:** pitch_lab z10 -2.08 crosses -2, but the tape convention reads -1.63 and IEF r5 is 0.4 against > 20. This is the dead duration-driven form, so it is not near arming.
- **W17:** bank breadth is 91% at r5 <= 20, but median r63 is 11.9 (broken form), and the arm is a cost bar no single session moves.

**Live state legs that are ON, with the rest of the arm unmet:** W6 (SPY more than 1% off its high, blocked by midterm), W13 (+0.568pt yield rise, DX r21 80.2), W39 (HYG 2.45% off its high, no anchor until 12-28), W52 (^TNX at its 252 max, anchor 10-27), W5 (TLT, IEF and LQD all at 252 lows) and W50 (VIX/VIX3M up to 0.882 against 0.90, anchor 10-26).
