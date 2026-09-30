# Surface map, 2026-09-28 (Monday)

State: `data/pitch_state.json` built 05:11 ET. Freshest bar 2026-09-25 (the prior session), so prices are current. One stale ticker (LEG), not used. Dial ma10(63d) 80.9, down from 88.6 21 sessions ago, so still fragile, and the exposure leg is at 0.0x. P/C fear off (34th pctile, data 09-25). Signals on: NYSE Net Highs only (5d EMA -282 with SPY 0.59% below its 252 high). Book staged: 2 OVS shorts (COHU, MXL, both semis, overflow). Event sleeve flat, trend sleeve in CASH. Seasonal board meta is stale (asof 2026-08-05) and carries 0 A+B setups, so it contributes nothing today.

Entry-timing constraint that shapes the whole map: every order enters at TODAY's close (lag 1 off the 09-25 bar) or tomorrow's open, and a hold is at most 10 sessions. A calendar anchor more than about 8 sessions out can only be reached as a long run-in, and one that needs a later entry close (for example the 09-30 quarter-end close) cannot be placed from this morning at all.

Scoreboard read: 7 graded ideas lifetime. Grade B is 5-0 at +0.60R and grade C 1-1 at -0.24R. By axis, event_fingerprint 2 (+0.62R), interaction_cell 2 (+0.29R), inversion 2 (+0.31R), relative_value 1 (+0.10R). That is still a handful, so there is no axis tilt beyond a mild preference for event-anchored work.

## 1. Calendar events x asset classes

Events in the [-5, +15] window: NFP 10-02 (td 4), CPI 10-14 (td 12), PPI 10-15 (td 13), monthly opex 10-16 (td 14). Flow dates not in the macro file: month-end and quarter-end 09-30 (td 2), the Q4 turn 10-01, and earnings for MU and CAG on 09-30, NKE on 10-01 and PEP on 10-08. Outside the window: FOMC 10-28 (td 22), VIX expiry 10-21 (td 17), election 11-03 (td 26).

| event | class | verdict |
|---|---|---|
| NFP 10-02 | all ten classes | **CHECK as C2.** One fingerprint table across all ten proxies, run as the only placeable form: a run-in from today's close (k=-4) to the print close and to print+1. It is conditioned on today's state (DX 21d rank >= 85, ^TNX at a 252 high), and the pre-specified legs are short DX and long GLD. Every other cell of the table is reported and charged as a search. |
| NFP 10-02 | rates | Folded into C2. The standing NFP x TLT cells are watchlist 0 (midterm-dead, parks to 2027-01) and 36 (prior-surprise band, which needs a CPI inside the hold and has none). |
| NFP 10-02 | volatility | Folded into C2. Watchlists 32 and 34 are the pre-print compression cells, and both are blocked by the dial (ma10 80.9 against the <= 68 and < 50 arms). The anchor also needs a k=-2 entry (09-30), which today cannot place. |
| CPI 10-14, PPI 10-15, opex 10-16 | all ten classes | **Not examined today.** At td 12 to 14, a hold entered at today's close would have to run 12 to 14 sessions to reach the print, which is past the 10-session ceiling, and any pre-print anchor needs an entry close that later mornings own. Watchlists 2, 42 and 57 carry these cells to their mornings. |
| Month-end / quarter-end 09-30 | rates | **Dismissed, registry.** The month-end TLT parent died twice: the divergence gate adds +0.04pp, and the ungated anchor's index-extension mechanism decayed (registry lines 1126 and 1765). The QE-12 TLT-against-SPY rebalancing window carries no quarter-end premium (+0.218% against +0.233% at ordinary month-ends). TLT at its 252 low is also the worst bucket of the gated cell (-0.581%). |
| Month-end / quarter-end 09-30 | us_large | **Dismissed, registry.** Short SPY across the quarter turn is wrong-signed, the quarter turn equals an ordinary month turn (+0.11pp, t 0.28), and the month-end anchor on equities is closed (registry line 2265). |
| Month-end / quarter-end 09-30 | us_small | **CHECK as C8.** The Q4 turn crossed with IWM at a 63d floor (rank 0.4) is a cross the registry has not run: it closed the month-end anchor on the index but never gated it on a small-cap washout. |
| Month-end / quarter-end 09-30 | dollar_fx | **Dismissed, watchlist 54.** It was measured on 09-21 and is off: no-FOMC quarters pay +0.197pp over ordinary month-ends, below the +0.25pp bar. |
| Month-end / quarter-end 09-30 | credit | **Dismissed.** Watchlist 60 is December-only, and September is the one losing quarter (-24.7 bp). |
| Month-end / quarter-end 09-30 | volatility | **Dismissed, registry.** Long SVXY residual QE-3 to QE+1 is closed (line 5336). |
| Month-end / quarter-end 09-30 | international | **Dismissed, registry.** The EEM quarter-end short is dead, and its long flip is parked to December (watchlist 62). FXI across Golden Week is closed in both signs. |
| Month-end / quarter-end 09-30 | gold, metals, energy | **Not examined.** No flow mechanism ties a quarter-end to these. The commodity-index roll (the GSCI 5th-9th business day in October) is the nearest, and falsifying it needs curve positioning the repo lacks (honesty rule), so it cannot be verified. |
| Quarter-end reversal (sectors) | sectors | **Watchlist 55, PASS today.** Its anchor is the 09-30 close, which today cannot place. The 09-30 morning owns it. |
| MU and CAG 09-30, NKE 10-01, PEP 10-08 | single names / semis | **Dismissed, registry.** The pre-print washout lane is dead on liquid names (NKE is exactly that state), and the turned 63d laggard into the print is dead, with MU in the state at 1-2. The ungated print premium is +0.164% on names, and an SMH translation would dilute it further while overlapping the OVS semis shorts staged today. |

## 2. Tape extremes, by class (full 218-name tape sorted)

| class | what sits at an edge | verdict |
|---|---|---|
| us_large | SPY is 0.59% below its 252 high and QQQ/^NDX 0.40% below. DIA 21d rank 9.9. Tape breadth: 49.5% of names above their 200d, 19.3% with a 21d rank above 50, and NYSE net highs -282. This is extreme narrowness at an index high. | **CHECK as C7** (a nearest-neighbour analogue of this joint state). The NYSE-divergence short and TLT forms are registry-closed (kA_a1_nyse_div), and watchlist 20 needs SPY more than 2% off its high. |
| us_small | IWM 63d rank 0.4 (-5.7%), 21d -5.4%, while QQQ is +4.8% over 21d, a 10.2pp size spread. | **CHECK as C1** (relative value on the size spread) **and C8** (the Q4 turn). |
| rates | ^TNX at its 252 high with a 63d rank of 100. TLT at a fresh 252 low (63d rank 0.4), IEF 0.35% and LQD 0.06% off their lows, and TLT/IEF volume 2.0x/2.3x on a flat Friday. | **Mostly dismissed, watchlist and registry.** The TLT-at-low family is dead in every form tried (watchlists 5, 16, 18, 29, 65 and the registry). Watchlist 65 needs a -1.25% day and Friday was -0.13%. The volume spike on a flat day has no mechanism the repo can test. Rates enter today through C2 (NFP) and C3 (bond vol). |
| credit | HYG 5d rank 4.4 (z10 -1.26) and LQD at its 252 low, while SPY sits within 1% of its high. | **CHECK as C4** (credit not confirming the index high, which inverts watchlist 23). Watchlist 53 needs HYG z10 <= -2, and today reads -1.26. |
| gold | GLD 20.7% below its 252 high (21d -6.6%) with DX 21d rank 88.5. GDX 19.8% off its high. | **CHECK inside C5** (gold outright after a dollar-driven drawdown) **and in C2's table.** Watchlists 3 and 13 are not armed (GDX 5d rank 30.6, DX rank the wrong side). |
| metals | SLV 44.9% below its 252 high and 41.7% above its low, so silver is the bust leg of a blow-off. | **CHECK as C5** (silver against gold after the bust). Watchlist 28 needs GLD, SLV and GDX each <= -2% on one day, and Friday was up in all three. |
| energy | USO +16.5% over 21d (rank 79) but -3.1% Friday and -3.6% over 5d. XLE flat over 21d (-0.03%), so a 16.5pp crude-over-equity gap. UNG +6.9% 5d, then -3.6% Friday on 3.2x volume. VLO +50% over 63d. | **CHECK as C6** (the first -3% day after a crude thrust) **and C9** (XLE against USO on the 21d gap, which has to confront the 63d registry kill). UNG: watchlist 67 did not re-fire on a down day, and the 09-23 natgas pitch is inside the repeat window. **Dismissed.** |
| dollar_fx | UUP z10 2.00 and 0.24% off its 252 high, DX 21d rank 88.5. | **CHECK in C2** (short DX run-in across NFP). The bare-washout and QE dollar cells are parked or off (watchlists 22, 54). The 09-16 long-dollar pitch is inside the repeat window, and C2 is the opposite side at a different anchor. |
| international | EFA 21d rank 10.3, FXI 17.2% off its high with a 63d rank of 94, EWJ +2.21% Friday. | **Dismissed.** No edge-of-range state that is not the dollar in translation. FXI's intact-thrust break is watchlist 9 (21d rank 19, not armed). EWJ's single day has no event in the repo calendar to anchor it. EEM is flat everywhere. Kept only as a column in C2 and C7. |
| volatility | VIX 14.9, 21d range at the 14th pctile, compressed 22 days. ^MOVE +38% over 21d (rank 95.6) and +19% over 5d. SVXY 0.39% off its high. | **CHECK as C3** (bond vol at an extreme while equity vol sleeps). Watchlist 64 is the MOVE-spike-day cell, and Friday's MOVE was -8.2%, so it is not armed. Watchlist 49 needs a 10% VIX crush, and Friday was -5.1%. |
| sectors / names | Utilities z10 -2.2 to -3.1 (DTE, CMS, PEG, DUK, NEE, XLU). Insurers HIG -2.8 and ALL -1.97. Staples at 252 lows (CPB, TAP, MCD, HRL). Semis thrust (AMD, INTC, QCOM +12-13% over 5d). META 5d rank 98.4. PAYX -12.7% over 5d post-print. | **Dismissed.** The utilities washout with TLT hit is watchlist 61 (out of sample only). Sector-washout families are dead ([21], [33], [47]; 33 needs SPY below its 200d). The semis thrust fade is the OVS book's own trade (COHU and MXL staged today), so a pitch there duplicates the scanner. The insurer flush has no mechanism the repo can measure (catastrophe or pricing-cycle news). Post-earnings drift on single large caps is covered by the book's print lanes, which are registry-dead. |

## 3. Seasonal and cycle cells

- Midterm year, late September into Q4. The midterm Q4 low is the famous cycle cell, and the factor-seasonality memory says the famous cells have been arbitraged away since 2013. It is a conditioner here: C8 carries a midterm split, and C2 and C7 report one.
- September seasonality is finished. The natgas September front-contract cell (watchlist 45) is out of its month.
- Board candidates are stale (asof 2026-08-05) and none is A or B. **Nothing to take.**

## 4. Watchlist (68 active, 0 expired)

Every entry gets a verdict line with today's value in `01_watchlist_verdicts.md` (merged below once it is computed). From the tape alone, none of the price-armed entries fires today. The date-armed entries whose date falls inside the window (55 on 09-30) cannot be placed from this morning.

## 4. Watchlist verdicts (68 active)

Values are as of the 2026-09-25 close. Live readings come from `w_live_values.py` and `w_live_values2.py`. The one CHECK re-run is `w_55_window_dressing_rev.py`, all in this folder. W2's own script was re-run unmodified to update its LOYO floor. Ranks are `pitch_lab.pct_rank` (trailing-252 percentile of the n-day return).

- [0] NFP x TLT at the 52w floor, non-midterm -- DATE-PARKED. The arm is the first non-midterm NFP (2027-01); the state is live (TLT 0.00% above its 252 low, NFP 10-02) but 2026 is midterm.
- [1] Long LQD / short HYG at joint extremes -- PASS. There are 4 declustered episodes against >= 8 spanning 3 years ex-2018, and the joint state is off (HYG 2.04% below its high against within 0.5%; LQD 0.06% above its low meets its 2% leg).
- [2] SVXY overnight into CPI -- PASS. Re-run today (N=102 2018+, including the 08-12 and 09-11 prints), the LOYO floor is 21.9 bps against 40-50 bps; the next CPI eve (10-13) is 11 sessions out.
- [3] GLD on a miner thrust the metal has not joined -- PASS. GDX r5 is 30.6 against >= 95; GLD is 20.67% below its 252 high against within 10%, and GLD r63 is 46.8 against >= 50.
- [4] XLE on a 5-6% crude thrust -- PASS. USO 1d was -3.11% against the [5,6)% band; the statistical legs are also unmet (band residual sign p 0.314 against <= 0.10, and the [4,5) bucket at -0.186pp against >= 0).
- [5] TLT with the IG complex pinned at lows -- PASS. The kept-minus-deleted gap is +0.043pp against >= 0.35pp, and decluster-then-filter is N=4 at p 0.3125 against <= 0.05; the price state is near-live (TLT 0.00%, IEF 0.35%, LQD 0.06% above lows).
- [6] SPY on a SKEW spike alone -- PASS. SKEW r5 is 23.8 against >= 95 and SPY is 0.59% below its high against more than 1%; 2026 is midterm, which blocks the cell until 2027.
- [7] Fade a crude thrust out of a deep base with a print in the hold -- PASS. USO r5 is 22.6 against >= 90 and r63 is 75.0 against <= 20; there are 4 post-2020 episodes against >= 8.
- [8] IHI 21d-rank-100 thrust -- PASS. IHI r21 is 16.3 against 100 (19.7% below its high); the reference-class Cochran p is 0.544 against < 0.05.
- [9] FXI 5d break inside a thrust while EEM holds -- PASS. FXI r5 is 36.5 against <= 20 and r21 is 19.4 against >= 80 (EEM 5d +1.42%); the EEM residual is still -0.277% against > 0.
- [10] TLT November month-position -- DATE-PARKED. The entry window is tdom 4-12 of November, about 2026-11-05 to 11-17.
- [11] Short SPY at its high while TLT sits at its low -- PASS. The joint state is off (SPY 0.59% below its high against within 0.5%; TLT 0.00% above its low meets the 1% leg), with no joint day since 08-14, so the de-concentrated mean is unchanged at +0.039% against 0.15% (5x cost).
- [12] SPY on a VIX pop in a calm tape -- PASS. VIX r21 is 54.0 against <= 25 and VIX 1d was -5.11% against >= +5%; the increment Welch t is +1.09 against >= 2.0.
- [13] Gold on an unconfirmed rate rise -- PASS. The yield leg is ON (21-session ^TNX +0.520pt against +0.20pt) but DX r21 is 88.5 against <= 15.
- [14] XLK over XLV after a rotation gap -- PASS. The XLV-minus-XLK 1d gap is -0.31pp against >= 3.0pp; the tape legs are met (SPY 0.59% off its high, ATR 0.86% of price).
- [15] Short the dollar on an unconfirmed rate rise -- PASS. DX r21 is 88.5 against <= 20 (TNX r21 99.6 meets >= 65); the magnitude-floor form is 3.9 bps against 7.5 bps.
- [16] Short TLT after a big up day near its low -- PASS. TLT 1d was -0.13% against >= +1.5% (at its low, 0.00%); the [1.0,1.5) band is still wrong-signed at -0.241%.
- [17] Short KRE / long XLF on a bank-breadth washout -- PASS. Bank breadth is 4 of 11 (36%) at r5 <= 20 against >= 70%, with median r63 at 14.7; the ex-crisis mean is +0.102% against +0.35%.
- [18] IEF against 0.523 TLT at a yield high with dose -- PASS. OOS episode 1 (09-09 signal) realized -65.0 bp at h=8; episode 2 signalled 09-23 (^TNX at its 252 max, 252d change +103.7bp) and stays open to the 10-06 close, against the need for >= 3 episodes averaging >= +22.1 bp.
- [19] Narrow energy thrust cluster -- PASS. The count at z10 >= 2 is 0 of 11 (best USO -0.71) against 2 or 3.
- [20] Survivorship-free breadth with the index further off its high -- PASS. Leg (b) is ON (raw-21d fragility 41.7 against <= 50) but leg (a) fails (SPY 0.59% below its high against more than 2.0%).
- [21] Sector washout near a 52w high, family form -- PASS. No SPDR is at r5 <= 5 within 5% of its high (lowest r5 is XLU at 6.7); Cochran p is 0.789 against < 0.10.
- [22] Bare dollar washout -- DATE-PARKED. The arm is the first trigger in a non-midterm year (2027 at the earliest); DX r21 is 88.5 against <= 2 anyway.
- [23] HYG at a fresh high while the index is not -- PASS. HYG is 2.04% below its high against within 0.05%, SPY is 0.59% off against >= 2.0%, and dial ma10 is 80.9 against < 50.
- [24] SMH / laggard family at a 63d floor -- PASS. No cell-C episode (r63 <= 5, 252d >= 40%, r5 < 15) has signalled after 09-15, against >= 20 needed; XBI is the only name at the floor (r63 3.2) and its r5 is 27.8 against < 15 (SMH r63 9.9, r5 81.7).
- [25] Rates repricing with zero credit stress -- PASS. HYG is 2.04% below its high against within 0.25% (IEF 0.35% and LQD 0.06% meet their 1.5% legs); the tight rung has 1 episode against >= 8.
- [26] Long IEF out of the Jackson Hole close -- DATE-PARKED. The next non-midterm anchor is 2027-08-27.
- [27] Pooled laggard still falling, 29 ETFs -- PASS. No member holds r21 >= 90 & r63 <= 10 today (EEM has dropped out), so there is no candidate for the r5 < 15 print.
- [28] Short SLV after a first metals break -- PASS. There has been no faithful break since 09-24 (09-24 -0.30/-0.93/-1.29; 09-25 +0.44/+0.90/+0.56 for GLD/SLV/GDX against each <= -2%), so the OOS score is 0 of 10 first breaks.
- [29] Duration at a yield high with MOVE mid-range -- PASS. The MOVE trailing-252 level percentile is 96.4 against the [40,50) band (the ^TNX 252-high leg is ON); there are 7 episodes against the 25-30 needed.
- [30] Small-cap December month-end overnight -- DATE-PARKED. The first eligible date is 2027-12-31, and it needs the month permutation to clear first.
- [31] XLE at a fresh high on a down-SPY session, h=21 -- PASS. XLE is 5.34% below its 252 high on a +0.54% SPY day; the h=21 family P is 0.051-0.062 against < 0.05.
- [32] SVXY into a print out of a (5,8.5] VIX range -- PASS. The rel-range percentile is 20.6 on the cell's own 21d/252 definition (the production signal reads 14) against (5, 8.5], dial ma10 is 80.9 against <= 68.0, and alpha sign p is 0.32 against <= 0.10.
- [33] Pooled sector triple floor below the 200d -- PASS. SPY is +7.70% above its 200d SMA against below.
- [34] SPY into a print out of a dead VIX range -- PASS. Dial ma10 is 80.9 against < 50; rel-range reads 20.6 on the pitch definition (14 on the production signal) against <= 15, and the next pre-print session is 10-01 (NFP 10-02).
- [35] Short the index / long vol across an extended closure -- DATE-PARKED. The forward test needs two new >= 3-day closures, the next being the 2026-11-26 and 2026-12-25 boundaries (verify Thanksgiving's gap against the index).
- [36] Post-NFP TLT after a moderate prior miss with CPI in the hold -- PASS. The 10-02 print's h=3 hold ends 10-07, so CPI 10-14 is outside; the prior (09-04) surprise is not in macro_release_history (frozen 08-07), and drop-2021 is -0.043% against > 0.
- [37] SVXY at the first close after an extended closure -- DATE-PARKED. The next >= 3-day closure is the Christmas boundary (12-24 close to 12-28), well beyond 10 sessions.
- [38] SPY over IWM with the dial in [56,70) -- PASS. Dial ma10 is 80.9 against [56, 70), which needs a fall of at least 10.9 points.
- [39] Long HYG at the first close back from an extended closure -- DATE-PARKED. The state leg is ON (HYG 2.04% below its 252 high, more than 1%), but the anchor is the first close after a >= 4-day closure, next 2026-12-28.
- [40] Short IEF with commodities at a high and a print in the hold -- PASS. DBC is 3.15% below its 252 high against at a high, and the placebo rank is 5 of 11 against 1-2; an h=5 hold from today ends 10-05 with no print inside.
- [41] SPY across the September PPI-then-CPI pair -- PASS. The 12-month permutation P is 0.1618 against < 0.05 and the pair gate is -0.115pp against > 0; October is CPI-then-PPI, so the next pair is September 2027.
- [42] TLT from the PPI release close at a yield high -- PASS. The charged permutation is 0.2682 against < 0.05; new episodes stand at 0-1 (the 09-10 PPI with ^TNX at its 252 max paid TLT -0.09% at h=3), and the next PPI anchor is the 10-15 close, 13 sessions out.
- [43] SPY with HYG and ^TNX both at 252 highs -- PASS. ^TNX is at its high (ON), but HYG is 2.04% below against within 0.5% and SPY is 0.59% below against within 0.5%.
- [44] SVXY on a VIX-expiry x FOMC settle session -- PASS. The 2026-09-16 collision added another 2018+ loser (SVXY 1d -0.43%), so the 2018+ mean stays negative against positive at >= 60% on >= 16; the next collision is 2027-03-17.
- [45] NG=F September seasonal -- PASS. The mechanism leg is unmet (September ranks 5 of 12 at h=5, behind May +3.53% and April +2.90%); the NG=F vehicle leg is satisfiable.
- [46] SPY run-in to an FOMC x VIX-expiry collision, non-midterm -- DATE-PARKED. The next non-midterm collision is 2027-03-17 (confirmed in macro_events; 2026-09-16 was a midterm one).
- [47] Deep flush inside a top-decile trend, r5<=2 / r63>=90 corner -- PASS. No member of the 20 is at r5 <= 2 & r63 >= 90 (none even at r5 <= 5 & r63 >= 85); the excess is +1.577% against >= +1.743%, and the LOO floor is +0.436% against > +0.961%.
- [48] XLV against 0.71 SPY after a healthcare flush -- PASS. XLV r5 is 75.8 against <= 1 (IBB 75.4, XBI 27.8, IHI 53.2; 0 of 4 at r5 <= 5); the 8-complex common excess is +0.294pp against >= +0.33pp.
- [49] Hedged short SVXY after a 10% VIX crush -- PASS. VIX 1d was -5.11% against a >= 10% crush; the hedged cell is +0.335% against +0.60%, and the >= 12% dose bucket is +0.097% against >= the 10% cell.
- [50] SVXY into FOMC after a backwardated re-bid -- DATE-PARKED. The k=-2 anchor for the 10-28 FOMC is 10-26; VIX/VIX3M is 0.829 today against >= 0.90.
- [51] Short a large bank against XLF after an information shock -- PASS. Banks' max-of-12 P is 0.324 against <= 0.10, and 2018+ is +0.034% against >= +0.25%.
- [52] TLT across the FOMC announcement at a yield high -- DATE-PARKED. The anchor is the 10-27 eve close for FOMC 10-28; ^TNX is at its 252 max today, so the within-2% leg is currently ON.
- [53] HYG after a spread-driven flush -- PASS. HYG z10 is -1.26 on the tape convention (-1.59 on pitch_lab) against <= -2, and IEF r5 is 7.9, which would fail the > 20 leg anyway.
- [54] Long the dollar into the September quarter-end -- PASS. The non-FOMC quarter-end premium (>= +0.25pp needed) is still unmeasured; DX r5 is 81.0, so the >= 95 blocker is off.
- [55] Short top-2 / long bottom-2 SPDRs from the quarter-end close -- CHECK. The arm clears on the pre-read: 2018+ short pair +0.688% on 23-11 against +0.40%, and September 2018+ +2.545% on 7-1 (not wrong-signed); the anchor is the 09-30 close, so it cannot be placed this morning.
- [56] IWM from a non-September quad close after a washout -- DATE-PARKED. The next quad is 2026-12-18 (signal 12-17).
- [57] SPY from an opex close after a VIX crush, non-midterm -- DATE-PARKED. The first eligible opex is 2027-01-15.
- [58] Short CL=F from October tdom 13 -- DATE-PARKED. The anchor is the 2026-10-19 close, 15 sessions out.
- [59] 63d winner lagging 21d into its print -- PASS. SPY is +7.70% above its 200d against below.
- [60] LQD against beta-IEF into the December quarter-end -- DATE-PARKED. The anchor is the 2026-12-21 close.
- [61] XLU washout with TLT hit, delay rule -- PASS. The joint state is live (XLU r21 0.8, TLT r21 2.4), but 09-23/24/25 are in the 09-18 cluster, so the OOS count is 0 of 5 closed; the 09-18 episode marks at the 09-28 close.
- [62] EEM against beta-SPY into the December quarter-end -- DATE-PARKED. The anchor is the 2026-12-23 QE-5 close.
- [63] Long crude after a 21d thrust round-trips -- PASS. USO r5 is 22.6 against <= 3 (CL=F r5 9.5), with the r21 >= 90 print on 09-15 keeping the 10-session window open through 09-29; the CL=F form has not cleared sign p 0.05.
- [64] Hedged SVXY after a sub-tail MOVE spike -- PASS. MOVE 1d was -8.20% (3rd pctile of 2018+ moves) against a rise in the 90-97th band (+6.06% to +10.25%), on a +0.54% SPY day against < -0.75%.
- [65] TLT after a high-volume down day into a fresh low -- PASS. TLT 1d was -0.13% against <= -1.25%; the volume (2.05x against 1.5x), at-the-low (0.00%) and MOVE (-8.20% against < +8.7%) legs are all met.
- [66] Long MXN after a 1.5% carry-unwind day -- PASS. USDMXN 09-25 was +1.125% with MOVE -8.20% and VIX -5.11%, against >= +1.50% with MOVE or VIX up (09-24 was +1.479%, 2 bp short).
- [67] UNG volume-thrust re-fire, no EIA in the hold -- PASS. UNG 09-25 was -3.64% on 3.38x volume against >= +5% on >= 3x; OOS stands at 0 of 5 closed, and the 09-24 re-fire (+6.26% on 5.57x) is open to the 09-29 close.

Tally: 1 CHECK, 51 PASS, 16 DATE-PARKED, 0 EXPIRED.

### Fired or near today

**W55, the one CHECK.** The adapted re-run is `w_55_window_dressing_rev.py`. It reproduces the parent exactly: all quarters +0.367% on 105. On the script's own convention (rank by 63d at QE-10), the arm clears: 2018+ short pair +0.688% on 23-11 (sign p 0.029) against +0.40%, and September 2018+ +2.545% on 7-1.

Three things weaken that result:
- **One year carries September.** 2022 is +9.82 of the September 2018+ total.
- **The ranking convention matters.** Ranked at the QE close itself, September 2018+ is wrong-signed (-0.548%, 4-4).
- **The walk charge is unpaid.**

The QE-10 ranking (09-16) is short XLE and XLV, long XLY and XLU. The anchor is the 09-30 close, so this is a pre-read for the 09-30 morning run, not something to place today.

**Live state legs that are ON, with the rest of the arm unmet:**
- **W39:** HYG is 2.04% below its high, but it has no anchor until 12-28.
- **W20:** raw-21d is 41.7, but SPY is only 0.59% off its high.
- **W13:** the yield rise is +0.52pt, but DX r21 is 88.5.
- **W18:** OOS episode 2 is running to 10-06, and episode 1 lost 65 bp.
- **W52:** ^TNX is at its 252 max, but the anchor date is 10-27.
- **W61:** the joint state is live, but it is inside the 09-18 cluster.

**Within 10% of threshold:**
- **W47:** +1.577% against +1.743%, 9.5% short. Its LOO leg (+0.436% against +0.961%) is far off, and no member is live.
- **W48:** +0.294pp against +0.33pp, about 11% short (the family is not flushed).
- **W11:** its SPY leg is 0.09pp outside the 0.5% band, and TLT is at its low. The arm itself is a cost bar that only moves on a new joint instance.

## Candidates selected for checking (9)

Classes touched: us_large, us_small, rates, credit, gold, metals, energy, dollar_fx, volatility (nine of ten; international is a column in C2 and C7). Axes: relative_value, event_fingerprint, interaction_cell, inversion, historical_analogue, flow_mechanics. Calendar-anchored: C2 and C8. Price-state: C1, C3, C4, C5, C6, C7 and C9.

**C1. Long IWM against short QQQ after a record 21-day size spread** (relative_value, us_small x us_large, price state). QQQ minus IWM over 21 sessions is +10.2pp, and IWM's 63d rank is 0.4 while QQQ sits 0.4% off its high. The claim is that the size spread mean-reverts at h=5 to 10 once it reaches a trailing-252 extreme. The other side is momentum crowding into mega-cap tech, with small caps sold as the funding leg. It must beat both an all-days spread drift and the plain "QQQ strong" day.

**C2. NFP run-in across ten classes with the dollar thrusting and the ten-year at a 252 high** (event_fingerprint, all classes, calendar). Entry at the k=-4 close (today's order), exit at the NFP close (h=4) or NFP+1 (h=5). The pre-specified legs are short DX (UUP / DX futures) and long GLD, conditioned on DX 21d rank >= 85 and ^TNX within 1% of its 252 high. The claim is that a crowded dollar-and-yields positioning into payrolls resolves against the crowd. The window also contains the month-end and the Q4 turn, so it needs a turn-of-month-matched control (the same k=-4..0 window around non-NFP month turns).

**C3. Bond volatility at an extreme while equity volatility sleeps** (interaction_cell, volatility x rates, price state). ^MOVE 21d rank >= 95 with ^VIX's 21d range percentile <= 15 or VIX < 16. The claim is that rates vol transmits to equity vol. The trade is the SVXY residual against beta-SPY (short), or SPY outright, at h=1 to 10. The other side is volatility sellers extrapolating a calm equity tape while funding markets reprice. It must clear the mandatory SPY residual, and watchlist 64 (the MOVE spike day) is the adjacent cell.

**C4. Credit failing to confirm an index high** (inversion of watchlist 23, credit x us_large, price state). HYG 5d rank <= 5 and LQD within 1% of its 252 low, while SPY sits within 1% of its 252 high. Two trades, run as one decision: short SPY at h=5/10, or long HYG on the snap-back. The split by whether IEF is flushed alongside (duration-driven, as today) or not (spread-driven) is mandatory, because registry line 5282 and watchlist 53 say duration-driven HYG flushes do not bounce.

**C5. Silver against gold after the silver bust, and gold outright after a dollar-led drawdown** (relative_value, metals x gold, price state). SLV is 44.9% off its 252 high against GLD's 20.7%. First, does long SLV against beta-GLD pay at h=5 to 10 with the gold/silver ratio at a trailing-252 extreme after a >= 35% silver drawdown? Second, does GLD pay at h=5/10 when it sits >= 15% under its 252 high with DX 21d rank >= 85? Registry silver entries (about 30 lines) must be read first.

**C6. Crude's first -3% day after a 21-day thrust** (inversion, energy, price state). USO 21d rank >= 75 (it was 79 on 09-25, with 21d +16.5%) and a first close-to-close <= -3% day inside 10 sessions. The claim is that the thrust has cracked, so short CL=F/USO at h=3 to 5. Watchlists 7 and 63 are the neighbours, and 63 is the long flip of a round-trip short. USO's short side is structurally hurt (roll decay is not shortable, registry line 1124), so the front contract has to carry it.

**C7. Nearest-neighbour analogue of today's joint tape** (historical_analogue, cross-asset). The features: SPY distance to its 252 high, the share of the tape above its 200d (or IWM 63d rank as a proxy), ^TNX 63d rank, DX 21d rank, VIX level, MOVE 21d rank, GLD distance to its 252 high, and fragility ma10 where it exists. Take the 20 nearest declustered dates since 2004, report the forward 5d/10d on the ten class proxies with honest N, and name the one proxy whose analogue sign is most consistent. The claim is that this joint state has a characteristic resolution. The other side is whoever holds the index at its high.

**C8. Long IWM across the Q4 turn from a 63-day small-cap floor** (interaction_cell, us_small x calendar x cycle, calendar plus price state). Entry at the QE-2 close (today), exit at the turn (QE+1 to QE+3, h=3 to 5), gated on IWM 63d rank <= 5. The claim is that tax and window-dressing selling of losers into the quarter-end reverses at the turn, strongest for small caps. The ungated month-turn anchor on equities is registry-closed, so the gate has to earn the whole edge (gate attribution is decisive). A midterm split is required.

**C9 is listed below the second-wave note for numbering; it was checked in wave 1.**

## Second wave (added 05:30 after wave 1 killed six of nine)

Wave 1 on metals and energy: C5A's gate is not live (ratio at the 55th pctile), C5B died on gate attribution (dollar gate filtering -0.43pp), C6's gate is not live (09-25 was the third -3% day, not the first) and the short is wrong-signed, and C9's dose runs backwards. Wave 1 on equity internals: C1 flips after 2018, C8's gate subtracts, and C7 has no characteristic resolution. Cells reopened from the map:

**C10. Long duration after a bond-volatility spike starts to crush** (flow_mechanics, rates x volatility, price state). ^MOVE printed a +21.5% day on 09-23 (99.7th pctile) and then fell -8.2% on 09-25, with its 21d rank still at 95.6. The claim is that duration holders who de-risk on a rates-vol spike (vol-scaled risk parity, CTA vol targeting, bank VaR) re-add once implied vol turns down, so TLT/IEF pay at h=1 to 5 from the first >= 5% MOVE fall inside 5 sessions of a top-3% daily MOVE rise. This mechanism is written down before the check, per watchlist 65's condition. The registry's "MOVE/VIX at a trailing-year high, traded on duration" (line 2280) is the level form; this is the turn. The TLT-at-a-low family re-anchors, so filter_vs_reanchor is mandatory.

**C11. Long equity volatility into the midterm October** (interaction_cell, volatility x cycle x month, calendar). A midterm year, VIX 14.9 with its 21d range compressed for 22 sessions, and five weeks to the 11-03 election. The claim is that hedging demand into the midterm election builds through October, so ^VIX rises and SVXY lags from the QE-2 close to +10 sessions in midterm years more than in other years. N is small by construction (six midterms since 2002 on ^VIX, two in SVXY's -0.5x era), so this is a grade-C cell at best, and its competitor is the plain September-to-October VIX seasonal in all years. It must beat that, not zero.

**C12. Long utilities against beta-TLT after utilities fall faster than duration explains** (relative_value, sectors x rates, price state). XLU is -8.53% over 21d (z10 -2.23, 63d rank 0.4) against TLT's -4.41%, with DTE, CMS, PEG, DUK and NEE at z10 -2.4 to -3.1. The claim is that the XLU residual on a rolling TLT-and-SPY regression at a trailing-252 extreme low mean-reverts at h=5 to 10: the bond proxy has been sold for more than its duration. The other side is growth rotation and sector outflows. Watchlist 61 (utilities washout with TLT hit, long XLU outright) is out of sample only and the rate-sensitive family was wrong-signed on that form. C12 is the residual pair, not the outright, and it has to show that difference is real.

**C9. Long XLE against short USO on a 21-day crude-over-equity gap** (relative_value, energy, price state). USO +16.5% over 21d against XLE -0.03%. The claim is that equities price the crude spike as transitory and the gap closes. It must confront the dead 63d version (b5_xle_uso_divergence: "a 63d relative-performance spread is a bear-tape selector by construction") and show that the 21d gap is a different object, not a lookback neighbour.
