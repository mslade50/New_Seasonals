# Cell map, run 2026-09-28 (Mon). Tape asof 2026-09-28 (Mon), previewing Tue 2026-09-29

Sweep: 1,199 cells scanned, 90 fired (54 event / 36 price), BH pass 13 (crit p 0.018).
Prices fresh through 2026-09-28. Midterm year. Tuesday is td 20 of 21 in September: the second-to-last session of the month and of Q3.
Title names Tuesday 2026-09-29. Files are named for the run date, 2026-09-28.

Tape read (01_seam_and_tape.py): 10y +5.6bp to 5.240%, highest close since 2007-06-12; 5y +6.1bp to 5.068%. Five straight up closes in the 10y.
TLT -0.88% to 78.62, lowest close since 2023-11-09, five straight down closes (Sep 22-28), 5d -3.89%, MTD -4.36%, QTD -7.97%. IEF, LQD at 52w lows.
SPY -0.74% (1.33% under its Aug 13 high), QQQ -1.07%, IWM -0.69% (MTD -4.48% vs SPY +0.06%).
GLD -3.94% (worst since 2026-06-10), SLV -5.49%. GLD 5d -5.14%, 21d -10.58%, MTD -7.47%. UUP +0.28% to a 52w high.
VIX +8.07% to 16.07, MOVE +6.06% to 101.8. 4 of 11 sector SPDRs above their 200d (XLE XLF XLK XLV), XLP and XLV up on the day.

Roll seams and bad bars (rejected before selection):
- HE=F -12.43%: -11.44% opening gap, volume 9,261 -> 27,761. Front-month roll again (Friday's -12.91% bar was restated to -1.23%). DEAD for every HE=F cell (P2, P2b, P5, P6).
- CT=F: Open 0.0, broken bar. SB=F +6.11%: +6.29% gap. HG=F: volume 1,239 -> 45,225, contract switch. PL=F / PA=F: zero volume on 09-24/25, stale until today.
- GC=F -4.00% vs GLD -3.94%, SI=F -5.00% vs SLV -5.49%: clean. CL=F +0.95% vs USO +1.13%: clean. NG=F -1.66% vs UNG -3.05%.
- 6 tickers have no Monday bar (LBS=F ^AXJO ^HSI ^KS11 ^N225 ^SKEW). Asian cash closes; not used.

ENGINE DEFECT still live (memory: context-engine-month-end-defect): `month_window_anchors` treats Sep 24, 25, 28 as September's final three
because prices end Monday. Every E:month_end cell tonight carries that contamination. Drills drop the current month.

## Event lane (anchor = Mon Sep 28, h1 = Tue Sep 29)

| trigger | subject(s) | verdict | reason |
|---|---|---|---|
| E:month_end | EEM 454-389 sign p 0.018 **bh_pass**, era-unstable | DRILL | 07/08: the pooled cell is era-unstable (2018+ final three negative, 09-27 drill 08). Question: is the bid a quarter-end effect, and does September belong? |
| E:month_end (quarter-end subset) | UUP / DX-Y.NYB | DRILL | 07: UUP at a 52w high into the quarter's final two. Quarter-end dollar funding demand is a pre-specified hypothesis; the engine's pooled DXY month-end cell is null (-0.01%) |
| E:month_end (final 3) | TLT t 4.26, IEF t 5.48, ^TNX t -4.32, HYG sign p 0.0005, all **bh_pass** | DRILL (follow-on only) | Told last night as the bad-month final-three bid (item 1, headline). A re-telling is blocked unless it adds specificity. New fact: Monday's slot failed (TLT -0.88%). 02 asks what the last two sessions did in bad months when the first of the three fell. Pre-specified famous cell, BH-exempt |
| E:month_end | NG=F t 3.14 (solid hint) | DEAD | roll seam (NG contract expiry inside the window; confirmed 09-27 drill 02) |
| E:month_end | ^VIX +0.59% t 2.39 | SKIP | record 467-489 DOWN against a positive mean; a few spikes |
| E:month_end | SPY, ^GSPC, QQQ, IWM | SKIP as a cell | the turn window ran last night (item 2). The Sep 29 day-of-year slot is a different, sharper cut -> 05 |
| E:month_end | CL, HG, GC, SI, DX, EURUSD, JPY | SKIP | abs t < 1.8, era-unstable, or FX stamping |
| E:weekday_month (Tuesdays in Sep) | ^VIX 69-45 t 2.43 **bh_pass** | SKIP | told 09-21 (sep_tuesday_ex_labor_day). Repeat |
| E:weekday_month | CL=F -0.52% t -2.09 | SKIP | USO September Fridays told 09-24; a second oil September weekday cell is the same mine, bh fail |
| E:weekday_month | HG=F 44-68 sign p 0.024 | SKIP | mean -0.08%, t -0.59; contract switch today anyway |
| E:weekday_month | SPY, ^GSPC, QQQ, IWM, EEM, HYG, TLT, IEF, NG, SI, GC, DX, EURUSD, JPY, ^TNX | SKIP | abs t < 1.8; generic slot |
| E:seasonal_doy (Sep 29 +/-2) | ^GSPC 19-7 sign p 0.0145, SPY 18-8 p 0.038, IWM 18-7 p 0.022; midterm 5 of 6 each | DRILL | 05: DOY cells died on neighbour checks before (EEM 09-24). Test by month position (Sept's second-to-last session) and against the four days around it; compare with other months' second-to-last; era split. Swept, not pre-specified: owes BH, fails it -> at best [suggestive] |
| E:seasonal_doy | ^TNX, TLT, IEF, HYG, GC, SI, HG, CL, NG, DX, EURUSD, JPY, ^VIX, EEM, QQQ | SKIP | all-years sign p >= 0.07; midterm subsets N 4-6 p >= 0.06. HYG h5 midterm 4-0 (N 4) is noise. TLT DOY ran 09-03, NG 09-08 |
| Calendar: Tue 9/29 | nothing tracked | - | calendar line |
| Calendar: Wed 9/30 month and quarter end | - | footnote | last night held back TLT on September's last session (8 of 24 up); not re-run |
| Calendar: Thu 10/1 first session of Q4 | - | SKIP | midterm Q4 told last night |
| Calendar: Fri 10/2 NFP 08:30 | - | SKIP | td 4, outside the k1-3 window; Wednesday and Thursday runs own it. Calendar line |

## Price lane (anchor = Monday's print)

| trigger | subject(s) | verdict | reason |
|---|---|---|---|
| P7b down streak | TLT 63-31 h1, t 3.42, sign p 0.0006, era-stable, **bh_pass**, tag_hint solid | DRILL | 03. Engine counts every day of a streak >= 5 (overlapping, run >= 5), so N 94 is not 94 episodes. Re-anchor on the day a streak first reaches 5, decluster, era split, cluster note, local control, and the overlap with the month's final sessions. Swept cell, so BH matters: it passes |
| P5 bottom 5% (5d) | TLT 150-119 sign p 0.034, IEF t 2.16, LQD | SKIP as items | same state as P7b; folded into 03 as a condition. TLT's worst week at a 52w low drilled 09-24 (held back) |
| P5 / cap-dropped P5b | HYG, LQD, IEF, TLT 21d bottom 5% | SKIP | same family; credit month-end state drilled 09-27 (07) |
| P7b down streak | HYG 61-64 | SKIP | t 0.03, era-unstable |
| P9b stocks and bonds both down 50bp+ | SPY 147-122 t 1.0, TLT 140-132 | DRILL | 04: sharpen with gold. Monday had SPY -0.74, TLT -0.88, GLD -3.94 and UUP at a 52w high: everything but the dollar fell. The base P9b cell is null; the three-way version is the question |
| (own observation) gold -3.94% | GLD / GC=F | DRILL | 06: not in the sweep as a fired cell (GC did not clear 2 ATR; gold's ATR is high this year). GLD drops of 3.5%+ since 2004, forward h1/h5/h21, declustered, conditioned on the dollar at a 52w high. Own construction |
| P6 two-ATR down | SI=F 121-91 sign p 0.023, t 0.05, era-unstable | SKIP as a cell | mean zero, era-unstable; silver folds into 06 as the SLV leg |
| P2 / P2b / P5 / P6 | HE=F | DEAD | roll seam |
| P6 up / P5b bottom | CT=F | DEAD | broken bar (open 0) |
| P7b down streak | ZW=F h1 +0.325% t 2.16, hit 49.4% | SKIP | mean from skew, coin-flip record; grains not a lead subject |
| P5b bottom 5% | CHFJPY **bh_pass**, CADJPY, (cap-dropped) EURJPY NZDJPY AUDJPY GBPJPY | SKIP | FX stamping rule; yen crosses told 09-07 and 09-20 |
| P4 / P5 / P5b / P7 | USDTRY **bh_pass x2**, USDMXN, CAD=X, USDNOK, EURUSD **bh_pass**, GBPUSD, (cap-dropped) NZDUSD USDSEK CHF=X GBPUSD EURUSD | SKIP | FX stamping rule (Yahoo FX bars stamp ahead of the US afternoon); USDTRY is a managed-devaluation drift. The dollar runs on UUP, told 09-24 |
| P5 / P5b / P7 top | ^TNX, ^FVX | SKIP | the 10y ran 09-20, 09-23, 09-24. Footnote: resolution of the 09-24 IEF week-after item and the new high (highest since 2007-06-12) |
| P5 / P5b top | ^MOVE (112-181, 137-185), **bh_pass x2** | SKIP | MOVE reversion told 09-20 [solid], the jump 09-23. Footnote only |
| cap-dropped P5b | ^IRX | SKIP | killed 09-17 |
| cap-dropped P5b | ^RUT, IWM 21d bottom 5% | SKIP | IWM lag into the month turn killed 09-27 (2 of 8 since 2018) |
| (own observation) VIX +8.07% on S&P -0.77% | ^VIX | SKIP | the VIX-jump family ran 09-14, 09-16, 09-22; nothing new about an 8% lift from 14.87 |
| (own observation) breadth 55.2% / 4 of 11 sectors | SPY | SKIP | told last night |

## Tag-hint notes
- TLT month-end: pre-specified (index duration extension), BH-exempt. Last night's item; tonight only a follow-on can run.
- TLT P7b: swept cell, bh_pass on the engine's overlapping count. tag_hint solid is on N 94 overlapping days; the re-anchored episode count decides the tag.
- Sep 29 DOY: swept, bh fail. Cannot be [solid].
- Gold and the three-way down day are my own constructions, not sweep finds. Tag on N and era.

## Drill outcomes (filled after Stage C)

- 02 month-end, day one missed: bad months (MTD <= -2.93% at the anchor, N 57) where the first of the final three fell (N 25): last two up 14 of 25, +0.49%, median +0.32%, sign p 0.35; all months 176 of 289 (60.9%). Top two (2011-10, 2009-05) carry 75%. Final three as a whole up 8 of 25. The miss neither cancels nor sharpens the bid -> footnote resolution of last night's item 1.
- 03 TLT fifth straight down close: 63 prior streaks re-anchored on the 5th close (engine: 94 overlapping days). h1 40 of 63, +0.23%, t 2.2, sign p 0.022 (2018+ 15 of 26, t 0.84). h2 43 of 63, +0.554% vs +0.033% all days, t 4.24, sign p 0.0026; pre-2018 26 of 37 (t 3.49), 2018+ 17 of 26 (t 2.54); top two carry 15%. h5 33-30, coin flip. 5d <= -3% (as now, -3.89%): h1 15 of 27, h2 19 of 27. At a 52w low: N 3 (1 up next day). Month-end overlaps: 7, all up at h2; the 5 anchored with two sessions left in a month (as now) all up. Excluding the 7: h2 36 of 56, +0.47%. Run lengths: 40 stopped at 5, 16 at 6, 6 at 7, 1 at 8. -> PUBLISH [suggestive] (h2 is one of four horizons looked at, h1's 2018+ era is weak, so not solid despite bh_pass on the parent). HEADLINE.
- 04 three-way down day (SPY and TLT <= -0.5%, GLD <= -2%): 18 sessions since GLD listed (Nov 2004) incl. Monday, 8 in 2026. Declustered prior 16: SPY h21 13 of 16, +2.82% vs +0.98% base, t 2.13; pre-2018 4 of 6, 2018+ 9 of 10; top two (2020-03-31, 2026-03-26) carry 52%; the six 2026 episodes all up a month later. TLT h1 6 of 16. GLD h5 5 of 16 (0 of 6 pre-2018, 5 of 10 since: era flip, not used). Base P9b pair null. -> PUBLISH [suggestive] as a level with the forward stated with its concentration.
- 05 Sep 29 day-of-year: KILLED. By month position, September's second-to-last session is 15 of 26 for the S&P, -0.30% (2008 and 2002 carry it), midterm 2 of 6; calendar Sep 29 is 8 of 18. The engine's 19 of 26 is a trading-day-of-year alignment artifact. Welch t vs all months' second-to-last -1.03.
- 06 gold drop: GLD <= -3.5% raw count 2026 = 8 incl. Monday; 2008 = 10; every other year <= 5. Declustered prior 38: h1 22 of 38 (-0.20%, median +0.20%), h5 21 of 38, h21 21 of 38; all null vs base. Conditioned on UUP near a 52w high (N 10) or the 10y near a high (N 4): null. With 21d <= -8% (N 14): h1 4 of 14 up, h21 10 of 14 -> held back (four cuts tried, N 14). -> gold rarity folds into the 04 item as one sentence.
- 07 dollar at quarter-end: UUP final two at quarter-ends 33 of 77, -0.02%, no different from other month-ends (Welch t -0.30): the pre-specified quarter-end funding bid FAILS. First two sessions of the new quarter: 47 of 77, +0.157% vs -0.038% at other month turns, Welch t 2.06; pre-2018 25 of 43, 2018+ 22 of 34. DXY 2000+: 64 of 106, +0.141% vs -0.070%, Welch t 2.45. Near a 52w high at quarter-end: last2 5 of 10, next2 6 of 10. -> PUBLISH [suggestive].
- 08 EEM quarter-end: final two at quarter-ends 62 of 93, +0.497%, vs other month-ends 89 of 187, -0.017%, Welch t 2.09; beat SPY in 65 of 93. By quarter: Mar 19/23, Jun 14/24, Dec 17/23, Sep 12/23 (-0.22%). Ex-September 50 of 70, +0.73%; Sept vs other quarters Welch t -2.15; Sept 2018+ 3 of 8. Era: pre-2018 40 of 59 (t 2.35), 2018+ 22 of 34 (t 1.37). Top two carry 3%. SPY, EFA, IWM share the September dip (-0.32%, -0.33%, -0.27%). Overlaps last night's SPY Sep-Oct window finding; EEM and the final-two cut are new. -> PUBLISH [suggestive].
- 09 numbers: all quoted numbers recomputed. UUP highest since 2025-01-17; 10y highest since 2007-06-12. Resolutions: IEF -0.73% since Sep 23 (09-24 week-after item, Wednesday left); since Sep 22 SPY -1.00%, TLT -3.83% (09-22 quarter-end item: SPY ahead, the 2018+ pattern).

Final selection: tomorrow 1 EEM quarter-end final two, September excepted [suggestive]; 2 UUP quarter-end vs quarter turn [suggestive]; today 3 TLT fifth straight down close [suggestive, headline]; 4 SPY/TLT/GLD down together, gold's eighth 3.5% drop [suggestive]. No anecdotes. Both lanes represented.
