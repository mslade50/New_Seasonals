# Cell map, run 2026-09-27 (Sun). Tape asof 2026-09-25 (Fri), previewing Mon 2026-09-28

Sweep: 1,171 cells scanned, 87 fired (54 event / 33 price), BH pass 10 (crit p 0.0099).
Prices fresh through 2026-09-25. Midterm year. Monday is td 19 of 21 in September, the first of the month's and quarter's final three sessions.
The brief is titled for the next session, Monday 2026-09-28. Files are named for the run date, 2026-09-27.

Tape read (01_seam_and_tape.py): 10y +2.2bp to 5.184%, highest close since 2007-07-06, +52bp over 21 sessions; 5y -1.8bp (a twist).
TLT -0.13% to 79.32, lowest close since 2023-11-13, MTD -3.51%, QTD -7.15%. IEF +0.35%, LQD 0.06% above its 52w low.
SPY +0.54%, 0.59% under its 52w high (Aug 13); QQQ 0.40% under (Sep 22); IWM 7.34% under, MTD -3.82% vs SPY +0.81%.
Only 4 of the 9 original sector SPDRs above their 200d (XLE, XLF, XLK, XLV); 8 of 9 were 21 sessions ago.
VIX -5.1% to 14.87, MOVE -8.2% to 96.0 after +33% in two sessions. UUP -0.24% off Thursday's 52w high. USO -3.11%.

Roll seams and bad bars (rejected before selection):
- HE=F -12.91%: -11.27% opening gap, volume 10,848 -> 21,957. Front-month roll. DEAD for every HE=F cell (P2, P2b, P5, P6).
  (Thursday's HE=F -12.80% bar from the 09-24 run has been restated to -0.85%; futures bars get rewritten next pull.)
- SB=F +5.46%: all gap, zero intraday range. Seam. CT=F: Open 0.0, broken bar. PL=F: zero volume on 09-23/24, stale series.
- HG=F: volume 1,042 -> 25,665, contract switch. NG=F -1.40% vs UNG -3.64%.
- CL=F -2.29% vs USO -3.11%. Clean enough.
- Data warning: 8 tickers have no 2026-09-25 bar (LBS=F plus ^AXJO ^FCHI ^FTSE ^GDAXI ^HSI ^KS11 ^N225). Not holidays; a pull gap.

ENGINE DEFECT found in drilling: `month_window_anchors` (build_context_state.py:592) groups the price index by month
and marks the last 3 rows of each group. The current month is incomplete, so Sep 23-25 2026 are treated as
"September's final three". Every E:month_end cell tonight carries those 3 contaminated observations
(n_anchors 963 = 321 months x 3). Harmless in the pooled cells, material in `month_cell`: TLT's September
cell reads 35 of 75 up, and on complete months only it is 35 of 72 per session and 15 of 24 per block.
All my drills drop the current month. Not fixed tonight (engine change; reported to McKinley).

## Event lane (anchor = Fri Sep 25, h1 = Mon Sep 28)

| trigger | subject(s) | verdict | reason |
|---|---|---|---|
| E:month_end (final 3 sessions) | TLT 496-374 t 4.23, IEF t 5.41, ^TNX t -4.29, HYG sign p 0.0005, all **bh_pass** | DRILL | 02/07/08/09/10. Pre-specified famous cell (month-end duration extension), so it owes the sweep nothing. Questions: is September really a hole (month_cell 35/75)? where in the three sessions does the bid sit? does a bad month (TLT MTD -3.51%, lowest since Nov 2023) change it? what happens after? |
| E:month_end | NG=F +0.57% t 3.17 (tag_hint solid) | DEAD | roll seam: the NG contract expires three sessions before month start, inside this window. NG=F +0.55% per final-three session vs UNG -0.12% (02) |
| E:month_end | ^VIX +0.59% t 2.39 | SKIP | record 467-489 DOWN against a positive mean; the mean is a few spikes |
| E:month_end | SPY, ^GSPC, QQQ, IWM, EEM | DRILL | 08: one window per month (the engine pools three overlapping anchors per month, inflating t), split final-3 vs next-month first-2. IWM's September lag of 4.6pp vs SPY -> 06 |
| E:month_end | CL, HG, GC, SI, DX, EURUSD, JPY | SKIP | abs t < 1.8, or era-unstable, or FX stamping (Yahoo FX bars stamp ahead of the US afternoon) |
| E:weekday_month (Mondays in Sep) | ^VIX +3.02% t 3.46 **bh_pass** | SKIP | told 09-20 as [solid] with the 2x2 (mostly the weekday). A second telling is a repeat |
| E:weekday_month | NG=F 56-32 **bh_pass** | SKIP | told 09-13 as pre-roll Mondays. Tomorrow is the Oct contract's expiry Monday, the seam that cell excluded |
| E:weekday_month | TLT 50-31, IEF 50-32 (sign p 0.03) | SKIP | era-unstable; the month-end drill covers the bond slot tomorrow properly |
| E:weekday_month | SPY, ^GSPC, QQQ, IWM, EEM, HYG, CL, SI, GC, HG, DX, EURUSD, JPY | SKIP | abs t < 1.6; generic slot |
| E:seasonal_doy (Sep 28 +/-2) | all 18 subjects | SKIP | every all-years sign p >= 0.07 (N 19-26); midterm subsets N 4-6, p >= 0.11. GC/HG midterm 5-1 is anecdote-level noise. TLT DOY ran 09-03, NG DOY 09-08 |
| Calendar: Mon 9/28, Tue 9/29 | nothing tracked | - | calendar line |
| Calendar: Wed 9/30 month/quarter end | - | DRILL | covered by the month-end drills |
| Calendar: Thu 10/1 first session of Q4, midterm year | ^GSPC | DRILL | 03: the midterm Q4 is a pre-specified famous cell (Almanac four-year cycle). BH-exempt. N 6 since 2000 -> anecdote at best |
| Calendar: Fri 10/2 NFP 08:30 | - | SKIP | td 5, outside the k1-3 window; Wednesday/Thursday runs own it. Calendar line only. Note it is the 2nd session of the turn window |

## Price lane (anchor = Friday's print)

| trigger | subject(s) | verdict | reason |
|---|---|---|---|
| P2 / P2b / P5 / P6 | HE=F | DEAD | roll seam (above) |
| P5 bottom 5% (5d) | TLT 150-118 up h1, sign p 0.029; LQD; (cap-dropped) HYG | SKIP as items | TLT's worst week at a 52w low was drilled 09-24 (month-later 2-8, held back as a conflict). Tonight TLT's state feeds the month-end drill (item 1). HYG/LQD month-end state -> 07 |
| P5b bottom 5% (21d, cap-dropped) | TLT, IEF, LQD, HYG | SKIP | same family, folded into 02/07 |
| P5 / P5b top 5%, cap-dropped P5b | ^TNX, ^FVX | SKIP | the 10y ran 09-20, 09-23, 09-24. Footnote resolution only |
| P4 / P5 / P7 / cap-dropped P5b | ^IRX | SKIP | drilled and killed 09-17 (n 16, nothing at any horizon); ZIRP-era percent changes of a near-zero level |
| P5 / P5b / P6 down | ^MOVE (111-181, 136-185, 116-178), all **bh_pass** | SKIP | MOVE reversion told 09-20 [solid] and the jump 09-23. Friday's -8.2% after +33% is that reversion. Footnote |
| P4 / P5 / P5b / P7 | USDNOK, CAD=X, DX-Y.NYB, USDMXN, EURUSD, NZDUSD, GBPUSD, CHF=X, USDTRY, EURJPY, GBPUSD | SKIP | FX stamping rule; the dollar runs on UUP, told 09-24 |
| P5b bottom 5% | AUDJPY **bh_pass**, NZDJPY, GBPJPY, CHFJPY, CADJPY | SKIP | FX stamping; yen crosses told 09-07 and 09-20 |
| P5 / P7 / P5b | CT=F, CC=F, KC=F | DEAD / SKIP | CT broken bar; softs out of scope; KC flagged broken 09-20 |
| P9e curve steepen 2sd | SPY 81-62 t 0.9, TLT 65-68 | DRILL | 05: a twist (10y up, 5y down) with the 10y at a 52w high |
| (own observation) sector breadth | SPY within 0.6% of its high, 4 of 9 sectors above 200d | DRILL | 04: rarity and SPY forward. Sector ETFs are the condition only; subject SPY |

## Tag-hint notes
- The month-end bond cells are pre-specified (famous, mechanism: index duration extension at month end). BH-exempt; they pass anyway.
- The turn-of-month equity window is pre-specified (Ariel / Lakonishok-Smidt). The September split is a derived cut.
- The midterm Q4 is pre-specified. N 6.
- The sector-breadth cell is my own construction, not a sweep find. Tag on N of its forward cell.
- No cell tonight is published as [solid]: every item is a derived cut or small N.

## Drill outcomes (filled after Stage C)

- 02 month-end bonds by month (current month dropped): the September "hole" is mostly the engine defect. On complete months TLT's September final three run 15 of 24 blocks up, +0.28%, Welch t -0.37 vs other months. KILLED as a September story. By position (all months, per session): 3rd-last +0.061% / 54.3%, 2nd-last +0.126% / 56.4%, last +0.182% / 60.7%. The bid builds into the last session. September's LAST session is the exception, 8 of 24 up (-0.14%): a Wednesday cell, one of 36 month x position splits -> held back, footnote. HYG September block -0.28% is 2008 + 2011 (160% of the total) -> KILLED. NG=F seam confirmed.
- 03 midterm Q4: from the Sept anchor to year end, midterm 5 of 6 up, +2.83% mean, vs other years 16 of 20, +4.33%. Q4 alone 5/6 +3.63% vs 16/20 +4.28%. Worst close by end-October median -3.59% (midterm) vs -2.02%. Near-high midterms split: 2006 (at its high) +6.13%, 2018 (-0.52%) -14.02%. Midterm final three of September 1 of 6 up but three of the five downs are under 0.1% -> not used. -> PUBLISH [anecdote]: famous, and not distinct in this sample.
- 04 sector breadth: 2,291 near-high sessions (SPY within 1% of 252d high) since 2000; 5 had <= 4 of 9 above 200d (2023-05-30, 2023-06-01, 2023-06-07, 2026-09-21, 2026-09-25). N 1 declustered prior episode, DEAD as a forward stat, kept as a level. Looser 5-6 of 9: 16 declustered (21 td) with forward: SPY h21 +1.09% (10-6), h63 +1.41% (10-6, t 0.77) vs +2.11% after near-highs with >= 7 of 9 (158); worst close within 63 median -3.12% vs -2.55%. No edge. -> PUBLISH [suggestive] as a level, not a forecast.
- 05 curve twist: 10s5s 17.7bp, not extreme (252d max 46bp). Twist days N 25, null at every horizon; at a 52w-high 10y N 2. -> KILLED.
- 06 IWM lag into the month turn: spread <= -4pp N 15, h5 relative 40% up; 2018+ 2 of 8. Era flips. -> KILLED.
- 07 credit month-end state: HYG worst-quintile MTD final three 63.8% up (Welch t 0.59 vs rest), LQD 56.9% vs 66.7%, IEF 55.2% vs 64.5%. Credit and the belly do not share TLT's bad-month bid. -> footnote.
- 08 turn windows, one per month: TLT final3 +0.363% 62.6% t 4.68 (2018+ t 2.39), next2 -0.061% 47.1%. After a worst-quintile month (<= -2.93%, N 58): final3 +0.625% 37/58 t 2.78, next2 -0.358% 20/58 t -1.96, both eras. SPY h5 +0.408% 196/320 t 3.31 (2018+ 56/104); September's edition 13/26, -0.150%, Welch t -1.20, loss carried by 2008 + 2011; January 13/27 and February 14/27 look the same. EEM 2018+ final three negative -> not used.
- 09 exact numbers: recomputed everything quoted. Sign p: final3 37/58 = 0.024; next2 38 down of 58 = 0.012 (vs 50%), 0.036 (vs the 52.9% all-month down rate). Tomorrow's slot alone in bad months +0.007%, 33 of 58.
- 10 thresholds: the give-back holds from -1% to -4% MTD (next2 up 31-41%, mean -0.26 to -0.45%) vs +0.10%, 53.4% up in the 178 months down less than 1%. Not a cut artifact.

Final selection: tomorrow 1 TLT month-end after a bad month [suggestive, headline], 2 SPY Sep-Oct turn window [suggestive], 3 midterm Q4 [anecdote]; today 4 SPY near a high on 4 of 9 sectors [suggestive]. One anecdote, not the headline.
