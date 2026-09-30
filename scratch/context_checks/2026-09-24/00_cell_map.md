# Cell map, run 2026-09-24 (Thu). Tape asof 2026-09-24, previewing Fri 2026-09-25

Sweep: 1,181 cells scanned, 77 fired (36 event / 41 price), BH pass 1 (crit p 0.0001).
Prices fresh through 2026-09-24. Midterm year, next session is td 18 of 21 in September, 3 sessions before quarter end.
The brief is titled for the next session, Friday 2026-09-25. Files are named for the run date, 2026-09-24.

Tape read (01_seam_and_tape.py): 10y +4.8bp to 5.162%, its highest close since 2007-07-06, +19.4bp in two sessions, +52bp over 21 sessions and +77bp over 63.
TLT -1.29%, IEF -0.55% and LQD -0.71%, all three at 52-week closing lows. MOVE +9.57% to 104.58 (+33% in two sessions), its highest since 2026-03-30.
UUP +0.14% to its highest close since 2025-01-17. SPY -0.08%, QQQ -0.01% and IWM -0.09% on the day. VIX +3.2% to 15.67.
EM equities fell for a second day: EWW -1.24%, EWZ -1.23%, EWJ -1.28%.

Roll seams and bad bars (rejected before selection):
- HE=F -12.80%: an -11.93% opening gap with -1.00% intraday. This is a front-month roll. DEAD for every HE=F cell (P2, P2b, P5, P6).
- SB=F +4.96%: a +6.09% gap with -1.06% intraday. Oct expiry roll. Not a nugget.
- CT=F: Open printed 0.0, a broken bar. Not a nugget.
- HG=F +1.43%: volume 720 -> 24,544 with a +1.54% gap, so the series switched contracts. Not a nugget.
- NG=F +9.76% (gap +4.83%, intraday +4.70%) against UNG +6.26%. Partly a seam; the real move is about 6%.
- CL=F +2.82% against USO +2.86%. Clean.

## Event lane (anchor = today, h1 = Fri Sep 25)

| trigger | subject(s) | verdict | reason |
|---|---|---|---|
| E:weekday_month (Fridays in September) | CL=F -0.39% t -2.00, 50-63 | DRILL | 02: rerun on USO so roll Fridays drop out; where in the month it sits; era |
| E:weekday_month | ^TNX +0.41% t 1.67, 53-59 | DRILL | 02: in bp, not % of level; a median-negative cell whose mean may come from a few big days |
| E:weekday_month | EURUSD 61-37 sign p 0.0098 | SKIP | Yahoo FX bars stamp ahead of the US afternoon (the 09-17 repair). The mean is +0.045% anyway |
| E:weekday_month | NG=F 47-66, sign p 0.056 | SKIP | continuous NG=F seams at expiry, and Oct NG expires Mon Sep 28. Nat gas seasonality already ran 09-08 and 09-16 |
| E:weekday_month | ^VIX 47-66 down, mean +0.03% | SKIP | the record and the mean disagree, and era_stable is false |
| E:weekday_month | SPY, ^GSPC, QQQ, IWM (-0.10% to -0.13%, abs t <= 1.31) | SKIP | a generic slot. The post-expiry week these sit in ran 09-17 and 09-20 (footnote resolution only) |
| E:weekday_month | SI, HG, GC, TLT, IEF, HYG, JPY, EEM, DX | SKIP | abs t < 1.1, or era-unstable |
| E:seasonal_doy (Sep 25, +/-2) | EEM all years 6-17, sign p 0.017 | DRILL | 02: check whether the record survives day-by-day in the window or is one calendar spike. Midterm subset flips 3-2 up |
| E:seasonal_doy | QQQ midterm 5 of 6 up | SKIP | anecdote, and last night already used the midterm day-of-year frame on ^GSPC. A second night of it is a countdown re-telling |
| E:seasonal_doy | ^GSPC, SPY, IWM, TLT, IEF, TNX, HYG, GC, SI, HG, CL, DX, EURUSD, JPY, VIX | SKIP | every sign p >= 0.08 at N 19-26; midterm subsets N 4-6 with p >= 0.11. TLT DOY ran 09-03, NG DOY 09-08 |
| E:month_end / quarter end (Sep 28-30) | - | SKIP | the quarter-end SPY vs TLT window ran last night (09-22 run). Calendar line only |
| NFP Fri Oct 2 (td 6) | - | SKIP | outside the 5-session window. Calendar line only |

## Price lane (anchor = today's print)

| trigger | subject(s) | verdict | reason |
|---|---|---|---|
| P5:rank5 top 5% | ^MOVE 111-180 down, sign p 0.0001, **bh_pass** | SKIP | the only BH pass. MOVE's reversion after a top-5% week ran Sunday 09-20 as [solid], and last night ran the one-day jump with a low VIX. Footnote only |
| P5b:rank21 top 5% / P6 up | ^MOVE | SKIP | same family, same reason |
| P5:rank5 bottom 5% | TLT 150-117 up, sign p 0.025, h5 +0.40%, era-stable | DRILL | 03: TLT's own path after its worst 5-day return of the year at a 52w low; two-day -2.9% condition; era split; concentration |
| P5b:rank21 bottom 5% | TLT 181-150, era-unstable | DRILL | folded into 03 |
| (cap-dropped) P5/P5b | IEF, LQD, HYG | DRILL | 03 checks IEF and LQD alongside. HYG is cross-checked in 04 against IEF: HYG -1.05% vs IEF -1.71% on the week says duration, not spreads |
| P5 / P5b top 5% | ^TNX (5d 150-176, 21d 180-217 sign p 0.049) | DRILL | 04: the 10y at a 52w high on the same day UUP closes at a 52w high, with SPY and EEM forward. The one-day jump and its follow-through ran last night, so the question here is the dollar conjunction, with 10y-alone as the control |
| (cap-dropped) P4 / P5b | UUP, DX-Y.NYB | DRILL | 04, on UUP only (the 09-17 stamping repair) |
| P4 / P5 / P5b / P7 | ^IRX | SKIP | the 3-month bill at a 52w high was drilled and killed 09-17 (n=16, nothing at any horizon) |
| P2 / P2b / P5 / P6 | HE=F | DEAD | roll seam (see above) |
| P6 up / P8 up | NG=F (P6 32-34, P8 11-9) | SKIP | both base cells are null, and the bar is part seam; UNG +6.26% is the real move |
| P5 / P6 | CT=F | DEAD | broken bar (Open 0.0) |
| P4 / P5 / P5b / P6 / P7 / P8 | USDMXN, USDSEK, USDNOK, USDCNY, JPY=X, CAD=X, EURUSD, NZDUSD, AUDUSD, GBPUSD, GBPCHF, EURAUD, CHF=X, crosses | SKIP | Yahoo FX pairs stamp ahead of the US afternoon (09-17 repair). The dollar leg runs on UUP in 04. USDJPY's top-5% week was killed 09-22 on two 2008 episodes; the yen crosses ran Sunday |
| P5b (cap-dropped) | KC=F | SKIP | softs out of scope tonight; KC's bars were flagged as broken 09-20 |

## Tag-hint notes
- Only one cell carries bh_pass (MOVE). It is skipped for repetition. So every published item is a derived cut, tagged on N and never solid.
- No pre-specified famous hypothesis is in play tonight. The post-expiry week (Almanac) is resolved in the footnote only.

## Drill outcomes (filled after Stage C)

- 02 Friday cells: USO Sep Fridays 34-53 (2 flat), -0.41%, t -2.21, sign p 0.027 on decided days, Welch t -2.61 vs other Fridays. 15-35 before 2018, 19-18 since. November Fridays -0.44%, in both eras. 2026 is 0-3. -> PUBLISH as a tomorrow item [suggestive, single-era].
  ^TNX Sep Fridays: +0.54bp mean, median -0.25, and one day (2008-09-19 +33bp) carries it. DEAD. Fridays after a Thursday 10y 52w high: 18 of 41 up, -0.57bp. DEAD.
  EEM day-of-year: 17/23 down on the target tdoy only; neighbours 12-13/23 and 8-10/23. One calendar spike. DEAD.
  Post-expiry week: SPY +0.72%, IWM -0.86% since Sep 18. The S&P was up through Thursday in 7 of 26 years and only 2002 finished the week down. IWM trailed by >1pp through Thursday in 9 years and was still behind at Friday's close in all 9. -> footnote scoreboard only (cell told 09-17, re-told 09-20).
- 03 TLT: worst 1% week at a 52w low has N 5, DEAD. Worst 5% week at a 52w low: h1 7-10 (N 17). With a 21d decluster, a month later 2-8, -1.97% vs -0.66% local. Two straight -1.2% days to a 52w low: next session 6-1 +1.16%, top two 68%. -> KILLED. The yield-side version of the same move in 08/09 (N 21) splits 11-10 next session, so the 6-1 is definition-specific. The month-later 2-8 is held back as a conflicting cut. IEF and LQD in the same state: null or crisis-driven (LQD 2008/2020 carry 98%).
- 04 dollar + 10y: 10y and UUP at 52w highs together, 7 prior runs since 2008 (4 in 2022). SPY h21 6-1 but +0.90% vs +0.87% local; EEM 4-3, -0.21% vs +0.77%. With SPY within 2% of its high, only 2016-11-18. The control (10y high, UUP not) is flat too. -> PUBLISH [anecdote] as a level, with no edge claimed.
- 05 HYG residual vs IEF+SPY (504d daily fit, no look-ahead): 5d -0.71pp, z -2.23, 2.4th percentile since 2009-04. Near SPY highs (18 with forward): SPY h21 7-11, +0.22% vs +1.33% local. -> the cell map's "duration, not spreads" call was wrong on the IEF+SPY fit. See 07.
- 07 HYG robustness: IEF+IWM residual -0.38pp, z -1.01; IEF+SPY+IWM -0.62pp, z -2.00; adding LQD -0.52pp, z -1.86. Half of the miss is small-cap beta, and it opened on Monday's tech-led day. -> PUBLISH [suggestive] as a level with the caveat stated, not as a credit-stress call.
- 06 USO detail: see 02.
- 08 10y two-session jumps: next session null in every variant (N 14-158). One live cell: 15bp+ in two sessions with a 52w-high close today. N 20 with a forward week: 10y lower 13/20, -6.1bp vs +1.4bp local; IEF 14/20 up, +0.61% vs -0.03%, t 2.50, sign p 0.058. The stricter version (52w-high close on both days, N 14) is weaker: IEF h5 8-6.
- 09 detail: both eras positive (5/6 before 2018, 9/14 since). Top two carry 39%. 10 of 20 are in 2022. Ex 2022-23: 6/8. Today's exact position (second consecutive trigger day), N 7: IEF h5 5/7, top two 87%. The "last day of run" variant is look-ahead and is NOT used. -> PUBLISH as the HEADLINE [suggestive]. It is a derived cut and owes the sweep a multiplicity note.
- 10 exact figures recomputed.

Final selection: 1 tomorrow (USO Sep Fridays), 3 today (10y week-after [headline], HYG shortfall, UUP+10y [anecdote]). One anecdote. No solid tags.
