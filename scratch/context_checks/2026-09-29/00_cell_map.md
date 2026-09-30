# Cell map, run 2026-09-29 (Tue). Tape asof 2026-09-29 (Tue), previewing Wed 2026-09-30

Sweep: 1,217 cells scanned, 112 fired (72 event / 40 price), BH pass 19 (crit p 0.019).
Prices fresh through 2026-09-29. Midterm year. Wednesday is td 21 of 21 in September: the final session of the month and of Q3.
NFP is Fri Oct 2, 08:30 ET, 3 td ahead, so the k3 NFP cells fire tonight and their h1 IS Wednesday's month-end session.
Title names Wednesday 2026-09-30. Files are named for the run date, 2026-09-29.

Tape read (01_tape_and_seams.py):
- 10y +1.5bp to 5.255%, highest close since 2002-05-14 (yesterday's 5.240% was the highest since 2007-06-12). Six straight up closes. 5y -0.5bp.
- TLT -0.50% to 78.23, lowest close since 2023-11-06, SIXTH straight lower close (Sep 22-29), 5d -4.31%, MTD -4.84%, QTD -8.43%. Volume 2.36x its 63d mean.
- HYG -0.23% to 77.36, lowest since 2026-03-30, sixth straight lower close, volume 149.3M = 4.21x its 63d mean. LQD, IEF at 52w lows (vol 1.9x, 1.7x).
- SPY -0.18% (1.5% under its high), QQQ +0.19%, IWM -0.36% (MTD -4.83%, QTD -6.89%). XLU +1.17% on 2.4x volume.
- GLD +1.32% after Monday's -3.94%. USO -4.44% (CL=F -3.95%, clean: USO gap -3.05%). UUP +0.17%, highest since 2025-01-13.
- VIX -0.19% to 16.04. MOVE +4.70% to 106.6 (5d +35.7%). 4 of 9 original sector SPDRs above their 200d.

Roll seams and bad bars (rejected before selection):
- HE=F -10.96% on a -12.14% opening gap, volume 11,079 -> 27,357. The same front-month seam restated for a third night. DEAD for every HE=F cell.
- SB=F +7.90% on a +6.24% gap, volume 32k -> 85k: October sugar expires Sep 30. DEAD.
- SI=F volume 49 -> 37,702 and HG=F 653 -> 34,170: contract switches. GC=F +1.12% vs GLD +1.32%, clean.
- NG=F +0.47% on a +4.97% gap vs UNG -4.08%: seam. Not a fired cell.
- 6 tickers have no Tuesday bar (LBS=F ^AXJO ^HSI ^KS11 ^N225 ^SKEW). Not used.

ENGINE DEFECT still live (memory: context-engine-month-end-defect): the current month's last three rows count as month end in
`month_window_anchors`. Tonight's history includes Sep 25, 28, 29 as a "final three". Drills drop the current month.

Running items from last night (not re-told, footnote only): TLT's h2 from the fifth close needs Wednesday above 78.62 (+0.50%);
EEM +0.30% on the first of the final two; UUP +0.17%.

## Event lane (anchor = Tue Sep 29, h1 = Wed Sep 30)

| trigger | subject(s) | verdict | reason |
|---|---|---|---|
| E:nfp k3 | SPY 189-127 t 2.89 solid **bh_pass**; ^GSPC t 2.40, QQQ sign p 0.0008, IWM sign p 0.0087 **bh_pass** | DRILL | 02. NFP is always early in the month, so the k3 h1 session sits on the month turn. Is this cell the turn-of-month effect in disguise, and what does it do when h1 is the month's LAST session, as tomorrow? SPY k3 told 09-01; not blocked (19 td). Pre-specified event cell, but the k-offset is a swept choice |
| E:nfp k3 | ^VIX -0.85% 137-179 **bh_pass** | DRILL | 02, same split. The VIX falls on the month turn too |
| E:nfp k3 | EEM t 2.06 (bh fail), ^TNX, HG, NG, CL, SI, TLT, HYG, DX, EURUSD, GC, IEF, JPY | SKIP | abs t < 1.7 or bh fail; FX pairs out on stamping. EEM's month-end cell ran last night |
| E:month_end | TLT t 4.30, IEF t 5.52, ^TNX t -4.33, HYG sign p 0.0005, all **bh_pass** | DRILL (final session only) | 03. The bad-month final-three bid ran Sunday (headline) and TLT's five-down ran Monday. Tomorrow is the escalation to the final session itself: one re-telling allowed only with new specificity. Question: how much of the bid sits on the last session alone, at quarter-ends, after the first two of the three fell (both did), and on September's last (held back Sunday: 8 of 24). Pre-specified famous cell, BH-exempt |
| E:month_end | EEM 455-388 **bh_pass**, era-unstable | SKIP | told last night (item 1, quarter-end EEM). Repeat |
| E:month_end | NG=F t 3.07 (solid hint) | DEAD | roll seam, NG expiry inside the window (09-27 drill 02) |
| E:month_end | ^VIX +0.59% t 2.38 | SKIP | record 466-490 DOWN against a positive mean; a few spikes. Folded into 05 as a check only |
| E:month_end | SPY, ^GSPC, QQQ, IWM | DRILL via 05 | the pooled turn window ran Sunday (item 2). The final session of a quarter, and of September, is a different cut |
| E:month_end | CL, HG, GC, SI, DX, EURUSD, JPY | SKIP | abs t < 1.7, era-unstable, or FX stamping |
| E:weekday_month (Wednesdays in Sep) | ^VIX 44-69 sign p 0.012 **bh_pass** | SKIP | the September-weekday VIX family ran 09-20 (Mon) and 09-21 (Tue). Same mine, no mechanism |
| E:weekday_month | CL=F +0.53% t 2.32 | SKIP | oil September-weekday cells told 09-24 (USO Fridays); same mine, bh fail |
| E:weekday_month | SPY, ^GSPC, QQQ, IWM, EEM, HYG, TLT, IEF, NG, SI, HG, GC, DX, EURUSD, JPY, ^TNX | SKIP | abs t < 1.9, generic slot |
| E:seasonal_doy (Sep 30 +/-2) | SPY midterm 5 of 6 down -0.95%, ^GSPC 4 of 6, all-years 14-12 | DRILL | 05, with the month-position check. Sep 30 is September's final session, so the DOY cell is the quarter-end cell. Swept, owes BH, fails it -> at best [anecdote] on the midterm cut |
| E:seasonal_doy | ^VIX midterm 5 of 6 down -4.1%, h5 +8.9% | DRILL via 05 | same month-position test; N 6 |
| E:seasonal_doy | TLT, IEF, ^TNX, HYG, GC, SI, HG, CL, NG, DX, EURUSD, JPY, EEM, QQQ, IWM | SKIP | all-years sign p >= 0.10; midterm N 4-6 p >= 0.06. TLT DOY ran 09-03, NG 09-08 |
| Calendar: Wed 9/30 month and quarter end | - | 03 / 05 | |
| Calendar: Thu 10/1 first session of Q4 | - | SKIP | midterm Q4 told Sunday; TLT's first-two give-back told Sunday; UUP's quarter-turn told last night |
| Calendar: Fri 10/2 NFP 08:30 | - | 02 | k3 cells; the event session itself is Thursday night's job (k1) |
| Calendar: Mon 10/5, Tue 10/6 | nothing tracked | - | calendar line |

## Price lane (anchor = Tuesday's print)

| trigger | subject(s) | verdict | reason |
|---|---|---|---|
| P7b down_streak | TLT +0.29% 63-32 t 3.35 solid **bh_pass** (engine counts every day of a run) | DRILL | 03. Now the SIXTH lower close. Monday told the fifth (h2 43 of 63). New cut: runs reaching six, and runs that reach the month's final session still going. Must not re-tell the fifth-close stat |
| P7 up_streak | ^TNX 54-72 down next day | DRILL via 03 | mirror of TLT; the yield side at its highest close since May 2002 |
| P5 rank5 bottom | TLT +0.13% 150-120 sign p 0.039 | DRILL via 03 | same story, bh fail on its own |
| P5b rank21 bottom | TLT +0.03% era-unstable | SKIP | null, covered by 03 |
| P7b down_streak | HYG 61-65 null | DRILL | 04. The streak cell is null, but HYG traded 4.21x its 63d volume on a -0.23% day at a six-month low. Volume-spike HYG sessions are a new cut. Check whether spikes cluster at month-ends (index rebalance) before reading them as stress |
| P5 rank5 bottom | HYG -0.04% null, era-unstable | DRILL via 04 | |
| P5 rank5 bottom (cap-dropped) | IEF, LQD | SKIP | same bond selloff as TLT; 03 covers it |
| P5b rank21 (cap-dropped) | ^TNX, ^FVX, ^IRX, HYG, IEF, LQD | SKIP | same rates cluster; 03/04 |
| P5b rank21 (cap-dropped) | NZDJPY, CHF=X, GBPJPY, NZDUSD | SKIP | FX stamping |
| P5b rank21 (cap-dropped) | CT=F | DEAD | broken bars (open of zero Monday) |
| P5 / P5b top | ^MOVE 113-181 **bh_pass**, 138-185 **bh_pass** | SKIP | MOVE reversion told 09-20 and 09-23, its jump 09-23, and killed again last night. Repeat |
| P5 top | ^TNX, ^FVX | SKIP | null, era-unstable; the 10y level goes in as context via 03 |
| P5b rank21 bottom | EURJPY, AUDJPY, CHFJPY **bh_pass** | SKIP | FX stamping rule; yen crosses ran 09-27 |
| P5b rank21 bottom | CADJPY, ZW=F | SKIP | null; FX stamping |
| P2 / P2b 52w low | EURUSD (P2b N 14) | SKIP | FX stamping; null (12-13); the dollar leg ran last night on UUP |
| P2 / P2b / P5 / P6 | HE=F | DEAD | roll seam (-12.14% gap) |
| P6 two_atr up | SB=F | DEAD | roll seam (+6.24% gap, October expiry) |
| P6 two_atr down | USDCNY | SKIP | FX stamping; a -0.15% move; null |
| P4 z10 up | USDTRY **bh_pass**, USDNOK, CAD=X, USDMXN, UUP | SKIP | FX stamping; USDTRY is structural drift; UUP h1 null and told last night |
| P4 z10 down | EURUSD, NZDUSD, GBPUSD | SKIP | FX stamping, null |
| P5 / P5b | USDMXN, AUDUSD | SKIP | FX stamping, null |
| P7 up_streak | USDTRY **bh_pass**, USDMXN, CAD=X, CHF=X | SKIP | FX stamping; USDTRY structural |
| P7b down_streak | AUDJPY | SKIP | FX stamping, null |
| P8 sma200 cross up | USDNOK N 23 | SKIP | FX stamping, null |
| (not fired) USO -4.44% | - | SKIP | no trigger fired; tape line at most |
| (not fired) XLU +1.17% on 2.4x volume with the 10y at a 24-year high | - | SKIP | sector ETFs are breadth context only, never a subject |

Pre-specified vs swept: E:month_end (TLT/IEF/TNX/HYG) is a famous pre-specified cell and owes no BH. The NFP-k3 offset, the DOY cell,
the HYG volume cut and the streak-length cut are swept or derived and are tagged on N, never above what BH supports.

## Drill results and selection

- 02 + 06 (NFP k3): position control alone leaves SPY +0.13% t 2.2 and VIX -1.36% t -4.0, but the k3 session is almost always a
  Wednesday, and against the same weekday at the same month position SPY goes to -0.09% (t -1.4), VIX to -0.07%. KILLED as a
  payrolls effect; told as the kill inside item 2. Wednesday month-ends (37 of the 41 k3-on-last): SPY 24 of 41, Welch t 1.95,
  Nov 30 2011 and 2022 carry 60% -> held back.
- 03 + 07 (TLT final session): quarter-end finals 46 of 96, -0.01%, vs other months 129 of 193, +0.27%, Welch t -2.78; pre-2018
  -2.04, 2018+ -1.96; IEF -2.66; 10y -0.33bp vs -1.90bp. Sept 8/24, Dec 11/24. Quarter-end AND worst-fifth MTD: 7 of 16.
  -> PUBLISH item 1 (tomorrow), headline. Derived split of a pre-specified cell: [suggestive].
- 03 + 07 (TLT sixth close): 23 of 63 earlier five-runs reached six; next session 16 of 23, +0.43%, sign p 0.047; top two carry 57%.
  None of the 23 next sessions a quarter-end. -> PUBLISH item 3 (today), with the 10y level (highest close since 2002-05-14).
- 05 + 07 (SPY final session): 151 of 319 up since 2000, -0.02%, vs 55% for other sessions (sign p 0.004), era-stable (46% / 50%);
  quarter-ends no different (Welch 1.03). -> PUBLISH item 2 (tomorrow). DOY midterm 5 of 6 = September's final session, which is
  11 of 26 in all years (Welch 0.15 vs other finals) and 2018 was +0.01% -> KILLED, footnote.
- 04 + 07 (HYG): 4.21x volume, 27th session since 2007 at that ratio (16 in 2007-09, previous 2025-04-07). Spike forwards are null
  once 2008 is set aside. Six-close: next day 13 of 33, next week 24 of 33. -> PUBLISH item 4 (today) as a level.
- Nothing tagged [anecdote]; nothing [solid] (all four are derived cuts).
