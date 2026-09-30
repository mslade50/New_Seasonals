# Cell map, run date 2026-09-21 (Monday)

Asof session 2026-09-21 (Monday), next session 2026-09-22 (Tuesday), midterm year.
Prices fresh (core bar 2026-09-21). Sweep: 1181 cells scanned, 78 fired (36 event /
42 price), BH pass 3 at crit p 0.0048 (ES=F P1, ^IXIC P1, EURJPY P5b).
Caps: P5 dropped EURJPY/JPY/CAD, P5b dropped AUDNZD/^IRX. Nothing dropped matters
(yen crosses published Sunday, ^IRX killed Thursday).
Stale tape: LBS=F, ^AXJO, ^HSI, ^KS11, ^N225 (Tokyo holiday), ^SKEW. None used.

Recent briefs this cell map must not restate: 09-17 (post-Sep-expiry WEEK for
SPY/QQQ/IWM, expiry VIX, UUP), 09-20 (post-expiry MONDAY, VIX Mondays, ^TNX 5%, MOVE vs
VIX, crude break, yen crosses calm tape).

## Roll seams first (01_roll_seams.py)

The Sep equity index futures expired Friday and the continuous series moved to Dec
today. Futures minus cash on today's session: ES=F +0.75pp, NQ=F +1.11pp, YM=F +0.63pp,
against +/-0.10pp the three sessions before. Every index-futures cell that fired today
reads that carry jump as a price move.

| ticker | c2c | gap | intraday | verdict |
|---|---|---|---|---|
| HE=F | -10.60 | -11.78 | +1.34 | ROLL SEAM (third straight evening) |
| SB=F | +5.88 | +4.90 | +0.93 | ROLL SEAM |
| CT=F | +7.95 | n/a | n/a | BAD BAR (Open = 0) |
| KC=F | -6.81 | -5.04 | -1.86 | untrustworthy: Thu and Fri bars are OHLC-identical |
| CL=F | -8.31 | -3.54 | -4.94 | HALF SEAM: USO fell 3.68% today. The Oct contract expires tomorrow |
| ES=F / NQ=F | +2.24 / +3.94 | | | ROLL SEAM vs cash +1.49 / +2.83 |

CORRECTION owed: Sunday's crude item rested on a Friday CL=F bar of -6.32% that the feed
has since revised to -1.58% (Friday close 100.30). USO fell 0.96% on Friday. Sunday's item 5
was wrong, and the footnote says so.

## Event lane

| trigger | subject(s) | verdict |
|---|---|---|
| E:weekday_month, Tuesdays in September | ^VIX +1.89%, t 2.49, N 113 | SKIP: Sunday's 2x2 showed the VIX weekday cells are carried by the weekday itself. Slicing the month by Tuesday is the same cell cut a different way. Swept, bh fail |
| E:weekday_month | CL=F t -2.05 | SKIP: swept, bh fail, no mechanism, and crude's front contract rolls tomorrow |
| E:weekday_month | SPY/QQQ/^GSPC/IWM/TLT/IEF/HYG/GC/SI/EEM/JPY/DXY/EURUSD/^TNX/NG | SKIP: \|t\| < 1.7 everywhere. HG=F 43-68 has sign p 0.019 on a -0.09% mean, which is not worth a line |
| E:seasonal_doy Sep 22 | SPY/^GSPC/QQQ/IWM h5 (IWM 19 of 25 down, ^GSPC 19 of 26) | SKIP as published: it restates Thursday's post-expiry-week cell, and Sunday excluded it for the same reason. DRILL the one new thing, which is that the Monday went UP 1.5% (04) |
| E:seasonal_doy | TLT/IEF midterm 4 of 5 up, HYG midterm 4 of 4 down | DEAD: N 4-5 |
| E:seasonal_doy | NG=F h5 +7.2%, 17 of 25 | SKIP: natural gas seasonality already ran 09-08 and 09-16 |
| E:seasonal_doy | ^VIX, GC, SI, HG, CL, DXY, EURUSD, JPY, EEM, ^TNX | SKIP: sign p >= 0.08 at every horizon, midterm splits N 6 |

No top-tier event inside five sessions. Next: month end Wed 09-30 (td 7), NFP Fri 10-02
(td 9). quiet_hint false because the price lane fired.

## Price lane

| trigger | subject | verdict |
|---|---|---|
| P1 first 52w high in 30+ days | ^IXIC (BH pass, 33-14, t 3.21, N 47) | DRILL (02). Swept, BH pass, so it is eligible for a tag on its own |
| P1 | ES=F (BH pass) | DEAD: roll seam. ^GSPC closed 0.44% BELOW its 52-week high |
| P1 | NQ=F | DEAD: roll seam. ^NDX closed 0.58% below its high |
| P1b first 52w high in 90+ days | ^IXIC (N 20, t 0.83) | DRILL (02): the longer-drought version is null, and that belongs in the ^IXIC item if it runs |
| P1b | NQ=F | DEAD: roll seam |
| P2/P2b new 52w low | HE=F | DEAD: roll seam |
| P4 up | ^IRX z10 3.02 | SKIP: Thursday drilled the 3-month bill at a 52w high (n 16, null) and the edge here is negative |
| P4 up | ^FVX | SKIP: Sunday killed the whole-curve top-5% cells (sign flips across 2018) |
| P4 down | NZDUSD | SKIP: t 1.01, and the FX bars stamp before the US afternoon |
| P5 bottom 5d | HE=F, KC=F | DEAD: seam / stale feed |
| P5 top 5d | BTC-USD (solid hint, t 2.72, N 231, bh fail) | DRILL (06): day-level overlap inflates t. Swept with no BH pass, so it tops out at suggestive |
| P5 top 5d | ETH-USD | SKIP: era unstable, h1 null |
| P5 top 5d | HG=F | SKIP: h1 -0.07%, t -0.72. Copper already ran 08-20 |
| P5 top 5d | NQ=F | DEAD: roll seam |
| P5 top 5d | ZC=F, CHFJPY | SKIP: null |
| P5b bottom 21d | KC=F | DEAD: stale feed |
| P5b bottom 21d | NZDJPY, EURJPY (BH pass), GBPJPY, CHFJPY, CADJPY | SKIP: published Sunday (item 6) and still the same picture. EURJPY's BH pass is a +0.09% mean |
| P5b top 21d | ^FVX, CHF=X | SKIP: Sunday kill / null |
| P6 2-ATR down | HE=F, KC=F | DEAD: seam / stale |
| P6 2-ATR up | ^NDX (N 21, h1 -1.41%, 9-12) | DRILL (02): the base cell is bear-market rallies. Today's is 0.6% under a 52w high, so cross it with proximity to the high |
| P6 2-ATR up | QQQ (N 15) | fold into 02 with ^NDX |
| P6 2-ATR up | NQ=F, ES=F | DEAD: roll seam inflates the move |
| P6 2-ATR up | CT=F | DEAD: bad bar |
| P6 2-ATR up | BTC-USD | fold into 06 |
| P7 up streak | ETH, HG, CADJPY, CAD, JPY, AUDNZD | SKIP: all h1 \|t\| < 1.8, FX bar-timing objection |
| P9 stocks and bonds up 50bp+ | SPY, TLT | SKIP: edge -0.005 / +0.005. Thursday killed the 1% version (n 19, 10-9) |

## Off-sweep drills from the tape (not triggers, flagged by reading the tape)

- **VIX closed HIGHER (+0.41% to 14.87) on a +1.55% SPY day.** P9d needs VIX +5%, so it did
  not fire. Mondays lift the VIX (Sunday item 2), so the drill has to control for weekday. DRILL (03).
- **Narrowness**: QQQ +2.77%, SPY +1.55%, IWM +0.52%, ^NYA +0.37%, ^DJI +0.71%. The 21d
  NDX-minus-RUT spread is now +8.25pp (Sunday held it at 6.4pp). The one-day version was killed
  Thursday. The 21d level moved materially since Sunday. DRILL (05).
- Crude: USO -3.68% is the real move. Not a fired cell, and the seam contaminates CL=F
  history on this date. Correction only, no nugget.
- EEM +2.69%: not a fired trigger, SKIP.

Revised before selection: E:weekday_month ^VIX was first marked SKIP above as a
weekday re-slice. That was wrong on inspection: last night's 2x2 covered Mondays only, and
Tuesdays in September sit at +1.86% against +0.04% on other-month Tuesdays, Welch t 2.34. It
moved to DRILL (05) before anything was selected.

## Drill outcomes

| script | outcome |
|---|---|
| `01_roll_seams.py` | Table above. Index futures rolled to December today: ES=F / NQ=F / YM=F beat cash by +0.75 / +1.11 / +0.63pp. USO -3.68% vs CL=F -8.31%. Friday's CL=F bar is now -1.58% (Sunday published -6.32%). |
| `02_ixic_breakout.py` | PUBLISH (item 2) + PUBLISH (item 3). A: 48 first-in-30-day ^IXIC highs, today's +2.26% is the biggest breakout session (next: +2.17%, 2012-09-06). Previous high 2026-06-02, 111 days. h1 33-14 is carried by droughts of 30-89 days (21-6, +0.35%); droughts of 90+ days, today's case, are 11-8 at +0.01%. Long-drought h21 15 of 19 higher (+1.46% vs +0.80% all days, base-rate sign p 0.098), but it is all post-2018 (7 of 8, +3.46%; pre-2018 +0.01%). Era h1 stable (0.27 / 0.20). Top-2 = 29%. B: the engine's 2-ATR up-day cell (N 21, h1 -1.41%) holds 12 cases 4-46% below the high (mean h1 -2.63%). The 7 within 1% of a high are +0.31% next day (4 of 7) and higher a week later in 7 of 7. C: the +2% day within 1% of a 52w high, declustered 10td: 30 episodes. h1 15-15, +0.05%. h5 22 of 30, +0.55% vs +0.22% all days, base-rate sign p 0.051. Both eras positive (13 of 16 / 9 of 14). Concentration -1%. The 209 +2% days NOT near a high: h1 -0.13%, h5 -0.18%. |
| `03_vix_up_on_rally.py` | KILL, footnote only. VIX higher on 38 of 651 SPY +1.25% days (5.8%), 17 of them Mondays (14% of Monday rallies vs 1.5-7.8% other days). The 11 with prior VIX < 20: SPY 5 of 11 higher next day, 8 of 11 a week later, 5 of 11 a month later. Three of the 11 are 2026 (May 6, Aug 4, today). The state is a rarity with no forward content. |
| `04_post_expiry_up_monday.py` | KILL as a nugget, footnote only (it would be the third telling of the post-expiry week). After an UP Monday, Tue-Fri was 3-6 at -0.76%. After a DOWN Monday it was 6-11 at -0.66%. The Monday's sign changes nothing. Monday +1%: N 2 (2001, 2010), DEAD. |
| `05_sept_tuesday_vix.py` | PUBLISH (item 1). The Sep-Tuesday cell is the Tuesday after Labor Day: 26 of them average +6.28%, 22-4, sign p 0.0003. All post-Monday-holiday Tuesdays: 130, +5.04%, 97-33. The other 87 September Tuesdays: +0.54%, 47-40, t 0.72, Welch vs other Tuesdays +1.19. Back-half-of-September Tuesdays (day >= 15): 60, -0.26%, 31-29. Labor Day Tuesday era: 14 of 18 pre-2018, 8 of 8 since. |
| `06_ndx_rut_spread_8pp.py` | PUBLISH (item 4). Spread +8.26pp, trailing-year rank 98.4, widest since 2026-05-20. 8pp+ with NDX 21d > 0 and RUT 21d < 0: 20 declustered episodes. Spread h21 narrower in 13 of 20, -1.46pp vs +0.71pp local control, t -1.74. Both eras (-1.41 / -1.49). Concentration -4%. ^GSPC h21 15 of 20 higher, +1.12% vs +0.49% local, t 0.87, base-rate sign p 0.17, worst -17.2% (2009-02). |
| `07_btc_5d_top.py` | PUBLISH (item 5). The engine's h1 +0.75% t 2.72 is overlap: 77 declustered episodes give h1 +0.17%, the same as the +0.18% control. h5 52 of 77 higher, +2.28% vs +0.90%, t 2.25, base-rate sign p 0.013. Era pre +1.31% / post +2.81%, same sign. Top-2 -1%. Swept with no BH pass, so it tops out at suggestive. |
| `08_era_checks.py` | Base-rate sign tests + era splits quoted above. |

## Selection

Tomorrow's tape: 1 (VIX Tuesday, the only event-lane cell that survived drilling).
Today in context: 2 (^IXIC), 3 (^NDX near-high 2% day), 4 (NDX-RUT spread), 5 (BTC).
All tagged suggestive. No anecdotes. Headline = item 2's long-drought next-day record.
None of the five fingerprints is repeat-blocked.
