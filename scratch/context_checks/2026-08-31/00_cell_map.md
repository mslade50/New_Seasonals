# Cell map — run date 2026-08-31

- asof session: 2026-08-31 (Monday, final session of August, td 21 of 21)
- next session: 2026-09-01 (Tuesday, td 1 of 21 in September)
- cycle: midterm (year % 4 == 2)
- prices_fresh: TRUE, core bar 2026-08-31
- sweep: 1185 cells scanned, 83 fired (54 event / 29 price), BH crit p 0.0166, 11 pass
- stale tape: LBS=F, ^AXJO, ^FTSE, ^HSI, ^KS11, ^N225, ^SKEW (7 of 98). Foreign
  cash indices are price-lane only anyway; none is a candidate tonight.

## What last night already spent

The 2026-08-30 brief previewed THIS Monday, and it published at full
specificity the cell that would otherwise be tonight's obvious lead:
"SPY — the turn into September", anchored on Monday's close, 13 of 26 up at
-0.260%. Anchored on Monday's close means h1 = Tuesday 2026-09-01. That is
exactly tomorrow. It also spent the month-end IEF/^TNX duration bid, the
August-Monday VIX floor cell, the BTC z10 cell, and the grain roll-gap
finding. All five are off the table tonight; a countdown re-telling of any of
them is the exact failure the novelty rule exists for.

## Event lane

| trigger | verdict |
|---|---|
| E:turn_of_month (18 subjects) | SPLIT. SKIP(SPY, ^GSPC, QQQ, IWM: the September turn was published 2026-08-30 at full specificity, same anchor, same h1). DRILL(EEM): n=560, +0.288%, 341-215, t 4.02, era-stable, BH pass, the strongest cell in the sweep and never published here. Needs its September arm and a control before it can carry a tag. DRILL(HYG) as the credit companion: n=464, 263-195, t 2.18, BH pass, and HYG sits 0.14% off a 52-week high. SKIP(GC=F t 2.34 but BH fail; SI=F sign p sits exactly on the crit line; CL=F, HG=F, NG=F, TLT, IEF, ^TNX, DX-Y.NYB, EURUSD=X, JPY=X, ^VIX: abs t below 2.5 or era-unstable, and a September arm would only thin them further). |
| E:weekday_month (Tuesdays in September, 18 subjects) | DRILL(^VIX) with prejudice: n=110, +1.797%, 66-44 up, t 2.32, era-stable, but it is the SAME CELL FORM as last night's item 4 (weekday x month VIX off a low base) two nights running, and it fails BH. It publishes only if the September arm says something the August arm did not. DRILL(HG=F): 41-67 down, sign p 0.0139, the only member of this group that cleared BH, and copper closed +1.94% today. SKIP(CL=F -0.617% at t -2.48, era-stable but BH fail, and crude just printed +3.49%, so the cell and the state disagree with no story to join them). SKIP(SPY, ^GSPC, QQQ, IWM: all abs t below 1.3, and the equity September story is spent). SKIP(remaining 12: abs t below 1.7, no BH). |
| E:seasonal_doy (Sep 01 plus or minus 2, 18 subjects) | DRILL(DX-Y.NYB): 19-7 up all years at sign p 0.0145, and 6 of 6 up in midterm years at +0.533%, sign p 0.0156. Two independent small-sample records pointing the same way on the exact calendar slot tomorrow occupies. Best untouched event cell in the sweep. DRILL(HG=F): 5 of 6 up in midterms at +1.078%, pairs against the September-Tuesday copper cell above, so the two share one script. SKIP(TLT: h5 midterm 0-for-5 is striking, but E:seasonal_doy|TLT published 2026-08-17, 10 td ago, and its h1 arm is 4-1 on n=5, not enough to re-tell). SKIP(SPY, ^GSPC, QQQ, IWM: sign p 0.16 to 0.42, and spent). SKIP(remaining 11: no arm below sign p 0.10). |
| Calendar, next 5 sessions | Tue Sep 1 nothing scheduled. Wed Sep 2 nothing. Thu Sep 3 nothing. Fri Sep 4 08:30 ET employment report, 4 td ahead, so it is NOT tomorrow's tape and the pre-NFP anchor (k=1..3) does not include tonight. PUBLISH in the calendar block only; an NFP nugget tonight would be a countdown. Mon Sep 7 is Labor Day, market closed, which is why the week is four sessions. |

## Price lane

| trigger | verdict |
|---|---|
| P1:new_52w_high ^TNX | DRILL, and this is tonight's lead candidate. The 10-year yield closed 4.758, its first 52-week high in 30+ days, +0.81% on the session and +6.32% over 63 days, and ^FVX printed one too. The engine's base cell is n=24 with era_stable false, so the number it hands me is not publishable as it stands. What matters is the cross-asset arm the sweep cannot see: what equities and duration did next. |
| P3:drop50_after_high and P3b:drop100_after_high, ^GDAXI | SKIP(t 0.67 and 0.86, sign p 0.34 and 0.18, era-unstable on the 50bp arm. A foreign cash index reversal with no edge over its own drift is not worth a line). |
| P4:z10_extreme up (ZC=F, ZW=F, ZS=F, CT=F, ^BVSP) | SKIP(ZC=F: repeat_blocked, published 2026-08-27, 2 td ago). SKIP(ZW=F, ZS=F, CT=F: last night established these bars are contract rolls and today's complex is the same instruments. A z10 computed across a roll measures plumbing. Gated on drill 01 regardless). SKIP(^BVSP: t 0.59, edge -0.004). |
| P4:z10_extreme down (USDCNY=X) | DEAD(mean -0.018% on a 174-day cell, t -1.26, hit 46.0%. The yuan at a 52-week low is a real macro fact, but the cell carries no number worth Scott's attention). |
| P5:rank5_extreme bottom (KC=F) | DRILL, gated. Coffee -9.21% today and -16.88% over five sessions, the 0.8th percentile of its own year and the largest single-session move in the 98-name tape. Last night tagged the 08-28 coffee bar as 96% gap, so none of this is believable until the roll test is rerun. |
| P5:rank5_extreme top (ZW=F, CT=F, ZS=F, ZC=F) | SKIP(all four are the rolled grain and softs contracts, same gate and same reason as P4 up. ZW=F clears BH at sign p 0.0066, which is exactly the trap: a strong statistic computed across an instrument change). |
| P5b:rank21_extreme (BTC-USD, ETH-USD, ZC=F, CT=F, ZW=F, SB=F) | SKIP(BTC-USD: solid on paper at n=300, t 2.87, BH pass, but "bitcoin is stretched and the base cell says it keeps going" was published last night as item 1 of the today lane. Different fingerprint, same fact, and the rule is one fact once). SKIP(ETH-USD: t 1.37, era-unstable). SKIP(the four ags: roll-gated). |
| P6:two_atr_day down (KC=F, LE=F) | KC=F folds into the coffee drill. SKIP(LE=F: 43.5% hit with a positive mean, t 0.69, and live cattle roll gaps are a known contaminant in this cache). |
| P6:two_atr_day up (USDTRY=X, ZC=F, CT=F) | DEAD(USDTRY=X: n=38, t -0.11, and a managed-devaluation series is not a return distribution). SKIP(ZC=F, CT=F: roll-gated). |
| P7:up_streak (ZS=F, ZW=F, ^BVSP, JPY=X) | SKIP(all four abs t below 1.6; the two grain members are roll-contaminated on top of that). |

## Untriggered states worth a cell of their own

The sweep enumerates triggers, not conjunctions, and tonight's most specific
fact is a conjunction it cannot see. From the tape block:

- HYG closed 0.14% below its 52-week high. TLT closed 1.44% ABOVE its 52-week
  low, IEF 0.61% above, LQD 0.75% above. Credit at its highs, duration at its
  lows, on the same day the 10-year printed a 52-week high in yield.
- ^MOVE +6.13%, the largest one-day move in the tape, from a 21-day rank of
  23.4. Bond vol waking up from a low base. ^VIX +3.4% to 14.92, still 52.0%
  below its own 52-week high.
- SPY -0.30% and 1.39% off its high; ^DJI -0.70%; breadth 67.8% above the
  200-day against 66.7% twenty-one sessions ago.

DRILL: the credit-versus-duration divergence as an explicit two-sided cell.
DRILL: the bond-vol jump off a low base as its companion.

## Drill queue

1. `01_kc_roll_check.py` — gate. Is coffee's -9.21% a price move or a roll?
2. `02_tnx_52w_high.py` — 10y yield 52-week high, first in 30+ days: the
   cross-asset follow-on for SPY, TLT, HYG and the yield itself, with era
   split and concentration.
3. `03_credit_vs_duration.py` — HYG near its 52-week high while TLT sits near
   its 52-week low, and what has followed.
4. `04_dollar_sep1.py` — DX-Y.NYB on the first session of September, all years
   and midterm years, against the all-months first-session control.
5. `05_eem_turn.py` — EEM turn-of-month, the September arm, controls, eras.
6. `06_sep_tuesday_vol.py` — the September-Tuesday VIX cell and the copper
   pair, tested against last night's August-Monday form so the brief does not
   run the same trick twice.
7. `07_move_jump.py` — MOVE up 5% or more from a bottom-third 21-day rank.

## Multiplicity

Nothing tonight is a pre-specified famous hypothesis except turn-of-month
itself, and the turn-of-month members that could have used that exemption are
the ones last night spent. Every other candidate came out of the sweep and
owes the BH line at crit p 0.0166 before it can be tagged solid.

## Stage C outcomes (appended after the drills ran)

| drill | verdict |
|---|---|
| 01 KC=F roll gate | KILLED the whole commodity lane. Coffee's -9.21% is 91% gap on 9,638 lots after five sessions of 35 to 60, the expiring contract going quiet. Corn 94% gap, soybeans 98%, cotton no open at all on 954x median volume. P5-bottom KC=F, P6-down KC=F, P5-top ZW/CT/ZS/ZC and P4-up all dead as return cells. Published instead as the roll finding, ranked last. |
| 02 ^TNX 52w high | SURVIVED, inverted. SPY h5 is 12-12 with a +0.171% median and the -0.456% mean is 109% two 2022 days; eras +0.329% / -1.242%. The publishable content is that the reflex is wrong, plus the yield's own one-session momentum (16-8, sign p 0.076). Published rank 4. |
| 03 credit vs duration | KILLED as a forecast. 13 episodes since 2008, every horizon inside noise (SPY h21 +0.462% at t 0.59, sign p 0.29). The raw-state-day version looks strong (180 days, 135-45, t 4.99) and that is pure overlap: declustering removes it entirely. The STATE is still a true and specific description of today, so it went into the ^TNX nugget as a clause rather than as its own item. |
| 04 / 04b / 04c dollar Sep 1 | KILLED, and this was the closest call of the night. 04 restated the engine's 19-7 / 6-of-6 cell as the calendar fact and got 16-10 / 3-3. 04b showed why: the trading-doy match lands on a real month boundary only 5 times in 26 and once in 6 midterms, so the two cells are different populations. 04c settled it. Neighbouring doys whipsaw (167 is 19-8, 169 is 19-7, 170 is 7-19, 171 is 9-17, 164 is 7-20), the pooled doy 160-180 window is +0.0048% against +0.0007% for the rest of the year, and the doy-169 record is pre-2018 (14-4) with nothing since (5-3). One column of a 21-column scan. |
| 05 / 05b EEM turn | SURVIVED as a RELATIVE cell only. The absolute September arm dies exactly as SPY's did (24-22, +0.057%), which is the check that mattered. EEM minus SPY holds: 329-229 on 558 anchors at t 3.55 against a flat 2627-2685 control, 18 of 24 years positive, top two episodes 1% of the total, and 16-7 on the exact month-end-into-September slot. Era caveat published: 217-134 becomes 112-95 after 2018, and 2026 runs 7 of 15 against it. Headline. |
| 06 Sep Tuesday vol | BOTH SURVIVED, on the separation test rather than on the raw number. ^VIX: 60.0% on September Tuesdays against 45.6% all sessions, 45.3% other-month Tuesdays and 45.4% September non-Tuesdays, so neither marginal explains it. That is what earned it a slot despite repeating last night's construction, and the bottom-third arm (49-23) is quoted as the state-matched arm, not as the cell. Tagged suggestive, not solid: sign p 0.022 misses the BH line at 0.0166. HG=F: 42-69 with a flat -0.060% mean because the three best days exceed the total, the only weekday-month member that cleared BH. |
| 07 / 07b MOVE jump | SURVIVED with its own deflator attached. A 6% MOVE day is 7.95% of sessions and 2026 has already had six since June, so the jump is not the news. Conditioned on a bottom-third 21-day rank: 58 episodes, SPY higher ten sessions later 45 times (77.6% against a 60.9% base rate and a 63.0% local control), sign p 0.00002, both eras holding at 27-6 and 18-7, top two episodes -3% of the total, and declustering at min-gap 10 or 21 changes not one number. t is 1.86 only because of the left tail (-9.46%, May 2022), so the record is the statistic quoted. |

Published: 6 nuggets (1 solid, 4 suggestive, 1 anecdote), 3 per lane, 401 words.
Killed outright: the entire ags and softs lane, the dollar seasonal, the
credit-duration forecast, and the four turn-of-month equity members last night
had already spent.
