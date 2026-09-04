# Cell map — run 2026-08-24 (Monday)

asof session 2026-08-24 (Mon) | next session 2026-08-25 (Tue) | midterm year
prices_fresh = True, core bar 2026-08-24. Both lanes live.
sweep: 1195 cells scanned, 63 fired (36 event / 27 price), BH crit p 0.0096, 5 pass.

Next session position: td 17 of 21 in August, 4 sessions from month end, so
Tuesday OPENS the month's final five sessions. No top-tier release scheduled;
nearest set piece is Jackson Hole on Friday 2026-08-28 (+4 td), nearest print is
NFP on 2026-09-04 (+9 td).

## Engine hints I am not inheriting blind

- `tag_hint` is a floor. BTC-USD arrives `solid` on the strength of an
  overlapping-day count and is graded on the declustered numbers instead (drill
  08). CT=F arrives `suggestive` and is dead on data (below).
- Pre-specified vs swept. `E:weekday_month`, `E:seasonal_doy` and the
  August-month-end window are famous pre-specified calendar hypotheses and do
  not owe BH a correction. Every P-lane cell below WAS found by the sweep and
  does owe it.
- `dropped_by_cap`: P5b:rank21_extreme truncated at 8, dropping DX-Y.NYB,
  EURUSD=X, JPY=X, GBPUSD=X. The dollar drop is the same state the 2026-08-23
  brief already published (DXY 21d bottom-5% continuation), so nothing is lost
  by the cap tonight and it is not recomputed.

## Data guard first (drills 01, 02, 03)

The memory note on continuous-futures roll gaps applies to five of tonight's
seven price-lane futures subjects, so the session decomposition ran before any
verdict was written.

| ticker | session | gap | intraday | call |
|---|---|---|---|---|
| ZC=F | +6.46% | +5.48% | +0.93% | **REAL.** Corn's >+3% gaps cluster in delivery months (Sep 18, Dec 10, May 8, Mar 6); August has exactly one in 593 sessions and it is today. Not a roll slot. |
| ZW=F | +2.82% | +3.41% | -0.57% | **REAL**, same reasoning, 1 of 592 August sessions. |
| GC=F | +1.86% | +1.07% | +0.78% | **REAL**, ordinary decomposition. |
| LE=F | -4.23% | -1.84% | -2.43% | real move, but 51 of 61 comparable LE=F sessions are gap-dominated and 21 of 61 land in May. Thin livestock contract, not a nugget subject. |
| KC=F | -4.72% | -9.44% | +5.20% | **DEAD(unresolvable).** A -9.4% gap that recovers +5.2% intraday, in a ticker whose >3% gaps cluster in the H/K/N/U/Z delivery months. Cannot separate roll from crash, so it does not publish. |
| CT=F | +1.08% | Open=0 | n/a | **DEAD(corrupt bar).** Today's bar prints Open 0.00 and Close 88.01 ABOVE its own High of 82.90. 236 CT=F bars in history fail Close<=High or Open>0. Cotton's 52-week-high triggers fire off a broken bar. |

Vintage note, not tonight's job: the 2026-08-23 brief quoted KC=F at -10.6% on
2026-08-21. Today's cache has that session at -1.33%. The bar was revised.

## Event lane

| trigger | verdict |
|---|---|
| `E:weekday_month` Tuesdays in August | **SKIP(noise).** 18 subjects, and the largest abs(t) in the group is JPY=X at -2.18 on a -0.112% mean with a 51-63 record and sign p 0.15. ^GSPC is -0.019% at t=-0.19. Nothing here survives its own control, let alone Scott. |
| `E:seasonal_doy` Aug 25 +/-2, GC=F / SI=F | **DRILL then SKIP(repeat).** Gold's all-years cell is 18-7 up at sign p 0.0216 and the midterm subset is 2-4 down. Drill 10 confirmed and sharpened it (cool-entry years 16-4, sign p 0.0059). It is the same cell the 2026-08-16 brief published as `E:seasonal_doy|GC=F|midaug_vs_stretched_entry` (n=25, hit 76.0, sign p 0.0073). Same subject, same month, same stretched-entry split. That is a repeat with a moved number, so it does not go in. |
| `E:seasonal_doy` Aug 25, NG=F | **DEAD(era flip).** The engine's hint is the midterm h5 cell, 6 of 6 down at -5.38%, sign p 0.0156. Drill 11 widened it: all years is 10-15 up at -0.31%, t=-0.17, and the era split flips hard, pre-2018 -3.18% against 2018+ +5.80%. One of the six midterm legs is -0.04%. A 6-observation subset of a coin flip whose parent sign flips at 2018 is not publishable under the era rule. |
| `E:seasonal_doy` Aug 25, DX-Y.NYB | **SKIP(covered).** h1 18 down of 25, sign p 0.0378, but the dollar was the closing nugget of 2026-08-23 and the 200-day cross was 2026-08-19. |
| `E:seasonal_doy` Aug 25, TLT / HYG | **SKIP(repeat).** `E:seasonal_doy|TLT` published 2026-08-17, td_since 5, and its h5 window from that anchor ends today. A second late-August TLT nugget five sessions later is the countdown re-telling the rules ban. Drill 10 also kills it on merit: TLT's August last-5 is 18-6 up, but its own non-August last-5 control is 157-107 at t=4.37, so August is not the fact, month end is. |
| August month-end window | **DRILL** (09, 10, 12). Not in `cells_index`: `E:month_end` anchors the last THREE sessions and today is 4 out, so the window Tuesday opens is invisible to the sweep. Inventory gap, logged below. |
| `E:jackson_hole` 2026-08-28 | **SKIP(banned re-telling).** k=4, outside the engine's k in 1..3. Published 2026-08-23 as the VIX symposium-session cell. It is not the next session, so it earns no escalation re-telling, only a Calendar line. |
| `E:cpi` `E:nfp` `E:ppi` `E:fomc_*` `E:opex` `E:quad_witching` `E:vix_expiry` `E:election` | **SKIP(out of window).** Nearest is NFP at +9 td. Calendar block only. |
| `E:holiday_pre` / `E:holiday_post` | **SKIP(no closure nearby).** Labor Day is 2026-09-07. |

## Price lane

| trigger | subjects | verdict |
|---|---|---|
| `P4:z10_extreme` up | BTC-USD (z10 3.19, `solid`, BH pass), ETH-USD, USDTRY=X, ZC=F | **DRILL** (08) for BTC, and the drill's job is the decluster: the sweep counts 295 overlapping bars and the same cell has 63 episodes. USDTRY **SKIP(degenerate)**, a managed depreciation trending one way is not a forward-return claim, same call as 2026-08-23. ETH folded into BTC. ZC=F handled in 03/04. |
| `P5:rank5_extreme` top | BTC-USD, GC=F, ZC=F | **SKIP(same state)** for BTC. GC=F t=1.13 on the sweep's rank cell, which is the wrong cut; the magnitude cut is drill 07. |
| `P5b:rank21_extreme` top | BTC, ETH, ZC=F, GC=F, SB=F | GC=F **DRILL** (07) but on MAGNITUDE, not percentile: 21d +15.79% is the 99.6th percentile of gold's full history, and the percentile cell the sweep computed (t=-0.50) buries that. SB=F at a 52-week high with rank63 100.0 is real but sugar is not a subject Scott reads; **SKIP(subject)**. |
| `P5b:rank21_extreme` bottom | USDZAR=X (BH pass), HE=F, UUP | **SKIP.** USDZAR t=2.41 is the dollar-washout state already published 2026-08-23 wearing an EM hat. HE=F **DEAD(roll)** per the standing memory note. UUP is DXY again. |
| `P5`/`P5b`/`P4` down | LE=F, USDCNY=X | **SKIP(subject/magnitude).** LE=F real but thin, USDCNY h1 +0.098% on a managed rate with a 48-48 record. |
| `P6:two_atr_day` up | ZC=F | **DRILL** (03, 04). |
| `P6:two_atr_day` down | KC=F, USDHKD=X, USDCNY=X, LE=F | KC=F **DEAD** per the guard table. USDHKD is the peg band. |
| `P7:up_streak` | USDTRY=X, ^FTSE, CT=F | **SKIP.** USDTRY degenerate, ^FTSE t=-0.53, CT=F dead on data. |
| `P1`/`P1b:new_52w_high` | CT=F | **DEAD(corrupt bar)**, see the guard table. |
| NOT FIRED but live on the tape | QQQ 5d -3.23% (6.8th percentile of its year) while ^NYA closed 0.38% from a 252-day high, EFA 0.61% and HYG 0.11% from theirs | **DRILL** (05, 06). No trigger sees this: the P-lane measures each subject alone and the P9 family carries no index-versus-breadth pair. Second inventory gap. |
| NOT FIRED but live | ^VIX +4.76% on a session SPY fell 0.29% | **SKIP(published).** The 2026-08-17 brief ran `P:vol_bid_shallow_tape`, VIX up 5%+ on a session the index fell under 1%. Today is 4.76% against -0.29%. Same state, inside 5 sessions, and the number has not moved enough to re-tell. |
| NOT FIRED but live | SI=F 21d +17.58% while 40.07% BELOW its 252-day high | **DRILL then DEAD** (07). A thrust inside a drawdown is a different animal, but there are 6 episodes and 5 with a full forward, and the control cell is 52 episodes of nothing. Useful only as the contrast line inside the gold nugget: what happens 33 times in 6,501 gold sessions happens 392 times in silver. |

## Selected

1. `P4:z10_extreme|QQQ_vs_NYA|narrow_weakness` (today) - drills 05, 06
2. `P5b:rank21_extreme|GC=F|magnitude_15pct` (today) - drill 07
3. `P6:two_atr_day|ZC=F|august_overnight_gap` (today) - drills 01, 02, 03, 04
4. `P4:z10_extreme|BTC-USD|decluster` (today) - drill 08
5. `E:month_end|^GSPC|august_last5_entry_split` (tomorrow) - drills 09, 10, 12

Two anecdotes (gold n=10, August last-5 n=13), which is the budget. Headline is
nugget 1, which is `suggestive`.

## Inventory gaps found tonight

1. `E:month_end` anchors only the last three sessions of a month. The last five
   is the conventional window and it is invisible to the sweep on the day it
   opens, which is exactly the day it is relevant.
2. No index-versus-breadth pair in the P9 family. QQQ in the bottom decile of
   its own year while ^NYA sits at a 52-week high fires nothing, and it is the
   single most legible thing on tonight's tape.
3. The price lane has no bar-integrity screen. CT=F fired two 52-week-high
   triggers off a bar whose Close exceeds its own High, and 236 of its bars fail
   that check. A one-line OHLC sanity gate would have dropped both.

Engine-side changes, logged here, not made tonight.
