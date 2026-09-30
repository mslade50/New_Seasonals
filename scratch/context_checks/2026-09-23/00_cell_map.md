# Cell map, run 2026-09-23 (Wednesday)

- asof session 2026-09-23 (Wed), next session 2026-09-24 (Thu), midterm year, September
- prices_fresh = true (core bar 2026-09-23). Warnings: LBS=F, ^AXJO, ^HSI, ^KS11, ^N225, ^SKEW have no 09-23 bar (Asia holiday or dead feed; none is a nugget candidate)
- sweep: 1181 cells scanned, 87 fired (36 event / 51 price), 2 BH passes at crit p 0.0016: E:weekday_month JPY=X, P5 ^MOVE
- dropped_by_cap: P4 ^IRX / CAD=X / USDNOK; P5 CAD=X; P5b ^IRX + seven yen/FX crosses; P6 GBPUSD / HYG. Examined below.
- no top-tier event in the next session; price triggers fired, so NOT a quiet tape
- Tonight's tape is a rates shock: 10y +14.6bp to 5.114 (highest close since 2007-07-12, biggest one-day rise since 2025-04-07's +17.0bp), 5y +15.5bp to 4.997 (highest since 2007-07-13), MOVE +21.5%, TLT/IEF/LQD at 52w lows, SPY -0.72, IWM -1.84, VIX +6.8% to 15.18, DXY +0.59.

## Roll seams checked first (01_seam_and_tape.py)

- CL=F -1.99% on a -4.97% opening gap, volume 76k -> 351k (contract roll). USO +3.30%. CL=F cells SKIP on seam.
- HE=F -11.23% on a -10.47% opening gap. Seam (and the feed restated yesterday's -9.25% HE bar to +1.44%). HE=F cells DEAD.
- NG=F +7.02% on a +6.61% gap; UNG +0.09%. Seam. DEAD.
- SB=F +7.05% on a +5.63% gap, volume 29k -> 82k. Seam. DEAD.
- HG=F +0.47% on a +2.09% gap, volume 900 -> 39k (December switch). HG's up-streak cell is a seam artifact. DEAD.
- SI=F -1.52% vs SLV -4.23%, PL=F -3.59% vs PPLT -4.39%: the ETFs carry the real move (SI=F switched contract, volume 88 -> 48k).
- CT=F: broken bar (open 0). DEAD.
- ES=F / NQ=F: clean (gaps -0.02 / -0.06).

## Event lane (next session = 2026-09-24)

| trigger | verdict | note |
|---|---|---|
| E:weekday_month JPY=X (Sept Thursdays up 72 of 114, BH pass) | DRILL (06) -> SKIP | hit rate without magnitude: mean +0.06%, t 1.04. Best of 12 Thursday-month cells (next best April 56.5%); all Thursdays 53%. Against the Thursday base rate sign p 0.019, which BH on 60 weekday x month cells would not keep. No mechanism (Gotobi Thursdays run the other way, -0.12%). Footnote. |
| E:weekday_month ^VIX (Sept Thursdays -0.38%, 48-65) | SKIP | t -0.63; VIX weekday cells told three nights running. |
| E:weekday_month all other subjects | SKIP | abs(t) < 1.3, edges under 0.2%. Bare weekday x month, swept. |
| E:seasonal_doy ^GSPC/SPY/QQQ/IWM (midterm 6 of 6 down, sign p 0.016 each) | DRILL (05) -> PUBLISH as decomposition | same six years for all four subjects, so one observation four times. ^GSPC: 2002 -1.38, 2006 -0.25, 2010 -0.83, 2014 -0.58, 2018 -0.04, 2022 -1.72. Pinned to the structural slot (Thursday after September quad witching) it is 12 up / 14 down 2000-2025, -0.09%, and 4 down / 2 up in midterms (2002 +1.82, 2018 +0.28). All-years calendar cell 17 of 26 down, sign p 0.084. Swept, not pre-specified. Honest "the 6-for-6 is a calendar pick" item. Anecdote-level parent; the claim rests on the N=26 structural slot, so suggestive. |
| E:seasonal_doy ^TNX (18 of 26 lower, sign p 0.038) / IEF midterm 5 of 5 | DRILL (08 C) -> KILL | my recovery of the pick dates in 08 C was broken (matched June/July sessions), so no clean decomposition; swept p 0.038 on 36 seasonal cells cannot carry anything, and today's shock owns the rates story. |
| E:seasonal_doy TLT (published 09-03) / NG=F (09-08) | SKIP | told this month; NG=F bar is a seam tonight. |
| E:seasonal_doy HYG, GC, SI, HG, CL, DX, EURUSD, JPY, VIX, EEM | SKIP | all-years sign p >= 0.08; midterm splits N=4-6 noise. SI/HG/CL seams tonight. |
| calendar, next 5 sessions (Sep 24-30) | none on macro_events | month and quarter end Sep 30 (5th session). Quarter-end rebalancing told last night; E:month_end does not fire until Sep 28. |
| post-expiry IWM lag (told 09-17) | footnote scorecard only | IWM -0.77% vs SPY +0.80% since the Sep 18 close, two sessions left. Not a nugget (repeat). |

## Price lane (asof 2026-09-23)

| trigger | verdict | note |
|---|---|---|
| P6 ^TNX / ^FVX 2-ATR up; P5b ^TNX/^FVX top-5% 21d; P4 ^FVX z10 | DRILL (02) -> PUBLISH | 10y +12bp to a 52w closing high: 20 declustered episodes since 2003. Next-session yield: 11 of 20 up overall, but 8 of 11 since 2018 (+3.5bp) vs 3 of 9 before (-1.6bp). SINGLE-ERA, so suggestive and the era flip is stated. S&P next session 8 of 20 higher, -0.32% vs +0.07% local, t -1.36; with the S&P down 0.5%+ that day (today -0.75%) 3 of 12, sign p 0.073. Derived cut, not solid. 10bp and 2.5-sd variants weaker on SPY h1 and noisier on the yield; noted. The 5% level itself was told Sunday; tonight is the jump, not the level. |
| (derived) IWM vs SPY after the 10y jump | DRILL (03, 08 A) -> PUBLISH, headline | at a 52w high: IWM trailed SPY next session 17 of 20, -0.30pp, sign p 0.001, 7 of 9 pre-2018 and 10 of 11 since; 8 of 10 when IWM had already fallen 1%+ that day. 8 of the 20 are 2022. Parent (12bp any level): 89 of 137, -0.26pp, t -4.17, both eras, GFC removed t -3.57. Caveat carried to footnote: at any yield level, when IWM already fell 1%+ and lagged SPY by 0.5pp+, 14 of 27 (mostly 2020), so the 52w-high condition does the work. |
| P5 ^MOVE top-5% week (BH pass) | DRILL (09 A) -> footnote, not a nugget | declustered 127 episodes: lower next session 81 of 127, a week later 90 of 127 (-2.19%). Single-session 12%+ weeks like today 27 of 39 lower a week later. But with the VIX under 20 the single-day jumps split 15-15 (04). Sunday told MOVE reversion (21d, calm VIX); retelling the reversion would be a near-repeat and today's calm-VIX sub-cell does not support it. |
| P6 ^MOVE 2-ATR up (+21.5%) | DRILL (04, 08 B) -> PUBLISH | third 21%+ one-day jump of 2026 (Mar 12, Mar 20), a count only 2020 matches since 2002. MOVE +12% with VIX < 20: SPY higher a week later 21 of 30, +0.56% vs +0.28% local, 70% both eras, sign p 0.021. Threshold-sensitive: 60% at 10% (+0.08%, below local), 71% at 15% (N=14). MOVE itself 15-15 a week later. Suggestive. |
| P9b SPY / TLT both down 50bp+ | SKIP (folded) | SPY h1 +0.13%, era-unstable; the 10y-jump cell is the sharper version of the same day. |
| P1 UUP first 52w high in 30+ days | SKIP | 11-11 next day. Dollar-after-FOMC told 09-16. |
| P4 DX-Y.NYB / UUP / USDSEK stretched up | SKIP | DXY t -1.49; USDSEK t -2.46 (sign p 0.045) is a peripheral cross, one of 8 kept by cap, no BH. |
| P4 EURUSD / NZDUSD / GBPUSD stretched down; P5 GBPUSD/GBPCHF; P5b NZDUSD/EURUSD/GBPUSD | SKIP | edges < 0.1%, hit ~50%. Mirror of the dollar. |
| P4 / P5b KC=F stretched down, 21d bottom | SKIP | KC=F carried flat O=H=L=C bars last week (09-18); feed. |
| P4 ^IRX, CAD=X, USDNOK (dropped by cap) | SKIP | ^IRX killed 09-17; CAD/NOK peripheral FX, null. |
| P5b yen crosses (dropped by cap) | SKIP | told 09-20. |
| P6 GBPUSD 2-ATR (dropped by cap) | SKIP | GBPUSD cells null all week. |
| P6 HYG 2-ATR down (dropped by cap) | SKIP | HYG -0.72%, credit moved with duration; LQD at a 52w low is the rate move, not spread. No drill. |
| P6 IEF / LQD 2-ATR down | SKIP (folded) | carried by the 10y cell; IEF h1 t -0.89, LQD t 0.18. |
| P6 PL=F 2-ATR down; P6 SB=F up | SKIP / DEAD | PL=F on contract switch (PPLT -4.39% real; h1 +0.14%, t 1.05). SB=F seam. |
| P3 / P3b QQQ, NQ=F, ^NDX, ^IXIC, EWJ down after a 52w high | DRILL (09 B) -> SKIP | QQQ h1 +0.18% vs +0.07% local, h5 +0.30% vs +0.38% local: no edge at a week. With a 10bp 10y jump the same day: N=2, DEAD. AUDNZD null. |
| P2 / P2b HE=F 52w low; P5 HE=F bottom week; P6 HE=F 2-ATR down (engine tag solid) | DEAD | roll seam; the solid tag_hint is not inherited. |
| P7 HG=F / CAD=X / JPY=X up streaks | SKIP | HG seam; FX streaks null. |
| P7b CL=F down streak | SKIP | CL=F roll; USO rose 3.30% today, so the streak is not real on the ETF. |
| P8 NG=F 200d cross up | DEAD | seam (UNG +0.09%). |
| P8 USDSGD 200d cross up | SKIP | 11-8, peripheral. |

## Extra price-state drills (not engine triggers)

| cell | verdict | note |
|---|---|---|
| TLT at a 52w low with SPY within 2% of its 52w high (07) | KILL | 8 declustered episodes since 2006; SPY a month later +0.24% vs +1.68% local, 5-3; IWM 3-5. Same state printed 07-31 and 09-10 this year. Anecdote with no clean record. |
| Precious metals (SLV -4.23%, PPLT -4.39%, GLD -1.80%) | SKIP | not drilled; futures on contract switch, and the metals leg adds a sixth item to a rates night. |

## Selection

Tomorrow: 1 (Sep 24 calendar decomposition). Today: 3 (IWM vs SPY after the 10y jump [headline], 10y follow-through by era, MOVE jump). 4 nuggets, no anecdote tags. Every item derived or swept, so nothing is tagged solid.
