# Cell map, run 2026-09-22 (Tuesday)

- asof session 2026-09-22 (Tue), next session 2026-09-23 (Wed), midterm year, September
- prices_fresh = true (core bar 2026-09-22). Warnings: LBS=F, ^AXJO, ^HSI, ^KS11, ^N225, ^SKEW no 09-22 bar (Asia closed/holiday or dead feed; none are nugget candidates tonight)
- sweep: 1181 cells scanned, 75 fired (36 event / 39 price), 1 BH pass (NQ=F P7 up-streak) at crit p 0.0017
- dropped_by_cap: P5 JPY=X (drilled, 07), P5b GBPJPY=X / NZDJPY=X (yen crosses told Sunday, SKIP)
- no top-tier event in the next session; price triggers fired, so NOT a quiet tape

## Roll seams checked first (01_seam_checks.py)

- CL=F -6.19% = -4.00% opening gap (October contract expired today) + -2.28% in the bar. USO -2.75%. Every CL=F cell tonight re-measured on USO.
- HE=F -9.25% = -9.66% opening gap, +0.46% in the bar. Seam. DEAD.
- NG=F +11.50%: bar is broken (low 2.982 above open 2.830). UNG +5.85%. NG=F cells DEAD.
- HG=F / SI=F / PL=F / PA=F: continuous contract switched to December today (HG volume 739 -> 41,940; SI 97 -> 49,787; PL 2 -> 25,356). SI=F +2.71% vs SLV +1.84%, PL=F +2.24% vs PPLT +1.84%. No copper ETF in the cache, so HG's +3.27% cannot be split into move and roll. Metals cells SKIP on seam.
- ES=F / NQ=F: clean today (gaps -0.05% / -0.02%), but NQ=F's five-day streak includes Monday's +1.10% roll gap into December.

## Event lane (next session = 2026-09-23)

| trigger | verdict | note |
|---|---|---|
| E:weekday_month ^VIX (Sep Wednesdays -1.13%, 43-69) | DRILL -> SKIP | 08/09: survives removing FOMC days and the post-Labor-Day Wednesday (-1.20%, 24 of 69 up), but it sits in the FIRST HALF of September: the two Wednesdays before the expiry run -2.41% and -1.52%. Tomorrow's slot, the Wednesday after quad witching, is 13-13 at -0.16%. Does not apply to tomorrow. Footnote. Also third straight night of a VIX weekday cell. |
| E:weekday_month CL=F (+0.56%, t 2.44) | SKIP | continuous-contract cell; September Wednesdays straddle the CL roll, and USO carries tonight's crude item. No mechanism beyond EIA day, which fires every week. |
| E:weekday_month all other subjects | SKIP | \|t\| < 2, edges under 0.25%. Bare weekday x month, swept. |
| E:seasonal_doy ^GSPC/SPY/IWM/QQQ (Sep 23: ^GSPC 19 of 26 down) | DRILL (10) -> PUBLISH as decomposition | post-expiry-week weakness was told Thu/Sun/Mon at the 5-session and Monday level; tonight's new cut is the Wednesday itself cleaned of FOMC days: S&P up 4 of 17, -0.32%, sign p 0.025, both eras same sign. New cell, new number, not a countdown. |
| E:seasonal_doy TLT/IEF/^TNX (TLT 17 of 23 up, 10y 19 of 26 down) | DRILL (03, 10) -> KILL, folded into the S&P item | 9 of the 24 TLT slots were FOMC decision days and all 9 rose (+0.88%); ex-FOMC TLT is 9 of 15 at +0.08%, 1 of 4 since 2018. Tomorrow is not an FOMC day. The late-September run to month-end flips at 2018 (11 of 16 up before, 2 of 8 since). |
| E:seasonal_doy HYG (h5 13 of 18 down) | SKIP | sign p 0.08, credit ETF only since 2007, no mechanism. |
| E:seasonal_doy NG=F (h5 +6.9%) | SKIP | natural-gas seasonality already run twice this month (Sep 8 and Sep 16 opex); NG=F bar broken tonight. |
| E:seasonal_doy GC/SI/HG/CL/DX/EURUSD/JPY/VIX/EEM | SKIP | sign p >= 0.07 all-years; midterm splits are N=5-6 noise. |
| calendar, next 5 sessions (Sep 23-29) | none on macro_events | month-end/quarter-end is Sep 30 (td+6); the final six sessions of the quarter start tomorrow -> quarter-end rebalancing drill (04), pre-specified hypothesis (pension/balanced-fund rebalancing), not a sweep find, owes BH nothing. |
| quarter-end rebalancing (04) | DRILL -> PUBLISH | QTD SPY-TLT gap +8.13pp (69th pct of 95 quarter-ends). Gap >= 8pp: 29 cases, SPY-TLT -0.75% over the final six sessions, SPY ahead in 12 of 29. Pre-2018 5 of 17 (-1.47%, sign p 0.072); 2018+ 7 of 12 (+0.28%). Rank corr -0.24 / -0.22 by era, so the continuous tilt survives weakly while the top bucket flipped. Pearson 2018+ -0.67 is March 2020 (+16.2%), not reported. |

## Price lane (asof 2026-09-22)

| trigger | verdict | note |
|---|---|---|
| P1 / P1b ^NDX, QQQ first 52w high in 30+/90+ days | SKIP | told last night on ^IXIC (first high in 111 days, drought split). NDX lagged a day; same event. |
| P2 / P2b HE=F new 52w low | DEAD | roll seam, -9.66% opening gap. |
| P3 / P3b ZC=F drop after 52w high | SKIP | h1 -0.24%, t -1.08; nothing. |
| P4 EURUSD, NZDUSD stretched down | SKIP | h1 edges < 0.1%, hit ~50%. |
| P4 HE=F stretched down | DEAD | seam. |
| P4 ^IRX stretched up | SKIP | 3-month bill at a 52w high drilled and killed Thursday. |
| P4 CAD=X, USDMXN stretched up | SKIP | era-unstable, hit 45-55%. |
| P5 HE=F bottom 5% | DEAD | seam. |
| P5 CL=F bottom 5% (5d) | DRILL (02) on USO -> PUBLISH | USO -10.98% over 5 sessions, five straight down closes, 28.6% above its 200d. 5d <= -10% while above the 200d: 12 episodes since 2006, 5 in 2026 (today the 6th), only 2 before 2018. h5 7 of 12 up +2.30% (local +0.40%), h10 9 of 12 up +3.33% sign p 0.073, h21 6 of 12. Anecdote (N=12). Top two carry 73% of h5. Also corrects Sunday's CL=F -6.32% claim onto a clean basis. |
| P5 BTC-USD top 5% (5d) | SKIP | told last night, declustered. |
| P5 ETH-USD, HG=F, NQ=F, QQQ, ^NDX top 5% | SKIP | ETH/QQQ/NDX/NQ h1 null; HG seam. |
| P5 JPY=X top 5% (dropped by cap) | DRILL (07) -> KILL | 139 declustered: h1 -0.31% but top two (Oct/Dec 2008) = 92%; the 63d-low snap cut (N=14) shows nothing, 2 since 2018. |
| P5b KC=F bottom 5% (21d) | SKIP | KC=F carried flat duplicate bars this week (Sunday/Monday notes); feed. |
| P5b EURJPY, CHFJPY, CADJPY, NZDUSD, GBPUSD bottom 5% | SKIP | yen crosses told Sunday; h1 edges < 0.1%. |
| P5b ^IRX, AUDNZD top 5% (21d) | SKIP | ^IRX killed Thursday; AUDNZD null. |
| P6 NG=F 2-ATR up | DEAD | broken bar (low above open); UNG +5.85% is real but NG told twice this month. |
| P6 HE=F 2-ATR down | DEAD | seam (engine tag_hint solid, not inherited). |
| P7 QQQ / NQ=F / ^NDX 5+ up closes (NQ=F BH pass) | SKIP | BH pass is the sign test on 353 overlapping days; the mean edge is +0.03% (NQ) / +0.05% (QQQ), and NQ=F's streak includes Monday's roll gap. Hit rate without magnitude. |
| P7 HG=F, CAD=X, JPY=X up streak | SKIP | HG seam; FX null. |
| P7b CL=F 5+ down closes | folded into USO item (02) | measured on USO. |

## Extra price-state drills (not engine triggers)

| cell | verdict | note |
|---|---|---|
| VIX -4.44% to 14.21 on an unchanged S&P (06, 08) | DRILL -> PUBLISH, tagged suggestive, with its fragility stated | VIX -4%+ on an S&P close <= 0 with the VIX under 16, declustered 5td: 22 since 2004, S&P lower a week later in 14 of 22, -0.79% vs +0.23% local, t -2.37, both eras negative; VIX higher in 15 of 22. Cut-fragile: a 3% drop gives 50 cases at -0.16%. Swept/derived, not BH-eligible, never solid. |
| NDX 52w high on a flat/red S&P, XLF -1.97% (05) | KILL | 115 of 743 NDX high days had the S&P red; 14 declustered episodes of NDX +0.5% at a high with S&P <= 0, S&P h5 6 of 14 up, top two = 73%. XLF is a sector ETF, never a subject. |

## Selection

1. Quarter-end rebalancing, SPY vs TLT (tomorrow, suggestive, pre-specified) -> headline
2. Wednesday after September quad witching, S&P ex-FOMC + TLT decomposition (tomorrow, suggestive)
3. VIX drop on an unchanged S&P under 16 (today, suggestive)
4. USO five-session 11% slide (today, anecdote)
