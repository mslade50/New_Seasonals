# Cell map — run 2026-08-16 (Sunday)

- asof session: 2026-08-14 (Friday), prices fresh, core bar 2026-08-14
- next session: 2026-08-17 (Monday), midterm year, 11th td of August, 10 td to month end
- 1,213 cells scanned, 88 fired (54 event / 34 price), BH crit p 0.009, 7 pass
- No top-tier event in the window. VIX expiry Wed 8/19 (3 td), monthly opex Fri 8/21 (5 td).
  So the next session is the Monday of expiry week, which is what the whole event lane is about tonight.
- Novelty: all 88 fingerprints are new (`times_published` 0 everywhere); the last five sessions
  published custom drill fingerprints, none of which match a base cell tonight. `delta_suppressed` false.
- Thematic self-policing, since the flags cannot see it: corn/grains (8-12), Brazil/LatAm (8-12),
  SKEW (8-11 and 8-13), TLT August and TLT-at-52w-low (8-10, 8-13), SPY joint 52w highs (8-13),
  VIX August Fridays (8-13). Those get a higher bar than the flag state imposes.

## Event lane

| trigger | verdict |
|---|---|
| `E:vix_expiry` k3, 18 subjects | **DRILL** (02). Anchor is Friday's close, so h1 is Monday of expiry week. QQQ 188-129 up, sign p 0.0008, bh_pass; SPY 185-133; ^GSPC 182-137. The August subset is far stronger: QQQ n=26 mean +0.70% hit 80.8% t=3.19, SPY +0.43% 76.9% t=2.29. ^VIX +1.04% t=2.47 across all months but the August subset is -1.55% and 30.8% up. Confounded with the plain Monday cell by construction; the drill has to separate them or neither publishes. |
| `E:weekday_month` Mondays in August, 18 subjects | **DRILL** (01). QQQ 76-40 up, sign p 0.0005, bh_pass, era-stable 68.4%/59.5%, while SPY on the same 116 anchors is 63-51 and +0.007% mean. A hit-rate cell with no mean, which is exactly the thing that needs its magnitudes shown. ^VIX +2.35% t=2.27, the mirror image of the August Friday cell published 8-13, so it needs to add specificity or stay out. |
| `E:seasonal_doy` Aug 17 ±2, 18 subjects | **DRILL** (03 equities, 04 gold). ^GSPC midterm 6-0 up, mean +1.11%, sign p 0.0156; QQQ 5-1 +1.67%; IWM 5-1 +1.26%. N=6, anecdote tier at best, and it sits inside a seasonal window that is famously weak later, so the drill has to check the window around it. GC=F all-years h5 20-5 up, +0.71%, sign p 0.002, which is the cleanest seasonal number on the board. |
| calendar: VIX expiry Wed 8/19 | **PUBLISH** via 02, and in the calendar block. |
| calendar: opex Fri 8/21 | **PUBLISH** in the calendar block; the week framing carries it. |
| calendar: Jackson Hole Fri 8/28, NFP 9/4, CPI 9/11 | **SKIP** as nuggets (10, 15 and 19 td out, nothing in the window makes them tomorrow's business). Calendar block only. |
| calendar: NFP 8-7, CPI 8-12, PPI 8-13 | **SKIP**, already printed and already covered in the 8-10 through 8-13 briefs. |

## Price lane

| trigger | subject | verdict |
|---|---|---|
| `P1:new_52w_high` | ZC=F | **SKIP(published 8-12)**. Corn's 52-week high was the 8-12 headline. Tonight's +7.98% is a new bar but the same cell, and h1 is 10-12 down with edge -0.02%. |
| `P1:new_52w_high`, `P1b` | EURCHF=X | **SKIP(no macro read)**. A euro-Swiss cross at a 16-observation high with a 0.06% h1 and no era stability is not a macro fact. |
| `P4:z10_extreme` up | GC=F | **DRILL** (04). z10 2.18 and 21d rank 92, and the base cell is flat at h1 (+0.001%) and negative at h5 (-0.20%) against a seasonal window that is the year's best. The contradiction is the nugget. |
| `P4:z10_extreme` up | USDTRY=X | **SKIP(mechanical)**. A managed devaluation makes every momentum cell look like a solid: 401-177 up at h1. It measures the crawl, not a state. |
| `P4:z10_extreme` up | SB=F, GBPCHF=X, EURCHF=X, USDHKD=X | **SKIP(no edge, no relevance)**. Edges of -0.05% to -0.004%, none era-stable, none macro subjects. |
| `P5:rank5_extreme` bottom | HE=F | **DRILL** (06), folded with P6 and P8. A -14.38% session is the tape's whole story tonight. |
| `P5:rank5_extreme` bottom | ^MXX | **SKIP(covered 8-12)**. LatAm weakness went out as the Brazil divergence nugget four sessions ago; Mexico is the same trade with a smaller N. |
| `P5:rank5_extreme` bottom | LE=F | **SKIP(same complex)**. Live cattle is the hog story's neighbour, not an independent one; edge -0.01%. |
| `P5:rank5_extreme` top | ZW=F, ZC=F | **SKIP(published 8-12)**. Grains again. Wheat's h1 record is 161-175 down against a +0.24% mean, which is a mean carried by tails, not a fact worth a line. |
| `P5:rank5_extreme` top | USDTRY=X | **SKIP(mechanical)**, as above. |
| `P5b:rank21_extreme` bottom | HE=F | folded into 06. |
| `P5b:rank21_extreme` bottom | CHFJPY=X | **SKIP(no edge)**. 188-148 up at +0.06%, edge 0.04%. |
| `P5b:rank21_extreme` top | ^GDAXI, EURCHF=X, SB=F, USDTRY=X | **SKIP(no edge)**. Best of them is the DAX at -0.002% h1 with a negative edge; a 21-day top-5% state that predicts nothing. |
| `P6:two_atr_day` down | HE=F | **DRILL** (06). n=115, h1 -0.47%, t -2.77, era-stable, tag hint solid. |
| `P6:two_atr_day` down | USDCNY=X, LE=F | **SKIP**. CNY's 2-ATR move is 0.165% and the cell has a 0.92 sign p; cattle as above. |
| `P6:two_atr_day` up | ZC=F | **SKIP(published 8-12)**. |
| `P7:up_streak` | ^NYA | **DRILL** (07). Five up closes into a 52-week high, and it is the broad tape index, so it can carry the breadth question the tape block raises (65.5% above the 200d against 70.1% twenty-one sessions ago). |
| `P7:up_streak` | USDTRY=X, AUDJPY=X | **SKIP**. Mechanical and 55.9% with a 0.06% edge respectively. |
| `P7b:down_streak` | BTC-USD | **DRILL** (05). Five down closes, 35.2% under its 52-week high, and crypto has not been in a brief. N=66 base cell, 41-25 up, sign p 0.032. |
| `P7b:down_streak` | ^BVSP | **SKIP(published 8-12)**. |
| `P7b:down_streak` | ^FTSE | **SKIP(no edge)**. -0.078% h1 on 119, edge negative. |
| `P7b:down_streak` | USDMXN=X | **SKIP(no edge)**. -0.002% h1, sign p 0.44. |
| `P8:sma200_cross` down | HE=F | folded into 06. |
| `P8:sma200_cross` up | USDBRL=X | **SKIP(N and relevance)**. 16 observations, -0.06% h1, and Brazil is covered. |

## Hints I am not inheriting

- `tag_hint` **solid** on `P4:z10_extreme|USDTRY=X` and `P6:two_atr_day|HE=F` is downgraded. USDTRY is dropped
  outright. Hogs will publish at whatever its own drill supports, which given the anchor count is anecdote or
  suggestive, not solid.
- `bh_pass` true on `E:vix_expiry|QQQ|k3`, `E:vix_expiry|SPY|k3`, `E:vix_expiry|^GSPC|k3`,
  `E:weekday_month|QQQ`, `P4:z10_extreme|USDTRY=X`, `P4:z10_extreme|GC=F`, `P7:up_streak|USDTRY=X`.
  The two that matter tonight are **pre-specified, not swept**: the Monday effect and expiry-week drift are
  named hypotheses that predate this engine, so they neither need the sweep correction nor get credit from it.
  Everything else in the price lane is a swept cell and is tagged accordingly.

## Stale and missing

- `^GSPTSE` absent from master_prices; `LBS=F` stopped printing 2023; `^AXJO`, `^HSI`, `^KS11`, `^N225` have no
  2026-08-14 bar. None are subjects tonight, and foreign cash indices are excluded from the event lane anyway.

## Drill list

1. `01_august_monday.py` — August Mondays, QQQ against SPY, against all Mondays, against August non-Mondays.
2. `02_expiry_week_monday.py` — is the expiry-week cell anything beyond a Monday, and what does August do to it.
3. `03_midterm_doy.py` — the ^GSPC 6-0 midterm doy record, and the window it sits inside.
4. `04_gold_seasonal_stretch.py` — gold's mid-August seasonal, conditioned on entering stretched.
5. `05_btc_downstreak.py` — five down closes in bitcoin while the US index sits at a high.
6. `06_hogs.py` — a -14.4% session, where it ranks and what followed.
7. `07_nya_streak_breadth.py` — five up closes into a 52-week high with breadth already narrowing.
8. `08_verify_brief_numbers.py` — every number that goes in the brief, printed once, transcribed not remembered.

## Verdicts after the drills ran

- **01 / 02 / 02b PUBLISH.** The August expiry-week Monday is a real interaction and not a
  restatement of either parent: SPY 20-6 at +0.43% (t 2.29, sign p 0.0047) against 57.8% for
  expiry-week Mondays in all months and 43-47 for August's other Mondays. VIX inverts on the same
  session, 18 of 26 lower against 584-395 up on every other Monday. Both published.
- **03 KILLED, and the engine's own flag with it.** Reconstructed one read per year, the ^GSPC
  midterm doy cell is 4-2 at h1, not 6-0, and 73% of that sits in 2010. The 42-session window
  behind it is -2.28% and 2-4 in midterms, but 2022 at -16.8% IS the effect: the other five
  average +0.6%. Nothing publishable. The stronger midterm read is inside nugget 1 (6-0 on the
  index and IWM at the expiry-week anchor), a different and better-populated cell.
- **04 PUBLISH.** Gold's mid-August window is 19-6 at h10 (sign p 0.0073) and the 15 prior
  already-stretched August entries are 7-8 and flat. The contradiction is the item.
- **05 / 05b PUBLISH as [anecdote].** 11 episodes across 8 years, 1-10 at h21. Also caught a
  definitional trap worth remembering: the tape block's `dist_52w_high_pct` is a 252-ROW window,
  which on a 7-day-a-week series is eight months. Bitcoin is 35.2% below its 252-row high and
  49.7% below its true 12-month high. The brief quotes the 12-month number and labels it.
- **06 / 06b KILLED, and this one generalises.** The HE=F -14.38% "session" is a front-month ROLL
  in the continuous series: overnight gap -14.06%, intraday -0.37%, range 0.9%. All 43 sessions
  at or below -8% in HE=F land mid-month in a lean hog contract month (21 in August, 13 in
  October), mean gap -13.57% against mean intraday -0.69%. CL=F, which rolls without a gap of
  that size, spreads its -8% days across all twelve months. So FOUR of tonight's fired price
  triggers on HE=F (P5, P5b, P6, P8) are one data artifact, and `P6:two_atr_day|HE=F` arrived
  carrying a `solid` tag hint. LE=F shows the same signature on 2 sessions, both May 1.
- **07 PUBLISH, reframed.** The breadth conditioning collapsed: on an equity-only panel the
  narrowing is 19 of 21 names against 20 three weeks ago, and only 3 prior episodes match, so the
  divergence framing was dropped from the brief. What survived is the streak null: 5+ up closes
  are worth +0.79% a month out (88-44, t 2.68) unless they end at a 52-week high, where 51
  episodes give +0.11% against +0.52% unconditional.
- Reconciled the engine's breadth number on the way: `pct_above_sma200` is computed over ALL 98
  subjects, FX pairs and VIX included, despite `breadth_series`'s docstring saying "equity-index +
  sector panel". 65.5 / 70.1 reproduces exactly on the 98-name panel; equity-only reads 90.5 / 95.2.
