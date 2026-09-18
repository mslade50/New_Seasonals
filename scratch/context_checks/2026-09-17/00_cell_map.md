# Cell map — run 2026-09-17 (Thu), asof session 2026-09-17, next session 2026-09-18 (Fri)

Cycle: **midterm** (year%4==2). Prices **fresh** (core bar 2026-09-17). Sweep:
1217 cells scanned, 116 fired (72 event / 44 price), BH crit p 0.0024, 3 pass.

Setup in one line: tomorrow is **September quad witching + monthly opex**, two
sessions after Wednesday's FOMC decision, in a midterm year. Today was a
post-decision relief session: SPY +1.13%, QQQ +1.73%, TLT +1.11%, VIX -12.8%
to 15.44, and the 3-month bill yield closed at a 52-week high.

Novelty state: `delta_suppressed=false`, **zero** repeat-blocked fingerprints.
Three fingerprints have published before, none inside 5 sessions:
`E:opex|HG=F|k1` (19 td), `E:seasonal_doy|TLT` (9 td), `E:seasonal_doy|NG=F`
(7 td). The last four briefs were FOMC-heavy (9/13, 9/14, 9/15, 9/16 all led
with `E:fomc_decision` cells); nothing published yet touches opex, quad
witching or the post-opex week, so the whole event surface for tomorrow is
unpublished.

---

## Event lane — calendar inside the next 5 sessions

| when | event | verdict |
|---|---|---|
| 2026-09-18 (next session) | **monthly opex** | see `E:opex` below |
| 2026-09-18 (next session) | **quad witching** | see `E:quad_witching` below |
| 2026-10-02 (11 td) | nfp | `SKIP(too far; nothing in the next 5 sessions)` |
| 2026-10-14 / 10-15 (19-20 td) | cpi / ppi | `SKIP(too far)` |
| 2026-10-21 (24 td) | vix expiry | `SKIP(too far)` |
| 2026-10-28 (29 td) | fomc decision | `SKIP(too far; just traded one)` |
| 2026-11-03 (33 td) | election | `SKIP(too far, but it is what makes the midterm split live)` |

Nothing in the window after tomorrow. Tomorrow carries the whole event lane,
which is the right shape for a quad-witching brief.

## Event lane — fired trigger groups

### `E:opex` — monthly opex on 2026-09-18, anchor = today (k1), 320 anchors

| subject | engine cell | verdict |
|---|---|---|
| `^VIX` | n=320, h1 -1.01%, 107-210 down, t -2.63, edge -1.27, era-stable, **bh_pass** | `DRILL` — the strongest thing in the sweep, but the VIX already fell 12.8% today. The interesting question is whether the opex-day decline survives arriving pre-crushed. Script 03. |
| `QQQ` | n=320, h1 -0.18%, t -2.47, 150-168 down, era-stable | `DRILL` — folded into the quad-witching work (script 01); the monthly cell mixes 3 quiet months into every witching month. |
| `HG=F` | n=312, h1 +0.27%, t 2.96, era-stable, tag solid | `SKIP(published 2026-08-20 at the same number, materially_moved=false; a copper-opex re-telling 19 td later is the countdown failure mode in a different costume)` |
| `SPY` `^GSPC` `IWM` | h1 -0.07 / -0.06 / -0.02%, \|t\| < 1.3 | `DRILL` — not as monthly opex (nothing there), but as the September-quad subset. Script 01. |
| `CL=F` | n=312, h1 +0.14%, t 1.05, era-unstable | `SKIP(no signal and it flips across 2018)` |
| `NG=F` | n=312, h1 +0.13%, t 0.76, era-unstable | `SKIP(and a natgas-opex cell published 2026-09-16 at k2; too close)` |
| `HYG` | n=233, h1 +0.06%, t 1.78, 131-97 up, p 0.033 | `SKIP(mean is 6 bp; a hit-rate tilt with no magnitude is not worth a Scott line)` |
| `JPY=X` `GC=F` `DX-Y.NYB` `EEM` `TLT` `IEF` `SI=F` `EURUSD=X` `^TNX` | all \|h1\| < 0.09%, \|t\| < 1.6 | `SKIP(no magnitude)` — DX-Y.NYB picked back up under quad witching below |

### `E:quad_witching` — quad witching on 2026-09-18, anchor = today (k1), 106 anchors

| subject | engine cell | verdict |
|---|---|---|
| `SPY` | n=106, h1 -0.175%, t -1.76, 49-57 down, era-stable | `DRILL` — base cell is weak on purpose: it pools March/June/Sep/Dec. September is the month with the reputation. Script 01. |
| `^GSPC` `QQQ` `IWM` | h1 -0.13 / -0.13 / -0.08% | `DRILL` with SPY, same script. IWM matters because small caps are the stretched leg (21d rank 7.9). |
| `^VIX` | n=106, h1 -0.95%, 42-63 down, p 0.032, era-**unstable** | `DRILL` into script 03 alongside the opex cell; the era instability is disqualifying on its own and has to be checked, not inherited. |
| `DX-Y.NYB` | n=106, h1 +0.084%, 65-41 up, sign p 0.0125, era-stable | `DRILL` — 8 bp is small, but the dollar is the live tape story (5d rank 92, USD/CHF at a 52w high) so the conjunction is worth pricing. Script 07. |
| `TLT` | n=96, h1 +0.17%, t 1.49, era-stable | `SKIP(t 1.49 on a pooled-month cell; the September subset will be N~26 and I would rather spend that N on equities)` |
| `^TNX` | n=106, h1 -0.17%, t -0.71 | `SKIP(no signal)` |
| `HG=F` `CL=F` `NG=F` `SI=F` `GC=F` `EEM` `HYG` `EURUSD=X` `IEF` `JPY=X` | \|t\| <= 1.6, mostly era-unstable | `SKIP(no signal)` |

### `E:weekday_month` — Fridays in September, 113 anchors

| subject | engine cell | verdict |
|---|---|---|
| `CL=F` | n=113, h1 -0.38%, t -1.93, 50-62 down, era-stable | `SKIP(bare day-of-week x month cell with t under 2 and sign p 0.17; this is exactly the kind of cell the 1217-cell sweep manufactures)` |
| `^TNX` | n=113, h1 +0.40%, t 1.63 | `SKIP(same)` |
| `NG=F` | n=113, h1 -0.20%, 46-66 down, p 0.045, h5 +2.07% | `SKIP(natgas published 9/13 and 9/16; the h5 ramp is the seasonal_doy cell below, and it is the same fact told twice)` |
| `^VIX` | n=113, h1 +0.07%, era-unstable, h5 +2.70% | `SKIP(h1 dead; the h5 lift is the post-opex-week story I am drilling properly in script 02, not as a Friday cell)` |
| everything else | \|t\| <= 1.4 | `SKIP(no signal)` |

### `E:seasonal_doy` — same trading day of year (+/-2), Sep 18, midterm phase

This is where the famous cell lives. h1 is a coin flip everywhere; h5 is not.

| subject | engine cell | verdict |
|---|---|---|
| `QQQ` | h5 all-years n=26, -1.12%, **18-8 down, sign p 0.038**; h5 midterm n=6, -1.87%, 5-1 down | `DRILL` — script 02. Calendar-day anchoring is the wrong axis though: the cell that matters is "the 5 sessions after September quad witching", which is a clean event anchor, not a +/-2 day-of-year smear. Re-anchor and recompute. |
| `SPY` `^GSPC` | h5 -0.90 / -0.85%, 17-9 down, sign p 0.084; midterm -1.89 / -1.90% | `DRILL` with QQQ, script 02. |
| `IWM` | h5 all-years -1.00%, 15-10; **h5 midterm n=6, -3.06%, 5-1 down** | `DRILL` — script 02 and script 06. Small caps are already at a 21d rank of 7.9, so the conjunction is live rather than decorative. |
| `NG=F` | h5 all-years **+8.87%, 19-6 up, sign p 0.0073** | `SKIP(the single loudest number in the sweep and I am still dropping it: E:seasonal_doy|NG=F published 2026-09-08, 7 td ago, and NG=F opex published 2026-09-16. Three natgas-seasonal nuggets in eight sessions is a countdown re-telling with different arithmetic. Re-open in October.)` |
| `^VIX` | h5 all-years +3.87%, 17-9 up; h1 midterm +4.68%, 5-1 up on n=6 | `DRILL` into script 03 — the mirror of the equity h5 cell, and the honest way to say the post-opex week thing twice without saying it twice. |
| `HYG` | h5 midterm 4-0 down, -0.89% | `DEAD(n=4)` |
| `TLT` `IEF` `^TNX` | h5 +0.78 / +0.13 / -1.30%, sign p >= 0.16 | `SKIP(no signal; TLT also published 9/3 at this fingerprint)` |
| `GC=F` `SI=F` `HG=F` `CL=F` `DX-Y.NYB` `EURUSD=X` `JPY=X` `EEM` | all sign p >= 0.10 on h1 and h5 | `SKIP(no signal)` |

---

## Price lane — fired trigger groups (lane is live, prices fresh)

| trigger | subject(s) | verdict |
|---|---|---|
| `P1:new_52w_high` | `CHF=X` n=17, h1 -0.30%, **3-14 down**, t -4.16, sign p 0.0064, era-stable | `DRILL` — USD/CHF printed its first 52w high in 30+ days and the next session has faded it 14 of 17 times. Small N, real magnitude, live tape. Script 07. |
| `P2 / P2b:new_52w_low` | `HE=F` (lean hogs) | `DRILL-then-likely-kill` — HE=F is down 11.5% today and sitting on a 52w low, but continuous lean-hog futures roll gaps fire price triggers as fake moves. Verify before believing any of it. Script 08. |
| `P3:drop50_after_high` `P3b` | `^FVX` `^TNX` | `SKIP(\|t\| < 0.4 on every variant; yields reversing off a high says nothing)` |
| `P3c:pop50_after_low` | `IEF` n=14, 8-6 up | `DEAD(n=14 and a 57% hit rate is nothing)` |
| `P4:z10_extreme` (down) | `KC=F` n=90 t 1.46; `NZDJPY=X` n=133 t 0.53 | `SKIP(coffee and a yen cross with no US-session relevance tomorrow)` |
| `P4:z10_extreme` (up) | `^IRX` n=244, h1 +1.40% but edge **-1.48** | `DRILL` — the printed cell is unusable (the all-days control is enormous because IRX drifts), but the underlying fact is the live one: the 3-month bill yield closed at a 52-week high with z10 2.54 while the 10-year FELL 1.2% today. Recompute properly. Script 05. |
| `P4:z10_extreme` (up) | `USDMXN=X` n=155, t -0.15 | `SKIP(no signal)` |
| `P5:rank5_extreme` (bottom) | `HE=F` `KC=F` `^VVIX` `NZDUSD=X` | `SKIP(softs and FX noise; \|t\| <= 1.9, and HE=F is the roll-gap suspect)` |
| `P5:rank5_extreme` (top) | `USDNOK=X` `USDMXN=X` `CAD=X` `CHF=X` | `DRILL` as a dollar-breadth conjunction only, not one at a time. Script 07. |
| `P5b:rank21_extreme` (bottom) | `KC=F`, and the yen crosses `NZDJPY` `EURJPY` `GBPJPY` `CHFJPY` | `SKIP(the yen-cross cells are real but tiny: EURJPY 195-148 up, sign p 0.0064, mean +0.084%. A 8 bp mean is below the noise floor of anything Scott can use as context, and four correlated crosses is one cell told four times.)` |
| `P5b:rank21_extreme` (top) | `^IRX` (rolled into script 05), `ZC=F` n=432 t 2.10, `^FVX` t -0.31 | `SKIP` for corn (no relevance) and `^FVX` (no signal) |
| `P6:two_atr_day` (down) | `HE=F` t -2.83 tag solid; `CC=F` n=88 60% up; `KC=F` | `SKIP(softs; HE=F handled in script 08)` |
| `P6:two_atr_day` (up) | `ES=F` n=15, 9-6 up, t -0.78 | `DEAD(n=15 on continuous futures with roll contamination, and no signal anyway)` — the underlying "today was a big up day" fact is drilled on SPY cash in script 04 instead |
| `P7:up_streak` | `USDSEK=X` n=138, h1 -0.19%, **52-86 down, t -2.61, sign p 0.0024, era-stable, bh_pass** | `DRILL` — one of only three BH survivors in 1217 cells. Script 07. |
| `P7:up_streak` | `USDTRY=X` n=426, 302-124 up, bh_pass | `SKIP(degenerate: the lira devalues monotonically, so "up streaks continue" is the trend itself, not a finding. era_stable=false confirms it.)` |
| `P7:up_streak` | `USDNOK=X` `CAD=X` `CHF=X` | `SKIP` individually, `DRILL` as dollar breadth, script 07 |
| `P7b:down_streak` | `EURUSD=X` n=148, 89-58 up, sign p 0.0084, mean +0.068% | `DRILL` as the other side of the same dollar coin, script 07 |
| `P7b:down_streak` | `HE=F` `AUDUSD=X` `KC=F` | `SKIP(no signal / roll suspect)` |
| `P8:sma200_cross` | `GBPUSD=X` n=21, 10-11, t -0.63 | `SKIP(no signal; the cable 200d break is the dollar story again and script 07 owns it)` |
| `P9:stocks_bonds_up` | `SPY` n=314, h1 +0.034%, edge -0.005; `TLT` +0.025% | `DRILL` — the printed cell is a nothing, which is itself informative, but the 50 bp threshold is far too loose for what actually happened today (SPY +1.13% AND TLT +1.11% AND VIX -12.8%). Tighten it. Script 04. |

**Triggers that did NOT fire**, checked so the absence is on the record:
`P10/P10b/P10c` (VIX term structure inversion, VIX +10%) — the term structure
did the opposite today, VIX3M/VIX widened to 1.20 on a 12.8% VIX collapse, and
there is no `VIX -10%` trigger in the inventory. That is a genuine gap in
`PRICE_TRIGGERS` and it is the single most characteristic thing about today's
tape, so script 04 computes it by hand. `P11/P11b` (breadth crossing 80%/20%
above the 200d) — breadth sits at 57.5%, down from 64.4% 21 sessions ago, well
inside both rails. `P12` (macro print vs consensus) — zero US releases today.

---

## Engine hints I am not inheriting

- `tag_hint` downgrades taken: `E:opex|^VIX|k1` arrives `solid` and I will not
  publish it as `solid` without checking whether the pre-crushed condition
  holds (script 03). `E:opex|HG=F|k1` arrives `solid` and is being dropped
  entirely on novelty, not on statistics.
- `bh_pass` provenance: the three BH survivors are `E:opex|^VIX|k1`,
  `P7:up_streak|USDSEK=X` and `P7:up_streak|USDTRY=X`. Of the cells I plan to
  publish, **the September post-quad-witching week and the quad-witching
  session itself are pre-specified famous hypotheses** — the "week after
  September opex is the worst week of the year" claim predates this sweep by
  decades and was not found by it, so it does not owe the multiplicity
  correction. Every other publishable cell either carries `bh_pass` or will be
  tagged at `suggestive`/`anecdote` where the correction is not load-bearing.

## Drill queue

| # | script | question |
|---|---|---|
| 01 | `01_sept_quad_witching.py` | the quad-witching session itself, September only, all years + midterm, vs all quad witchings and all Fridays |
| 02 | `02_post_quad_week.py` | the 5 sessions after September quad witching: SPY/QQQ/IWM, era split, concentration, midterm subset |
| 03 | `03_vix_opex.py` | VIX into opex when it arrives already crushed; VIX through the post-opex week |
| 04 | `04_everything_rally.py` | SPY +1% and TLT +1% on the same session, with and without a 10%+ VIX collapse |
| 05 | `05_front_end_52w_high.py` | 3-month bill yield at a 52w high while the 10-year falls |
| 06 | `06_iwm_stretched_into_opex.py` | IWM 21d rank in the bottom decile going into September opex week |
| 07 | `07_dollar_squeeze.py` | USDSEK streak, USD/CHF 52w high, dollar breadth, and the quad-witching DXY cell |
| 08 | `08_he_roll_check.py` | is the lean-hog collapse a contract roll |

---

## What the drills returned (written after running them)

| # | script | outcome |
|---|---|---|
| 01 | `01_sept_quad_witching.py` | The pooled witching cell is weak because it pools months, and splitting by month does NOT rescue it: September's own session is 11-15 up at t -0.80. What it DID surface is the conjunction that matches tomorrow exactly. A September witching landing two sessions after an FOMC decision has happened 6 times (2013, 2014, 2019, 2020, 2024, 2025) and IWM fell on all six, mean -0.66%, t -3.60, every one between -1.27% and -0.21%. SPY 1-5, -0.35%. **PUBLISH as an anecdote (n=6).** Separately, the September witching session is 1-7 up across 2018+ on all four equity subjects against 10-8 up before, which is the era note that belongs inside that item rather than a second nugget, since four of the six FOMC dates sit in it. |
| 02 | `02_post_quad_week.py` | The core of the brief. The 5 sessions after September quad witching: SPY 7-19 up, -0.90%, sign p 0.0145, against +0.19% for every 5-session window; IWM 6-20 up, -1.79%, t -2.89; QQQ 9-17, -0.90%. The relative version is cleaner still: **IWM minus SPY -0.89pp, IWM ahead only 6 of 26, t -3.44, sign p 0.0047**. December is the mirror (SPY +0.74% 17-9 up, IWM +0.98% 19-7 up), which is what makes this a September effect rather than a witching effect. VIX +5.92%, 18-8 up. **PUBLISH (headline + index item).** |
| 03 / 03b | `03_vix_opex.py`, `03b_vix_opex_cross.py` | The BH-surviving opex VIX cell survives its own hardest condition. When the VIX arrives already down 8%+ it falls AGAIN on the opex session 20 of 25, mean -2.71%, median -3.11%, sign p 0.002. The level split looked like a contradiction (the 15-17 bucket is flat at +0.33%) and 03b resolved it: that flatness is the no-big-drop cases. Restricted to anchors below a 20 VIX it is 13 of 16 down, median -3.04%, sign p 0.011. Era honesty: 16 of 18 down pre-2018, only 4 of 7 down since. **PUBLISH with the era caveat.** |
| 04 | `04_everything_rally.py` | **KILLED.** `P9:stocks_bonds_up` tightened to 1% on both legs plus an 8% VIX collapse gives n=19, SPY next session 10-9 up, 5 sessions out 12-7 up with the mean wrecked by 2008-11-04. Only 1 of the 19 followed an FOMC decision. No claim survives. Also confirmed the trigger inventory has no `VIX -10%` rule, only the +10% side, which is why the most characteristic feature of today's tape fired nothing. |
| 05 | `05_front_end_52w_high.py` | **KILLED.** ^IRX first 52w high in 30+ td is n=16 with nothing at any horizon; the broader "at a 52w high" version flattens the 10y-3m spread a further 8.5 bp over 21 sessions (47 of 72, sign p 0.0064) but that is mechanical and has no bearing on tomorrow. The conjunction with a 1%+ fall in the 10-year has exactly ONE precedent (2019). |
| 06 | `06_iwm_stretched_into_opex.py` | **KILLED as its own nugget, kept as a sharpener.** The post-witching decline does not care how IWM arrives: beaten up -1.33% (n=4), middle -2.33% (n=11), strong -1.60% (n=5). And a bottom-quartile 21d rank is normally a mild POSITIVE for IWM's next week (+0.35% against +0.22% unconditional), so this one week runs against IWM's usual oversold behaviour. That contrast goes in the headline body. |
| 07 / 07b | `07_dollar_squeeze.py`, `07b_dollar_breadth.py` | 07 passed a return series into `pct_rank`, which takes prices, so every breadth count came back zero; 07b fixed it. **A BH survivor died here**: USDSEK's 5-day up streak is 52-86 down undeclustered, but declustered at 5 td it is 38-49 with sign p 0.142, and the 2018+ mean is -0.06% against -0.27% before. Overlapping streak days were doing the work. USD/CHF's 52w high is n=14, 3-11 down, sign p 0.029, and 0-6 since 2018, which is real but too thin beside better material. What survived is **DXY on the September witching session: 19-7 up, +0.188%, t 2.30, sign p 0.014**, and September is the only witching month that carries it. Its live caveat is sharp: the two years the dollar arrived with a 5-day percentile above 75 it fell both times, and today it is at 92.1. **PUBLISH with that caveat.** |
| 08 | `08_he_roll_check.py` | **KILLED, and the kill is publishable.** HE=F did not fall 11.53%. It opened at 69.625 against a 78.675 prior close, a -11.50% gap, and then traded -0.04% intraday inside a 1.37% range on ordinary volume. That is the October contract rolling to December. All five HE=F triggers today (P2, P2b, P5, P6, P7b) are artifacts. Goes in as a tape note outside the numbered items. |
| 09 | `09_rarity_checks.py` | Half kept. All seven dollar pairs closed with a 5-day return in the top 5% of their year, which has happened on 9 sessions since 2000, and six of the eight priors are one week in March 2020. No forward claim is legitimate on 3 distinct episodes and the numbers agree (DXY 5-3 up next session, 2-6 up over five). **PUBLISH as a rarity with the null stated.** The whole-curve-at-52w-highs cell is n=12 after declustering and says nothing; **killed**. |
| 10 | `10_qqq_led_relief.py` | **KILLED.** QQQ beating IWM by 100 bp with both legs up is n=200 declustered and the forward spread is a null (IWM ahead 104 of 200 at h=5). The narrower version with a VIX collapse looked live (IWM down 43 of 70 the next session, -0.87%) but its dates are 2000-2002, 2008 and 2020, and the era split flips (+0.61pp pre-2018, -0.11pp since). Bear-market relief-rally artifact. |
| 11 | `11_headline_robustness.py` | The headline holds everything thrown at it. IWM minus SPY: pre-2018 -0.76pp, 2018+ -1.17pp with IWM ahead 0 of 8, so the sign does not flip and the modern era is the stronger half. Top two episodes carry 29% of the total; drop them and it is still -0.68pp at t -2.93, 6 of 24, sign p 0.011. SPY outright drop-two gets WORSE, -1.01%, 6-18 up. The VIX mirror drop-two is +3.92%, 16-8 up. The midterm subset does NOT sharpen it (n=6, -0.50pp, IWM ahead 2 of 6) and the brief says so rather than quoting the better-looking pre-election cell. |

## Final slate

Six nuggets, two anecdotes (the FOMC-witching conjunction and the dollar
rarity), headline is `suggestive`. Tomorrow's lane leads because a top-tier
scheduled event owns the next session. The lean-hog roll note ships as a
non-lane `## Tape note` section so it does not consume a third anecdote slot.

Multiplicity: the post-September-opex-week cell is the pre-specified famous
hypothesis and does not owe the sweep's BH correction; it is tagged
`suggestive` on N anyway. The opex VIX cell carries `bh_pass` in its own
right. The DXY witching cell and the FOMC conjunction are swept cells tagged
at `suggestive` and `anecdote`, where the correction is not load-bearing, and
neither is claimed as `solid`.
