# Cell map — run date 2026-08-25 (Tuesday)

asof session 2026-08-25 | next session 2026-08-26 (Wednesday) | midterm year
prices_fresh = True, core bar 2026-08-25. Full price lane live.
sweep: 1213 cells scanned, 92 fired (54 event / 38 price). BH crit p 0.0038, 4 pass.
Capped: P5b:rank21_extreme dropped DX-Y.NYB, UUP, EURUSD=X. DXY's 21d bottom-5%
state was published 2026-08-23, so the cap costs nothing tonight.

Novelty: no fingerprint is repeat_blocked, delta_suppressed = false. But the
2026-08-24 brief spent nuggets on corn's August gap, gold's 21d +15%, BTC z10,
QQQ-vs-NYA and the final-five-of-August cell. Those states are all still live
tonight and are treated as SPENT unless a drill produces a genuinely different
cell.

## Data integrity first (00_integrity.py, 01_grain_roll.py)

The price lane arrives dominated by commodities. Four of the loudest fired
cells fail a bar check or a roll check:

- **CT=F — DEAD.** Tonight's bar is Open 0.00, High 82.90, Low 81.96,
  Close 88.04. The close prints 6.2% above its own high and the open is zero.
  Both P1:new_52w_high and P1b:new_52w_high_90 fired on that bar and both are
  artifacts. Cotton was dropped for the same class of fault on 2026-08-24.
- **KC=F — SKIP(integrity).** The -11.36% session is a -9.25% overnight gap
  plus -2.32% intraday. Coffee's >3% gaps cluster in Mar/May/Jul/Sep/Dec, its
  contract months, and the 2026-08-21 bar closes below its own low. An
  unverifiable double-digit move is not worth a nugget.
- **LE=F — SKIP(roll).** The -4.20% session is a -2.99% gap. Ten of 26 years
  carry a >3% gap in the Aug15-Sep05 window and every one lands Sep 1-4: the
  Aug-to-Oct cattle roll. The Aug contract goes off the board this week.
  Fires P4 stretched-down, P5 bottom-5%, P6 down and P7b. All SKIP.
- **ZC=F / ZW=F — SURVIVE the roll check, and it is decisive.** Corn's +4.93%
  gap is the ONLY |gap|>3% inside Aug15-Sep05 in 27 years, against a 0.42%
  median for that window's largest gap; corn's >3% gaps live in September (18)
  and December (10), not August (1, tonight). Wheat's +2.97% is the largest
  since 2002. The 5-day advance underneath is intraday: +2.22%, +1.27%,
  +1.42%, +1.34% on gaps of -0.11%, -0.05%, -0.37%, +0.26%. Real move.
  NOTE: today's cache puts corn's gap on 08-25; the 08-24 brief attributed a
  near-identical gap to 08-24. That bar was revised. Footnote it.
- ZS=F clean (+0.58% gap). Its annual Aug 15-16 roll gap is a different event.
- SI=F: dist_52w_high -40% against a +77% 252d return is an old-series
  artifact. Silver stays out of every nugget tonight.

## Event lane

| trigger | verdict |
|---|---|
| `E:jackson_hole` k3 (Fri 2026-08-28) | **DRILL** (04). Only real scheduled subject. Engine base cells are soft: ^GSPC 17-9 up at +0.31%, t 1.48, sign p 0.084; SPY the same; IWM t 1.77; HYG 13-6. NG=F is the largest at -0.95%, t -2.39, but that is the sweep's tail on 25 obs and August natgas. The tellable cell is the RUN INTO the symposium and the symposium session itself, not one day-3 mean. 2026-08-23 published a JH VIX symposium-session nugget, so a VIX re-tell is banned; equities and the h3 window are untouched. Pre-specified? No. Jackson Hole drift is not a famous pre-registered hypothesis, so it owes the sweep its correction. |
| `E:seasonal_doy` (Aug 26, +/-2) | **DRILL** (05, 06). Two cells stand out and neither is a swept tail: DX-Y.NYB h5 is 21-5 up, sign p 0.0012, mean +0.50%, and it is the cell with the sharpest live contrast because DXY enters at the 0.8th percentile of its own 21d returns. ^VIX midterm h1 is 5-0 up among the six matching years, mean +4.64%. HYG h5 16-3, sign p 0.0022, is real but credit-at-highs was published 2026-08-23 and the h5 cell adds little. TLT/IEF h5 both 16-7, sign p 0.047, weaker versions of the same duration story. SKIP the rest. |
| `E:weekday_month` (Wednesdays in August) | **SKIP(generic)**. ^GSPC 72-47 up on 119 obs, sign p 0.014, but the mean is +0.099% against a +0.066% edge, t 1.03. A bare day-of-week x month cell at a tenth of a percent is exactly the minutiae that is not worth Scott's attention, and it fires every single day. Its only interesting form is "the Wednesday before Jackson Hole", which is the k3 cell 04 already owns. |
| `E:month_end` | **not fired, correctly.** Aug 26 is session 18 of 21, td_from_month_end 3, and the trigger needs < 3. It arms tomorrow night. The 2026-08-24 brief already spent the final-five-of-August cell, so nothing is lost. |
| `E:opex` `E:vix_expiry` | past (-2 and -4 td). Published 08-20 and 08-23. **SKIP(spent)**. |
| `E:cpi` `E:nfp` `E:ppi` `E:fomc_decision` `E:quad_witching` | 8 to 17 td out, nothing anchored. **SKIP(too far)**. Calendar only. |
| `E:holiday_pre` `E:holiday_post` `E:election` `E:fomc_minutes` `E:fomc_intermeeting` | did not fire. Labor Day is 2026-09-07, outside the anchor window; the election is 49 td out. **SKIP(not live)**. |

## Price lane

| trigger | verdict |
|---|---|
| `P5:rank5_extreme` ZC=F top5% + `P5b` ZC=F top5% + `P6:two_atr_day` ZC=F up + `P7:up_streak` ZC=F + `P4:z10_extreme` ZC=F | **DRILL** (02, 03). Corn is the tape. 5d +13.17% and 21d +16.05%, both the 100th percentile of the trailing year, z10 2.95, closing at a 52-week high 18.6% above its 200-day mean. Wheat z10 2.00 and beans z10 2.21 alongside. The engine only has the single-subject base cells; the joint grain state is the cell worth writing and nobody has published it. |
| `P4:z10_extreme` BTC-USD, ETH-USD; `P5/P5b` BTC-USD | **SKIP(spent 2026-08-24)**. BTC z10 declustered was last night's #5 and the state has not changed enough to re-tell. Countdown re-tellings are banned and this is the same number. |
| `P5b:rank21_extreme` USDZAR=X, JPY=X (both bh_pass) and `P4`/`P7` USDTRY=X (bh_pass) | **SKIP(structure, not news)**. USDTRY 70.8% hit on 418 obs is a managed devaluation trend, not a market fact about tomorrow; JPY and ZAR bottom-5% cells are carry mean-reversion at +0.09% and +0.35%. Real, tiny, and about nothing scheduled. |
| `P5b:rank21_extreme` SB=F, GC=F; `P3/P3b` SB=F | **SKIP(weak)**. Sugar's post-52w-high reversal cells run t -0.40 and -0.34 with edges of -0.15%; gold's 21d top-5% cell is t -0.50. Gold's 21d extreme was published 2026-08-24 in a sharper form. |
| `P5:rank5_extreme` ^BVSP, EWZ, TLT, GC=F | Brazil t -0.30 and EWZ t 0.50, **SKIP(weak)**. TLT top-5% 5d t -0.72, **SKIP(weak)**, and TLT +2.22% over 5d is a duration move the doy cells cover better. |
| `P7:up_streak` ^MXX ^FTSE ^BVSP GC=F ZC=F CT=F; `P7b` LE=F | **SKIP**. Every equity-index streak cell is inside +/-0.06% with t under 1. CT=F and LE=F are dead on the checks above. |
| `P6:two_atr_day` KC=F, USDCNY=X, LE=F | Coffee and cattle dead above. USDCNY -0.158% is a 2-ATR day only because the yuan's ATR is 0.07%; t 0.76 and a 43% hit. **SKIP(degenerate)**. |
| `P1/P1b/P3/P3b` | CT=F and SB=F only, both handled above. **DEAD / SKIP**. |
| `P8` `P9..P9f` `P10` `P11` `P12` | **did not fire**. No 200d cross in the universe, no joint stocks-bonds or dollar-gold day (SPY +0.32% with TLT +1.10% is a bull-both day, no trigger), VIX fell on an up day, breadth crossed nothing, no US print today. Noted so the absence is on the record. |
| Not triggered but on the tape: **CL=F -4.59%** | **DRILL** (07, cheap). Crude's whole -4.59% was intraday (-4.61%) against a +0.02% gap, so it is real, and it is the mirror of the grains. It fires no trigger because 4.52% ATR makes it a 1-ATR day. Worth one script to see whether energy-down-while-grains-up is a cell or a coincidence. |

## Drill queue

02 grains joint stretch (corn/wheat/beans simultaneously extended)
03 corn: 52w high + top-5% 5d, forward, era, concentration, controls
04 Jackson Hole: run into the symposium, h3 and the symposium session, midterm
05 DXY: the late-August doy cell, controls, and the oversold entry
06 VIX: the Aug-26 doy in midterm years
07 crude down 4%+ while the grain complex rallies
