# Cell map — run 2026-09-20 (Sunday)

Run date 2026-09-20. asof session 2026-09-18 (Friday). Next session 2026-09-21 (Monday).
Prices fresh through 2026-09-18, core bars SPY/^GSPC/QQQ/TLT all present. Price lane LIVE.
Sweep: 1153 cells scanned, 58 fired (36 event / 22 price), BH crit p 0.0056, 4 pass.
Cycle: midterm. Next session is td 14 of 21 in September, 7 sessions from month end.

## Standing constraint carried in from Thursday 2026-09-17

That brief spent FIVE nuggets on the week after September quad witching: SPY/QQQ h5
(19 of 26 down, -0.90%, sign p 0.0145), IWM minus SPY (20 of 26, -0.89pp, 8-for-8 since
2018), the Fed-conjunction expiry cell, the pre-crushed VIX into expiry, and the dollar on
the expiry. Its footnote additionally spent the midterm subset ("2 of 6", the weaker half),
the post-expiry-week VIX leg (+6.5% median, 18 of 26) and IWM's 21d-rank conditioning.

The window it previewed IS tomorrow. Under the novelty rule an event escalating from
upcoming to next-session earns one re-telling only if it adds NEW specificity. Thursday
left almost none on the table: every angle the engine surfaces tonight for this cell is a
number Thursday either printed or explicitly held out. So the seasonal_doy equity subjects
are SKIPped rather than re-told, and the calendar line carries the reminder instead.

## Event lane

| trigger / subject | verdict |
|---|---|
| `E:weekday_month` ^VIX (Mondays in Sep, n 87, h1 +3.05%, t 3.46, 56-31, BH pass, era-stable) | **DRILL** — the strongest never-published cell for tomorrow. The engine controls against ALL days, which cannot separate "September" from "Monday". Needs the 2x2. |
| `E:weekday_month` NG=F (n 87, +1.32%, t 2.85, BH pass) | SKIP(repetition) — natural gas seasonality published 09-08 and 09-16, held back again on 09-17. This would be the third telling in two weeks. |
| `E:weekday_month` ^GSPC/SPY/QQQ/IWM (h1 -0.10 to -0.23%, abs t < 1.2, era-unstable) | SKIP(null) — day-of-week x month on the indices is a coin flip and the era split flips. |
| `E:weekday_month` TLT/IEF (60.5% hit, sign p 0.037, t 1.5) | SKIP(weak) — the hit rate is the only interesting part, the mean is +0.16% and it is era-unstable. Carried into the rates drill as a possible follow-on, not as its own nugget. |
| `E:weekday_month` GC=F, CL=F, EEM, ^TNX, HYG, SI=F, HG=F, JPY=X, EURUSD=X, DX-Y.NYB | SKIP(null) — all abs t < 1.6, none both era-stable and strong. |
| `E:seasonal_doy` SPY/^GSPC h5 (n 26, -1.06%/-1.08%, 7-19, sign p 0.0145) | SKIP(repetition) — this is Thursday's nugget 2 restated on a different anchor. |
| `E:seasonal_doy` IWM h5 (n 25, -1.51%, 8-17, p 0.054; midterm -2.42%) | SKIP(repetition) — Thursday's nugget 1 plus the midterm figure its own footnote printed. |
| `E:seasonal_doy` NG=F h5 (+9.28%, 18-7, p 0.0216) | SKIP(repetition) — see above, third telling. |
| `E:seasonal_doy` ^TNX h5 (n 26, -1.16%, 8-18, p 0.0378) | **DRILL** — not covered Thursday, and it points the opposite way to a tape arriving with the 10-year at 4.998 and a 21d rank of 97. Folded into the rates drill as the seasonal control. |
| `E:seasonal_doy` HYG h5 midterm (n 4, 0-4) | DEAD — n 4. |
| `E:seasonal_doy` TLT/IEF/GC=F/SI=F/HG=F/CL=F/DX-Y.NYB/EURUSD=X/JPY=X/^VIX/EEM/QQQ | SKIP(null) — every one has sign p > 0.08 on the all-years h5 cell, and the midterm cells are n 5-6 with split records. |
| Calendar, next 5 sessions | Nothing scheduled Mon-Fri. Next tracked prints NFP Fri Oct 2, CPI Wed Oct 14, PPI Thu Oct 15, VIX expiry Wed Oct 21, FOMC Wed Oct 28, election Tue Nov 3. Quiet-tape contract does NOT bind: the price lane fired 22 cells. |

## Price lane

| trigger / subject | verdict |
|---|---|
| `P2`,`P2b`,`P5`,`P6` HE=F (-13.05% session, 52w low, 2-ATR down) | **DRILL first** — Thursday diagnosed Thursday's HE=F print as a pure roll gap (-11.53% headline, -11.50% opening gap, -0.04% inside the bar). Friday is a second consecutive double-digit print, so verify before anything else. If it is an artifact all four cells die together. |
| `P6` CC=F (-7.66%), `P5`/`P5b`/`P7b`/`P4` KC=F (-4.59%, 21d -22.9%, rank 0.4) | **DRILL** — same roll-seam check. Softs are a coherent group and no soft-commodity cell has run in this product. The cells are weak on their own (KC=F h1 +0.17 to +0.31%, all abs t < 1.9), so this publishes only if the check makes the move itself the story. |
| CL=F -6.32% session (fired NO trigger, its ATR is too wide) | **DRILL** — the largest genuine-looking macro move on the tape, missed by the sweep on a threshold. Exactly the near-miss case the spec says to recompute rather than ignore. Roll check first, crude rolls monthly. |
| `P5b` ^TNX top 5% 21d (n 398, h1 -0.14%, 179-216, sign p 0.049) + ^FVX (n 398, p 0.088) | **DRILL** — the base cell is thin but the live state is far more specific than the trigger: ^IRX at a 52-week high with z10 3.2, ^FVX 21d rank 98, ^TNX 97 and 0.16% off its 52w high, MOVE +13% over 21d. Condition on the whole curve, not one tenor. |
| MOVE 21d rank 81.7 / +13.2% while ^VIX sits -52% from its 52w high and -18.2% under its 200d | **DRILL** — no trigger covers bond vol against equity vol. Best novel cell on the board. |
| ^NDX 21d rank 50 (+0.74%) against ^RUT 21d rank 6 (-5.69%) and ^DJI rank 8.3 | **DRILL** — a 6.4pp 21-day large-growth-over-small spread. Distinct from Thursday's IWM-vs-SPY cell, which was calendar-anchored; this is price-state anchored and uses the Nasdaq leg Thursday never touched. |
| `P5b` five JPY crosses in the 21d bottom 5% at once (CHFJPY 0.8, NZDJPY 2.0, EURJPY 2.8, CADJPY 3.2, GBPJPY 3.6; EURJPY BH pass) | **DRILL** — the engine scored them one at a time. The simultaneity is the cell, and it sits against USDJPY 5d rank 94.8, so the yen is bid against everything except the dollar. |
| `P4` ^IRX z10 3.2 stretched up (n 245, t 1.28) | SKIP(killed 09-17) — "the 3-month bill at a 52-week high (n 16, nothing at any horizon)" was drilled and killed Thursday. Its state feeds the rates drill as a describer only. |
| `P4` USDSEK z10 up (n 126, h1 -0.17%, t -2.46, sign p 0.045) | SKIP(killed 09-17) — Thursday declustered USDSEK's streak cell to 38-49 and sign p 0.142. Same instrument, same week; no reason the z10 form survives declustering either. |
| `P4` NZDUSD, ^FVX stretched | SKIP(null) — abs t < 1.1. |
| `P6` BTC-USD 2-ATR up (n 82, h1 +0.54%, 39-43 DOWN, t 0.96) | SKIP(null) — the record contradicts the mean. ETH +7.67% noted here, no cell. |
| `P7` USDTRY 5+ up streak (n 427, 71% hit, BH pass) | DEAD(degenerate) — the lira trends by construction against a depreciating carry, so a streak cell is a tautology. Never a nugget subject. |
| `P5` HE=F bottom-5% 5d (n 310) | dies with the roll check. |

## Tag-hint and BH notes

- Engine `tag_hint` treated as a ceiling throughout. The only `solid` hints tonight are
  ^VIX and NG=F on September Mondays and HE=F on the 2-ATR cell. HE=F is an artifact and
  NG=F is repetition-blocked, so at most one solid hint survives to the brief.
- BH: the September-Monday VIX cell passes in its own right (crit p 0.0056, sign p 0.0048).
  Everything else published tonight is swept and is capped at suggestive or anecdote.
- Nothing tonight is a pre-specified famous hypothesis EXCEPT the post-expiry week, which
  is the cell being skipped for repetition.

## Drill plan

1. `01_roll_seams.py` — HE=F, CL=F, CC=F, KC=F: gap share vs intraday range. Gate for 3 cells.
2. `02_bondvol_vs_equityvol.py` — MOVE 21d rank high while VIX under its 200d.
3. `03_curve_top5.py` — whole-curve 21d top-5% with ^IRX at a 52w high; SPY/TLT/^TNX forward.
4. `04_ndx_rut_spread.py` — 21d NDX-RUT spread extreme, forward spread and forward SPY.
5. `05_sept_monday_vix.py` — the 2x2: Monday x September, plus concentration and era.
6. `06_yen_cross_cluster.py` — 4+ yen crosses simultaneously at 21d bottom 5%.
7. `07_crude_break.py` — conditional on the roll check clearing crude.

---

# Drill outcomes (written after Stage C)

| script | outcome |
|---|---|
| `01_roll_seams.py` | HE=F -13.05% is a ROLL SEAM (gap -11.79 of -13.05, intraday -1.44, range 2.62) — all four HE=F cells dead, second straight session of it. KC=F -4.59% also a seam (gap -3.98) and its 09-16/09-17 bars are open=high=low=close, so the coffee cells are dead twice over. SB=F +4.94% and CT=F -2.52% are seams too. CL=F -6.32% is REAL (gap -0.83, intraday -5.53, range 3.15) and so is CC=F -7.66% (gap -1.89, intraday -5.88). |
| `02_bondvol_vs_equityvol.py` | PUBLISH. n 56 declustered episodes 2003-2026, spread over 22 years. MOVE h5 -3.17% (14-42, sign p 0.0000, t -3.01) against a +0.45% local control; h21 -4.56% (15-40, p 0.0000). VIX does NOT follow: 24-32 at h5, 28-27 at h21, mean +3.14% against a +2.93% control. SPY sits on its base rate (h5 +0.34% vs +0.27%; h21 +1.27% vs +1.19%) despite a 39-16 record. |
| `03_curve_top5.py` | DEAD as a forward claim. All three tenors at 21d rank >= 95 is only 8 episodes, 7 with forwards, and the top-2 carry 202% of the ^TNX h5 total with the era split flipping sign (+3.07% pre-2018, -9.10% since). Relaxing to rank >= 90 (25 episodes) makes every forward null and the SPY h5 era split flips too. The curve is a LEVEL story, not a forward-return story. |
| `03b_tnx_five_pct.py` | PUBLISH the level. ^TNX closed 5.006 on 2026-09-16; only 566 of 6712 sessions since 2000 carry a 5% handle and they stop in 2007. Friday's 4.998 is the highest close since. |
| `03c_tnx_gap.py` | The gap is 4,817 sessions / 19.2 years from 2007-07-19 (5.028). 2026-09-16 was a scheduled FOMC decision day. In between the 10-year ranged 0.499 (2020-03-09) to 5.006. Only four prior "first 5% close after 60+ sessions below" events, so the forward record is n 4 and carries nothing; said so in the brief. |
| `04_ndx_rut_spread.py` | DRILLED, HELD for length. 52 episodes at spread rank >= 95 with NDX 21d >= 0 (live rank 94.4, just under). h21 records are strong and era-stable (^DJI 35-17 sign p 0.009, ^GSPC 34-18 p 0.018, hit 65.7% pre-2018 and 64.7% since) but the MEANS are null and below control (^GSPC +0.003% against a +0.687% local control) because the worst case is -25.2%. A real finding, cut only to keep the body inside 400 words and because nugget 4 already carries a "the alarming state is nothing" message. |
| `05_sept_monday_vix.py` | PUBLISH, but NOT as the engine framed it. The 2x2 kills the attribution: September Mondays +3.05% (n 87), other-month Mondays +1.87% (n 1172, t 7.51), September non-Mondays +0.11% (n 452), everything else -0.16%. Welch t for Sep-Mon vs other-month-Mon is only +1.29. So it is a Monday seam that September decorates. Survives dropping 2008/2011/2020 (+2.60%, t 3.07). Era: pre-2018 41-18 (p 0.0019) vs 2018+ 15-13 (p 0.425). |
| `05b_postexpiry_monday.py` | PUBLISH, headline. The Monday after the September expiry: S&P down 18 of 26, mean -0.47%, median -0.35%, sign p 0.0378, t -2.37, Welch -2.51 against other-month Mondays. The cross-month table is the control that makes it: October's equivalent Monday is +0.64% and up 20 of 26 (p 0.0047). VIX leg 20-6 up (+3.71%) but Welch only +1.43 vs a generic Monday, so the VIX half is mostly nugget 2's seam. Era fade is real and disclosed: 4-14 pre-2018 (p 0.0154), 4-4 since, up in each of 2022-2025; 2008 and 2021 carry 45% of the total. |
| `06_yen_cross_cluster.py` | MISLEADING AS POOLED. 4+ of 6 crosses at a 21d bottom-5% gives SPY h21 +2.54% (20-6, p 0.005) and VIX -13%, but the episode years are 2007/2008/2022 heavy — it is a panic-washout detector, and Friday is not a panic. |
| `06b_yen_calm_tape.py` | PUBLISH the conditioned version. Splitting the 31 episodes by vol state inverts it: the 23 STRESSED episodes give SPY h21 +2.73% (17-6, p 0.017), the 8 CALM ones (VIX >= 5% under its 200d, which is Friday at -18.1%) give SPY h5 -1.54% with 1 of 8 higher (p 0.035) and VIX +10.3% with 7 of 8 higher (p 0.035, t 2.96). n 8 so it ships as an anecdote, with the disclosure that 2 of the 8 are 2026-08-04 and 2026-09-08 and that 2008-09-02 and 2012-05-24 carry 57% of the h5 total. |
| `07_crude_break.py` | PUBLISH the frequency. 2026 already has 7 declustered 5%+ down sessions in crude, and 6 of the 23 episodes of "down 5%+ while more than 5% above its 200d average" since 2000 are this year, tying 2022. Forward is weak and disclosed: crude lower again at both h1 and h5 in 15 of 22 (p 0.067), and 12 of the 23 episodes are 2022 plus 2026. |
| `08_era_checks.py` | Settles two tags at [solid]. Monday VIX seam: positive in every 5-year block, pre/post-2018 +1.65%/+2.30% at t 6.54/4.15, non-Mondays -0.16%/-0.09% in both; weakest block is 2024-2026 at +1.22%, t 1.44, which the brief states. MOVE h5 reversion: negative in every 5-year block, pre/post-2018 -2.88%/-3.59%, both sign p < 0.02; 2008-2012 is the flat block at -0.45%. |

## Final slate: 6 nuggets, 2 tomorrow / 4 today

Tags: 2 solid (Monday seam, MOVE reversion), 2 suggestive (post-expiry Monday, crude),
2 anecdote (the 5% handle, the calm-tape yen cell) — exactly at the anecdote cap, and the
headline is the suggestive cell, not an anecdote.

The post-expiry Monday IS the one permitted escalation re-telling of Thursday's cell. It
earns it on new specificity: Thursday priced the five-session window, this is the single
session Scott faces, on a statistic Thursday never computed, against a cross-month control
table Thursday never had, with an era fade Thursday never disclosed.
