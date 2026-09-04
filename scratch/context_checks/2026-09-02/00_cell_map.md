# Cell map — run 2026-09-02 (Wed), asof session 2026-09-02, next session 2026-09-03 (Thu)

Prices FRESH through 2026-09-02 (SPY, ^GSPC, QQQ, TLT all print). Both lanes live.
Sweep: 1199 cells scanned, 85 fired (54 event / 31 price), 0 cleared BH at crit p 0.0.
Cycle: midterm year. Next session is td 3 of September, 21 sessions in the month.
Stale tape: LBS=F, ^AXJO, ^HSI, ^KS11, ^N225, ^SKEW carry no 2026-09-02 bar.
Cap: P5:rank5_extreme dropped ^IRX only. Bills are not a subject tonight; nothing recovered.

Novelty state is live (`delta_suppressed` false). One block: `P4:z10_extreme|ZC=F`,
published 2026-08-27, 4 td ago. It is also a roll artifact, see drill 01.

## The session, in one paragraph

Nothing scheduled today and nothing tomorrow; payrolls land Friday 2026-09-04, two
sessions out. SPY rose 0.44% to 765.16, 1.64% under its 52-week high, and ^VIX fell
6.98% to 15.20 with a 63-day return of -29.3%, rank 7.1. The bond market went the other
way: ^TNX closed at 4.796, AT a 52-week high, ^FVX 0.11% off its own, and ^MOVE rose for
a fifth straight session, +14.79% over five days, 5d rank 93.3. That gap between bond vol
and equity calm is the 99.4th percentile of its own history. The yen rallied hard against
everything, four of six crosses printing 2-ATR down closes. ^BVSP closed up for an
ELEVENTH straight session. The four largest movers on the tape (KC=F -13.12%, ZC=F +4.03%,
and the ZS/ZW 52-week highs) are a continuous-contract roll seam for a third night.

Last night's brief owns the pre-payrolls SPY drift cell, the IEF/LQD 52-week lows, the
TLT down-streak and the ^VIX pop. All four are off the board tonight, and the payrolls
countdown re-telling is the specific thing the novelty rule bans.

## Event lane

| trigger | verdict |
|---|---|
| `E:nfp` SPY / ^GSPC / QQQ / IWM arms (k=2) | **SKIP(countdown re-telling)**. This is last night's headline moved forward one session. Last night published the k=3 anchor at 190-127 and quoted this very leg inside it ("175-144 at -0.029% into Thursday"). SPY k=2 is 174-143 at -0.024%, QQQ 181-134 at +0.055%, IWM 162-149 at -0.012%. Same family, no new specificity, and the rule against "3 days to CPI, then 2 days to CPI" exists for exactly this. |
| `E:nfp` ^VIX arm (k=2) | **DRILL -> PUBLISH (headline)**. n=317, h1 mean +0.728% with a 139-178 DOWN record, sign p 0.016: the mean and the record disagree, which is the thing worth resolving. Drill 04 resolves it (the mean is a 5% tail, trimmed mean +0.115%) and finds the real cell one leg further out at h=2, the print itself: 200 of 319 down. Drills 04b and 04c condition it on the live state and build the matched control. Not a repeat of last night's ^VIX item, which was about a same-day pop with no forward claim. |
| `E:nfp` NG=F arm (k=2) | **DRILL -> PUBLISH**. The largest abs(t) in the whole payrolls group and era-stable: n=309, -0.424%, t -2.16, 136-172, sign p 0.027. No mechanism, so the entire question is whether the controls kill it. Drill 05 runs four of them. |
| `E:nfp` GC / SI / HG / CL / DX-Y / EURUSD / JPY / TLT / IEF / ^TNX / HYG / EEM arms | **SKIP(no edge)**. Every one inside +/-0.11% with abs(t) < 1.5, and TLT, IEF, ^TNX, HYG, CL, DX-Y, EURUSD, JPY are era-unstable on top of it. IEF is the emptiest cell on the board: 140-140, mean +0.016%, sign p 0.66. |
| `E:weekday_month` (Thursdays in September) JPY=X | **DRILL -> PUBLISH(folded)**. The best record in the group by a distance: 70-39 up for the dollar, n=109, sign p 0.0019, era-stable, and the engine's own t of 1.25 correctly says the magnitude is tiny. Drill 02 re-derives it on the native index (111 anchors, 70-41), splits the eras, runs two matched controls and checks concentration. Folded into the yen nugget rather than given its own slot, because it and the 2-ATR cluster point the same way and one fact appears once. |
| `E:weekday_month` ^VIX / SPY / QQQ / everything else | **SKIP(no edge, and near-duplicate of a published form)**. ^VIX is the strongest at -0.293%, 47-63, sign p 0.076, and last night's map already rejected the Wednesday version of this trigger as a weekday rotation of a published cell. SPY -0.028% at t -0.26 and ^GSPC -0.002% at t -0.02 are nothing. |
| `E:seasonal_doy` (Sep 03 +/-2, one pick per prior year) | **SKIP(dead / near-dead)**. SPY 26 anchors at +0.004%, 15-11, sign p 0.28. The two arms with a record are IEF and EEM at 16-7, sign p 0.047, on n=23 with no control and no mechanism, and every midterm arm is n=5 or 6. ^TNX at 9-17, sign p 0.084, is the most tempting and is still a swept 26-anchor cell with a matching-year tolerance of +/-2 days. Nothing here is worth a slot against six live states. |
| `E:month_end` / `E:turn_of_month` | **N/A**. Next session is td 3 of 21. Outside both windows. |
| `E:holiday_pre` / `E:holiday_post` | **N/A**. Labor Day is 2026-09-07 and the pre-holiday session is Friday 2026-09-04, so this trigger belongs to Thursday night's brief. Flagged here so its absence is not read as a miss, and it is why Friday is doubly loaded. |
| `E:cpi` `E:ppi` `E:fomc_decision` `E:vix_expiry` `E:opex` `E:quad_witching` `E:jackson_hole` `E:election` | **N/A**. Outside their anchor windows: PPI 5 td, CPI 6 td, FOMC and VIX expiry 9 td, opex and quad witching 11 td, election 43 td. Calendar block only. |

## Price lane

| trigger | verdict |
|---|---|
| `P6:two_atr_day` down — EURJPY, GBPJPY, CHFJPY, NZDJPY | **DRILL -> PUBLISH**. Four separate arms of one event. The per-cross base cells are n=21-34 and EURJPY at t 2.99 is the only one that looks like anything alone. The content is the JOINT state, which the engine never asks about: how often do four or more yen crosses do this on the same session. Drills 02 and 02b. |
| `P6:two_atr_day` down — USDCNY=X | **SKIP(no edge)**. n=111, +0.098% at t 0.76, 48-48. A managed currency's 2-ATR day is a different object from a floating one, and pooling them would be the error the engine's own side_fn rule exists to stop. |
| `P6:two_atr_day` down — KC=F | **DEAD(roll seam)**. -13.12% on a 78% overnight gap, into a session whose PRIOR bar traded 8 contracts against a 20-day median of 5,838. The move is measured off a stale expiring-contract close. Third night of this. |
| `P6:two_atr_day` up — EWZ | **DRILL -> SKIP(era flip + concentration)**. n=40, 26-10 up, sign p 0.040 looked publishable. Drill 06 splits it: pre-2018 is 27 episodes at +0.060%, 48% up, and the entire cell lives in 11 modern episodes at +1.291%. The tightest version (2-ATR up AND 5d rank >= 95) is n=12 with 73% of its total in two episodes, and ^BVSP, the underlying, goes the OTHER way over the same window (-0.570%, 7-9). EWZ +4.16% against ^BVSP +3.05% with USDBRL -1.65% says a chunk of tonight's EWZ print is the dollar leg, not Brazil. |
| `P7:up_streak` — ^MOVE | **DRILL -> PUBLISH**. n=100 at h=1 is nothing (45-55), which is why the engine's tag_hint undersells it: the cell is at h=5, where 55 episodes go 16-39 down, median -3.31%, sign p 0.001, against a local +/-126td control of +0.44%. Drill 03b, after drill 03 was thrown out for computing its ranks on a union-calendar panel and reading ^VIX's 63d rank as 48.0 against the engine's 7.1. |
| `P7:up_streak` — ^BVSP | **DRILL -> PUBLISH(as an anecdote)**. The trigger fires at 5+ and pools everything above; the live streak is ELEVEN, which only 9 sessions in 26 years have reached. Drill 07. The forward sample at that depth is n=5 and splits 1-4, so it publishes as a rarity with an explicit refusal, not as a claim. |
| `P7:up_streak` — AUDNZD=X | **SKIP(no edge)**. n=192, -0.010% at t -0.30, 95-96. |
| `P4:z10_extreme` up — ZC=F, ZW=F, ZS=F | **DEAD(roll seam + repeat_blocked)**. Drill 01: on 2026-09-01 ZC traded 5,480 against a 136,567 median, ZS 340 against 8,488, ZW 339 against 58,336, then all three printed 94,937 to 309,668 today. The "52-week highs" are measured off near-untraded prior closes. ZC is separately repeat_blocked. |
| `P4:z10_extreme` up — EWZ, ^BVSP | **SKIP(folded)**. Same country as the P6 and P7 arms; handled in drills 06 and 07, not given a third fingerprint. |
| `P4:z10_extreme` down — USDCNY=X | **SKIP(no edge)**. n=174, -0.018% at t -1.26, 80-87, and a pegged-band currency's z-score is not the same statistic as a floating one's. |
| `P5:rank5_extreme` bottom — KC=F | **DEAD(roll seam)**. Same bar as the P6 arm. |
| `P5:rank5_extreme` bottom — CHFJPY, GBPJPY, NZDJPY, NZDUSD | **SKIP(folded)**. The three yen crosses are the same event as the P6 cluster and go into drill 02. NZDUSD alone is n=270 at -0.014%, t -0.21, 127-140: nothing. |
| `P5:rank5_extreme` top — ^FVX | **SKIP(level, not rank; and adjacent to a published item)**. n=331, -0.113% at t -0.45 with sign p 0.19 is empty on its own, and the interesting fact about ^FVX is that it closed 0.11% from a 52-week yield high, which is the LEVEL story last night's brief already told through ^TNX. Tonight's new rates content is the vol surface, not the level, and it is in the ^MOVE nugget. |
| `P5:rank5_extreme` top — EWZ, ^BVSP | **SKIP(folded)**. Drills 06 and 07. |
| `P5b:rank21_extreme` — BTC-USD | **SKIP(off-subject, parked again)**. n=302, +0.728% at t 2.84, era-stable, the only `solid` tag_hint on the board. It was parked last night for the same reason and the reason has not changed: BTC fell 0.15% today, the state is 21-day momentum rather than anything tonight printed, and it competes against six live states in the asset classes that moved. Not rejected. Worth a slot on a genuinely quiet night. |
| `P5b:rank21_extreme` — ETH-USD, AUDJPY=X | **SKIP(no edge)**. ETH n=188 at t 1.31, 99-89, era-unstable. AUDJPY n=328 at +0.014%, t 0.29, 166-161, era-unstable, and AUDJPY was the one yen cross that did NOT print a 2-ATR day tonight (-0.05%), which is itself the honest limit on the cluster nugget and is stated in it. |
| `P5b:rank21_extreme` — ZC=F, ZW=F, ZS=F, SB=F | **DEAD(roll seam)** for the grains. SB=F is the one softs bar that survives drill 01 (0-17% gap share, volume 125,318 against a 111,024 median, so it is real trade), but n=442 at +0.068%, t 0.64, 219-216 is nothing to publish. |
| `P1` / `P1b` / `P2` / `P2b` 52-week extremes | **N/A(did not fire)**. Noted because ^TNX closed AT a 52-week high and did not trigger: these fire only on the FIRST such print in 30 or 90 calendar days, and the 10-year has been making them for weeks. That is a correct non-fire, not a miss, and it is why the level story is last night's rather than tonight's. |
| `P3` reversals, `P8` 200d cross, `P9`-`P9f` cross-asset, `P10`-`P10c` VIX term structure, `P11` / `P11b` breadth, `P12` surprise-conditioned | **N/A(did not fire)**. No release printed today, so P12 has nothing to condition on. Breadth sits at 65.5% above the 200d against 62.1% 21 sessions ago, well inside both thresholds. P10 did not fire on a 6.98% VIX drop because the trigger is the +10% side and the term-structure crosses; ^VIX3M -3.27% keeps the curve in normal contango. |

## Data caveats carried into the brief

- **Grains, coffee and cotton are excluded**, third night running. Drill 01 has the volumes.
  CT=F additionally printed an Open of 0, a corrupt bar rather than merely a roll.
- **SI=F's 52-week-high distance is wrong** in the tape: -42.71% from a 52-week high while
  +61.28% above its 52-week low and +60.52% over 252 days. Those cannot all be true.
  Silver is not a subject tonight, so this is noted rather than chased.
- **Drill 03 is retained but not used.** Its ranks came off a union-calendar panel and read
  ^VIX's 63d rank as 48.0 against the engine's 7.1. Drills 03b and 03c redo it on native
  indices and match the engine exactly. Kept in the folder because the error is the point.

## Tag and multiplicity accounting

Zero cells cleared Benjamini-Hochberg tonight (crit p 0.0, 85 fired of 1199 scanned), so
nothing is tagged `[solid]` on swept evidence and nothing claims to be. Two of the five
published nuggets are pre-specified rather than swept and say so: the payrolls event cell
and the September-Thursday calendar cell are famous-hypothesis forms, not search output.
The rest are swept states and each carries its own control instead of a p-value. One
`[anecdote]` (^BVSP at eleven), against a budget of two, and it is not the headline.

## Drills

| file | what it settled |
|---|---|
| `01_roll_gap_check.py` | Grains, coffee, cotton are a roll seam. Killed 8 fired cells. |
| `02_yen_cluster.py` | 4 of 6 crosses at 2-ATR down; USDJPY 14 of 16 up next session. Sep-Thursday cell re-derived with controls. |
| `02b_yen_cluster_era.py` | Era split and every episode listed. Direction holds both eras, magnitude is crisis-era. |
| `03_bondvol_vs_equityvol.py` | THROWN OUT, union-calendar rank bug. Kept as the record of it. |
| `03b_move_streak_clean.py` | Native-index ranks match the engine. ^MOVE h=5 cell with eras, controls, concentration. |
| `03c_vol_divergence_cross.py` | The MOVE/VIX cross adds nothing beyond MOVE alone. Gives the 99.4th-percentile gap. |
| `04_prenfp_vix_calm.py` | The h=1 mean is a 5% tail. The cell is at h=2. |
| `04b_payroll_vol_crush.py` | Era split, midterm and September subsamples. |
| `04c_vix_matched_control.py` | The matched control, on exactly the published condition. |
| `05_natgas_prenfp.py` | Four controls, all flat-to-positive. Not a first-week-of-month effect. |
| `06_brazil_stretch.py` | EWZ continuation is modern-era only and the local index disagrees. Killed. |
| `07_bvsp_long_streak.py` | Eleven up closes: 9 sessions in 26 years. Forward sample n=5, no claim. |
