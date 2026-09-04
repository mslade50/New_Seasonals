# Cell map — run 2026-08-12 (Wed), asof session 2026-08-12, next session 2026-08-13 (Thu)

Prices FRESH (core bar 2026-08-12). Both lanes live. 1185 cells scanned, 87 fired
(54 event / 33 price). BH crit p 0.0, so zero swept cells clear BH at alpha 0.10 —
nothing publishes as `[solid]` on swept evidence alone tonight.

State: CPI printed this morning. SPY +0.25% (0.10% under its 52w high), QQQ +0.73%,
IWM / ^RUT / ^NYA / HYG / EWJ all closed AT 52w highs, VIX -4.78% to 14.55,
^MOVE -7.48% to 72.1, ^TNX 4.682 (-0.04%), TLT 0.23% above its 52w LOW,
IEF 0.86% above, LQD 0.75% above. Gold +1.96%, corn +10.13%. PPI tomorrow 08:30 ET.

Cap note: `P4:z10_extreme` truncated at 8, dropped USDCNY=X and USDTRY=X. Neither is
a subject I would have published; no recompute owed.

## Event lane

| trigger | verdict |
|---|---|
| `E:ppi` (18 subjects, k1, n~317) | **DRILL**. The only two subjects with a real number are the bond legs: IEF n=286 +0.061% 58.0% hit t=2.62 sign p 0.0038, TLT n=286 +0.115% 57.0% t=2.38 sign p 0.0105. Both era-stable. Equities are empty (SPY -0.009% t=-0.12, ^GSPC -0.010%, QQQ -0.024%), which is itself worth one line. Three things to resolve before this can publish: (a) PPI usually prints on a THURSDAY, so this cell and `E:weekday_month` below may be the same days wearing two hats; (b) tomorrow's PPI lands 1 td after a CPI, a specific and checkable sub-cell; (c) TLT is sitting on a 52w low, so the conditional matters. -> `01_ppi_bonds.py` |
| `E:weekday_month` — Thursdays in August | **DRILL**, folded into 01. TLT n=107 67-39 up sign p 0.0058; ^TNX 43-73 down sign p 0.0047; IEF 63-42 sign p 0.0407. Same direction as the PPI cell and almost certainly the same sessions. Publishing both as if they were two findings would be the exact multiplicity failure the footnote is supposed to guard. Resolve the overlap or drop one. |
| `E:weekday_month` — ^VIX +1.53% mean | **SKIP(no edge)**. n=117, t=1.77, hit 45.3%, era-unstable, and the direction contradicts the record (53-63 down with a positive mean = two outliers). Nothing to say. |
| `E:weekday_month` — other 15 subjects | **SKIP(no edge)**. |t| <= 1.2 or hit inside 47-53% for SPY, QQQ, ^GSPC, IWM, EEM, HYG, GC=F, SI=F, HG=F, CL=F, NG=F, DX-Y.NYB, EURUSD=X, JPY=X. |
| `E:seasonal_doy` — Aug 13, SPY/^GSPC | **DRILL**. All years n=26 +0.228% h1. Midterm n=6 is 5-of-6 DOWN h1 (-0.12%) then 5-of-6 UP h5 (+1.34%). N=6 is anecdote tier at best and the two halves point opposite ways, which is what a 6-observation cell looks like when it is noise. Widen to an Aug 10-18 midterm window and see whether the shape survives at usable N before believing any of it. -> `07_seasonal_aug_midterm.py` |
| `E:seasonal_doy` — TLT/IEF Aug 13 midterm h5 | **DRILL**, same script. TLT 5-for-5 up h5 (+1.26%, sign p 0.0312), IEF 5-for-5 (+0.36%). A perfect record at N=5 is one coin flip from meaningless, but it points the same way as the two cells above, so it is worth knowing whether the mid-August bond bid is real or three views of the same handful of Augusts. |
| `E:seasonal_doy` — 14 other subjects | **SKIP(no edge / N<=6 with split records)**. QQQ, IWM, ^TNX, HYG, GC=F, SI=F, HG=F, CL=F, NG=F, DX-Y.NYB, EURUSD=X, JPY=X, ^VIX, EEM. Nothing reaches even a clean 5-of-6. |
| Calendar, next 5 sessions | PPI Thu 08-13 (covered above). VIX expiry Wed 08-19 and opex Fri 08-21 are 5 and 7 td out: **SKIP(too far)** — telling Scott about opex tonight and again next Thursday is the countdown failure mode. Jackson Hole 08-28, **SKIP(too far)**. Next FOMC 09-16, 24 td: **SKIP(too far)**. NFP 09-04. |

## Price lane

| trigger | verdict |
|---|---|
| `P4:z10_extreme` ^GSPC / SPY / ^IXIC / ^NYA / ES=F stretched up | **DRILL**. ^GSPC n=120 49-71 down sign p 0.0274 h1 -0.059%, but era-unstable and the mean is trivial. The interesting part is not in the engine's cell: SPY's 5d return is +0.35% (45.6th percentile) and its 21d realized vol is 13.7 against 0.62x its 63d norm. The z is high because the DENOMINATOR collapsed, not because the tape ran. Split the historical cell on that and see whether the two kinds of z10>=2 behave differently. -> `02_z10_vol_compression.py` |
| `P1:new_52w_high` EWJ | **SKIP(no edge)**. n=31, 15-15, t=-1.11 on a negative mean, era-unstable. Japan at a 52w high is real but the cell says nothing. |
| `P1:new_52w_high` ZC=F | **DRILL**, merged into the corn script. Corn closed +10.13%, a >=2 ATR up day, at a 52w high, 21d return in the top 5% of its year. Three triggers on one instrument on one day is the tape's loudest single event. The base cell is empty (n=22, 10-12) so the nugget has to be about the SIZE and rarity of the move, not the forward return. -> `05_corn_thrust.py` |
| `P6:two_atr_day` ZC=F up, `P5:rank5_extreme` ZC=F top 5%, `P5b:rank21_extreme` ZC=F | folded into `05_corn_thrust.py`. Individually all empty (h1 +0.036% / +0.108% / +0.185%, none era-stable except P5b). |
| `P7b:down_streak` ^BVSP | **DRILL**. n=144, +0.401% h1, 59.0% hit, t=1.86, sign p 0.0184, era-stable. Brazil closed down a 6th straight session with its 5d return in the 0.4th percentile of its year, on the same day US indices printed 52w highs. The cross is the story, not the streak. -> `06_brazil_divergence.py` |
| `P5:rank5_extreme` ^BVSP / EWZ bottom 5% | same script. EWZ n=314 +0.246% t=1.01 era-stable; ^BVSP n=315 +0.173% t=0.96 era-unstable. Weak alone. |
| `P8:sma200_cross` HE=F, `P6` HE=F down, `P5` HE=F bottom 5% | **SKIP(out of scope)**. Lean hogs -12.9% is the largest move on the tape and lean hogs are not macro context for this reader. The `P6` HE=F cell is the only `solid` tag_hint in the whole sweep (n=115, t=-2.77) and I am declining it on relevance, not on statistics. |
| `P6:two_atr_day` LE=F down, ZS=F up | **SKIP(out of scope)**. Live cattle and soybeans, same reason. |
| `P2/P2b:new_52w_low` USDMXN=X | **SKIP(weak)**. n=21 15-6 up sign p 0.0392 looks tempting but the 90-day version is n=13 8-5 and h1 +0.09%, i.e. the effect halves as the cell tightens, and the mean is a tenth of the peso's daily range. Noted: the peso at a 52w high against the dollar with US equities at highs is a coherent risk-on picture, but I have no cell that pays for a sentence. |
| `P7b:down_streak` USDMXN=X | **DEAD**. h1 -0.001%, 101-102. |
| `P7:up_streak` CADJPY=X / AUDJPY=X | **SKIP(no edge)**. AUDJPY n=252 55.6% sign p 0.0444 is the better of the two and the mean is +0.066%, inside the spread of a yen cross. Carry pairs grinding up is the same risk-on tape the equity nuggets already carry. |
| `P5:rank5_extreme` CAD=X bottom 5% | **SKIP(no edge)**. t=2.10 on a +0.077% mean, n=275. Statistically alive, economically a rounding error, and the loonie is not tomorrow's subject. |
| `P5:rank5_extreme` SB=F / GBPJPY=X / CADJPY=X top 5%, `P5b` SB=F | **SKIP(no edge)**. All |t| < 1 or negative means on positive tags. |
| `P4:z10_extreme` GC=F / SB=F / CT=F | **SKIP(repetition)**. Gold's z10 is 2.38 and the 5d is +5.26%, but gold was last night's item 4 (`E:cpi\|GC=F\|k1\|gold_already_run`, 08-11) and the engine's cell is empty anyway (n=286, h1 +0.005%). Two gold nuggets in two nights on a different slice of the same run is the repetition rule's target even though the fingerprint differs. |
| `P6:two_atr_day` USDCNY=X down | **SKIP(no edge)**. n=110, 48-47, sign p 0.92. |

## Triggers that did NOT fire but where the state is nearly there

- `P9` family (stocks and bonds together, dollar and gold): did not fire tonight. The
  underlying state — US indices at 52w highs while TLT/IEF/LQD sit within 1% of 52w
  lows — is live, but `P9:level_divergence|SPY_TLT|high_low` PUBLISHED ON 2026-08-10 at
  a headline number of -0.05, two sessions ago, and the state has not changed. **SKIP(repeat_blocked in spirit)**.
  Re-telling it with IWM substituted for SPY would be the same nugget with a new
  fingerprint, which is laundering, not novelty. What IS new and unpublished is the
  INTERNAL split: ^NYA closed at a 52w high while QQQ is 2.90% below its own. That is a
  different claim (breadth leading the Nasdaq) and gets its own drill. -> `03_nya_high_qqq_lag.py`
- `P10` VIX term structure: VIX 14.55 / VIX3M 18.53, contango 27%, nothing inverted, no
  trigger. ^MOVE fell 7.48% today with ^TNX unchanged, which is not in the trigger
  inventory at all. Bond vol collapsing on a CPI print while the yield does not move is
  the cleanest single description of today's bond market and it feeds the PPI thread.
  **DRILL** as an off-inventory cell. -> `04_move_crush.py`
- `P11` breadth: 64.4% of the panel above its 200d, from 69.0% 21 sessions ago. No cross
  of the 80/20 lines. **SKIP(no trigger)**.
- `P12` consensus-conditioned: `releases_today` is empty, so no actual-vs-consensus cell
  exists for today's CPI. **DEAD(no data)** — worth noting that the CPI print itself is
  therefore un-drillable on surprise, and every CPI claim tonight would be a level claim.

## Novelty ledger

`delta_suppressed` false. Every fingerprint in tonight's sweep carries `is_new: true`.
Flag state holds 11 records from 08-10 and 08-11. The two live constraints:
- SPY/TLT high-low divergence, published 08-10 -> excluded above.
- Gold, published 08-11 -> excluded above.

## Multiplicity accounting

BH crit p is 0.0 and `bh_pass_count` is 0, so NOTHING tonight publishes as `[solid]` on
the strength of a swept p-value. Of the cells I intend to work:
- PPI-day bond behaviour is a PRE-SPECIFIED hypothesis (a scheduled inflation print's
  effect on duration), not a sweep discovery. It does not owe BH a correction. Cap at
  `[suggestive]` regardless, because era stability and the Thursday confound are the
  real questions and both are decided in 01.
- z10 / breadth / MOVE / corn / Brazil cells are all sweep-discovered or self-selected
  from tonight's tape. Anything from those is `[suggestive]` at best and the
  vol-compression split in 02 is explicitly a post-hoc conditioning, which the text
  will say.

## Drill queue

1. `01_ppi_bonds.py` — the PPI bond bid, the Thursday confound, day-after-CPI, TLT at a low
2. `02_z10_vol_compression.py` — z10>=2 from a small denominator vs a big numerator
3. `03_nya_high_qqq_lag.py` — broad index at a 52w high with the Nasdaq lagging
4. `04_move_crush.py` — bond vol collapse with the yield unchanged
5. `05_corn_thrust.py` — the size and rarity of a +10% corn session at a 52w high
6. `06_brazil_divergence.py` — Brazil's 6th down close against US highs
7. `07_seasonal_aug_midterm.py` — does the Aug-13 midterm shape survive a wider window
