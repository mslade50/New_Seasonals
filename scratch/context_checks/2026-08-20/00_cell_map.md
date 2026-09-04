# Cell map — run date 2026-08-20 (Thursday)

- asof session: 2026-08-20 (Thu), prices_fresh=True, core bar 2026-08-20
- next session: 2026-08-21 (Friday) = **August monthly opex**
- cycle: midterm year, month position td 15 of 21, 6 td from month end
- sweep: 1213 cells scanned, 99 fired (54 event / 45 price), BH crit p 0.0052, 7 pass
- warnings: 5 tickers stale (LBS=F, ^AXJO, ^HSI, ^KS11, ^N225), all foreign cash or a
  dead contract, none is a nugget subject tonight
- novelty: `E:seasonal_doy|TLT` repeat_blocked (published 2026-08-17). Recent briefs
  already spent `E:opex|TLT|k3_run` (08-18) and `E:opex|NG=F|k2` (08-19), so an opex
  nugget tonight must be a different subject AND must be the opex session itself
  (k1, h1 = the opex bar), never another countdown step.

## Session recap driving the price lane

QQQ -0.72% (5th consecutive down close, cumulative -2.9%), SPY -0.84%, IWM -1.34%,
TLT -0.82%, ^TNX +0.92% to 4.696, ^VIX +7.52% to 16.01, GC=F +1.91% to 4575
(21d +10.3%), DX-Y.NYB flat on the day but 21d rank 2.0. Breadth 64.4% above the
200d, down from 69.0% 21 sessions ago.

## Calendar inside the next 5 sessions

| date | event | td | verdict |
|---|---|---|---|
| 2026-08-21 | monthly opex | +1 | **PUBLISH/DRILL**, the whole event lane tonight |
| 2026-08-28 | Jackson Hole | +6 | SKIP(outside the 5td window; calendar line only) |

Nothing else lands before 2026-09-04 NFP. No top-tier print ahead, but opex is a
scheduled event, so the quiet-tape contract does not apply.

## Event lane

| trigger / subject | verdict |
|---|---|
| `E:opex` ^VIX k1 | **DRILL**. n=319, -0.99%, 107-209 down, t=-2.58, era-stable, BH pass. Best cell in the sweep. Midterm subset is much weaker (n=79, -0.50%, t=-0.69) and the August month cell is 6-20 at only -0.50% mean. Record and mean disagree; the drill has to resolve which one publishes. |
| `E:opex` HG=F k1 | **DRILL**. n=311, +0.263%, t=2.89, 170-141, era-stable. Crosses with the copper doy cell below (19-6, sign p 0.0073). Two independent August-21 copper cells pointing the same way is worth one script. |
| `E:opex` QQQ k1 | **DRILL**. n=319, -0.184%, t=-2.48, but the record is a coin flip (149-168, sign p 0.19) and the mean is small. Only interesting crossed with tonight's QQQ down streak. |
| `E:opex` SPY / ^GSPC k1 | **DRILL**. Bare cell is nothing (-0.07%, 162-156, t=-1.24). SPY enters this opex with a 5d return in the 9.5th percentile of its year. Opex-after-a-down-week is the conditioning that makes it a claim. |
| `E:opex` CL=F k1 | SKIP(era_stable=False, t=1.07, and 08-18 already published a CL=F August weekday cell) |
| `E:opex` TLT / IEF / ^TNX k1 | SKIP(TLT +0.06% t=1.18, IEF +0.01%, ^TNX +0.03% t=0.22, all noise; TLT opex was also spent on 08-18) |
| `E:opex` HYG k1 | SKIP(n=232, +0.057%, t=1.77, a 6bp edge is below anything worth a sentence) |
| `E:opex` GC=F / SI=F k1 | SKIP(GC=F t=1.22 era_stable=False, SI=F t=0.21; the gold story tonight lives in the weekday cell, not here) |
| `E:opex` JPY=X / DX-Y.NYB / EURUSD=X k1 | SKIP(t 0.34 to 1.54, edges of 1-5bp) |
| `E:opex` NG=F k1 | SKIP(t=0.73, era_stable=False, and NG=F opex was published 08-19, so a re-telling would be the banned countdown) |
| `E:opex` IWM / EEM k1 | SKIP(IWM t=-0.26, EEM t=1.12, both era_stable=False) |
| `E:weekday_month` GC=F (Aug Fridays) | **DRILL**. n=114, +0.225%, t=2.33, 66-48, era-stable. Gold is up 10.3% over 21d at z10 1.68, so the honest question is whether the cell survives being already stretched. |
| `E:weekday_month` ^VIX (Aug Fridays) | SKIP(subsumed by the stronger opex VIX cell; publishing both would double-count the same Friday bar) |
| `E:weekday_month` TLT / IEF (Aug Fridays) | SKIP(t=1.73 / 1.57, and the bond lane was spent on 08-17 and 08-18) |
| `E:weekday_month` NG=F, ^TNX, CL=F, SI=F, HG=F, EURUSD=X, JPY=X, HYG, EEM, DX-Y.NYB, SPY, ^GSPC, IWM, QQQ | SKIP(all abs(t) < 2 with sub-20bp edges; the bare weekday cell is the weakest family in the sweep and only earns space when it is the only lane firing) |
| `E:seasonal_doy` HG=F | **DRILL**. 19-6 up, sign p 0.0073, +0.50% mean, midterm 5-1. Folded into the copper script. |
| `E:seasonal_doy` JPY=X (18-8, p 0.038), EURUSD=X (16-6, p 0.026), EEM (7-16, p 0.047) | SKIP(bare doy cells at n=19-26 with no mechanism, and doy is the most heavily swept family here; none clears the 0.0052 BH bar) |
| `E:seasonal_doy` SPY / QQQ / ^GSPC / IWM | SKIP(all coin flips: 12-14, 9-16, 11-15, 11-13; midterm subsets are n=6 anecdotes) |
| `E:seasonal_doy` TLT | SKIP(repeat_blocked, published 2026-08-17, number has not moved) |
| `E:seasonal_doy` GC=F, SI=F, CL=F, NG=F, DX-Y.NYB, ^TNX, IEF, HYG, ^VIX | SKIP(sign p 0.10 to 0.66, nothing separable from day-count noise) |

## Price lane

| trigger / subject | verdict |
|---|---|
| `P7b:down_streak` QQQ | **DRILL**. n=96, +1.20%, 61-35, t=4.08, era-stable, BH pass, midterm hit 66.7%. Strongest price cell and it is about tomorrow directly. Two checks first: the live streak is only 5 days and -2.9% deep while the cell's best outcome is +12.2%, so concentration and a depth split are mandatory. |
| `P7b:down_streak` ^NDX | **DRILL**. n=91, +0.77%, t=3.11, era-stable. Same underlying as QQQ, used as the longer-history confirmation inside the QQQ script rather than as its own nugget. |
| `P7b:down_streak` NQ=F | SKIP(t=1.79, third redundant look at the same index) |
| `P7b:down_streak` USDHKD=X | SKIP(BH pass but the mean is 0.010%, a pegged currency, degenerate cell) |
| `P7b:down_streak` ^FCHI | SKIP(t=0.69, era_stable=False) |
| `P7b:down_streak` EWJ | SKIP(t=1.15, era_stable=False) |
| `P9b:stocks_bonds_down` SPY / TLT | **DRILL**. SPY +0.125% t=0.96, TLT +0.018% t=0.31, both era_stable=False, so the bare cell is dead as published. Today's version came with VIX +7.5% and 10y +9bp; the version conditioned on a real vol bid is worth one script. If it does not separate, it dies. |
| `P5b:rank21_extreme` DX-Y.NYB / UUP | **DRILL**. Dropped by the per-trigger cap, and the dollar at a 21d rank of 2.0 with gold up 10.3% over the same window is the most macro-relevant state in the tape. Recomputed rather than inherited, per the cap rule. |
| `P5b:rank21_extreme` JPY=X | **DRILL**. n=385, +0.087%, 223-162, sign p 0.0011, BH pass, era-stable. The same dollar-weakness state from the other side; folded into the dollar script as a cross-check. |
| `P5b:rank21_extreme` USDZAR=X | SKIP(BH pass, but an EM cross is not a subject this brief needs and the mechanism is carry, not anything about tomorrow) |
| `P5b:rank21_extreme` EURUSD=X, SB=F, HE=F | SKIP(t -1.06 / 0.50 / -0.15) |
| `P5b:rank21_extreme` BTC-USD, ETH-USD, USDTRY=X | SKIP(BTC t=2.78 is real but crypto momentum published 08-19 on the same tape; ETH era_stable=False; USDTRY is a devaluation trend, degenerate) |
| `P4:z10_extreme` BTC-USD | SKIP(t=3.06, but 08-19 published the ETH/BTC joint thrust, so this reads as a re-telling) |
| `P4:z10_extreme` USDTRY=X | SKIP(BH pass but degenerate: a currency in permanent decline scores 404-179 by construction) |
| `P4:z10_extreme` ZC=F, SB=F, CT=F, USDMXN=X | SKIP(abs(t) 0.01 to 1.34) |
| `P5:rank5_extreme` BTC-USD | SKIP(t=2.61 but duplicates the P4 and P5b crypto cells; one crypto nugget max and it was spent 08-19) |
| `P5:rank5_extreme` ZW=F, ETH-USD, CT=F, ZS=F, ZC=F, HE=F, CHF=X | SKIP(all abs(t) < 1.5) |
| `P5:rank5_extreme` dropped-by-cap EURUSD=X, EWJ, ^SKEW | SKIP(drop list examined; EURUSD and EWJ are covered by the dollar drill's frame, ^SKEW is a derived index whose 5d rank has no forward cell worth the space) |
| `P6:two_atr_day` KC=F | SKIP(coffee -8.7% is the day's biggest move and n=54 at t=0.94 says nothing follows it, a genuinely dead cell) |
| `P6:two_atr_day` BTC-USD, ZC=F, CHF=X, USDCNY=X | SKIP(abs(t) 0.39 to 0.84) |
| `P7:up_streak` EURJPY=X | SKIP(t=-2.16 is the only one with a pulse, but a cross-rate streak is not a subject for this brief and it collides with the dollar drill) |
| `P7:up_streak` CL=F, GBPJPY=X, CT=F | SKIP(abs(t) 0.12 to 0.57) |
| `P1:new_52w_high` / `P1b` CT=F, ZC=F | SKIP(cotton and corn, abs(t) 0.04 to 0.63, and ZC=F's 52w high was published 08-17) |
| `P8:sma200_cross` EURUSD=X | SKIP(n=14 anecdote at t=-0.76, and the dollar 200d story was published 08-19 via DX-Y.NYB) |
| `P8:sma200_cross` CAD=X | SKIP(t=0.36) |

## Hint handling

- `tag_hint` downgrades on the table: `E:opex|^VIX|k1` and `E:opex|HG=F|k1` arrive as
  `solid`. Whether they publish at that tier depends on what the drills find in the
  August and midterm subsets, both far weaker than the pooled cell.
- `bh_pass` accounting: monthly opex behaviour and the post-down-streak bounce are
  **pre-specified famous hypotheses**, not sweep discoveries, so they do not owe the
  multiplicity correction. `E:opex|^VIX|k1`, `P7b:down_streak|QQQ` and
  `P5b:rank21_extreme|JPY=X` clear it anyway. The copper doy cell (sign p 0.0073) was
  found BY the sweep and does not clear the 0.0052 bar, so it cannot be tagged solid
  on its own; it publishes only if the independent opex cell corroborates it.

## Drills queued

1. `01_qqq_downstreak_depth.py` — QQQ 5+ down closes split by streak depth, concentration, era
2. `02_qqq_streak_into_opex.py` — the streak's next session being a monthly opex
3. `03_vix_opex_august.py` — VIX on opex sessions, August and midterm subsets
4. `04_spy_opex_weak_week.py` — opex Friday conditioned on the week into it
5. `05_copper_aug_opex.py` — HG=F opex cell crossed with the Aug-21 doy cell
6. `06_dollar_rank21_floor.py` — dollar 21d bottom-5% recovered from the cap drop, plus the JPY cross-check
7. `07_gold_aug_friday_stretched.py` — August Fridays in gold, conditioned on being stretched

## Drill results, and what they changed

| drill | outcome |
|---|---|
| `01_qqq_downstreak_depth.py` | The pooled cell is real and era-stable (n=96, +1.202%, 61-35, t=4.08) but the magnitude is a function of the hole. Streaks deeper than -4% pay +1.686% (n=61, t=3.83); shallower ones pay +0.359% (n=35, t=1.86, 21-14). Tonight's is -2.89% over 5 closes, the shallow kind. **PUBLISH** with the split, not the pooled number alone. |
| `02_qqq_streak_into_opex.py` | The crossing is n=5 (4-1) and the difference in means versus a non-expiration next session is -0.042 pp. **The expiration label adds nothing to the streak cell**, which is itself worth one clause. Separately: QQQ on expiration bars is -0.184% (n=318, t=-2.47) while the record is 149-169, so that cell is a left-tail effect, not a frequency one. Folded into the QQQ nugget as the caveat rather than published on its own. |
| `03_vix_opex_august.py` | **PUBLISH.** August expirations are 6-20 down for VIX with a median of -2.45% and sign p 0.0047. The pooled -0.50% mean is wrecked by 2015-08-20 (+46.45%), so the record and the median carry it, not the mean. The live conditional is stronger and larger: expirations with VIX up over the prior 5 sessions are 38-85 down, mean -2.02%, t=-2.62, n=123. |
| `04_spy_opex_weak_week.py` | The expiration BAR after a weak week is nothing (SPY +0.040%, t=0.19). The follow-on is the cell: expiration bar plus the week after, entering on a sub-20th-percentile 5d return, pays +1.222% on 39-18, t=3.31, n=57, era-stable both sides, still t=2.89 after dropping the best two episodes. QQQ agrees at +1.287% on 40-18. **PUBLISH as the headline.** |
| `05_copper_aug_opex.py` | **PUBLISH, downgraded to suggestive.** Expiration bar +0.263% (n=311, t=2.89) against a +0.045% all-day drift and a +0.044% local control; h5 +0.600% on 182-129, sign p 0.0016. But the effect roughly halves post-2018 (+0.309% to +0.172%, t 2.68 to 1.18), so the `solid` hint is downgraded. No August-specific claim: that subset is n=25 with 71% of the total in 2009 and 2001. The doy cell corroborates rather than double-counts, since only 6 of the 93 sessions in that calendar window were also expiration anchors. |
| `06_dollar_rank21_floor.py` | **KILLED.** Recomputed the cap-dropped DXY cell and it is empty: 21d return in the bottom 5% of its year pays -0.020% next session on 190-192, +0.053% at h21, against a zero drift. JPY=X looked real at the day level (h5 +0.257%, 240-173, sign p 0.0006) and dies on declustering (57 episodes, +0.135%, t=0.70), so its BH pass is overlap inflation. The joint DXY-floor-with-gold-ripping state is the same story: gold h21 -0.635% at the day level, +1.207% at t=1.42 across 23 declustered episodes. No dollar or gold-macro nugget tonight. |
| `07_stocks_bonds_down_volbid.py` | **KILLED, and the kill publishes.** Stocks and bonds both down 50bp+ with VIX up 5%+ pays +0.079% next session on 84-81, against +0.049% for any session. Post-2018 it is -0.129% at a 44.0% hit, two episodes exceed the whole total, and declustering at 5td leaves +0.035% at t=0.20. Ships as an explicit null, which is what tonight's tape state deserves. |
| `08_gold_aug_friday_stretched.py` | **PUBLISH as a conditioning kill.** August Fridays in gold are +0.231% on 63-46, t=2.32, n=109. Split by how extended gold walked in, the whole cell lives in the calm version: 21d rank <= 85 pays +0.292% on 53-34 (t=2.61), rank > 85 pays -0.013% on 10-12. Gold enters tomorrow at a 21d rank of 88, so the seasonal cell that fires does not describe the state it fires in. |

## Final slate (6 nuggets, 0 anecdotes)

Tomorrow: SPY expiration-week follow-on [solid], ^VIX August expiration [suggestive],
HG=F expiration week [suggestive], GC=F August Friday conditioning kill [suggestive].
Today: QQQ 5-down-close streak depth split [solid], SPY/TLT/vol-bid null [suggestive].
