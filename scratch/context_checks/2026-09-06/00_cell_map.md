# Cell map — run 2026-09-06 (Sunday)

asof session 2026-09-04 (Fri) | next session 2026-09-08 (Tue) | Labor Day Mon 09-07 closed
midterm year | prices_fresh = True, freshest core bar 2026-09-04
sweep: 1217 cells scanned, 103 fired (72 event / 31 price), 12 cleared BH at crit p 0.0126

Three dates: run 2026-09-06 names these files, asof is 09-04, the brief previews 09-08.

## Data integrity gate (run first, 01_bar_integrity.py)

The 2026-09-02 brief excluded grains/coffee/cotton as continuous-contract roll
seams. STILL CONTAMINATED on the 09-04 bar. Prior-session volume against the
09-04 session:

| ticker | prev-bar volume | 09-04 volume | printed session move | verdict |
|---|---|---|---|---|
| KC=F | 18 | 15,188 | -9.70% | seam, not a move |
| ZC=F | 1,359 | 191,255 | +4.17% | seam |
| ZW=F | 129 | 100,203 | -0.44% | seam |
| ZS=F | 200 | 105,402 | +0.31% | seam |
| CT=F | 137 | 24,040 | +4.05%, Open = 0.00 | corrupt bar |
| CC=F | 2 | 11,927 | +1.95% | seam |
| SI=F | 71 | 40,410 | -0.23% | seam |
| GC=F | 72 | 186,451 | -0.32% | seam |
| HG=F | 1,659 | 29,974 | +1.38% | seam |
| PL=F / PA=F | 0 | 18,245 / 4,474 | +0.01% / -1.82% | seam |

Clean: SB=F, CL=F, NG=F, HE=F, LE=F, all FX, all indices, crypto, vol.

**Consequence**: every price-lane cell whose subject is a grain, a soft or a
precious/base metal is DEAD tonight on the state side, whatever its history
says. The 52w-high distances for SI=F (-41.9%) and PL=F/PA=F (-36%) are the
same artifact and are not quotable levels. This also removes the largest
printed session move in the tape (KC=F -9.70%) from consideration entirely.

## Event lane

| trigger | subjects | verdict |
|---|---|---|
| `E:holiday_post` | 18 | **DRILL**. This is the session Tuesday IS. Strongest cell in the sweep: ^VIX n=246 +3.83% h1, hit 69.5%, t 7.10, era-stable, BH pass. Equities flat beside it (SPY -0.003%, ^GSPC -0.019%). Generic-across-all-holidays is not specific enough to publish: drill to Labor Day itself, and condition on entering the holiday with vol already crushed (VIX 63d rank 9.1 tonight). Pre-specified in form (the holiday effect is famous), so it does not owe the sweep a BH correction, but the conditioning is search output and gets a control. -> 02, 03 |
| `E:holiday_post` GC=F/SI=F | 2 | **SKIP(seam)**. Both BH-passed (t 3.19 / 3.05) and both are metals whose current bar is a roll seam. The history is fine; I will not build a nugget whose subject I cannot honestly describe tonight. |
| `E:holiday_post` HYG | 1 | SKIP(weak). -0.098% at t -2.43, sign p 0.036, but HYG sits 0.41% off a 52w high with 2.6% realized vol. Nothing to say. |
| `E:seasonal_doy` (Sep 08) | 18 | **DRILL**. Heavy overlap with `E:holiday_post` by construction: Labor Day is the first Monday of September, so the Sep-08 trading-day-of-year cell IS mostly post-Labor-Day sessions. They disagree on VIX sign (holiday_post all-holidays +3.83%, Sep-08 midterm -2.22% with 5 of 6 down) and that contradiction is worth resolving rather than picking the flattering side. ^TNX midterm h5 +2.035%, 5 of 6 up, is separately relevant with the 10-year 0.25% off a 52w high. -> 02, 05 |
| `E:seasonal_doy` TLT | 1 | **SKIP(repeat_blocked)**. Published 2026-09-03 at -1.258%, 1 td ago. |
| `E:weekday_month` (Tuesdays in September) | 18 | **DRILL, then mostly SKIP**. n=111. ^VIX +1.867% t 2.43 collides with the holiday cell on the same session and is the weaker of the two; HG=F 41-68 down, sign p 0.011, BH pass, is a metal (seam) with no mechanism and I am not publishing a September-Tuesday copper cell off a sweep. Use only as the control layer for 03: is the post-holiday VIX pop just "a September Tuesday"? -> 03 |
| `E:ppi` (Sep 10, k=3) | 18 | **SKIP(anchor)**. At k=3 the h1 return is the session two days BEFORE the print, not the print. Nothing in the group is about PPI. NG=F +0.447% t 2.48 BH-passed and was already published 2026-08-10 at a similar number; SI=F/HG=F are seams; the equity arms are +0.05% noise. The PPI/CPI week belongs in the brief as calendar and as the 05 rates conditioning, not as a k=3 drift cell. |

Calendar entries inside five sessions, each with a verdict:
- Mon 09-07 Labor Day, closed. Drives `E:holiday_post`. PUBLISH via 02/03.
- Tue 09-08, Wed 09-09: nothing scheduled. This is why the brief leans on the
  holiday and state cells rather than an event.
- Thu 09-10 PPI, Fri 09-11 CPI: outside the h1 window, SKIP as cells, carried
  in the Calendar block and as the forward context for 05.
- Wed 09-16 FOMC + VIX expiry, Fri 09-18 opex/quad witching: 7 and 9 td out,
  Calendar block only.

## Price lane

| trigger | subject | verdict |
|---|---|---|
| `P5:rank5_extreme` bottom | KC=F | **DEAD(seam)**. The -9.70% is 18 contracts of stale prior bar. |
| `P5:rank5_extreme` bottom | EURJPY, CHFJPY, CADJPY, GBPJPY, AUDJPY, NZDJPY | **DRILL**. Six yen crosses in the bottom 5% of their year at once. EURJPY BH-passed (sign p 0.0004), CADJPY BH-passed (0.0046). -> 04 |
| `P6:two_atr_day` down | EURJPY | **SKIP(repeat_blocked)**. Published 2026-09-02. |
| `P6:two_atr_day` down | JPY=X | **SKIP(repeat in substance)**. Published 2026-09-03 at +0.102%, materially_moved so not hard-blocked, but the next-day-snapback claim is exactly what ran the last two nights and it failed twice in a row. Publishing a third telling of the same reflex would be the countdown failure mode with a different label. The yen goes in only on a genuinely new cell. -> 04 |
| `P6:two_atr_day` down | CADJPY, KC=F | SKIP. CADJPY 13-14 at t -0.05 is empty; KC=F is a seam. |
| `P7b:down_streak` | NZDJPY, CHFJPY | folded into **04**. Three crosses on 5+ down closes at once is part of the new framing. KC=F seam. |
| `P8:sma200_cross` down | NZDJPY | folded into **04**. First 200d cross in 63+ sessions. n=22, 11-11, t 0.01 on its own, so it is a state descriptor, never a forward claim. |
| `P5b:rank21_extreme` top | BTC-USD | **DRILL**. n=304, +0.734% h1, t 2.88, hit 56.6%, era-stable, BH pass, h5 +2.45%. The interesting conditioning is that this 21d top-5% (+26.8%) sits 17.7% BELOW the 52w high in a year that is -8.6%. -> 06 |
| `P5b:rank21_extreme` top | ETH-USD | SKIP(dup). Same trade as BTC, weaker (t 1.35, era-unstable). One crypto nugget, not two. |
| `P5b:rank21_extreme` top | ZC=F, ZW=F, ZS=F | **DEAD(seam)**. |
| `P5b:rank21_extreme` bottom | ^FCHI | **DRILL, low priority**. 21d rank 4.8 against ^GDAXI 34.9. n=400, hit 57.2%, sign p 0.0022, BH pass, but `era_stable: False`, so it publishes with the era split stated or not at all. -> 07 |
| `P4:z10_extreme` up | ZC=F, ZS=F | DEAD(seam). |
| `P4:z10_extreme` up | ^BVSP, AUDNZD | SKIP(empty). ^BVSP t 0.83 edge 0.014; AUDNZD t -1.41. Brazil already led a nugget 2026-09-03. |
| `P4:z10_extreme` down | EURAUD | SKIP(empty). t -0.37. |
| `P2/P2b:new_52w_low` | EURAUD | SKIP(thin, not macro). n=24 / n=14 on a cross Scott has no read on, t 1.64 / 0.07. |
| `P3c:pop50_after_low` | ^VIX3M | folded into **03**. 3-month vol printed a 52-week low then popped 1.09%. n=64, +0.52% h1, t 1.77, h5 +2.57%. On its own it is suggestive; as the state Tuesday's holiday cell fires into, it is the point. |
| `P5:rank5_extreme` top | EWZ | SKIP(dup). Brazil, published 09-03. |
| `P7:up_streak` | USDTRY | SKIP(not macro). A managed-devaluation series; 421 anchors of a one-way drift is a currency-regime artifact, not context. |
| dropped by cap | USDTRY, ^BVSP, AUDNZD | Reviewed and all three are SKIP above anyway. Nothing lost. |

## Not fired but examined (the sweep's blind spots tonight)

- **The whole curve at 52-week yield highs.** ^TNX 4.784% is 0.25% from its
  52w high, ^FVX 0.15%, ^IRX 63d rank 94.0; TLT sits 1.44% off its 52w LOW and
  IEF 0.44%. No P-trigger fires on "near an extreme without printing one", so
  this is invisible to the sweep and it is the most consequential standing
  state into a PPI/CPI week. **DRILL -> 05.**
- **Realized vol at 8.2% annualized with SPY 1.0% off a 52-week high.**
  Also unfired: no trigger tests compression itself. vol_vs_63d 0.67, VIX 63d
  rank 9.1, VIX3M 9.5, VVIX 13.9. **DRILL -> 03** (this is the state the
  holiday cell fires into, so the two merge).
- **Dow 21d rank 14.7 against Nasdaq 44.8.** Real but small in absolute terms
  (-0.87% vs +0.58% over 21 sessions). SKIP(thin) unless 07 finds nothing.
- Breadth 59.8% above the 200d, 62.1% 21 sessions ago. Unremarkable, no
  P11 cross. SKIP.

## Drill queue

- 01 bar integrity (done, gates everything above)
- 02 post-Labor-Day specifically, and its overlap with the Sep-08 doy cell
- 03 post-holiday vol conditioned on a crushed entry, vs the September-Tuesday control
- 04 the yen basket: a new cell, not the reflex bounce
- 05 the curve at 52-week yield highs into a CPI week, midterm September
- 06 bitcoin's 21d top-5% inside a down year
- 07 France against Germany at a 21d extreme


## Post-drill outcomes (written after stage C, corrections included)

Two verdicts above were WRONG and the drills overturned them. Left in place
rather than edited so the reasoning stays auditable.

1. **The overlap claim was wrong.** The map asserted `E:seasonal_doy` (Sep 08)
   and `E:holiday_post` are "mostly the same days". They are not: only 3 of the
   26 Sep-08 doy anchors are post-Labor-Day sessions (12%). The post-Labor-Day
   session is always a Tuesday and lands Sep 2 to Sep 8, so the two cells are
   near-independent. The contradiction the map wanted to resolve was never a
   contradiction.
2. **The real overlap is with `E:weekday_month`.** 02/03 found the September-
   Tuesday VIX cell (n=111, +1.87%, t 2.43) is entirely the 26 post-Labor-Day
   sessions: remove them and 85 anchors give 45-40 at +0.52%, t 0.67. That
   retires the engine's September-Tuesday ^VIX cell as an independent claim
   and became the lead nugget's sharpest line.

Drill results:

| # | verdict | outcome |
|---|---|---|
| 02 | PUBLISH | post-LD ^VIX 22-4, +6.28%, med +3.66%, t 3.30, sign p 0.0003 vs a +0.13% / 44.8% control. Equity leg 9-17 at -0.28%, midterm 1-5. Both shipped. |
| 03 | PUBLISH | eve session only -1.05%, so the pop is NOT weekend-decay reversal; the 2-session round trip is +5.10%. The low-vol conditioning I expected to matter DIED at n=3 and is not in the brief. Era +4.56% / +10.17%, top-2 = 37%, survives dropping them at +4.28%. |
| 04 | **DROPPED, empty** | USDJPY 63d rank <= 2 is nothing at every horizon (h1 +0.09% t 0.67, h5 negative). Basket breadth 5+ crosses at a 5d low gives a hit-rate-only h1 (32-17) with no magnitude, which is the same reflex already published 09-02 and 09-03. NZDJPY's 200d cross is n=24 at sign p 0.076 on a subject Scott has no read on. No yen nugget on the level framing. |
| 05 | PARTIAL | (A) shipped as an equity-side median deficit, +0.74% vs +1.45% neighbourhood over 21 sessions on 58 episodes; yields themselves do nothing (h21 +1.46% vs a +1.74% control). (B) the yields-high + VIX-crushed divergence is genuinely rare, 5 episodes since 2003 including this week, but n=4 usable is too thin for any forward claim and was cut rather than published as an anecdote. (C) ^TNX midterm Sep-08 h5 (+1.78%, 5-1) is the sign-flipped mirror of the TLT cell published 09-03 and repeat-blocked, so SKIPPED. |
| 06 | PUBLISH | the engine's BH-passing `[solid]` hint DOWNGRADED to suggestive on the split: the continuation is a new-highs effect (24-10, +10.15%) and Friday's deep-below-high version is 12-9 at +4.58% against an unconditional +4.05%. Two episodes carry 65%. |
| 07 | **SKIP, thin** | the base ^FCHI cell has a real h21 (+1.82%, 56-32, t 2.57, sign p 0.0069) but it is not about tonight. The live conditioning, France weak while Germany is mid-pack, is n=4 declustered and points the other way (h21 1-3). No nugget. |
| 08 | PUBLISH | new cell built from Friday's actual signature: 10y yield up while USDJPY falls 1%+, 96 declustered episodes in 26 years, USDJPY h1 72-24 at +0.38%, t 5.56 against a +0.00% / 51.3% neighbourhood. Shipped WITH the disclosure that a plainer form of the reflex led both briefs this week and missed twice. The ^GSPC h5 leg era-splits badly (+0.58% / -0.05%) and no equity claim is made at that horizon. |

Shipped: 5 nuggets, 3 tomorrow / 2 Friday, zero anecdotes, 401 words.
Sender QA: clean, no hard or soft findings.
