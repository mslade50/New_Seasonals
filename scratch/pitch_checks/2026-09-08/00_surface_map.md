# Surface map — 2026-09-08 (Tuesday, first session back from the Labor Day closure)

Bars are 2026-09-04 for 215 of 218 names. `DX-Y.NYB` and `^VIX` carry a
2026-09-07 holiday quote (^VIX's is a carried stub, do not trade off its
`ret_1d` of +5.30%); `LEG` is stale to 2026-08-27. The state file's
"216 tickers stale vs 2026-09-07" warning is that artefact, not a broken
cache: 2026-09-04 IS the prior session and the cache is current.

**The single most important fact about this morning: there are no new bars
since yesterday's run.** 2026-09-05/06/07 were the closure. Yesterday's
stand-down swept 20 candidates over ~1,700 cells against the identical tape,
targeting the identical entry session (2026-09-08). Every price-state cell it
killed is killed on today's numbers too, because they are the same numbers.
The only axis that moved is the CALENDAR, which advanced one session.

So this map weights the event lane heavily and treats the price-state lane as
a re-verification exercise plus a hunt for shapes the 3,600-line registry has
not run.

## Regime

| reading | value |
|---|---|
| fragility dial, 10d-MA 63d (the sizing statistic) | **88.0** as of 2026-09-04, 99th percentile of the 2016+ series; raw 21d 67.2, raw 5d 45.9 |
| dial 21 sessions ago | 57.2 — this is a fast climb, not a plateau |
| P/C fear state | OFF, equity P/C 52nd percentile (data 2026-09-04, 2 bd old) |
| fragility signals ON | VIX Range Compression only. **Note the 2026-09-07 correction: the production signal reads OFF in `risk_dashboard_v2.py:495-500` because it also requires VIX above its 20d SMA. No candidate may claim it is firing.** |
| exposure leg | 0.0x (raw-21d 67.2 > 50) |
| trend sleeve | CASH (dial 87.2 > 50) |
| staged scanner signals | zero |
| SPY | 770.19, -0.99% off its 52w high, +8.41% over its 200d, 107 sessions since a 5% pullback, 343 since a 10% |
| cycle | midterm (year%4==2), September |

The book is fully de-risked and holding nothing. That means a pitch today has
essentially no systematic overlap to disclose, which raises rather than lowers
the bar: nothing is being hedged or diversified, it is a naked directional add
in the dial's 99th percentile.

## 1. Every live calendar event x every asset class

Events in the window: **PPI 2026-09-10 (+2 td), CPI 2026-09-11 (+3 td), FOMC
decision 2026-09-16 (+6 td), VIX expiry 2026-09-16 (+6 td), opex 2026-09-18
(+8 td), quad witching 2026-09-18 (+8 td)**, and NFP 2026-09-04 (-1 td) behind
us. Next NFP 2026-10-02 (+18 td). Election 2026-11-03 (+40 td).

Structural facts worth stating before the grid, because they set most of the
verdicts:
- **PPI and CPI land on CONSECUTIVE sessions (Thu/Fri), and both sit inside a
  pre-FOMC window.** That triple is the only genuinely new coordinate on the
  board. The registry has PPI-on-equities (dead, 323 events), PPI-on-the-curve
  ("real but exactly one session wide"), and one line on the CPI-then-PPI pair
  at +0.002% on N=55 — the REVERSE ordering, which is what we have, is not
  recorded as measured.
- **September quad witching is already killed as "an FOMC anchor in costume"
  (2026-09-04).** Its FOMC-in-window half reads +2.382% over 11 years at 90.9%,
  which is live this year, but that number is a decomposition of a kill and the
  honest object it points at is the pre-FOMC run-in, not quad.
- **The pre-FOMC run-in in a midterm year is the event sleeve's T2 trade and it
  is SHORT.** T2 stages MOC on 2026-09-10 (4 sessions before the decision),
  gated on SPY 21d rank < 50 — SPY reads 31.3, so T2 will fire. Any long-SPY
  idea in this window is opposite the house sleeve and owes that in `overlap`.
- **The closure lane is dead.** 697 cells yesterday, nothing clears a
  family-wise bar; the extended-closure anchor adds nothing to an ordinary
  weekend on any class. Not re-run.
- **`data/macro_release_history.parquet` is frozen at 2026-08-07**, verified
  again this morning. The last CPI and PPI prints in it predate the August
  releases, so NOTHING may be conditioned on the prior CPI or PPI surprise
  today. That kills the whole surprise-conditioned event lane before it starts.

| event x class | verdict |
|---|---|
| PPI/CPI x us_large | **CHECK (C1).** The run-in to a back-to-back print pair has never been measured. PPI-alone on SPY is registry-dead (-0.009%, N=317) and post-NFP equity direction is dead; the pair is the new object. |
| PPI/CPI x us_small | CHECK inside C1 as the second vehicle. IWM r21 21.4 and -2.98% off its high, so it is the laggard leg of any large-vs-small expression. |
| PPI/CPI x rates | **CHECK (C2), with the tdom control mandatory.** The registry's standing correction is that "CPI/PPI/FOMC work on duration" is a trading-day-of-month profile: long TLT into CPI reads +0.178% raw and +6.7 bps against a tdom-matched control. Any duration-into-prints cell that does not carry that control is dead on arrival. |
| PPI/CPI x credit | Not examined as a lane of its own. HYG's CPI cell is already registry-flagged for the drifting-asset hit-rate trap, and this family has failed five times to produce a credit-specific residual (LQD = 0.485*IEF, HYG = 0.189*IEF + 0.446*SPY). Dismissed: the vehicle cannot express anything the rates or equity legs do not already carry. |
| PPI/CPI x gold_miners | Dismissed. GLD pre-CPI is registry-dead at +0.040% against GLD's own +0.092% h=2 drift, the GDX pre-CPI cell runs the other way and was killed, and the only cell with a pulse is gold ON the print day, which is not tradeable from here. |
| PPI/CPI x other_metals | Dismissed by the same entry plus the 2026-08-31 silver-complex work; SLV is -43.35% off its 52w high with nothing calendar-shaped attached. |
| PPI/CPI x energy | **CHECK (C7/C8).** This is the cross that has not been run: not energy washing out into a print (registry-dead, the CPI anchor SUBTRACTS) but the commodity complex at a 52-WEEK HIGH going into a back-to-back inflation print. DBC is -0.19% off its 252d high with r21 87.7, USO r5 87.7, XOP r21 91.7. |
| PPI/CPI x dollar_fx | Dismissed on a counted zero. The 2026-08-07 "count occurrences first" entry measured the dollar-into-CPI conditional cell at ZERO occurrences in 318 events, and DX into CPI is -3.5 bps, under both the 1.5 bp futures and the 6 bp UUP round trip. DX r21 39.3 is mid-range anyway. |
| PPI/CPI x international | Dismissed. The country-decoupling family is closed after five members (EWZ twice, FXI, SMH/QQQ, EFA) and no US macro print has ever shown a foreign-specific residual in this repo. EFA -0.42% and EWJ -0.19% off their highs is a real leadership fact (below) but it is not event-shaped. |
| PPI/CPI x volatility | **CHECK (C3), and note the live blocker.** Watchlist 33's arm disqualifies today by name: 2026-09-08 is the PPI k=-2 anchor at a runway of 1 session, and the entry requires >= 3. The CPI k=-2 anchor is tomorrow. So the parked cell is NOT tradeable today and C3 is the different question — what a two-print-in-two-sessions cluster does to the vol complex, where the runway conditioner is structurally 1 by construction. |
| FOMC x us_large | Dismissed as a standalone: the pre-FOMC drift is the most-documented cell in the repo, the production Pre-FOMC Rally signal reads OFF, and the sleeve trades the midterm version SHORT from 09-10. Folded into C1 as a conditioner instead. |
| FOMC x us_small | Dismissed with the quad kill: the IWM/SPY laggard gate on the Sept-quad run-in is worth +0.006pp and the 63-day floor applied year-round pays -0.266% over 160 episodes at a 45.0% hit. |
| FOMC x rates | Folded into C2's conditioner set (FOMC inside the hold), not run standalone; the duration-into-FOMC cell shares the tdom confound above. |
| FOMC x volatility | Dismissed. VIX expiry and opex are ONE anchor (189 shared of ~200), the VIX-expiry-week drift is registry-dead as mid-month position plus noise, pre-expiry short-vol carry is dead, and the September VIX settle was killed again on 2026-09-07 as the 2026-08-07 corpse at P(corpse mask) = 1.0000. |
| FOMC x energy / metals / credit / FX / international | Not examined. Reason: with the decision 6 sessions out, any pre-FOMC cell on a non-equity class has to clear both the tdom control and the FOMC-drift parent, and the repo has no instance of a non-equity FOMC cell surviving either. The cost of the look exceeds its prior. |
| opex/quad x every class | Dismissed wholesale on the 2026-09-04 kill: the Sept run-in is an FOMC anchor in costume, the reference class over 16 index and industry ETFs is homogeneous (Q 13.50 on 15 df, I-squared 0.0%) with IWM 6 of 16, and every year has a quad. |
| vix_expiry x every class | Dismissed, same anchor as opex; see above. |
| NFP (behind us) x every class | Dismissed. Yesterday's and 09-04's work closed the post-NFP lane on equities, rates (midterm-blocked, watchlist 0) and vol. Watchlist 37's prior-surprise conditioner is unreadable with the release file frozen. |
| election (+40 td) | Outside any 1-10 td horizon. Not examined. |

## 2. Tape extremes by class

All readings 2026-09-04 unless stated.

**us_large** — SPY 770.19, -0.99% off its 52w high, r5 41.7 / r21 31.3 / r63 47.6,
ATR 0.81%, 21d realised vol 8.2%. QQQ -3.54% off, r63 30.6. DIA r21 15.1, the
weakest of the majors. Nothing is at an extreme; this is a quiet index a
percent under its high. **No price-state candidate here.**

**us_small_breadth** — IWM -2.98% off its high, r21 21.4, z10 -0.53. Lagging but
not extreme. Enters only as C1's and C6's second vehicle.

**rates** — the sharpest picture on the board. **^TNX 4.784, within 0.25% of its
52-week HIGH**, r21 69.0 / r63 75.8. **TLT within 1.44% of its 52w low, IEF
0.44%, LQD 0.25%** — the whole investment-grade complex pinned at the floor.
Verdicts: watchlist 5 (TLT floor, freshness leg cleared 09-07) still needs TLT
to CLOSE at or below 81.44 and it sits at 82.21, so **PASS, and note the
standing constraint that it may never be expressed as a resting limit at that
level** (its entire edge is session +2; the session a limit fill adds pays
-0.302% at a 38.9% hit). Watchlist 18 (IEF vs 0.523 TLT with the 10y at a 252d
max) — **PASS**, the max touch lapsed on the 09-04 bar (4.784 against a 4.796
max set 09-01). Watchlist 30 (yield high x bond vol mid-band) — **PASS**, ^MOVE
level percentile 62.7 against the [40,50) band. Watchlist 10 (November TLT) —
**PASS**, parks to November. Watchlist 0 (post-NFP TLT) — **PASS**,
midterm-blocked to 2027-01.

**credit** — **HYG within 0.41% of its 52-week HIGH** at 2.6% realised vol, the
calmest instrument on the tape; LQD within 0.25% of its 52w LOW. Verdicts:
watchlist 26 (IG at lows while HY prints a high) — **PASS**, HYG needs <= 0.25%
off its high and reads 0.41%, and the entry's own second arm (a genuine
credit-specific residual) is unmet in any case. Watchlist 1 (long LQD / short
HYG at joint extremes) — **PASS**, the state is live but the arm is >= 8
declustered episodes over >= 3 years ex-2018 and the corrected count is FIVE
episodes in two years. Watchlist 41 (HYG out of the closure) — **PASS and
today is exactly the anchor it named**: the arm requires HYG more than 1% below
its 252d high OR 21d realised vol above 4.4%, and HYG reads -0.41% and 2.6%,
which is the dead bucket the entry was killed in.

**gold_miners** — GLD -17.97% off its 52w high, r21 54.4, 21d realised vol
27.5%. GDX -14.31% off, r21 79.8, r63 64.3, but -2.20% on the day. NEM r21 90.5.
Verdict: watchlist 3 (miner-led thrust the metal has not joined) — **PASS**,
GDX r5 41.3 against the >= 95 arm and GLD -17.97% against the within-10% arm,
both legs failing. The short-GLD analogue was killed outright yesterday. No
candidate.

**other_metals** — SLV -43.35% off its 52w high (that distance is the April-2026
spike unwinding), r21 54.4. FCX r5 17.5 after -4.87% in five sessions, +19.65%
over its 200d. Verdict: watchlist 29 (short silver after a complex break) —
**PASS**, the complex bounced and the depth arm needs a -4.00% break. No
candidate.

**energy** — the leadership block. **DBC -0.19% off its 52-week high**, r21 87.7.
USO r5 87.7 / r21 86.5, +32.07% over its 200d. XOP r21 91.7, XLE r21 83.3, both
within 1.6% of their 52w highs. OIH r63 23.0 is the laggard of the group.
Verdicts: watchlist 4 (XLE on a crude thrust in the 5-6% band) — **PASS**, USO's
1-day move is -0.09%. Watchlist 7 (fade a crude thrust out of a deep base) —
**PASS**, USO r63 63.1 against the <= 20 deep-base leg. Watchlist 19 (narrow
energy thrust cluster) — **PASS**, the count of the 11-name complex at z10 >= 2.0
is 0 against the [2,3] arm. Watchlist 32 (XLE at a fresh high on a down-SPY
session) — **PASS**, XLE -1.60% off its 252d max. **But the class carries C7/C8/C9**:
none of those entries is about the commodity index itself printing a 52-week
high into an inflation print, which is the cross nothing has run.

**dollar_fx** — UUP -1.82% off its high, r21 32.1, r63 17.1. DX-Y.NYB 99.176
(2026-09-07 quote), r21 39.3, r63 16.3, ATR 0.43%. Verdicts: watchlist 15
(short the dollar on an unconfirmed rate rise) — **PASS**, DX r21 39.3 against
<= 20. Watchlist 13 (gold on an unconfirmed rate rise) — **PASS**, same leg.
Watchlist 23 (the bare dollar washout) — **PASS**, midterm-parked and DX is not
washed out. The dollar is mid-range on every measure; there is no candidate here
and the class is honestly empty rather than unexamined.

**international** — **EFA -0.42% and EWJ -0.19% off their 52-week highs** while
SPY is -0.99%, QQQ -3.54% and IWM -2.98% off theirs. EEM r5 73.4 after +1.82% on
the day; EWZ r5 95.2 with z10 1.60; FXI -12.47% off. Verdict on the obvious
candidate (foreign developed printing highs the US has not): **dismissed on the
closed family.** The country-decoupling family has five dead members and EFA's
sustained-leadership form is named in the 2026-08-25 entry as one of them; the
2026-08-18 EWJ work died on the reference class at P(max-of-10 >= EWJ) = 0.477.
Reviving it needs a mechanism this state does not supply. Watchlist 9 (FXI break
inside an intact thrust) — **PASS**, FXI r5 62.3 against <= 20.

**volatility** — ^VIX 15.3 (2026-09-07 stub), -50.72% off its 52w high. **^VIX3M
17.61, within 1.09% of its 52-week LOW**, r63 9.5. Term structure ~15% contango.
**^SKEW 151.58 with a 21-day return rank of 98.0**, the second-highest r21 on the
whole 218-name tape, and +12.51% over 21 sessions; its trailing-252 LEVEL
percentile was 49.2 on 09-03, so the spike is in the rate of change, not the
level. SVXY -0.27% off its 52w high, r63 84.1, z10 1.18. UVXY -74.50% off.
Verdicts: watchlist 6 (skew spike, 5-day form) — **PASS**, ^SKEW r5 61.1 against
the >= 95 arm, and its arm additionally requires a non-midterm year. Watchlist 12
(vol pop inside a calm tape) — **PASS**, wrong-signed. Watchlist 33 (SVXY into a
print out of a compressed range) — **PASS**, disqualified on runway as above.
Watchlist 38 (SVXY at the first close after a closure) — **PASS**, the entry
disqualifies 2026-09-08 by name at runway 2. The ^VIX3M 52-week floor and the
SKEW-over-VIX3M ratio are both registry-dead (2026-08-27 and 2026-08-31) and are
not re-opened. **The 21-DAY skew form is none of those and is not on the
watchlist: it is C4/C5/C6.** Yesterday's own measurement, taken as the control
leg of a conjunction it killed, put SKEW's 21-day rank alone at SPY +0.333%
(108-58, sign p 0.0001) and SVXY +1.374% (59-36, p 0.0117). That parent is live
today at 98.0 and has never been checked on its own terms.

**sectors and subsectors** — **ITA r21 2.8** (-9.75% over 21 sessions, -10.90%
off its 52w high), the deepest 21-day reading on the tape and watchlist 40's
live cell. **XLI r21 7.1** but -6.03% off its high. XLRE r5 21.8 / r21 15.9 /
r63 17.9. **SMH r63 5.2** with a +2.61% session and MU +67.65% over its 200d, the
widest intra-complex dispersion on the board; IBB r63 96.8 and XBI r63 86.1 are
the other side of that rotation. XLF r63 89.3. IHI r5 14.3. A **food and staples
subgroup flush** sits under a flat sector: CPB -8.59% and GIS -7.85% and TSN
-6.32% over five sessions with r5 of 2.8 / 4.0 / 2.8, HRL r21 3.2, SYY z10 -1.89,
while XLP itself is only r5 27.4 and -4.98% off its high. Two utilities have
blown up idiosyncratically (EIX -19.10% in five sessions at 96.2% realised vol,
PCG -13.86%) while XLU r21 is 47.6.

Verdicts: watchlist 40 (ITA) — **CHECK, and it is the cheapest deep dive
available**, because its arm is not a number but a 13-name subsector reference
class that has never been run, and the cell is live at its deepest reading.
Watchlist 21 (the sector washout as a family) — **PASS**, XLI clears the washout
leg but is -6.03% off its high against a within-5% clause, and the pooled family
is registry-dead. Watchlist 34 (pooled sector triple floor) — **PASS**, only XLI
holds all three floors, down from nine names. Watchlist 25 (SMH at a 63d floor
in a top-decile year) — **PASS**, the still-falling arm now fails outright at r5
56.7 after the +2.61% session. Watchlist 28 (pooled r21>=90 & r63<=10) — **PASS**,
no holder anywhere on the tape. Watchlist 8 (IHI at r21 100) — **PASS**, r21 50.4.
Watchlist 14 (XLK vs XLV rotation gap) — **PASS**, the one-day gap is -1.11pp
against a >= +3.0pp rung. Watchlist 17 (KRE vs XLF breadth washout) — **PASS**,
the arm is a cost threshold no session moves. Watchlist 22 (XLU washout with the
long end hit) — **PASS**, XLU r21 39.4 against <= 5. **The food-subgroup flush is
NOT on the watchlist and is not in the registry: it is C10/C11**, and it is
the same reference-class question as ITA, which is why the two share a checker.

**Two single-name blowups deliberately not pitched.** EIX and PCG are the two
weakest names on the tape by a distance (-19.10% and -13.86% over five sessions,
r5 2.0 and 2.0, realised vol 96.2% and 86.1%). The registry's 2026-09-03 entry on
a utility down >= 20% in five sessions names this as the definition of an
unverifiable mechanism: no news source is available to this product, so a
regulatory or wildfire repricing is indistinguishable from a flush. Dismissed on
that ground, not on a number.

## 3. Seasonal and cycle cells

September, trading day 6 of the month, midterm year. Cycle state is a
CONDITIONER on everything above rather than a lane of its own, and it is a
consequential one today:
- The board's own regime read: midterm book win% 56.4 vs 64.9 all-years over
  1,099 trades, +0.24R vs +0.43R. Three sleeves flagged "fade in midterm".
- Watchlist 0, 23 and 27 are all parked to a non-midterm year and stay parked.
- Watchlist 6's arm requires a non-midterm year, which blocks the 5-day skew
  form independent of its rank leg.
- The Labor Day seasonal is dead in both directions (long 0-for-8 since 2018,
  the post-holiday short is a fixed September trading-day-4 anchor wearing a
  holiday label, and IWM's forward-10 after the holiday is +0.713% against
  -1.452% before).
- **Verdict: no standalone seasonal candidate.** Every September cell in reach
  is either registry-dead or is the pre-FOMC run-in in costume. Cycle enters
  C1, C4 and C7 as a mandatory split instead.

## 4. Watchlist — all 42 entries carry a verdict

Covered by name in sections 1-3 above: 0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13,
14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 28, 29, 30, 32, 33, 34, 38, 40, 41.
Remaining:

- **2** (SVXY overnight into CPI) — PASS today, CHECK on 2026-09-10. The anchor
  is the CPI-eve close, which is Thursday, not today. Its own arm is a
  beta-neutral drop-best-year floor of 40-50 bps against 19.7 today.
- **11** (short SPY at a 52w high with the long end at a 52w low) — PASS. SPY
  -0.99% off against a <= 0.5% touch; TLT +1.44% above its low against <= 1%.
  Both legs fail, the SPY one by 49 bp.
- **16** (short TLT after a big up day from the low zone) — PASS. TLT's 1-day
  move is +0.17% against a >= +1.5% thrust rung.
- **20** (cross-sectional new-high breadth with the index off its high) — PASS,
  both legs failing: SPY -0.99% off against a > 2.0% arm, and raw-21d fragility
  67.2 against <= 50.
- **24** (HY at a fresh high while the index has not) — PASS. HYG -0.41% off its
  252d high against the <= 0.05% touch, and the dial arm (< 50) fails at 88.0.
- **27** (IEF out of Jackson Hole) — PASS, midterm-blocked and the anchor is
  weeks gone.
- **31** (December small-cap month-end overnight) — PASS, parks to December.
- **35** (long SPY into a print out of a dead VIX range) — PASS on the dial arm,
  which is 88.0 against a requirement well below it.
- **36** (short the index across an extended closure) — PASS and it is now
  HISTORY: the closure it would have traded ended this morning. Its arm needs
  two NEW closures cleared by a forward-written rule; the next boundaries are
  2026-11-26 and 2026-12-25.
- **37** (post-NFP duration on the prior print's surprise) — PASS, and
  structurally unreadable: `macro_release_history.parquet` is frozen at
  2026-08-07 so the prior print's surprise cannot be read this morning.
- **39** (SPY vs IWM in the dial's 56-70 band) — PASS. The dial is 88.0, some
  18 points above the band, and yesterday's work showed [70,80) pays +0.071% on
  a 6-8 record, so the extreme is the part of the mask with no content.

Nothing on the watchlist arms today except **40 (ITA)**, whose arm is a piece of
work rather than a number, and it is being done.

## 5. Axis feedback read

The scoreboard carries 5 graded ideas lifetime (avgR +0.174, 4 of 5 positive;
B-grades +0.448 on 3, C-grades -0.237 on 2). Per-axis it is 1 or 2 ideas per
axis. That is too few to steer selection and no axis is being weighted up or
down on it. Recording the read and moving on, per the skill.

## 6. Candidates selected from this map

Twelve, over four checkers, five novelty axes, and the asset classes us_large,
us_small, rates, volatility, energy/commodities and sectors/subsectors. At
least one is event-anchored (C1, C2, C3, C7, C8, C9) and at least one price-state
anchored (C4, C5, C6, C10, C11, C12), and C7/C8/C9 cross the two modes, which
is the specific failure the 2026-08-07 rewrite was aimed at.

| id | candidate | axis | class |
|---|---|---|---|
| C1 | Long SPY (and IWM) MOC into the back-to-back PPI+CPI pair, held to the far print | event_fingerprint | us_large / us_small |
| C2 | Long duration (IEF, TLT) across the consecutive-print pair, tdom-controlled | event_fingerprint | rates |
| C3 | Long SVXY / short vol across a two-print-in-two-sessions cluster, the structurally-zero-runway case watchlist 33 excludes | interaction_cell | volatility |
| C4 | Long SPY on ^SKEW's 21-DAY rank at or above 90, the never-checked parent of watchlist 6's 5-day form | inversion | us_large / volatility |
| C5 | Long SVXY on the same 21-day skew state | instrument_translation | volatility |
| C6 | The cross-sectional form: SPY over IWM on a 21-day skew spike | relative_value | us_large / us_small |
| C7 | Long the commodity complex (DBC, USO) at a 52-week high into a back-to-back inflation print | interaction_cell | energy / commodities |
| C8 | Short duration on the same commodity-high-into-print state, the rates expression of C7 | interaction_cell | rates / energy |
| C9 | Energy leadership continuation (XLE, XOP at r21 >= 83) with a print inside the hold | event_fingerprint | energy |
| C10 | Long the food and staples flush basket against XLP, the subgroup washout inside an intact sector | relative_value | sectors |
| C11 | The short side of C10, which is where this repo's four prior confirmations point | inversion | sectors |
| C12 | **Watchlist 40's arm**: run the 13-name subsector reference class on ITA's washout-under-a-high-index cell and either arm it or kill it | historical_analogue | subsectors |


---

# Post-check corrections and outcome (appended after stage C)

All twelve candidates were killed. The morning stands down. Three corrections
this map owes, written here rather than silently edited, because the map is the
audit artefact.

**Correction 1 — the novelty claim in section 1 was FALSE.** This map said the
PPI-then-CPI ordering "is not recorded as measured". It is. Checker 1
reproduced it from the 2026-08-10 work: PPI-then-CPI at print-2, h=1, reads
**-0.1135% on N=127** against the registry's -0.071% on N=133. Today's ordering
is the NEGATIVE side of a pair the repo already measured. The correct reading of
that registry line is that BOTH orderings are recorded and the one we have is
the worse one.

**Correction 2 — watchlist 40's published ITA numbers do not reproduce, and its
quoted worst episode is an artefact.** `_survey_lib.align()` in the 2026-09-07
folder applies union-reindex-then-ffill to a FORWARD-RETURN series, which is
NaN for the last lag+h rows by construction. The ffill carries the last
resolvable value into them, minting a phantom episode dated **2026-09-02** whose
booked returns are -3.097% (h=5) and **-4.942% (h=10) — which is exactly the
"worst episode -4.94%" the watchlist entry quotes.** Clean: N=42 / +0.629% at
h=5 and N=28 / +1.443% at h=10, against the published 43 / +0.543% and 29 /
+1.223%. Checker 2 found the same bug independently on the SVXY leg, where it
smeared one h=10 return across **11 sessions ending on today's own anchor, 3 of
them inside the live mask**. Any other 2026-09-07 cell built on `_survey_lib`
with a live trigger carries the same phantom. This is a scratch survey helper,
not production, and `pitch_lab` is unaffected.

**Correction 3 — the ^SKEW figures carried out of 2026-09-07 are RAW, not
excess.** SPY's +0.333% is +0.142pp over an all-days control at h=5 and
**-0.111pp at h=10**; SVXY's +1.374% is +0.738pp pooled and -0.016pp at h=10.

## What each candidate died of

| id | verdict | substantive kill |
|---|---|---|
| C1 | KILL | the pair gate selects the worse half (-0.115pp against a +0.024pp complement); placebo rank 8 of 11 |
| C2 | KILL | mechanism falsified in-window — only 7.9% of the hold accrues in the two release gaps |
| C3 | KILL | definitionally the dead runway-1 bucket (N=139 vs N=139); the reframing is wrong-signed on the unlevered vehicle |
| C4 | KILL | a filter that does not filter (+0.021pp), dose response backwards, midterm sign flip |
| C5 | KILL | no vol-specific residual; 70% of the surviving cell is 2023 |
| C6 | KILL | mechanism falsified in-window — the IWM leg pays more than the SPY leg |
| C7 | KILL | regime concentration — four inflation-shock years hold >100% of the total |
| C8 | KILL | the live joint state is the negative half; the rescuing 5-for-5 cell is a lucky subset, not a filter |
| C9 | KILL | the state's own gradient runs backwards and the live band is its worst cell |
| C10 | KILL | generic short-term reversal with a food label — +0.041pp at t +0.11 against a broad flushed basket |
| C11 | KILL | mechanism falsified in-window — the object is positive at every horizon 1 through 10 |
| C12 | KILL | class-wide effect, ITA is the top draw of 13 correlated subsectors; max-of-13 P = 0.6892 against a <= 0.10 arm |

## Watchlist consequences

- **Entry 40 (ITA) is RETIRED, not re-parked.** Its arm ran and returned
  0.6892 against a required 0.10, and ITA's observed excess sits BELOW the
  null's median best draw on all four bases. Per the reference-class precedent
  a revival needs P below 0.05.
- **Entry 33 (SVXY pre-print compression) gains an amendment** rather than a
  duplicate: the second-print rung is a distinct anchor on the same 2026-09-11
  session the entry already names.
- Two new entries from C8 and C1.
