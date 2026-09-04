# Cell map — run 2026-08-18 (Tue), asof session 2026-08-18, next session 2026-08-19 (Wed)

Midterm year. Prices FRESH (core bar 2026-08-18). 1231 cells scanned, 93 fired
(72 event / 21 price). BH crit p 0.0, 1 pass (USDTRY, see below). No cells
dropped by cap, no engine errors. `delta_suppressed` false.

Top-tier ahead: NONE. VIX expiry is tomorrow (2026-08-19), monthly opex Friday
(2026-08-21), Jackson Hole 2026-08-28. So the tomorrow lane is process events,
not prints, and the price lane fired 21 cells, so this is not a quiet tape.

## Novelty state carried in

Last night (2026-08-17) published five nuggets. That constrains tonight:

- `E:seasonal_doy|TLT` is `repeat_blocked` (published 2026-08-17, td_since 1).
  Not eligible tonight in any form.
- `E:vix_expiry|*|k2` published last night as the headline. Tomorrow the event
  ESCALATES from "two sessions out" to "tomorrow is the expiry". The spec
  allows exactly one re-telling on escalation and only with NEW specificity.
  The k1 cell is a different measurement, not a countdown: k2's h1 was the
  pre-expiry session (SPY +0.190%, t 2.87), k1's h1 is the expiry session
  itself, and the engine has it at SPY -0.104%, t -1.53. Opposite sign, same
  event. That is new content, and the drill has to establish the pairing
  rather than just re-quote the event name.
- `P:vol_bid_shallow_tape|^VIX` published last night (VIX +5%+ on a decline
  under 1%). Today repeated the shape: VIX +4.28% on SPY -0.68%. Same claim,
  no new number. NOT eligible.
- `P1:new_52w_high|ZC=F|two_atr` published last night. Corn added another
  +4.89% today. A second consecutive outsized up day, a 5d return in the 100th
  percentile of its own year, and the whole grain complex extreme at once are
  all facts last night's cell did not contain, so a corn nugget is only
  eligible if the drill is about the EXTENSION or the COMPLEX, never about the
  breakout again.

## Event lane — every fired trigger group gets a verdict

### E:vix_expiry (18 subjects, n_anchors 319, k=1, h1 = the expiry session)
- **SPY / ^GSPC — DRILL.** SPY -0.104% t -1.53, ^GSPC -0.091% t -1.35, both
  era-stable, edge -0.143pp / -0.123pp vs all days. Weak on its own and would
  be SKIP in isolation. It is a DRILL because of the pairing with last night's
  k2 cell: the run-up into the expiry and the expiry session are opposite
  signs off the same 319 anchors. Drill 01 tests whether that is one round
  trip or two unrelated numbers, plus the August and midterm conditioning.
- **SI=F — DRILL.** +0.225%, 176-135, sign p 0.0116, t 1.98, era-stable, the
  best single subject in the group. Folded into drill 01 as the cross-asset
  half so the expiry item is not purely an equity claim. Note silver fell
  4.08% today, so the state and the cell point opposite ways.
- **GC=F — SKIP.** +0.124% t 1.95, p 0.0347, real but a weaker twin of SI=F;
  publishing both is one fact twice.
- **EURUSD=X — SKIP(thin).** t 2.31 but the mean is +0.085%, inside the pair's
  own daily noise, and the sign-test p is 0.1006. Not worth a line.
- **TLT — SKIP.** +0.073% t 1.28. Also collides with the blocked TLT seasonal.
- **^VIX — SKIP(covered).** 144-173 down, p 0.0727, era_stable false, and the
  August version of this anchor was last night's item 2.
- **QQQ, IWM, ^TNX, HYG, CL=F, NG=F, HG=F, EEM, IEF, DX-Y.NYB, JPY=X — SKIP.**
  All |t| < 1.6 with edges inside 0.12pp; nothing to say.

### E:opex (18 subjects, n_anchors 319, k=3, h1 = tomorrow)
- **TLT + IEF — DRILL.** TLT +0.141% t 2.46, 161-127, sign p 0.0258,
  era-stable; IEF +0.064% t 2.12, 155-128, era-stable. Two independent
  durations of the same trade agreeing is the reason this is interesting and
  not just one lucky series, and ^TNX sits at -0.236% t -1.51 on the other
  side of the same claim. Drill 02 checks whether the bid is the whole opex
  week or only this anchor, the era split, and concentration.
- **SI=F — SKIP(duplicate).** +0.260% t 2.20, but silver already carries the
  VIX-expiry cell and the two anchors are two sessions apart on overlapping
  windows. One silver claim per brief.
- **GC=F — SKIP.** t 1.49.
- **^TNX — folded into drill 02** as the confirming side, not its own nugget.
- **HG=F — SKIP.** t -1.04, era_stable false.
- **SPY / ^GSPC / QQQ / IWM — SKIP.** All |t| < 1.0, edges around -0.1pp.
- **EEM, EURUSD, HYG, JPY=X, ^VIX, CL=F, NG=F, DX-Y.NYB — SKIP.** |t| < 1.5,
  nothing survives its own control.

### E:weekday_month — Wednesdays in August (18 subjects, n_anchors 118)
- **CL=F — DRILL.** -0.385%, t -2.06, n=114, era-stable, edge -0.377pp. The
  strongest |t| in the group and the only one with a NAMED mechanism: the EIA
  petroleum status report prints Wednesday 10:30 ET, so the weekly inventory
  number lands inside this bar. Drill 03 has to separate "Wednesdays" from
  "August Wednesdays" or the August framing is fake precision.
- **HG=F — SKIP(no mechanism).** -0.251% t -1.92, era-stable, but copper has
  no Wednesday release and this reads as the same commodity-Wednesday shape
  without the print that explains crude.
- **^GSPC / SPY / QQQ — SKIP(weak mean).** ^GSPC 71-47 with sign p 0.0169
  looks tidy, then the mean is +0.098% at t 1.01, a hit-rate artifact with no
  magnitude. Publishing a 60% hit rate worth a tenth of a percent misleads.
- **EEM — SKIP.** 62-42 p 0.031, mean +0.109% t 0.77. Same artifact.
- **SI=F, GC=F, NG=F — SKIP.** t < 2, and gold/silver already contested above.
- **^VIX, TLT, IEF, HYG, ^TNX, DX-Y.NYB, JPY=X, EURUSD, IWM — SKIP.** |t| < 1.1.

### E:seasonal_doy — same trading day of year (+/-2), Aug 19, midterm phase
- **QQQ + IWM midterm — DRILL.** QQQ 6-0 up, mean +0.465%, sign p 0.0156;
  IWM 6-0 up, mean +0.839%, sign p 0.0156. Two indices, same six midterm
  years, both perfect. N=6 each, so this is anecdote-tier by the tag rules and
  can never headline, but a clean record with a real per-event magnitude is
  publishable as context. Drill 04 has to check whether it is six independent
  years or one repeated regime, and whether the all-year cell agrees.
- **SPY / ^GSPC all-years — folded into drill 04** as the wider control
  (SPY 18-8 p 0.0378, mean +0.253%).
- **TLT — BLOCKED.** `repeat_blocked` true, published 2026-08-17.
- **HYG — SKIP(N).** midterm N=4, 4-0, p 0.0625. Too thin next to two 6-0
  cells that say the same risk-on thing.
- **IEF, ^TNX, GC=F, SI=F, HG=F, CL=F, NG=F, DX-Y.NYB, EURUSD, JPY=X, ^VIX,
  EEM — SKIP.** Every midterm sign test is 0.19 or worse; most are 3-3.

## Price lane — every fired trigger group gets a verdict

- **P4:z10_extreme USDTRY=X — SKIP(degenerate).** The sweep's ONLY BH pass
  (n=582, 69.1% hit, t 3.16, bh_pass true) and it is still not a nugget. The
  lira depreciates structurally, so "stretched up, keeps going up" is the
  instrument's drift wearing a trigger's clothes, and the cell fires
  perpetually rather than marking news. Tag_hint `solid` is a floor I am
  declining to take. Recorded explicitly because skipping the one BH pass
  needs to be visible.
- **P4:z10_extreme SB=F — SKIP.** Sugar +3.62% today, z10 2.79, but the cell
  is t 0.08 and era_stable false. The state is real, the history is empty.
- **P1 / P1b / P6-up / P5-top ZC=F — DRILL.** Corn at a 52-week high on
  +4.89%, 5d +11.68% at the 100th percentile of its own year. Published last
  night as a breakout, so drill 05 must be about what is new: the second
  consecutive outsized up day, and the fact that soybeans (5d 99.2 pctile) and
  wheat (97.2) are extreme at the same time. A three-grain simultaneity cell
  is a different question from corn's own 52-week highs.
- **P5:rank5 ZS=F / ZW=F, P7:up_streak ZS=F — folded into drill 05.**
- **P7b:down_streak + P5:rank5-bottom + P5b:rank21-bottom HE=F — DRILL(kill
  check).** Lean hogs 5d -15.80%, 21d -20.29%, rank21 0.4. A move that size
  demands the roll-gap check before it is believed at all; the continuous
  contract stitches expiries and HE=F is the known offender in this repo.
  Drill 07 exists to kill it, not to publish it.
- **P5:rank5-bottom LE=F — same treatment**, folded into drill 07.
- **P6:two_atr_day PL=F down — SKIP.** Platinum -3.46%, cell +0.143% t 1.06,
  126-94 p 0.0217. A hit-rate tilt with no magnitude, and platinum is a thin
  subject for a macro brief.
- **P6:two_atr_day USDCNY=X down — DEAD.** -0.128% clears 2 ATR only because
  managed-CNY realised vol is near zero. Degenerate.
- **P7b:down_streak ^BVSP — SKIP.** +0.386% t 1.84, 85-63 p 0.042, era-stable.
  The most publishable of the leftovers, but Brazil is peripheral to Scott's
  next session and it loses the ranking to better items.
- **P7b:down_streak ^FCHI — SKIP(era).** 78-50 p 0.0083 looks strong, then
  era_stable is false and the mean is +0.132% at t 0.72.
- **P7:up_streak AUDJPY=X — SKIP.** +0.067% t 1.66. Below the noise floor of
  the cross.
- **P5b:rank21 CHFJPY=X — SKIP.** +0.061% t 0.80.
- **P5b:rank21 SB=F, EURCHF=X — SKIP.** t 0.50 and t -0.75.
- **P5:rank5-top ZW=F, ZS=F — see drill 05.**

## Not fired but present in the tape, and looked at anyway

- **QQQ -1.69% against SPY -0.68% — DRILL.** No trigger covers relative
  performance, so the sweep is silent on the most distinctive thing the
  session did: a 1.01pp large-cap-tech underperformance on a day the index
  barely moved. Drill 06 builds the cell from the tape rather than inheriting
  it. Flagged here as a tape-derived cross, not an engine trigger, so the
  provenance is on the record.
- **TLT 81.66, 0.38% off its 52-week low** — the state behind last night's
  blocked TLT item. Not republished.
- **^VIX 15.84, 17.1st percentile of its own year on 21d** — context for the
  expiry item, not a nugget.

## Drill queue

01 vix expiry: the run-up vs the expiry session itself (SPY, ^GSPC, SI=F)
02 opex week: the bond bid at k3 (TLT, IEF, ^TNX)  + 02b the k3->k0 window
03 august wednesdays in crude, against all wednesdays (CL=F)
04 aug 19 seasonal, midterm phase (QQQ, IWM, SPY control)
05 grains: corn's second leg and the three-grain simultaneity
06 qqq underperforming spy on a shallow index decline + 06b the spread leg
07 kill check: HE=F / LE=F continuous-contract roll gaps
08 rotation day: sector dispersion under a shallow index decline (added)
09 the remaining fired price cells: ^BVSP, PL=F, SI=F (added)
10 ^BVSP: the streak that is actually live, not the one that fired (added)
11 ^BVSP: verify the run-length rarity claim before printing it (added)

## Drill outcomes — four cells killed, five survive

**KILLED 05, grains.** The simultaneity cell (corn, soy and wheat all >=95th
pctile on 5d, 24 declustered episodes since 2002) is FLAT at h1: -0.118% on
12-12 for the basket. It looks alive at h10 (+1.688%, 15-9, edge +1.395pp) and
then the top two episodes carry 45% of the total, which the concentration rule
requires me to disclose and which kills it at that N. Corn's own extension
cell, two consecutive 1.5-ATR up days into a 252-day high, is 11 episodes and
6-5 at h1. Nothing here is publishable and last night's breakout may not be
retold, so grains are OUT entirely.

**KILLED 06, QQQ underperformance.** The index legs are noise (every horizon
inside 0.2pp of control). The relative leg looked real: QQQ/SPY -1.100% over
21 sessions on 50-75, t -2.01, against +0.094% for all 21-session windows,
holding at t -3.00 after dropping the two largest episodes. Part D killed it.
The MIRROR cell, QQQ OUTperforming by a point on a shallow UP day, also gives
QQQ/SPY -1.578% at h21 on 45-64, t -2.72. Both sides drag, so the cell is a
ratio artifact rather than rotation, and the era split (-1.437% pre-2018 vs
-0.197% after) puts most of what is left in the dot-com unwind.

**KILLED 07, livestock — and this one was the point of the drill.** HE=F's
5d -15.80% is a CONTRACT ROLL, not a market event: a single -14.47% overnight
gap on 2026-08-17 against -2.15% of intraday movement across the last ten
sessions, 79% of the 21-day move sitting in gaps, and a median daily range
that FELL to 0.58% from 1.11% over the prior 250 sessions. A real -20% move
does not come with a quieter tape. LE=F is the same story at 157% gap share.
Both excluded from the brief. The corn control confirms the method reads the
other way when a move is genuine (29% gap share over 21 days), though today's
single +4.89% corn session was itself +5.16% gap and -0.26% intraday.

**KILLED 08, rotation day.** Built as a tape-derived cross after noting XLK
-2.47% against XLV +1.60% under a 0.68% index decline. The comparison that
matters inverts it: shallow declines with TOP-decile sector dispersion return
+0.635% over 21 sessions against +0.914% for shallow declines in the bottom
half of dispersion, and h1 is -0.128% with 68% of that in two episodes. High
dispersion is mildly WORSE than low, the edges are inside noise, and today's
spread is the 83rd percentile anyway, not the top decile the cell required.

**METHOD NOTE, and it nearly cost a nugget.** Drill 11 measured what followed
each completed ^BVSP down-run from the run's LAST down close and got "next
session +1.74%, 6 of 6 up". That statistic is worthless: a run ENDS precisely
because the following session was positive, so the record is 6-0 by
construction. Any measure anchored on a run's end inherits the look-ahead.
The publishable version anchors on the streak length as of tonight, not
knowing whether it ends. Not published in any form.

**SURVIVES 02/02b — the headline.** The three sessions into monthly opex are
a risk-off tilt across four instruments at once: TLT +0.262% (171-116, t 2.92,
n=287) against +0.051% for all three-session windows, IEF +0.091% (166-121,
t 2.11), LQD +0.098% (177-110), ^TNX -0.371% (133-185) and SPY -0.162%
(168-150) against +0.116%. Era-stable both sides of 2018 (+0.285 / +0.219),
positive in 19 of 25 calendar years, and t RISES to 3.03 when the two largest
episodes are dropped, so nothing is episode-carried. Mechanism is honest and
conditional: TLT pays +0.619% (85-43, t 4.32) when SPY falls over the window
and -0.026% when it does not, correlation -0.315. This is a flight-to-quality
tilt, not a standalone bond bid, and the brief says so.
  MULTIPLICITY: TLT's single-day k3 cell did NOT clear the sweep's BH line, so
  the [solid] tag does not rest on a swept p-value. It rests on E:opex being a
  standing calendar trigger the engine enumerates every month rather than a
  search hit, on four instruments agreeing, on era stability, on the per-year
  record and on the drop-the-biggest test. Recorded here as the rule requires.

**SURVIVES 03 — and the mechanism does not.** August Wednesdays in crude are
-0.385% (n=114, t -2.06) where Wednesdays ex-August are +0.127% (n=1226) and
August non-Wednesdays +0.075% (n=458). Both differences clear t 2: -0.512pp
at t -2.49 and -0.460pp at t -2.18. So the CROSS is the finding, not either
main effect, which is exactly what the drill was for. Era-stable and stronger
recently (-0.306% pre-2018, -0.550% after), negative in 17 of 27 Augusts, top
two episodes 23% of the total. Two caveats go in the brief: the hit rate is
47.4% so this is magnitude and not frequency, and COPPER shows the same
August-Wednesday tilt (-0.251%, t -1.92) with no Wednesday release at all,
which undercuts the tidy EIA explanation rather than supporting it.

**SURVIVES 01, reframed to silver.** The equity half is weak on its own: SPY's
expiry session is -0.103% (t -1.52) against +0.190% (t 2.86) for the run-in
that published last night, correlation -0.173, and it is post-2018 where the
drag sits (-0.188%). August inverts even that (+0.134%, 15-11), so the
escalation re-telling cannot lead with SPY. SI=F is the group's strongest
subject and was not mentioned last night: +0.212% on the expiry session,
175-135, sign p 0.013, n=310, positive in BOTH eras (+0.171 / +0.295) with the
top two episodes contributing -0% of the total. Silver leads the item.

**SURVIVES 04 as an anecdote.** QQQ 6-0 and IWM 6-0 across six midterm years
spanning 2002 to 2022, so six independent regimes rather than one repeated,
per-event magnitudes tight (QQQ +0.24% to +0.77%), and the all-years cell
agrees at 18-8, sign p 0.038. Control passes: a random midterm-August session
pays QQQ +0.035% against +0.047% for all days, so the day is doing the work,
not the month. Caveat that must be printed: the neighbouring sessions do NOT
repeat it, with the anchor+2 cell at 1-5, so the exact-day precision is
coincidence rather than structure.

**SURVIVES 10 as the today lane.** The engine fired ^BVSP's 5+ down-close cell
(+0.484%, 48-30, sign p 0.027, era-stable) but the live streak is ELEVEN, and
the cell does not describe it. Conditioned on 7+, which is the real state, the
bounce is gone: h1 +0.162% on 6-8, h5 -1.223% on 5-8, h10 -1.399% on 5-8,
against +0.243% and +0.478% for all windows. The next-session bounce peaks at
streak 5 (+0.495%, 53-33) and fades to nothing by 7. Verified by drill 11 that
the streak is real (12 distinct closes, no zero-change sessions) and that only
ONE longer run exists since 2000: 13 sessions ending 2023-08-17.

## Final slate

1. opex risk-off tilt, TLT/IEF/LQD/^TNX/SPY  [solid]     HEADLINE
2. crude on August Wednesdays                [suggestive]
3. silver into the VIX expiry                [suggestive]
4. Aug 19 seasonal, midterm years            [anecdote]
5. ^BVSP eleven down closes                  [anecdote]   today lane

Two anecdotes, which is the cap. Headline is the solid cell. Both lanes
present. Four tomorrow-lane items and one today-lane item, which reflects that
three of the four today-lane candidates died on their drills.
