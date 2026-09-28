# Pre-registration: next-day conditioning rules for the NQ/ES opening breakout

Date 2026-09-28. Author McKinley Slade (drafted by Claude). Status: DRAFT for owner review. Not approved. Nothing in this file is live, and no forward data has been examined for this question.

Drafter's recommendation: do not proceed with P1/P2 as a separate control while the 1.25 range skip is live; hold S1 pending the forward tier; the owner decides.

Candidate: `artifacts/research/qqq_open_breakout_20260923/current_candidate/` (break-even OFF, decision 2026-09-24). Live service: `open_breakout/`, runbook `docs/open_breakout_runbook.md`. Related prereg: `docs/prereg_open_breakout_range_filter_2026-09-25.md` (prior-range skip at ratio >= 1.25, live in the pilot since 2026-09-28).

Repo rule (CLAUDE.md, "Pre-registration"): any new dial-conditioned control needs a written prereg (gates, decision rule, sensitivity) before the study runs. These rules condition on the strategy's own prior-day P&L, not on the fragility dial, but they change what trades and at what size, and the short-side effect they target is confounded with the risk-dial short gate. They are held to the same rule. They are per-strategy and per-market, not a book-wide throttle. This file must be committed before any forward data is examined for this question and before the RTY tier is run.

Basis: the RANGE-FILTERED ledger throughout. The live pilot skips any market-session with prior-range ratio >= 1.25, so that is the book these rules would act on. Every trigger, outcome, test and sample count below is on range-filtered sessions unless marked unfiltered. The unfiltered ledger is a sensitivity only.

## Motivation (post hoc, stated plainly)

On 2026-09-28 the owner asked whether the candidate autocorrelates with itself. The descriptive study (`autocorr/RESULTS.md`) found a small negative lag-1 effect: after a day of +2R or more the next day averaged about -0.15R, against about +0.3R after a losing or flat day. A follow-up (`autocorr/lag1_rule/RESULTS.md`) found the effect sits almost entirely on next-day SHORT trades, whether the big day was long or short; longs after big long days showed nothing. The owner then proposed three rules: no shorts the day after a big win, no longs the day after a big short win, and larger shorts (perhaps 1.5x) the day after a losing day, on the theory that sloppy non-trending action signals an unhealthy market. The composite study (`autocorr/lag1_rule/composite/RESULTS.md`) ran them on the frozen, unfiltered ledger: the data's version of the first rule is "no shorts after a big win of EITHER side" (the owner's literal long-only trigger keeps +5.8R of +23.8R, because it leaves in the worst cell); the second is weakly supported; the third adds real R but mostly through added exposure, with a 17% deeper max drawdown, almost all of it NQ.

The overlap study (`composite/overlap_walkforward/RESULTS.md`) then showed that the skip arms are largely the range filter seen again. A big breakout win is usually a wide-range session, so the next morning's ratio is usually high: 57% of sessions after a +2R win are already range-skipped, against 22% otherwise (odds ratio 4.7). 78% of the unfiltered R1+R2 gain (30.5R of 39.1R) sits on days the shipped filter already skips. On the filtered ledger R1+R2 adds +8.6R, CI [-13.9, +32.1], and -7.2R without its five best days. The short slope on prior-day R shrinks from -0.109 [-0.179, -0.040] unfiltered to -0.081 [-0.195, +0.028] filtered. Walk-forward on the filtered ledger over 2021-2026, R1+R2 earns -4.3R. The forward tier should therefore expect a result near zero for P1/P2. S1 mostly sits on days the filter keeps and retains about 64% of its R.

None of this was pre-specified. The lag-1 grid ran 486 cells (27 rules x 3 series x 3 eras x 2 costs), the direction cells 378 rows (the side split chosen after seeing them), the composite grid 306 rows, the filtered composite grid as many again, plus the overlap, single-statistic and walk-forward grids, and 217 logged looks in the parent study: well over 1,700 looks on one 2018-2026 sample, the one the candidate itself was selected on. All NQ and ES history in hand was used in discovery.

## Hypotheses

H1 (skip arm): on the filtered book, next-day short R falls with the prior day's own R, so that shorts after a big win (and longs after a big short win) are worth skipping. H0: the short slope on prior-day R is zero or positive.

H2 (scale arm): shorts on the session after a losing day earn more per trade than the candidate's shorts in general, by enough that scaling them up improves return per unit of drawdown. H0: they do not beat the unconditional short mean, or the drawdown-adjusted return does not improve.

## Variables (frozen, point-in-time at the prior session close, per market)

- R unit: the candidate's own per-trade risk, a stop of 0.25 x prior full-session TR. A trade's R is net P&L per contract over the stop distance in dollars per contract, net of fees and slippage (`net_r`).
- Trigger session: the market's immediately preceding XNYS trading session. Day R = the summed net R of the shadow session's entries in that market that session, counted only if the session was range-armed (ratio < 1.25, from the ratio the shadow manifest records). A range-skipped session is 0R, as in the research filtered ledger.
- Big win day: day R >= +2.0R. Dominant side: the side with the larger |summed R| that session; a one-sided day takes that side; an exact tie is not short-dominant. Losing day: day R < 0 with at least one entry.
- No trigger (no skip, no scale) when the trigger session: had no entry; was range-skipped; was halted, failed or never ran in the shadow; was roll-excluded or an early close; or is separated from the current session by a full XNYS holiday (a weekend alone is not a gap). The research reached back to the previous eligible session across such gaps. This is a declared difference; gate 3 reports how many in-sample triggers it removes.
- Own-market trigger is primary. Pre-declared alternates: the either-market trigger as in the research (the big-win test fires if either market's most recent trigger session qualifies; the losing-day test is always own-market), and a +1.5R big-win threshold.

## Pre-registration (frozen; do not alter)

- P1: skip all SHORT entries in a market on the session after a big win of either side in that market.
- P2: skip all LONG entries in a market on the session after a big SHORT-dominant win in that market.
- S1 (secondary arm): 1.25x risk on SHORT entries on the session after a losing day in that market. The owner asked for 1.5x. 1.25x is the drafter's default: in-sample the gain, the added exposure and the added drawdown are all linear in the multiplier, so 1.25x buys half the exposure increase for half the R and half the drawdown cost, and nothing in the data picks 1.5x. Pre-declared alternate: 1.5x, evaluated by the same rule on its own. On the full in-sample the walk-forward's drawdown cap (1.10x the book's drawdown) rejects every multiplier above 1.0, and out of sample drawdown ends 14 to 24 percent above the book's.
- A skip beats a scale. With the own trigger they cannot overlap.
- Attempts are not freed by a skip. A skipped entry counts toward the three-entry limit, and its phantom position (fill at the trigger, stop at 0.25 x TR, exit at the stop or 15:55) blocks new entries as the real one would have, so every later trade that session is the one the candidate took. This is how the research counted it.
- Live one-contract cap: S1 cannot act while the pilot is capped at one contract. It multiplies whole contracts and floors, so 1.25x first changes a position at 4 contracts and 1.5x at 2. P1 and P2 can act now.

The +2R threshold stays primary as frozen, although the walk-forward never chose it: on every expanding window 1.5R won on in-window Sharpe. That preference is recorded here, not acted on. P1 and P2 are decided together as one skip arm. No other threshold, trigger, side split, multiplier or cost case may be reported as confirmatory. An alternate that passes while its primary fails ships nothing.

## Pre-declared analyses

Confirmatory statistic for P1/P2 (replaces the per-cell means of the research). Pairs are consecutive eligible sessions of one market where the prior session traded and the next session has at least one short. y = the next session's summed short R; x = the prior session's own summed R. Pooled OLS of y on x with a market intercept. Reported with it: the bottom-minus-top tercile gap in mean y (terciles of x cut within each market), and the same slope and gap for next-day longs as a placebo. P2 has no separate confirmatory test; it ships only with P1, and its skipped trades' R is reported beside.

Parameter-stability check (reported, never deciding). Expanding window from 2018, one test year at a time: choose threshold {1, 1.5, 2, 3}R x scope {own, either} by in-window day Sharpe of P1+P2, then k in {1.0, 1.25, 1.5} by in-window Sharpe of P1+P2+S1, subject to in-window max DD <= 1.10 x the unconditioned book's (k = 1.0 if nothing qualifies). Apply the choice to the test year only and stitch the out-of-sample years. At decision time it is rerun with the forward sessions appended as further test periods.

## Out-of-sample plan

There is no clean NQ/ES holdout. Three tiers, in order of weight.

(a) Forward tracking on the shadow journal, the deciding tier. The first counted session is the first whose 09:00 prepare starts after this file's commit. Triggers and outcomes come from the shadow (client 927480), restricted to range-armed sessions; live pilot fills are a cross-check only.

Required samples and pace, from filtered in-sample rates (2018-01-02 to 2026-08-28, 8.65 years):
- P1/P2: 120 forward short pairs. The filtered sample had 499, about 58 a year, so about two years (near late 2028). The skipped trades themselves are rare on the filtered book: 57 shorts and 22 longs in 8.65 years, about 9 a year (15 a year in 2023-26).
- S1: 60 qualifying shorts. 347 in-sample, about 40 a year (57 a year in 2023-26), so 13 to 18 months.
Shorts trade only when the prior-session legacy risk score is 20 or higher, so all of these rates move with the regime.

Power, stated before the fact. The in-sample filtered short slope had a 95% interval about +/-0.11 at 499 pairs, so at 120 pairs the 90% half-width is about 0.19R per R. A PASS needs an observed slope near -0.19 or steeper; the in-sample estimate was -0.081. The expected outcome is INCONCLUSIVE and nothing ships. The owner accepts this.

Sessions where the shadow halted, failed or never ran are excluded from the primary and from triggering. A secondary count adds them replayed with the research engine on IB same-contract bars, flagged as replays.

(b) RTY as a pseudo-out-of-sample instrument. Unfiltered RTY ledgers exist from the range-filter RTY tier (`range_filter_rty/RTY_{base,stress}_none_trades.csv`, with `RTY_features.csv` for the ratio). No lag-1 or next-day cut has been computed on them; only trade counts by side were read while drafting. The RTY tier uses the RTY filtered ledger (ratio >= 1.25 sessions dropped) and RTY's own day R. For it to count, all must hold:
1. The ledger is from `current_candidate/engine.py` unchanged with RTY settings, and the same harness reproduced the frozen NQ and ES ledgers exactly (recorded in `range_filter_rty/run_log.json`; re-confirm the engine hash).
2. The RTY session audit matches `es_rty/RTY_session_audit.csv`.
3. At least 250 filtered short pairs and 150 S1-eligible shorts at base costs.
The RTY book is weak (mean R 0.02 base, -0.11 stress), stated beside every RTY result.

(c) In-sample era split, 2018-22 versus 2023-26, supporting only, never decisive.

## Primary metric and decision rule

Forward tier (decides), each arm separately. Bootstrap throughout: stationary block bootstrap over union-calendar dates (mean block 10 sessions, NQ and ES on the same date move together), 10,000 draws, percentile interval, seed 20260928.

P1+P2, on the forward short pairs:
- PASS: the upper bound of the 90% interval of the short slope is below 0, the bottom-minus-top tercile gap is positive, and the long placebo slope's 90% interval includes 0. If the placebo is also reliably negative, the effect is not side-specific and the arm does not pass as specified.
- FAIL: the short slope point estimate is at or above 0.
- INCONCLUSIVE: anything else. Extend once by 60 more pairs, apply the same rule to all 180, then stop. A second INCONCLUSIVE is final and counts as not passing.
Reported beside, never deciding: summed and mean R of the actual P1 and P2 skipped trades, with the 90% interval; a one-sided month sign test (months where the skipped trades summed below zero); day Sharpe with and without the rules.

S1, at the 60-trade count, evaluated at 1.25x (and separately at 1.5x as the alternate):
- PASS only if both hold: (i) mean R of the qualifying shorts at 1.0x exceeds the mean R of all filtered shadow shorts in the same window; and (ii) summed R divided by max drawdown in R, on the combined NQ+ES filtered shadow day series over the whole forward window, improves with S1 applied versus not. Test (ii) is required because the in-sample gain came with deeper drawdowns, the full-sample drawdown cap rejects every multiplier above 1.0, and out of sample the walk-forward's drawdown ended 14 to 24 percent above the book's.
- Otherwise not passing. If the forward max drawdown is under 5R, the ratio is not yet measurable; extend once by 30 qualifying shorts and decide then.
Reported beside: the month sign test, day Sharpe with and without, a plain t-stat on per-trade R.

RTY tier (corroborates or blocks), filtered, base costs, same bootstrap.
- P1+P2: short slope. SUPPORTS if the 90% upper bound is below 0; CONTRADICTS if the point estimate is at or above 0; else NEUTRAL.
- S1: mean R of post-loss shorts minus other shorts. SUPPORTS if the 90% lower bound is above 0; CONTRADICTS if the point estimate is at or below 0; else NEUTRAL.
Shipping an arm requires its forward PASS and an RTY result that is not CONTRADICTS. RTY alone can never ship an arm.

## Gates before the study runs

1. This prereg is committed to main before any forward data is examined for this question and before the RTY tier. The commit hash is recorded in the results file.
2. Forward tier only: the live pilot has at least 10 clean sessions of journal (ran to 16:01 ET without halting), so shadow and live fills can be compared before any R is counted.
3. Forward tier only: the trigger is computed from the same journal the service writes (the shadow journal's per-market day R, masked by the manifest ratio). Parity check before first use: on the overlap of shadow sessions with research-engine replays of the same sessions, day R agrees within 0.05R and the trigger classification agrees on every session. The check also reports how many in-sample triggers the holiday-gap and halted-session exclusions remove. Failure stops the forward tier until fixed.

## Sensitivity (pre-declared, reported, never used to change the primary)

- Big-win threshold +1R, +1.5R, +2R, +3R; own versus either trigger.
- S1 at 1.0x (pure conditioning test), 1.25x and 1.5x.
- Unfiltered versus filtered ledger (shadow outcomes on all sessions, triggers from unfiltered day R).
- Base versus stress costs (in-sample and RTY; forward reports shadow and live fill bases separately).
- NQ versus ES; eras 2018-22 versus 2023-26 (in-sample and RTY).
- Forward primary with and without replayed sessions.

## What ships

If P1+P2 PASSES and RTY does not CONTRADICT: a per-market side-skip in the `open_breakout` 09:00 inputs step. `build_manifest` reads the prior session's filtered day R from the shadow journal and writes per market `prior_day_r`, `prior_day_side`, `skip_short_next_day` and `skip_long_next_day`. The service does not arm the skipped side and keeps the phantom-attempt accounting above. One config block (for example `next_day_filter: {enabled, big_win_r, trigger}`), mirrored in `current_candidate/config.json`, default OFF.

Fail-open, on purpose. If the prior session's day R cannot be computed (shadow did not run, journal unreadable, halted session), both flags are false and the market trades normally. The range filter fails the other way (closed, to skip), and the asymmetry is deliberate: the range filter guards against a state it treats as harmful, while this rule removes a minority of trades from a positive-expectancy book, so failing open only loses the filter's small benefit for one session. A data problem never blocks trading. The owner can override this and choose fail-closed in writing.

If S1 PASSES and RTY does not CONTRADICT: a per-side size multiplier (`short_after_loss_mult`, default 1.0) applied to the whole-contract count after every existing sizing step and cap, floored. It has no effect while the pilot is capped at one contract.

Each arm is turned on only by a second written approval from McKinley after the results file is reviewed, through the runbook's live-money change process. If an arm FAILS or ends INCONCLUSIVE, nothing ships for it and it is not re-tested on the same forward data with a different threshold or multiplier.

## Stop conditions and invalidation

- Any change to the candidate's frozen rules (trigger or stop fraction, 11:30 cutoff, three attempts, 15:55 exit, short gate definition, reinstating the break-even) invalidates pairs and trades accumulated before it; they are reported separately and the count restarts.
- A change to the range filter's threshold or mode, or turning it off, changes the basis: the count restarts on the new basis, and earlier data is reported separately.
- A change to the risk-dial short gate (score, smoothing, threshold or source) restarts S1's forward count, because the post-loss short effect is confounded with the regime that gate selects. P1/P2 results are split before/after and reported both ways.
- A vendor restatement or replay change that flips more than 10% of trigger classifications restarts the forward tier.
- An entry-mode or sizing change does not invalidate, since the metrics are in R; its date is recorded and a before/after split added.
- If P1, P2 or S1 is adopted live before its forward tier completes (as the owner did with the range filter on 2026-09-25), the adoption date is recorded here and in the runbook, and the forward sample is reported split before/after that date. The shadow stays unconditioned, so the decision still uses the full shadow sample.
- The study stops early only if the candidate itself is retired or suspended. No early stop for a favorable interim result; no interim looks.

## Appendix: what we already know (IN-SAMPLE, post hoc, not pre-specified)

Sources: `autocorr/lag1_rule/composite/` (`rules.csv`, `by_year.csv`) and `composite/overlap_walkforward/` (`overlap.csv`, `rules_filtered.csv`, `walkforward.csv`), 2018-01-02 to 2026-08-28, own trigger, +2R, base costs, fractional size, flat-R path. Legend: R1 = P1, R2 = P2, R3 = S1 at 1.5x, R1b = the owner's literal "skip shorts after a big LONG win".

Overlap of big-win days with range-skipped days (unfiltered ledger, market-days; hi = next-day ratio >= 1.25):

| series | big & hi | big & lo | not big & hi | not big & lo | P(hi given big) | P(hi given not big) | odds ratio |
|---|---|---|---|---|---|---|---|
| NQ | 134 | 96 | 391 | 1,437 | 58.3% | 21.4% | 5.1 |
| ES | 109 | 89 | 420 | 1,440 | 55.1% | 22.6% | 4.2 |
| pooled | 243 | 185 | 811 | 2,877 | 56.8% | 22.0% | 4.7 |

Fisher p < 1e-10 in every row. R1+R2's unfiltered gain: 30.5R of 39.1R on hi days. R3's: 92% on lo days.

Composite rules, COMB, unfiltered (before 562.3R, Sharpe 1.43, max DD 41.8R) beside filtered (before 558.6R, Sharpe 1.51, max DD 35.9R):

| rule | dR unf. | Sharpe / DD after, unf. | 95% CI unf. | trades affected, filt. | dR filt. | Sharpe / DD after, filt. | 95% CI filt. | ex top-5 filt. |
|---|---|---|---|---|---|---|---|---|
| R1 | +23.8 | 1.51 / 40.6 | [-0.9, +48.5] | 57 S skip | +1.7 | 1.52 / 37.3 | [-17.8, +21.7] | -11.0 |
| R1b | +5.8 | 1.45 / 41.9 | [-13.0, +24.5] | 35 S skip | +1.8 | 1.52 / 35.9 | [-13.9, +18.0] | -9.7 |
| R2 | +15.4 | 1.48 / 41.7 | [-11.1, +38.8] | 22 L skip | +6.9 | 1.53 / 35.9 | [-4.2, +18.1] | -0.3 |
| R3 (1.5x) | +113.8 | 1.51 / 48.9 | [+50.3, +181.2] | 347 S scale | +73.1 | 1.57 / 42.2 | [+28.0, +123.1] | +45.5 |
| R1+R2 | +39.1 | 1.55 / 40.5 | [+2.0, +76.9] | 57 S, 22 L | +8.6 | 1.54 / 36.5 | [-13.9, +32.1] | -7.2 |
| R1+R2+R3 | +153.0 | 1.62 / 47.5 | [+75.6, +233.1] | + 347 S | +81.7 | 1.60 / 42.9 | [+28.1, +140.7] | +54.1 |
| 1.5x every short | +154.8 | 1.45 / 52.3 | [+63.2, +250.1] | all S | +143.7 | 1.51 / 43.5 | [+58.9, +234.0] | +93.2 |

Unfiltered R3 at 1.25x: +56.9R, DD 45.4R. Filtered by market, R1+R2: NQ +1.4R, ES +7.2R; by era +5.9R (2018-22) and +2.7R (2023-26). Unfiltered by market: R1+R2 NQ +10.9R (ex-top-5 -0.4R), ES +28.2R; R3 NQ +78.4R, ES +35.5R, and under stress ES +1.1R.

Direction 2x2, unfiltered, pooled pairs, mean next-day R by dominant side (baseline / after a +2R day / after a loss): S->S +0.33 / -0.32 / +0.53; L->S +0.43 / -0.20 / +0.80; S->L -0.06 / -0.20 / -0.03; L->L +0.02 / +0.04 / +0.02.

Single statistic, next-day side R on prior-day own R (95% intervals):

| ledger | cost | next side | n pairs | slope per 1R | tercile means bottom / mid / top | bottom minus top |
|---|---|---|---|---|---|---|
| filtered | base | short | 499 | -0.081 [-0.195, +0.028] | +0.42 / +0.58 / +0.01 | +0.41 [-0.12, +0.96] |
| filtered | base | long (placebo) | 790 | -0.039 [-0.120, +0.043] | +0.04 / +0.23 / +0.05 | -0.01 [-0.37, +0.36] |
| unfiltered | base | short | 833 | -0.109 [-0.179, -0.040] | +0.47 / +0.37 / -0.03 | +0.50 [+0.10, +0.93] |
| unfiltered | base | long (placebo) | 1,185 | -0.046 [-0.098, +0.008] | +0.10 / +0.01 / -0.01 | +0.12 [-0.16, +0.40] |
| filtered | stress | short | 479 | -0.037 [-0.158, +0.075] | +0.15 / +0.42 / -0.07 | +0.22 [-0.32, +0.80] |

Walk-forward, expanding window, stitched out-of-sample 2021-2026 (1,358 days), base:

| ledger | variant | dR | 95% CI | Sharpe before -> after | max DD before -> after |
|---|---|---|---|---|---|
| unfiltered | WF R1+R2 | +6.0 | [-52.4, +61.8] | 1.32 -> 1.40 | 41.8 -> 44.8 |
| unfiltered | WF R1+R2+S1 | +76.9 | [+0.0, +158.9] | 1.32 -> 1.47 | 41.8 -> 51.8 |
| filtered | WF R1+R2 | -4.3 | [-39.7, +30.3] | 1.49 -> 1.51 | 35.9 -> 41.0 |
| filtered | WF R1+R2+S1 | +43.1 | [-10.7, +100.1] | 1.49 -> 1.60 | 35.9 -> 41.0 |
| filtered | fixed 2R own, R1+R2 | +4.1 | [-18.0, +26.7] | 1.49 -> 1.53 | 35.9 -> 36.5 |
| filtered | fixed 2R own 1.5x, R1+R2+S1 | +66.3 | [+18.2, +119.2] | 1.49 -> 1.61 | 35.9 -> 42.9 |

Chosen parameters: threshold 1.5R in every filtered window (2R never); scope flips between own and either; k is set only by the drawdown cap (1.5x in 2022-2024, 1.0x in 2021, 2025 and 2026), and on the full 2018-2026 sample the cap rejects every k above 1.0 on both ledgers.
