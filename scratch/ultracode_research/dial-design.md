# Dial Design — which fragility dial, smoothing, threshold; is a combination better?

Run date: 2026-07-02. Track: DIAL DESIGN.

## Setup

- Trades: `data/backtest_trades_full.parquet`, non-OVS only (the live multiplier
  exempts OVS), signal dates 2016-08+ (30-day MA warm-up), joined as-of signal date
  (merge_asof, 7-day tolerance) to each dial's smoothed score from
  `data/rd2_fragility.parquet`. N = 1136 non-OVS trades with all three dials
  non-NaN (vs 1153 in the established finding — difference is the longer warm-up
  plus requiring 21d/63d columns, which start later than 5d).
- Smoothing = trailing simple MA of each dial's daily score (live basis is the
  10d MA of 63d).
- All significance = Welch t on MONTHLY mean R (signal month), flagged vs
  unflagged. Robustness = leave-one-year-out (LOYO) refits of that t.
- Code: `dial_setup.py`, `t1_overlap.py`, `t3_sweeps.py`, `t4b_fixed50.py`,
  `t5_hysteresis.py` (this directory).

Top-decile (p90) cutoffs of each dial's 10d-MA daily distribution: 5d >= 28.8,
21d >= 38.9, 63d >= 54.9 — these reproduce the thresholds in the track brief.

## Test 1 — overlap matrix (are the dials the same signal?)

Day level (2,443 days, 245 flagged per dial by construction):

| P(col | row) | 5d | 21d | 63d |
|---|---|---|---|
| 5d  | 1.00 | 0.49 | 0.39 |
| 21d | 0.49 | 1.00 | 0.69 |
| 63d | 0.39 | 0.69 | 1.00 |

Jaccard: 63d-21d 0.53, 21d-5d 0.33, 63d-5d 0.24. Any-dial-flagged = 438 days
(17.9%); all three = 90 days (3.7%). Trade level is similar (63d-flagged trades:
57% also 21d-flagged, 37% 5d-flagged).

So the dials are NOT redundant — 5d in particular flags mostly different days.
The question is whether the non-63d days carry any R-degradation. They don't (Test 2/3).

## Test 2 — combinations vs the single 63d rule (monthly-clustered)

Non-OVS, N=1136. avgR_hi = flagged trades, avgR_lo = rest.

| rule | N_hi | %trades | avgR_hi | avgR_lo | t | p | LOYO t range |
|---|---|---|---|---|---|---|---|
| 63d >= 50 (established) | 242 | 21.3 | +0.19 | +0.65 | -2.73 | .010 | [-2.95, -2.05] |
| 63d p90 (>=55) | 171 | 15.1 | +0.17 | +0.62 | -2.08 | .049 | — |
| 21d p90 (>=39) | 144 | 12.7 | +0.03 | +0.62 | -1.68 | .106 | [-1.88, -1.13] |
| 5d p90 (>=29) | 166 | 14.6 | +0.13 | +0.62 | -0.87 | .391 | — |
| any dial p90 | 291 | 25.6 | +0.17 | +0.68 | -2.31 | .024 | [-2.58, -1.73] |
| all dials p90 | 64 | 5.6 | +0.13 | +0.57 | -0.50 | .631 | — |
| 2-of-3 p90 | 126 | 11.1 | -0.01 | +0.62 | -1.36 | .192 | — |
| 63d>=50 OR 21d p90 | 257 | 22.6 | +0.19 | +0.65 | -2.75 | .009 | [-2.99, -2.06] |
| 63d>=50 OR 5d p90 | 321 | 28.3 | +0.23 | +0.67 | -2.17 | .033 | — |

Verdict: no combination beats 63d>=50 in a way that matters. "any dial p90" adds
120 trades of coverage but weakens both t and the LOYO floor (-1.73 vs -2.05).
OR-ing 21d p90 onto 63d>=50 adds 15 trades and +0.02 of t — noise-level gain for
a second parameter. Adding 5d dilutes. Requiring agreement (all/2-of-3) destroys
the signal by cutting N.

## Test 3 — threshold stability curves (10d MA)

63d, fixed-value sweep (the decisive table):

| thr | N_hi | avgR_hi | avgR_lo | t | p |
|---|---|---|---|---|---|
| 35 | 446 | +0.42 | +0.63 | -1.23 | .223 |
| 40 | 372 | +0.37 | +0.64 | -2.19 | .031 |
| 42.5 | 322 | +0.28 | +0.65 | -2.32 | .023 |
| 45 | 293 | +0.26 | +0.65 | -1.91 | .062 |
| 47.5 | 272 | +0.21 | +0.66 | -3.10 | .003 |
| 50 | 242 | +0.19 | +0.65 | -2.73 | .010 |
| 52.5 | 202 | +0.09 | +0.65 | -2.62 | .016 |
| 55 | 171 | +0.17 | +0.62 | -2.08 | .049 |
| 60 | 109 | +0.09 | +0.60 | -1.54 | .145 |
| 65 | 81 | +0.20 | +0.58 | -1.09 | .297 |

A broad significant plateau at 40-55 (t between -1.9 and -3.1), fading above 55
purely from thinning N (only 109 trades / 13 months above 60). 50 sits inside the
plateau; the in-sample peak (47.5) is not meaningfully better and should not be
chased. 21d dial (10d MA): nothing until p95 (thr 46.5, N=88, t=-2.26, 9 months —
too thin). 5d dial: dead at EVERY threshold p60-p95 (best t = -0.87). Percentile
sweeps per dial in `t3_sweeps.py` output.

## Test 4 — MA window sensitivity (5..21)

63d dial, FIXED thr=50 (the production candidate):

| MA_w | 1 | 3 | 5 | 8 | 10 | 13 | 15 | 18 | 21 |
|---|---|---|---|---|---|---|---|---|---|
| t | +0.72 | -1.26 | -2.21 | -3.07 | -2.73 | -3.34 | -2.81 | -1.81 | -1.43 |
| N_hi | 209 | 211 | 217 | 243 | 242 | 237 | 233 | 214 | 212 |

- Raw score (w=1) has NO signal — confirms the established finding; the smoothing
  carries the information.
- Robust plateau w=5..15 (t -2.2 to -3.3). The live 10d MA sits in the middle of
  the plateau, not on a peak. Fades at 18-21 (avgR_hi rises to +0.33..0.36 —
  a too-slow MA keeps flagging after the episode's damage window has passed).
- 21d dial improves monotonically with window: at w=21, thr=p90 (~36), t=-3.93,
  N=145, LOYO t range [-4.67, -3.14] — the strongest LOYO of anything tested.
  BUT: 123 of its 145 flagged trades are already flagged by 63d>=50 (Jaccard
  0.47), and the 22 uniquely-flagged trades have avgR +0.49 — i.e. it catches
  zero incremental damage. A 21d score under a 21d MA is just another slow
  composite converging on the same information as the 63d dial. It was also
  found by scanning 9 windows x 3 dials (27 cells), so its headline t carries
  selection inflation. Runner-up, not winner.
- 5d dial: no window works (best |t|=0.87).

## Test 5 — hysteresis vs plain threshold (63d, 10d MA, 10.0 yrs of daily data)

| config | flips/yr | episodes | %days on | median ep len | eps<=5d | eps<=10d | N_tr | avgR_hi | t | p |
|---|---|---|---|---|---|---|---|---|---|---|
| plain 50 | 2.6 | 13 | 14.0 | 19d | 0 | 2 | 240 | +0.19 | -2.67 | .011 |
| hyst 50/45 | 2.4 | 12 | 15.8 | 23.5d | 0 | 1 | 263 | +0.20 | -3.08 | .004 |
| hyst 50/40 | 1.8 | 9 | 17.5 | 46d | 0 | 0 | 296 | +0.32 | -2.63 | .012 |
| plain 55 | 2.2 | 11 | 10.0 | 16d | 2 | 4 | 169 | +0.18 | -2.03 | .055 |
| hyst 55/45 | 2.2 | 11 | 13.1 | 19d | 0 | 1 | 224 | +0.29 | -2.31 | .030 |

(N_tr here differs by ~2 from Test 2 because the regime is evaluated on the daily
series then mapped to signal dates, vs thresholding the joined per-trade value —
same convention live sizing would use.)

Whipsaw is already a non-problem: the 10d MA limits the plain-50 rule to 2.6
regime changes/yr with ZERO episodes of 5 days or less over 10 years. Hysteresis
50/45 trims one episode and slightly improves t (-3.08, LOYO not separately run
but the added band 45-50 has plateau-consistent behavior); 50/40 halves the flip
rate but dilutes avgR_hi to +0.32 by hanging on through the 40-50 recovery band.
Conclusion: hysteresis is cosmetic here. It becomes relevant only if the MA is
shortened or the pending taper (50->60) is replaced by a hard on/off multiplier.

## 2026 YTD inversion (inherited caveat, re-checked)

Persists under every variant: 2026 flagged trades avgR +0.49 to +0.55 (N~25) vs
unflagged -0.09 to -0.12 (N~68). Whatever rule ships, 2026 is the year that
would have hurt; the LOYO floors above (drop-2021 worst, t~-2.0) already price
this in.

## Recommendation

**Production rule: 63d dial, 10-day MA, threshold 50, plain (no hysteresis),
non-OVS only** — i.e. exactly the basis of the pending taper rec (1.0x through
50, taper to 0.5x by 60). Every dimension of this study independently lands on
it: 5d is dead at all thresholds and windows; 21d at the live smoothing is too
thin; no combination improves discrimination or the LOYO floor; thr 50 sits
mid-plateau (40-55) rather than on the in-sample peak; MA 10 sits mid-plateau
(5-15); and the 10d MA already caps regime flips at 2.6/yr with no <=5-day
whipsaw, so a hysteresis band adds a parameter for cosmetic benefit. If an
on/off band is ever wanted for operational stability, 50/45 is free (t=-3.08)
— but it is not needed for the taper.

**Runner-up: 21d dial, 21-day MA, threshold ~36 (its p90)** — best LOYO range of
anything tested ([-4.67, -3.14]) but 85% overlapped with the 63d rule, zero
incremental damage caught (+0.49 avgR on its 22 unique flags), and found via a
27-cell scan. Keep it as a shadow/confirmation series, not the sizing basis.

## Caveats

- Fragility history is a current-vintage reconstruction; composite edge weights
  carry calibration lookahead (inherited from the established findings).
- Above thr 55 the sample thins fast (171 trades / 17 months; 81 above 65) —
  the taper's 0.5x-by-60 endpoint is an extrapolation on ~110 trades.
- Threshold and MA sweeps are multiple comparisons; conclusions rest on plateau
  breadth, not any single cell's p-value.
- 2026 YTD inverts the signal (flagged trades outperformed). One year, N=24.
- N=1136 vs 1153 in the prior study (warm-up + all-dials-non-NaN requirement);
  headline numbers reproduce (63d>=50: +0.19 vs +0.65, t=-2.73 vs -2.86).

## Adversarial verification (2026-07-02, independent recompute)

Recomputed from scratch (`verify_dial-design.py`, no code reuse; merge_asof
backward 7d tol, rolling-mean min_periods=1, Welch t on monthly mean R,
LOYO by signal-year). Base N = 1146 vs their 1136 (min_periods/warm-up
convention; all decisive numbers unaffected). All 8 decisive claims CONFIRMED:

| claim | theirs | recompute |
|---|---|---|
| 63d>=50 headline | N=242, +0.19/+0.65, t=-2.73 p=.010, LOYO [-2.95,-2.05] | N=242, +0.187/+0.639, t=-2.75 p=.009, LOYO [-2.97,-2.06] |
| any-dial-p90 combo | t=-2.31, LOYO floor -1.73, N=291 | t=-2.33, LOYO [-2.59,-1.74], N=291 — worse than 63d alone, confirmed |
| 63d>=50 OR 21d p90 | t=-2.75, +15 trades | t=-2.76, N=257 — noise-level gain, confirmed |
| 5d dial dead | best t=-0.87 over thr p60-p95 x MA 1-21 | best t=-0.88 (w=10, p90); all other cells \|t\|<0.6 |
| 63d thr plateau 40-55 | t -1.9..-3.1, peak 47.5 (-3.10) | t -1.92..-3.12, peak 47.5 (-3.12); 35 insignif (-1.35), 60+ fades |
| MA window plateau 5-15 | -2.2..-3.3; w=1 +0.72; w=18/21 -1.8/-1.4 | -2.24..-3.32; w=1 +0.72; w=18/21 -1.78/-1.40 |
| whipsaw | plain 50: 2.6 flips/yr, 0 eps<=5d; hyst 50/45 2.4, t=-3.08 | 2.6 flips/yr, 13 eps, 0<=5d; hyst 50/45: 2.4, t=-3.14, N_tr 265 — marginal, confirmed cosmetic |
| runner-up 21d/21MA/36 | t=-3.93, LOYO [-4.67,-3.14], 123/145 overlap, 22 uniq +0.49 | thr 35.9, t=-4.04, LOYO [-4.80,-3.08], 124/146 overlap, 22 uniq +0.48 |
| 2026 inversion | flagged +0.49..+0.55 (N~25) vs -0.09..-0.12 (N~68) | +0.49 (N=24) vs -0.09 (N=68) |

Robustness probes beyond their tests:
- Join convention: live-style ffill(limit=5) join gives identical N_hi=242,
  t=-2.75 — the 7d-tolerance merge_asof is not doing anything.
- Coarser clustering: quarterly-clustered t=-3.66 (p=.002) — the monthly
  result is not a clustering-granularity artifact. (Raw per-trade t=-4.67,
  correctly not used.)
- No new lookahead found beyond what they flag: p90 cutoffs are full-sample
  (mild), but the shipped rule is a fixed 50 on the 63d dial, unaffected;
  fragility current-vintage reconstruction caveat inherited and stated.

Verdict: recommendation stands as written. The 21d/21MA runner-up's t is even
a touch stronger on recompute (-4.04) but the redundancy finding (85% overlap,
+0.48R on unique flags) also reproduces, so demoting it to a shadow series
remains the right call.
