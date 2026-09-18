# Would smoothing `nyse_net` improve the recovery-reset warning?

> **Superseded by the ship, 2026-09-18.** `warning_severity` is now EMA5-based
> in production, so `study.py`'s `raw_1d` replication assertion no longer holds
> against it and the script will fail at that check. Everything below describes
> the state of the code when the study was run. `study.py` is left unmodified.

Research only. Nothing here changes `nyse_risk.py` or any production file.
Reproduce with `python scratch/nyse_smoothing_study/study.py` (writes
`results.csv` plus `coverage.csv`, `era_split.csv`, `loyo.csv`, `timeliness.csv`,
`delay_placebo.csv`, `last24m.csv`, `last30_sessions.csv`, `aug_sep_2026.csv`,
and the full console transcript in `output.txt`).

---

## Protocol (fixed before any result was looked at)

**Sample.** SPY adjusted closes from `data/master_prices.parquet` joined to
`nyse_net` from `data/market_breadth.parquet`. Breadth runs 1995-01-03 to
2026-09-17, 7,978 rows, no duplicate dates and no 0/0 placeholder rows (those
were already dropped upstream). SPY in the cache starts 2000-01-03, so SPY is
the binding constraint. Two sessions have SPY but no breadth reading,
2020-06-12 and 2020-06-19. After requiring the 252-session rolling high and the
longest variant lookback (21), the eligible sample is **6,440 sessions,
2000-12-29 to 2026-09-17** (about 25.7 years).

**Trigger, replicated not retyped.** The script imports
`nyse_risk.warning_severity` and builds `distance` with the same expression
`compute_nyse_main` uses, `(1 - SPY / SPY.rolling(252).max()).clip(lower=0)`.
The near-high zone is `distance <= 0.03`; severity is 1.0 below 2% and 0.6 from
2% through 3%. Severity tiers scale the dial's magnitude, not the fire set, so a
fire is `severity > 0`. The script asserts that the raw variant's fire set
equals `nyse_risk.warning_severity > 0` on all 6,717 SPY sessions and prints the
check (345 fire days on the full calendar). That assertion passes.

- `FIRE(variant)` = near_high AND variant series < 0
- `CONTROL(variant)` = near_high AND variant series >= 0

The control is variant-specific on purpose. The component's job is to split
near-high days into dangerous and healthy, so each rule is graded against the
days its own rule calls healthy.

**Variants.** raw 1d (incumbent), SMA {3,5,8,13,21}, EMA {5,8,13,21} (`ewm`
span, `adjust=False`), plus `persist_3of5` (raw < 0 on at least 3 of the last 5
sessions) as the cheap comparator. A missing breadth reading blanks every
variant for its whole trailing window, so an unknown reading can never confirm
anything. All variants are scored on one identical calendar.

**Episodes.** A fire day starts an episode if no fire occurred in the previous
10 sessions. Control days are declustered by the same rule for their clustered
standard errors.

**Returns.** lag-0 close to close, `C[i+h]/C[i] - 1`, h in 5/10/21/42/63, for
all fire days and for episode starts. Drawdown within horizon is
`min(C[i+1..i+h])/C[i] - 1` at h = 21 and 63, plus P(dd <= -5%).

**Statistics.** Day level uses a cluster-robust standard error of the mean,
clustering by episode with a G/(G-1) adjustment, reported as t versus the
variant's own control. Episode level uses the exact one-sided sign test from
`pitch_lab.sign_test`, P(at least the observed count of negative forward
returns) with p set to the control's negative rate. Episode N runs 21 to 78, so
the sign test is the decision statistic and t-stats are context only.

**Sensitivity.** Pre-2010 / 2010+ split and a leave-one-year-out sweep on the
21d episode mean. No threshold scanning beyond the grid above. The whole grid is
reported. The question asked of the split is whether the **ordering** is stable,
not which cell won.

---

## Table 1. Coverage

`state_on_days` applies the shipped reset-and-fade loop (reset on a non-negative
reading, 63-session linear fade, killed past 20% off the high) with the
variant's own series supplying both the negativity and the reset.

| variant | fire days | episodes | eps/yr | control days | state ON days | % of near-high days |
|---|---|---|---|---|---|---|
| raw_1d (incumbent) | 345 | 78 | 3.03 | 3249 | 570 | 9.60 |
| sma3 | 289 | 58 | 2.26 | 3305 | 559 | 8.04 |
| sma5 | 268 | 44 | 1.71 | 3326 | 581 | 7.46 |
| sma8 | 253 | 34 | 1.32 | 3341 | 538 | 7.04 |
| sma13 | 253 | 30 | 1.17 | 3341 | 550 | 7.04 |
| sma21 | 244 | 25 | 0.97 | 3350 | 549 | 6.79 |
| ema5 | 244 | 34 | 1.32 | 3350 | 619 | 6.79 |
| ema8 | 224 | 30 | 1.17 | 3370 | 575 | 6.23 |
| ema13 | 215 | 27 | 1.05 | 3379 | 592 | 5.98 |
| ema21 | 215 | 22 | 0.86 | 3379 | 584 | 5.98 |
| persist_3of5 | 273 | 46 | 1.79 | 3321 | 567 | 7.60 |

Smoothing cuts fire days modestly (345 to 215 at the extreme) but cuts distinct
episodes by a lot (78 to 22). It does **not** cut time spent in the ON state,
which sits in a 538 to 619 band for every variant including the incumbent. The
fade dominates the total, so what smoothing changes is the number of separate
alarms and their length, not the fraction of history under warning.

---

## Table 2. Forward SPY returns, all fire days (%)

`t` is versus the variant's own control, clustered by episode.

| variant | 21d mean | 21d t | 63d mean | 63d t | control 21d | control 63d |
|---|---|---|---|---|---|---|
| raw_1d | -0.250 | -1.69 | +0.168 | -1.84 | +0.753 | +2.490 |
| sma3 | -0.541 | -1.84 | -0.383 | -2.02 | +0.759 | +2.496 |
| sma5 | -0.585 | -1.65 | -0.510 | -1.94 | +0.754 | +2.487 |
| sma8 | -0.421 | -1.27 | -0.288 | -1.70 | +0.736 | +2.459 |
| sma13 | -0.255 | -1.04 | -0.019 | -1.59 | +0.726 | +2.445 |
| sma21 | -0.320 | -1.04 | -0.275 | -1.73 | +0.729 | +2.458 |
| ema5 | -0.802 | -1.90 | -1.254 | -2.34 | +0.759 | +2.516 |
| ema8 | -0.652 | -1.44 | -0.946 | -2.02 | +0.741 | +2.476 |
| ema13 | -0.526 | -1.24 | -0.666 | -1.82 | +0.731 | +2.456 |
| ema21 | -0.502 | -1.22 | -0.733 | -1.76 | +0.730 | +2.461 |
| persist_3of5 | -0.495 | -1.61 | -0.318 | -1.93 | +0.750 | +2.481 |

Unconditional baseline over the eligible sample: 21d +0.816%, 63d +2.503%.

On day counts the short EMAs look best (ema5 63d -1.25% at t = -2.34 against
raw's +0.17% at t = -1.84). Hold that thought until Table 6c, because a smoothed
series fires later inside the same deterioration, and a later day is closer to
the damage.

---

## Table 3. Forward SPY returns, episode starts (%)

This is the timely-warning question: what does the first alarm of a
deterioration tell you?

| variant | N | 5d mean | 21d mean | 63d mean | 21d % neg | 21d sign p |
|---|---|---|---|---|---|---|
| raw_1d | 78 | +0.324 | +0.666 | +2.147 | 35.9 | 0.270 |
| sma3 | 58 | -0.008 | +0.211 | +1.911 | 46.6 | 0.014 |
| sma5 | 44 | -0.358 | +0.338 | +1.458 | 45.5 | 0.046 |
| sma8 | 34 | +0.038 | +0.984 | +1.783 | 30.3 | 0.670 |
| sma13 | 30 | +0.456 | +1.726 | +2.571 | 24.1 | 0.880 |
| sma21 | 25 | +0.450 | +1.234 | +0.371 | 20.8 | 0.929 |
| ema5 | 34 | -0.318 | +0.173 | +0.866 | 47.1 | 0.049 |
| ema8 | 30 | -0.232 | +1.343 | +1.173 | 40.0 | 0.236 |
| ema13 | 27 | +0.059 | +1.505 | +1.256 | 30.8 | 0.644 |
| ema21 | 22 | +0.260 | +1.196 | +1.693 | 23.8 | 0.863 |
| persist_3of5 | 46 | -0.250 | +0.413 | +1.304 | 40.0 | 0.176 |

Every variant's episode-start 21d mean is **positive**, including the
incumbent's. The near-high warning does not forecast a negative 21d or 63d
return from its first alarm under any smoothing, and the ordering is not
monotone in window length (sma5 +0.34, sma8 +0.98, sma13 +1.73, sma21 +1.23).
That non-monotonicity is the signature of noise across 22 to 78 episodes, not a
window-length effect.

---

## Table 4. Max drawdown within horizon, h = 63 (%)

This is the metric that matters, since the component is a drawdown warning.

| variant | fire-day mean dd | fire-day P(dd<=-5%) | episode mean dd | episode P(dd<=-5%) |
|---|---|---|---|---|
| raw_1d | -6.12 | 51.7 | -3.88 | 30.7 |
| sma3 | -6.85 | 59.5 | -4.54 | 35.7 |
| sma5 | -7.14 | 60.9 | -4.70 | 40.5 |
| sma8 | -7.13 | 60.0 | -4.23 | 36.4 |
| sma13 | -6.95 | 58.1 | -3.98 | 31.0 |
| sma21 | -7.09 | 60.3 | -5.41 | 50.0 |
| ema5 | -7.68 | 66.5 | -4.70 | 37.5 |
| ema8 | -7.79 | 67.5 | -4.54 | 37.9 |
| ema13 | -7.69 | 66.0 | -4.55 | 34.6 |
| ema21 | -7.76 | 67.6 | -4.96 | 47.6 |
| persist_3of5 | -6.87 | 57.8 | -4.71 | 44.4 |

Control (near-high, breadth non-negative): mean dd63 about -3.68%, P(dd<=-5%)
about 23.8%. All days: -5.12%, 34.1%.

Two real things here. The near-high negative-breadth day genuinely is more
dangerous than the near-high positive-breadth day, by every variant and on both
clocks, which supports the component existing at all. And smoothing does raise
the drawdown hit rate, from 30.7% to 35-50% at episode level. The ordering
across windows is again not monotone (sma13 at 31.0% sits below sma5 at 40.5%
and sma21 at 50.0%).

At h = 21 the same shape holds and is weaker: incumbent episode mean dd -2.13%,
P(dd<=-5%) 12.8%, against control -1.91% / 8.7%. The smoothed cells reach about
-2.8% mean (sma5 -2.75%, persist_3of5 -2.81%) with hit rates spread 11.5% to
23.8%, that last from ema21's 21 episodes.

---

## Table 5. Era split, 21d episode edge (episode mean minus control mean, pp)

| variant | pre-2010 (N ep) | 2010+ (N ep) |
|---|---|---|
| raw_1d | -1.43 (18) | -0.21 (60) |
| sma3 | -2.49 (12) | -0.75 (46) |
| sma5 | -1.52 (7) | -0.80 (37) |
| sma8 | -0.89 (5) | -0.21 (29) |
| sma13 | -1.19 (6) | +0.63 (24) |
| sma21 | +0.15 (6) | -0.11 (19) |
| ema5 | -2.43 (5) | -0.89 (29) |
| ema8 | -0.59 (5) | +0.24 (25) |
| ema13 | -1.03 (6) | +0.32 (21) |
| ema21 | -0.04 (5) | -0.20 (17) |
| persist_3of5 | -1.46 (9) | -0.65 (37) |

Spearman rank correlation of the edge between eras: **0.69**. That is moderate,
not stable, and it is computed on 11 points where the pre-2010 cells carry 5 to
18 episodes each. Four variants flip sign across the split (sma13, sma21, ema8,
ema13). Every variant's edge is weaker post-2010 than pre-2010, the incumbent
included, which is the usual breadth-signal decay story and not a smoothing
question.

## Table 6. Leave-one-year-out, 21d episode mean

For a warning signal the number that matters is the **worst** leave-one-out
mean, that is the least negative.

| variant | N ep | full mean | LOYO min | LOYO max (floor) | worst year dropped |
|---|---|---|---|---|---|
| raw_1d | 78 | +0.67% | +0.47% | +0.88% | 2007 |
| sma3 (best smoothed by full-sample edge) | 56 | +0.21% | +0.03% | +0.47% | 2007 |
| persist_3of5 | 45 | +0.41% | +0.17% | +0.68% | 2007 |

No variant's 21d episode mean goes negative under any single-year exclusion.
Both survive LOYO in the sense that nothing depends on one year, and both fail
the more important test, which is that the level never gets below zero.

---

## Table 6b. What smoothing actually selects (diagnostic, not an edge)

For each of the incumbent's 78 episode starts, the variant's first fire in the
window [-5, +21] sessions. CONFIRMED means the variant also fires on that
deterioration. dd63 is measured from the **incumbent's** anchor for both groups,
so the two sit on one clock.

| variant | confirmed | missed | median lag (td) | dd63 confirmed | dd63 missed | P5 confirmed | P5 missed |
|---|---|---|---|---|---|---|---|
| sma3 | 54 | 24 | 2 | -4.71% | -2.13% | 37.3% | 16.7% |
| sma5 | 39 | 39 | 4 | -5.33% | -2.54% | 47.2% | 15.4% |
| sma8 | 29 | 49 | 7 | -5.24% | -3.12% | 48.1% | 20.8% |
| sma13 | 21 | 57 | 10 | -5.55% | -3.28% | 50.0% | 23.6% |
| sma21 | 18 | 60 | 8 | -7.59% | -2.80% | 76.5% | 17.2% |
| ema5 | 38 | 40 | 4 | -5.10% | -2.82% | 42.9% | 20.0% |
| ema8 | 28 | 50 | 6 | -5.71% | -2.91% | 53.8% | 18.4% |
| ema13 | 21 | 57 | 8 | -5.56% | -3.27% | 55.0% | 21.8% |
| ema21 | 15 | 63 | 6 | -6.27% | -3.33% | 71.4% | 21.3% |
| persist_3of5 | 42 | 36 | 3 | -5.25% | -2.32% | 47.5% | 11.4% |

Reference, all 75 incumbent episodes with a 63d window: mean dd63 -3.88%,
P(dd<=-5%) 30.7%. The near-high positive-breadth control sits at 23.8%.

The pattern is monotone and clean across the whole grid. The incumbent fires
that smoothing throws away land **below the healthy-day control** on drawdown
risk (11-24% versus 23.8%), which says the dropped fires are one-day breadth
blips that mean nothing.

**Caveat, and it is a big one.** The CONFIRMED label uses breadth information
from up to 21 sessions after the incumbent's anchor, while dd63 is measured from
that anchor. A deterioration that turns into a drawdown keeps printing negative
breadth, so part of this separation is circular by construction. Table 6b says
what smoothing selects. It is not a tradeable edge, and the `sign_p_confirmed`
column in `timeliness.csv` should be read the same way.

## Table 6c. The delay placebo (the decisive test)

A smoothed series fires later, and a later anchor sits closer to the damage.
Placebo = the incumbent's episode start pushed forward by the variant's median
lag, using no breadth information at all. If the placebo matches the variant,
smoothing is just waiting.

| variant | median lag | N ep | variant dd63 | placebo dd63 | variant P5 | placebo P5 | P5 gain (pp) | gain / sigma |
|---|---|---|---|---|---|---|---|---|
| sma3 | 2 | 56 | -4.54% | -3.89% | 35.7% | 32.0% | +3.7 | 0.60 |
| sma5 | 4 | 42 | -4.70% | -3.89% | 40.5% | 32.0% | +8.5 | 1.18 |
| sma8 | 7 | 33 | -4.23% | -3.97% | 36.4% | 32.0% | +4.4 | 0.54 |
| sma13 | 10 | 29 | -3.98% | -4.29% | 31.0% | 30.7% | +0.4 | 0.04 |
| sma21 | 8 | 24 | -5.41% | -4.19% | 50.0% | 33.3% | +16.7 | 1.73 |
| ema5 | 4 | 32 | -4.70% | -3.89% | 37.5% | 32.0% | +5.5 | 0.67 |
| ema8 | 6 | 29 | -4.54% | -4.07% | 37.9% | 33.3% | +4.6 | 0.53 |
| ema13 | 8 | 26 | -4.55% | -4.19% | 34.6% | 33.3% | +1.3 | 0.14 |
| ema21 | 6 | 21 | -4.96% | -4.07% | 47.6% | 33.3% | +14.3 | 1.39 |
| persist_3of5 | 3 | 45 | -4.71% | -3.86% | 44.4% | 33.3% | +11.1 | 1.58 |

The sigma column is an indicative noise yardstick, not a formal test, because
the variant and placebo samples overlap heavily.

Every variant beats its own delay placebo on mean dd63, by 0.3 to 1.2 percentage
points, and nine of ten beat it on the 5% hit rate. So smoothing is not purely
lag. But **no cell clears 1.8 sigma**, the gain is not monotone in window length,
and the two largest gains (sma21 +16.7pp, ema21 +14.3pp) come from the two
smallest samples (24 and 21 episodes), which is the classic shape of noise. The
two mid-size cells with a real-looking gain, sma5 (+8.5pp, 1.18 sigma) and
persist_3of5 (+11.1pp, 1.58 sigma), agree with each other closely, which at
least says the effect is **persistence** rather than any specific weighting
scheme.

---

## Table 7. Practical, last 24 months (502 sessions, 2024-09-17 to 2026-09-17)

| variant | fire days | state ON days | episodes |
|---|---|---|---|
| raw_1d | 63 | 94 | 15 |
| sma3 | 50 | 92 | 9 |
| sma5 | 43 | 51 | 6 |
| sma8 | 42 | 48 | 4 |
| sma13 | 39 | 45 | 4 |
| sma21 | 38 | 44 | 4 |
| ema5 | 43 | 98 | 7 |
| ema8 | 38 | 46 | 5 |
| ema13 | 32 | 38 | 5 |
| ema21 | 33 | 42 | 5 |
| persist_3of5 | 42 | 69 | 7 |

Note `ema5`: fewer fires than the incumbent (43 vs 63) but **more** ON days
(98 vs 94), because an EMA takes longer to climb back above zero so the reset is
rarer and each fade runs to term. Smoothing trades many short warnings for fewer
long ones. It does not reduce exposure to the warning state.

## Aug-Sep 2026

First fire of the current episode, and what it cost:

| rule | first fire | SPY at that close |
|---|---|---|
| incumbent (raw 1d) | 2026-08-17 | 772.67 |
| sma3, sma5, ema5, ema8 | 2026-08-18 | 767.45 |
| sma8, persist_3of5 | 2026-08-20 | 762.60 |
| sma13, ema13 | 2026-08-31 | 767.05 |
| ema21 | 2026-09-01 | 761.78 |
| sma21 | 2026-09-02 | 765.16 |

SPY closed 762.60 on 2026-09-17 and troughed at 757.39 on 09-15, so the whole
episode is a 2% pullback so far and the incumbent's 10-day head start over
sma13/ema21 bought little.

What it did cost the incumbent is flicker. Raw `nyse_net` printed non-negative
on 08-19, 08-25, 08-26, 08-27 and 08-28, and each of those cleared the
component's state and both smoothing queues outright. The dial's NYSE
contribution went to zero and rebuilt from scratch three separate times inside
eleven sessions. sma5 broke twice for one day each, ema5 once, and sma13 and
ema21 never broke at all once they turned on. Day-by-day detail is in
`last30_sessions.csv` and `aug_sep_2026.csv`.

One thing worth flagging that is not a smoothing question: on 2026-09-16, the
worst raw breadth day of the stretch at -161 with SPY 3.06% off its high, every
variant including the incumbent stops firing because the near-high zone ends at
3%. The fade carried the state through, so the dial did not collapse, but the
3% edge is a hard cliff on the one day the tape looked worst.

---

## Conclusion

1. Smoothing does not improve the warning where it counts. From the first alarm
   of a deterioration, every variant's 21d and 63d forward SPY return is
   positive, the incumbent's included, and the ordering across window lengths is
   not monotone in either era.
2. Smoothing does raise the drawdown hit rate, 30.7% to 35-50% on P(dd63 <= -5%),
   but most of that is lag. Against a placebo that just waits the same number of
   days, no variant clears 1.8 sigma and the two biggest gains come from the two
   smallest samples.
3. The one substantive thing smoothing buys is fewer false alarms and far less
   flicker: it cuts 78 episodes to 22-58 while leaving total ON time unchanged
   at 538-619 days, and the incumbent fires it discards score below the
   healthy-day control on drawdown risk.
4. The cost is 2 to 10 trading days of lag and, for the longer windows,
   dropping half to three quarters of the incumbent's episodes, including real
   ones (missed episodes still average -2.1% to -3.3% dd63).
5. **Would I change the shipped 1d trigger? Not on this evidence.** If the
   flicker in Aug 2026 is the actual complaint, the fix is the reset rule rather
   than the series, since a one-day non-negative print wiping both queues is
   what produced it. A 5-period smoothing is the most defensible candidate if
   one is wanted anyway (sma5 and persist_3of5 agree, which points at
   persistence rather than a weighting scheme), but it should go through a
   pre-registered protocol with the reset semantics specified, not ship off this
   grid.
