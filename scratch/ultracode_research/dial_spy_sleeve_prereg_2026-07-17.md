# Pre-registration: dial-gated SPY sleeve ("clean-air sleeve")

Registered 2026-07-17, at the moment of the threshold decision, before any
further variants are tested. Studies behind it: scratch/dial_decile_fwd*.py,
dial_jurisdiction.py, dial_spy_strategy.py, dial_spy_execution.py,
dial_threshold_curve.py, dial_sharpe_decomp.py (all 2026-07-16/17).

## Frozen specification

- **Rule**: long SPY when BOTH hold as of the prior close:
  (a) SPY close within 5% of its trailing 252d max (min_periods 60),
  (b) 10d MA of the 63d fragility dial (data/rd2_fragility.parquet, the
      sizing statistic) **< 20**. Otherwise T-bills.
- **Threshold 20 chosen by McKinley** over the marginally-better-scoring 22
  explicitly as an anti-overfit round number; the threshold curve is flat
  (overall Sharpe ~1.0-1.2 for T in 22-45; in-market Sharpe declines
  monotonically with T). The per-dollar-deployed argument (in-market Sharpe
  ~2.2, maxDD -6%, capital free 76% of the time for the systematic book)
  is the reason for a tight T over a participatory one.
- **Hysteresis (named now, deliberately NOT backtested at registration)**:
  enter when dial < 20; exit when dial >= 25 or the near-high condition
  fails. The 5-point exit buffer is a stated convention to cut the ~37%
  whipsaw-spell fraction, not a tuned value. It gets ONE historical test at
  evaluation, reported win or lose, never scanned.
- **Execution**: signal from the 21:15 UTC risk-report append; MOO (TIF=OPG)
  at the next NYSE open; assume 2 bps/side all-in for SPY.
- **Sizing**: unset here — a capital-allocation decision (candidate: the
  exposure-leg pattern, fraction of NAV). Not part of this registration.

## Reference performance under the frozen spec (in-sample, 2016-07..2026-07)

CAGR 8.2%, overall Sharpe 1.05, in-market Sharpe 2.16, maxDD -6.1%,
time-in-market 24%, ~6.8 switches/yr (dial_threshold_curve.py, T=20 row,
no hysteresis). These are the numbers to beat/match, not evidence — the
threshold family was examined in-sample and the dial history before
2026-07-02 is recompute vintage computed with today's signal code.

## Gates before any live capital

1. **One-shot hysteresis run** (enter<20 / exit>=25): report vs the frozen
   no-hysteresis reference. No re-tuning of either number afterward.
2. **PIT forward gate**: paper-track the rule on the append-only PIT segment
   only. Minimum 2 quarters AND >= 6 signal transitions before sizing
   anything. Metric: sign agreement of regime calls with forward 21d SPY
   (does flat-time actually coincide with sub-baseline tape), not CAGR
   (2 quarters of CAGR is noise).
3. **Overlap audit vs exposure_leg**: this sleeve and the exposure leg are
   near-duplicates (index long, dial kill). Before going live, either merge
   the specs (the leg gains the near-high condition and the tighter
   threshold — needs its own B2-style replay) or cap combined index
   exposure explicitly. Running both unreconciled double-counts the theme.
4. Standard disclosures on any output: recompute-vintage caveat, today's-
   code lookahead caveat, one-decade/three-drawdowns caveat.

## What would kill it

- PIT-segment regime calls no better than coin-flip after gate 2's window.
- The hysteresis run revealing the reference depended on whipsaw luck
  (reference materially degrades under the stated exit buffer).
- The overlap audit concluding the exposure leg already captures the theme
  (in which case fold the near-high condition into the leg's replay instead
  and close this as merged, not shipped).
