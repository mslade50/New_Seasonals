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

## Amendment 2026-07-17 (same day, McKinley): price-exit confirmation

The price exit is amended to require **2 consecutive closes** below the 95%-
of-252d-high line (the dial exit at >=25 stays single-evaluation — its input
is already 10d-MA-of-5d-smoothed and cannot whipsaw daily). Chosen from a
5-variant comparison (scratch/dial_exit_variants.py: single close / 2-consec
/ trigger-low close / trigger-low intraday stop) on SIMPLICITY — all
variants agreed within noise (CAGR 9.0-9.3%, Sharpe 1.13-1.17), turnover
fell ~26% (3.4 -> 2.5 round trips/yr). IN-SAMPLE SELECTION, disclosed; the
PIT gate judges the amended package as a whole. Amended reference: CAGR
9.3%, Sharpe 1.17 overall / 2.20 in-market, maxDD -5.6%, in-market 29%.

## Gates before any live capital

1. **One-shot hysteresis run** — CLOSED POSITIVE 2026-07-17: dial hysteresis
   (20/25) vs no-hysteresis reference improved CAGR +0.8pp and Sharpe +0.10
   at unchanged turnover (scratch/dial_exit_variants.py, A vs B). The
   reference did not depend on whipsaw luck. No re-tuning of 20/25 afterward.
2. **PIT forward gate**: paper-track the rule on the append-only PIT segment
   only. Minimum 2 quarters AND >= 6 signal transitions before sizing
   anything. Metric: sign agreement of regime calls with forward 21d SPY
   (does flat-time actually coincide with sub-baseline tape), not CAGR
   (2 quarters of CAGR is noise).
3. **Co-exposure statement — RESOLVED 2026-07-17**: the exposure leg is NOT
   traded live (McKinley), so the audit target shifts to the trend sleeve.
   Resolution: trend sleeve cut 0.6x -> 0.3x NAV (trend_sleeve.py, same
   day), capping worst-case clean-air-day US index exposure at ~43% NAV
   (25% dial-sleeve SPY + 6% each trend SPY/QQQ/IWM). Expected NAV vol:
   trend ~2.4%/yr, dial ~1.5%/yr, combined ~3.4%/yr (monthly corr +0.37;
   scratch/overlay_vol_estimate.py). Sleeve sizing: nominal 25% NAV
   (proposed 2026-07-17; final sign-off when gate 2 clears).
4. Standard disclosures on any output: recompute-vintage caveat, today's-
   code lookahead caveat, one-decade/three-drawdowns caveat.

## What would kill it

- PIT-segment regime calls no better than coin-flip after gate 2's window.
- The hysteresis run revealing the reference depended on whipsaw luck
  (reference materially degrades under the stated exit buffer).
- The overlap audit concluding the exposure leg already captures the theme
  (in which case fold the near-high condition into the leg's replay instead
  and close this as merged, not shipped).
