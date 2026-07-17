# Pre-registration: RMW seasonality ETF confirmation (2026-07-17)

Written BEFORE running the confirmation. The French-factor study
(scratch/factor_seasonality_study.py, data through 2026-05) found two
quality-factor cells that survive every era including post-2013:

- H1 JANUARY: RMW negative in January (junk rallies). Full sample
  -0.96%/mo t=-3.9; post-2000 -0.76 t=-1.6; post-2013 -0.71 t=-1.3.
- H2 JULY: RMW positive in July. Full sample +0.66 t=+2.6; post-2000
  +1.12 t=+2.2; post-2013 +0.61 t=+1.3.

## Test

Vehicles: QUAL (iShares MSCI USA Quality, 2013-07+) primary; SPHQ
(2005-12+) secondary with an index-provenance caveat (tracked Value Line
Timeliness, then S&P High Quality Rankings, quality-index only since
~2010; pre-2010 SPHQ months are excluded). Benchmark SPY. Monthly
total-return spreads from adjusted closes.

Predictions, conditional on the vehicles actually loading on RMW:
1. LOADING GATE (informativeness, not pass/fail): full-overlap monthly
   regression beta of (QUAL-SPY) on RMW >= 0.20. If beta < 0.20 the ETF
   cannot express the factor and the test is UNINFORMATIVE — record that,
   do not count it as a failure of the factor cells.
2. H1: mean January (QUAL-SPY) spread NEGATIVE with t <= -1.0.
3. H2: mean July (QUAL-SPY) spread POSITIVE with t >= +1.0.
   (t thresholds are deliberately modest: ~13 monthly observations per
   cell; direction + magnitude consistency with beta x factor-cell is the
   real content.)
4. MAGNITUDE: observed cell spread within [0.25x, 4x] of beta x the
   French cell mean (sign-correct). Outside that band = suspicious even
   if the t passes.

## Decision rules (pre-committed)

- BOTH H1+H2 pass (loading gate met): draft a sizing proposal for a
  QUAL/SPY monthly tilt (underweight quality in Jan, overweight in Jul)
  through the seasonal/trend execution path. Pilot conviction only.
- ONE passes: park; recheck after the next occurrence of the failing
  month. No trade.
- BOTH fail with wrong sign at t beyond +/-1: drop the RMW-seasonality
  thread entirely.
- Loading gate fails: the thread survives but needs a different vehicle
  (e.g. long-short baskets), which raises costs enough that it likely
  parks indefinitely.

## Not tested here, explicitly

No new months, no other factors, no threshold search. Any cell not named
above that looks good in the output is NOT a finding (multiple testing);
it goes back through a fresh pre-registration if it seems worth it.
