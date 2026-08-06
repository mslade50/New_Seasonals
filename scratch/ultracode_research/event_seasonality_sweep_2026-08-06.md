# Event-calendar x seasonality sweep (2026-08-06)

Data: `data/macro_events.csv` (new macro event calendar, 2000-2027) joined to
SPY/QQQ/IWM/TLT from master_prices, 2000+. Scripts:
`scratch/event_seasonality_sweep.py` (broad grid, results in
`scratch/event_sweep_results/`) and `scratch/event_sweep_drilldown.py`
(per-year tables for the survivors). All returns are close-to-close, no
costs, no sizing. Nothing here is shipped; every candidate below needs its
own pre-registered protocol per house rules.

Method note: the broad sweep scanned ~600 cells, so isolated |t| around 2
means little on its own. The three survivors were kept because each has a
prior in the literature (Lucca-Moench pre-FOMC drift; quarter-end September
weakness; the Santa window) and each held up under drop-best-year and
cross-ticker checks.

## Survivor 1: Pre-FOMC drift, ex-midterm years

Long SPY from the close 4 sessions before a scheduled FOMC decision through
the decision-day close (td -3..0). 2000-2026:

- All meetings: +30 bps/meeting, t 2.05, N 212, hit 59%
- Ex-midterm years: **+53 bps/meeting, t 3.11, N 159, hit 62%**
- Midterm years only: -38 bps, t -1.34, N 53, hit 49%
- Confirmed on QQQ (+62, t 2.47) and IWM (+57, t 2.60) ex-midterm
- 90% of ex-midterm years positive; worst year 2011 (-1500 bps across its
  meetings); drop-best-year mean still +358 bps/yr
- The runup conditioning (G1): the drift is a mid-tape phenomenon. Trailing
  5d return below the 20th percentile kills it (-11 bps). Above the 80th it
  is +33 but the post-window goes slightly negative, consistent with the
  dashboard's Pre-FOMC Rally fragility signal.

The midterm inversion rhymes with the OVS midterm evidence already in the
book (cycle_risk_mults). Whatever makes midterm years hostile to short-vol
also shows up around FOMC windows.

Candidate spec: 8 windows/yr in non-midterm years, 4-day hold, ~32 days/yr
in market, roughly +420 bps/yr gross at full notional. Could run as a small
overlay sleeve like exposure_leg (which already has dial kill rules that
would need to compose), or as sizing tailwind/headwind on dip-buys whose
hold straddles a decision.

## Survivor 2: September post-quad-witching risk-off rotation

From the September opex close to month-end (roughly 7 sessions), 2000-2025:

| leg | mean | t | hit |
|---|---|---|---|
| SPY | -87 bps | -1.43 | 42% |
| QQQ | -115 bps | -1.62 | 46% |
| IWM | -157 bps | -1.95 | 35% |
| TLT | **+95 bps** | +1.75 | 62% |

The post-opex week alone is sharper: IWM -179 (t -2.88, hit 23%), TLT +109
(t +3.02, hit 73%). First-week-of-September baseline is flat (-7 bps), so
this is specifically the back half after expiry, not generic September.
Drop-best-year makes the SPY number stronger (-120, t -2.21), so it is not
one crash year. 2008/2011/2022 are the tails but 2013, 2014, 2015, 2019,
2021, 2023 all bled too.

Candidate spec: long TLT and/or short IWM (or simply de-rate new equity-long
staging) from Sep opex close to month-end. The TLT leg is the cleanest and
cheapest to run. Note TLT history starts 2002 and duration had a brutal
2022 (-459 bps worst), so the bond leg is not a free hedge in inflation
regimes.

## Survivor 3: December post-opex into year-end (Santa, anchored)

From the December opex close to the last session of the year (~7 sessions),
2000-2025:

- SPY +66 bps, t 2.19, hit 62%; IWM +85 bps, t 2.30, hit 65%
- Post-opex week alone: IWM +98, t 2.41, hit 73%
- QQQ does NOT confirm (-18 bps): the effect tilts small-cap/value, which
  fits the tax-loss / window-dressing story
- Robust both directions: drop-best +53 (t 1.87), drop-worst +76 (t 2.60)
- G2 check: conditioning on the Dec FOMC day sign adds nothing (+47 either
  way), so the anchor is opex, not the Fed

Candidate spec: long IWM Dec opex close -> year-end. One trade/yr, N=26 is
modest; grade as a book-adjacent seasonal ticket rather than a strategy.

## Dead / not worth pursuing

- CPI day effects: nothing at day level in any era, overnight or intraday,
  SPY or TLT. Surprising given 2022 lore, but the unconditional day carries
  no edge. (Direction-conditioning on the print itself would need surprise
  data, out of scope.)
- NFP: overnight-into-release +7 bps (t 1.8) on SPY, too thin to trade;
  TLT mirror -9 (t -1.7). Turn-of-month with vs without NFP: no spread.
- FOMC minutes days: nothing.
- Jackson Hole: into-keynote +59 bps (t 1.6, N 26) fades to nothing 2013+.
  Not tradeable, N tiny.
- Elections: pre-election week +268 bps (t 1.8) but N=13 and the midterm
  cell is 5-of-6; log it as folklore-consistent, park until N grows.
- Santa conditioned on Dec FOMC sign: no conditioning value (see above).
- Post-FOMC fade overall: the -10/-13 bps td+1/+2 seen 2016+ does not hold
  over the full sample (t 0.05 all years); it concentrates in midterm years
  (-47 bps, t -1.37). The tradeable form is already captured by staying out
  of midterm FOMC windows (Survivor 1's exclusion), not by shorting.

## Suggested gates before anything ships

1. Pre-registered protocol (decision rule, entry/exit, sizing, kill
   criteria) written BEFORE any further backtesting, per the fragility
   discipline.
2. Episode-clustered t on the windows as specified (each window one obs,
   already the drilldown convention).
3. LOYO floor: drop each year in turn, require the effect to hold.
4. Execution reality: windows enter at a known close; model next-open entry
   slippage as the pessimistic variant.
5. Overlap audit vs the existing book: Survivor 1 overlaps the exposure_leg
   overlay and the Pre-FOMC Rally fragility windows; Survivor 2 overlaps
   FAMILY4 dip-buy entries (a de-rate there would interact with frag bands);
   Survivor 3 overlaps the Seasonal sheet tickets. None of these should ship
   as an independent sleeve without checking combined exposure on the same
   dates.
