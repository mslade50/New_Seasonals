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

## Addendum (same day): opex as a turning point after outsized moves

Question from McKinley: does opex mark reversal turning points after big
moves INTO expiry? Study: `scratch/opex_reversal_study.py`. Pre-window
return td -5..0 and -10..0 into the opex close, z-scored by trailing 21d
realized vol (measured before the window), bucketed, forward windows
+1..+5 / +1..+10 / to month-end. SPY/QQQ/IWM, 2000+, ~318 opex.

**Downside: yes, strongly.** Outsized selloffs into opex bounce after:

- SPY z10 < -1.5 (N=19): fwd5 +213 bps t 3.1, to month-end +263 t 3.1,
  hit 74-79%
- Quad-only, same cut (N=7): +388 / +469 / +422 (2018-12, 2020-03,
  2022-06 in the list, but 2022-12 and 2024-12 failed)
- QQQ confirms hard: quad z10 < -1 (N=15) fwd5 +282 t 4.2, hit 93%
- IWM confirms (z5 < -1.5: fwd5 +211 t 2.9; quad +387 t 2.6)
- Failures on record: 2018-10 (-395 fwd5), 2019-05, 2022-12, 2024-12,
  2026-03. Roughly one in four keeps falling.

**Upside: no.** Big rallies into opex do not reverse. SPY big-up buckets
are flat (t 0 to -1.1); QQQ big-up actually CONTINUES (+72/+96 bps,
t 1.6-1.7); IWM monthly extreme-up continues (+279 fwd10, t 2.3). The only
whiff of upside fade is SPY quad z10 > +1.5 at N=9 (-75 fwd5, t -1.8),
too small to trade. The asymmetry is the finding: expiry releases a
downside pin, it does not cap strength.

**Refinement of Survivor 2:** the Sep/Jun post-quad drag lives in the MILD
bucket (calm tape into expiry: SPY quad mild fwd5 -42 t -1.6, QQQ -65
t -2.2). When the tape already sold off hard into quad expiry the drag
inverts to a bounce. Any Sep post-quad de-rate should carry a washout
exception (skip the de-rate when z10 into opex < -1).

Overlap note: the washout-bounce cell is close kin to the existing dip-buy
family (Indices Oversold Bounce et al). The marginal content is the timing
anchor (expiry release), so the natural form is a sizing boost or entry
timing on dip-buys landing in the week after opex, not a new strategy.

## Addendum 2 (same day): UVXY / vol around the same events + seasonality

Study: `scratch/uvxy_event_study.py`. UVXY 2011-10+ (adjusted; structural
decay -37.5 bps/day, t -3.1 — THE hurdle for long vol and THE harvest for
short vol), ^VIX 2000+ for the clean seasonal shape. `vix_expiry` added to
the macro calendar (computed, 30 cal days before next month's SPX opex).

Event findings (all vs the -150 bps/4d decay baseline):

- **Pre-FOMC vol crush is real and mirrors T1**: UVXY td-1 -133 (t -2.4),
  td0 -153 (t -2.2); the -3..0 window shorts for +352 bps ex-midterm
  (t 2.3, 72% of windows). In MIDTERM years the crush disappears (-59,
  n.s.) — same regime split as the equity drift.
- **Dec opex -> year-end vol crush mirrors T4**: short side +693 bps,
  t 3.3, 87% hit (15 windows).
- CPI-day crush exists (td0 -106, t -2.0) but is thin after decay.
- VIX-expiry roll pressure shows at td-2 (-143, t -2.2); not tradeable
  alone. The big VIX +309 day after NFP is the Friday->Monday quotation
  effect (VIX weekend decay), NOT tradeable — UVXY shows nothing there.
- **Sep post-quad long UVXY (T3 companion)**: +562 bps mean, 64% hit, but
  t 1.4 with -2459/-1657 tails — noisy; T3's IWM short is the better
  vehicle.

Seasonality:

- Monthly decay concentrates in **July (-94 bps/day, t -3.1) and November
  (-100, t -2.6)**; August is the only positive UVXY month while VIX
  builds +52/+60 bps/day through Aug-Sep.
- Naive long-vol-for-the-fall (Aug 1 -> Oct 15) LOSES: -836 bps avg, 43%
  hit — decay eats the seasonal VIX rise. The exception is MIDTERM years:
  +2653 bps, 3-for-3 (2014/2018/2022). N=3, curiosity only, but aligned
  with every other midterm risk finding.
- **Short UVXY Nov 1 -> Dec 31: +2625 bps avg on the short side, t 4.5,
  80% hit (12 of 15).** Mostly decay harvest timed at the calmest
  seasonal window (Nov is the only month VIX FALLS on average).

Verdict / tails: the tradeable vol expressions largely DUPLICATE the
equity sleeve (pre-FOMC crush ~ T1, Dec crush ~ T4) with far worse tails:
worst single pre-FOMC short window -45% of shorted notional (2011-11),
and Nov-Dec windows see intra-window UVXY run-ups of +54% to +84% in a
third of years (2018: +84%, window closed -36%). NAKED UVXY SHORTS ARE
NOT RECOMMENDED at book size; the defined-risk implementation (UVXY put
spreads / VIX call spreads for the midterm long-vol pocket) is an options
venture outside current execution infra. PARKED: revisit if/when options
staging exists. If a small cash short is ever run despite this, the
Nov-Dec window at <= 2% NAV notional with a hard 50% buy-stop is the only
cell with the stats to justify it.

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
