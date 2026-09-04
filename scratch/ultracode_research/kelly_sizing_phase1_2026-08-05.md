# Kelly Sizing — Phase 1 Repo Archaeology (2026-08-05)

Status: phase 1 complete; research only. No live/config/engine file was changed.

## Bottom line

The brief has the right economic framing — optimize static relative multipliers
on flat-$750k daily MTM streams, retain the 250 bps per-strategy staging cap,
separate OVS P1/P2, shrink noisy means, and treat overflow results as
survivorship-flattered. Four current-state facts need correcting before the
Kelly work can be trusted:

1. Production no longer has pooled 500L/250S daily caps. They were removed on
   2026-07-16; only the 250 bps per-strategy/day staged-risk cap remains.
2. The live strategy book now has 15 strategies, not 14. Monthly Weak Close was
   added after the current ledger vintage and therefore has no ledger rows.
3. `data/backtest_trades_full.parquet` is a local 2026-07-21 build at git
   `5f0dd33` (4,694 rows, last signal 2026-07-17), while the current repo is at
   `d97df8b`. It predates the large-gap derates, 3x-fade stacking, the OLV
   signal-recency ladder, the St OS Sznl earnings derate, and Monthly Weak
   Close. It is not a current-book sizing basis.
4. OVS's 2,429 ledger rows are not 2,429 independent trades. Scale-outs book
   near/far tranches as separate rows. Collapsing rows by
   `(tier, ticker, signal date, entry date, entry price)` gives 1,243 filled
   signals: 651 P1 and 592 P2. P2 is therefore not N=407 in this vintage; that
   older figure predates removal of the engine-only P1-budget gate. Episode N
   will be smaller again.

Primary sources: `CLAUDE.md` sizing/cap/strategy sections;
`strategy_config.py`; `pages/strat_backtester.py::process_signals_fast` and
`get_daily_mtm_series`; `scripts/build_trade_ledger.py`; ledger parquet
metadata; and the direct OVS collapse described above.

## Current framework

### Decision variable and basis

`risk_bps` is per-signal risk against a flat $750,000 sizing account, not a
capital allocation. The current GRM is 1.5 and scales base `risk_bps`, OVS
P1/P2 bps and P2 aggregate-cap percentage, earnings override bps, and the OLV
overflow override at import. Daily MTM marks each filled position from entry
through exit using forward-filled closes, then reconciles the exit day to the
booked fill so the daily vector sums exactly to realized PnL. This makes
strategy daily streams the right first-order Kelly objects: signal frequency,
holding periods, overlap, scale-outs, and realized gap/slippage tails are
embedded in dollars per day.

Sources: `strategy_config.py` GRM loop; `pages/strat_backtester.py` lines
2028-2131; `scripts/build_site.py::build_strategy_daily` and
`build_trade_mtm`.

### Current base sizes and live overlays

| Strategy | Nominal bps | Effective bps at GRM 1.5 | Static/dynamic sizing features relevant to the replay |
|---|---:|---:|---|
| 52wh Breakout | 35 | 52.5 | No size overlay; dial entry gate is a signal filter, not a multiplier |
| Weak Close Decent Sznls | 35 | 52.5 | Hard-coded seasonal tier 0.66x/1.0x/1.5x; fragility 0.25x at >=50 |
| Oversold Low Volume | 35 liquid / 25 overflow | 52.5 / 37.5 | 21td signal-recency ladder 0.5/0.7/1.0; -10..0 TD earnings replacement 10 nominal, composed with recency; 50% ticker/NAV notional cap; vol-confirmed stop |
| Overbot Vol Spike | P1 40 / P2 8 | 60 / 12 | P2 aggregate cap 1.125% effective; midterm 0.75x; +/-10 TD blackout; 40/60 near/far scale-out; Friday EOD-DD; ATR-Extended precedence |
| LT Trend ST OS | 30 | 45 | Earnings blackout in the engine when configured; otherwise no size overlay |
| St OS Sznl | 40 | 60 | -5..-1 TD earnings replacement at 6 nominal / 9 effective |
| 3x ETF Overbot Fade | 40 | 60 | Same-ticker stacking enabled; no fragility cut |
| 3x Bear ETF Overbot Fade | 25 | 37.5 | Pilot/frozen; stacking; fragility 0.25x at >=50; same-day signal de-rate 10% per extra signal, floor 0.30 |
| 3x Leader Gap Fade | 25 | 37.5 | Pilot/frozen; no fragility or same-day derate; 250 bps/day cap is its cluster backstop |
| Indices Oversold Bounce | 35 | 52.5 | Fragility 0.25x at >=50; same-date/same-tradeable overlap clamp with MonFri to 20 nominal each |
| SPY QQQ MonFri Reversion | 35 | 52.5 | Fragility 0.25x at >=50; T+1 large-gap half-size; overlap clamp with Indices OS |
| Sector BO | 25 | 37.5 | No size overlay |
| Monday Dip | 30 | 45 | Fragility 0.25x at >=50; T+1 large-gap half-size |
| ATR Extended Gap Up | 40 | 60 | Takes precedence over same-symbol OVS after its T+1 gate |
| Monthly Weak Close | 30 | 45 | Pilot/frozen; fragility 0.25x at >=50; absent from current ledger vintage |

Sources: imported current `STRATEGY_BOOK` inspection; `CLAUDE.md` sections
Sizing Conventions through OLV Entry-Order Live Window; `daily_scan.py` sizing
steps 2-5c.

The active per-signal order is more nuanced than the brief's shorthand. WCDS's
seasonal tier is applied first. Current live carriers then pass through
fragility, recency, cycle, and earnings replacement; the replacement clobbers
all earlier multipliers except OLV recency. ADV and OLV notional caps act on
shares. Same-day derating and overlap clamping occur before order staging's
250 bps per-strategy cap. The engine reaches the same result for current
carriers, but its generic code orders earnings before cycle/fragility and
applies the overlap clamp after its daily-cap pass. That ordering mismatch is
currently low-impact because no present carrier combines the conflicting
fields and the overlap pair does not approach 250 bps/day, but it matters when
designing a generic replay.

### Daily caps and nonlinear response

The current production cap is 250 bps of staged risk per strategy per signal
date, applied pro rata. The pooled 500L/250S layer is retained only as dormant
engine counterfactual machinery. OVS P2 also has its own daily aggregate cap,
and OLV has a per-ticker notional cap. These make a per-strategy multiplier's
response nonlinear on cluster days.

`process_signals_fast(risk_multipliers=...)` is unsuitable for the required
engine replay because it multiplies both trade risk and that strategy's daily
cap. A faithful replay must deep-copy the book and rescale its size fields while
leaving `cap_bps=250` fixed. Independent OVS P1/P2 proposals require separate
changes to `risk_bps/path1_bps` and `path2_bps` (and an explicit policy for
whether the P2 aggregate cap stays fixed or scales); a single strategy
multiplier cannot express the split.

Sources: `pages/strat_backtester.py::process_signals_fast`, especially sizing
step 3c and the post-loop cap; `scratch/grm_replay_study.py` comment explaining
why `risk_multipliers` cannot be used.

### Ledger and MTM mechanics

`scripts/build_trade_ledger.py` builds liquid plus six overflow passes twice on
identical candidates: compounded sizing and flat sizing. It joins flat shares,
risk, PnL, and pre-cap `Size_Mult` onto the compounded rows, then writes trade
and book-daily parquets. The site recomputes per-strategy/tier daily MTM by
grouping the flat-shaped ledger and calling `get_daily_mtm_series`; per-trade
vectors use the same close-mark and exit-reconciliation convention.

The flat daily PnL is additive and is the correct decomposition basis. Raw
ledger row counts and raw sums of `R_Multiple` are not valid OVS sample or
portfolio statistics after scale-out: the two tranches divide dollars/risk but
create two R observations. Standalone empirical-log checks must first collapse
OVS tranches back to the filled-signal level; correlated Kelly can use the
additive daily PnL stream directly.

There is one currently dormant parity caveat: the live/report path can apply a
2% ADDV participation cap to the dynamic overflow universe, while
`build_trade_ledger.py` itself does not. `OVERFLOW_UNIVERSE_ACTIVE` defaults
off, so this does not affect the present static-overflow ledger; it must be
stated if the environment is enabled for the research replay.

## Inherited results and negative-result discipline

### GRM replay

The saved GRM curve reports, for GRM 1.0/1.25/1.5/1.75, Sharpe
1.891/1.866/1.846/1.825 and max drawdown -8.94%/-10.78%/-12.60%/-14.44% of
$750k. Annual PnL/maxDD is approximately flat near 1.66. This supports the
interpretation of GRM as risk appetite rather than a sharply identified
growth optimum over that range.

But the saved CSV is an archival probe, not a current-book curve. It used the
now-removed pooled caps and the older engine/book vintage. At GRM 1.5, the
same cap study's per-strategy-only variant produced $3.947m total flat PnL,
versus $3.823m with the pooled layer, with identical -$94,471 maxDD and
-$44,244 worst day. Phase 3 must extend the method on a fresh scratch-only
current-book pass rather than append points to the old CSV.

Sources: `scratch/grm_replay_study.py`;
`scratch/grm_replay_results.csv`; `scratch/cap_impact_results.csv`.

### Cap study

On the archived vintage:

- Uncapped: $5.126m total PnL, Sortino 3.658, maxDD/worst day -$118,125
  (-15.75% NAV).
- Current-style per-strategy-only cap: $3.947m, Sortino 3.189, maxDD/worst day
  -$94,471 (-12.60% NAV).
- Adding the removed pooled layer reduced PnL another $124,593 and Sortino by
  0.095 while changing neither maxDD nor worst day.

Thus the current 250 bps cap cost about 23.0% of uncapped total PnL and 0.469
Sortino on that vintage; the brief's “25% and 0.56” describes the obsolete
two-layer production stack versus no caps. The retained cap is still
load-bearing because it alone bounds the worst day.

Source: `scratch/cap_impact_study.py` and
`scratch/cap_impact_results.csv`.

### Risk-dial and conditional-sizing graveyard

The following are inherited nulls and are out of scope for resurrection:

- book-wide fragility throttles, tapers, and dial-conditioned daily caps;
- the OVS fragility tilt (failed the point-in-time gate);
- high-fragility boosts above 1.0x and low-fragility boosts;
- sub-50 ramps, 5d confirmation, 21d “fast confirm,” hedge/VXX attachments,
  and trend-sleeve dial gates.

The only live/evidenced dial sizing is the static per-strategy fragility band
already embedded in the streams. Kelly recommendations remain static strategy
scalars and must not re-score or condition on the dial.

Sources: `scratch/ultracode_research/RISK_DIALS_2026-07-16.md`, especially
sections 3B and 4; `scratch/ultracode_research/PORTFOLIO_RESEARCH_2026-07-02.md`.

### Deliberate expectancy costs

The OLV recency ladder, OVS scale-out, Monday/MonFri gap derates, fragility
cuts, and 3x Bear same-day de-rate are appetite/variance controls that can cost
expected return. A Kelly optimizer may mechanically prefer their removal.
That is not a sizing recommendation: the study holds all overlays fixed and
only tests static base-bps/path scalars.

## Implications for phases 2-3

1. Use a scratch-only current-engine rebuild as the primary dataset and retain
   the 2026-07-21 ledger only as an archival cross-check. Do not run
   `build_trade_ledger.py::main`, because it overwrites shared data artifacts;
   call the engine in memory and write all research outputs under `scratch/`.
2. Treat the current book as 15 strategies, with three frozen pilots exactly
   as instructed. Monthly Weak Close will be represented only after the fresh
   replay and remains fixed regardless of its estimate.
3. Collapse scale-out tranches for empirical single-strategy log/Kelly checks;
   use additive daily MTM streams for correlation-aware allocation.
4. Report full-book allocation because it matches deployed exposure, but make
   liquid-only mean/covariance and “exclude overflow PnL” decisive
   sensitivities. No size change may be supported by overflow-only edge.
5. Use the established episode convention as the starting point: a new episode
   after more than five trading days without a signal (equivalent to the
   Leader Gap study's >7 calendar-day rule). For very dense OVS, also show the
   existing monthly-cluster convention and use the more conservative effective
   N for shrinkage.
6. Keep the per-strategy 250 cap fixed in every replay. The removed pooled caps
   stay off. Explicitly state whether an OVS P2 recommendation scales or fixes
   the P2 aggregate cap; the default research choice should keep the cap fixed
   as a separate appetite constraint.

## Decisions needed before phase 3 locks

Recommended defaults, subject to McKinley's confirmation:

1. **Drawdown constraint:** require the stationary-block bootstrap probability
   of a one-year max drawdown worse than 20% of the $750k NAV to be below 5%.
   Phase 2 can keep this symbolic, but phase 3 needs the numerical threshold
   and horizon.
2. **Data basis:** use the scratch-only current-engine rebuild as primary and
   the supplied 4,694-row July ledger as a frozen-vintage comparison, rather
   than treating the stale ledger as the authoritative current book.
3. **OVS P2 cap under a path-size proposal:** keep the effective 1.125%-of-NAV
   aggregate cap fixed while changing P2 per-signal bps. This cleanly separates
   relative edge sizing from the existing cluster appetite limit.

