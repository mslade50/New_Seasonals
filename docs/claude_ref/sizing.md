# Sizing: GRM, tilt, caps, ladder, overlap clamp, frag bands, P/C fear bands, gap derate

Moved close to verbatim from CLAUDE.md on 2026-09-23. CLAUDE.md keeps the live rules;
this file keeps the history, evidence and detail. Path fixes applied on the move.

## Sizing Conventions — GLOBAL_RISK_MULTIPLIER + overlays (2026-05-27)

`strategy_config.GLOBAL_RISK_MULTIPLIER` (currently **1.5**) scales the whole
book at import time: every execution `risk_bps`, OVS `path1_bps` / `path2_bps`
/ `path2_daily_cap_pct`, every `earnings_size_override.risk_bps`, and the
`OVERFLOW_RISK_OVERRIDES` in daily_scan / daily_portfolio_report. The dicts in
strategy_config SOURCE are nominal; everything downstream (scan, engines,
reports, staged `Risk_Amt`/`Risk_Bps`) sees SCALED values. **All bps in this
doc are nominal unless marked effective** — e.g. OLV liquid 35 nominal = 52.5
effective, overflow 25 = 37.5. (OLV additionally sizes by signal recency —
first iteration in a trailing 21td window 0.5x, second 0.7x — see "Ladder
Sizing"; a flat 35→18 cut shipped and was replaced by a ladder form the same
day, 2026-07-29, re-based to signal recency 2026-07-30.)

GRM evidence trail (2026-07-16, scratch/grm_replay_study.py — the constant
shipped 2026-05-27 with none): full-ledger replay at GRM 1.0/1.25/1.5/1.75
with caps FIXED at prod (250 per-strat, 500L/250S pooled, flat $750k).
Risk-adjusted metrics are nearly scale-invariant — Sharpe 1.89/1.87/1.85/1.83,
annPnL/maxDD ~1.66 flat, maxDD -8.9%/-10.8%/-12.6%/-14.4% NAV scaling
slightly sub-linearly (fixed caps clip the tail). No cliff anywhere in the
range: the setting is a clean risk-appetite dial, and 1.5 (~$157k/yr ann
flat PnL at -12.6% worst DD) is defensible. Results:
scratch/grm_replay_results.csv.

**Base-bps tilt (D3.2, shipped 2026-09-04, `strategy_config.STRATEGY_BASE_TILT`)**:
applied ONCE at import, after the GRM block, to `execution['risk_bps']` only
and folded into `OVERFLOW_RISK_OVERRIDES`; never through the engine's
`risk_multipliers` (that path scales the per-strategy cap), never on the
earnings override, never on OVS `path1/path2_bps`. Table: 52wh Breakout
0.70, Weak Close Decent Sznls 0.75, Sector BO 0.87, ATR Extended Gap Up
1.10, 3x ETF Overbot Fade 1.27, SPY QQQ MonFri Reversion 1.30; every other
strategy 1.0 (the dict is total over the book; a missing name defaults to
1.0 with a loud `[TILT]` line; `tests/test_base_bps_tilt.py`). A Sharpe and
drawdown lever, PnL-neutral: full-history replay 2010+ Sharpe 2.20 -> 2.30,
maxDD -12.4% -> -11.2%, worst-21d -9.5% -> -7.2%, annual PnL -2.7%. Same
day the Weak Close Decent Sznls seasonal-rank size tiers (1.5x >= 65,
0.66x 33-50) were RETIRED on both sides (D3.1; inverted against the edge,
`tests/test_wcds_size_tiers.py`). Evidence: sizing_due_diligence_2026-09-02
and `artifacts/verify_2026-09-04/sizing_d31_d32/`. Fortnight authority:
`docs/plan_2026-09-04.md`.

daily_scan per-signal sizing order (mirrored in strat_backtester step 3b):
base bps (tier x GRM x tilt) -> 2b fragility band -> 2c signal-recency ladder rung
(carrier: OLV {window_td: 21, mults: [0.5, 0.7, 1.0]} since 2026-07-30; the
old open-position-count ladder machinery survives dormant, carrier-less) ->
2c2 cycle-year mult -> 2d earnings size override (REPLACES the base but
COMPOSES with the 2c recency mult since 2026-07-30 and with the 2b frag
band mult since 2026-08-24 — cycle/tier are still clobbered; itself
GRM-scaled; two carriers: OLV -10..0 TD -> 10 bps nominal / 15 effective;
St OS Sznl -5..-1 TD -> 6 bps nominal / 9 effective, added 2026-07-30 —
the no-stop 5d hold straddling an imminent print held every ledger tail
loser [-5..-1 cell N=9 avgR -0.50 vs +0.32 outside]; small-N appetite
haircut, guard tests/test_earnings_size_override.py, evidence
scratch/stos_earnings_proximity.py) -> shares -> ADV participation cap -> per-ticker notional cap
(OLV, 2026-07-20) -> 5c same-day signal de-rate (post-pass; 3x Bear fade —
see its section below).

## Daily Risk Caps (aligned 2026-07-10)

Two stacked pro-rata caps on STAGED (pre-fill) risk, applied after all
per-signal sizing. All values are EFFECTIVE (not GRM-scaled):
- **Per-strategy: 250 bps/day.** Each strategy's staged risk per signal date
  independently capped; rows scale by cap/total. Live:
  `order_staging.PER_STRAT_DAILY_CAP_BPS` (was 200 from inception —
  raised to 250 on 2026-07-10 to close a silent live-vs-ledger divergence:
  the engine always modeled 250, so live trimmed 200-250 bps days up to
  20% while the ledger booked full size; the drift came from the engine
  default being anchored to the POOLED 2.5% cap, not the per-strategy
  one). Engine: `process_signals_fast(cap_bps=...)`, default 250.
  `PER_STRAT_DAILY_CAP_DOLLARS` in order_staging holds per-strategy dollar
  overrides (currently empty).
- **Pooled per-direction caps: REMOVED 2026-07-16** (in place 2026-07-10 to
  2026-07-16 at long 500 / short 250 bps). The cap-impact study
  (`scratch/cap_impact_study.py` + `cap_impact_results.csv`) showed the
  pooled layer bound on the SAME net-positive cluster days as the
  per-strategy cap and cost ~$125k/23y with IDENTICAL maxDD and worst day —
  pure redundancy. Removed together: `order_staging` pooled stage (staged
  side totals still printed), `build_trade_ledger` POOLED_*_CAP_BPS = None,
  `daily_portfolio_report` call site, strat_backtester UI defaults (0 = off).
  The engine's `max_long_risk_bps`/`max_short_risk_bps` machinery is
  retained for counterfactuals (sequential-after-per-strategy semantics,
  fixed 2026-07-16; guard: `tests/test_pooled_cap_sequential.py`).
  Context: caps overall cost 25% of total return and 0.56 Sortino over 23y;
  the per-strategy 250 is kept because it alone bounds the worst single day
  (-$44k vs -$118k = -15.75% NAV uncapped, which by itself was the entire
  uncapped maxDD).

Aligned sites — change together: `order_staging.py` (OneDrive) constants,
`scripts/build_trade_ledger.py` POOLED_*_CAP_BPS, `daily_portfolio_report.py`
call site, `pages/strat_backtester.py` UI defaults + `cap_bps` fallback (250)
in `process_signals_fast`.

**Do NOT fragility-condition these caps** (negative result, codified
2026-07-16): dial-scaled pooled or per-strategy caps are the failed
book-wide throttle re-skinned — rest-of-book at dial >=50 shows no
significant degradation (p=.47 clustered), the aggregate PIT t was -0.23,
and the taper variant cost -11.4R — on the costliest possible surface (four
aligned sites incl. one out-of-repo, scalar-to-series engine change). The
book's only evidenced dial-sizing hook is per-strategy `frag_risk_bands`.
Evidence: scratch/ultracode_research/RISK_DIALS_2026-07-16.md.

## Ladder Sizing (OLV signal-recency form, 2026-07-30)

`execution['signal_recency_ladder']` has ONE carrier: OLV at
**{window_td: 21, mults: [0.5, 0.7, 1.0]}** — the rung is the count of that
ticker's OLV SIGNAL days (shared filter mask, fill-independent) in the
trailing 21 sessions before the signal day: 0 prior -> 0.5x, 1 prior ->
0.7x, 2+ -> full. The earnings size override COMPOSES with this mult (it
replaces the BASE bps only — a first-iteration pre-earnings signal is
10 x 0.5 bps nominal); every other overlay is still clobbered by the
override. A deliberate risk-appetite footprint trim aimed at OLV's weakest
legs (leg-1 avgR +0.56-0.82 across dial bands vs +1.1-1.4 for leg-3+;
OLV's open notional had doubled vs the 2018-2020 norm and carried 49% of
2026's intraday trough dollars), NOT a PnL-positive rule — it costs
expectancy by design.

It replaced the ONE-DAY-OLD open-position-count ladder [0.5, 1, 1]
(2026-07-29, itself the replacement for a same-day flat 35->18 cut). Why
the re-base: the open-count form reset to 0.5x whenever a chain had fully
exited (even a day later), was blind to still-unfilled working limits (a
day-2 signal before day-1's limit filled ALSO got 0.5x), and jumped
straight to full size on the second leg. Signal-recency counting fixes all
three and grades the second iteration at 0.7x.

Implementation — aligned sites (change together):
- `strategy_config.py` OLV execution `signal_recency_ladder` (source of
  truth; mults NOT GRM-scaled)
- `daily_scan.py` sizing step 2c: recomputes the fired ticker's mask
  (`filters.live_signal_mask`) and counts the trailing window
  (`filters.recency_prior_from_mask` — last bar excluded); mult carried
  into step 2d's earnings override
- `pages/strat_backtester.py`: candidate-recency pre-pass counts prior
  candidate df-positions per (strategy, ticker) from the RAW candidate list
  (candidates ARE mask days, so engine == scan; known bound: pre-cutoff
  signals invisible for the first window of a run) + `_recency_mult` in the
  earnings override
- order_staging needs nothing (takes scanner-staged sizes as-is)
- Guard: `tests/test_olv_stop_and_cap.py` (config invariants, consecutive
  grading, fill-independence, window expiry, override composition)

History: the ORIGINAL ladder (OLV-only 2026-04-22 to 2026-07-20, removed in
the stop/gate package) was the OPPOSITE bet — a mild 0.85 first-rung
discount graded UP — and flat 1.0x beat it ($654k vs $605k [0.85,1,1] vs
$627k [0.85,1,1.15] / 21y; evidence scratch/olv_package_sim.py). Do not
confuse the two: today's recency ladder is an appetite cut that accepts
that drag. The old open-position-count machinery
(`execution['ladder_multipliers']`, `daily_scan.load_open_position_counts`,
the engine's open-count rung) survives dormant with NO carriers.

## Cross-Strategy Overlap Clamp (2026-05-12)

`strategy_config.CROSS_STRATEGY_OVERLAP_OVERRIDES`: when the named strategies
fire on the SAME signal date and SAME tradeable ticker (compared after
`SPOT_TO_TRADEABLE` aliasing, ^GSPC->SPY ^NDX->QQQ), each side's risk is
clamped to `risk_bps_when_overlapping`. Currently one pair: Indices Oversold
Bounce + SPY QQQ MonFri Reversion -> 20 bps nominal each (GRM-scaled at
import since 2026-08-12 = 30 effective; it was unscaled before, making the
clamp 20 EFFECTIVE against documented intent). It is an ABSOLUTE clamp on
the row's staged Risk_Amt (a row already below the clamp is untouched), and
it keys on STAGED signals — both sides firing — regardless of which limits
later fill. Applied in daily_scan step 5b and replayed in
`pages/strat_backtester.py` sizing step 3b3c from a candidate pre-pass
(2026-08-12; the old post-pass clamped only FILLED pairs and ran after the
per-strategy cap, booking the one-fills leg at full size vs live's clamp).


## Fragility Risk Bands (2026-07-02)

Per-strategy fragility sizing via `execution['frag_risk_bands']` =
`[[lo, hi, mult], ...]` on the 10d-MA 63d risk-dial score as of signal date
(first match wins, `lo <= score < hi`; missing/stale score or no bands = 1.0x).
REPLACED the retired book-wide ramp (1.25x boost -> 0.10x floor): the boost had
no edge case, and only specific pockets degrade at high fragility. Current
bands: the dip-buy FAMILY4 (Weak Close Decent Sznls, SPY QQQ MonFri Reversion,
Monday Dip, Indices Oversold Bounce) run `[[50, 999, 0.25]]`, as do the
family-analogy carriers 3x Bear ETF Overbot Fade (2026-07-07) and Monthly Weak
Close (2026-07-31); the rest of the book (including OVS, and OLV again since
2026-08-25 — see the retired-band note below) is 1.0x at all scores. Unlike the old ramp, the ENGINE
REPLAYS the bands point-in-time, so ledger and live agree (finding #26 closed
for this scheme). Evidence: scratch/ultracode_research/PORTFOLIO_RESEARCH_2026-07-02.md.

**OLV band — RETIRED 2026-08-25 after one session; OLV is bandless again.**
McKinley chose to hedge the Aug-2026 high-dial episode manually as a one-off
(short index) rather than carry standing sizing machinery, and the EOD book
cap below was disabled at the same time. The paragraph is kept as
institutional memory; nothing in it is live. A SPY-hedge study run the same
day (scratchpad, not committed) found: OLV's daily PnL is ~8% SPY-explained
overall (beta 0.76x notional), ~32% at dial >= 65 (beta 1.38x, n=67); its
worst drawdowns (Jul-2026 -$59k/21d, Jan-2025, Jul-2024, Jul-2021) hit with
SPY flat-to-up at dial 13-51 — idiosyncratic clusters a SPY hedge cannot
touch; every dial-gated SPY-short variant lost money on drift (OLV only
trades above the 200-SMA); and the rule "dial >= 65 AND OLV > 50% NAV" had
ZERO historical days — Aug-2026 is the first. Original record follows.

**OLV 0.5x at dial >= 65 (2026-08-24) was an APPETITE decision, not an
evidenced edge — do not cite it as one.** McKinley: "I know the data doesn't
support that but I don't care." The data, for the record (ledger 2016+, 299
OLV trades with a dial reading): edge dilutes with the dial (avgR +0.98 <30,
+0.68 30-50, +0.55 50-70, +0.60 70-85, -0.06 >=85 on N=7) but there is NO
damage signal — 21 trades at >=70, zero stop-outs, nothing below -1.4R;
OLV's worst drawdowns (July 2026 oil cluster -$35.8k, 14 legs open) hit at
dial ~20. What the cut bounds is structural: OLV has no resting stop, no
aggregate concurrent-exposure cap, and on 2026-08-24 carried ~$628k
(~84% of the $750k base) across 10 names with the dial at 89.5 — a level
seen on 20 days in the whole series (Dec 2021 and Aug 2026). Halving above
50 would have cost ~19R / ~$29k flat over 10y. Threshold 65 is McKinley's
(it coincides with the St OS Sznl dial_filter, nothing was scanned).
Composition (the 2026-07-16 prereg's gate 5, decided here): the band
COMPOSES with the earnings size override in both scan 2d and engine 3b3b —
pre-earnings at dial >= 65 = 10 x recency x 0.5 bps; cycle/tier mults are
still clobbered by the override. P/C fear override (same day, McKinley):
`pc_fear_bands = OLV_PC_FEAR_BANDS` = {on: [[65,999,1.0]], off:
[[65,999,0.5]]} — the cut is LIFTED to full size when the lag-1 P/C fear
state is ON, the family's "washed-out positioning" logic; fear OFF or
stale/missing P/C keeps the cut (plain table == 'off', so fail-closed = cut).
No 1.25x boost, no zeroing. The prereg's PIT gates become the post-ship review — they
can retire the band, they are not its justification. The engine replays it
point-in-time, so the next ledger rebuild halves the 25 historical OLV
trades at dial >= 65 (14 in 2021, 11 in 2026; ~-5.6R / -$8.7k flat —
ledger/live parity preserved). Guards:
`tests/test_frag_risk_bands.py`, `tests/test_pc_fear_bands.py` (OLV must
stay OUT of the P/C family).

PIT edge-weight gate (roadmap step 5, run 2026-07-03, scratch/pit_reestimate.py
+ pit_extract_signals.py): the fragility composite's signal weights were
re-estimated on expanding windows (vintage Y-1 weights score year Y, 2018+)
to remove calibration lookahead. Results: PIT-vs-current series corr 0.94,
>=50 day agreement 92%. FAMILY4 throttle SURVIVED (hi -0.10 vs lo +0.63,
clustered t=-1.96 p=0.057; LOYO floor ~1.4-1.5 sigma; negative in 6 of 9
years) — stands, conviction one notch lower than the current-weights grading.
The OVS [21,44) 0.75x tilt FAILED (PIT t=-1.34; even current weights only
t=-0.63 on 2018+ — its z=-3.0 lived in untestable 2016-17) and was REMOVED
per the pre-agreed gate; OVS is fully exempt again. The aggregate book-wide
>=50 effect also fails PIT (t=-0.23), vindicating the family-only design.
Residual lookahead the PIT gate cannot cure: signal definitions/parameters
are today's code. Re-examine FAMILY4 at +20 high-frag family trades (~2029).

Three aligned sites -- change together (order_staging needs nothing: it takes
scanner-staged sizes as-is since 2026-06-11):
- `strategy_config.py` execution `frag_risk_bands` (source of truth)
- `pages/strat_backtester.py` sizing step 3b3 (`frag_band_mult_at`, reads
  data/rd2_fragility.parquet point-in-time; pre-2016 signals -> 1.0x)
- `daily_scan.py` sizing step 2b (`frag_band_mult`, today's score, stamps
  Sizing notes; scan-summary email shows active band tilts)
- Guard: `tests/test_frag_risk_bands.py` (config invariants + boundary
  behavior + engine/live parity). Replay parity vs the research cells:
  scratch/parity_check_frag_bands.py (FAMILY4 74@0.25x exact, OVS 226/230
  @0.75x exact with 4 cap-interaction deviations of 0.0004 on one day).

## P/C Fear-Conditioned Family Bands (2026-08-05)

The 6 `frag_risk_bands` carriers (FAMILY4 + 3x Bear Fade + Monthly Weak
Close) select their band TABLE by the equity put/call FEAR STATE — trailing
252d percentile of the 10d-MA CBOE equity P/C (`data/cboe_putcall.parquet`,
2006-11+, `update_cboe_putcall.yml` with `--assert-fresh-bd 2` fail-loud),
**lag-1 by construction** (the state for signal date D uses the newest row
dated <= D-1 bday; measured: the 21:30 UTC scrape only ever has D-1).
The job runs TWICE a day since 2026-08-06 — a 4:10 AM ET local dispatch that
collects the prior session once CBOE publishes overnight, plus the original
21:30 UTC backstop. **Live band selection is byte-unchanged by the AM run**
(pc_fear selects by DATA date, and the AM scan already got exactly the row
lag-1 wants); what it fixes is later consumers, above all the 7 AM Daily
Pitch, which used to see a reading two business days behind the session it
was trading. `strategy_config.PC_FEAR_BANDS` (NOT GRM-scaled):
- fear ON (pctile > 85):  `[[0,50,1.25],[50,999,1.0]]` — boost in calm tape,
  FULL size in the dial>=50 zone (washed-out positioning = capitulation)
- fear OFF: `[[0,50,1.0],[50,999,0.0]]` — dial>=50 without washout is
  ZEROED (staged at 0 shares, visible in email/tabs, never ordered)
- P/C stale (> 3 bd, `pc_fear.STALE_BD`) / missing / pre-2007-11: fail
  CLOSED to the strategy's plain `frag_risk_bands` (incumbent 0.25x book)
Stale-DIAL semantics unchanged (fail-open 1.0x, book-wide convention).

SHIPPED AHEAD OF THE PREREG GATES as an explicit McKinley appetite decision
(2026-08-05) — the evidence base is 19 fear-ON hi-frag trades / 3 episodes;
the prereg's gates (PIT re-bucket, 2 new OOS episodes, LOYO) now run as the
POST-SHIP REVIEW: scratch/ultracode_research/
family_pc_fear_band_prereg_2026-08-05.md. Multiplier set is CLOSED:
{1.25, 1.0, 0.25, 0.0}. POST-SHIP REVIEW PART 1 RAN 2026-09-04
(scratch/ultracode_research/pcfear_review_2026-09-04/): every runnable gate
passed on both dial vintages (1a -2.82 / -2.22 sigma; 1b +0.79R on n=21;
leg B within 0.1R; LOYO min remainder +0.42), so all three legs STAND;
gate 2 (two new fear-ON hi-frag episodes) has accrued nothing because fear
has been OFF since 2026-08-04. The Aug-2026 zeroed set (6 signals, +2.21R,
+$8.5k at 1.0x) is one episode and does not move a multiplier.
**Leg-C shadow tracking is mandatory**:
`build_trade_ledger.build_pcfear_shadow` re-runs the engine with
`pc_fear_enabled=False` -> `data/backtest_trades_pcfear_shadow.parquet` so
the zeroed cell keeps accruing evidence for the "+20 hi-frag family trades"
re-exam.

Aligned sites — change together (order_staging needs nothing: takes staged
sizes as-is; a 0-share row stages as a non-order):
- `pc_fear.py` — state definition + table selection + note formatting
  (source of the math); `strategy_config.PC_FEAR_BANDS` (source of truth)
- `daily_scan.py` — 3b2 state load + console line, sizing 2b table
  selection + per-signal Sizing notes ("P/C 71%ile (fear OFF) + dial 55 ->
  0.00x — ZEROED"), P/C liveness footnote in EVERY scan email
- `pages/strat_backtester.py` — 3b3 `frag_band_mult_at` PIT lag-1 replay +
  `process_signals_fast(pc_fear_enabled=)`
- `scripts/build_trade_ledger.py` — pcfear shadow pass (best effort)
- Guards: `tests/test_pc_fear_bands.py`, `tests/test_frag_risk_bands.py`
  (state-matched parity), `tests/test_cboe_putcall.py` (feed freshness +
  the row guard below)
- KNOWN GAP: the site risk tab's sizing_state block still serializes the
  incumbent `frag_risk_bands` only (stale-state view) — fear-conditioned
  display not yet built.

**Feed row guard (2026-09-18)**: `cboe_putcall.py` validated FRESHNESS only,
so a bad row could size the book silently. It now also validates each row,
both on scrape and as a purge over the cache on load, so a polluted local or
R2 copy self-heals: the date must be an NYSE session (weekend, NYSE holiday
set built in-module because Columbus Day and Veterans Day are federal
holidays the NYSE trades through, Good Friday, plus an explicit
`NYSE_SPECIAL_CLOSURES` list) and `equity` must land in [0.15, 3.0], a band
set outside the measured extremes of 0.32 and 2.40 so a real capitulation
print still passes. Rejections print one loud line and never raise. Purged
on ship: 2025-01-09 equity 0.00, a Carter day of mourning when the NYSE was
closed and CBOE served a page anyway (5000 rows to 4999, nothing else
dropped, R2 republished). LEFT AS-IS by McKinley's call: 2024-01-10's 1.55
equity print, a one-day spike that reverted but is in band and on a real
session.


## Large-Gap-Up Size Derate (2026-07-21)

Per-strategy HALF-SIZE on a gap-up open, carried by the two liquid dip-buys
Monday Dip and SPY QQQ MonFri Reversion via
`execution['gap_size_derate'] = {threshold_atr: 0.25, mult: 0.5, dir: 'up'}`.
When the T+1 session open gaps more than `threshold_atr * ATR` ABOVE the signal
close, the dip-buy edge roughly halves (avgR ~0.45 -> ~0.23 in the ledger; the
bounce partly plays out at the open and the `Limit(Open-0.25ATR)` entry fills at
a worse price), so the trade is sized at `mult`. Deliberate risk-appetite
haircut, NOT a PnL win — the gap-up bucket is still net positive; the ledger
cost is ~-$16k (Monday Dip) / -$40k (SPY QQQ MonFri) flat over 23y, cutting
gap-up trades' Size_Mult ~0.97 -> ~0.48 while leaving all non-gap trades
byte-identical.

Key facts:
- **Sizing overlay, not a filter.** Composes multiplicatively with 3b3 frag
  bands (a high-fragility gap-up day = 0.25 x 0.5 = 0.125x). Distinct from SPY
  QQQ MonFri's `use_t1_gap_kill` (settings), a Friday-ONLY full DROP at 0.5 ATR
  enforced in `filters.get_historical_mask`. The kill runs first (removes the
  candidate); this derate then half-sizes whatever it leaves that still gaps
  > 0.25 ATR (non-Friday signals + Friday 0.25-0.5 ATR gaps). Both stay
  configured — neither replaces the other.
- **Only knowable at the open**, so live it is STAMPED by daily_scan and
  APPLIED by order_staging at the IBKR T+1 open (like MonGapKill / OVS 2-path);
  the scan itself never applies it. Fails OPEN (full size) on a missing
  open/ATR/signal-close — a haircut isn't worth dropping a valid fill.

Aligned sites — change together:
- `strategy_config.py` — execution `gap_size_derate` on both strats (source of
  truth; NOT GRM-scaled, it's a pure multiplier).
- `pages/strat_backtester.py` — `gap_derate_mult()` helper + sizing step 3b5
  (engine sees `entry_row['Open']` directly, so ledger == live). Drives the
  ledger + `daily_portfolio_report.py`.
- `daily_scan.py` — stamps `GapDerate_ATR` / `GapDerate_Mult` / `GapDerate_Dir`
  on every staging row (empty for strats without the field).
- `order_staging.py` (OneDrive) — reads the stamps, halves `Quantity` +
  sets `_GapMult` (so the daily caps see the reduced risk) + labels the row
  `DERATE_GAP`; gated on `path_label == ''` so a killed/gated/OVS row is never
  touched. Enforced at the live open right after the MonGap block.
- Guard: `tests/test_gap_size_derate.py` (config carriers, helper boundaries
  up/down, fail-open, frag-band composition, kill-coexistence).

## Sizing step sequence (verified against the code 2026-09-23)

The per-signal sizing order. Checked against the step comments in `daily_scan.py` and
`pages/strat_backtester.py` on 2026-09-23.

**Live scan (`daily_scan.py`), in order:**
1. Base bps = tier (liquid or `OVERFLOW_RISK_OVERRIDES`) x GRM x base-bps tilt, all applied at import.
2. Step 2b: fragility risk band. The table is picked by the P/C fear state that step 3b2 loads before the loop (`pc_fear_bands` carriers); everyone else uses plain `frag_risk_bands`.
3. Step 2c: signal-recency ladder rung (OLV). The dormant open-position-count ladder ("2c-old") has no carriers.
4. Step 2c2: cycle-year multiplier (OVS midterm 0.75x).
5. Step 2d: earnings size override. It REPLACES the base bps and composes with the 2c recency mult and the 2b band mult. Cycle and tier mults are clobbered.
6. Shares, then the ADV participation cap (`ADV_PARTICIPATION_CAP`, overflow), then the per-ticker notional cap (`ticker_notional_cap`, OLV).
7. After the strategy loop, step 5b: cross-strategy overlap clamp.
8. Step 5c: same-day signal de-rate (3x Bear fade).

Applied later by order_staging at the IBKR T+1 open, because they need the open: the gap-size derate (from the `GapDerate_*` stamps), the OVS 2-path multiplier, and `T1_Open_Filters`. Then the per-strategy 250 bps/day cap on staged risk.

**Engine (`pages/strat_backtester.py`, `process_signals_fast`), in order:** base risk with the
recency ladder mult (step 3b), 3b2 cycle-year, 3b3 fragility band (PIT, lag-1 P/C table), 3b3b
earnings override (composes with recency and the band, clobbers cycle), 3b3c overlap clamp
(mirrors scan 5b), 3b4 same-day de-rate (mirrors scan 5c), 3b5 gap-size derate (the engine sees
the entry-day open directly), then 3c, the user-configured per-strategy risk multiplier from the
backtester UI. The per-strategy daily cap is `cap_bps` (default 250).

**What the check found.** The documented scan order is correct. The engine applies the cycle
mult (3b2) before the fragility band (3b3), while the scan applies the band (2b) first. Both are
plain multipliers and both sides run the earnings override last, so the result is the same. The
engine also has step 3c, a UI-only multiplier with no live counterpart. The engine replays the
OLV notional cap inside the fill loop rather than as a numbered sizing step. No reference to
`adv_share_cap` or the ADV participation cap turned up in `pages/strat_backtester.py`, so that
cap looks live-only (scan-side, overflow tier). Confirm before relying on engine/live parity for
thin overflow names.
