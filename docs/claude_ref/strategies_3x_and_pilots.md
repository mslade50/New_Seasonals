# 3x fades, pilots and the trend sleeve

Moved close to verbatim from CLAUDE.md on 2026-09-23. CLAUDE.md keeps the live rules;
this file keeps the history, evidence and detail. Path fixes applied on the move.

## 3x Bear ETF Overbot Fade + Same-Day Signal De-rate (2026-07-07)

The 13 bear-equity 3x names (`strategy_config.LEV3X_BEAR_EQ`) were carved out
of the generic 3x ETF Overbot Fade (now 29 tickers) into a looser bear-only
fade: short thresholds 85->80, 21d consec 3->1, SAME 126/252d < 65 leader
exclusion — that filter is LOAD-BEARING (it keeps the fade from shorting
sustained bear markets; dropping it collapsed avgR +0.66 -> +0.28 with
-14R/-16R years in 2020/2022). Universes are disjoint by construction so the
two fades can never fire the same ticker on the same day. 25 bps nominal
(vs parent 40) because every signal in the loosened config lands 2020+
(one-regime sample). Fading an overbought inverse ETF = buying a market
selloff, so the strategy carries the FAMILY4 `frag_risk_bands` [[50,999,0.25]].
Evidence: scratch/lev3x_fade_class_study.py + lev3x_fade_bear_episodes.py.

Position stacking (2026-07-28): BOTH 3x Overbot Fades run `max_one_pos: False`
— consecutive-day re-fires open additional full-size legs in the same ticker
(observed depth <= 3). This ALIGNED THE MODEL TO LIVE: daily_scan /
order_staging never enforced one-pos (eq_order_entry's dup guard keys on
staged date), so live always stacked; the ledger was the side that under-
counted (first live stack: SQQQ 2026-07-24 + 07-27 legs). Backtest impact of
adopting stacking: generic fade +12.7R/23y (maxDD unchanged), bear fade
+10.2R with ~1.5x wider maxDD/worst-5d — accepted; marginal-leg edge is
episode-concentrated (bear: mostly Apr 2024). Only the per-strategy 250
bps/day cap bounds a stack (it sees same-day staged risk, NOT open legs).
The 3x Leader Gap Fade keeps `max_one_pos: True` (guarded by its test).
Evidence: scratch/lev3x_fade_stacking_study.py + lev3x_fade_stacking_results.csv.

Same-day signal de-rate — new generic sizing overlay, currently bear-fade
only: `execution['same_day_signal_derate'] = 0.10` sizes each of the day's
signals at `max(floor, 1 - 0.10*(n-1))` where n = that strategy's SIGNAL
count that day (ex-ante staged count, NOT fills — only ~1/3 of signals fill,
but high signal count itself marks the violent-selloff days where per-trade
edge degrades). `same_day_derate_floor` = 0.30. Composes multiplicatively
with frag bands (April 2024: 5 signals x high fragility -> 0.15x). Evidence:
scratch/lev3x_fade_bear_sizing_rule.py (same totR, worst 2-day window
-6.2R -> -4.5R).

Aligned sites -- change together (order_staging needs nothing: takes staged
sizes as-is):
- `strategy_config.py` execution `same_day_signal_derate` /
  `same_day_derate_floor` + shared formula `same_day_derate_mult()`
- `pages/strat_backtester.py` sizing step 3b4 (counts staged candidates
  per (strategy, day) in a pre-loop pass, post earnings-blackout)
- `daily_scan.py` post-pass 5c (after the cross-strategy overlap clamp;
  runs post-loop because n is only known after the strategy's ticker loop;
  counts per (Strategy_Name, Scan_Source); rescales Shares/Risk_Amt/Notional
  and stamps Sizing notes)
- Guard: `tests/test_same_day_derate.py` (carve-out partition, filter
  invariants, formula boundaries, single-carrier assertion)


## 3x Leader Gap Fade (pilot, 2026-07-10)

Capitulation fade on 3x ETFs whose UNDERLYING is spiking on fear. Universe =
`LEV3X_ALL` minus `LEV3X_BULL_EQ` (21 names: 13 bear-eq + TMF/TMV + 6 cmdty).
Filters: 2/5/10/21d rank > 80 (consec 1) AND 252d rank > 95 — the leader is
REQUIRED, the inverse of the other two 3x fades' <65 exclusion, so same-day
same-ticker cross-fire with them is impossible by construction (a ticker
cannot be <65 and >95 at once). Tape gate: T+1 open > close + 0.25 ATR,
resolved LIVE by order_staging's generic `T1_Open_Filters` gate (fail-closed;
scanner only stamps the JSON spec — it cannot see tomorrow's open). Entry:
Limit (Open + 0.75 ATR), OVS convention. 2-day time exit, NO STOP: stops
1.0-2.0 ATR (day-1 and day-2 armed) all destroyed the edge — adverse
excursion > 1 ATR is the normal path before the reversal (non-bull +23.7R ->
-39.8R at 1.0 ATR). The demanding entry IS the risk control (worst no-stop
trade -2.95R). Bull-eq exclusion is STRUCTURAL: every selectivity layer makes
bull-eq worse (strictest cell 0-for-7, avgR -1.28; losses span five bull
regimes 2018-2026) — do not re-add.

Sizing: 25 bps nominal (x GRM). Deliberately EXEMPT from frag_risk_bands and
same_day_signal_derate — the edge lives on exactly the high-fragility
multi-signal days those overlays would cut (Sept 2022, Apr 2025). Tail risk
is bounded instead by the per-strategy 250 bps daily cap (engine + live
aligned book-wide 2026-07-10 — see "Daily Risk Caps"; a 7-signal day at
37.5 eff = 262.5 bps trims ~5%). Validation
(2026-07-10): 31 trades / 15 episodes 2011-2025, avgR +0.80, PF 2.82,
episode-clustered t = 2.17, LOYO floor 1.55, drop-best-episode +9.4R @
t = 1.79, bootstrap P(<=0) = 2.1%. Pilot conviction — consider 40 bps only
after clean out-of-sample quarters.

Aligned sites — change together:
- `strategy_config.py` — the entry + `LEV3X_BULL_EQ` (source of truth)
- `pages/strat_backtester.py` / `daily_scan.py` — nothing bespoke; flows
  through generic paths (perf filters, T1_Open_Filters stamp, 0.75 ATR
  limit parse, max_one_pos, per-strategy daily cap)
- `order_staging.py` (OneDrive) — generic T1 gate enforces the gap at the
  IBKR T+1 open; per-strategy cap via the book-wide 250 bps default (a
  strategy-specific override existed for a few hours on 2026-07-10 and was
  removed when the book default aligned at 250)
- Guard: `tests/test_lev3x_leader_gap_fade.py`. Studies:
  `scratch/lev3x_fade_leader_*.py` (expansion, stops, entries, ovs_entry,
  class_split, bulleq_clusters, bulleq_strict, validation, capcheck,
  book_parity)

## Monthly Weak Close (pilot, 2026-07-31)

Monthly-scale dip-buy on SPY+QQQ: a month that CLOSES in the bottom 15% of
its own high-low range (signal fires only on the month's last trading day)
while the ticker is above its 200d SMA (~= the 10-month MA trend gate; the
gate is LOAD-BEARING — ungated, 2000-01/2022-style signals ride the next
bear leg, worst -17.9%). Entry: persistent limit at signal close - 0.25 ATR,
live T+1..T+2 (`fill_window_days: 2`); fills ~half the signals but captured
~90% of close-entry total PnL with better per-fill stats (the missed half
bounces immediately). 5d hold, 2 ATR target, NO stop (`stop_atr` 1.0 is the
sizing risk unit only). 30 bps nominal, FAMILY4 frag band [[50,999,0.25]] by
family analogy. ~1.1 signals/yr; SPY+QQQ same-month signals are
near-duplicates. Validation (gated cell, 2003+): close-entry N=30 avgR
+1.55%, clustered t=4.08; limit cell N=15, 15-for-15, avg +2.79%/fill;
h21 research variant LOYO floor t=3.76, cluster bootstrap P(<=0)=0.0000.
Longer holds carry more per-trade edge (h21 +3.0%/trade) but ~half the
in-market Sharpe (2.6 vs 4.7 filtered); the 5d/target form was chosen for
slot efficiency.

Implementation — all generic paths, nothing bespoke:
- `filters.py` `use_month_range_pos` / `month_range_pos_max` (month-end
  detection: next-row month roll, final row via US-bday calendar so the
  month-end PM scan AND the next-AM scan both grade it)
- `strategy_config.py` "Monthly Weak Close" (source of truth)
- daily_scan / strat_backtester: shared mask + generic persistent-limit,
  fill-window, frag-band, no-stop paths; order_staging needs nothing
- Guard: `tests/test_monthly_weak_close.py` (filter semantics + config
  invariants); engine parity: scratch/monthly_weak_close_engine_parity.py
  (30/30 signal dates, 15/15 fills, exit types identical vs research cell).
  Evidence: scratch/monthly_weak_close_mr*.py.

## Trend Sleeve (pilot, 2026-07-02)

`trend_sleeve.py` + `.github/workflows/trend_sleeve.yml`: monthly 12-ETF
trend-following ballast at 0.3x NAV (cut from 0.6x on 2026-07-17 to cap
overlap with the dial-gated SPY sleeve; combo = 12-1 momentum AND 10-month MA,
long/flat, inverse-vol slots capped 20%, cash otherwise). Universe = SPY QQQ
IWM EFA EEM FXI VNQ GLD SLV DBC TLT LQD — NO USO (roll decay) and NO UUP
(capital inefficiency: 20% slot for +0.00%/mo contribution + K-1; costs 2022
+0.5% -> -1.9%, accepted). Sector-ETF / intl-single expansion tested and
REJECTED (equity slots crowd out diversifiers, 2008/2022 flip negative);
exhaustion scale-down overlay REJECTED (Sharpe flat). Signals on the month's
last trading-day close, staged MOO (TIF=OPG) to the `Trend` Sheets tab for
next-session execution; held-share state in `trend_sleeve_state.json` (R2 —
the month-end run computes DELTAS against it; if staged orders were never
executed, clear the state or the next rebalance is wrong). The workflow runs
weekdays 21:35 UTC (AFTER update_master_prices' 21:10 PM cron — the script
hard-fails if today's close is missing) and no-ops except on the last trading
day; `Execute_On` (next ET trading day after the run) gates submission.
FULLY AUTOMATED end-to-end: order_staging.py (`load_trend_rows`) reads the
tab on Execute_On morning and emits naked-MOO rows (appended AFTER risk caps,
excluded from PA/execution_2); eq_order_entry.py places them as MKT/OPG
parent-only (Exit_Condition_Time='NONE' -> no exit legs — positions unwind
via future rebalance SELL rows). Ballast ONLY — it loses ~-0.4%/mo in
high-fragility months (frag_risk_bands handles that hole). Scale to 1.0x of
the fraction only after 2 clean quarters. Studies: scratch/tf_universe_study.py,
scratch/ultracode_research/trend-following.md + trend_prework_gates.md.
