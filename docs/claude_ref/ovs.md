# OVS (Overbot Vol Spike)

Moved close to verbatim from CLAUDE.md on 2026-09-23. CLAUDE.md keeps the live rules;
this file keeps the history, evidence and detail. Path fixes applied on the move.

## OVS Strategy — Earnings Blackout + 2-Path Sizing + Friday-only EOD-DD

Overbot Vol Spike has special-cased execution as of the 2026-04-30 merge.

### Earnings blackout (±10 trading days)
The OVS execution dict in `strategy_config.py` carries `earnings_blackout_td: 10`. Signals within ±10 trading days of an earnings announcement are dropped. Tickers with no earnings data in `data/earnings_calendar.parquet` (commodity ETFs, indices, futures, FX) **pass through** — NaN-as-True, mirroring the `Not Between` behavior in `pages/backtester.py`.

Implementation:
- `earnings_filter.py` — shared module with `load_earnings_dates_map()`, `signed_offset()`, `in_blackout(window=10)`. Loads `data/earnings_calendar.parquet`.
- `daily_scan.py` — applies the filter inline during the strategy loop (drops the signal before the dict is built).
- `pages/strat_backtester.py` — pre-pass that drops candidates from the chronological loop entirely, so the daily portfolio report's PnL reflects what live would do.

### Two-path execution (replaces the prior 30/20 bps + 1.3× ATR-sznl-5d sizer)
The OVS execution dict carries (nominal; ×GRM 1.5 at import → 60 / 12 / 1.125%):
- `path1_bps: 40` — full size on a decisive open gap
- `path2_bps: 8` — reduced size on a mild gap
- `path2_daily_cap_pct: 0.75` — 0.75% of ACCOUNT_VALUE aggregate cap on path-2 risk (was 1.0 pre-2026-05-01)

Decision happens in `order_staging.py` (in `C:\Users\McKinley Slade\OneDrive\trading_ibkr\`) using IBKR's T+1 session open vs the signal's close + 0.25 ATR threshold. Same scheme for liquid AND overflow universes.

| T+1 open vs close | Path | Per-trade size |
|---|---|---|
| Open > Close + 0.25 ATR | **Path 1: Decisive** | 40 bps nominal / 60 effective (full) |
| Close < Open ≤ Close + 0.25 ATR | **Path 2: Mild** | 8 bps nominal / 12 effective, capped at the 0.75% nominal path-2 aggregate (pro-rata scale-down across all path-2 rows that day) |
| Open ≤ Close | **Skip** | 0 |

Scanner-side stamps `Path1_Bps`, `Path2_Bps`, `Path2_Daily_Cap_Pct` columns on every OVS staging row so order_staging can compute the multiplier without importing strategy_config.

### Scale-out (live 2026-06-17, engine-modeled 2026-07-16)
Every OVS P1/P2 primary-account row is split by order_staging into two
independent single-target brackets: **near = 40% of shares @ 1 ATR, far =
60% @ 2 ATR** (a tranche that rounds below 1 share = no split, single
full-size 2-ATR bracket; PA is never split). Deliberate short-book VARIANCE
SMOOTHING, not PnL-maximizing — the 2026-07-01 audit measured scale-outs as
-R vs full-size 2 ATR and McKinley accepted that trade-off explicitly
(2026-07-16). The engine books two tranche rows per fill (`Tranche` column:
near/far/'' ) with the same share split; EOD-DD days book as one row (both
live tranches exit at the same close); entry-day targets stay uncredited on
the near tranche (book convention). Aligned sites — change together:
- `strategy_config.py` OVS execution `scaleout_near_frac` /
  `scaleout_near_tgt_atr` (source of truth; NOT GRM-scaled)
- `order_staging.py` (OneDrive) `OVS_SCALEOUT_NEAR_FRAC` /
  `OVS_PROFIT_TAKER_ATR_MULT` + `_split_scaleout_for_primary`
- `pages/strat_backtester.py` tranche booking in `process_signals_fast`
- Guard: `tests/test_ovs_scaleout.py`

### Same-symbol precedence + P1-budget gate history (2026-07-16)
**ATR Extended Gap Up > OVS**: when both fire on the same symbol and the ATR
row passed its T+1 open gate, the OVS row is dropped (both short the same
blow-off; never double the slot). Live in order_staging since before 2026-07;
modeled in the engine pre-pass since 2026-07-16 (engine keys on ATR-Ext
candidates, whose mask already includes the T+1 gate — matching live's
Quantity > 0 condition). Guard: `tests/test_ovs_scaleout.py`.
**P1-budget gate REMOVED**: the engine-only rule "kill all P2 when the day's
P1 risk exceeds 60% of the per-strategy cap" fired on ~170 historical ledger
days but never existed live. Removed 2026-07-16 (decision: match live). The
P2 aggregate daily cap is live and stays.

### Entry-day drawdown stop (EOD-DD, Friday entries only)
The OVS execution dict carries `eod_dd_atr: 0.25` and `eod_dd_weekdays: [4]`. If a Friday-entered OVS trade is more than 0.25 ATR offside vs the entry-day fill by 15:58 ET, exit at the entry-day close. Mon-Thu entries skip the check entirely — those positions get the full hold window instead. Weekday list uses Python conventions (Mon=0..Fri=4); empty/missing = all weekdays.

Aligned across four systems — change `eod_dd_weekdays` in one place and they all move together:
- `strategy_config.py` — execution dict (single source of truth)
- `pages/strat_backtester.py` — reads `execution['eod_dd_weekdays']`, gates the EOD-DD block on `df.index[entry_idx].weekday() in [...]`. Drives both the backtester page and `daily_portfolio_report.py`.
- `pages/backtester.py` — UI multiselect lets you override per-run for exploration (separate from the prod-locked rule above).
- `order_staging.py` (in `C:\Users\McKinley Slade\OneDrive\trading_ibkr\`) — hardcoded `weekday() == 4` gate on the STP-with-goodAfterTime=15:58 leg. Update both sides if you change the rule.
- Regression coverage: `tests/test_eod_dd.py` Cases C/D assert Fri fires + Tue skipped under `[4]`.

### Reference
- Trading-day arithmetic: `compute_signed_earnings_offsets()` in `pages/backtester.py` (np.busday_count + USFederalHolidayCalendar).
- Earnings parquet: `data/earnings_calendar.parquet` — 117k rows, 946 tickers, FMP-backfilled, includes forward dates.
- 2-path validation note (2026-04-29): 12 of 13 OVS signals on that date would have been killed by the blackout — only USO survived because no earnings data.

## Cycle-Year Risk Tilt (OVS, 2026-06-10)

OVS runs at 0.75x risk in midterm years (year%4==2). Evidence: all six
midterm years 2006-2026 underperform (avgR +0.19 vs +0.49 non-midterm),
leave-one-year-out stable, damage concentrated in P1 decisive-gap entries
(+0.63 -> +0.23 avgR). ~1.5 sigma after episode clustering -> shrunk-Kelly
0.75x, not the full-conviction 0.4x. Validated by LOYO, NOT by re-running
the backtest with the rule on (in-sample rules flatter themselves).

Three aligned sites -- change together:
- `strategy_config.py` OVS execution `cycle_risk_mults: {2: 0.75}` (source of truth)
- `pages/strat_backtester.py` sizing step 3b2 (generic: any strategy with the field)
- `daily_scan.py` sizing step 2c2 (stamps the mult into Sizing notes)
(order_staging needs nothing since 2026-06-11: the OVS P1 fixed-dollar
target and its `OVS_CYCLE_MULTS` were removed -- P1 takes the scanner's
staged size as-is, so the tilt flows through like every other overlay.)

The live-vs-backtest divergence found during this work (P1-only live with a
fixed $3,000 target, mild gaps dropped) was RESOLVED 2026-06-11:
order_staging trades both paths again, P1 at scanner qty x1.0 and P2 at
scanner qty x (Path2_Bps / Path1_Bps) from the row stamps plus the P2
aggregate daily cap, matching the 2-path scheme the ledger models
(P2 = 407 trades, +0.20 avgR, +82R/24y). The engine's `ovs_p1_only`
parameter remains for counterfactuals.
