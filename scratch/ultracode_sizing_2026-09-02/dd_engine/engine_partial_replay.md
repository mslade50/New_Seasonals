# GAP 2 partial engine confirmation (WP5 step 4, existing levers only)

Run 2026-09-02, `engine_partial_replay.py` + `episodes_postprocess.py` in this
folder. Results: `engine_partial_replay.json`; per-scenario trade and daily
parquets under `trades/`. Nothing in the repo was modified; no R2 writes.

## Harness

Same pipeline as `scripts/build_trade_ledger.py`'s flat pass: full-history
candidates from 2003 (24,676 candidate signal-dates, 21 book passes, 828-ticker
static overflow tier), `process_signals_fast(..., cap_bps=250,
overflow_active=True, flat_sizing=True, pooled caps None)`, daily series from
`get_daily_mtm_series` per strategy (book = sum). Full run, not truncated:
precompute ~5 min, 18 engine passes + daily series ~9 min (821s total).

Lever mechanics found in the engine (no repo edits needed):
- GRM: the book is GRM-scaled at import, so every `risk_bps` / `path1_bps` /
  `path2_bps` / `path2_daily_cap_pct` / `earnings_size_override.risk_bps` in a
  deep copy is multiplied by 1.25; `pages.strat_backtester.GLOBAL_RISK_MULTIPLIER`
  is patched to 1.875 because the overflow path computes `nominal x GRM` itself.
- Overflow exclusion: the engine reads `OVERFLOW_RISK_OVERRIDES.get(strat_name)`
  GENERICALLY for any ticker outside LIQUID_PLUS_COMMODITIES, so LT Trend ST OS,
  St OS Sznl and 52wh Breakout work as carriers by patching the module dict. All
  four carriers' liquid universes sit entirely inside the liquid set, so the
  override never leaks onto a liquid-pass row. (The override CLOBBERS the row's
  bps, so the tilt is folded into the override values for those four.)
- Tilt: `risk_bps` and `earnings_size_override.risk_bps` x tilt on liquid and
  overflow variants. WCDS's internal seasonal 1.5x/0.66x multiplier is engine
  code and unchanged.
- P2 cap: `path2_daily_cap_pct` = 1.0 x GRM.
- Clamp: the engine imports `strategy_config.CROSS_STRATEGY_OVERLAP_OVERRIDES`
  PER CALL, so patching the module list works; new pairs at 20 x GRM effective
  (37.5 at 1.875), the existing IOB+MonFri pair re-based the same way.
- Cap absorption is measured by re-running each scenario with `cap_bps=0` and
  with `ticker_notional_cap` stripped, and comparing per-trade `Risk $`.

## Baseline reproduction (A vs data/backtest_trades_full.parquet)

| | rows | flat PnL | total R |
|---|---|---|---|
| ledger (gha:33608560596, built 08:30 UTC) | 4696 | $4,137,131 | 2125.7 |
| A (local master through 2026-09-02) | 4696 | $4,140,675 | 2122.2 |

Trade-key diff: 0 only-in-ledger, 0 only-in-A. Every strategy identical to the
dollar except OLV (+$3,544: a vol-confirm/limit re-book on the 3 sessions the
local master has beyond the GHA vintage). Harness trusted.

## Scenario table

| | 2010+ ann % | Sharpe | maxDD | worst day | worst 21d | trades | tot R | 250-cap bound days | cap PnL foregone |
|---|---|---|---|---|---|---|---|---|---|
| A baseline (GRM 1.5) | 27.10 | 2.20 | -12.38 | -5.90 | -9.52 | 3799 | 1781 | 6.1% | $1.09M |
| B GRM 1.875 | 32.67 | 2.17 | -15.51 | -7.38 | -11.90 | 3797 | 1777 | 8.5% | $1.50M |
| C B + overflow excl. | 31.28 | 2.17 | -15.21 | -7.20 | -10.10 | 3797 | 1777 | 8.3% | $1.48M |
| D C + tilt | 31.20 | 2.26 | -13.11 | -6.16 | -8.01 | 3796 | 1775 | 8.2% | $1.50M |
| E D + P2 cap 1.0 + clamps | 30.85 | 2.25 | -12.62 | -6.16 | -8.01 | 3796 | 1778 | 8.3% | $1.50M |
| L levers only at GRM 1.5 | 26.50 | 2.28 | -10.25 | -5.03 | -7.20 | 3797 | 1780 | 6.1% | $1.13M |

| | 2016-07+ ann % | Sharpe | maxDD | worst day | worst 21d | trades | tot R | 250-cap bound days | cap PnL foregone |
|---|---|---|---|---|---|---|---|---|---|
| A | 34.03 | 2.70 | -7.34 | -4.89 | -6.30 | 3041 | 1486 | 8.0% | $1.08M |
| B | 40.82 | 2.67 | -9.22 | -6.13 | -7.73 | 3039 | 1483 | 10.9% | $1.47M |
| C | 39.33 | 2.67 | -8.74 | -5.97 | -6.66 | 3039 | 1483 | 10.5% | $1.45M |
| D | 39.27 | 2.70 | -8.28 | -5.56 | -5.76 | 3038 | 1481 | 10.8% | $1.48M |
| E | 38.78 | 2.69 | -8.28 | -5.56 | -5.76 | 3038 | 1484 | 10.9% | $1.49M |
| L | 33.38 | 2.72 | -6.87 | -4.52 | -5.42 | 3039 | 1486 | 8.1% | $1.12M |

All percentages are of the flat $750k base; maxDD/worst-day/worst-21d are
dollar drawdowns of the flat cumulative PnL over $750k. Ticker cap (OLV 50% NAV)
bound on 3-5 signal-days in every scenario, ~$13k foregone over 17 years:
irrelevant at this scale.

Step decomposition, annual pts (2010+ / 2016-07+): GRM step +5.57 / +6.79;
overflow exclusion -1.39 / -1.49; tilt -0.08 / -0.06; P2 cap + clamp pairs
-0.35 / -0.49. Net E vs A: +3.75 / +4.75.

## Cap absorption of the GRM step (B vs 1.25 x A)

Book, 2010+: A $3.512M -> B $4.233M against a linear $4.390M. Realised step
1.205x; 17.9% of the step absorbed (20.1% on 2016-07+). Staged risk itself only
rose 1.217x. Per strategy (2010+), PnL_B / (1.25 x PnL_A) and the share of
signal-days where the 250 cap bound, A -> B:

| strategy | PnL_B/1.25A | risk_B/1.25A | cap-bound days A -> B |
|---|---|---|---|
| Overbot Vol Spike | 0.90 | 0.94 | 16.6% -> 21.2% |
| 3x ETF Overbot Fade | 0.90 | 0.92 | 19.6% -> 26.1% |
| Weak Close Decent Sznls | 0.91 | 0.97 | 6.9% -> 12.4% |
| Oversold Low Volume | 0.95 | 0.98 | 1.0% -> 1.0% (ticker cap 1.4% -> 1.9%) |
| St OS Sznl | 0.95 | 0.98 | 0.0% -> 3.2% |
| LT Trend ST OS | 0.96 | 0.98 | 1.6% -> 5.4% |
| ATR Extended Gap Up | 0.99 | 0.99 | 0.0% -> 1.9% |
| 3x Leader Gap Fade | 1.02 | 0.97 | 0.0% -> 5.9% |
| all others (52wh, MonFri, IOB, Sector BO, 3x Bear, Monday Dip, MWC) | 1.00 | 1.00 | 0 -> 0 |

The 250 cap is fixed in effective bps, so a 1.25x GRM is a 20% nominal cut in
the cap for the cluster strategies. In E the cap forgoes $1.50M of $4.00M
(2010+), i.e. the cap is now the largest single lever in the book.

The P2 cap raise (0.75 -> 1.0 nominal) is a near no-op: D -> E changes 325 OVS
tranche rows on 30 signal days but moves only $1.6k of risk and +$1.3k PnL,
because those days are also 250-cap-bound and the per-strategy cap re-trims
whatever the P2 cap released. The clamp extension costs ~$37k / 17y (MonFri
-$21k, Monday Dip -$11k, Monthly Weak Close -$10k, IOB +$5k).

The tilt is PnL-neutral by construction (D vs C -$12k / 17y) and is where all
the risk improvement comes from: maxDD -15.2 -> -13.1, worst-21d -10.1 -> -8.0,
Sharpe 2.17 -> 2.26. Drivers: 52wh 0.70 (-$154k, 52wh led the 2016, 2018 and
2020 episodes), WCDS 0.75 (-$59k), against MonFri 1.30 (+$126k), OLV 1.17
(+$57k), 3x ETF Fade 1.27 (+$35k), ATR Ext 1.10 (+$25k). Year effect E/A:
2013 0.86 (52wh year), 2015 2.15, everything else 1.02-1.44.

## WP5 step-4 criteria on E

| criterion | 2010+ | 2016-07+ | reading |
|---|---|---|---|
| annual gain >= +4 pts vs today | +3.75 FAIL | +4.75 PASS | window-dependent; the levers eat 1.8 of the step's 5.6 pts |
| maxDD at GRM-1.5-equiv no worse than A by >1 pt | E/1.25 = -10.09 vs A -12.38 PASS; L -10.25 PASS | -6.62 vs -7.34 PASS; L -6.87 PASS | levers narrow the drawdown by ~2 pts, not widen it |
| worst-21d no worse by >10% | E raw -8.01 vs A -9.52 PASS (equiv -6.41) | -5.76 vs -6.30 PASS | passes even unscaled |
| 2016+ worst episode is not the Jun-Jul 2026 OLV stack | n/a | 2021-01-21..27, OVS -$52k (squeeze week) PASS | see caveat below |

Caveat on criterion 4: with tilt + GRM alone a 2026-07-22..29 episode enters
E's 2016+ list at #3 (-6.97%: 3x Bear Fade -$20.7k, MonFri -$11.9k, OLV
-$6.7k), and OLV's own Jun-Jul 2026 21d trough grows from -4.98% NAV (A) to
-6.06% (E), already larger than the book's worst 21d in that window (-5.76%);
other strategies' gains cushion it. In L (levers at 1.5) the 2026-06-12..07-01
OLV-only leg (-$40.9k) appears as episode #5. The stack is at book-worst-21d
scale before the depth ladder, pullback tilt and flow up-size are applied.

What this run CAN judge: the multiplier, the exclusion, the tilt, the P2 cap
and the clamps, with the engine's real P2 cap, ticker cap, per-strategy cap and
staged-candidate clamp keyed. Fills were NOT assumed linear (the two-cap
absorption above is the non-linearity).

What it CANNOT judge (levers with no engine code): OLV depth ladder re-key +
`OLV_COMPOSITE_CLIP` 1.5, `open_leg_mults` 0.8/1.2 (WCDS, LT Trend), OVS
bottom-extremity 0.7x, IOB same-day index clone 0.5x, WP8 flow up-size 1.2x and
cap relief 250 -> 375, WP10 OLV pullback 1.15x, WP1 `MARGIN_TRIM`, WP4 costs.
Criterion 1 is the one these decide: flow cap relief targets exactly the
$1.5M the 250 cap forgoes in E, while OVS extremity and the 0.8 solo cut go
the other way; +3.75 vs +4.0 is inside that band. Criterion 4 is specifically
about the OLV composite (ladder x tilt x pullback x flow, clip 1.5) and is
therefore NOT settled by a run where OLV carries only tilt 1.17 x GRM.
Criteria 2 and 3 have ~2 pts and ~1.5 pts of headroom for the unbuilt levers
to consume.

## Worth discussing

1. The +4 gate turns on the window: 2010+ fails by 0.25 pt, 2016-07+ passes by
   0.75. The brief names both windows without saying which one decides; the
   overflow exclusion alone (-1.4 pts) is what pulls 2010+ under.
2. Cap absorption is ~18-20% of the step and lands on OVS, 3x ETF Fade and WCDS
   (cap-bound days 17->21%, 20->26%, 7->12%). Holding 250 effective while
   stepping GRM is a nominal cap cut for the cluster strategies; the flow cap
   relief is the only proposed lever aimed at it and it is unbuilt. The
   standing rule "250 stays fixed in effective bps until GRM > 3.0" is a
   real choice, not a neutral default.
3. WP7.4 (P2 cap 0.75 -> 1.0) does nothing in the engine (+$1.3k / 17y) because
   the 250 cap binds over it on the same days. Either drop it or pair it with
   cap relief; on its own it is a config change with no replayable effect.
4. The tilt's entire risk benefit comes from cutting 52wh and WCDS, and it costs
   2013 14%. The OLV 1.17 raise pushes the Jun-Jul 2026 OLV 21d trough to -6.06%
   NAV, past the book's worst 21d, before any depth/pullback/flow multiplier
   exists; the composite clip needs to be modelled before that raise ships.
