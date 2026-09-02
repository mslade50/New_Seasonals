# Dynamic sizing: implementation brief (2026-09-02)

Audience: the agent implementing the Sizing Hierarchy Plan
(`dynamic_sizing_plan_2026-09-02.html`). Read CLAUDE.md first; every
guardrail there applies. This brief tells you what to build, in what
order, where it lives, and what "done" means. It does not re-argue the
evidence; the plan and `scratch/dynamic_sizing_results*_2026-09-02.json`
hold that.

## Ground rules (from CLAUDE.md, restated because they bind here)

- Engine and live must agree. Anything that sizes a staged order in
  `daily_scan.py` is replayed point-in-time in
  `pages/strat_backtester.py::process_signals_fast`; `order_staging.py`
  (OneDrive `trading_ibkr`) takes staged sizes as-is unless the control
  needs the T+1 open.
- Sizing is in ATR risk (bps of NAV), never notional. Do not add a
  notional cap anywhere.
- `data/rd2_fragility.parquet` is the only dial source, 10d MA of the
  63d column, lag-1 for anything that trades the next session. Rows
  before 2026-07-02 are a recompute vintage; say which vintage a study
  used.
- Any NEW dial-conditioned control needs a pre-registered protocol on
  disk BEFORE its study runs (P1 to P3 below). Do not adopt a re-scored
  stats JSON into the live path.
- Do not build a book-level throttle, taper, vol target, or
  drawdown-state scaler. Do not build an OLV sleeve risk cap. Do not
  rebuild ML meta-labelling. These are closed negative results.
- Tests are guards. Every shipped control gets a test that pins config,
  boundary behaviour, and engine/live parity, next to the existing ones
  in `tests/`.
- Commit on main, promptly, with the session trailer. Do not push
  blind; re-check `git log` first (concurrent sessions).

## Work packages, in order

### WP1 (ship): same-ticker cross-strategy clamp for every dip-buy pair

Mechanism exists: `strategy_config.CROSS_STRATEGY_OVERLAP_OVERRIDES`,
applied in `daily_scan.py` step 5b and replayed in
`pages/strat_backtester.py` sizing step 3b3c from a candidate pre-pass.
Today it holds one pair (Indices Oversold Bounce + SPY QQQ MonFri
Reversion, 20 bps nominal each, GRM-scaled).

1. Add pairs: Monday Dip + Weak Close Decent Sznls; SPY QQQ MonFri
   Reversion + Weak Close Decent Sznls; Monthly Weak Close + SPY QQQ
   MonFri Reversion; Monthly Weak Close + Indices Oversold Bounce;
   Monday Dip + Indices Oversold Bounce. Keep the existing pair. Use the
   same `risk_bps_when_overlapping: 20` nominal convention.
2. Confirm the pre-pass keys on the tradeable ticker after
   `SPOT_TO_TRADEABLE` aliasing (it does for the existing pair).
3. Test: extend the overlap-clamp test to the new pairs; a same-day
   same-ticker fire of each pair clamps both rows; a different-ticker
   fire does not.
4. Rebuild the ledger locally (`scripts/refresh_view.py`, no `--upload`)
   and print the vintage diff; expect only same-day same-ticker family
   rows to change (14 pairs since 2010).

Acceptance: tests green; ledger diff touches only those rows; no
change to any row outside the dip-buy family.

### WP2 (ship): Indices Oversold Bounce SPY+QQQ same-day clamp

Same clamp machinery, keyed WITHIN one strategy on same-day multi-ticker
fires. Prefer a generic execution field so it is data, not code:

```
"same_day_index_clone": {"mult": 0.5}
```

- `daily_scan.py`: post-pass beside 5c (same-day derate): if a strategy
  carries the field and stages more than one ticker on the same signal
  date, scale every such row by `mult`.
- `pages/strat_backtester.py`: same rule in the candidate pre-pass that
  already counts staged candidates per (strategy, day) for 3b4.
- Carrier: Indices Oversold Bounce only. MonFri MUST NOT carry it (its
  both-fire days are its best days: 3.1x PnL at 2.1x sd). Pin that in
  the test (single-carrier assertion, like `test_same_day_derate.py`).
- Composition: multiplies with frag bands and the earnings override
  like the other post-pass overlays.

Acceptance: config invariant test (only IOB carries it); a two-ticker
IOB day stages both at 0.5x; a one-ticker day is untouched; engine
replay matches on a fixture.

### WP3 (ship, appetite rule): 52wh Breakout 0.5x at >= 5 open legs

The dormant open-position-count machinery (`execution['ladder_multipliers']`,
`daily_scan.load_open_position_counts`, the engine's open-count rung)
may be keyed per TICKER (it was OLV's original ladder). Check first.

- If it is per-strategy: carrier `52wh Breakout`,
  `ladder_multipliers: [1,1,1,1,1,0.5]` with the last rung sticky (>= 5
  open -> 0.5). Document the rung semantics in the config comment.
- If it is per-ticker: add `open_leg_derate: {"from_n": 5, "mult": 0.5}`
  mirroring `same_day_signal_derate` but counting the strategy's OPEN
  positions (filled legs; the scan reads the nightly positions the same
  way `load_open_position_notionals` does for the OLV ticker cap).
  Engine side counts `open_positions` for the strategy at the
  candidate's entry, matching the scan's fill-based count. Known bound,
  same as the OLV ticker cap: unfilled working limits are invisible to
  both sides, so parity holds.
- Stamp the mult into the Sizing notes; the scan email shows it.
- Record in CLAUDE.md as an APPETITE rule on N=23 (avgR 0.08, win 35%)
  with re-exam at +20 trades in the cell; retire if the cell's avgR
  exceeds 0.4 then.

Acceptance: test pins carrier + boundary (4 open -> 1.0x, 5 open ->
0.5x); engine/scan parity on a fixture; ledger diff shows only 52wh
legs entered with >= 5 open changing (23 trades since 2010).

### WP4 (build): live-vs-ledger haircut measurement

Goal: the ratio of realised live R to ledger R on matched trades, plus
the fill rate of staged limits, published as one number with a date.

- Source of live fills: Trade_Signals_Log (written by
  `verify_fills.py`) and the Trade Log DO ring (`/fills`, 14d retention,
  so start accumulating now; the Sheets log is the long history).
- Match key: (Strategy, Ticker, Signal Date) to the ledger's
  `backtest_trades_full.parquet`. Compare R_Multiple; also record
  staged-but-unfilled counts per strategy.
- Script: `scripts/build_live_vs_ledger.py`, writes
  `data/live_vs_ledger.json` (committed; small) with N matched, ratio,
  bootstrap CI, per-strategy table, and `asof`. Run in the PM
  daily_screener job (best effort), and surface on the site Execution
  tab (`build_site.py` copies it; `execution.js` renders one card).
- The plan's 50% haircut is replaced by the measured ratio once
  N_matched >= 150; until then print "placeholder 50%".

Acceptance: script runs on the current Sheets log; JSON validates;
site card renders with N and the placeholder flag; test on a fixture
log.

### WP5 (build): GRM frontier script + rule

- `scripts/grm_frontier.py`: stationary block bootstrap (mean block
  10, 3,000 paths, 252 sessions, seed fixed) of the flat-basis daily
  book PnL from `data/backtest_daily_pnl.parquet` (2016+ window and
  full window both printed), at GRM multiples 0.75 to 2.0 and haircuts
  {measured, 0.5, 0.25}. Output: P(maxDD > 10/15/20%), median 1y PnL,
  5th-pct maxDD, and the largest GRM satisfying the rule.
- The rule, as a constant block at the top of the script with a
  comment pointing at the plan: `DD_THRESHOLD = 0.15`, `P_BUDGET = 0.05`,
  `HAIRCUT = live_vs_ledger ratio or 0.50`.
- Schedule: quarterly via the risk_report workflow (`--frontier` flag,
  prints the table into the email) or a Task Scheduler entry; do not
  auto-change GRM. GRM changes stay a human edit to
  `strategy_config.GLOBAL_RISK_MULTIPLIER` with the frontier output in
  the commit message.
- Reference implementation of the bootstrap: section 5 of
  `scratch/dynamic_sizing_study_2026-09-02.py` and section D of
  `..._study2_...py`.

Acceptance: script reproduces the plan's table (GRM 1.5, 50% haircut:
P(DD>15%) about 3.3%) within bootstrap noise on the same seed.

### WP6 (build, display only): OLV stack n_eff on the hedge panel

- For the OLV legs open in the selected account, compute
  `n_eff = (sum r_i)^2 / (r' C r)` where r_i is each leg's ATR risk in
  dollars and C the trailing-63d daily return correlation matrix of the
  tickers (from `master_prices.parquet`; the scan already reads it).
- Ship it beside the panel's existing beta arithmetic: "OLV: 9 legs,
  n_eff 2.4, single-bet risk $X". Build-side in `scripts/build_betas.py`
  (add a `corr` block for the currently open OLV tickers) so the
  browser only reads numbers; `assets/execution.js` renders.
- No orders, no sizing, no `data-mutation` controls (the panel's
  contract).

Acceptance: test for the n_eff helper (identity C -> n; all-ones C ->
1); panel renders with the betas.json extension present and degrades
silently without it.

### WP7 (prereg study P1): dial-armed book beta hedge

Write the protocol file FIRST:
`scratch/ultracode_research/beta_hedge_prereg_2026-09-02.md`, copying
section 6 P1 of the plan verbatim (series, β̂ window, arming 50/45 with
65/60 sensitivity, episode definition, decision statistic, ship rule,
paper period). Commit it before running anything.

Study script `scratch/beta_hedge_pit_study.py`:

1. Daily flat book PnL from the ledger build's daily parquet; SPY from
   master_prices.
2. Dial vintages: current-weights series (rd2_fragility 63d, 10d MA)
   and the PIT vintage-lagged series produced by
   `scratch/pit_reestimate.py` (re-run it if its output is stale; it
   scores year Y with weights fit through Y-1, 2018+). PIT is primary.
3. β̂ = 63-session OLS of book return on SPY, lag-1, clipped [-1, 2].
4. Hedge PnL on armed days = -β̂ × SPY_ret × NAV, less 2 bps of hedge
   notional per arm event.
5. Episodes: armed runs separated by >= 21 unarmed sessions. Report
   per-episode hedge PnL, mean, episode-clustered t, full-sample
   equal-vol Sharpe hedged vs unhedged, LOYO by year.
6. Apply the ship rule from the prereg. Either outcome is filed in
   CLAUDE.md under the fragility section's negative results or as a
   new "Book beta hedge" section.

If it passes: paper-track first. Add an `armed` state and a target
contract count to the hedge panel (`β̂ × NAV / (50 × ES)` for MES);
log one full armed episode in `data/beta_hedge_paper.jsonl` (R2
canonical, like the event journal) before any live order. Live
mechanics come after paper and are their own brief: MES orders through
the execution bridge with native MOC encoding (the 2026-08-21 rule),
strategy tag `Book_Beta_Hedge` in the orderRef, quarterly roll from
the panel's roll-off list, and an execution-report attribution entry.

### WP8 (prereg study P2): exemption re-test at dial 65

Protocol file first:
`scratch/ultracode_research/exemption_retest_65_prereg_2026-09-02.md`
(section 6 P2 of the plan). Then `scratch/exemption_retest_65.py`:
OLV, LT Trend ST OS, OVS trades with PIT signal-date dial >= 65, 2018+;
avgR vs the same strategy at < 50; episode-clustered t; LOYO; drop-best-
episode. Rule: a `[[65, 999, 0.5]]` band only if clustered t <= -2.0 on
PIT weights AND the effect survives dropping the best episode. If WP7
shipped, run on β-hedged trade returns. Either outcome filed.

### WP9 (ship after engine confirmation): strategy base-bps half-tilt

The walk-forward evidence is done (plan L1d; `scratch/dynamic_sizing_study3_2026-09-02.py`
and `..._study4_...py`): a half-blend toward shrunk Sigma^-1 mu weights on
the per-strategy daily series, re-fit each January on data through the
prior December, beat the current allocation in 12 of 13 held-out years
2014-2026 (+4 pts NAV/yr, same maxDD, PnL/maxDD +18%). It met the
pre-stated gate. What remains is the engine confirmation and the wiring.

Shipping multipliers (fit through 2025; `mult = clip(0.5*w + 0.5, 0.6, 1.4)`):

| Strategy | mult | | Strategy | mult |
|---|---|---|---|---|
| 52wh Breakout | 0.73 | | Monday Dip | 1.02 |
| Indices Oversold Bounce | 0.83 | | ATR Extended Gap Up | 1.04 |
| Sector BO | 0.84 | | 3x ETF Overbot Fade | 1.15 |
| Weak Close Decent Sznls | 0.84 | | SPY QQQ MonFri Reversion | 1.16 |
| St OS Sznl | 0.92 | | LT Trend ST OS | 1.19 |
| Overbot Vol Spike | 1.00 (cap-bound, held) | | Oversold Low Volume | 1.29 |
| 3x Bear Fade, 3x Leader Gap Fade, Monthly Weak Close | 1.00 (too few active days) | | | |

Steps:

1. **Engine confirmation (P3).** Run `process_signals_fast(risk_multipliers=<table>)`
   on the full ledger (flat basis) via a copy of `scratch/grm_replay_study.py`;
   compare to the unmodified ledger over 2014-2026. PASS if the engine
   reproduces at least half the walk-forward PnL gain (>= +2 pts NAV/yr)
   with maxDD no worse by more than 1 pt and worst-21d no worse by more
   than 10%. FAIL means cap interactions ate it (a raised strategy binds
   its 250 bps daily cap more often): re-derive the weights with a
   walk-forward on ENGINE output instead of the daily series, and re-gate.
   Either outcome is filed in CLAUDE.md.
2. **Wiring.** New `strategy_config.STRATEGY_BASE_TILT = {name: mult}`
   applied at import right after `GLOBAL_RISK_MULTIPLIER` scales
   `risk_bps` (and OVS `path1_bps`/`path2_bps` if OVS ever carries a
   tilt; today it does not). It scales the BASE bps only; every overlay
   (frag bands, recency ladder, earnings override, gap derate, same-day
   derate, cross-strategy clamp) composes on top unchanged. The earnings
   size override REPLACES the base, so it must be tilted too (multiply
   `earnings_size_override.risk_bps` by the same mult) or the override
   silently un-tilts the strategy; pin this in the test.
3. **Annual re-fit.** `scripts/fit_strategy_tilt.py`: reads the
   strategy_daily payload (or rebuilds it from the ledger), fits the
   expanding-window shrunk Sigma^-1 mu (cov shrinkage 0.3 to diagonal,
   mu shrunk 0.5 toward equal-Sharpe, mean |w| = 1), prints the clipped
   table and a diff vs the live table. Run each January; the table
   change is a human commit with the script output in the message. A
   strategy whose raw w falls below 0.6 two Januaries running is
   flagged as a retirement question, not clipped further.
4. **Tests.** `tests/test_strategy_tilt.py`: every live strategy has a
   mult in [0.6, 1.4]; OVS and the three thin strategies are 1.0; the
   scaled `risk_bps` equals nominal x GRM x mult; the earnings override
   is tilted; the ledger diff after rebuild changes Risk_flat by exactly
   the mult per strategy (spot-check three).
5. **Site/email.** Sizing notes stamp "tilt 0.73" beside the base bps;
   the scan email's sizing footer lists non-1.0 tilts. Total in-sample
   book vol is unchanged by construction (weights are vol-matched), so
   GRM is untouched.

Order: WP9 step 1 can run in parallel with WP1-WP3; steps 2-5 only after
step 1 passes.

## Do-not-build list (closed)

- Book-level scalar throttle, taper, dial Kelly table, EWMA or VIX vol
  target, open-leg 1/sqrt(n): nine overlays, none above baseline at
  equal vol, LOYO 2 to 5 of 11 years.
- OLV sleeve open-risk cap: -$70k/16y for -$3.4k worst-21d; cuts the
  higher-edge legs. The disabled `olv_book_cap.py` task stays disabled.
- Drawdown-state exposure scaling: next-21d mean is 25 to 41% ann. in
  every drawdown state; no forecast content.
- MonFri SPY+QQQ clone clamp: both-fire days are its best days.
- Changing the ATR risk unit's vol power: avgR flat across vol terciles.

## CLAUDE.md updates to make with each package

- WP1/WP2/WP3: extend the "Cross-Strategy Overlap Clamp" section and
  add a short "Stack controls (2026-09-02)" note listing the IOB clone
  clamp and the 52wh open-leg derate as appetite/variance rules with
  their N and re-exam triggers.
- WP4/WP5: a "Sizing basis measurement" note: where the haircut lives,
  the GRM rule's three constants, and that GRM stays a human edit.
- WP7/WP8/WP9: file the prereg paths beside the existing ones under
  "Pre-registration requirement", and the outcome under negative
  results or a new section once run.

## Evidence files

- `scratch/kelly_read_book_2026-09-02.py`, `..._rungs_...py`,
  `..._same_ticker_...py`: Kelly reads of the ledger.
- `scratch/dynamic_sizing_study_2026-09-02.py` and
  `scratch/dynamic_sizing_results_2026-09-02.json`: regime buckets,
  variance forecasting, within-strategy adds, overlay sims + LOYO, DD
  frontier, DD-state test, correlation by regime.
- `scratch/dynamic_sizing_study2_2026-09-02.py` and
  `..._results2_...json`: beta hedge by regime, OLV sleeve cap replay,
  index clones, haircut frontier, cluster structure, 52wh stack cell.
- Article text: `scratch/kelly_read_source_article_2026-09-02.md`.
