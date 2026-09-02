# Sizing Hierarchy Revised: implementation brief (2026-09-02, second pass)

Audience: the agent implementing `sizing_optimal_plan_2026-09-02.html`.
This SUPERSEDES `dynamic_sizing_implementation_brief_2026-09-02.md` wherever
the two differ (differences are flagged inline). Read CLAUDE.md first. The
ground rules from the earlier brief still apply: engine and live must agree,
sizing is ATR risk in bps of NAV, PIT dial for any dial claim, prereg before
any new dial-conditioned study, tests as guards, commit on main.

Evidence: `scratch/ultracode_sizing_2026-09-02/` (13 studies, ~180 scripts,
per-study `*_results.json`), plus `_evidence_pack.json` and `_proposals.json`
(the four competing plans). The plan's refutation ledger (section 11) records
why each rule is shaped the way it is; do not "improve" a multiplier without
re-reading it.

Closed results (do not build, do not re-scan): the list at the end of plan
section 13. Add it to CLAUDE.md's negative-results registry as part of WP0.

## Order of work

Phase 0 (days, no sizing change): WP0 documentation, WP1 margin guard, WP2
IBKR what-if, WP3 headroom monitor, WP4 fills harvest + engine costs.
Phase 1 (one release, engine-confirmed): WP5 GRM 1.875 + overflow exclusion,
WP6 keep-adjusted tilt, WP7 within-strategy rules, WP8 flow rules.
Phase 2 (paper then live): WP9 hedge, WP10 OLV pullback tilt.
Phase 3 (prereg files first): WP11.
Standing rules: WP12 GRM step rule, WP13 sizing base.

Every Phase 1 item ships behind ONE engine confirmation run (WP5 step 4).

---

### WP0 Documentation and registry

- CLAUDE.md: new section "Sizing Hierarchy Revised (2026-09-02)" summarising
  the three-wall GRM rule, the composition rules (max-not-product, OLV
  composite clip, guard gates reliefs), and the closed list from plan
  section 13 appended to the existing negative results.
- Replace the first plan's L3 current-weights trade table wherever it is
  cited with the PIT table: book [65,999) +0.006R vs <50 (t 0.04); OLV
  +0.235R; OVS P1 [50,999) -0.104R (t -0.84); LT Trend [50,999) -0.375R
  (t -2.33, 9 episodes). Drop the "65+ book avgR halves" claim.

### WP1 Margin-feasibility guard (order_staging, OneDrive trading_ibkr)

Purpose: the broker's constraint made visible one day early. It is NOT a
risk control and NOT a notional cap (CLAUDE.md's no-notional-caps rule is
about sizing; this trims only to stay solvent through the 15:45 ET Soft
Edge Margin expiry, which is when IBKR auto-liquidates with no call).

1. Rates table (replace with the what-if numbers from WP2 when available):
   single stock and sector ETF 15%; broad index ETF 8%; small-cap index 10%;
   3x ETF long 45%, 3x ETF short 90% (until WP2 confirms PM treatment);
   short stock under $16.67: max(15% x notional, $5/share); under $5: 100%.
2. Before the 9:31 chain: Req_proj = sum over open positions (live marks x
   rate) + 1.10 x sum over staged entries at full size (notional x rate),
   on LIVE NLV (`ib.accountSummary NetLiquidation`).
3. Gates, in this order:
   - Req_proj > 0.60 x NLV: set `relief_off = True` for the day (WP8 flow
     up-size and cap relief disabled; the per-strategy cap stays 250).
   - Req_proj > 0.70 x NLV: scale every staged entry's Quantity by
     (0.70 x NLV - Req_open) / (1.10 x Req_staged), floor 0, label the row
     `MARGIN_TRIM`, never touch open positions.
   - Req_open alone > 0.85 x NLV: email alarm, no new entries.
4. 15:30 ET re-check (new small task or a step in an existing PM runner):
   recompute Req_open on live marks; alarm above 90% of NLV.
5. Log gross/NLV, blended rate, Req_proj/NLV and any trim on every scan
   email and the hedge panel (WP3).
6. Engine: add a `MARGIN_TRIM` partial-size row type to
   `process_signals_fast` so the guard's cost can be replayed (the plan's
   estimate is 0.5 trims/yr at GRM 1.875, 2.6/yr at 3.0, each on a top-1%
   PnL day; unmeasured until this exists).
7. Tests (OneDrive): rates table, the three gates, order of operations
   (cap relief first, then trim), a fixture day with 2023-02-03's book.

### WP2 IBKR what-if and exposure fee (one afternoon, human-driven)

Run Risk Navigator what-if margin on the ledger books of 2013-11-04,
2016-06-14, 2019-06-26 and 2023-02-03 at 1.0x, 1.25x and 1.5x current size,
and read the Exposure Fee Calculation Report at current, 1.5x and 2x gross.
Record: the 3x short rate (45 vs 90), the concentration trigger, and the
requirement/NLV on each book. These four numbers replace the stylised rates
in WP1 and decide whether GRM 2.25 is reachable without WP11's futures
routing. Reconstruct each book from `data/backtest_trades_full.parquet`
(open on that date: Entry Date <= d <= Exit Date, Shares_flat x Entry Price).

### WP3 Headroom and factor display (hedge panel, display only)

Add to `assets/execution.js` / `scripts/build_betas.py`: gross/NLV and
gross/base by class; blended stylised TIMS rate; projected requirement/NLV
with the 60/70/85/90% lines; -30% one-day equity scenario loss / NLV (the
exposure-fee screen); rolling 126d book beta with up/down-day split;
effective N (126d) and active strategies; 12-1 momentum-factor loading;
per-strategy share of SPY covariance while armed; OLV stack
n_eff = (sum r_i)^2 / (r' C r); theme variance shares; hedge state. Nothing
here sizes anything and the panel keeps its no-`data-mutation` contract.

### WP4 Measurement: fills harvest, engine costs

1. Schedule `trade_journal.py` (OneDrive) daily; harvest the DO `/fills`
   ring before its 14-day retention drops rows; match on orderRef
   `SYMBOL|ACTION|Strategy|Date` to ledger keys; write
   `data/live_vs_ledger.json` (N matched, live R / ledger R, bootstrap CI,
   fill rate of staged limits, per-strategy table, asof). Site card on the
   Execution tab. Until N >= 150: live R = ledger R x 0.60 is the working
   assumption; Sheets `Fill_Status` is yfinance-modelled and is never cited
   as live evidence.
2. `process_signals_fast`: commissions $0.005/share, $1 floor, both sides;
   ADV-bucketed half-spread on MKT / MOC / stop exits (1.5 bps > $100M
   63d dollar ADV, 3 bps $25-100M, 6 bps $5-25M, 12 bps below); limit
   entries and target limits stay cost-free. Expect ledger avgR 0.453 ->
   ~0.428. Rebuild the ledger; print the vintage diff.
3. `scripts/grm_frontier.py` (from the earlier brief WP5) now prints the
   THREE-WALL table (plan section 3) plus the drawdown disclosure rows at
   haircuts {measured or 0.40, 0.27, 0.71}; it no longer applies the
   P(DD > 15%) <= 5% rule. Constants: `HAIRCUT`, `GUARD_LINE = 0.70`,
   `GAP_BOUND = 0.15`, `K999_BOOK = 1.28`.

### WP5 GRM 1.5 -> 1.875, overflow longs excluded, engine confirmation

1. `strategy_config.GLOBAL_RISK_MULTIPLIER = 1.875`. Preconditions: WP1
   live, WP2 run with the four books under 85% of live NLV at 1.25x.
2. Overflow-tier longs keep today's effective bps: set
   `OVERFLOW_RISK_OVERRIDES` nominals x (1.5 / 1.875) = 0.8 for OLV (25 ->
   20), LT Trend ST OS (30 -> 24), St OS Sznl (40 -> 32), 52wh Breakout
   (35 -> 28). Three of these are NEW carriers: wire them through
   daily_scan, daily_portfolio_report and strat_backtester's
   `overflow_active` path exactly as OLV's existing override. Overflow
   SHORTS (OVS, ATR Ext) take the step.
3. ADV participation: every order notional <= 1.0% of the 21d median
   dollar ADV (0.4% for LT Trend ST OS, St OS Sznl, WCDS, IOB whose avgR is
   under 0.4R); hard refusal above 5%. Re-express the existing participation
   cap in these terms; stamp the participation on the staging row.
4. ENGINE CONFIRMATION (the one gate for all of Phase 1): run
   `process_signals_fast` with GRM 1.875, WP6 multipliers, WP7 rules, WP8
   rules, caps, P2 cap, ticker cap and cross-strategy clamp all re-applied,
   on the full ledger (flat basis), and compare to the unmodified ledger over
   2010-2026 and 2016-07+. PASS if: annual PnL gain >= half the package-C
   replay's (>= +4 pts NAV/yr at 1.875 vs today), maxDD no worse than
   today's by more than 1 pt at GRM 1.5-equivalent (i.e. the levers, not
   the multiplier, must not widen the drawdown), worst-21d no worse by more
   than 10%, and the 2016+ worst episode is not the Jun-Jul 2026 OLV stack
   (that is the signature of the composition failure the clip exists to
   prevent). FAIL: fix the composition (WP7 clip / WP8 gate), not the
   multipliers, and re-run.
5. Tests: `tests/test_grm_step.py` pins GRM, the four overflow nominals,
   and that overflow-long effective bps are unchanged from the GRM-1.5
   values.

### WP6 Keep-adjusted allocation tilt (replaces the earlier brief's WP9)

Multipliers (fit through 2025; `mult = clip(0.5 w + 0.5, 0.7, 1.3)` with
`w = Sigma_tilde^-1 mu_tilde`, mu shrunk by keep_s and halfway to an
equal-Sharpe prior; robust_bayes_03_allocation.json `keep_mult_0.7_1.3`):

| Strategy | mult | Strategy | mult |
|---|---|---|---|
| 52wh Breakout | 0.70 | LT Trend ST OS | 1.04 |
| Weak Close Decent Sznls | 0.75 | Monday Dip | 1.09 |
| Sector BO | 0.87 | ATR Extended Gap Up | 1.10 |
| St OS Sznl | 0.88 | Oversold Low Volume | 1.17 |
| Indices Oversold Bounce | 0.89 | 3x ETF Overbot Fade | 1.27 |
| Overbot Vol Spike | 1.00 (held) | SPY QQQ MonFri Reversion | 1.30 |
| 3x Bear Fade, 3x Leader Gap Fade, Monthly Weak Close | 1.00 | | |

Wiring as in the earlier brief's WP9 (`STRATEGY_BASE_TILT` applied at import
after GRM; tilt the earnings override too; sizing-notes stamp; annual
`scripts/fit_strategy_tilt.py` refit with keep_s from
`estimation_haircut_results.json`, judged overrides OVS 0.48 and 52wh 0.42,
pilots capped at keep 0.50). Retirement flag: raw w < 0.6 two Januaries
running. 2027 Q1 re-read: retire the OLV and LT Trend raises if OLV PnL/risk
stays below 0.5 and LT Trend below 0.15 across the first two non-midterm
quarters.

### WP7 Within-strategy rules

1. **OLV ladder re-key.** `signal_recency_ladder` gains a depth rung:
   `mult = max(recency_rung, depth_rung)`, depth_rung by the count of open
   OLV legs (FILLED positions plus WORKING OLV entries within their T+3
   window) at signal time: 0 -> 0.5, 1-2 -> 0.7, 3+ -> 1.0. daily_scan reads
   filled legs from the nightly positions (as `load_open_position_notionals`
   does) and working entries from the previous day's staged rows still
   inside their fill window; the engine counts open_positions plus
   candidates whose fill window is open. Same-ticker adds full size; ticker
   cap unchanged. Add `OLV_COMPOSITE_CLIP = 1.5`: the product of tilt x
   ladder x pullback (WP10) x flow (WP8) on any OLV leg is capped at 1.5x
   pre-GRM (frag/PC state does not apply; OLV is bandless).
2. **Weak Close and LT Trend solo/adds.** New generic execution field
   `open_leg_mults: {"none_open": 0.8, "adds": 1.2}` (adds = any leg
   entered with >= 1 same-strategy leg open or working). Carriers: Weak
   Close Decent Sznls, LT Trend ST OS only. Same-sector clusters are NOT
   cut. Composes with frag/PC bands, tilt, flow, and the 250 cap.
3. **OVS bottom-extremity tier.** In daily_scan sizing (from the
   rank_ret_{2,5,10,21}d columns the filter already reads) and in the engine
   pre-pass: `mult = 0.7 if mean(rank_2d, rank_5d, rank_10d, rank_21d) < 94
   else 1.0`. No top-tier boost. Stamp the mean rank on the staging row.
4. **OVS path-2 cap.** `path2_daily_cap_pct: 0.75 -> 1.0` nominal. Review
   after one year: second half-step to 1.5 if the capped cell's realised
   avgR stays above the uncapped cell's.
5. **Clones.** IOB SPY+QQQ same-day 0.5x each (`same_day_index_clone`,
   IOB only; MonFri must NOT carry it, pin in the test).
   `CROSS_STRATEGY_OVERLAP_OVERRIDES` extended to Monday Dip + WCDS,
   MonFri + WCDS, Monthly Weak Close + MonFri, Monthly Weak Close + IOB,
   Monday Dip + IOB at 20 bps nominal.
6. **52wh Breakout: NO depth rule.** The earlier brief's WP3 (0.5x at >= 5
   open) is WITHDRAWN. If McKinley wants the Feb-2014 tail insured, the only
   defensible form is 0.5x at >= 6 open labelled tail insurance with a
   re-exam at +15 legs; default off.
7. Tests: `tests/test_olv_depth_ladder.py` (rung table, working-limit
   counting, composite clip), `tests/test_open_leg_mults.py` (carriers,
   boundary, parity), `tests/test_ovs_extremity.py` (cut-point, no boost,
   engine parity), `tests/test_same_day_index_clone.py`, extended
   overlap-clamp tests.

### WP8 Flow rules (family 5d raw candidate count)

Shared state: `flow_state.py` computes, from the SAME raw filter masks
daily_scan evaluates (before earnings blackout, T+1 gates, fill windows),
the trailing-5-session candidate count per family inclusive of today.
Families: dip_buy {WCDS, MonFri, Monday Dip, IOB, Monthly Weak Close, St
OS Sznl}, oversold_hold {OLV, LT Trend ST OS}, short_fade {OVS, 3x ETF
Overbot Fade, ATR Ext Gap Up}, bear_etf_fade {3x Bear, 3x Leader},
breakout {52wh, Sector BO}. Thresholds: dip_buy >= 6, oversold_hold >= 7,
short_fade >= 104. NEVER count staged rows (OVS runs ~16x its fills; every
threshold would shift). The engine reproduces the count from
`generate_candidates_fast` output per date; see
`flow_conditional_01_dump_candidates.py` for the exact construction.

1. **Cap relief.** `cap_bps(strategy, day) = 375 if family_hi_flow else
   250`, taking the MAX with any other relief (never the product). Applied
   in order_staging's per-strategy cap stage and the engine's cap post-pass.
   Breakout and bear_etf_fade families excluded. Off when WP1's
   `relief_off` is set.
2. **Up-size.** New sizing step after 2b (frag band) and before 2c (OLV
   ladder): `mult = 1.2 if family_hi_flow (dip_buy also requires 10d-MA 63d
   dial < 50) else 1.0`. No down-size leg. No breakout carrier. Composes
   multiplicatively; inside the OLV composite clip. Off when `relief_off`.
3. Post-ship review file: `scratch/ultracode_research/flow_rules_review_2026-09-02.md`
   with the rule "PnL per unit risk on the first 60 hi-flow trades >= 1.3x
   the concurrent lo/mid-flow trades, else retire", plus the OVS same-day
   >= 5 cluster tilt as a registered second-stage refinement.
4. Tests: `tests/test_flow_state.py` (mask-based counting, thresholds,
   family membership, dial gate), cap relief max-not-product, guard gate.

### WP9 Dial-armed whole-book beta hedge (paper then live)

Prereg file first: `scratch/ultracode_research/beta_hedge_prereg_2026-09-02.md`
(copy plan section 8: arm 50 / release 45, whole book, ES/MES, 126d or
0.5 x 63d + 0.5 x expanding beta clipped [-1, 2], ratio 1.0x never above,
native MOC entry, quarterly roll; second-stage VIX gate registered with its
decision rule). Then:

1. Hedge panel: armed state, beta_126 vs beta_63, target MES contracts =
   beta x sizing base / (5 x ES), realised armed beta and drift-to-date,
   logged daily to `data/beta_hedge_paper.jsonl` (R2-canonical, event-journal
   convention).
2. Paper period: one full armed episode. Ship if realised armed beta is
   within [0.2, 0.8] and the panel's attribution reconciles.
3. Live: `hedge_moo.py` (OneDrive, own clientId, own Task Scheduler entry,
   activation flag like `pitch_moo_enabled.flag`), native `orderType MOC /
   tif DAY` (the 2026-08-21 encoding rule with the verify-the-reject guard),
   orderRef strategy tag `Book_Beta_Hedge`, execution-report attribution
   entry, roll from the panel's roll-off list. Futures margin is additive:
   WP1 must include the hedge's margin in Req_proj.
4. Tests: OneDrive `test_hedge_moo.py` (encoding, arming hysteresis, contract
   arithmetic, never above 1.0x), repo `tests/test_beta_hedge_state.py`.

### WP10 OLV market-pullback tilt (confirm and ship)

`execution['market_state_bands']` on Oversold Low Volume: SPY close /
rolling-252d high - 1 in [-0.10, -0.03) -> 1.15x, else 1.0x (never a cut on
the at-highs side). daily_scan evaluates SPY at the signal close from the
cache; the engine replays from master_prices point-in-time. Inside the OLV
composite clip. Ship rule (already met on the study series, confirm in the
engine replay): clustered t >= 1.5 on 2010+ and >= 10 of 15 walk-forward
years better. Retire if the cell's avgR on the next 30 trades is below
OLV's unconditional mean. If WP9 ships first, re-test on hedged returns
before any move toward the fitted 1.5x.

### WP11 Pre-registrations (files first, then studies)

Write each protocol under `scratch/ultracode_research/` BEFORE running
anything, with the decision rule as stated in the plan:

- `ltt_band50_prereg_2026-09-02.md`: LT Trend ST OS `[[50, 999, 0.75]]`,
  plain table, PIT dial; adopt only if on the live PIT series extended by
  two NEW hi-dial episodes the [50, 999) cell's clustered t <= -2.0,
  survives drop-best-episode, episode signs >= 6 of 11; adds cell scored
  separately and exempt if >= 0.4R.
- `hedge_vix_gate_prereg` (inside WP9's file): adopt only if not worse than
  plain 50/45 across the next two OOS armed episodes.
- `ovs_cluster_tilt_prereg_2026-09-02.md`: >= 5 OVS signals staged -> 1.25x;
  episode-clustered t >= 2 with 2026 included, LOYO floor > +0.20, no
  worst-day deterioration beyond 0.5 pt, P1/P2 split reported.
- `b52_dial_exit_prereg_2026-09-02.md` (P4, exit rule): shorten 52wh holds
  while dial >= 50; PIT, episode-clustered, LOYO; ship only if armed-window
  PnL improves with clustered t >= 1.5.
- `family_band_hedged_retest_2026-09-02.md`: once WP9 has two live armed
  episodes, re-score the FAMILY4 0.25x / fear-off zero cell on beta-hedged
  trade returns; fallback if the 2026-08-05 review fails is
  `[[50, 999, 0.5]]`, not 1.0x.
- Index legs to futures (`index_legs_futures_design_2026-09-02.md`): SPY /
  QQQ / DIA / IWM legs of the dip-buy family in MES / MNQ / MYM / M2K at
  the same ATR risk; this is the precondition for GRM >= 2.5 and lifts the
  plain-PM wall from m ~1.6 to ~2.3. Design only in this brief.

### WP12 GRM step rule (standing)

`scripts/grm_frontier.py --walls` prints, quarterly and on any 20% NLV move:
(i) g(m) curvature at the current haircut; (ii) projected p99 and max
requirement days at plain PM on live NLV and on the base, with rules-based
3x and 30% concentration as sensitivities; (iii) k_999 x p99 open ATR risk
x m vs 15% NAV; and the drawdown disclosure table. Step to 2.25 on the first
of: NLV >= ~$840k; WP11's futures routing live; one clean WP1 trim through a
real cluster day. 2.5-3.0 only with the futures routing AND a measured live
keep >= 0.5 (WP4). GRM changes remain a human edit with the script output
in the commit message; never more than one step per quarter. The
per-strategy 250 cap stays fixed in effective bps until GRM exceeds 3.0.

### WP13 Sizing base (standing)

`ACCOUNT_VALUE` stays 750,000 while live primary NLV < 750,000 (a switch is
a 16% cut). When NLV > 750,000 at a quarter end: base = max(prior base,
0.5 x 750,000 + 0.5 x NLV), reset quarterly, never below the prior quarter.
Full compounding only if the measured live keep is >= 0.75. No downside
floor (the guard on live NLV already tightens with a drawdown). WP1 always
uses live NLV regardless of the base.

## Removed from the earlier brief

- WP3 (52wh 0.5x at >= 5 open): withdrawn (cuts a 1.71R cell).
- The P(DD > 15%) <= 5% GRM rule: replaced by WP12's three walls.
- The flat 50% shrink in the tilt: replaced by keep_s (WP6).
- P3 as written: absorbed into WP5 step 4 (one confirmation for Phase 1).
- P2 at the 65 line: replaced by WP11's LT Trend prereg at 50 on PIT; OLV
  and OVS exemptions confirmed on PIT.
