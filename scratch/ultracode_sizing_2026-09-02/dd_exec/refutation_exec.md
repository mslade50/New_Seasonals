# Refutation ledger: execution reality, live/engine parity, margin and liquidity

Reviewer lens 2 of 3 (email gap 1). Read-only pass, 2026-09-02. Repo code as of
c60fc3d4; OneDrive trading_ibkr as on disk. Numbers computed in
`dd_exec/exec_adv_cap_check.py` -> `exec_adv_cap_check.json` (ledger vintage
2026-09-02, flat $750k, master_prices 21d median dollar ADV).

Conventions: "scan" = daily_scan.py run_daily_scan sizing loop (steps 2b..2d at
lines 2900-2990, ADV cap 3031, ticker cap 3057, post-passes 5b 3227 / 5c 3282);
"staging" = OneDrive order_staging.py pull_and_stage_orders (OVS gate ~1000-1075,
gap derate ~1216-1240, P2 cap 1445-1478, per-strategy cap 1480-1500, positions
print 1532); "engine" = pages/strat_backtester.py process_signals_fast (pre-passes
905-1060, sizing 1290-1440, ticker cap 1883-1920, cap post-pass 2063-2090).

## Two facts that reshape several verdicts

F1. The 250 bps cap already binds on OVS almost every day. Ledger 2016+: OVS is
cap-bound on 97.1% of its signal-days (451 days), and 107% of its PnL sits on
bound days (unbound days are net losers). The step to GRM 1.875 therefore adds
ZERO staged risk to OVS on those days: placed risk is already several times the
cap, so the pro-rata scale just falls from s to s/1.25. Book-wide, PnL-weighted,
the GRM step is worth at most 1.18-1.19x (2016+: 1.182; 2025+: 1.176), not
1.25x; 35.5% of all strategy-days are bound now (2016+), rising only to 36.0%
at 1.875 because the unbound days are mostly far below the cap. Other bound
shares (2016+, PnL on bound days): 3x ETF Fade 22.5% of days / 57% of PnL;
Weak Close 11% / 44%; OLV 2.9% / 19%. The unconstrained_growth_01b "cap-aware"
study got eff mult 1.24 at m=1.25 because it used FILLED risk per day (p99 239
bps, max exactly 250) as the placed risk; placed risk on OVS days is 10-30
signals x 60 bps. The practitioner replay (grm1.875_capfixed, total 4.84M vs
4.00M = 1.21x) recovered the cap from Size_Mult and is closer to right. So the
"linear through 3x" claim is a filled-risk artefact; the practitioner's own row
says the step is ~1.21x and it degrades further at 2.25 (short_fade 82% of any
up-size is absorbed by the cap per flow_conditional cap_absorbs_share_of_upsize).

F2. The live ADV participation cap is a NO-OP today. daily_scan applies
`adv_share_cap` only for `_scan_source == 'Overflow'` and only when
`load_overflow_meta()` returns data, which it does only when
`OVERFLOW_UNIVERSE_ACTIVE=1` (overflow_universe.py:197 returns {} otherwise; the
sheets pull shows Addv_63d = nan on every 2026 row). The liquid tier has never
had one. Any "re-express the existing participation cap" wording in 1.2 is
re-expressing nothing; the rule has to be built from scratch on master_prices for
ALL tiers, and the liquid tier is where the historical extremes live (WCDS liquid
p99 11%, max 32% of ADV in unconstrained_growth_03).

## Phase 0

### 0.1 Margin-feasibility guard - STANDS-WEAKENED (as a projection), REFUTED (as a ruin control)

(a) Visibility: computable, but not where the brief puts it. order_staging IS the
9:31 chain (register_order_chain_task.ps1 fires it at 9:31); "before the 9:31
chain" would run pre-open with no session opens, before the OVS P1/P2/SKIP gate
and the gap derate that set today's staged notional. The guard must sit inside
pull_and_stage_orders AFTER `_GapMult`/`_PathLabel` (line ~1377) and before the
per-strategy cap (1480), i.e. exactly where `_PostOpenRisk` is built. Inputs:
live NLV is one `ib.accountSummary()` call on the already-open primary
connection (today only the PA is queried, `fetch_pa_account_value` 370-398);
marks via `ib.portfolio()` (order_staging currently uses `ib.positions()` at
1532, avgCost only). The DO `/book` snapshot carries NLV + marks but NO margin
fields (book_snapshot.py 105-115 collects NetLiquidation only; the 2026-07-01
futures design doc already flagged this).

(b) Strongest attack: the stylised Req_open is redundant and worse than what the
broker hands over for free. `accountSummary` tags `FullMaintMarginReq`,
`LookAheadMaintMarginReq`, `ExcessLiquidity` ARE the broker's requirement on the
open book, house uplifts and concentration add-ons included. On a day IBKR raises
house margin (the Nov-2020 +35% case in the plan's own ledger) the stylised table
reads 15% while the account is already at the wall. Use the broker number for
Req_open and stylise (or `whatIfOrder`) only the staged increment.

(c) "Never touch open positions" leaves the liquidation channel open. With a
flat 15% rate, projected requirement R/NLV maps to gross G = R/0.15. From the
70% trim line (G = 4.67x NAV) a book-beta move X breaches 100% when
0.70(1-X)/(1-4.67X) = 1, i.e. X = 7.6%. From the 85% alarm line (G = 5.67x),
X = 3.1%: a -3.1% day on an 85%-of-NLV open book is a 15:45 liquidation, and
the guard's only response at 85% is "no new entries". The 15:30 re-check alarms
at 90%; that is 15 minutes of hand-flattening on the worst day of the year.
Since 2016 the ledger has 24.7 days/yr where a -30% equity shock exceeds 30% of
NAV (growthmax_1). Either the guard trims open positions (a real partial-exit
path, MOC/DAY, the olv_book_cap.py machinery already does this for OLV) or the
70% line is a notional cap in disguise that still does not close the channel.

(d) Working limits are invisible to Req_proj. Persistent limits (OLV T+3, 52wh
and Sector BO 10-day windows, MWC T+2) from the previous 1-9 sessions are neither
"open positions" nor "today's staged entries". IBKR does not reserve margin for
resting orders, so the fills of yesterday's limits land on the same 15:45 clock
with no guard seeing them. Same blindness as the OLV ticker cap (CLAUDE.md
"KNOWN BOUND"). At GRM 1.875 that is up to 3 x OLV's daily staged notional.

(e) Engine: the brief's WP1 step 6 (`MARGIN_TRIM` row type) needs a projected
requirement series inside process_signals_fast (open positions x marks from
price_matrix, plus staged notional). Doable, but until it exists every guard day
is an engine/live divergence on exactly the days that set maxDD, and 1.10's
`relief_off` gate cannot be replayed at all.

What would change the verdict: Req_open from `FullMaintMarginReq`; a partial-exit
row type live and in the engine; working-limit notional counted in Req_proj.

### 0.2 IBKR what-if - STANDS (human step). Add: run the what-if with the hedge
(2.1) on, and ask for the account's leveraged-ETF short rate explicitly; the
Risk Navigator answer for a hypothetical book does not show house uplifts that
apply to THIS account category unless the book is entered as positions.

### 0.3 Headroom display - STANDS. Display only; the DO snapshot needs the margin
tags added to book_snapshot.py before "projected requirement/NLV" can be a live
number rather than a stylised one.

### 0.4 Fills harvest - STANDS-WEAKENED. The live-vs-ledger json already shows
the parity problem the harvest must measure: of 853 staged rows (Mar-Aug 2026),
FILLED-status rows are in the ledger only 69% of the time (21 of 68 disagree),
and 23% of ledger trades in the window were never staged live. Sheets
`Fill_Status` is blank on 681 of 882 rows. N >= 150 matched fills at the current
pace (~68 modelled fills in 5 months) is 2027.

### 0.5 Engine cost model - STANDS. One parity note: OVS tranche rows would be
charged the $1 commission floor twice unless costs are booked per order (the
engine books two rows per fill, `Tranche` near/far).

### 0.6 Haircut - STANDS (outside this lens).

## Phase 1

### 1.1 GRM 1.5 -> 1.875 - STANDS-WEAKENED

Turned on F1: the step is worth <= 1.18-1.21x of PnL, not 1.25x, and 0x for OVS
(13.8% of 2016+ PnL). Margin: max historical requirement day at 1.25x is 77% of
the $750k base under flat 15% (robust_bayes req_max 0.615 on 2013-11-04) and
91% of the $632k live NLV; the plan's 84% uses the 2016+ maximum (2023-02-03,
0.554). Under rules-based 3x margin the 2023-02-03 book is at 99% of base at
1.0x and 124% at 1.25x (robust_bayes rules3x). The step is only safe if 0.2
comes back at plain PM for the 3x shorts. Verdict changes to STANDS if the plan
restates the growth claim at ~1.2x and the what-if confirms the 45% rate.

### 1.2 Overflow longs excluded + participation rule - STANDS (wiring), STANDS-WEAKENED (rule)

(g) The three "new carriers" need no new wiring: all three consumers already read
the dict generically. daily_scan.build_effective_strategy_book 173-176 loops
`if s['name'] in OVERFLOW_RISK_OVERRIDES`; daily_portfolio_report 126-127 same;
strat_backtester 1307-1310 `OVERFLOW_RISK_OVERRIDES.get(strat_name)` for any
ticker outside `_LIQUID_SET` when `overflow_active`. Editing the dict is the
whole change (tests/test_grm_step.py should pin the four values). Two traps:
(i) the engine keys on ticker-not-in-liquid-set, not on scan tier, so a
strategy whose NATIVE universe strays outside LIQUID_PLUS_COMMODITIES would get
the override on its liquid pass - Weak Close and Sector BO both carry XLC, which
is not in the liquid set; harmless today (neither is in the dict) but a trap the
moment either is added. (ii) 1.3's tilt on OLV (1.17), LT Trend (1.04), St OS
(0.88), 52wh (0.70) either applies to the overflow rows (then "effective bps
unchanged" is false: OLV overflow 37.5 -> 43.9, 52wh 52.5 -> 36.75) or the
override path needs an explicit tilt exemption. The brief does not say which.

ADV rule at GRM 1.875, 2025-01..2026-08 ledger orders (tranches merged), 21d
median dollar ADV: nothing above 5% (hard refusal never fires); above 1%:
ATR Ext overflow 14.3% of orders (n=14, max 2.99% CSGS), OVS overflow 3.4% of
146 (max 2.83% WDS), LT Trend overflow 3.0% (max 3.58% DBA), 3x Bear Fade 5%
(EDZ 2.46%), all other liquid cells 0%. The 0.4% rule on LT Trend/St OS/WCDS/IOB
bites 3.6% of 83 orders. So the rule is cheap, but it binds on exactly the
short-fade overflow cells (ATR Ext, OVS) that 1.2 says "take the step", which
partly cancels the step it is written beside. Where it must live: scan step
between shares and the ticker cap (3031-3047), for BOTH tiers, keyed on
master_prices not overflow_meta; stamp participation; staging can only
re-enforce from the stamp. Engine: `processed_dict` has Volume, so a per-signal
`adv21` is a one-line addition next to `atr` in generate_candidates_fast.

### 1.3 Allocation tilt - STANDS-WEAKENED

Engine trap that breaks the 1.11 confirmation if the tilt is fed through
`risk_multipliers`: the cap post-pass scales the cap by the multiplier
(`cap_dollars = day_equity * _effective_cap * _strat_mult_cap`, strat_backtester
~2080). MonFri at 1.30 would get a 325 bps engine cap against a fixed 250 live
(order_staging 1486 uses `PER_STRAT_DAILY_CAP_BPS` only). Plan section 5 says
the confirmation is `process_signals_fast(risk_multipliers=...)`; the brief says
import-time `STRATEGY_BASE_TILT`. Only the import-time form is live-parity.
Composition: the cross-strategy clamp (20 nominal = 37.5 effective at 1.875) is
absolute and untilted, so a MonFri 1.3 row (52.5 x 1.25 x 1.3 = 85 bps) is cut
to 37.5 on any day WCDS fires on SPY/QQQ (WCDS universe includes both); intended
but worth stating. WCDS's in-code Sznl tier (1.5x / 0.66x, scan 2887, engine
1313) composes on top of 0.75 - the replays carried that basis mismatch.

### 1.4 OLV depth ladder + composite clip - STANDS-WEAKENED

(a) "Working OLV limits within T+3" at 4:47 AM: the scan can see them, but not
from where it looks today. `load_open_position_notionals` (1962-2025) reads the
`Portfolio` tab = daily_portfolio_report's ENGINE-modelled open positions
(yfinance bar-touch fills), not the broker book. Working entries would come from
`Trade_Signals_Log` rows (Strategy=OLV, Date in the last 3 sessions, Fill_Status
not FILLED/EXPIRED; verify_fills is yfinance-modelled, blank on 77% of rows), or,
since the 2026-08-28 local-primary cutover, from OneDrive `eq_placed_orders.json`
/ the live open orders. AM and PM bookends both append the same signal (0.3% of
rows are exact duplicates; the two bookends are distinct timestamps), so the
count must dedupe on (ticker, date) and must not count today's own PM-staged rows
as "working" in the AM re-run. Engine: candidates are processed in signal order
and fills are resolved by look-ahead, so at candidate k "working" = prior OLV
candidates within 3 sessions with `entry_date > signal_date` or no fill; the
ticker cap already uses the `entry_date <= signal_date` split (1898-1905).
Computable on both sides.

(b) The rule shipped is not the rule tested. within_strategy_adds counted
`n_open` = FILLED legs entered before the signal day; the refutation ledger
then "modified" it to filled + working. Working entries can add up to 3 days of
names, so the 3+ rung (full size) triggers earlier than in every number cited
(+$125k raw, +$16k equal risk). Direction is size-up on cluster days. Parity is
also only model-to-model: a depth of 3 "filled" legs is 3 modelled fills, and
the live-vs-ledger json puts modelled-vs-live fill agreement at 69%.

Composite clip: max product = tilt 1.17 x ladder 1.0 x pullback 1.15 x flow 1.2
= 1.61, so 1.5 binds only when all three up-levers coincide. Two definitional
gaps: the tilt is baked into `risk_bps` at import, so the clip needs the tilt
factor carried separately; and the earnings override (2d / 3b3b) REPLACES base
risk and composes only with `_recency_mult` and `_fbm`, so "product of all OLV
overlays" is undefined inside the override window unless the override is
rewritten to compose with depth, pullback and flow.

### 1.5 WCDS / LT Trend solo-adds - UNTESTABLE as written

The definition of "adds" decides the rule. The evidence is same-DAY cluster adds
(WCDS "same-day adds 0.73R vs solo 0.21R"). At 4:47 AM all of a day's WCDS
signals are staged together with no prior leg "open or working" (WCDS entry is a
single-day `Limit (Open +/- 0.25 ATR)`, LT Trend is persistent with hold 1, so
nothing works beyond T+1). Under the brief's wording ("entered with >= 1
same-strategy leg open or working") a 5-signal day is either 5 solo legs at 0.8x
(nothing open yet) or, if same-day staged rows count as working, 5 adds at 1.2x
and a lone signal is 0.8x - which is same_day_signal_derate inverted, and in the
engine the answer would depend on candidate order within the day unless computed
in a pre-pass like 3b4. Pick one, write it down, and re-score; the 250 cap
binding "7 to 9 days each" changes with the choice.

### 1.6 OVS extremity tier - STANDS (with two locations)

Live: `last_row['rank_ret_{2,5,10,21}d']` is what build_live_filters already
reads (daily_scan 1574-1583); one line in the OVS sizing branch, stamp the mean.
Engine: `signal_data` carries only `rank_ret_126d`/`252d` (generate_candidates
~735-740); the four short ranks must be added from `processed_dict`, and the
multiplier must ALSO be folded into the P2 pre-pass
(`_base_risk_p1 = starting_equity * _p1_bps/1e4 * _ovs_mult * _cyc_m`, ~990),
otherwise the engine's P2 aggregate scale is computed on un-derated risk while
staging computes it from the derated `Risk_Amt` (1445-1478). Effect claim is
overstated by F1: on 97% of OVS days the cap redistributes, so 0.7x on the
bottom cell is a reallocation of a fixed 250 bps toward the top cells, not a
"20% smaller OVS footprint"; the extremity study's "-$22k if not redeployed" is
the wrong counterfactual for a cap-bound strategy.

### 1.7 OVS path-2 cap 0.75 -> 1.0% - STANDS-WEAKENED

Staging applies the P2 cap BEFORE the per-strategy cap (1445 then 1480); the
engine mirrors that. On the 97% of OVS days at the 250 cap, a larger P2 pool
(112.5 -> 150 bps effective) mostly shifts the fixed 250 from P1 rows to P2
rows on mixed-gap days. The "+$20k/24y, worst extra day -$1.4k" was estimated
as if the two caps were independent. The engine confirmation (1.11) will price
it; the claimed number should not be cited until then.

### 1.8 Index clone clamps - STANDS

IOB SPY+QQQ 0.5x: a (strategy, date) count post-pass beside 5c, mirrored in the
engine's 3b4 pre-pass. Clamp extension: 5b (3227) and the engine pre-pass
(1039-1058) are generic over `CROSS_STRATEGY_OVERLAP_OVERRIDES`; Monthly Weak
Close trades SPY/QQQ directly while MonFri/IOB use ^GSPC/^NDX, and both sites
alias via SPOT_TO_TRADEABLE, so the new pairs collide correctly. No staging
change.

### 1.9 Flow-aware cap relief - REFUTED as written

The families in the brief are not the families that were tested.
flow_conditional_results.json `data.families`: St OS Sznl -> oversold_hold,
3x Bear ETF Overbot Fade -> dip_buy, 3x Leader Gap Fade -> short_fade. The brief:
St OS Sznl -> dip_buy, 3x Bear and 3x Leader -> a new bear_etf_fade family that
is excluded. Moving St OS Sznl (197-name universe) out of oversold_hold and into
dip_buy changes both count distributions; every threshold in the brief
(dip_buy >= 6, oversold_hold >= 7, short_fade >= 104) was calibrated on the
study's membership, and the study's own f5 tercile edges are 5 / 8 / 97
(hi = >= 6 / >= 9 / >= 98), not 6 / 7 / 104. Until the brief's families are
re-scored the rule has no evidence behind it.

Beyond membership, three parity holes:
(a) The count needs every family member's mask for the trailing 5 sessions
BEFORE any signal is sized. daily_scan sizes inline inside the strategy x ticker
loop (2860+), so this is a new pre-pass over all (strategy, ticker) masks, not
"a step after 2b". `check_signal_live` already evaluates the full mask per
ticker (filters.py 769-773, last row of `live_signal_mask`), so the cost is
retention, not computation. (b) Live masks strip T+1 gates
(`_LIVE_STRIP_KEYS`), engine candidates include them (ATR Ext's mask has the
NextOpen gate, MonFri's Friday gap kill). Live counts >= engine counts for
short_fade and dip_buy; hi-flow triggers slightly more often live. (c) The count
base is the universe: the ledger and today's live overflow tier are both the
static CSV_UNIVERSE minus liquid, but the moment OVERFLOW_UNIVERSE_ACTIVE=1
turns on the 1270-name dynamic screen the OVS candidate count jumps and the
short_fade threshold is stale. Staging cannot compute masks: the relief must be
a per-row stamp (`Cap_Relief_Bps`) that order_staging reads at 1490, with the
guard's `relief_off` overriding the stamp.

Cap absorption (flow_conditional `cap_absorbs_share_of_upsize`): short_fade 0.82,
oversold_hold 0.25, dip_buy 0.15. With the relief the short_fade up-size is
still mostly a cap question, which is why the relief and the up-size cannot be
scored separately.

### 1.10 Family flow up-size - REFUTED as written (same membership defect), plus

The up-size is baked into the scanner's `Shares`/`Risk_Amt` at 4:47 AM, but
`relief_off` is decided in order_staging at 9:31. Turning it "off for the day"
means order_staging must UNDO a multiplier it never applied: a `Flow_Mult` stamp
and a divide-through before `_PostOpenRisk`, the same shape as `_GapMult`. New
path, and the engine cannot replay it without WP1 step 6 (see 0.1(e)). Ordering:
the brief puts the up-size after 2b and before 2c, which leaves it inside the
earnings override's clobber zone (2d replaces base and keeps only recency and
frag); either compose it explicitly or accept that pre-earnings OLV/St OS
signals never see it.

### 1.11 Engine confirmation gate - STANDS-WEAKENED

Two things make the gate unable to say what it is asked to say. (i) If run with
`risk_multipliers`, the engine cap scales with the tilt (1.3 trap above) and the
run is not live-parity. (ii) The composition order in plan section 10 ("... per-
strategy cap -> cross-strategy clamp -> same-day derate -> margin guard") is the
opposite of both code sites: 5b clamp and 5c derate run in daily_scan BEFORE
staging's cap; engine 3b3c/3b4 run before the cap post-pass. Cap and derate do
not commute (derate-then-cap gives cap/placed x r_i; cap-then-derate gives
m x cap/placed x r_i). The gate must state which order it replays; the code
order is the one live uses. Also the OVS P2 pre-pass must carry every new OVS
multiplier (1.6, any flow up-size) or the engine's P2 cap diverges from staging.

## Phase 2

### 2.1 Dial-armed beta hedge - UNTESTABLE (permissions and order type), STANDS (arithmetic)

Futures on the primary: execute_order.py has a FUT branch (tick snap, whatIf
init-margin gate, `LIVE_MAX_FUT_CONTRACTS` default 3, CME/CBOT/NYMEX/COMEX
allowlist), `futures_sizing.py`/`futures_front.py` exist, `contract_reference`
lists ES (50x) and MES (5x). `exec_agent_seen.jsonl` holds zero FUT commands and
no log shows a futures fill: trading permissions on the primary have never been
exercised through this stack. The 2026-07-01 design doc's open question ("does a
dedicated futures username/port exist") is still open. Size: beta 0.45 (mean
armed beta in practitioner_02) x $750k = $337k = ~10 MES at ES ~6,700, ~$25k
overnight margin ($2,455/MES per CME 2026 schedule) = ~4% of live NLV; at beta
1.0, 22 MES / ~$55k / 8.7%. The default 3-contract cap in execute_order makes
`hedge_moo.py` a separate runner, as the brief says. Margin is additive: IBKR
margins CME index futures under SPAN outside the PM stress set, so the short ES
gives no relief against the long SPY legs; the plan states this and 0.2 should
confirm it with the hedge in the what-if. "Native MOC entry": MOC is an
equities order type; CME Globex has no MOC for ES/MES (the exchange mechanism is
TAS, and IBKR's futures order-handling pages do not list MOC for Globex), so the
2026-08-21 encoding rule does not transfer. The paper period must place a real
MES order at 15:59 ET and record what IBKR accepts (MKT at 15:59 with
`outsideRth=True` is the likely form). Roll: the first arming after 2026-09-18
straddles the Sep/Dec roll; the panel's roll-off list is stock time-exits, not
futures expiries.

### 2.2 OLV market-pullback tilt - STANDS

SPY close / rolling-252d high is a ratio, scale-invariant under the adjusted
cache, readable at the scan (SPY is in master_prices) and in the engine
(processed_dict). Two placements to fix: it must sit before 2d or be composed
into the override, and the AM bookend scores it on the settled bar while the PM
bookend scores the provisional one (same as every other close-keyed input).

## WORTH DISCUSSING (owner decisions)

1. The GRM step buys ~18-21%, not 25%, and buys OVS nothing. OVS is cap-bound
   97% of days with 107% of its PnL there; the "linear through 3x" claim came
   from filled risk, not placed risk. Decide whether the step is still worth its
   margin cost when 14% of the book's PnL is untouched by it, and whether the
   honest lever for OVS is the cap (which the plan freezes to 250 until 3.0).

2. The guard as specified does not close the ruin channel it was written for.
   From an 85%-of-NLV open book a -3.1% book-beta day breaches 100% at 15:45,
   and the guard's answer at 85% is "no new entries"; working limits from prior
   sessions are invisible to Req_proj. Either accept a partial-exit path on open
   positions (olv_book_cap.py already does this for OLV, MOC/DAY, owner
   clientIds) or accept that the guard is a stage-day trimmer and the 15:30
   alarm is the real control. Also: use `FullMaintMarginReq` from the broker for
   Req_open rather than a stylised table that cannot see house uplifts.

3. Flow rules (1.9/1.10) cannot ship on the current evidence: the brief's family
   membership (St OS Sznl, 3x Bear, 3x Leader) differs from the study's, the
   thresholds differ from the study's tercile edges (6/7/104 vs 6/9/98), and the
   relief_off gate requires staging to reverse a scanner-applied multiplier that
   the engine cannot replay. Re-score on the brief's families or adopt the
   study's; and decide whether the dynamic overflow universe (1270 names,
   currently OFF) is switched on before or after the thresholds are set, because
   it moves the short_fade count base.

4. 1.4 and 1.5 ship rules that were not the rules measured: depth counting
   working limits (untested, size-up direction) and an "adds" definition that
   is either a same-day count rule or nothing. Both need one sentence of
   definition and a re-run before 1.11, not after.

5. The participation rule is new, not a re-expression: today's ADV cap is a
   no-op (gate OFF, meta {}), liquid tier never had one. At 1.875 it bites only
   ATR Ext overflow (14% of orders > 1%) and a handful of OVS/LT Trend overflow
   orders (max 3.6% of ADV); nothing reaches the 5% refusal. Decide whether a
   rule that mostly trims the overflow short fades is meant to coexist with
   "overflow shorts take the step".

6. Hedge readiness is a permissions and order-type question before it is a
   sizing question: no futures order has ever gone through this stack, and MOC
   does not exist on Globex. The paper episode should start with one MES
   round-trip at the close, not with the beta arithmetic.
