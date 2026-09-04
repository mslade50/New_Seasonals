# Portfolio Research Integration — 2026-07-02

Integrates ten adversarially verified research tracks: five fragility deep-dives
(family-throttle, proxy-falsification, dial-design, ovs-exemption, inversion-2026)
and five sleeve studies (trend-following, momentum-rotation, factor-sleeves,
crisis-alpha, book-gaps). Every decisive number cited below reproduced under
independent recompute unless marked otherwise. Detail files and verification
sections live in this directory.

Shared caveat that rides with everything: the fragility history is a
current-vintage reconstruction, and the composite edge weights carry calibration
lookahead (full-sample event study in `signal_horizon_stats.json`). Component
states are point-in-time; the weights are not. This affects all fragility
results equally and is the top item on the open-questions list.

---

## 1. FRAGILITY SIZING — final recommended design

### The design

**Basis (unchanged, now positively validated):** 63d fragility dial, 10-day
trailing MA, evaluated as of signal date. Threshold 50, plain, no hysteresis,
no multi-dial combination. Dial-design confirmed every dimension independently:
the 5d dial is dead at every threshold and window (best clustered t = -0.88,
CONFIRMED), the raw unsmoothed score has no signal (t = +0.72 exactly
reproduced), thr 50 sits mid-plateau (40-55 all t -1.9 to -3.1), MA 10 sits
mid-plateau (5-15 all t -2.2 to -3.3), and the 10d MA already limits regime
flips to 2.6/yr with zero episodes of 5 days or less in a decade, so hysteresis
buys nothing. Proxy-falsification confirmed the composite is not replaceable by
any simple SPY/VIX proxy or OR-combination at matched N (best solo alternative,
low 3y realized-vol percentile, opens less than half the gap). Keep the
composite.

**Schedule, per scope:**

| Scope | Rule | Evidence status |
|---|---|---|
| Low-frag boost (all strategies) | **Kill the 1.25x boost.** 1.0x everywhere below the throttle zone. | Unanimous, all five fragility tracks. Established: score<25 avgR ~= 25-50 avgR, no boost case. |
| FAMILY4 = {Weak Close Decent Sznls, SPY QQQ MonFri Reversion, Monday Dip, Indices Oversold Bounce} | **0.25x when 63d MA10 >= 50.** 1.0x below. | Family >=50 avgR -0.283 (N=74) vs +0.607 below (N=379), clustered t=-2.30 p=0.032, CONFIRMED exactly; survives every single-year exclusion incl. COVID (p 0.03-0.09). |
| Rest of non-OVS book | **1.0x at all scores. No taper.** | Rest at >=50: +0.394 (N=168) vs +0.660 below, p=0.47 after clustering, CONFIRMED. No detectable degradation. |
| OVS | **Keep exempt at >= 44 (affirmatively, not just unproven). Add 0.75x when score in [21,44), stacking with the midterm 0.75x to 0.5625x where both apply.** | Mid-band dip: trade-weighted block bootstrap z=-3.0, LOYO worst -2.14, survives ex-midterm (z=-2.33); discounted to ~2 sigma effective (edge-sensitive to ~1.8, null under equal-weight monthly). 55+ recovery is real and NOT 2022 (2022 is the band's worst year; ex-2022 +0.658, N=36). All CONFIRMED. |

**Monitored hypotheses (logged, not sized):**
- frag >= 50 AND rv21 3y-percentile **<= ~25%** (ultra-calm quintile): the toxic
  cell (N=43, avgR -0.56, win 30%, essentially two episodes 2021+2024). Note the
  direction: the proxy-falsification report's own text had this INVERTED
  (verification refuted the "rv-elevated" labeling); the correct condition is
  deepest complacency, realized vol in its calmest quintile while frag is high.
  Do not encode the >75% version anywhere.
- 21d dial / 21d MA / thr ~36 as a non-sizing shadow series (best LOYO of
  anything tested, [-4.80, -3.08] on recompute, but 85% redundant with the 63d
  rule and window-scan selected).
- Re-examine the family throttle after ~20 more family trades accrue at >= 50.

### What changed vs the pending rec, and why

Pending rec was: kill the 1.25x boost, 1.0x through 50, book-wide taper to 0.5x
by 60, OVS exempt. Three changes:

1. **The book-wide taper is replaced by the family-only 0.25x cut.** This is
   the one genuine conflict between tracks and I am siding with family-throttle
   deliberately. Proxy-falsification, inversion-2026, crisis-alpha,
   trend-following, momentum-rotation and factor-sleeves all say "proceed with
   the pending taper", but none of them studied the schedule design; they
   inherited it as background while answering different questions (is the
   composite a proxy, is 2026 a refutation, should a sleeve replace it). The
   family-throttle track is the only one that ran the design comparison, and
   its numbers reproduced exactly: the established book-wide >=50 degradation
   is mostly a family effect (family carries 62% of the high-frag R shortfall
   with 31% of the trades); in replay the family-only 0.25x is the only design
   that RAISES total R (+15.7R to 642.1) while improving avgR per unit risk
   (0.543 to 0.585), whereas the book-wide taper costs -11.4R, including -5.3R
   in the inverted 2026, because it throttles strategies that are fine or
   better at high frag (LT Trend +0.31/78, OLV +0.35/70, St OS Sznl +0.65/12,
   ATR Ext Gap Up +1.98/5, all reproduced). The inversion-2026 anatomy
   independently supports narrowing: the taper's realized cost concentrates in
   post-break dip-buys (historically +0.417, N=60, the profitable side of the
   high-frag sample), and the family, whose damage is the pre-break
   configuration, placed zero trades in the 2026 episode. The two tracks
   converge on the same conclusion from different directions.

   The honest asterisk, stated in full: the family-vs-rest interaction never
   reaches significance (t=-1.42 p=0.165, only 16-22 high-frag months of
   power); the three named strategies were first noticed via their returns
   (mitigated by Indices Oversold Bounce, admitted purely on mechanism,
   reproducing the edge-disappears pattern at +0.69 -> +0.09, though it decays
   to zero rather than flipping negative like the trio); 2021 supplies 31 of
   the 74 high-frag family trades (ex-2021 survives at p~0.06); and the 0.25x
   level is mildly in-sample (any deep cut works because the pocket EV is
   negative, -20.9R over 74 trades). This is a portfolio-construction judgment
   with consistent supporting evidence, not a proven interaction. The fallback
   if the judgment is wrong: the rest of the book at 1.0x is exactly what the
   data says (p=0.47), and the design self-corrects cheaply because the family
   is only ~7 high-frag trades/yr.

2. **OVS gets a mid-band 0.75x tilt instead of blanket exemption.** New
   finding, verified end to end (including a 100% Size_Mult path-decode
   cross-check). The evidence grade (~2 sigma effective after the disclosed
   discounts) buys 0.75x under the book's shrunk-Kelly precedent (1.5 sigma ->
   0.75x for midterm), not the 0.5x that the headline z=-3.0 would suggest.
   Cost if pure noise: ~7R/decade. The top-end exemption is strengthened from
   "unproven" to "affirmatively correct": applying the pending taper to OVS
   would have taxed its second-best regime (+0.48 avgR at 55+, broad-based
   across 2021/2024/2026, with 2022 the band's worst year).

3. **The taper shape itself is retired with the book-wide scope.** Dial-design
   noted the 0.5x-by-60 endpoint rested on ~110 trades above 60; the family
   cut replaces a two-parameter ramp on thin data with a single threshold on
   the strongest cell.

What did NOT change: the dial, the smoothing, the threshold, the
kill-the-boost decision, and the OVS top-end exemption all survive intact and
are now positively validated rather than assumed.

### What fragility sizing does not fix (so nobody expects it to)

Two verified negative results bound the mandate. First, the dial does not flag
the book's worst dollar months: high-frag months average +$14,462/mo, 75%
positive (N=16, CONFIRMED), while the worst-decile months sit at median frag 19
vs base 20.6. The frag problem is per-trade avgR dilution, not monthly losses.
Second, the worst realized drawdown in the ledger (June 2026, -22.9R, 39
trades) happened entirely below frag 50 (avg 28.9, max 44.3, CONFIRMED): one
correlated OLV energy cluster (-20.4R, OXY x7, USO x6, PBR, LYB, DBC...) as oil
broke while SPX sat calm at highs. No SPX-level dial has jurisdiction there.
The binding risk-control gap is an OLV same-sector/same-direction concentration
cap, which outranks every throttle refinement on the roadmap below.

---

## 2. NEW SLEEVES — ranked against the book-gaps rubric

The rubric (book-gaps §5, verified): worst-month profile 30%, standalone
quality after costs 25% (Sharpe >= 0.5), correlation 20% (|rho| <= 0.25 to book,
own SPY beta <= ~0.3), capacity/execution 15%, dead-zone fill 10%, and a
pass/fail materiality gate of >= $25-30k/yr net (3.3-4.0% of the $750k NAV; the
"2% of NAV" label in the original rubric was a verified mislabel). The
re-weighted target regime matters: demand performance in the 29 worst-decile
months and 31 SPY <= -4% months, give little credit for "works at frag 55+"
alone, since the book already makes +$14.5k/mo there and the sizing design in
§1 handles the per-trade problem. Marginal-Sharpe hurdle: a sleeve improves the
book iff S_sleeve > rho x 2.16.

### #1. Multi-asset trend following — PILOT SMALL

The only proposal that clears standalone quality, correlation, and materiality
simultaneously. Ex-bonds 12-ETF combo (12-1 momentum AND 10m MA, long/flat,
inverse-vol, 5 bps/side): full-universe primary spec Sharpe 0.86 / CAGR 5.2% /
maxDD -4.5% over 303 months (CONFIRMED to the second decimal, excess t=4.33);
ex-bonds variant Sharpe 0.84/0.90/0.93 across sub-periods, so the edge does not
rest on the dead bond tailwind. Corr to book +0.117 (N=282, exact); in the 66
losing book months the sleeve made +0.21%/mo with 65% hit; combined monthly
Sharpe 2.21 -> 2.44 at 1x (CONFIRMED). At 0.5-1.0x NAV that is roughly
$20-50k/yr, clearing the materiality gate at the 1.0x end. Marginal-Sharpe
condition: 0.86 > 0.117 x 2.16 = 0.25, passed with room.

Rubric failures, stated plainly: it does not carry the knife-catch profile
(2020 Feb-Mar -3.2%; 2015-08 down alongside the book; positive in only 6 of the
book's 12 worst months) and it LOSES money in concurrent high-fragility months
(-0.23%/mo, N=16, still negative at t+1; the verifier's episode-clustered test
made this stronger, t=-3.55 p=0.007 across 7 episodes). So it is rejected as
the fragility fix and admitted purely as strategic ballast: dollars on idle
capital (avg open risk is 1.5% of NAV, 17% of days have zero open trades), real
crisis alpha in extended bears (2008 +8.0%, 2022 +0.8%, both exact), and it is
the only candidate whose long/flat structure de-risks itself. Long/short is
dead (Sharpe 0.15-0.32 depending on convention); skip it. Largest unremovable
bias: universe hindsight (2026-chosen 16-ETF menu), so 0.86 is an upper bound.
Whether it fills the Jul-Sep/midterm dead zones was not tested; it is testable
from `tf_monthly_series.parquet` in an afternoon and should be part of the
pilot gate.

Missing data/infra: none for the ex-bonds variant (all tickers + ^IRX already
in master_prices and maintained nightly; BIL backfill optional for the cash
leg). First implementation step in this repo: a `scripts/monthly_trend_scan.py`
that computes month-end signals from master_prices and stages 2-4 MOC (or
next-open `TIF=OPG`) orders to a new `Trend` Sheets tab, mirroring the Seasonal
pipeline; run it from a monthly GHA cron on the last trading day. Do not put it
in STRATEGY_BOOK (it is not a signal-scan strategy; it is a rebalance basket).
Execute on schedule: a full-month delay drops Sharpe to 0.55 (exact).

### #2. Crisis alpha — PARK the tactical VXX-proxy; REJECT puts outright

The tactical gated VXX-proxy (5% NAV, on at frag63-MA10 >= 55, off below 50,
11 round trips in 10 years) is the one structure with the right shape: +$39,081
verified to the dollar, 7/11 episodes paid, LOEO-stable (worst +$22.6k
ex-COVID), lag/hysteresis-insensitive, -0.14 corr to book dollar PnL, ~zero
calm drag. But it fails the materiality gate by an order of magnitude
(~$3.9k/yr) and is statistically indistinguishable from zero (t=+1.24 p=0.22,
N=119, exact), on a gate that is maximally exposed to the calibration-lookahead
caveat. It is cheap optionality, not a sleeve. Park it: revisit only after the
PIT weight re-estimation (roadmap step 5) removes the lookahead objection, and
never let it justify keeping size on elsewhere.

Puts are rejected with prejudice, and this includes superseding the in-repo
prior art: `tests/backtest_put_hedge.py`'s "net positive at all thresholds"
does not survive adding skew (+4 vol pts per 10% OTM) and haircuts. Always-on
3M 5%-OTM burns -7%/yr of NAV (verified -7.18%); gated at thr 55 it is
breakeven (+$9.4k/10y) and flips negative dropping any of three episodes;
gated at 50 it is -$36.7k outright. Mark notes.md's put-hedge table superseded.

The structural disqualifier applies to all crisis alpha here: the dial is a
pre-correction detector, not a crash detector. It missed Volmageddon (frag max
35.6), exited 2020-03-02 before the COVID crash proper, read max 4.9 through
the entire 2022 bear, and was on in only 2-3 of the book's 12 worst months
(all exact). And the book's realized curve has no crisis to insure: maxDD
-$62,073 against +$2.35M PnL since 2016 (exact).

### #3. Factor sleeves — REJECT

No static ETF tilt beats SPY on the honest common window (SPY 0.99 vs USMV
0.93, MQUV 0.98; SPHQ 1.06 and MTUM 1.03 inside noise, all CONFIRMED). Every
long-only tilt loses money in high-fragility months (USMV -0.66%/mo, MQUV
-1.19%/mo vs book +1.93%, N=16, exact), and every 0.5x-NAV addition lowers the
combined Sharpe and deepens maxDD (dilution slightly LARGER on the verifier's
clean window). Fragility-timed SPY->USMV is nothing (+4-5% cumulative per
decade, episode-clustered p=0.28-0.71, sign flips ex-2025). The SPY->BIL
rotation that looked alive is not a factor result: 73% of its gain is the
single COVID episode, insignificant under both clustering conventions
(p=0.13-0.25), and it duplicates the sizing throttle. Single-stock factor
screens: do not buy PIT fundamentals data to re-ask a question the clean ETF
version already answered negatively. Nothing to build. Park idle cash in BIL
as cash management if wanted; that is an ops decision, not a sleeve.

### #4. Momentum rotation — REJECT

Sector rotation is SPY beta with 25%/mo turnover (net CAGR 11.45% vs SPY
11.68%, corr 0.86, alpha never significant and decaying to +0.85%/yr t=0.39
since 2016, all CONFIRMED). Country rotation is strictly worse than SPY
(Sharpe 0.58, maxDD -66.8%, exact) on a 9-10 name universe, flattered by the
hindsight RSX drop and still failing. The single-stock book's +5%/yr "alpha"
is unratable: master_prices is today's membership (8 of 1,114 tickers
delisted, verified literally), and the recent-period concentration of the
alpha is the bias signature. Decisively anti-mandate: every variant loses
-0.4% to -1.8%/mo in the 16 high-frag months and about -1% to -1.5% in the
book's 12 worst months, with corr to book no better than plain SPY (+0.15 to
+0.24). Nothing to build, and the missing data (point-in-time constituents +
delisted histories) is not worth buying for this.

---

## 3. SEQUENCED ROADMAP — next 1-6 months, smallest reversible first

Each step names its validation gate. Nothing scales past its gate.

**Step 1 (week 1): kill the 1.25x low-frag boost.**
Unanimous across tracks, zero evidence of a boost case, pure removal.
Touch the aligned sizing sites (strategy_config / strat_backtester /
daily_scan / order_staging). Gate: replay parity, the ledger rebuilt with
boost off matches the research replay's below-50 cells; one scan cycle
(AM+PM) produces sane Sizing notes.

**Step 2 (weeks 1-3): OLV same-sector/same-direction concentration cap —
study, then ship.**
This outranks all throttle work: June 2026 was -20.4R of one oil bet
re-signaled ~30 times at low fragility, the worst DD in the ledger, and no
fragility design touches it (verified: zero of the 39 June trades at >= 50).
Design study first (cap definition: max N open same-sector-same-direction OLV
positions, or aggregate risk bps per sector per direction; needs a sector map
for the overflow universe). Gate: LOYO replay on the full ledger showing the
cap removes most of the June-2026-type cluster damage without materially
taxing OLV's +0.6R baseline; monthly-clustered before/after. Ship into
daily_scan (skip/queue signals over the cap) and daily_portfolio_report.

**Step 3 (weeks 2-4): ship the family throttle + OVS mid-band tilt.**
FAMILY4 0.25x at 63d-MA10 >= 50; OVS 0.75x in [21,44) stacking with midterm;
rest of book 1.0x. Implement as a generic `frag_risk_bands` execution field in
strategy_config (mirroring `cycle_risk_mults`), read by strat_backtester
sizing, daily_scan (stamp the mult into Sizing notes), and order_staging.
Gate: the four-site alignment test (change the field once, all four surfaces
move); replay parity with `verify_family-throttle.py` cells; then a standing
monitor that logs every family trade signed at >= 50 (expect ~7/yr) with a
scheduled re-examination at +20 trades. Rollback is a one-line config
deletion.

**Step 4 (month 2): log the monitored hypotheses.**
Add to the daily risk report or a small JSON monitor: (a) frag >= 50 AND rv21
3y-pctile <= 25% (corrected direction) as a flagged-not-sized kill-switch
candidate, (b) the 21d/21MA/36 shadow series divergence from the 63d rule.
Gate: none needed, monitoring only, but write the corrected rv direction into
the code comment so the inverted version can never be shipped by accident.

**Step 5 (months 2-3): point-in-time re-estimation of the composite edge
weights.**
The single biggest residual caveat on everything above. Re-derive
`signal_horizon_stats.json` on expanding windows (weights as of date t use
only data through t), regenerate the fragility history, and re-run the §1
headline cells (family -0.283/74, OVS mid-band z, 63d>=50 t=-2.75). Gate: if
the family and OVS effects survive on the PIT-weighted series at >= ~1.5
sigma, the designs stand; if they collapse, unwind step 3 (one-line rollback)
and revert to no throttle while the evidence is re-graded. This also unblocks
any future reconsideration of the VXX-proxy attachment.

**Step 6 (months 3-4): phase-gate LOYO study (research only).**
Test applying the family throttle only while SPY is within ~2% of its 52w
high, releasing once >3% off. Historical accounting flipped the book-wide
taper's cost from -11.4R to +10.8R (verified), but the split was formulated
after seeing 2026 and the clustered evidence is p=0.24. Under the family-only
design the stakes are smaller (the family cut cost just -1.1R in 2026), so
this is a refinement, not a fix. Gate: LOYO with monthly clustering on the
family subset; adopt only if the pre-break concentration of family damage
(pre-break -0.42, N=40 for the trio) holds at p < ~0.05 across drop-years.

**Step 7 (months 3-6): trend-following pilot at 0.5x NAV.**
Ex-bonds 12-ETF combo long/flat, monthly basket via a new `Trend` tab +
monthly GHA cron. Before real orders: (a) recompute with next-open execution
at daily granularity (the same-close convention was verified by neither side;
the 0.55 full-month-delay bound brackets it but the true number is unknown),
(b) test the Jul-Sep/midterm dead-zone fill from the saved series. Gate to go
live small: next-open Sharpe within ~0.1 of same-close. Gate to scale to 1.0x
after 2 quarters live: realized fills track the model within costs, no margin
competition observed on OVS spike days, and the combined monthly series
behaves (corr to book stays under ~0.25).

Explicitly NOT on the roadmap: any put structure, any factor sleeve, any
momentum rotation, any always-on long vol, book-wide fragility taper, and a
hysteresis band (cosmetic, verified).

---

## 4. OPEN QUESTIONS

**Testable now:**
1. Do the fragility effects survive point-in-time weight calibration? (Step 5.
   The decisive unknown; every fragility result inherits it.)
2. Does the family damage concentrate pre-break strongly enough to phase-gate?
   (Step 6; current evidence p=0.24, suggestive.)
3. What is trend's true next-open execution cost, and does it fill the
   Jul-Sep/midterm dead zones? (Step 7 pre-work; an afternoon each.)
4. What cap form best neutralizes OLV sector clusters without taxing its
   baseline? (Step 2 study.)
5. OVS P2 live-vs-backtest divergence: the ovs-exemption track added real
   evidence for the live retirement (P2 has no calm-band edge, -0.036 over 81
   trades where OVS is best, and no fragility structure). Decide: flip the
   ledger engine to `ovs_p1_only` or re-enable P2 live. Currently the site and
   reports model a scheme that live does not trade.
6. How much of the overflow single-stock PnL (52% of book PnL, survivorship-
   inflated) is real? Partially testable with a delisted-names dataset;
   matters for the materiality of everything else.

**Not testable except with time:**
7. Is the family-vs-rest interaction real? The cell accrues ~7 trades/yr at
   >= 50; the +20-trade re-examination (roughly 2029) is the honest horizon.
   Until then this is a judgment call, correctly labeled.
8. Is the frag>=50/rv-ultra-calm toxic cell (two episodes) a real interaction
   or an artifact? Monitored, next fragility episode decides.
9. Base-rate uncertainty on inversions: 2 of 7 measurable years inverted, 13
   distinct >=50 episodes in a decade, effective N ~2. No amount of recompute
   fixes this; it is why every fragility multiplier here is partial (0.25x on
   a small family, 0.75x on a band) rather than aggressive.
10. Whether the knife-catch archetype (2015-08, 2012-05, 2008-01, 2024-04:
    fast 4-6% corrections from calm) is hedgeable at all at this account size.
    Every candidate examined fails it: trend is too slow, puts bleed, the VIX
    gate fires on a different signature, factors and rotation are long beta.
    The current answer is the OLV cap plus the family throttle plus accepting
    the residual; a genuinely fast convex overlay remains an open design
    problem with no funded candidate.
