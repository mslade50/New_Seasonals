# Family-Level Fragility Throttle — Analysis (track: family-throttle)

Run date: 2026-07-02. Scripts: `scratch/ultracode_research/family_throttle.py`,
`scratch/ultracode_research/family_throttle_dd.py`.

Data: `data/backtest_trades_full.parquet` joined as-of signal date to the live
sizing basis (`data/rd2_fragility.parquet` col `63d`, `.rolling(10, min_periods=1).mean()`).
Window 2016-08-01 .. 2026-06-30. OVS excluded throughout (exempt live).
Non-OVS N = 1153 trades. All significance tests are monthly-clustered Welch t
(trade R averaged within signal-month, then t across months).

## 1. Family definition — the ex-ante argument

Mechanism, not returns: **short-horizon (2-day hold) LONG mean-reversion in
index / broad-market ETFs**. These strategies buy a falling broad-market
instrument and need the dip to bounce within ~2 sessions. At high fragility the
composite is literally built from signals that flag stored downside energy
(vol compression, complacency, credit divergence), so a 2-day index dip-buy is
the single most regime-exposed structure in the book: it is long beta crash
risk exactly when the dial says crash risk is elevated. Single-stock dip-buys
(OLV, LT Trend ST OS, St OS Sznl) revert on idiosyncratic flows and hold
longer; shorts benefit from fragility.

| Strategy | Instruments | Direction | Median hold | In family? |
|---|---|---|---|---|
| Weak Close Decent Sznls | sector/index ETFs | Long | 2d | Yes (named in task) |
| SPY QQQ MonFri Reversion | SPY, QQQ | Long | 2d | Yes (named in task) |
| Monday Dip | IWM, DIA, SMH | Long | 2d | Yes (named in task) |
| **Indices Oversold Bounce** | ^GSPC, ^NDX | Long | 2d | **Yes — by mechanism.** Identical structure: long index, 2-day hold, buys oversold. Excluding it would be indefensible; it differs from the named three in no mechanistic dimension. |
| **3x ETF Overbot Fade** | leveraged 3x ETFs | **Short** | 2.5d | **No — by mechanism.** It fades strength (short overbought), the opposite exposure: at high fragility a short in an overextended levered ETF is a tail hedge, not a falling-knife catch. (Empirically moot: N=1 at frag>=50.) |

- **FAMILY4** (primary) = Weak Close + MonFri + Monday Dip + Indices Oversold Bounce.
- **CORE3** = the three named strategies (shown as sensitivity).

**Outcome-picking audit.** CORE3 were originally flagged in the prior study *by*
their 55+ band returns, so a CORE3-only family is partially outcome-selected.
The mitigant is Indices Oversold Bounce: it was added here purely by mechanism
(its high-frag returns were not part of the established findings), and it shows
the same degradation — avgR **+0.693 (N=107) below 50 vs +0.087 (N=21) at >=50**.
A strategy admitted to the family on mechanism alone reproducing the pattern is
the closest thing to an out-of-definition confirmation this dataset can give.

## 2. The family effect

Per-strategy avgR (N) by frag band on the live basis:

| Strategy | <50 | >=50 | 55+ |
|---|---|---|---|
| Weak Close Decent Sznls | +0.596 (131) | -0.589 (20) | -0.909 (14) |
| SPY QQQ MonFri Reversion | +0.541 (106) | -0.221 (24) | -0.441 (15) |
| Monday Dip | +0.584 (35) | -0.630 (9) | -0.878 (7) |
| Indices Oversold Bounce | +0.693 (107) | +0.087 (21) | +0.176 (19) |
| — rest of book — | | | |
| Oversold Low Volume | +0.677 (216) | +0.353 (70) | +0.326 (39) |
| LT Trend ST OS | +0.305 (76) | +0.312 (78) | +0.292 (65) |
| St OS Sznl | +0.478 (20) | +0.649 (12) | +1.309 (7) |
| ATR Extended Gap Up | +0.934 (43) | +1.984 (5) | +2.369 (4) |
| Sector BO | +1.364 (29) | -1.031 (2) | — |
| 3x ETF Overbot Fade | +0.890 (68) | +1.479 (1) | +1.479 (1) |
| 52wh Breakout | +0.398 (80) | — | — |

Aggregates at frag>=50 (74 family / 168 rest trades, 18 / 22 months):

| | frag <50 | frag >=50 | >=50 vs <50, clustered |
|---|---|---|---|
| FAMILY4 | +0.607 (379) | **-0.283 (74)**, totR -20.9, win 38% | **t=-2.30, p=0.032** |
| REST (non-OVS) | +0.660 (532) | +0.394 (168) | t=-0.74, p=0.466 |
| Family minus rest at >=50 | | diff -0.68R | t=-1.42, p=0.165 |

**The headline structural finding: the established book-wide >=50 degradation
is mostly a family effect.** Remove the family and the rest of the book's
high-frag avgR is +0.394 vs +0.660 below — a gap, but noise-level after
clustering (p=0.47). Decomposing the total R shortfall at >=50 (vs the <50
baseline avgR): family carries **+68.2R of the 109.2R shortfall (62%) with
only 31% of the trades**.

Honesty caveat: the family-vs-rest *interaction* at >=50 is NOT significant
(t=-1.42, p=0.165; only 16-22 high-frag months of data). What IS significant
is the family's own degradation (p=0.032) while the rest shows none (p=0.47).
With this sample you cannot statistically prove the family is *worse than* the
rest at high frag — you can show the damage lives there and nowhere else at
detectable strength.

## 3. LOYO and episode exclusions

FAMILY4-minus-rest diff at >=50, dropping each year (clustered t):
diff ranges **-0.55 to -0.86, negative under every drop**; t from -0.74
(ex-2020, weakest) to -1.57. Never significant, always same-signed. CORE3
variant: diff -0.73 to -0.86, same picture.

FAMILY4's own >=50-vs-<50 damage under exclusions:

| Exclusion | >=50 avgR (N) | <50 avgR (N) | clustered t | p |
|---|---|---|---|---|
| full sample | -0.283 (74) | +0.607 (379) | -2.30 | 0.032 |
| ex 2020 (COVID) | -0.191 (71) | +0.623 (319) | -1.93 | 0.067 |
| ex 2021 | -0.304 (43) | +0.586 (357) | -2.01 | 0.066 |
| ex 2022 | -0.289 (69) | +0.611 (323) | -2.19 | 0.040 |
| ex 2023 | -0.283 (74) | +0.586 (340) | -2.19 | 0.039 |
| ex 2024 | -0.303 (52) | +0.598 (348) | -2.05 | 0.059 |
| ex 2025 | -0.272 (69) | +0.552 (332) | -1.81 | 0.086 |
| ex 2026 | -0.312 (72) | +0.607 (372) | -2.34 | 0.030 |

Damage is spread, not one episode: 2020 -7.4R (3 tr), 2021 -7.9R (31 tr),
2024 -5.2R (22 tr), 2025 -2.2R (5 tr), 2022 -1.0R (5 tr). Excluding COVID or
any single year of 2021-2024 leaves the point estimate near -0.2 to -0.3R and
the clustered t between -1.8 and -2.3 (p 0.03-0.09). **The family high-frag
damage survives every single-year exclusion**, though several drop to
borderline (p 0.06-0.09) — expected with 74 trades.

## 4. Design comparison (replayed on the ledger, R-weighted)

Multipliers applied to R (equivalently to risk); OVS untouched. avgR/unit =
sum(R x mult) / sum(mult) = return per unit of risk deployed.

| Design | totR | risk units | avgR/unit | worst DD (R) |
|---|---|---|---|---|
| baseline (no throttle) | 626.4 | 1153.0 | 0.543 | -22.9 |
| (a) book-wide taper 1.0→0.5 over 50-60 | 615.0 | 1069.9 | 0.575 | -22.9 |
| **(c) family-only 0.25x at >=50, rest untouched** | **642.1** | 1097.5 | **0.585** | -22.9 |
| (c) family-only 0.5x at >=50 | 636.9 | 1116.0 | 0.571 | -22.9 |
| (b) family 0.25x + book taper on rest | 619.0 | 1041.1 | 0.595 | -22.9 |
| (b) family 0.5x + book taper on rest | 613.7 | 1059.6 | 0.579 | -22.9 |
| (c') CORE3-only 0.25x at >=50 | 643.5 | 1113.3 | 0.578 | -22.9 |

- **Family-only 0.25x is the only design that RAISES total R** (+15.7R vs
  baseline) while also improving efficiency (0.543 → 0.585). It cuts a pocket
  whose EV is negative (-20.9R across 74 trades), so shrinking it adds money.
- The book-wide taper (a) *costs* -11.4R of totR for a smaller efficiency gain
  (0.575), because at >=50 it also throttles strategies that are fine or better
  there: ATR Ext Gap Up +1.98 (5), St OS Sznl +0.65 (12), LT Trend +0.31 (78),
  OLV +0.35 (70).
- Combined (b, 0.25x) has the best avgR/unit (0.595) but gives back 23R of totR
  vs the pure family cut, to suppress a rest-of-book effect that is
  statistically indistinguishable from zero (p=0.47).

**Which years pay** (totR delta vs baseline): family-only 0.25x gains 2020 +5.6,
2021 +5.9, 2024 +3.9, 2025 +1.6; loses 2019 -0.7, 2026 -1.1, 2018 -0.2. Four
gain years, trivial losses. Book taper (a): gains ~0 anywhere; loses 2022 -2.6,
2024 -4.0, **2026 -5.3** (2026 is the inverted year: non-OVS >=50 ran +0.489 vs
-0.093 below 50, and the book taper eats it; the family had only 2 high-frag
trades in 2026, so the family cut costs just -1.1R).

**Drawdown:** the worst realized-R DD (-22.9R) is the June 2026 episode —
39 trades, ALL at frag<50 (avg frag 28.9) — so no throttle touches it; that is
why the DD column ties. The **second-worst** DD is where the designs separate:
baseline -18.5R (2021-11-16 → 2022-01-10) shrinks to **-11.4R** under the
family cut (and -10.3R remains under the book taper). The family cut removes
the deepest throttle-addressable episode; nothing addresses June 2026, which
is a low-fragility drawdown by construction.

## 5. Verdict

**Family-level is defensible ex-ante, with one honest asterisk.**

For: (1) the mechanism is coherent and stated without reference to returns —
2-day long index dip-buys are the book's purest long-crash-risk structure, and
the fragility dial is a crash-risk meter; (2) Indices Oversold Bounce, admitted
purely on mechanism, independently reproduces the degradation (+0.69 → +0.09);
(3) the excluded candidate (3x ETF Overbot Fade) is excluded on direction, not
performance; (4) the family effect survives LOYO and every single-year/COVID
exclusion with a stable point estimate; (5) once the family is removed the
rest-of-book high-frag effect collapses to noise (p=0.47), meaning the
book-wide throttle was always mostly a family throttle wearing a book costume.

The asterisk: the three named strategies entered the conversation because of
their 55+ returns, so the definition is not perfectly clean, and the
family-vs-rest interaction never reaches significance (p=0.13-0.47 across LOYO,
16-22 high-frag months). This is a portfolio-construction judgment supported by
consistent evidence, not a statistically proven interaction. The IOB
confirmation and the mechanism story are what keep it on the right side of
outcome-picking.

## 6. Caveats

- Fragility history is a current-vintage reconstruction; composite edge weights
  have calibration lookahead (full-sample event study). Inherited from the
  established findings; affects all designs equally.
- 74 family trades at >=50 across ~18 months in 8 calendar years. Cells like
  Monday Dip 55+ (N=7) are indicative only.
- The 0.25x level within the tested 0.25-0.5 range is mildly in-sample (0.25
  beats 0.5 because the pocket EV is negative, so more cut = better; any deep
  cut works). The >=50 threshold is inherited from the established findings,
  not re-optimized here.
- Replay assumes R scales linearly with the multiplier (no capacity/fill
  effects) — safe at these sizes.
- 2026 YTD inversion is a live warning for ALL fragility throttles; the family
  design minimizes exposure to it (2 trades, -1.1R) but does not escape it.

## 7. Recommendation

Replace the pending book-wide taper with: **family-only multiplier 0.25x when
the live basis (63d MA10) >= 50, family = {Weak Close Decent Sznls, SPY QQQ
MonFri Reversion, Monday Dip, Indices Oversold Bounce}; rest of book (incl.
OVS) untouched at 1.0x; kill the 1.25x low-frag boost.** In replay this adds
+15.7R vs baseline (vs -11.4R for the book-wide taper), lifts avgR per unit
risk 0.543 → 0.585, halves the 2021-22 drawdown episode (-18.5R → -11.4R), and
cost only -1.1R in the inverted 2026. Re-examine after ~20 more family
high-frag trades accrue.

## Adversarial verification (2026-07-02, verify_family-throttle.py)

Independent recompute from `data/backtest_trades_full.parquet` +
`data/rd2_fragility.parquet` (63d, 10d MA, as-of signal date), fresh script
`scratch/ultracode_research/verify_family-throttle.py`. Two join variants run:
strict ffill(limit=5) drops 7 trades whose signal dates fall in a Feb-Apr 2017
gap in the fragility index (N=1146); unlimited ffill keeps them (N=1153,
matching the researcher). All 7 are low-frag 2017 trades; every high-frag cell
is identical under both joins. The researcher's stated "ffill limit 5" was
therefore not what their code did, but the deviation is immaterial.

Verdicts on the decisive claims:

1. FAMILY4 >=50 vs <50: CONFIRMED. -0.283 (N=74, totR -20.9, win 38%) vs
   +0.614 (N=377) strict join / +0.607 (379) their join. My clustered
   t=-2.40 p=0.025 (18 vs 92 months); theirs -2.30/0.032. Same conclusion.
2. Rest non-OVS >=50 vs <50: CONFIRMED. +0.394 (168) vs +0.658 (527),
   t=-0.78 p=0.443 (theirs -0.74/0.466). No detectable degradation after
   clustering — but note the point estimate still drops 0.26R; this is
   absence of evidence with 22 high-frag months, not proof of flatness.
3. Shortfall decomposition: CONFIRMED. Using the pooled <50 baseline
   (+0.640): family +68.3R of +109.6R total = 62%, with 74/242 = 31% of
   high-frag trades.
4. Interaction: CONFIRMED as disclosed. diff -0.68, t=-1.42 p=0.165.
   LOYO (FAMILY4): diff -0.55 (ex-2020) to -0.73 (ex-2018), always negative,
   p 0.13-0.47, never significant. The stated upper bound -0.86 comes from
   the CORE3 variant, not FAMILY4 — FAMILY4-only range is -0.55..-0.73.
5. Episode exclusions: CONFIRMED. Point estimates match exactly (ex-2020
   -0.191/71; ex-2021 -0.304/43; ex-2024 -0.303/52; ...). My p-values are
   slightly STRONGER than reported (ex-2020 p=0.051 vs 0.067, ex-2022 0.032
   vs 0.040, ex-2024 0.049 vs 0.059) — the report is conservative. Note the
   2020 damage is only 3 trades worth -7.4R (avg -2.5R/trade, COVID gap
   tails); ex-2020 is the honest stress test and it survives at p~0.05-0.07.
6. IOB out-of-definition confirmation: CONFIRMED numerically — +0.693 (107)
   vs +0.087 (21), and +0.176 (19) at 55+. Adversarial caveat: IOB degrades
   to roughly ZERO, it does not flip negative like the named three (which
   run -0.22 to -0.63 at >=50). It supports "edge disappears at high frag"
   for the mechanism, but the family's negative avgR is still driven
   entirely by the three outcome-flagged strategies. The mitigant is real
   but weaker than "reproduces the pattern" suggests.
7. Design (c) family-only 0.25x: CONFIRMED exactly. totR 642.1 (+15.7 vs
   baseline 626.4), avgR/unit 0.5859 (vs 0.5439). Yearly deltas match
   (2020 +5.6, 2021 +5.9, 2024 +3.9, 2026 -1.1).
8. Design (a) book taper 50-60: CONFIRMED exactly. totR 615.0 (-11.4),
   avgR/unit 0.5757; 2026 delta -5.3R, 2024 -4.0R, 2022 -2.6R.
9. Second-worst DD: CONFIRMED with an attribution correction. Baseline
   -18.5R (2021-11-16 -> 2022-01-10) reproduces exactly; book taper leaves
   -10.3R in that window (matches). But under the family cut the 2021-11
   episode shrinks to -4.9R, NOT -11.4R — the -11.4R (my -11.3R) is the
   fam-cut curve's NEW second-worst DD, a different episode (2021-06-16 ->
   2021-07-19) that no >=50 throttle touches. The error is conservative:
   the family cut improves the named episode MORE than claimed.
10. June 2026 worst DD: CONFIRMED exactly. -22.9R, 2026-06-03 -> 2026-07-01,
    39 trades, avg frag 28.9, max frag 44.3, zero trades at >=50. No
    fragility throttle addresses it.

Residual concerns (none fatal, all should ride with the recommendation):
- The family's negative point estimate rests on ~59 trades in three
  outcome-flagged strategies plus 21 near-zero IOB trades; 2021 alone is
  31 of the 74 high-frag trades. Ex-2021 survives (p~0.06) but the cell is
  thin. The "re-examine after ~20 more family high-frag trades" clause is
  the right guard.
- 0.25x vs 0.5x is mildly in-sample (researcher discloses); any deep cut
  works because the pocket EV is negative.
- Inherited lookahead (current-vintage fragility reconstruction, full-sample
  composite weights) affects all designs equally, as stated.
- Replay is cost-free R-scaling; since the throttle only REDUCES size, cost
  omission cannot flatter it.

Bottom line: every decisive number reproduces within tolerance (one DD
mis-attribution that understates the design's benefit; one LOYO range that
mixed CORE3 into a FAMILY4 claim). The honest asterisk — insignificant
family-vs-rest interaction, outcome-flagged origin of the named three — is
correctly disclosed in the report. The family-only throttle beating the
book-wide taper on totR, efficiency, and 2026 robustness is real in replay.
