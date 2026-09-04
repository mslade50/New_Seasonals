# The 2026 Inversion — anatomy, components, base rates, and what it implies for the throttle

Run date: 2026-07-02. Track: `inversion-2026`.

Scripts (all under `scratch/ultracode_research/`):
- `inv2026_q1_q3_q4.py` — trade lists, yearly tables, base rates, bootstraps
- `inv2026_composition.py` — strategy-mix decomposition, June energy cluster
- `inv2026_components.py` — fragility component reconstruction + episode attribution
- `inv2026_prepost.py` — pre-break vs post-break phase split
- `inv2026_throttle_replay.py` — pending-rec replay + LOYO of the phase split

Join: non-OVS trades from `data/backtest_trades_full.parquet` to the live basis
(63d fragility, 10d MA, as-of signal date, 5-day tolerance) from
`data/rd2_fragility.parquet`. N=1136 joined non-OVS trades 2016-08..2026-06.
(Slightly below the 1153 in the established findings because of the 5-day
merge tolerance; headline numbers reproduce: 2026 >=50 +0.489 / <50 -0.093.)

## Headline

The "2026 inversion" is real arithmetic but not a regime message. It decomposes
into two unrelated, fully identifiable events:

1. **The >=50 side (+0.489, N=24)** is a single fragility episode
   (2026-02-13..2026-03-24, the only >=50 window of the year) in which **19 of
   24 trades were signed AFTER SPY had already broken >3% below its 52w high**.
   Post-break high-frag dip-buys have ALWAYS been the profitable side of the
   high-frag sample: 2016-2025 avgR **+0.417 (N=60)** vs pre-break **-0.068
   (N=125)**. 2026's post-break trades made +0.671 (N=19) — the historical
   pattern, not its refutation.
2. **The <50 side (-0.093, N=68)** is entirely June: 39 trades, **-21.5R**, of
   which **-20.4R came from one correlated OLV energy-complex cluster** (OXY
   -5.2R over 7 re-signals, PBR -3.7, USO -3.5, LYB -3.0, DBC -2.3, CL=F -1.6,
   BP -1.3, WLK -1.0) as oil broke down while SPX stayed calm. Ex-June, 2026
   below-50 avgR is **+0.523** — normal. Monthly-clustered, 2026's below-50
   months average +0.600 vs +0.675 for 2016-25 (t=-0.21, p=0.84): nothing.

The fragility dial has no jurisdiction over an oil-sector selloff during a calm
SPX; the June damage is an OLV sector-concentration problem, not a throttle
problem.

## Q1 — the actual trades

### 2026 frag >= 50 (N=24, totR +11.7, avgR +0.489, win 71%)

All signed 2026-02-23..2026-03-23 inside the one episode. Mix: LT Trend ST OS
x13 (avgR -0.003), OLV x8 (+0.786, five of them the same ticker HE), Indices
Oversold Bounce x2 (+0.744), ATR Extended Gap Up x1 (USO short, +4.00R — the
single biggest trade of the year).

| Date | Strategy | Ticker | frag(MA) | raw 63d | SPY dd | R |
|---|---|---|---|---|---|---|
| 02-23 | LT Trend ST OS | BANC | 69.5 | 76.6 | -2.1% | +1.00 |
| 03-04 | LT Trend ST OS | CUK | 80.7 | 81.3 | -1.7% | -1.19 |
| 03-05 | LT Trend ST OS | LUV | 80.9 | 76.2 | -2.3% | +0.01 |
| 03-06 | ATR Ext Gap Up | USO (short) | 80.4 | 70.7 | -3.6% | +4.00 |
| 03-10..03-19 | OLV | HE x5 | 58-78 | 49-64 | -2.9..-5.4% | +0.41..+1.67 |
| 03-12 | Indices OS Bounce | ^GSPC, ^NDX | 73.4 | 58.2 | -4.5% | +0.68, +0.81 |
| 03-12..03-23 | LT Trend ST OS | PH, STLD, XLI, ^MXX, FLS, SCCO, WT, MCK, NOC | 54-73 | 45-58 | -4.5..-5.8% | mixed, ~0 |
| 03-16..03-19 | OLV | NUE, GILD, D | 58-67 | 49-54 | -4.1..-5.4% | +1.27, -1.01, +0.98 |

Concentration: top-3 trades sum +6.9R of the +11.7R. Drop the single USO +4R
winner and avgR falls to **+0.336** (N=23); drop top-2, +0.276. Median +0.44.

**Composition effect.** The strategies that produced the historical >=50
damage did not trade in the episode: Weak Close Decent Sznls (hist >=50 avgR
-0.589, N=20), SPY QQQ MonFri (-0.221, N=24), Monday Dip (-0.630, N=9) — zero
2026 >=50 trades among them. The 2026 mix was 54% LT Trend ST OS (hist >=50
+0.374) and 33% OLV (+0.297). Reweighting historical per-strategy >=50 rates by
2026's mix predicts **+0.316** — vs actual +0.489 (and +0.336 after removing
one trade). More than half of the apparent inversion (+0.49 vs pooled hist
+0.15) is strategy mix; the rest is one USO trade. Note LT Trend ST OS actually
did WORSE than its own >=50 history (-0.003 vs +0.374).

### 2026 frag < 50 (N=68, totR -6.3, avgR -0.093, win 37%)

By month: Jan +0.76 (7), Feb +0.89 (4), Mar +2.00 (1), Apr +0.25 (8), May +0.26
(9), **Jun -0.551 (39, totR -21.5)**. By strategy: OLV -0.500 x44 (totR -22.0)
vs its own 2016-25 below-50 baseline of +0.990 (N=171); every other strategy
roughly at baseline (ATR Ext Gap Up +0.861 x10, MonFri +0.893 x3, etc.).

June by ticker: OXY 7 signals -5.2R, PBR 3 -3.7, USO 6 -3.5, LYB 3 -3.0, DBC 3
-2.3, CL=F 3 -1.6, BP -1.3, WLK -1.0; the whole energy/commodity complex is
N=30, totR -20.4, avgR -0.680; everything else in June: N=9, totR -1.1. OLV
re-signals the same bleeding ticker daily, so the 39 June "trades" are perhaps
5-6 independent bets on one macro leg (oil breakdown).

## Q2 — what the fragility components looked like

Reconstruction: all 7 rd2 signals recomputed from `master_prices.parquet`
(SPY OHLC, ^VIX, 11 sector ETFs, 375 S&P names available) via the actual
`pages/risk_dashboard_v2.py` functions, then `compute_fragility_timeseries`.
Validation vs the frozen parquet: 63d corr **0.87 in 2026** (MAD 5.7 pts), 0.82
full-history. Good enough for attribution; exact levels differ a few points.

Signal fires into the episode (reconstruction):
- **Distribution Dominance** (63d edge 2.42): ON 2025-12-30..2026-02-26 — 35
  consecutive sessions of volume-confirmed distribution while SPY sat within 2%
  of its high.
- **Defensive Leadership** (edge 4.0, the largest): ON 2026-02-03..2026-03-04 —
  risk-off stocks leading on both 50d and 200d breadth with SPY near highs.
- **Low Absorption Ratio** (edge 3.5): fired repeatedly Dec 31-Feb 6.
- **Dispersion** (edge 2.91): declustered single-day fires Feb 10, Feb 26, Mar 13.
- VIX Range Compression, Seasonal Rank Divergence, Pre-FOMC: not material.

Attribution on episode dates (63d contributions, regime mult 1.1-1.2x, calm
mult 1.05): mid-February the composite was driven DD 2.4 + DL 4.0 + LowAR 2.9 +
Disp 2.5 out of max ~15.5 → raw score mid-70s to 81.

**This is the historically meaningful mix.** Defensive Leadership dominated the
Sep-Oct 2018 episode (preceded the Q4-2018 -20%) and the Nov 2021-Jan 2022
episode (peak 95, preceded the 2022 bear). Distribution + defensive rotation +
low absorption near all-time highs is exactly the "everyone on the same side
while the tape churns" configuration the dial was built to catch. And it
worked as a market warning: SPY fell **-9.1%** from the mid-February high to
the 2026-03-30 trough.

The trade-level "inversion" happens because of timing mechanics, not because
the warning was wrong: the near-high signals stop firing once the break starts
(near-high filters), the raw score decays through the drawdown, but the **10d
MA (the live sizing basis) stays >=50 for weeks into the dip** — by 2026-03-12
the raw 63d score was 58 and falling (48.9 by 03-19) while the MA still read
73/58. Meanwhile the book's high-frag population during a dip consists of
dip-buyers (LT Trend ST OS, OLV, Indices OS Bounce), and the dip bounced (SPY
back to new highs by June). So the throttle window overlapped the post-break
recovery trades, which have always been fine.

## Q3 — is 2026's below-50 weakness (-0.09) unusual?

Below-50 avgR by year (non-OVS):

| Year | N | avgR | med | win% |
|---|---|---|---|---|
| 2016 | 40 | +0.245 | +0.35 | 57 |
| 2017 | 43 | +0.952 | +0.98 | 63 |
| 2018 | 101 | +0.414 | +0.44 | 62 |
| 2019 | 72 | +0.579 | +0.53 | 65 |
| 2020 | 114 | +0.962 | +0.80 | 74 |
| 2021 | 72 | +0.897 | +1.10 | 79 |
| 2022 | 84 | +0.335 | +0.47 | 62 |
| 2023 | 104 | +0.827 | +0.89 | 74 |
| 2024 | 57 | +0.640 | +0.68 | 70 |
| 2025 | 139 | +0.905 | +1.17 | 73 |
| **2026** | **68** | **-0.093** | **-0.44** | **37** |

No prior year is close to negative; per-trade, 2026 is unprecedented. But:
- **It is one month and one sector.** Ex-June: +0.523 (N=29), a normal year.
- **Monthly-clustered it disappears**: 2026's six below-50 month-means average
  +0.600 vs +0.675 for the 99 prior months (t=-0.21, p=0.84). June (-0.551) is
  a bad month, not an outlier month — the -0.09 per-trade figure is an artifact
  of June holding 39 of 68 trades (OLV re-signaling the same bleeding tickers).
- Regime: the June losses happened at frag 21-45 because the dial measures SPX
  fragility, and SPX was calm at highs while oil broke. Structurally, no
  SPX-level composite can catch this; it is OLV sector-concentration risk.

## Q4 — base rate of single-year inversions

Years with >=5 trades on both sides of 50:

| Year | lo N / avgR | hi N / avgR | diff (hi-lo) | inverted? |
|---|---|---|---|---|
| 2018 | 101 / +0.414 | 16 / +0.016 | -0.397 | no |
| 2020 | 114 / +0.962 | 6 / +0.023 | -0.939 | no |
| 2021 | 72 / +0.897 | 102 / +0.202 | -0.696 | no |
| 2022 | 84 / +0.335 | 13 / +0.393 | **+0.058** | marginal yes |
| 2024 | 57 / +0.640 | 65 / +0.148 | -0.492 | no |
| 2025 | 139 / +0.905 | 14 / -0.199 | -1.105 | no |
| 2026 | 68 / -0.093 | 24 / +0.489 | **+0.582** | yes |

- **2 of 7 measurable years invert** (2022 barely; 2026 loudly). 2016, 2017,
  2019, 2023 had <5 high-frag trades (the score rarely reached 50).
- **No year-to-year predictive content**: the diff sequence flips sign without
  pattern (-0.94 → -0.70 → +0.06 → ... → -0.49 → -1.11 → +0.58); with 7
  observations nothing can be claimed, and the 2022 inversion was followed by
  a year with no high-frag sample at all.
- Bootstraps (iid and month-block) put P(diff >= +0.58 | 2016-25 pools) at
  ~0.0001, but this drastically overstates rarity honestly measured: 2026's
  diff is one 27-day episode against one June sector cluster — effective
  independent N is about 2. The whole decade contains only **13 distinct >=50
  episodes**. Per-trade or even per-month resampling is the wrong null here;
  treat the bootstrap as decoration, not evidence.

## The phase finding (new, and the one thing 2026 actually teaches)

Splitting ALL >=50 trades by SPY's distance from its 52w high at signal date:

| | pre-break (dd > -2%) | transition | post-break (dd < -3%) |
|---|---|---|---|
| 2016-2025 | **-0.068 (N=125, win 46%)** | +0.514 (N=33) | **+0.417 (N=60, win 73%)** |
| 2026 | -1.186 (N=1) | +0.044 (N=4) | +0.671 (N=19, win 74%) |

- The historical high-frag damage is entirely a **pre-break** phenomenon, and
  within pre-break it concentrates in the short-horizon index dip-buyers
  (Weak Close/MonFri/Monday Dip: -0.422 pre-break, N=40; other strategies
  +0.098, N=85).
- LOYO on the pre/post gap: positive in every leave-one-year-out (+0.33 to
  +0.88), but dropping 2021 removes 87 of 126 pre-break trades and dropping
  2024 moves pre-break to +0.04 — the level is 2021/2024-heavy even though the
  sign is stable. Monthly-clustered pre vs post: t=-1.21, p=0.24 — **not
  significant**. Suggestive, not proven.
- Mechanically this is the 10d-MA lag: ~24-25% of all ">=50 (MA)" trades in
  both eras were signed when the RAW 63d score was already back under 50 —
  the throttle stays on through the dip and cuts recovery trades.

## Replay: what the pending rec would have done

Pending rec (not applied): no boost, 1.0x through 50, linear taper to 0.5x at
60. Replayed on realized R (in-sample accounting, gross of the risk-adjustment
argument — the throttle's justification is variance reduction, not total R):

| Year | delta R from taper | throttled trades |
|---|---|---|
| 2018 | +0.6 | 16 |
| 2021 | -0.2 | 102 |
| 2022 | -2.6 | 13 |
| 2024 | -4.0 | 65 |
| 2025 | +0.1 | 14 |
| **2026** | **-5.3** | 24 |
| **2016-2026 total** | **-11.4R** | 218+24 |

In 2026 the pending taper would have cut ~5.3R of winners while doing
**nothing** about the June -21.5R (all below 50). A phase-gated variant (taper
only while SPY within 2% of its 52w high) flips the decade total to +10.8R
(2021 +4.1, 2024 +3.9, 2026 +0.6) — but that comparison is an in-sample rule
fit built on a hypothesis this very study generated; it is a candidate for a
proper LOYO study, NOT a validated change.

## Caveats

- Fragility history is a current-vintage reconstruction (calibration lookahead
  in the edge weights from `signal_horizon_stats.json`); component signal
  states are point-in-time. My component attribution is itself a reconstruction
  (corr 0.87 with the frozen 2026 series, MAD ~6 pts) using 375 of ~505 S&P
  names — directionally reliable, not exact.
- All 2026 cells are tiny and clustered: the >=50 sample is 1 episode, the HE
  ticker repeats 5x, June is ~6 independent bets rendered as 39 trades. Every
  "2026 avgR" in this report has an effective N far below its nominal N.
- Survivorship in master_prices affects the SP500 panel used for DL/dispersion
  reconstruction, mildly.
- The pre/post-break split was formulated after seeing 2026 — treat its
  economics (dip-buyers do fine buying confirmed dips; the risk is being long
  at unconfirmed highs) as the prior, and the p=0.24 clustered test as the
  honest evidence level.

## What 2026 does and does not imply for shrinking the throttle

**Does NOT imply:**
- It does not refute the throttle premise. The configuration the throttle
  exists for — pre-break trades at fragile highs, especially short-horizon
  index dip-buyers — was almost entirely absent from 2026's high-frag sample
  (1 trade). You cannot lose an edge you never bet against.
- It does not argue for widening or shrinking the 50/60 taper on its own: the
  below-50 losses were a sector event invisible to (and out of scope for) an
  SPX fragility dial.
- The dial itself worked in 2026 as a market-level warning: the DD+DL+LowAR
  mix at the February highs preceded a -9.1% drawdown.

**Does imply:**
- The throttle as designed (10d MA, no phase awareness) spends most of its
  throttled R cutting POST-break dip-buys — the historically profitable
  +0.42-0.67 side — because the MA stays >=50 for weeks into a correction.
  2026 is the third episode (after 2022, 2024) where the realized cost of the
  taper was concentrated there.
- June 2026's -21.5R says the binding risk-control gap is OLV same-direction
  sector concentration (7 OXY signals, 30 correlated energy longs in a month),
  not fragility sizing.

## Recommendation

Keep the pending rec (kill the 1.25x boost, 1.0x through 50, taper to 0.5x by
60) — 2026 gives no basis to shrink it further and no basis to abandon it. But
before locking it, run one follow-up study: **phase-gate the taper** (apply it
only while SPY is within ~2% of its 52w high, release once the break is >3%
confirmed), validated LOYO with monthly clustering. Historical accounting says
the gate turns the taper's -11.4R decade cost into +10.8R while preserving the
protection in the pre-break zone where all the historical damage lives
(-0.068 avgR, N=125); current evidence level is suggestive (clustered p=0.24),
which is why it needs the study rather than immediate adoption. Separately and
with higher priority than any throttle tuning: add a same-sector/same-direction
concentration cap to OLV — June 2026 was -20.4R of one oil bet resignaled 30
times, at low fragility, and no SPX-level dial will ever catch that.

## Adversarial verification

Verified 2026-07-02 by an independent recompute
(`scratch/ultracode_research/verify_inversion-2026.py` — fresh join of the
ledger to `rd2_fragility['63d'].rolling(10,min_periods=1).mean()`, as-of signal
date, ffill limit 5d, non-OVS; SPY 52w-high drawdown from `master_prices`;
no code reuse from the study scripts). My join lands N=1146 (vs the study's
1136; tolerance-implementation difference, immaterial — every headline cell
matched).

| Claim | Verdict | Recompute |
|---|---|---|
| 2026 >=50: +0.489 N=24, one Feb 13–Mar 24 episode, +0.336 ex-USO | **CONFIRMED** | +0.489 (N=24, win 71%), span 02-23..03-23, the year's only MA>=50 window is exactly 2026-02-13..2026-03-24; drop the USO +4.00R -> +0.336 (N=23). Strategy mix matches (LT Trend x13 -0.003, OLV x8 +0.786, Indices OS x2 +0.744, ATR Gap Up x1). |
| Composition-adjusted expectation +0.316 (23/24 covered) | **CONFIRMED (with a footnote)** | Their +0.316 reproduces only when ATR Extended Gap Up is excluded from coverage: (2×+0.017 + 13×+0.374 + 8×+0.297)/23 = +0.317. My join has 4 historical >=50 ATR Gap Up trades (avg +1.48); full 24/24 coverage gives **+0.365**. Either number supports the same conclusion (mix explains most of +0.489 vs pooled hist +0.153) — full coverage strengthens it. |
| Phase split: pre-break -0.068 (N=125, 46%) vs post-break +0.417 (N=60, 73%); 19/24 of 2026 post-break | **CONFIRMED** | Exact: pre -0.068 N=125 win 46%, mid +0.514 N=33, post +0.417 N=60 win 73%. 2026: pre 1 / mid 4 / post 19 (+0.671). Clustered pre-vs-post t=-1.21 p=0.24 also reproduces — correctly reported as NOT significant. |
| 2026 <50 = June: 39 trades -21.5R, energy -20.4R; ex-June +0.523; clustered t=-0.21 p=0.84 | **CONFIRMED** | June <50: N=39, totR -21.48. Their 8 named losers sum -21.54 (N=27); the "-20.4R, N=30 complex" additionally nets in CVE/GLNG/PBA (+1.14 combined) — internally consistent, not cherry-picked (I checked every June ticker; non-complex June names total -1.1R). Ex-June <50: +0.523 (N=29). Clustered month-means: 2026 +0.600 vs hist +0.655, my t=-0.15 p=0.88 (theirs -0.21/0.84; same conclusion, join-count noise). |
| Base rate: 2/7 measurable years invert; 13 distinct >=50 episodes | **CONFIRMED** | Measurable years {2018,2020,2021,2022,2024,2025,2026}; inverted = 2022 (+0.058) and 2026 (+0.582), exact. Distinct MA>=50 runs 2016-2026: exactly 13 (unchanged under gap-merge <=7 days). |
| Taper replay: 2026 -5.3R, decade -11.4R, all June losses below 50 | **CONFIRMED** | My replay (mult = max(0.5, 1-0.05·(s-50)) above 50): 2026 -5.28R on 24 trades; 2016-2026 total -11.37R on 242 throttled trades (2022 -2.55, 2024 -4.00 match their table). Zero June-2026 trades at >=50. Also reproduced the phase-gated variant: +10.79R decade (2021 +4.09, 2024 +3.95, 2026 +0.59) — matches +10.8, and I agree it is in-sample accounting on a hypothesis this study generated (their own caveat is correct). |
| Component mix (DD 35 fire-days + DL + LowAR, corr 0.87) + SPY -9.1% into Mar 30 | **UNCERTAIN on components, CONFIRMED on the checkable envelope** | Did not re-run the rd2 signal reconstruction (would inherit the same survivorship/calibration caveats). Independently checkable pieces all hold: SPY -9.1% exactly (693.52 on 2026-02-02 -> 630.35 trough on 2026-03-30); the MA-lag mechanics reproduce (03-12 raw 58.2 vs MA 73.3; 03-19 raw 48.9 vs MA 58.2); the episode window matches. Treat the DD/DL/LowAR attribution as plausible but unverified. |

Verifier's assessment: no methodology errors found. The study's own caveats are
the right ones (effective N≈2 for the inversion, phase split formulated
post-hoc at clustered p=0.24, replay is realized-R in-sample accounting, June
"39 trades" ≈ 6 independent bets). One correction of record: the
composition-adjusted expectation should be quoted as +0.32 to +0.37 depending
on ATR Gap Up coverage. The recommendation (keep the pending rec; LOYO-study
the phase gate before adopting; prioritize an OLV sector-concentration cap)
follows from numbers that all reproduce.
