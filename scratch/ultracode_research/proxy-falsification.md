# Is the fragility score just a proxy? — Falsification test

Run date: 2026-07-02.
Scripts: `scratch/ultracode_research/proxy_falsification.py` (main),
`scratch/ultracode_research/proxy_supplement.py` (robustness),
joined frame cached at `scratch/ultracode_research/proxy_joined.parquet`.

## Verdict up front

**No. The composite is not replicated by any of the five simple proxies, alone
or in pairwise OR combinations, at matched trade counts.** The frag>=50 split
(+0.187 vs +0.647, monthly-clustered t=-2.86, p=0.007) is 1.5-3x stronger than
the best simple alternative, and the fragility effect survives controlling for
every proxy individually and for the union of all six. The key structural
reason: **fragility is a complacency measure, negatively correlated with VIX
and realized vol** — "VIX 10d-MA above X" or "SPY below 200d" select nearly
disjoint (often opposite) trade sets and cannot mimic it directionally.

One genuine addition emerged: within frag>=50, an *elevated 3y realized-vol
percentile* marks the truly toxic subset (avgR -0.544, N=45, win 31%) vs
+0.354 for frag-high/rv-calm — but that interaction rests on essentially two
episodes (2021, 2024) and should be treated as a watch item, not a rule.

## Setup

- Trades: non-OVS ledger trades with signal date >= 2016-08-01, joined as-of
  signal date (merge_asof, 5-day tolerance = live ffill limit) to the live
  fragility basis (63d column of `data/rd2_fragility.parquet`, 10d MA).
  **N = 1136** after dropping rows missing any proxy (vs 1153 in the parent
  study; the 17 lost rows are 200d-SMA / 3y-percentile warmup at the margin).
- Proxies, all point-in-time from `data/master_prices.parquet` (SPY, ^VIX
  adjusted closes; VIX close unaffected by adjustment; SPY proxies are all
  relative levels recomputed per run, safe per the dividend-adjustment rule):
  1. `vix` — VIX close; `vix_ma10` — its 10d MA
  2. `dist200` — SPY close / 200d SMA − 1
  3. `rv21_pct` — 21d realized vol (close-close, annualized), percentile rank
     within rolling 756d (3y)
  4. `dd252` — SPY drawdown from rolling 252d high
  5. `days_since_5dd` — trading days since SPY last closed >=5% below its
     252d high
- frag>=50 covers 242 trades (21.3%). Each proxy is thresholded two ways:
  **matched-N** (its own top 21.3% tail, in the direction correlated with high
  fragility) and **top decile**. All t-stats are monthly-clustered
  (per-month mean R, Welch t between month groups).

## 1. Rank correlations: what the score actually co-moves with

Spearman at non-OVS signal dates (daily-basis values similar):

| proxy | rho vs frag | note |
|---|---|---|
| days_since_5dd | **+0.555** | closest relative — frag is a calm-duration measure |
| dist200 | +0.398 | high frag = SPY *extended above* trend |
| dd252 | +0.285 | high frag = *near highs*, not in drawdown |
| rv21_pct | **-0.231** | high frag = *low* realized vol |
| vix_ma10 | **-0.171** | high frag = *low* VIX |
| vix | -0.011 | ~orthogonal |

Fragility is anti-correlated with every "market is stressed" variable and
positively correlated with the complacency variables. So the naive
falsification candidates ("VIX MA above X", "SPY below 200d") are pointed the
wrong way from the start. The tests below therefore condition each proxy on
its **frag-aligned** tail (low-VIX, extended-above-200d, calm etc.), which is
the only version that could possibly replicate the score.

## 2. Solo conditioning: can any proxy reproduce +0.17 vs +0.65?

Matched-N (top 21.3%) and top-decile tails, monthly-clustered t:

| proxy (aligned dir) | thr | N_hi | avgR_hi | avgR_lo | t | p |
|---|---|---|---|---|---|---|
| vix (low) | <=15.4 | 243 | +0.452 | +0.576 | -0.87 | 0.389 |
| vix top10% | <=13.3 | 114 | +0.431 | +0.562 | -1.17 | 0.251 |
| vix_ma10 (low) | <=14.6 | 242 | +0.376 | +0.596 | -1.04 | 0.302 |
| vix_ma10 top10% | <=12.7 | 114 | +0.621 | +0.541 | +0.87 | 0.393 |
| dist200 (high) | >=+10.7% | 242 | +0.473 | +0.570 | -1.25 | 0.218 |
| dist200 top10% | >=+13.1% | 115 | +0.530 | +0.551 | +0.08 | 0.936 |
| rv21_pct (low) | <=0.26 | 245 | +0.377 | +0.597 | **-1.90** | 0.060 |
| rv21_pct top10% | <=0.12 | 116 | +0.283 | +0.579 | -1.63 | 0.113 |
| dd252 (shallow) | >=-0.5% | 242 | +0.357 | +0.601 | -1.44 | 0.154 |
| days_since_5dd (long) | >=103d | 242 | +0.462 | +0.573 | -0.35 | 0.728 |
| **FRAG >=50** | 50 | 242 | **+0.187** | **+0.647** | **-2.86** | **0.007** |

- Best simple proxy: low 3y realized-vol percentile, t=-1.90 p=0.06 — a real
  whiff of the same complacency effect, but the gap it opens (+0.38 vs +0.60)
  is less than half the fragility gap (+0.19 vs +0.65) and not significant
  under clustering.
- days_since_5dd — the proxy *most correlated* with frag (rho 0.56) — has **no
  R signal at all** (t=-0.35). Correlation with the score is not the same as
  carrying its information: frag>=50 is not simply "it's been calm a while".
- Pairwise OR rules (all 15 combinations, matched-N tails,
  `proxy_supplement.py`): best is dist200|dd252 at t=-2.04 (+0.425 vs +0.609);
  everything else |t|<1.5. The union of all six tails (54% of trades) shows
  nothing (t=+0.37).
- In-sample-tuned sweeps as an overfit ceiling: no `vix_ma10>=X` threshold
  works in either direction (all |t|<0.9 — high-VIX trades are actually the
  book's *best*, +0.7-0.8 avgR). `dist200<=-5%` (deep correction) gets
  t=-2.50 but N=70, is tuned in-sample, and — decisive — has **zero overlap**
  with frag>=50 (0 of 70 trades). Excluding those 70 trades entirely leaves
  the frag split intact (+0.187 vs +0.685, t=-3.20 p=0.003). It is a
  different (also interesting) regime, not an explanation of this one.

## 3. Double sorts: does frag survive controlling for each proxy?

Matched-N proxy strata; frag>=50 vs <50 within each stratum:

| control | within proxy-HI: frag t (p) | within proxy-LO: frag t (p) |
|---|---|---|
| vix | -1.67 (0.119) | **-2.27 (0.031)** |
| vix_ma10 | -1.87 (0.090) | **-2.29 (0.028)** |
| dist200 | **-2.33 (0.046)** | -1.62 (0.114) |
| rv21_pct | **-3.08 (0.010)** | **-2.56 (0.013)** |
| dd252 | **-2.58 (0.015)** | **-2.20 (0.036)** |
| days_since_5dd | -0.88 (0.388) | -1.46 (0.160) |
| any-proxy-high union | **-2.90 (0.006)** | -0.65 (0.529) |

The fragility split points the same direction inside **every** stratum (12/12
cells) and stays significant in at least one stratum of every control except
days_since_5dd (where both strata still show a ~0.45R gap in avgR — the split
into 105/137 frag-hi trades just guts the month counts). Within the
all-proxies-calm stratum the frag gap persists in magnitude (+0.438 vs
+0.664) but not significance (N_hi=92, t=-0.65) — flagged honestly; the
strongest frag months co-occur with at least one mildly elevated simple
proxy.

Reverse direction — does the proxy add anything within frag strata?

- Within frag<50: **nothing** adds (all |t|<=1.48). Below the throttle line
  the simple proxies are noise.
- Within frag>=50: only rv21_pct (t=-2.81 p=0.018) and dd252 (t=-2.19
  p=0.039) discriminate. The rv interaction is the notable one:

### 2x2, frag >= 50 x rv21_pct (matched-N)

| cell | N | avgR | medR | win% | months |
|---|---|---|---|---|---|
| frag<50, rv-calm | 694 | +0.666 | +0.676 | 67 | 89 |
| frag<50, rv-elevated | 200 | +0.584 | +0.574 | 66 | 43 |
| frag>=50, rv-calm | 197 | +0.354 | +0.438 | 63 | 21 |
| **frag>=50, rv-elevated** | **45** | **-0.544** | **-0.805** | **31** | **9** |

Read: fragility-high alone roughly halves avgR; fragility-high *while realized
vol has already started ranking up* is where the actual losses live — the
"stored energy releasing" phase. Threshold-free confirmation:
Spearman(frag, R) = -0.200 (p=0.002) within rv-elevated trades vs -0.029 (ns)
within rv-calm.

**Caveat on the interaction:** the 45-trade toxic cell is 24 trades from 2021
and 19 from 2024 (plus 2 strays). LOYO on the frag-split-within-rv-HI holds
when dropping 2021 (t=-3.62) but softens dropping 2024 (t=-1.92, p=0.099).
Two episodes; directionally consistent but not independently proven. The
damage composition matches the parent study's short-horizon-dip-buyer finding
(Weak Close -1.16, Monday Dip -1.15, MonFri -0.67 in cell; LT Trend/OLV only
about -0.1).

**LOYO on the main effect within the rv-calm stratum** (the harder test —
frag with its best complement stripped out): all 11 drop-one-year runs keep
t between -1.80 and -2.96 (only drop-2021 crosses p=0.08; every other year
p<=0.04). The core frag effect is not the rv interaction in disguise.

## 4. Why the composite wins (mechanism, not just stats)

frag>=50 months: 2017-10, 2018-08..10, 2019-12, 2020-02, 2021-05..2022-01,
2024-04, 2024-08..12, 2025-02, 2026-02/03. These are *pre-drawdown complacency
peaks* — the score fires while VIX is low, SPY is extended above trend, and
realized vol is quiet, i.e. exactly when every one-variable stress proxy says
"all clear". The composite's cross-signal structure (vol suppression +
crowding + correlation-instability, gated by regime) is doing state detection
none of the single variables can, which is why its top tail and the simple
proxies' top tails overlap so little (Jaccard vs frag>=50: vix 0.05,
vix_ma10 0.07, rv 0.10, dd 0.13, dist200 0.20, days_since 0.28).

## Caveats

- Inherited from the parent study: the fragility history is a current-vintage
  reconstruction with calibration lookahead in the composite edge weights
  (full-sample event study). None of the five simple proxies has that problem
  — that IS the one argument for them, and they still lose. But a fair
  statement is: "the composite beats zero-lookahead alternatives *assuming
  its reconstruction is representative of what live would have produced*."
- frag>=50 spans only 23 distinct months / ~6 episodes over 10 years;
  monthly clustering handles serial correlation within months but episodes
  span months (2021 H2 is 7 of the 23). The parent study's LOYO stability
  claim carries the weight here; this study adds LOYO within strata.
- The rv-interaction cell (N=45, 9 months, 2 episodes) is suggestive, not
  established.
- Survivorship in master_prices does not bite here: all proxies are SPY/^VIX.
- N=1136 vs parent 1153 (proxy warmup drop); headline numbers reproduce
  (+0.187/+0.647, t=-2.86 p=0.007 — identical to established finding).

## Recommendation

Keep the composite; do not replace it with a simple proxy — proceed with the
pending taper rec (kill 1.25x boost, 1.0x through 50, taper to 0.5x by 60) on
the composite basis. The lookahead concern the simple proxies were meant to
solve is better addressed by fixing the composite's weight calibration
(point-in-time re-estimation of `signal_horizon_stats.json`) than by
downgrading to a proxy that measurably doesn't work. Log the
frag>=50 & rv21_pct>74th-3y-percentile interaction as a monitored hypothesis
(it would currently gate ~4% of trades); revisit after the next fragility
episode rather than encoding it now on two episodes of evidence.

## Adversarial verification (2026-07-02)

Independent recompute: `scratch/ultracode_research/verify_proxy-falsification.py`
(+ `verify_diag*.py`). Fresh join from raw parquets (my N=1145 vs their 1136 —
9-row difference from proxy warmup handling; all cross-checks also run on
their cached `proxy_joined.parquet`).

### Confirmed

- **Claim 1 (frag>=50 baseline)**: exact on their frame (+0.187/+0.647,
  t=-2.86 p=0.007, N=242/894). My independent join: +0.187/+0.641, t=-2.77
  p=0.009, N=242/903. CONFIRMED.
- **Claim 2 (rv21 calm tail, matched-N)**: exact on their frame (+0.377/+0.597,
  t=-1.90 p=0.060). Caveat: on my 1145-row join the same rule gives t=-2.10
  p=0.038 — "not significant" is fragile to 9 trades. The substantive claim
  (gap less than half of frag's, +0.22R vs +0.45R) holds either way. CONFIRMED
  with fragility note.
- **Claim 3 (VIX rules fail)**: vix_ma10 low tail t=-0.93 p=0.36 (theirs
  -1.04). High-VIX sweep: every threshold 15-30 gives POSITIVE t (high-VIX
  trades are the book's best; max t=+1.53 at >=19, above their "all |t|<0.9"
  but same conclusion — no high-VIX rule works as a risk-off filter). CONFIRMED.
- **Claim 4 (rank corrs)**: -0.160 vix_ma10, -0.219 rv21_pct, +0.396 dist200,
  +0.276 dd252, +0.539 days_since_5dd (all within 0.02 of reported);
  days_since aligned tail t=-0.55 (theirs -0.35), no signal. CONFIRMED.
- **Claim 5 (union control)**: union share 0.538; within union-high frag split
  +0.028/+0.617, t=-2.80 p=0.008 (theirs +0.033/+0.632, -2.90/0.006). CONFIRMED.
  (Naming note: "any proxy elevated" means any proxy in its frag-ALIGNED
  complacent tail, not stress-elevated.)
- **Claim 8 (deep correction disjoint)**: 0 of 70 dist200<=-5% trades have
  frag>=50; excluding them frag split t=-3.10 p=0.004 (theirs -3.20). The deep
  rule itself: t=-2.42, in-sample tuned as they flagged. CONFIRMED.

### REFUTED: the rv-interaction direction is INVERTED throughout Section 3

`proxy_supplement.py` builds `masks[p]` as the **frag-aligned** tail
(`v = proxy * sign(rho)`; for rv21_pct sign=-1), so `masks['rv21_pct']` is the
**bottom ~21% of rv percentile — ultra-CALM**, not elevated. The md then labels
that mask "rv-elevated" everywhere in Section 3. Proof from their own cached
frame: frag>=50 & rv21_pct >= 0.772 (a real top-tail) contains **5 trades**
(4x2020, 1x2018, avgR -0.96), and no rv threshold reproduces the reported 2x2
under the "elevated = high rv" reading. Under the corrected reading everything
reproduces on my independent frame:

- Toxic cell = frag>=50 & rv21_pct <= ~0.24 (**calmest** quintile): N=43,
  avgR -0.562, med -0.805, win 30%, 41/43 from 2021+2024; strategy composition
  matches the md exactly (Weak Close -1.16, Monday Dip -1.15, MonFri -0.67).
- Frag split within ultra-calm: t=-2.86 p=0.015; drop-2021 t=-3.84,
  drop-2024 t=-1.69 (their -3.62 / -1.92 pattern).
- "LOYO within rv-calm stratum" (claim 6) is actually LOYO within the
  NOT-ultra-calm stratum (rv_pct > 0.24): my 11 drop-years t in [-3.04, -1.83],
  p<=0.077 — numbers reproduce, label backwards. The test's substance (frag
  effect survives with the toxic complement stripped) still stands.
- Threshold-free: Spearman(frag,R) = -0.213 (p=0.001) within ULTRA-CALM,
  -0.024 (ns) within the rest — the md's -0.200/-0.029 with strata swapped.

Consequences:
1. The mechanism narrative ("realized vol has already started ranking up",
   "stored energy releasing") is backwards. The toxic subset is frag-high while
   realized vol sits in its calmest 3y quintile — deepest complacency, BEFORE
   any vol expansion. (Arguably more coherent with the complacency thesis, but
   it is the opposite of what the report says.)
2. **The recommendation's kill-switch condition is wrong as written.**
   "frag>=50 AND rv21-3y-percentile > ~75%" selects 5 trades (2018/2020) on
   their own data, not the 45-trade cell. The condition their analysis actually
   found is frag>=50 AND rv21-3y-percentile <= ~24%. Do not log the >75%
   version.

### Net assessment

The core anti-proxy findings (claims 1-5, 8) are solid and independently
reproduced; the composite is not replicated by any simple proxy at matched N,
and the frag effect survives every control. The single-episode caveats the
researcher flagged remain fair. But the one "genuine addition" — the rv
interaction — has its sign flipped end to end (a variable-direction bug
propagated from `masks` into the prose), and the monitored-hypothesis
recommendation inherits the error. Keep the composite; if the interaction is
logged at all, log it as frag>=50 & rv21_pct<=~25th-3y-pctile.
