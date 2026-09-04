# OVS Exemption and Its U-Shape — Fragility Multiplier Investigation

Run date: 2026-07-02. Track: `ovs-exemption`.

Scripts (all under `scratch/ultracode_research/`):
- `ovs_frag_analysis.py` — band tables, equal-weight monthly-clustered t-tests, LOYO, throttle replay
- `ovs_frag_followup.py` — midterm confound, 0-3 peak test, P1/P2, earnings composition, half-sample split
- `ovs_frag_bootstrap.py` — trade-weighted monthly block bootstrap (10,000 reps)
- `ovs_frag_loyo_boot.py` — LOYO on the bootstrap basis + 55+ checks
- `ovs_frag_edges.py` — band-edge sensitivity + signal density

## Setup

Joined `data/backtest_trades_full.parquet` OVS trades (Strategy contains "Overbot Vol") to the
live sizing basis: `rd2_fragility.parquet` 63d column, 10d rolling mean, as-of signal date
(merge_asof, 5-day tolerance). Window 2016-07-25 .. 2026-06-30, **N=820** OVS trades.
Bands follow the established decile cuts: [0,3), [3,21), [21,44), [44,55), [55,100].

**Path marker exists in the ledger.** `Size_Mult` cleanly encodes it: P1 (decisive gap) = 1.0
or 0.75 (midterm tilt), P2 (mild gap) = 0.15/0.2 and pro-rata cap variants. Cross-checked
against the recomputable gap `(T+1 Open − Signal Close)/ATR > 0.25`: perfect agreement
(687 P1 / 422 P2 full history; 542/278 in the frag window). Midterm years (year%4==2)
contribute 194 trades.

## Band table (all OVS, frag window)

| band | N | avgR | medR | win% | totR |
|---|---|---|---|---|---|
| 0-3 | 360 | +0.600 | +0.745 | 71.7 | +216.1 |
| 3-21 | 130 | +0.382 | +0.353 | 65.4 | +49.6 |
| 21-44 | 230 | +0.121 | +0.010 | 50.4 | +27.8 |
| 44-55 | 55 | +0.390 | +0.384 | 61.8 | +21.4 |
| 55+ | 45 | +0.481 | +0.592 | 60.0 | +21.6 |

By path: the U-shape is a **P1 phenomenon**. P1: 0-3 +0.785 (N=279) → 21-44 +0.167 (N=135)
→ 55+ +0.421 (N=22). P2 is flat-weak everywhere below 44 (0-3 **−0.036** N=81; 21-44 +0.055
N=95) and only positive at 44+ (+0.342/+0.538, N=35/23). P2 shows no fragility structure at
all (mid vs calm z=+0.39). Both tiers (Liquid/Overflow) show the same mid-band dip.

## Q1 — Is the 21-44 weakness statistically real?

Two clustered tests disagree, and the disagreement is informative.

**Equal-weight monthly means (the test used for the book-wide finding): nothing.**
21-44 monthly-mean +0.215 (40 mo) vs frag<21 +0.276 (57 mo), t=−0.34 p=0.74. Vs all-other
t=−0.62 p=0.54. LOYO: every drop-year t between −0.02 and −0.71, all p>0.5. Even the 0-3
peak fails this test (0-3 +0.275 vs rest +0.266, t=+0.04 p=0.97).

**Trade-weighted monthly block bootstrap (resample months with replacement, keep all trades
in a drawn month, 10k reps): a real dip.**
- 21-44 vs frag<21: diff **−0.422R, z=−3.02, p=0.006** (Na=230, Nb=490)
- 21-44 vs frag<3: −0.480R, z=−2.86, p=0.018
- LOYO stable: drop-any-single-year z ranges **−2.18 to −3.30** (worst: drop 2020, 2023, or
  2026). P1-only version even stronger: z −2.24 to −3.49, all years.
- Non-midterm years only: z=−2.35 (vs <21) / −2.48 (vs <3) — survives the midterm confound.
- Drop BOTH 2025+2026 together: z=−1.84 (still ~2σ on 121 mid-band trades).

**Reconciliation.** The dip lives in trade-heavy months (2026-06: 22 trades, 2026-01: 19,
2018-01: 18; months with ≥5 trades hold 177 of 230 mid-band trades). Equal-weighting months
dilutes that to nothing; trade-weighting (which is what P&L experiences, since sizing is
per-trade) sees ~3σ. Both are honest clustered tests; the economically relevant one is
trade-weighted, but the equal-weight null warns that per-episode the effect is modest — the
damage compounds because OVS fires *often* inside bad mid-band months.

**Edge sensitivity (the main weakness of the finding).**

| band def | avgR (N) | diff vs below | z |
|---|---|---|---|
| [15,40) | +0.155 (221) | −0.410 | −2.99 |
| [18,42) | +0.174 (227) | −0.373 | −2.69 |
| [21,44) | +0.121 (230) | −0.422 | **−3.02** |
| [25,44) | +0.197 (184) | −0.282 | −1.70 |
| [21,50) | +0.135 (264) | −0.407 | −3.07 |
| [25,50) | +0.203 (218) | −0.277 | −1.79 |
| [30,50) | +0.175 (168) | −0.289 | −1.87 |

Moving the lower edge from 21 to 25 halves the z: the 21-25 sliver (46 trades, avgR ≈ −0.18)
carries a chunk of the dip, and the established book-wide findings flagged 21-27 as episode-
artifact territory. Half-sample split: dip mild in 2016-2021 (+0.255 vs 0-3 +0.547), strong in
2022-2026 (+0.056 vs +0.659) — present in both halves but not equally.

**Verdict on Q1: real at roughly 2σ effective strength** — trade-weighted z≈3.0 and LOYO-stable
at the canonical cuts, but edge-sensitive down to ~1.8σ, invisible in equal-weight monthly
terms, and sitting on a reconstructed fragility history with calibration lookahead. Not the
~3σ it first appears; clearly more than nothing.

## Q2 — Interactions with known OVS structure

**Midterm tilt (0.75x, year%4==2).** Large overlap: the 21-44 band is **40.9% midterm trades
vs 23.7% base rate** (55+ is 44.4%; 0-3 only 14.7%). Midterm mid-band avgR +0.022 (N=94) vs
non-midterm mid-band +0.189 (N=136). But the dip is not merely the midterm effect repackaged:
within non-midterm years only it is still z≈−2.4. Within midterm years the band contrast
disappears in equal-weight terms (t=−0.02) because everything midterm is weak. Practical
consequence: any mid-band multiplier stacks with the 0.75x tilt on ~41% of its hits (0.75 ×
0.75 = 0.5625x; a 0.5x band mult would produce 0.375x — over-shrinkage for this evidence).

**P1 vs P2.** The mid-band dip is concentrated in P1 (calm-band P1 +0.785 collapsing to +0.167;
z=−3.33, LOYO −2.24..−3.49). P2 has no fragility structure (z=+0.39) and, notably, **no calm-band
edge either** (0-3: −0.036, N=81) — its only positive cells are at frag ≥44. In the frag window
P2 runs +0.113 avgR (N=278) vs P1 +0.563 (N=542). This is side evidence for the unresolved
live-vs-backtest P2 divergence (live already retired P2; the ledger still models it): the
backtest's P2 R is weakest exactly where most trades occur.

**Earnings blackout.** Already applied upstream in the ledger build (strat_backtester pre-pass),
so no independent interaction is measurable. Composition check: ~90% of trades in every band are
on tickers present in `earnings_calendar.parquet`; the pass-through (no-data) minority shows
mid-band avgR −0.329 but on N=21 — noise, no action.

## Q3 — Would a mid-band-only throttle be defensible ex-ante?

**Mechanism.** OVS is the book's only short: it fades single-name overbought vol spikes.
A U-shape has a coherent story. At calm (0-3, 44% of trades) spikes are idiosyncratic
exhaustion — the classic fade, and the strategy's home regime (signal density 0.50 trades/day
vs 0.19 at 55+). At high fragility (55+) the short side picks up a beta tailwind and overbought
spikes in a stressed tape are squeezes/bear-rallies that fail — the fade works again. The mid
band (rising-but-not-extreme fragility) maps onto speculative uptrend/grind regimes — Jan 2018
melt-up, Feb-Mar 2021 meme era, the 2025-26 grind — where an overbought single name is a
momentum leader, not an exhaustion candidate, and the fade gets steamrolled. Plausible ex-ante,
though constructed with the data in hand; the honest label is "mechanism-consistent, not
mechanism-derived."

**Replay bookkeeping (NOT validation).** A 0.5x mult on 21≤score<44: totR +336.6 → +322.7
(−13.9R, −4%), avgR per unit risk +0.411 → +0.458, worst R-drawdown −10.4 → −10.0 (negligible).
A 0.75x mult costs half that (−7.0R) for half the concentration relief. The band still carries
positive total R (+27.8R over 230 trades), so zeroing it is strictly unsupported.

**Shrunk-Kelly sizing.** Book precedent: ~1.5σ (midterm) → 0.75x, not full-conviction. This
evidence is ~2σ effective (3σ headline, discounted for edge sensitivity, the equal-weight null,
and the reconstruction caveat). That supports **0.75x, not 0.5x**. 0.5x would require the
~3σ headline to be taken at face value across all its fragilities.

## Q4 — Is the 55+ recovery just 2022?

**No — 2022 is the *worst* 55+ year.** 55+ by year: 2017 +2.00 (1), 2020 +0.33 (2), 2021 +0.64
(12), **2022 −0.23 (9)**, 2024 +0.74 (10), 2026 +0.54 (11). Excluding 2022: avgR **+0.658
(N=36)**; excluding 2020 and 2022: +0.677 (N=34). Block bootstrap 55+ vs 21-44: z=+1.57 all
years, **z=+2.69 ex-2022**. Both paths participate (P1 +0.421, P2 +0.538). The recovery is
broad-based across 2021/2024/2026 and consistent with the short-side mechanism above and with
the established book contrast (the strategies that die at 55+ are the *long* index dip-buyers).
N=45 is small, but the direction is unambiguous: applying the pending book-wide taper
(0.5x by 60) to OVS would tax its second-best regime.

## Verdict

**Keep OVS exempt from the book-wide fragility ramp — the exemption at the top end (score ≥44)
is affirmatively correct, not just unproven.** Its 55+ edge (+0.48 avgR, not 2022-driven,
z≈+2.7 vs mid-band ex-2022) is the opposite of the non-OVS book, matching its short-side
mechanism.

**One specific change: add an OVS-only 0.75x multiplier when the live score (63d, 10d MA) is
in [21,44).** Trade-weighted, monthly-clustered evidence is z≈−3.0 and LOYO-stable, surviving
removal of midterm years (z≈−2.4) — but edge sensitivity (~1.8σ at a 25 lower edge), the
equal-weight monthly null, and the fragility-reconstruction lookahead caveat discount it to
~2σ effective. By this book's shrunk-Kelly convention that buys a 0.75x partial tilt, not the
0.5x floated in the prompt. It stacks with the midterm 0.75x to 0.5625x — acceptable given the
midterm concentration in this band is part of the observed damage. Expected cost if the effect
is pure noise: ~7R over a decade; expected concentration relief in trade-heavy mid-band months
(the 2026-01/2026-06 pattern) is the actual point.

Secondary observation (for the P2 track, not actioned here): backtest P2 has no calm-band edge
(−0.04 avgR over 81 trades where fragility is lowest) and no fragility structure; its retirement
live looks better-supported than the ledger's aggregate +0.11 suggests.

## Caveats

- Fragility history is a current-vintage reconstruction; composite edge weights carry
  calibration lookahead (established caveat, inherited here).
- Band edges [21,44) come from the same decile framework the book-wide study used —
  the specific 21 lower edge is the most favorable cut (25 → z≈−1.8). The tilt threshold
  inherits that selection.
- OVS overflow tier is single stocks from today's universe (survivorship); no reason it
  biases one fragility band over another, but levels are inflated.
- Equal-weight monthly tests show no band effect anywhere for OVS, including the 0-3 peak;
  the entire result is trade-weighted. If you require the equal-weight standard used for the
  non-OVS finding, the correct verdict is "keep full exemption, no change."
- 55+ and 44-55 cells are N=45/55; those conclusions are directional, not precise.

## Adversarial verification

Independent recompute 2026-07-02 (`verify_ovs-exemption.py`, fresh code, fresh join, fresh
bootstrap with own seed, 10k reps). Bottom line: **every decisive claim reproduces within
seed/rounding tolerance.** No lookahead beyond the already-flagged fragility-reconstruction
caveat, no clustering violations, no cherry-picked window found. Details and the few
divergences:

- **Join / N.** I get **N=821** (P1=543, P2=278, midterm=194), window 2016-07-25..2026-06-30,
  vs their 820/542. The single extra trade is **KMB 2017-02-16 (R=-2.25, P1, EOD-DD, score
  6.8)** — it lands in the 3-21 band, which is why their 3-21 row reads +0.382/N=130/totR 49.6
  vs my +0.362/N=131/totR 47.4, and their base totR 336.6 vs my 334.3. Immaterial to every
  conclusion; note the exclusion slightly *flatters* the non-mid bands, i.e. it did not help
  their dip case. Size_Mult-vs-recomputed-gap agreement: **100.0%** on all 821 joined trades
  (and full-history split 687 P1 / 420 P2 / 2 NaN matches their 687/422 with NaN counted as P2).
- **Band table.** Exact: 0-3 +0.600 (360), 21-44 +0.121 (230, totR +27.8, 28% of trades),
  44-55 +0.390 (55), 55+ +0.481 (45). P1/P2 x band matrix exact (P1 0-3 +0.785 N=279 ->
  21-44 +0.167 N=135; P2 0-3 -0.036 N=81).
- **Bootstrap.** 21-44 vs <21: diff **-0.416R, z=-2.94, p_emp=0.008** (theirs -0.422/-3.02/
  0.006 — seed noise). Vs 0-3: -0.480, z=-2.87. LOYO worst **z=-2.14** (drop 2023; theirs
  -2.18, drop-any-year range mine -2.14..-3.23). Non-midterm-only **z=-2.33** (theirs -2.35).
  Edge sensitivity confirmed: [25,44) z=-1.66 (theirs -1.70); 21-25 sliver N=46, avgR -0.186.
- **Equal-weight monthly null: even MORE null in my recompute.** t=**-0.10, p=0.92** (theirs
  -0.34/0.74; the one-trade join diff moves it). The claim "no signal under the book-standard
  test" is if anything understated.
- **One shading to note:** drop-2025+2026-together gives me **z=-1.78** vs their -1.84;
  calling that "still ~2 sigma" is a touch generous — it is below 2. Disclosed in their text,
  but the 2025-26 grind carries real weight in the result.
- **P1/P2.** P1 mid vs calm: diff **-0.618, z=-3.32**, LOYO -2.25..-3.46 (theirs -3.33,
  -2.24..-3.49). P2: z=**+0.39** exactly, calm-band -0.036 (N=81) exactly.
- **Midterm confound.** Mid-band midterm share 40.9% vs base 23.6% (theirs 23.7%); midterm
  mid-band +0.022 (N=94). Exact.
- **55+ / 2022.** By-year table exact: 2022 **-0.228 (N=9)** is the worst 55+ year; ex-2022
  **+0.658 (N=36)**, bootstrap z vs mid-band **+2.66** (theirs +2.69). Caveat I'd add: the
  +2.7 figure conditions on removing the worst year; all-years z is only +1.59 (they did
  report 1.57), so "55+ edge" rests on N=45 and is directional, as they say. Their defensive
  framing (it's not 2022-driven) is nonetheless correct.
- **Replay.** Delta **-13.9R** exact; avgR/unit-risk 0.407->0.454 (theirs 0.411->0.458,
  one-trade offset); worst R-DD **-10.4 -> -10.0** exact; 0.75x cost **-6.9R** (~7R claim OK).

Residual objections that survive (none fatal, all disclosed by the researcher): (1) the
[21,44) cut inherits selection from the decile framework and the bootstrap p is not
selection-adjusted — the 21->25 edge halving the z is the honest measure of that; (2) the
whole result is trade-weighted and vanishes equal-weight, so per-episode the effect is thin;
(3) fragility history is a current-vintage reconstruction (inherited caveat). All three are
priced into their ~2-sigma-effective grade. **Verdict: findings confirmed; the 0.75x-not-0.5x
recommendation is consistent with the evidence grade and the book's shrunk-Kelly precedent.**
