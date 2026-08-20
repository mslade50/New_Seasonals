"""Post-publish: append this morning's reusable kills to the negative registry."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REG = ROOT / "data" / "pitch_negative_registry.md"

BLOCK = """
## Method traps (2026-08-20, from a 10-candidate sweep that killed all 10)

- **A cross-sectional statistic needs a point-in-time percentile too, and the
  full-sample one can be the whole candidate.** The 2026-08-19 entry made this
  point for a single instrument's rank; it fires identically on a
  cross-sectional one. Yesterday's cross-sectional sd of daily returns across
  the 218-name tape reads the **89.3rd percentile of full history** and the
  **88.8th of a trailing 252 days**, and the dispersion cell was built on a
  >= 90th-percentile gate, so the state the morning was designed around did not
  actually fire on the only definition that is knowable in advance. A
  survivorship-free 11-sector-ETF cross-section reads 87.6. Compute the PIT rank
  of any breadth or dispersion measure before treating it as a trigger.
  (a1_c1_dispersion_round1.py, a1b_c1_round2.py)
- **A dose response whose immediately-lower neighbour is SIGNIFICANTLY
  wrong-signed is a stronger kill than any multiplicity number, and it is
  cheaper.** The TLT thrust cell at a 1.5% rung pays +0.638% at h=2 on 17
  episodes (13-4, sign p 0.0245, bootstrap 0.000); the same cell at a 1.0% rung
  **LOSES -0.165% on 33 episodes at a 36.4% hit (sign p 0.960)** and the excluded
  [1.0%, 1.5%) band is **-0.241% on 26 at 30.8% (sign p 0.986)**. The ladder
  peaks exactly at the pitched value and decays both ways. The rotation
  permutation over the 125-cell grid that was actually walked only said P(grid
  max |t| >= 3.30) = 0.167, i.e. suggestive; the parent said the sign is
  manufactured by the rung. Run the parent before pricing the search.
  (a2_c2_tlt_round1.py, a2b_c2_tlt_round2.py)
- **A rank extreme and a magnitude extreme select different populations, and on
  the dollar they disagree in SIGN.** The 21-day DXY washout pays **+0.237 /
  +0.165 / +0.509pp** excess at h=3/5/10 under `pct_rank(21) <= 2` (35 episodes,
  records 18-17 / 16-19 / 19-16) and is NEGATIVE under every magnitude threshold
  at h=3 and h=5: <=-2.32% gives -0.096 / -0.082 / -0.116, <=-3% -0.064 / +0.045
  / -0.010, <=-4% -0.098 / -0.079 / +0.019, <=-5% **-0.180 / -0.050 / -0.448**.
  The 2x2 locates it exactly: rank-extreme AND magnitude-extreme pays +0.060 /
  -0.127 / +0.194pp, magnitude-extreme but rank-ordinary pays -0.108 / -0.087 /
  -0.393pp, and the entire positive sign lives in **rank-extreme but
  magnitude-ORDINARY, +0.162 / +0.214 / +0.638pp** — the rank gate's only content
  is that the trailing year was quiet. So quote the level the rank bought AND the
  population it bought it from: the trigger set's median 21-day move is
  **-4.19%**, and 2026-08-20's rank of 0.79 buys **-2.32%**, the **91.3rd
  percentile of that population by depth**. The near-neighbour ladder is a knife
  edge on the GATE rather than the lookback: rank<=5 stays positive (+0.153 /
  +0.159 / +0.310) and **rank<=10 is wrong-signed at all three horizons**. The
  cell is separately 105% top-2 episodes at h=5 and its mechanism runs backwards
  inside its own trigger set (deep half +0.070% at a 39% hit against shallow half
  +0.276% at 53%). (c6_rank_vs_mag.py, c6b_registry_isolation.py)
- **The alphabetical placebo is now 6-for-7, and recording the miss matters more
  than recording the hits.** On the bank-breadth cell the signal-picked four
  names BEAT the alphabetically-first four by +0.589pp at h=3 and +1.333pp at
  h=10, market-relative — the first time the placebo has failed to kill a
  selection rule in this repo. The cell died to its reference class anyway
  (P(max group excess >= banks) = 0.761 across 12 industry groups). A placebo
  pass is not evidence of an edge; it only removes one way of being wrong.
  (a3b_c7_placebo_refclass_book.py)
- **An instrument that changed leverage mid-sample manufactures an inversion out
  of nothing, and this is the second time it has bitten in four sessions.** The
  "August post-opex short-vol cell inverts" reading is **entirely** the -1x SVXY:
  August h=3 splits into **pre-2018-02-28 -3.231% (N=6)** and **post-break
  +1.141% (N=8, 75% hit)**, and post-break August sits +0.934pp ABOVE SVXY's own
  drift. Measured rather than assumed: pre-break daily sd 4.56% and worst day
  -82.96%, against 2.32% and -21.43% after. Any SVXY grid that does not split at
  the break is reporting two securities as one. (b2_c4_postopex_vol_round1.py,
  b2b_c4_round2_and_book_finding.py)
- **A positive mean beside a negative median and a sub-50% up-rate is a
  left-tail description, not a direction, and a time exit cannot harvest it.**
  Spot ^VIX after August opex reads +1.182% at h=3 with a **median of -0.502% and
  a 42% up-rate**; top-2 episodes are **199% of total** (2015-08-20 alone
  +88.19%), drop-1 takes it to -2.299% and drop-2 to -3.364%. The same shape
  turned up independently on the TIP/IEF pair, where the ex-2008/09 h=5 cell has
  a **71% hit rate at a -4.09 bps mean**. Report the median beside the mean on
  any cell whose story is about tails. (b2b_c4_round2_and_book_finding.py,
  c9b_residual_and_era.py)
- **The offset placebo ladder finally missed, and the cell died anyway.** The
  ladder went into this morning 9-for-9 at closing event anchors. Long crude at
  Jackson Hole minus 6 ranks **1 of 16 at h=10** on USO and on CL=F, and beats
  the anchor-tdom-weighted unconditional August expectation properly (+2.145%
  observed against +0.632% expected on USO). It was killed by concentration
  instead: **dropping the best three years takes the h=10 excess from +1.552pp to
  -0.056pp**, exactly the unconditional late-August window. Lesson for future
  mornings: the ladder tests whether the ANCHOR is special, not whether the
  effect is real, and passing it buys one kill fewer rather than a survivor.
  (b3_c8_crude_jacksonhole_round1.py, b3b_c8_crude_round2.py)

## Cells swept and empty (2026-08-20)

- **Cross-sectional dispersion as a directional signal on the index, and it is
  genuinely NOT the dead fragility dial re-skinned.** Registry-collision check
  run properly for once: only 72 of 162 cell days have a dial reading at all
  (the series starts 2016-07-05), only 8 have ma10(63d) >= 50, only 3 sit on the
  post-2026-07-02 PIT vintage, and "cell AND dial >= 50" is N=7 — so this is the
  dispersion COMPONENT and the component is negative for the short. Gate
  attribution: dispersion alone pays the short **-0.649% at h=10 over 369
  episodes (edge -0.271pp)**, dispersion-and-NOT-quiet is worse at -0.766%, and
  the quiet-index leg alone is worth +0.061pp, so the joint +0.614pp is the
  intersection of a significantly negative leg and a nothing leg. High component
  dispersion is followed by SPY going UP relative to baseline, the opposite of
  the correlation-snapback story. Era pre-2018 +0.622% against 2018+ -0.418%,
  2008 alone +74.69pp of a +31.16pp total, and swapping the survivorship-selected
  tape for 11 sector ETFs flips the sign outright. Book-overlap by-product: 159
  ledger trades signal on the 162 trigger days and **112 are SHORT, 100 of them
  Overbot Vol Spike**, earning +$76.2k flat against +$16.6k for the 47 longs —
  where the book meets this state it is already short and profitably so.
  (a1_c1_dispersion_round1.py, a1b_c1_round2.py)
- **The run OUT of August opex, on IWM and on SPY, which closes the opex anchor
  in both directions.** The run INTO it died on 2026-08-07; this is the
  complement. IWM's August h=10 +1.603% ranks **5 of 120** in the month x horizon
  x vehicle grid it came from (grid excess sd 0.735pp, 20 of 120 cells clear
  |1.0pp|), the offset ladder disagrees with itself across adjacent horizons
  (true anchor 5 of 16 at h=3, 4 of 16 at h=5, 1 of 16 at h=10), August ranks
  only 5 of 12 months at h=5 and 3 of 12 at h=10, and the unconditional August
  tdom 10-14 window pays +0.852% over 130 starts against the anchor's +1.603%
  over 26. Midterm years pay **+0.393%**, below that unconditional window and
  roughly at IWM's plain 10-day drift. And the live state inverts it: with IWM
  near its 52-week high the anchor pays **-0.405% at h=10 over 87 anchors** with
  the opex gate contributing **-0.341pp**, independently reproducing the
  2026-08-17 "opex gate is an INVERTER" finding on a different instrument.
  (b1_c3_iwm_opex_round1.py, b1b_c3_iwm_round2.py)
- **The opex overnight/intraday decomposition.** Cost kills it before anything
  else: SPY's overnight legs sum to **+10.76 bps across the five post-opex nights
  against 45 bps of MOC-to-MOO cost, 0.24x** a 5x bar, and the best single night
  is 1.6x. The offset ladder puts the true anchor **10 of 16 at 1 night and 11 of
  16 at 5 nights** on SPY, 10 of 16 and 16 of 16 on IWM. And the two index
  vehicles disagree about the sign: against a tdom-matched non-opex placebo SPY's
  overnight excess is positive at every horizon (+0.100 to +0.159pp) while IWM's
  is negative at every horizon (-0.128 to -0.222pp), with IWM's INTRADAY leg
  carrying the sign instead. The dealer-hedging mechanism remains unfalsifiable
  in this repo — `option_surface_history` holds 1 row and
  `option_positioning_history` 90, all dated 2026-08-05.
  (b4_c10_opex_overnight.py, b4b_c10_overnight_ladder.py)
- **The dollar-washout trade expressed through EM, closing the country family
  from the FUNDING side.** Prior members broke on decoupling (EWZ twice, FXI,
  SMH/QQQ) or on sustained leadership (EFA); this one is a macro driver applied
  to the whole class, and it dies to the reference class like the rest.
  Permutation over 11 clean EM/intl vehicles on the same 19 episodes, two
  independent nulls (random anchors at min gap 21, and a circular shift
  preserving the trigger set's own spacing), 20,000 draws: **P(max name excess >=
  KWEB's +1.209pp) = 0.283 at h=5 and 0.641 at h=10**, and at h=10 KWEB's
  +0.764pp sits BELOW the null's median best-of-11 of +1.098pp. Being the only
  positive name of thirteen is what the null does on a correlated high-vol class.
  The mechanism's own longer test fails: FXI over 32 episodes back to 2004 is
  +0.092pp at h=10 (t +0.60) splitting **pre-2013 +1.377% against 2013+
  -0.255%**, and YINN — a 3x FXI, so the highest-beta version of the identical
  funding story — is +1.182pp at h=5 but **-1.716pp at h=10 on a 34.8% hit**.
  Two attack items resolved FOR the candidate and changed nothing: the trigger is
  not risk-on selection (SPY above its 200d on 72.2% of trigger episodes against
  a 71.3% base rate) and cost clears easily. (c5_round1.py, c5b_refclass.py,
  c5c_magnitude_fxi.py, c5d_refclass_clean.py)
- **Breakevens as a tradeable pair, long TIP against short IEF.** First
  examination of TIP in this repo and it fails the way the duration-pair family
  always has: the label says inflation, the arithmetic says duration. Beta(TIP on
  IEF) is **0.698** full sample (stable 0.714 / 0.710 / 0.672 by era), so an
  equal-dollar pair is a 0.30-unit duration short. Leg attribution at h=5: TIP
  alone +9.2 bps of excess, the short IEF leg removes **92%** of it, and the
  duration-neutral pair is **+0.3 bps = 0.91x** its own 6 bp round trip; the best
  full-sample cell anywhere is 1.24x. Adjacent horizons disagree about which leg
  carries it (h=3 is all TIP, h=5 is nothing), the EFA/SPY signature from
  2026-08-18 on a duration pair. The gold gate is a filter that does not filter
  (**+1.17 / +1.15 / -0.64 bps alone**, swinging 19 bps across three adjacent
  horizons while costing 22 of 46 episodes), and the joint cell flips era sign
  from **+37.76 bps pre-2018 to -27.04 bps** after, with three 2008-09 episodes
  carrying +117.86 bps of the h=3 total and being **opposite-signed at h=10**.
  Mechanism check: on joint-state days the residual's contemporaneous daily
  correlation is **+0.536 with SPY** against +0.212 with GLD and +0.089 with the
  10y yield level. (c9_round1.py, c9b_residual_and_era.py, c9c_h10_parent.py)
- **The bank-breadth washout inside an intact trend, and the insurance premise
  does not replicate.** On banks the intact-trend half is **+0.225% at h=5 on
  XLF** rather than the -0.789% loser the 2026-08-14 insurance cell described, so
  there was no inversion to trade; the short pays **-0.9 / -22.5 / -51.1 bps at
  h=3/5/10** against a 2 bp round trip. Reference class across 12 industry
  groups: fixed-effect common excess +0.096pp, **Cochran Q 6.65 on 11 df**,
  cross-group excess sd **0.394pp against a mean sampling SE of 0.552pp
  (dispersion ratio 0.71)**, and P(max group excess >= banks) = **0.761**. That
  replicates 2026-08-14's Cochran Q result on a wholly different set of groups,
  so "no industry label carries information" is now a two-sample finding. Two
  tests it PASSED, filed because they are unusual: tape over-selection runs the
  right way (trigger days below SPY's 200d **20.8% against a 25.4% base rate**),
  and the alphabetical placebo failed to kill for the first time. The KRE/XLF
  pair is parked on the watchlist at 1.5x cost ex-crisis. (a3_c7_banks_round1.py,
  a3b_c7_placebo_refclass_book.py, a3c_c7_kre_pair_teardown.py)

### Book finding, filed here because it is about the sleeve rather than a pitch

- **August must NOT be carved out of `V4_POSTOPEX_VOL` the way September is.**
  A recon grid that pooled across the 2018-02-28 SVXY leverage break appeared to
  show August inverting the post-opex short-vol cell (-0.73% at h=3 against the
  pooled +1.25%). It does not. Post-break, V4 exactly as specified — long SVXY,
  MOC on the opex close, exit MOC +3 — pays **August +1.115% over 8 anchors,
  5-for-8, median +0.471%, worst -1.51%**, against **rest-of-V4 +0.674% over 72**
  and **September -1.535% over 8 at a 0% hit rate** (0-for-8, bootstrap
  P(mean<=0) = 1.000). August beats the rest of the sleeve at exits +1, +3 and +5
  and trails slightly at +2 and +4. The whole "August inverts" impression is the
  pre-break -1x cell at -2.570% over 6, with 2015 alone at -20.41%. September's
  carve-out is confirmed and strong; no change to V4 is warranted. Honest caveat:
  August's post-break sample is 8 anchors and its bootstrap P(mean<=0) is 0.063,
  so it is not distinguishable from the rest of the sleeve in either direction.
  (b2b_c4_round2_and_book_finding.py)
"""

text = REG.read_text(encoding="utf-8")
if "Method traps (2026-08-20" in text:
    raise SystemExit("already appended")
REG.write_text(text.rstrip("\n") + "\n" + BLOCK, encoding="utf-8")
print("appended", len(BLOCK.splitlines()), "lines ->", REG)
