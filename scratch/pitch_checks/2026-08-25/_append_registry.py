"""Append the 2026-08-25 sweep's reusable findings to the negative registry."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
p = ROOT / "data/pitch_negative_registry.md"
text = p.read_text(encoding="utf-8")

ADD = """
## Method traps (2026-08-25, from an 11-candidate sweep that killed all 11)

- **The RECON contaminated the whole map: `px.pct_change(n)` on a wide
  union-calendar panel pads foreign-calendar holes into synthetic zero-return
  sessions.** This is the trap `pitch_lab._valid_pct_change` was written for
  (2026-08-19) and it was walked into at stage B1 rather than inside a check,
  so it propagated into every spread premise the surface map quoted and into
  three checkers' briefs. Found INDEPENDENTLY by two checkers within minutes.
  Measured damage: EEM-EFA 63d **-7.57pp / PIT 1.98 -> -4.67pp / PIT 5.56**,
  FXI-EEM 63d **+4.24pp / PIT 100.0 -> -0.03pp / PIT 90.5** (error +4.27pp and
  9.5 percentile points), OIH-XOP 63d -18.52 -> -16.78pp, SMH-SPY 63d
  -10.17 -> -7.78pp. **C8 died on it outright** - its named extreme did not
  exist and its trigger had last fired ten calendar days earlier. Two
  properties worth keeping: only windows SPANNING a hole are affected, so
  same-calendar US pairs and short lookbacks reproduce exactly (XLV-XLK 5d,
  GDX-GLD 21d, XLU-TLT 21d all 0.00 error); and **the magnitude of the error is
  a property of the PANEL, not the pair** - padding the recon universe (which
  carried `^TNX` and `DX-Y.NYB`) gives +4.24pp on FXI-EEM while padding a
  US-ETF-only panel gives -0.03pp. Rule: build every premise on valid sessions
  per ticker, and state the basis. A premise is the one number a whole morning
  rests on.
  (b1d_premise_padfill_audit.py, c3b_c8_premise_forensic.py,
  c5_padfill_basis_verification.py)
- **A multi-day spread trigger can be a single-day gap trigger wearing a
  longer lookback, and EPISODE CONTAINMENT is the test, not day-level mask
  overlap.** Day-level overlap between the 5-day XLV-XLK rung and the dead
  2026-08-19 one-day >=3pp gap form read only 32.3% / 17.1%, which looks like
  a different object. Episode containment says otherwise: **95.4% of rung>=8
  days and 100.0% of rung>=9/10 days contain a >=3pp single-day gap**, and the
  biggest single day is a median **48% of the whole 5-day spread**. Today's
  five daily gaps were [+4.07, +4.57, -1.58, +1.18, +1.82] and the +4.07pp
  print was **2026-08-18, the exact session the corpse was built on**, with it
  and 08-19 making 86.6% of the 9.98pp headline. When a window trigger is
  suspected of re-skinning a point trigger, ask what fraction of its episodes
  CONTAIN the point event, not what fraction of days coincide.
  (a1b_c1_round2.py)
- **The count-matched single-instrument control is the cheapest way to kill a
  spread candidate, and it should run before the battery.** Take the spread
  trigger's day count, then take the same number of days by the LONG leg's own
  drawdown alone. XLK 5d <= -9.82% (count-matched) pays **+1.069%** against the
  XLV-XLK spread trigger's **+0.555%**. The defensive leg was not adding
  information, it was subtracting selectivity - and the gate framing (rotation
  vs no-rotation, +0.564% vs +0.405%, worth +0.159pp) understated how badly,
  because it holds the threshold fixed instead of the sample size.
  (a1_c1_xlk_rotation_r1.py)
- **A registry BY-PRODUCT is not pre-specified until you check the prose
  against the script, and here they disagreed by 5 rank points.** The
  2026-08-24 kill report recorded "on days tech's 63-day rank is
  bottom-quintile while the index's is not, QQQ LONG pays +0.508% at h=5". The
  code behind that line used **SPY r63 >= 25**; the prose said 20. **The cell
  fires today only under the prose threshold** (SPY r63 is 23.8), and the
  +0.508% headline was additionally the gap-10 decluster, where gap 5 gives
  +0.297%. Edge over drift decays monotonically with the decluster gap
  (+0.207 / +0.187 / +0.128 / +0.142 / +0.066 pp at gaps 1/5/10/21/63, t 2.67
  -> 0.48). Two rules: quote a by-product from its SCRIPT, and a by-product
  written down to explain why something else died has not been falsified just
  because it was written down.
  (e1_c11_qqq_laggard.py)
- **A conditioning clause can be ANTI-selective, and the giveaway is that
  loosening it toward inert improves the number.** "QQQ 63d rank <= 20" alone
  is 337 episodes at +0.452%; adding "and the index is not" discards **226 of
  337** to move the mean **-0.036pp**, and the discarded half pays **+0.508%**
  against the kept half's +0.416%. The threshold ladder confirms the direction:
  SPY > 30 gives +0.014%, > 25 gives +0.222%, > 20 gives +0.416% - the closer
  the gate comes to doing nothing, the better it looks. When a gate's value
  rises monotonically as it approaches inert, it has negative information and
  the trade is the ungated parent.
  (e1_c11_qqq_laggard.py)
- **Today's fragility dial had no precedent in ANY cell examined, and that
  became the morning's most reusable single filter.** ma10(63d) = **89.5, the
  99.4th percentile of the entire 2016+ series**, with only 21 of 2453 days
  >= 85 and exactly one prior episode (2021-12-20..2022-01-11, max 95.2).
  Four independent cells were asked and all four answered the same way: C11's
  support **tops out at dial 80.4** with zero episodes >= 85 and the parent's
  [70,85) bucket at -0.806% on a 28.6% hit; C6's [80,200) bucket is **N=3 at
  -1.963%, 0% hit**; C8 has **never been observed above 85** and is 0-for-2
  above 65; C3's episodes top out at 68.6. Ask "what is the maximum dial this
  cell has ever been observed at" early - when the answer is below today, the
  candidate is out of sample regardless of its other statistics.
  (e1c_c11_cochranq_parentdial.py, a4b_c6_round2.py, c3_c8_eem_efa_r1.py)
- **An earnings anchor at the PRE-PRINT session is the worst rung on its own
  ladder, which is the second independent earnings-anchor ladder failure.**
  SMH into NVDA, ungated h=1: the true anchor ranks **16 of 16** at -0.286%
  against a +0.122% ladder mean over offsets -10..+5, with offset -10 paying
  +0.550% at t=2.35. The offset placebo ladder is now **11-for-11** in this
  repo and has been applied to macro anchors and single-name anchors alike.
  (b1_c2_smh_nvda_print.py)

## Cells swept and empty (2026-08-25)

- **The five-day tech-to-defensive rotation at a 99.6th-percentile extreme, in
  all four expressions.** The tape handed over a genuine one-in-250-day
  reading (XLV-XLK 5d = +9.98pp, 99.6th full-sample percentile) and every way
  of trading it failed. **Long XLK**: see the count-matched and containment
  entries above; additionally the rung ladder INVERTS where today sits (+0.566
  / +0.424 / +0.180 / +0.555 / **-0.408 / -0.490%** at rungs 5/6/7/8/9/10pp,
  today 9.98pp), the definition is fragile (at today's 99.64 percentile all
  three lookbacks are negative, at the 95th all three are positive), and it is
  a BEAR-tape selector for once - 20.0% of rung>=8 days sit above SPY's 200d
  against a 71.6% base, while today SPY is +8.1% above, so the near-high
  subclass is 5 days / 3 episodes all in 2026. **Short XLV**: -0.069% at h=3
  and +0.009% at h=5 (-1.4x and 0.2x cost, 34-44); the apparent edge is only
  that short-XLV's drift is -0.188%, and XLV ranks **9 of 9** by |t| across
  the SPDRs. **Cross-sectional losers-vs-winners**: today's dispersion is the
  84.5th PIT percentile, not an extreme; the gate is worth +0.009pp and the
  live band pays +0.017% against an unconditional +0.096%; 0.5x cost. **XLI
  intact-trend washout**: the "intact trend" clause is a negative-value filter
  (broken-trend complement +0.418% beats the joint cell's +0.234%), and the
  real h=7 shelf fails the family test at Cochran Q p=0.789 with permutation
  P=0.268. Note the four are ONE position: h=5 return correlations run 0.780
  (XLK/xsec), 0.715 (XLK/XLI), -0.586 (XLK/short-XLV), with OLS R-squared
  0.34-0.61.
  (a1_c1_xlk_rotation_r1.py, a1b, a1c, a2_c10_short_xlv_r1.py,
  a4_c6_xsec_reversal_r1.py, a4b, a3_c9_xli_washout_r1.py, a3b, a3c)
- **The NVDA print as a tradeable anchor for the semis complex, the first
  single-name earnings anchor examined in this repo.** Ladder kill above. Also
  settled: the edge is **not NVDA-specific and the gate is incoherent across
  the family** - the same rule on the other five big semi prints has the
  relative-low gate HELPING AVGO (+0.97pp) and MU (+1.32pp) while HURTING AMD
  (-0.50pp), INTC (-0.73pp) and TXN (-0.47pp), with pooled non-NVDA ungated
  h=3 at +0.085% over 465 prints. The one nominally positive object is the
  POST-print entry (h=1 +0.351%, t=2.00, edge +0.288pp) and it is offset +2 on
  the ladder, ungated, and 4.6x cost - under the bar and not the pitched
  trade. Tail for anyone revisiting: SMH's reaction-day sd is 2.42%
  full-history and **3.41% since 2020**, p01 -6.15%.
  (b1_c2_smh_nvda_print.py, b1b, b1c)
- **"The bond proxy was dumped and the bond was not" - the 2026-08-20 credit
  gate re-skinned on utilities, and it INVERTS across its own threshold
  walk.** Gate value by XLU rank21 rung: **+0.522 / +0.086 / -0.230 / -0.100pp
  at <= 2/5/10/15**. Reference class puts XLU **8 of 9** (Cochran Q 2.80 on
  8 df, p=0.946, I-squared 0, common excess +0.329pp, max is XLV). The
  mechanism is falsified inside its own window: the state where TLT WAS hit
  pays **+0.858% at h=5 on a 75.0% hit** over 28 episodes, sign p 0.006 - the
  rates-repricing seller is the good one, not the equity rotator. Not a
  duration trade either (beta_TLT -0.41, duration-neutral form +0.002%).
  100% mask overlap with the dead 2026-08-12 rank21<=5 cell. **Utilities are
  now dead in eight expressions.**
  (b1_c3_xlu_washout_tlt_fine.py, b1b, b1c)
- **Oil services versus E&P at a 63-day extreme, the first intra-energy pair
  examined here.** Pair wrong-signed at h=1/2/3/5 (-0.209% at h=5, 35-44,
  sign p 0.870, -1.7x cost); the one positive horizon is one episode
  (drop-best -0.018%, drop-best-2 -0.200%). Leg attribution: long OIH +0.763pp
  against short XOP **-0.439pp**, so the naked long pays +0.934% at 15.6x cost
  against the pair's 1.4x - the 2026-08-24 SPY/QQQ and 2026-08-19 EFA/SPY
  failure for the third time. Complex is one factor (PC1 83.0%, **1.42
  effective names of 4**). Book overlap: 13 energy-family ledger signals in
  these windows are **all Overbot Vol Spike SHORTS at avgR +1.083**.
  (c1_c4_oih_xop_r1.py, c4_book_overlap_and_confirms.py)
- **Fading a 99.6th-percentile gold-miner thrust, and it is not adjacent to
  the 2026-08-18 GDX/GLD corpse, it IS it.** P(corpse mask | this mask) =
  **0.924**, above the 91% same-object line, and that corpse's `outright`
  vehicle is literally `-r_gdx`. Wrong-signed at all ten horizons (-0.844% at
  h=5, 17-24, -16.9x cost); every live gate worsens it (today's r21>=37% rung
  **-2.958%**); the 8 genuinely-new days are the worst subsample (-3.919% at
  h=10, 1-for-6). The sign is on the LONG side (+0.844%, edge +0.575pp).
  Premise itself was sound - only **32 of 5,075 days** ever ran hotter. Book
  overlap quantifies a claim this registry had asserted from memory: **19
  miner-name ledger signals in this state, all 19 OVS shorts at avgR +0.492,
  27.1% of that family's lifetime signals in a state covering 4.1% of
  sessions = 6.6x concentration.**
  (c2_c7_gdx_thrust_fade_r1.py, c4_book_overlap_and_confirms.py)
- **Emerging versus developed at a 63-day extreme.** Killed on false premise
  first (see the pad entry). Granting the mask: short EFA carries **92% of the
  h=5 excess** while long EEM contributes +0.020pp and the attribution FLIPS
  SIGN at h=10; top-2 episodes are both late 2008 at **107% of total**; 2013+
  pays +0.047% at a 45.2% hit; the live shape with EFA near its high pays
  -0.113%. Naked long EEM edge **-0.008pp** - the signal says nothing about
  EEM. The dollar-regression test is the only one it passes.
  (c3_c8_eem_efa_r1.py, c3b)
- **Month-end on FX, which completes the month-end anchor to three asset
  classes (equities closed 2026-08-24, rates suspended 2026-08-24, FX closed
  here).** The mechanism is the 4pm London fix rebalancing flow and it is
  falsified in its own window: DXY's **ME-0 session pays -0.55 bp at a 45.6%
  hit** against an all-days base of +0.10 bp, and **+3.57 bp (wrong sign) from
  2020**; the window's total comes from ME-1/-3/-4, sessions the story does
  not name. The pre-specified signed regression on relative US-vs-foreign
  equity performance is slope -0.0157, **t -0.75, R-squared 0.0019**, with
  non-monotone terciles. The ME-5 spike (+5.03 bp, 55.9%) is noise: rotation
  null over the same 16-cell walk gives **P(max |t| >= 1.93) = 0.523**. Cost
  4.11x on the index and **0.38x and wrong-signed on UUP**, the only vehicle
  that trades as an ETF. August x midterm is N=7 at -0.156%.
  (d1_c5_monthend_fx_r1.py, d1b)

### Calendar finding, filed because it repeats yesterday's

- **For the SECOND consecutive session the macro anchor set was empty**, which
  makes 2026-08-24's note a pattern rather than a one-off. Jackson Hole is
  closed on seven asset classes and today was JH-3; post-opex is closed in
  both directions; NFP at +8 td sits at the horizon cap with its one live cell
  midterm-parked to 2027-01; CPI, PPI, FOMC and quad witching are all beyond
  +11 td. Month-end was the only non-macro anchor left and FX was its last
  unswept class, now closed. **The two calendar-anchored candidates a morning
  like this can still generate are a single-name earnings date and a
  flow-calendar position** - both were tried here and both died, so the
  inventory of anchors available in late August of a midterm year is now
  documented as exhausted.
"""

p.write_text(text.rstrip("\n") + "\n" + ADD, encoding="utf-8")
print("appended", len(ADD), "chars ->", p)
