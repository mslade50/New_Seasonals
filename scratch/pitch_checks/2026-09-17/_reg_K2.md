# K2 adjacent registry (breadth, calendar anchors, placebo ladders)
576:- **Placebo ladders keep earning their place, and 4-for-4 became 6-for-6.** The
577-  Jackson Hole TLT cell (+1.162%, 24 events, 17-7, sign p 0.0320) ranks 14th of
578-  127 offsets, with off=-9 paying +1.369%; the 98th-percentile contango cell ranks
579-  10th of 21, with every offset from -4 to -10 paying +4.55% to +5.88% at a
580-  90-100% hit; the macro-vacuum cell ranks 6th of 17 at h=10 and 14th of 17 at
581-  h=5. In all three the ladder is a PLATEAU rather than a spike, which is the
582-  signature of month position rather than an event. (b1b_c4a_round2.py,
583-  a1_c1_termspread.py, a2_c5_macro_vacuum.py)
584-- **An "event" cell inside one month owes a MONTH-OF-YEAR control, not just a
585-  trading-day-of-month one.** TLT's 10td lag-1 forward return runs Nov +1.059%,
586-  Aug +0.494%, Jun +0.498%, Jul +0.451% against Oct -0.432%, Sep -0.220%, Apr
587-  -0.240%. That is large enough to swallow a +1% conditional mean whole, and it is
588-  what the Jackson Hole cell turned out to be: the unconditional Aug 6-16 window
--
597:  placebo ladder beats the real anchor. Check the instrument's trailing return on
598-  trigger days before believing a premium story. (a1c_c1_intersection_cell.py)
599-- **"Positive in 9 of 9 years" is not evidence on its own.** Exactly 1 of the 27
600-  reference-class tickers achieved it and the global null expects ~0.3, so the line
601-  carries p~0.3. A perfect year record at ~2 episodes a year is a small number of
602-  coin flips wearing an impressive sentence. (r3_registry_failure_modes.py)
603-
604-## Cells swept and empty (2026-08-13)
605-
606-- **The ten sessions into Jackson Hole, all three cross-asset legs.** Rates: the
607-  anchor is decoration on an August seasonal (above), the mechanism loses in its
608-  own window (entering after the conference is -0.204%, +1 to +3 sessions -0.607%
609-  to -0.848%), 2018+ is +0.590% at 4-4, the duration-neutral residual against IEF
--
736:- **An earnings anchor owes the same placebo ladder as a macro anchor, and the
737-  first one tested failed it.** The pre-print washout's true anchor ranked **3rd
738-  of 15** offsets (shift -13 at +0.396% and shift -4 at +0.381% both beat the
739-  real +0.331%), the whole ladder sitting in a +0.121% to +0.396% band, so true
740-  minus placebo mean was +0.084pp. Stripping the print left the same washout gate
741-  earning +0.200% on the same names. The earnings lane is not exempt from the
742-  ladder just because the event is company-specific. The ladder is now 7-for-7.
743-  (c3e_liquid_ladder.py, c3c_preprint_round2.py)
744-
745-## Cells swept and empty (2026-08-14)
746-
747-- **A cell that beats every control except the local one, and is LIVE today, so
748-  it will be re-found.** Long SPY with VIX's LEVEL in its bottom decile while SPY
--
939:  Same sign at h=3 and h=10. The anchor placebo ladder is now **8 for 8**: across
940-  90 big-box clusters since 2006 the true anchor ranks **17 of 19** offsets, true
941-  minus placebo -0.333pp. The tradeable 3-name basket clears the alphabetical
942-  placebo (+0.118pp) but pays 0.7-1.3x its 15 bp round trip and is negative
943-  inside the live state. (e1_c7_xrt_retail_cluster.py, e1b)
944-- **GDX's maximal 21d thrust, closed by the reference class.** The identical rule
945-  on 15 names gives a cross-name excess of +0.025pp with observed sd 0.665pp
946-  against a sampling SE of 0.686pp — **dispersion ratio 0.97**, so the entire
947-  spread is noise; GDX ranks 2 of 15 and permutation gives **P(max >= GDX's
948-  +1.171pp) = 0.582**, a below-median draw from the null. The parent's edge lives
949-  only in the [20,26)% band and today's +26.01% sits in the losing half (h=5
950-  -0.858% on 2-8); magnitude-only h=10 runs +0.535% / **-2.224%** / +1.982% at
951-  the >=20 / >=26 / >=30 cuts, the same threshold instability that killed the
--
1385:- **The offset placebo ladder finally missed, and the cell died anyway.** The
1386-  ladder went into this morning 9-for-9 at closing event anchors. Long crude at
1387-  Jackson Hole minus 6 ranks **1 of 16 at h=10** on USO and on CL=F, and beats
1388-  the anchor-tdom-weighted unconditional August expectation properly (+2.145%
1389-  observed against +0.632% expected on USO). It was killed by concentration
1390-  instead: **dropping the best three years takes the h=10 excess from +1.552pp to
1391-  -0.056pp**, exactly the unconditional late-August window. Lesson for future
1392-  mornings: the ladder tests whether the ANCHOR is special, not whether the
1393-  effect is real, and passing it buys one kill fewer rather than a survivor.
1394-  (b3_c8_crude_jacksonhole_round1.py, b3b_c8_crude_round2.py)
1395-
1396-## Cells swept and empty (2026-08-20)
1397-
--
1758:  shape appeared independently on new-high breadth the same morning (0 of 9
1759-  sectors at a high +0.317%, 2 of 9 +0.310%, **3 of 9 -0.009%**, >=5 -0.136%),
1760-  which is two unrelated constructions agreeing that a LITTLE of a thrust state
1761-  is bullish and a LOT is not. Walk the count ladder in both directions before
1762-  choosing k; the interesting cell is usually not the extreme one.
1763-  (b2_c7_energy_z10_cluster_r1.py, c2_c6_breadth_attribution.py)
1764-- **A cell can pass every robustness test and die because its MECHANISM decayed
1765-  while its total return did not.** The month-end TLT parent is the strongest
1766-  duration cell in this repo and it survived re-derivation: month-matched
1767-  +0.346% at t=3.72 over 288 anchors, a clean exit-placebo **SPIKE** (ME+3
1768-  +0.065 / ME+0 **+0.430** / ME-3 +0.205 / ME-9 +0.067) rather than the plateau
1769-  that killed the Jackson Hole and August anchors, holdout 2014-2026 **+0.463%
1770-  at t=3.56** which beats its own in-sample half, top-2 episodes 8% of total,
--
1910:- **Cross-sectional new-high breadth with the index off its high.** Premise
1911-  false (above) and the gate is a negative-value filter: breadth alone pays
1912-  +0.104% at h=5 against an all-days +0.192%, i.e. **-0.086pp against doing
1913-  nothing**, while the index-distance leg alone pays +0.266% and carries the
1914-  cell. Tolerance walk costs 77% of the edge on a 0.75pp nudge (0.25% +0.452%,
1915-  1.00% +0.106%), and the two universes disagree about the gate's worth by an
1916-  order of magnitude. Bull-tape selector: **100.0%** of tape trigger days sit
1917-  above SPY's 200d against a 71.6% base rate, with the trend split having N=1
1918-  episode below it. And today's regime is outside the sample — median trigger
1919-  ma10(63d) is **24.8** with an all-time max of 80.6 against today's **89.5**,
1920-  and split on the live exposure-leg rule the edge is entirely in the
1921-  complement (leg-OFF +0.008% at a 50.0% hit, leg-ON +0.754% at 81.0%, t=3.10).
1922-  (c1_c6_newhigh_breadth.py, c2, c5)
--
2022:  +0.550% at t=2.35. The offset placebo ladder is now **11-for-11** in this
2023-  repo and has been applied to macro anchors and single-name anchors alike.
2024-  (b1_c2_smh_nvda_print.py)
2025-
2026-## Cells swept and empty (2026-08-25)
2027-
2028-- **The five-day tech-to-defensive rotation at a 99.6th-percentile extreme, in
2029-  all four expressions.** The tape handed over a genuine one-in-250-day
2030-  reading (XLV-XLK 5d = +9.98pp, 99.6th full-sample percentile) and every way
2031-  of trading it failed. **Long XLK**: see the count-matched and containment
2032-  entries above; additionally the rung ladder INVERTS where today sits (+0.566
2033-  / +0.424 / +0.180 / +0.555 / **-0.408 / -0.490%** at rungs 5/6/7/8/9/10pp,
2034-  today 9.98pp), the definition is fragile (at today's 99.64 percentile all
--
2969:  ranked 8 of 12 on its own placebo ladder. (b1b_family_common_excess_r2.py,
2970-  b2_crude_midterm_fomc_r1.py)
2971-- **An inverse-variance common excess is not the family's answer.** The
2972-  pre-FOMC family's fixed-effect common excess of **-0.274pp at z -2.51** is an
2973-  artifact of up-weighting the low-volatility rates, dollar and credit legs;
2974-  **equal-weighted it is -0.004pp** at a two-sided permutation P of 0.3754.
2975-  Report both, or the weighting invents an effect the family does not have.
2976-- **Two gate legs can each be positive alone and negative together, and the
2977-  discarded complement can beat both.** Silver's post-parabolic cell: the
2978-  drawdown leg alone +1.077%, the up-on-the-year leg alone +1.118%, the pitched
2979-  conjunction **-1.032%**, and the OPPOSITE year leg (drawdown with a negative
2980-  trailing year) **+1.563%**. "Still up huge on the year" cost 2.6pp against
2981-  its own complement. A conjunction owes both single-leg cells AND the
--
3019:  (permutation P 0.3754). The placebo ladder over k=-20..0 ranks the true
3020-  anchor **11 of 21** with the trough at **k=-6**, four sessions AFTER the
3021-  decision. Declustering verified rather than assumed: consecutive entry gaps
3022-  run min 23 / median 30 td against a 10 td window, **0 overlapping pairs of
3023-  211**. (b1_fomc_family_r1.py, b1b_family_common_excess_r2.py)
3024-- **Short the index at FOMC-10td in a midterm year**, i.e. the window the event
3025-  sleeve's T2 declines. Overlap measured rather than asserted: leg correlation
3026-  **0.679**, T2 owns 4 of the 10 held sessions, and the T2-free portion alone is
3027-  **+0.079% on 27-26 at sign p 0.5000**. It dies because the live rung is
3028-  wrong-signed -- the rank21 ladder pays +0.025% under 50, **-0.068% above 65
3029-  on a 41.2% hit** and -0.535% above 80, with the live lag-1 input at 67.5 --
3030-  which independently reproduces the sleeve's own frozen prereg cross-check at
3031-  a different offset. Two episodes are **75% of the +23.49pp total** and
--
3420:  eve pays -0.290% at a 34.6% hit over 26 years, the placebo ladder ranks k=0
3421-  **14 of 17**, gate-off across all 154 closures gives -0.076% so the Labor Day
3422-  gate selects the parent's WORSE half, and 2018+ is **0-for-8 at -0.932%**
3423-  (IWM 0-for-8 at -1.402%, t -4.34). Short from the first post-holiday close:
3424-  SPY h=7 -0.551% against a FIXED September trading-day-4 anchor at -0.528%,
3425-  IWM -0.617% against -0.617%, identical to three decimals, and the whole
3426-  number is 2001's 9/11 week at +10.11% on SPY. The folk claim that September
3427-  weakness begins after Labor Day fails its own calendar test on IWM, where
3428-  forward-10 after the holiday is +0.713% against -1.452% before.
3429-  (a2_labor_day_index.py, a2b_gate_attrib_and_inversion.py)
3430-- **September quad witching is an FOMC anchor in costume.** The ungated run-in
3431-  splits on whether an FOMC decision lands inside the window: **+2.382% over 11
3432-  years at a 90.9% hit and t 3.50 with one, -0.834% over 15 at 46.7% without.**
--
3445:  crude vehicles. The placebo ladder ranks the true anchor 5, 6, 5 and 11 of
3446-  17. The reference class over 7 energy plus 5 non-energy vehicles is
3447-  homogeneous (Q 7.64 on 11 df, I-squared 0.0%, pooled +0.075%) and its only
3448-  positive member is registry-dead UNG. Crossed with the live state, a
3449-  pre-holiday 21-day rank at or above 80, XOP has **zero prior observations**
3450-  against today's 94.8, XLE two averaging -1.42% and VLO four averaging
3451-  -2.74%. (c10_labor_day_energy.py)
3452-
3453-### Method finding, filed because it is the reason the morning shipped nothing
3454-
3455-The strongest statistic produced all morning was a **corpse-recovered sign
3456-flip** and it was not shipped. Long ^VIX and short the index ACROSS an extended
3457-closure is coherent, monotone in the extra calendar day, era-stable and
--
3889:  should run before the placebo ladder on any release-anchored cell.
3890-- **Decluster-then-filter and filter-then-decluster are NOT commutative, and the
3891-  gap is material.** Same C12 object: +0.406% on n=10 one way, **+0.218% on n=9**
3892-  the other, 0.188pp apart; TLT is 0.423pp apart. Fix and state the order.
3893-- **A permutation must be tested against the cell being DEFENDED.** Applied
3894-  throughout today after the 2026-09-08 correction (where a quoted 0.1618 was
3895-  testing March while September was the cell at issue, true value 0.7354). Today's
3896-  charged-versus-uncharged pairs, all against the defended statistic: C1 0.2575,
3897-  C11 0.7688 (uncharged 0.2712), C12 0.6785 (uncharged 0.0537), C12's post-release
3898-  rung 0.2682 (uncharged 0.0055), C7's SPY leg 0.9885 (uncharged 0.1050), C13
3899-  0.3150. **The charged-versus-uncharged gap is now the single most common cause
3900-  of death in this product** and it is worth reporting both numbers every time.
3901-- **State honestly when a permutation null OVER-charges.** C12's post-release
--
3985:inside the midterm bucket the collision gate is worth -1.673pp. The placebo ladder
3986-ranks the true anchor **1 of 11** on both the run-in and the settle, so the anchor
3987-is real and it belongs to the FOMC. The only non-dead direction is the short,
3988-which is the Event Sleeve's T2. The one genuinely volatility-specific object found
3989-all morning is the settle-session SVXY rung (alpha **+1.628% at t 5.91** over SPY,
3990-R-squared 0.797) and it is parked rather than dead, because its era break lands
3991-exactly on SVXY's -1.0x -> -0.5x re-levering in February 2018. (a3, a3b, a3c, a3d)
3992-
3993-## 2026-09-10 — both armed entries fired, and both paid the charge they owed
3994-
3995-Eleven candidates over four novelty axes and ten asset classes, four adversarial
3996-checkers, 29 check scripts, all eleven killed on substantive grounds. Two parked
3997-watchlist entries armed for the first time since being parked and both died on
--
4077:  the placebo ladder in both samples, showed no midterm inversion, survived
4078-  declustering at every gap and cleared cost 19.8x — and reduces to
4079-  `SVXY = a + 1.48*SPY` with a beta-charged alpha of **+0.292pp at sign p 0.32
4080-  on 11-8**. Its content is the SPY leg (+0.482%, 23-7 on the same anchors),
4081-  which is watchlist 35, already blocked on the dial. (d1b_round2.py)
4082-- **A cell can die a fraction of a percentile point above the live reading.** In
4083-  SVXY's tradeable -0.5x era the compression band is not a plateau: **(5,10]
4084-  pays +1.455% on n=12 at 10-2, t 3.76; (10,15] pays +0.188% on n=7 at 4-3, t
4085-  0.43.** Today reads **9.921**, i.e. **0.079 percentile points** below a
4086-  boundary above which the cell has no edge. Companion to the 2026-09-09 "one
4087-  bucket above the live reading" finding, at a finer scale. (d1b_round2.py)
4088-- **Reporting the charge on the MEAN and on the t can differ by an order of
4089-  magnitude, and the t-form is the honest one when the grid spans horizons.**
--
4103:  the true k=-2 anchor ranking **1 of 11** on the placebo ladder in both
4104-  samples, no midterm inversion, survival at every decluster gap (gap 42: N=12,
4105-  +1.158%, t 3.18) and 19.8x cost. It dies on the SPY charge, on where the live
4106-  reading sits, and on its own permutation: beta-charged alpha **+0.292pp at
4107-  sign p 0.32 on 11-8**; today's 9.921 is **0.079 percentile points** under a
4108-  boundary above which the -0.5x cell pays +0.188% on 4-3; and the charged
4109-  t-permutation over the 800-cell walk that produced it is **0.6747**. The state
4110-  is separately unobserved on the dial: **all 21 dial-covered armed anchors ran
4111-  at ma10(63d) <= 68.0 with a median of 1.2 against a live 87.66**, and the
4112-  three nearest clear-calendar anchors above 80 are **1-2 at -2.788%**.
