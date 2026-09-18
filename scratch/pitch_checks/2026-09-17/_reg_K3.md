# K3 adjacent registry (miner/metal ratio, energy relative, curve)
240:- **PPI on the curve is real but exactly one session wide.** The print session
241-  itself pays +0.115% (N=286, sign p 0.0105, +0.082pp tdom-matched, 2013+ only);
242-  every pre-print session is dead 2018+. Parked to the watchlist because it arms
243-  only on the eve of a release. A 52w-floor gate on it does nothing (+0.115% to
244-  +0.117% at a tenth the sample). (c2e_era_decider.py)
245-- **The quarterly Treasury refunding concession.** Mechanism falsified inside
246-  its own window: the predicted concession days (tdom 4-8) are where refunding
247-  months are MOST positive (+1.42 bps/day, 5 of 5 days), the cumulative
248-  difference grows monotonically to +0.741pp by tdom 16 with no kink at the
249-  auctions, and anchoring on the actual auction window inverts it to -0.249pp.
250-  February, a refunding month, is the worst of all twelve months at this entry
251-  and January, not one, is the best. The label does no work, and the tdom-6
252-  excess decays +41.6 bps pre-2009 to -7.7 bps 2018+, exactly as the 2008-09
--
377:  on a miner-led thrust and both fail the only test that mattered, which is what
378-  they add to a position the book already holds. SLV correlates +0.708 with the
379-  live GDX leg and paid **-2.716% at a 34.0% hit on the 50 episodes where that
380-  leg lost**; GLD correlates +0.724 and, added at 0.25x, 0.50x or 1.00x, leaves
381-  the book's hit rate at **58.3% at every weight** while widening the worst
382-  episode from -35.40% to -45.64%. That is size, not diversification. Check
383-  correlation against live exposure BEFORE pricing a second leg in the same
384-  complex. (a4_c4_slv_basket.py, a4_c4_c11_teardown.py)
385-- **Long IBB on healthcare 63d leadership.** Under the correct trigger the sign
386-  inverts: excess against its own drift is negative at every horizon from
387-  -0.267% (h=1) to -0.959% (h=10), bootstrap P(mean<=0) 0.985, record 41-47.
388-  Regressed on XLV the alpha is -0.126% at beta 1.04, so there is no biotech
389-  residual. The one positive slice is the rank>=100 bucket nested inside a
--
536:the PPI curve cell quoted "2018+ +0.133%". That is an average of two different
537-cells: +0.278% when a CPI printed on the eve and -0.017% when it did not. The
538-parent PPI cell has no modern-era edge outside the conditioner that happened to
539-be live on 2026-08-12. (a2_c1_gate_attribution.py)
540-
541-## Method traps (2026-08-13, from a 12-direction sweep that killed all 12)
542-
543-- **A single-ticker result has to be priced against its reference class, not just
544-  against its own bootstrap.** The morning's only survivor reached round 3 looking
545-  clean: long IHI on a 21d-rank-100 thrust out of a >=10% drawdown paid +1.499% at
546-  h=5 over 16 episodes (12-4), excess +1.267pp over its own drift, within-IHI
547-  bootstrap P(mean<=0) 0.0022, positive in 9 of 9 firing years, both eras
548-  positive, monotone in the rank gate and flat across the drawdown gate. Running
--
609:  to -0.848%), 2018+ is +0.590% at 4-4, the duration-neutral residual against IEF
610-  is +0.122% at a 50.0% hit, and today's rung has no precedent at all (zero JH
611-  anchors in the sample had TLT within 1% of its 52w low). Gold: 10-11 at +0.577%,
612-  92% of it two episodes, midterm -1.213% at 1-4, and the independent Aug 6-16
613-  midterm control agrees at -0.859%, t=-2.53. Dollar: 13-13 at +0.090%, drop-best
614-  flips the sign, the midterm cell is entirely 2022's +3.00%, and 9 bps is 4.5x a
615-  DX round trip against the 5x bar. The JH anchor is now examined on rates, gold,
616-  FX and (2026-08-11) small caps, and is empty in all four. (b1_c4_jh.py,
617-  b1b_c4a_round2.py, b1c_c4a_midterm_control.py)
618-- **The macro vacuum, i.e. a hold with no 08:30 release inside it.** A gate that
619-  does not filter: it agrees with plain "no FOMC decision inside the hold" on 278
620-  of 318 anchors, and dropping FOMC from the release set collapses the h=10 excess
621-  from +0.352pp to +0.051pp. The mechanism is falsified inside its own window,
--
645:  the same windows. No ETP delivers spot VIX; the futures curve is the only
646-  tradeable surface, and at 98th-pctile contango the roll drag swamps the spot
647-  rise in every direction-consistent vehicle. The kill verdict stands; the
648-  stated fallback reasoning was garbled and must not be reused as precedent.
649-- **The fragility dial's RATE OF CHANGE as a directional signal.** Flips sign on
650-  its own threshold (+0.716% at a 30-point 21d rise, +0.004% at 25, -1.132% at
651-  35), on its own lookback (-0.454% at 10d, -0.211% at 42d), and on its own
652-  VINTAGE (+0.716% on the sizing parquet, +0.061% on the research recompute, whose
653-  own last reading is ma10 41.64 against the sizing series' 72.24). Gate
654-  attribution: the SPY-near-high leg SUBTRACTS, and the level-only form is -1.034%
655-  at 2-4, the opposite sign and the dead book-wide throttle re-skinned. 87.3% of
656-  the cell's days are pre-2026-07-02 recompute vintage and the live state's
657-  PIT-only slice is ZERO days. Every defensive expression loses (long TLT / short
--
1041:- **Miner-versus-metal ratio reversion after a maximal thrust.** A different
1042-  object from the 2026-08-17 GDX outright kill, and it dies harder: the
1043-  beta-weighted short-GDX/long-GLD trade is wrong-signed at ALL TEN horizons in
1044-  all three vehicle forms, **-0.576% at h=5 over 51 episodes against the same
1045-  vehicle's +0.154% all-days control** (edge -0.730pp, bootstrap P(mean<=0)
1046-  0.913). Conditional on a maximal miner-over-metal thrust the miner keeps
1047-  outperforming, so the operating-leverage overshoot story is falsified by its
1048-  own sign. Beta-neutralisation is NOT what breaks it (equal-dollar -0.909% and
1049-  outright -1.153% are worse). The momentum mirror is separately dead on
1050-  definition fragility (the 90th-pctile neighbour flips the sign to +0.340%) and
1051-  cost (ex-2008 19.8 bps against a 10.2 bps two-leg round trip = 1.9x). Method
1052-  note: the map's "97.9th percentile" was a FULL-HISTORY percentile, i.e.
1053-  lookahead; the PIT trailing-252d rank is the tradeable statistic.
--
1466:  duration-neutral pair is **+0.3 bps = 0.91x** its own 6 bp round trip; the best
1467-  full-sample cell anywhere is 1.24x. Adjacent horizons disagree about which leg
1468-  carries it (h=3 is all TIP, h=5 is nothing), the EFA/SPY signature from
1469-  2026-08-18 on a duration pair. The gold gate is a filter that does not filter
1470-  (**+1.17 / +1.15 / -0.64 bps alone**, swinging 19 bps across three adjacent
1471-  horizons while costing 22 of 46 episodes), and the joint cell flips era sign
1472-  from **+37.76 bps pre-2018 to -27.04 bps** after, with three 2008-09 episodes
1473-  carrying +117.86 bps of the h=3 total and being **opposite-signed at h=10**.
1474-  Mechanism check: on joint-state days the residual's contemporaneous daily
1475-  correlation is **+0.536 with SPY** against +0.212 with GLD and +0.089 with the
1476-  10y yield level. (c9_round1.py, c9b_residual_and_era.py, c9c_h10_parent.py)
1477-- **The bank-breadth washout inside an intact trend, and the insurance premise
1478-  does not replicate.** On banks the intact-trend half is **+0.225% at h=5 on
--
1841:  the curve divides the edge by ~1.75 while the round trip stays flat — IEF is
1842-  the wrong vehicle for this shape by construction. **Amendment to W5**: the
1843-  shape belongs on TLT and only fresh — TLT on the drop-TLT rung, episode-first,
1844-  ex-2022 is +0.289% at a 61.1% hit, t 2.19, **9.1x cost**, N=18. Do not widen
1845-  the rung to reach a sample. (a3_c10_ief_ig_rungs.py, a3b)
1846-- **Copper, the first metal examined here that is neither gold nor silver.**
1847-  Premise false (above), and dead on its own numbers: the 52-week-high gate
1848-  alone pays +0.691% (N=124) and the intersection with the thrust **-1.424%**
1849-  (N=8, 3-5), negative at every horizon 2 through 10 (edge -0.637 to -2.486pp),
1850-  reference-class rank **23 of 29** at P(random member >= FCX) = 0.793, and at
1851-  the loose 10% rung FCX ranks 51 of 107 at P = 0.477. The one non-negative
1852-  variant is 0.0x cost with a day-level mean of -0.293% and a sign that flips on
1853-  the declustering gap (+0.479 / +0.705 / +0.529 / **-0.146** at gaps 5/10/21/63).
--
2072:  duration trade either (beta_TLT -0.41, duration-neutral form +0.002%).
2073-  100% mask overlap with the dead 2026-08-12 rank21<=5 cell. **Utilities are
2074-  now dead in eight expressions.**
2075-  (b1_c3_xlu_washout_tlt_fine.py, b1b, b1c)
2076-- **Oil services versus E&P at a 63-day extreme, the first intra-energy pair
2077-  examined here.** Pair wrong-signed at h=1/2/3/5 (-0.209% at h=5, 35-44,
2078-  sign p 0.870, -1.7x cost); the one positive horizon is one episode
2079-  (drop-best -0.018%, drop-best-2 -0.200%). Leg attribution: long OIH +0.763pp
2080-  against short XOP **-0.439pp**, so the naked long pays +0.934% at 15.6x cost
2081-  against the pair's 1.4x - the 2026-08-24 SPY/QQQ and 2026-08-19 EFA/SPY
2082-  failure for the third time. Complex is one factor (PC1 83.0%, **1.42
2083-  effective names of 4**). Book overlap: 13 energy-family ledger signals in
2084-  these windows are **all Overbot Vol Spike SHORTS at avgR +1.083**.
--
2116:  null over the same 16-cell walk gives **P(max |t| >= 1.93) = 0.523**. Cost
2117-  4.11x on the index and **0.38x and wrong-signed on UUP**, the only vehicle
2118-  that trades as an ETF. August x midterm is N=7 at -0.156%.
2119-  (d1_c5_monthend_fx_r1.py, d1b)
2120-
2121-### Calendar finding, filed because it repeats yesterday's
2122-
2123-- **For the SECOND consecutive session the macro anchor set was empty**, which
2124-  makes 2026-08-24's note a pattern rather than a one-off. Jackson Hole is
2125-  closed on seven asset classes and today was JH-3; post-opex is closed in
2126-  both directions; NFP at +8 td sits at the horizon cap with its one live cell
2127-  midterm-parked to 2027-01; CPI, PPI, FOMC and quad witching are all beyond
2128-  +11 td. Month-end was the only non-macro anchor left and FX was its last
--
2774:  on those refinements is **p 0.822 over a 56-cell grid**, so the
2775-  unconditional cell is the honest estimate and the state-matched h=3/h=5 forms
2776-  (+1.355%, +1.334%) are not real. The GLD-beta residual also survives
2777-  (+0.322pp edge, 61.1% hit), so this is NOT the closed "second metals leg is
2778-  size, not diversification" objection. Per-leg attribution on the same
2779-  trigger: short gold is **-2.6x cost** and short the miners **-1.1x**, both
2780-  with top-2 concentration at 202% and -233% of total.
2781-- **Fading the miners' 21-day outperformance of the metal at a 98th-percentile
2782-  spread.** Wrong-signed at every horizon (-0.186% at h=1 to **-1.718% at h=10
2783-  on 24-41**) with the percentile ladder monotone in the wrong direction, so
2784-  the more extreme the spread the worse the fade. The beta-neutral form is also
2785-  negative and the long-metal leg subtracts. The seven-pair miner-metal
2786-  reference class is homogeneous with a **negative** common excess (-0.021% at
--
2929:  pre-registration declared.** The parked duration-neutral flattener declared
2930-  `HS = (1,2,3,5,10)` and permuted 3 vehicles x 2 signs x 5 horizons to reach
2931-  P 0.018 -- and then parked its arm on **h=8, which is not in that grid**.
2932-  Charging the walk actually disclosed (3 vehicles x h=1..10 x 6 proximity
2933-  rungs = 180 cells, a floor, since the 6 lookbacks make it 1080) gives the
2934-  shipped cell **P 0.388** and the grid max **P 0.144**. A pre-registration
2935-  does not immunise a cell that was later read off a horizon the
2936-  pre-registration never charged. Re-read every parked arm against the grid
2937-  its own headline number sits in before treating the park as protection.
2938-  (a1_r2c_verdict.py)
2939-- **A cost arm can CLEAR and the cell still die, so resolve the arm and keep
2940-  going.** The flattener's stated arm was a two-leg round trip under 4.4 bps
2941-  and the honest answer is **3.59 bps (6.18x)** on a half-spread MOC basis,
--
3175:- **Miner-versus-metal, entered from the break side rather than the thrust
3176-  side, fails the same two ways.** Beta-neutral at h=3 the long miner leg is
3177-  +1.711pp and the short metal leg -1.365pp of a +0.458pp spread, so the long
3178-  side is 495% of it; permutation across twelve miner/metal pairs gives
3179-  family-wise P 1.0000 / 0.9947 / 0.8452 at h=1/3/5. The outright long miner
3180-  form is separately a **class effect with dispersion ratio BELOW 1 at every
3181-  horizon** (0.89 / 0.82 / 0.78 across 14 names), so the whole cross-name
3182-  spread is smaller than sampling noise. (a2_c4_gdx_gld_pair.py,
3183-  b1_c4_outright_miner_refclass.py)
3184-- **September weakness does not exist at the month position everybody assumes
3185-  it does.** Anchored at trading day 1, ^GSPC September pays -0.042% at h=3 on
3186-  15-11, an excess of **-0.080pp** over all other months, and the midterm
3187-  crossing is POSITIVE at +1.344% on 4-2 - wrong-signed for any short. Over the
--
3832:  VIX/VIX3M ratio **ROSE 0.8251 -> 0.8548 and the curve FLATTENED**. A whole
3833-  candidate (C4) was built on the sign of that number and died on the premise.
3834-  **Reindex `^VIX` to SPY's calendar before differencing.** `pitch_lab` is
3835-  unaffected and `00_recon.py` already reindexes; this bites anything that reads
3836-  `^VIX` standalone. Same class as the 2026-09-07 `align()` bug: a calendar
3837-  defect that survives because the number it produces looks plausible.
3838-  (a2c_c4_holiday.py)
3839-- **`^MOVE` is missing on 102 of the 5,992 SPY sessions inside its own span** and
3840-  prints an unchanged close on 1.4% of them; first bar 2002-11-12, spread series
3841-  usable from 2003-11-24. State the true N before quoting a MOVE-conditioned cell.
3842-  (d1_state.py)
3843-
3844-### Method traps
--
4114:- **The duration-neutral flattener armed on both legs for the first time and
4115-  failed the charge it had written down.** `^TNX` closed at **4.8370, exactly
4116-  its trailing-252 maximum**, with a 252-session change of **+79.1 bp** against
4117-  the +78 arm — a required close of 4.8260, cleared by **1.1 bp**. The cell
4118-  reproduces at **+34.7 bps over 29 declustered episodes, 22-7, t 3.48**, sign p
4119:  0.004 filter-then-decluster, 7.84x cost, and it is genuinely duration-neutral
4120-  (correlation with the yield change over the hold **+0.028**; residual after
4121-  the all-days rate beta +30.4 of the +34.7 bps). It dies on the declared
4122-  multiplicity charge (**0.7097** over 180 cells, 0.8897 over 1,080) and on the
4123-  clearance dose: **+0.366 bps of return per bp of clearance** at corr +0.279,
4124-  with <= 10 bp paying +11.3 bps on n=7 (2.55x cost) against > 10 bp at +42.1
4125-  bps on n=22 (9.52x), and **today's exact state — a FIRST crossing that clears
4126-  by <= 5 bp — is n=5 at +5.8 bps, 1.32x** against the entry's own 5x bar. Gate
4127-  attribution agrees the thrust leg does nothing alone (+4.3 bps on 120 days
4128-  against a +4.3 bps unconditional). The denominator-roll objection that flagged
4129-  it was TESTED AND FAILED as a kill: roll-driven crossings pay **+32.5 bps on
4130-  n=5 at 4-1** against price-driven +35.1 bps, so provenance is not the problem
4131-  and thinness is. (d2_curve_yield_thrust.py, d2b_round2.py)
--
4498:- **A five-session `^VIX` thrust that leaves the curve in contango is
4499-  dip-buying wearing a volatility label.** Plain SPY 5d <= -1% pays **+0.059 /
4500-  +0.179 / +0.261 / +0.189 pp** at h=1/3/5/10 on N=1639 against the labelled
4501-  cell's +0.041 / +0.023 / +0.132 / +0.221. Inside the dip, adding the thrust
4502-  contributes **+0.014pp (h=1), -0.107pp (h=5), +0.002pp (h=10)**. The thrust
4503-  dose response INVERTS: with contango held, rank [98,101) is the only negative
4504-  band at every horizon (**-1.249pp at h=5**) while [80,85) pays +0.223pp. The
4505-  contango gate itself is NOT decoration and that is recorded — retained
4506:  +0.415% (n=90) against discarded inverted-curve thrusts at -0.001% (n=172) —
4507-  it is simply gating an object already beaten by a five-day drawdown. The live
4508-  no-dip box (h=10, n=43, +0.834%, sign p 0.0010) charges to **P = 0.9810**, and
4509-  its own dip ladder shows the edge climbing as you demand more price STRENGTH
4510-  (>-1.0% +0.458pp, >-0.5% +0.763pp, >0.0% +1.302pp), which is momentum rather
4511-  than hedging demand. SVXY closed: residual **-0.190% at t -0.17**.
4512-  (k1_vix_thrust_contango.py, `_b`)
4513-- **The FOMC/`^VIX`-expiry same-date collision: the cycle owns the direction and
4514-  2026 is the wrong half.** Pooled k=3 is a good long (+0.489%, 29-13, sign p
4515-  0.010, **placebo rank 1 of 11**), and the split is monotone and total —
4516-  non-midterm +0.530 / +0.751 / +0.753 / +1.313 / +1.289 pp at k=2..6 against
4517-  midterm **-0.380 / -0.837 / -1.568 / -1.944 / -2.337**. The midterm short
4518-  fails its own ladder at **8 of 11** and its entire payoff is 3 episodes
--
4603:- **Long IEF against 0.523 TLT, watchlist 18's dose arm, killed on firing.**
4604-  252-session change +94.3 bp on 2026-09-11. The dose does not filter (above).
4605-  The 360-cell charge (3 vehicles x 10 horizons x 6 proximity rungs x
