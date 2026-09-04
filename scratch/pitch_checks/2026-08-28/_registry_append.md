
### Method traps (2026-08-28, from an 11-candidate sweep that killed all 11)

- **`searchsorted` fabricates anchors at the START of an index as well as the
  end, and this one produced a t=4.64 headline before it was caught.** The
  documented guard is `if loc >= len(dates): continue`, for a future event
  resolving to the end of the index. The mirror case is worse because it is
  silent: an event date BEFORE an instrument's first bar returns position 0, so
  every pre-inception anchor collapses onto the opening sessions. On the
  post-Jackson-Hole sweep all **11 pre-2011 conferences** landed on SVXY's first
  bars and one early value was counted **twelve times**, reporting SVXY h=7 at
  **+11.24%, t 4.64, n=26** against a real history of 14 Augusts. Any anchored
  sweep touching a late-inception vehicle (SVXY 2011, XLRE 2015, UUP 2007, GDX
  2006) needs BOTH guards. Promoted into `pitch_lab.anchor_positions`, which
  drops out-of-range anchors on both sides and returns the surviving anchor
  dates alongside the positions. (c6_post_jackson_hole.py,
  tests/test_pitch_lab.py)
- **The placebo offset ladder's record is 9-for-10, not 9-for-9, and saying so
  matters.** Long IEF one session AFTER the Jackson Hole close is the first
  event anchor in this repo whose ladder isolates k=0: the true anchor pays
  +0.228% (t 3.99, n=24) while every neighbouring offset runs -0.09% to +0.11%.
  It still died, on the midterm split and on family-wise multiplicity, but a
  killer quoted as undefeated invites the wrong inference when it finally
  misses. Record the ladder as a strong filter, not an oracle.
- **A homogeneous reference class is now the modal kill, and the fixed-effect
  common excess is frequently NEGATIVE.** Three of today's eleven died this way
  with the family's common effect pointing the wrong direction: 11 country ETFs
  at Cochran Q p 0.7879 / I-squared 0.0% / common excess **-0.230%**; 29 index
  and industry ETFs at p 0.8915 / 0.0% / **-0.228%**; 28 sector and industry
  ETFs at p 0.1671 / 20.5% / **-0.540% at t -3.14**. When the family mean is
  negative, a positive member is not a leader, it is the right tail of a
  negative distribution, and the permutation max-of-N p values behave
  accordingly (0.144, 0.135, 0.8875). Run the reference class BEFORE round 2,
  not after: it would have saved three round-2 passes today.
- **The joint state whose join subtracts is not a near-miss and must not be
  parked.** Four candidates today paid LESS than the plain state underneath
  them (round-trip breakout -0.048% against a low-63d-rank parent of +0.530%;
  the V that turned +0.370% against a momentum parent of +0.476%; the defensive
  washout with the index-high clause +1.182% on 6 episodes against +1.227% on
  34 without it; gold-and-equities below both parents AND below unconditional
  drift). No threshold rescues a negative interaction, so these leave no
  watchlist entry -- parking one would guarantee it is re-found and re-killed.
  Contrast with a cell blocked by a cycle year or a live reading, which IS
  parkable because a date or a number moves.

### Cells swept and empty (2026-08-28)

- **SPY at a 52-week high while its own 63-day return rank is bottom-quartile,
  the "round-trip breakout".** 138 days of 6,389, live today, and the
  interaction destroys both parents: the low-63d-rank leg alone pays **+0.530%
  at h=10 over 239 episodes (t 2.14)**, near-high alone +0.194%, the joint
  **-0.048% on 37 episodes** against own drift +0.457%, all days +0.377% and
  local +/-126td +0.518%. Threshold neighbours flip sign in both directions
  (r63<=15 -0.538%, r63<=35 +0.268%) and top-2 episodes are -12.64pp against a
  -1.76pp total. Separately CONFIRMED not to be the 2026-08-14 low-VIX
  near-high cell in disguise: overlap is 20 of 138 days and carries all the
  sign (+1.259% on the overlap, -0.172% off it). (a1_c1_roundtrip_breakout.py)
- **SVXY at a fresh 52-week high, as a price state rather than a term-structure
  state.** Post-2018-03 vehicle only, 47 fresh episodes. Ladder: offset -5
  +3.008% (t 9.42), -4 +2.721%, -3 +2.150%, -2 +1.472%, **true anchor +0.018%
  (t 0.04)** -- a monotone decay into the entry, which is the lagging-marker
  signature the registry already recorded for contango triggers, reproduced on
  a PRICE trigger. Trigger population's trailing 21-day return is **+9.497% at
  a 100% hit**. SPY-beta residual at beta 1.52 is **-0.449% (t -2.80)**, so the
  vehicle underperforms its own beta at the high. Both directions closed.
  (a2_c2_svxy_at_high.py)
- **Gold and the S&P both in the top decile of their 21-day returns, both
  directions.** Long is a filter that does not filter: joint -0.528% at h=5
  against a gold-only +0.418% and unconditional +0.237%; the 50/50 form is
  below both parents at every horizon. The fade is one fortnight: **top-2
  episodes 2008-12-16 and 2008-12-31 are 80% of the total** and ex-2008 the
  edge is ~1.2x cost. Reference class of 16 sibling pairs puts gold-vs-equities
  at **z +0.90** with USO-vs-IWM ahead of it, and a one-step lookback nudge
  (21d -> 10d) flips the sign. (a3_c10_gold_spx_joint_topdecile.py, a3b)
- **The month turn conditioned on a SECTOR washout, which was the last
  unswept form of the month-end anchor.** Ladder at h=5: **ME-5 +1.416%
  (t 2.79)** against the pitched **ME-1 +0.008% (t 0.02)**, and the
  three-sector form is a flat plateau with every t under 1.6. The washout
  conditioner is worse than nothing: bare ME-1 +0.105% (N=319), conditioned
  +0.008% (N=35), owning the basket every day +0.175%. Midterm is **-0.773%
  (N=13, t -2.11, 30.8% hit)**. 110-cell grid, best occupant at Sidak p 0.358.
  The month-end anchor is now closed on five classes.
  (a4_c11_sector_month_turn.py)
- **The country-ETF thrust from inside a drawdown, the INVERSION of the closed
  break-inside-an-intact-thrust family.** The drawdown clause subtracts
  (+0.673% bare, +0.463% joint, **+0.713% complement**; pooled -0.138pp over 11
  names) and today's own depth bucket is the worst of six (**(-15%,-10%]
  -0.289% at a 50.0% hit**). This reproduces the 2026-08-10 silver finding on a
  second asset class: **distance-from-high is a U-shaped noise carve, not a
  conditioner**, and that now holds on metals and on country equity. Family
  Cochran Q p 0.7879, I-squared 0.0%, common excess -0.230%.
  (b1_c3_thrust_in_drawdown.py, b1b)
- **The "V that turned", 21-day rank >= 90 with 63-day rank <= 10, pooled over
  29 ETFs.** Bare momentum +0.476% (N=3,521, t 6.43); joint +0.370% (N=189,
  t 0.88); complement +0.481% (t 6.47). The 63-day clause subtracts -0.106pp
  and discards 95% of the population. Rank and level forms disagree at
  **Jaccard 0.10** with the t-63 roll-off exceeding the day's own bar on 31.0%
  of trigger name-days, so the 2026-08-19 warning holds on the rank-LOW tail
  too. USEFUL RESIDUE, filed to watchlist: the sub-cell with a **5-day rank
  under 15** pays +1.437% (N=53, t 2.15, 67.9% hit) and there the 63-day gate
  adds +0.705pp, against +0.139pp in the already-bouncing half -- a pooled
  confirmation of watchlist 30 at 3x its episode count.
  (b2_c4_v_that_turned.py, b2b)
- **Sustained industry leadership at a double rank extreme into a 52-week high,
  tested on biotech, which closes the last unswept industry class.** Every
  clause subtracts (near-high alone +0.273% on 4,151 obs; double-rank plus
  near-high +0.200%; full cell **-0.159%**), the beta-neutral residual is
  **-0.023% (t -0.035)** on a measured beta of 0.983, and **92.2% of trigger
  days sit above SPY's 200d against a 71.6% base rate**. Family of 28: common
  excess **-0.540% (t -3.14)**, IBB 11 of 28, family-wise p 0.8875 -- the IHI
  shape for the third time. (b3_c5_biotech_leadership.py)
- **The POST-Jackson-Hole anchor on ten asset classes, which closes the
  conference in the only direction that was left.** 210 cells produce 10 at
  |t| >= 2 against an iid expectation of 10.5, and the best cell fails a
  permutation null at **P 0.065**. The duration pulse (IEF +0.228%, LQD
  +0.218% at h=1) is real and ladder-isolated but **midterm-inverted for the
  seventh time** (+0.037%, t 0.41, 33.3% hit on 6 anchors, against +0.292% and
  t 4.54 on 18), era-decayed to +0.137% at t 1.05 from 2020, one duration bet
  wearing four labels (IEF/TLT forward correlation **0.911**), and partly a
  month-position effect (lag-1 entry lands ME-1..ME-6 in 20 of 24 anchors).
  **Jackson Hole is now closed pre-speech on eight classes and post-speech on
  ten.** (c6_post_jackson_hole.py, c6b)
- **The whole defensive complex washed out while the index sits at a 52-week
  high**, which is the post-presidential-election rotation wearing a
  sector-breadth label. **62.5% of the 16 trigger days fall within 60 calendar
  days of a presidential election against a 9.1% base rate**, 8 of 16 within
  30 days, leaving two historical episodes outside that window. Top-2 episodes
  are **143% of the h=3 total**. The index-near-high clause subtracts
  (three-of-three alone +1.227% on 34 episodes against +1.182% on 6 with it)
  and breadth is non-monotone (2-of-3 beats 3-of-3). The rates reading fails
  independently: basket TLT loading **0.138**, and TLT's own forward return on
  trigger episodes is -0.291% at a 22.2% hit. (c7_defensives_washed_at_high.py,
  c7b)
- **The energy PULLBACK inside a thrust near a 52-week high, the rung between
  two already-closed cells.** Both clauses subtract monotonically: near-high
  2%/3%/5%/10%/none pays +0.482 / +0.409 / +0.813 / +1.080 / **+1.294%**, and
  the thrust ladder r21>=50/65/80 pays +0.926 / +0.409 / **-0.276%**. The
  near-high clause is the bull-tape selector at **100.0% of 27 trigger days
  above SPY's 200d against a 71.6% base**. Nine SPDRs homogeneous (I-squared
  0%), pooled pays +0.065% at h=7 against 6 bps, XLE ranks **9 of 9** by |t| at
  h=10 with permutation P 1.000. Premise correction worth keeping: **XLE's
  measured daily beta on CL=F is 0.112, not ~0.48** -- the levered-crude story
  does not apply to XLE at the index level. (c8_xle_pullback_in_thrust.py)
- **The dollar washout translated through DEVELOPED international, which closes
  the family the EM version opened on 2026-08-26.** Best horizon pays 2.7 bps
  = **0.34x** an 8 bps two-leg round trip, negative at five of six horizons,
  123-129 record. The identity does not exist to harvest: on gate episodes the
  dollar's own forward move is **+0.005%** (it mean-reverts UP) and the pair's
  slope on the dollar is -0.63. The beta worry was wrong in the idea's favour
  and did not save it -- measured **EFA beta on SPY 0.951**, so
  beta-neutralising moves +0.027% to +0.030%. Decisive for the lane: the
  already-dead EM version scores BETTER (+0.176% vs +0.027% at h=5), so the
  whole dollar-washout-through-international family is closed rather than
  parked. (c9_translation_channel.py)

### Calendar finding, filed because it is now five consecutive sessions

- **A fifth straight empty anchor set, and today the last open DIRECTION of the
  last anchor closed.** Jackson Hole was JH+0 for the first time in this
  product's life, so the post-speech anchor became reachable and was swept on
  ten classes; it is now closed. Nothing else moved: NFP at +5 td, PPI at +8
  and CPI at +9 are all closed on their own ladders, and FOMC, VIX expiry, the
  September opex and quad witching are all beyond the 10 td cap. The month-end
  anchor closed its fifth class (sector-conditioned). **The next genuinely new
  anchor remains the September FOMC on 2026-09-16, entering the horizon around
  2026-09-02, and it is already spoken for by the event sleeve's T1/T2** --
  and note the midterm T2 short is itself gated on SPY's 21-day rank being
  under 50, which reads 91.3 today, so even the sleeve's rule is off. The
  practical consequence is unchanged and now five sessions old: **a price-state
  sweep is the only honest search mode**, and today it produced eleven
  candidates and no survivor.
