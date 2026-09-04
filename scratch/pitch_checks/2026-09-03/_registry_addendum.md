
## 2026-09-03

- **A pre-specified gate can be BIMODAL, and the arm has to state which half
  it means.** Watchlist 33 armed on a date (the last session before a payrolls
  print, VIX 21-day relative-range percentile <= 15) and all three of its
  stated debts resolved in its favour. It died on a split nobody had run: the
  gate's dose response is not monotone. On clear-calendar print anchors,
  rel-range **(0,5] pays -0.096% over 25 anchors at 13-11**, against
  **+1.465% at an 82.4% hit for (5,10] and +2.034% at 78.6% for (10,15]**. It
  holds in both eras (pre-2018 5-6, post-2018 8-5) and is not the VIX level or
  the term structure: the 2x2 has (5,15] paying at both VIX-level buckets
  (+1.773 / +1.674) while (0,5] is dead at both (+0.054 / -0.259), and today's
  contango bucket [12,18)% is the cell's BEST (+0.804%, p 0.002). Today read
  **3.57**. The lesson generalises past this cell and past the 2026-09-02 ATR
  entry it rhymes with: **a threshold minted as "<= X" is a claim that the
  effect is monotone up to X, and that claim is testable the morning the arm
  fires.** The tell was visible and was waved through on 2026-09-02, where the
  threshold ladder showed thr<=5 at +0.106% on n=8 beside thr<=15 at +1.313%;
  it was called a small-N wobble instead of being measured as a band.
  (a9_c1_live_rung_verdict.py, a6_c1_round2.py)
- **The vol crush is not about payrolls, it is about a CLEAR CALENDAR, and the
  reference class was the thing that proved it rather than the thing that
  killed it.** The 2026-09-02 kill said the NFP/CPI/PPI/FOMC family-wise P of
  0.2766 made the event label arbitrary. Splitting on RUNWAY (sessions to the
  next scheduled print) explains the whole inversion: PPI's median runway is
  2 td with 43.9% at <= 1, against NFP's median 5 with 87.5% at >= 3, and
  gated PPI goes from **-2.150% to -0.113%** (SVXY) and **-4.929% to +0.374%**
  (short ^VIX) once the queued-print anchors come out. The split reproduces on
  CPI and FOMC, which is the generalisation test. Pooled clear-calendar,
  deduped: SVXY n=56 **+0.910%** at 38-17 (sign p 0.005); short ^VIX n=114
  **+1.975%** at t 3.717; monotone in runway (<=1 -0.805, >=2 +0.625, >=3
  +0.954, >=4 +0.900). Inside that subset NFP's family-wise P is **0.6181**,
  i.e. the label does no work, which is what coherence looks like. **A family
  permutation that says "your event is not special" is evidence FOR a pooled
  mechanism whenever the pooled cell is the stronger object.**
  (a3_c1_ppi_family.py, a6_c1_round2.py)
- **The fragility dial's one-session short-vol signal is real, tiny, and gone
  once the tape state is controlled.** Over 2,458 dial-covered days,
  corr(dial, next-session long SVXY) = **-0.0486, t -2.41**, LOYO stable at
  -0.043..-0.060, slope -0.066pp per 10 dial points. But the damage sits at
  70-80 (SVXY -0.307%, 45.6% hit, 7 episodes) and the [80,999) band is
  **+0.020%** on 3 episodes. Conditioned on a benign tape (contango > 10%, SPY
  within 3% of its high, VIX level pctile <= 30) the dial adds nothing:
  **+0.057% below 40, -0.019% at 40-70, +0.056% at 70+**. Also settled: the
  endogeneity defence is FALSE. corr(a compressed 21-day VIX range, the dial)
  is **-0.100**, mean dial 19.0 gate-ON against 25.2 OFF, Jaccard 0.167 with
  the production VIX Range Compression signal. A dead range is historically a
  LOW-dial state, so an 87.9 beside one is a genuinely foreign reading rather
  than the dial double-counting the entry. (a1_c1_dial_debt.py,
  a2_c1_dial_conjunction.py)
- **A short-vol event cell is substantially a levered equity bet on the print
  session, and the two must never be composed as separate ideas.**
  corr(SPY h=1, SVXY h=1) is **+0.626 on the gated payroll anchors (R-squared
  0.392, beta 1.75)** and **+0.755 on all payroll anchors (R-squared 0.570,
  beta 2.20)**. The gate raises SPY by +0.314pp and SVXY by +0.939pp, so at the
  measured beta roughly two thirds of the vol cell's edge is the equity move.
  There IS a vol-specific residual, which is why SVXY is the better vehicle for
  the view, but a slate carrying both is one position twice.
  (b4_c12_spy_nfp_vix.py)
- **A breadth COUNT that never fires without its own index already triggering
  is not a gate.** The twelve-name industrial and rail rank floor fires without
  XLI at its own 5-day rank floor on **0 days in 6,707 sessions**, at three
  independent threshold choices, and the subset it selects is the parent's
  WORSE half: h=10 XLI floor alone +1.006% (n=125) against count-ON +0.036%
  (n=34) and the count-OFF complement +0.960% (n=117). Test set membership
  before testing the effect. The 2026-09-02 "the pooled floor IS the book"
  charge did NOT reproduce in this form and is withdrawn for it: 289 ledger
  rows inside the windows is 6.2% against a 6.5% calendar share, only 2 on a
  complex ticker. (c1_c4_industrial_family.py, c6_c4_c10_robustness_dev.py)
- **^SKEW's 21-day RETURN rank is not a tail bid, and the two percentile
  conventions still disagree violently.** ^SKEW at 144.12 sits at the **49.6th
  percentile of its own trailing 252 days** (trailing-year median 144.18) and
  the 90.4th of full history, the documented median drift from 112.53. The
  99.6 that selected the candidate was a 21-day return rank, i.e. a rebound off
  a low. Separately the 2026-08-12 filter finding reproduces verbatim on the
  new form: skew r21 >= 95 alone pays +0.333% over 166 episodes at t 2.29, and
  adding range compression discards 140 of 166 to leave +0.094% at an edge of
  **-0.097pp**, sign-flipped at h=10 (-0.228% against +0.274%). The midterm
  block reproduces too, at **-1.106%** against +0.536% at h=5.
  (c2_c10_skew_r21_vs_dead_range.py)
- **A bond-vol bid predicts LOWER forward equity vol, not higher.**
  corr(^MOVE 5-day return, next-5d ^VIX return) = **-0.0506 over 5,876 days**,
  monotone against the "bond market sees something equities do not" story:
  bottom quintile of the bond-vol move gives forward ^VIX **+2.304%**, top
  quintile **-0.357%**, and the ordering survives inside a dead VIX range. The
  joint live state (^MOVE level >= 80th pctile AND 5-day rank >= 90 AND VIX
  range <= 15) has **zero payroll anchors in 24 years** and two across all four
  print kinds. Every loosened version loses on the pitched long-vol side,
  including 0-for-4 and 1-for-12 cells. (a5_c2_move_vix_divergence.py)
- **A catastrophe sequel is bimodal on news that is in no series here, which is
  the definition of an unverifiable mechanism.** A utility down >= 20% in five
  sessions while its sector is untouched: N=42, h=10 median **-0.692%**, hit
  47.6%, and **23.8% of episodes lose another 20%+ within ten sessions**
  against a 1.3% unconditional base rate on the same names. Universe-wide over
  9,988 declustered episodes the median is **0.000% at every horizon 1 to 10**.
  Two process notes for reuse: the universe-wide MEAN columns are unusable
  (best +34,344%, worst -1,200%, split and adjustment artefacts in the overflow
  tier) so quote medians and quantiles only; and every number is an upper bound
  because 998 of 1,010 analogue tickers still quote, the ledger survivorship
  caveat applying to price analogues as well. (c5_c5_catastrophe_sequel.py)
- **The placebo anchor ladder is now five-for-five.** Energy at a 52-week high
  into a payrolls print: XLE's live k=-2 rung ranks **8 of 17** at h=3 (best
  placebo k=+7 pays roughly three times it) and 9 of 17 at h=5; XOP 8 and 10,
  DBC 5 and 10, VLO 15 and 13. It also killed C3 outright, where TLT's live
  k=-2 ranks **DEAD LAST of 17**. Run the ladder before anything else on any
  at-a-state-into-an-event construction. (c3_c7_energy_high_nfp_placebo.py,
  b1_c3_rates_nfp.py)
- **Two conjunctions worth less than either leg alone, on the same morning.**
  Credit: investment grade at its 252-day low with high yield at its high pays
  LQD -0.222% at h=10 against +0.401% for the IG leg alone and +0.237% for the
  HY leg alone (TLT -0.224% against +1.012% and +0.235%), and the credit-
  specific residual of LQD on IEF is -0.005 / -0.059 / -0.245pp across h=1/3/10,
  so watchlist 1's "duration wearing a credit label" debt is confirmed rather
  than paid. Gold: the joint metal-washout-with-miners-bid state crossed with a
  payrolls anchor has **2 days in 22 years**, and its deep-drawdown half, which
  is the live one at -18.78%, is 0-for-3 at -1.667% (h=3) against +0.745% at a
  77.8% hit for the shallow half. (b3_c8_credit_nfp.py, b2_c6_gold_nfp.py,
  b6_c6_washout_rescue.py)
- **EWZ is EEM with a Brazil label on print days.** 63% of its daily variance
  is EEM at beta 1.056, EEM's own thrust cell is negative at every horizon, and
  across ten EM and international vehicles EWZ ranks 3, 3, 5 and 2 of 10 at
  h=2/3/5/10, never best, below the class median at h=5. The payrolls anchor
  subtracts: at h=5 the anchored cell pays **+0.005% against EWZ's own all-days
  drift of +0.244%**. The claimed dollar channel is ordinary daily correlation
  with a calendar label (-0.292 on 233 payroll days against -0.259 on all
  4,908). Top-2 episodes are 96% of the h=3 total.
  (c4_c9_ewz_nfp_reference_class.py)

### Calendar finding, filed because it changes how an event cell is specified

The repo's event cells have always been written as "anchor on event X". The
runway split says the correct specification is "anchor on event X **with the
next scheduled print at least three sessions away**", and the difference is
worth roughly 1.7pp on the short-^VIX leg. Runway is computable years ahead
from `data/macro_events.csv`, so it costs nothing to carry. Every future
event-anchored candidate owes its runway distribution up front.
