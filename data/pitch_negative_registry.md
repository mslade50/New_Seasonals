# Pitch negative-results registry

Dead ends the Daily Pitch pipeline must not re-pitch. Stage C checks every
candidate against this file (spec: daily_pitch_agent_spec_2026-08-06.html
section 5). An entry here does not mean "never think about this again"; it
means the obvious form of the idea was tested and failed, so a candidate that
collides with an entry must either be dropped or must state explicitly what is
different about its construction.

Format, one dead end per bullet, parsed by
`scripts/build_pitch_research_index.py`:

    - **short key** — why it is dead, with the study that killed it

The registry GROWS. Every stage-C kill with a reusable lesson gets appended
the same morning it happens, by the `/daily-pitch` skill.

## Risk-dial conditioning

- **book-wide fragility throttle / taper** — aggregate rest-of-book at dial >= 50 shows no significant degradation (p = .47 clustered), point-in-time aggregate t = -0.23, and the taper variant cost -11.4R. Only per-strategy `frag_risk_bands` on the dip-buy family survived. (scratch/ultracode_research/RISK_DIALS_2026-07-16.md)
- **dial-conditioned daily risk caps** — pooled or per-strategy caps scaled by the dial are the failed book-wide throttle re-skinned, on the costliest possible surface (four aligned sites, one out of repo). (CLAUDE.md "Daily Risk Caps"; RISK_DIALS_2026-07-16.md)
- **hi-fragility size BOOSTS above 1.0x** — the old ramp's 1.25x high-dial boost had no edge case in any bucket and was retired with the ramp. The only surviving boost is the P/C fear-ON calm-tape 1.25x. (RISK_DIALS_2026-07-16.md section 4)
- **sub-50 sizing ramps** — graded sizing below the 50 threshold added nothing over a flat 1.0x. (RISK_DIALS_2026-07-16.md section 4)
- **21d "fast confirm" shadow** — the 21d dial agrees with the 63d about 90% of the time, so a confirm layer is noise with a lag. No confirm semantics exist anywhere in the book. (CLAUDE.md risk-dials section)
- **5d dial for sizing** — failed every sizing test; it is a display-only context chip. (signal_horizon_stats work, RISK_DIALS_2026-07-16.md)
- **OVS fragility tilt** — the [21,44) 0.75x tilt failed the point-in-time gate (t = -1.34; even current weights only t = -0.63 on 2018+). OVS is fully exempt. (scratch/pit_reestimate.py, 2026-07-03)
- **trend-sleeve dial gate** — rejected; the sleeve is ballast and its high-fragility hole is handled by frag_risk_bands elsewhere. (trend-following.md, trend_prework_gates.md)

## Hedging and volatility vehicles

- **standing put hedges** — cost more than the drawdown they smoothed across every tested sizing. (RISK_DIALS_2026-07-16.md section 4)
- **VXX as a hedge proxy** — roll decay eats the payoff over any horizon the book trades. (RISK_DIALS_2026-07-16.md section 4)
- **naked short UVXY at size** — unbounded loss on a vol spike. Short vol in this book is expressed as LONG SVXY (a -0.5x ETP), which bounds the loss at the position. (event sleeve prereg 2026-08-06, V-trade addendum)
- **SVXY as a pre-FOMC (T1) leg** — 0.78 correlation to the equity expression of the same drift; it adds vehicle risk without adding a thesis. (event_seasonality_sweep_2026-08-06.md addenda)
- **post-CPI vol crush** — the effect died after 2018; the era check kills it. (event_seasonality_sweep_2026-08-06.md)
- **post-NFP, post-FOMC and post-VIX-expiry vol cells** — swept and empty. Opex ex-September is the only post-event vol cell that survived. (event_seasonality_sweep_2026-08-06.md; V4 in the event sleeve)
- **September post-opex vol crush** — September INVERTS the crush. That stress is the T3 short-IWM trade, not a short-vol trade. (event sleeve prereg 2026-08-06)

## Instruments and cost

- **UUP as a dollar vehicle** — roughly 6 bps of edge cannot pay the ETF's drag and spread. The same effect expressed in DX futures passed the cost check. (event_seasonality_sweep_2026-08-06.md, DX addendum)
- **USO in the trend sleeve** — roll decay. Excluded from the 12-ETF universe. (scratch/tf_universe_study.py)
- **UUP in the trend sleeve** — a 20% slot for +0.00%/mo contribution plus a K-1. (scratch/tf_universe_study.py)

## Calendar and factor seasonality

- **famous calendar factor cells** — arbitraged away post-2013 (turn-of-month, January effect and the classic factor-month cells). (memory factor-seasonality-research; scratch factor work)
- **Heston-Sadka seasonality on liquid names** — dead on this universe. (factor seasonality research)
- **RMW factor calendar cells** — parked; the ETF loading gate failed, so there is no tradeable expression. (factor seasonality research)
- **TLT mid-month gap leg** — dead in every form tested. The ungated SPY-laggard cell (+1.12%, t = 2.4) is the only parked survivor and is NOT shipped. (memory spy-tlt-midmonth-gap-research, 2026-07-31)
- **midterm-year pre-FOMC drift, ungated** — midterm years invert the Lucca-Moench drift, so the long form must exclude them and the short form needs the rank21 < 50 gate. (event sleeve prereg 2026-08-06)

## Sizing and accounting methodology

- **notional-denominated caps** — rejected by design. ATR risk caps plus time exits are the control. (memory no-notional-caps)
- **marginal-fill decompositions** — "new fills only" streams from a limit sweep are selection-biased by construction. Compare whole variants only. (memory no-marginal-fill-decompositions)
- **pooled per-direction daily caps** — bound on the same net-positive cluster days as the per-strategy cap and cost roughly $125k over 23 years with identical maxDD. Removed 2026-07-16. (scratch/cap_impact_study.py)
- **grading a rule by re-running the backtest with the rule on** — in-sample rules flatter themselves. Validation is leave-one-year-out plus episode clustering. (cycle-year tilt work, 2026-06-10)

## Strategy-structure dead ends

- **3x leader gap fade on bull-equity names** — every selectivity layer makes bull-eq worse; the strictest cell is 0-for-7 at avgR -1.28 across five regimes. Structural exclusion. (scratch/lev3x_fade_leader_bulleq_*.py)
- **stops on the 3x leader gap fade** — 1.0-2.0 ATR stops all destroyed the edge (+23.7R to -39.8R at 1.0 ATR). Adverse excursion beyond 1 ATR is the normal path before the reversal. (scratch/lev3x_fade_leader_stops.py)
- **dropping the <65 leader exclusion from the 3x bear fade** — collapses avgR from +0.66 to +0.28 with -14R/-16R years in 2020 and 2022. Load-bearing filter. (scratch/lev3x_fade_class_study.py)
- **ungated monthly weak-close dip buy** — without the 200d trend gate, 2000-01 and 2022-style signals ride the next bear leg (worst -17.9%). (scratch/monthly_weak_close_mr*.py)
- **OLV ladder graded UP by open-position count** — the original mild first-rung discount graded up lost to flat 1.0x ($605k vs $654k over 21 years). Today's recency ladder is an appetite cut that accepts that drag, not a PnL claim. (scratch/olv_package_sim.py)
- **sector-ETF and international single-name expansion of the trend sleeve** — equity slots crowd out the diversifiers and 2008/2022 flip negative. (scratch/tf_universe_study.py)
- **trend-sleeve exhaustion scale-down overlay** — Sharpe flat. (trend-following.md)
- **laggard-snapback continuation (SMH/QQQ form)** — long the deep 63d laggard that is snapping back does not continue; at h=5 the pair is flat (episode N=57, +0.27%, t=0.80) and the trigger over-selects bear tape by +29pp vs base rate, so it is a regime bet, not a relative-value edge. (scratch/pitch_checks/2026-08-07/c10_smh_qqq_laggard.py, r1_smh_qqq_inversion.py)
- **sector-vs-index pairs on a crowding or leadership trigger** — the trigger selects tape that BOTH legs share, so the spread is the difference of two near-identical drifts. XLV +0.377 vs SPY +0.412 at h=10; XLF +0.002 vs SPY +1.029 on the opex-anchored cell. Price the legs before the spread. (scratch/pitch_checks/2026-08-07/d7_xlv_crowded.py, d8b_opex_control.py)
- **fragility dial as a DIRECTIONAL signal (level or rate of change)** — the registry already killed the dial as a sizing rule; a directional read is the strictly stronger claim and fails at a lower bar. Dial ma10(63d) spiking from below 30 to above 50 at a 52w high gives 5 episodes, edge over buy-and-hold +0.26pp, and drop-best puts it below the control. The plain dial level is negative for SPY, so the "spike from calm" conditioning inverts the base relationship on 5 observations. (scratch/pitch_checks/2026-08-07/d1_dial_spike_calm_surface.py)

## Method traps (2026-08-07, from a 28-candidate sweep that killed all 28)

- **lag-0 forward returns on a MOC idea** — `fwd_ret` from the signal close measures a session you cannot trade, because the order enters at the NEXT close. Correcting SMH/QQQ from lag 0 to lag 1 took h=5 from t=2.30 to t=1.39; the entire nominal significance lived in the untradeable session. Every check must state its entry-lag convention. (scratch/pitch_checks/2026-08-07/r1_smh_qqq_inversion.py)
- **day-level t-stats on overlapping triggers** — declustering flipped UNG h=10 from +1.02% to -0.64% and GDX h=10 from +4.41% to -2.80%, and took an XLU/XLP pair from t=1.87 to t=0.12. The episode-level number is the only real one. (scratch/pitch_checks/2026-08-07/c7_natgas_floor.py, c8b_decluster_robustness.py, c2_xlu_xlp_spread.py)
- **conditional cells that underperform their own instrument's drift** — the decisive control is never zero, it is what the instrument does unconditionally over the same horizon and window. GLD pre-CPI (+0.040%) loses to GLD's own h=2 drift (+0.092%); AAPL's laggard cell has NEGATIVE excess at every headline threshold; a 10 td August-opex long underperforms a random 10 td SPY long. (scratch/pitch_checks/2026-08-07/d5_gold_pre_cpi.py, e2_aapl_laggard.py, d2_nfp_to_opex_run.py)
- **mid-cluster entry is not a fresh trigger** — a trigger 3-4 sessions into a run has different forward statistics from the episode-level mean that was measured. Conditioning the "low vol at the high" cell on today's actual cluster depth flipped its sign outright. Compute cluster depth before quoting an episode statistic as today's expectation. (scratch/pitch_checks/2026-08-07/e7_midcluster_check.py)
- **post-hoc sign flips recovered from a kill report** — a result found while hunting its opposite carries sign x era x horizon multiple comparisons before any threshold grid. The TLT-short inversion was nominally t=2.18 but Sidak over the implicit looks gives p~0.47, and both surviving inversions died on re-examination. Take the third idea from a candidate designed forward, not recovered from a corpse. (scratch/pitch_checks/2026-08-07/r1_smh_qqq_inversion.py, r2_tlt_short_inversion.py)
- **an era cut that isolates one macro episode** — "2018+" sounded like an era split and was actually a fence around 2021-2022: the TLT-at-the-floor short is 8 of 12 episodes in those two years, ex-2022 t=0.69, ex-2021-22 t=0.16, and pre-2018 the same trigger LOST. Check the episode year histogram before believing an era split. (scratch/pitch_checks/2026-08-07/r2_tlt_short_inversion.py)
- **rank gates in a quiet tape buy a fraction of the historical force** — the TLT trigger's ^TNX 63d rank >= 85 bought +31 bps of yield thrust today, the 3.8th percentile of the 26-episode distribution, against +47 to +129 bps for every winning episode. A percentile gate is not a magnitude gate; check the level the rank corresponds to today. (scratch/pitch_checks/2026-08-07/r4_inv2_regime_probe.py)
- **checking edge before checking N** — the "dollar into CPI" conditional cell (DX rank21 < 20 inside rank63 > 90) has occurred ZERO times in 318 CPI events since 2000. When the described state is rare, count occurrences first. (scratch/pitch_checks/2026-08-07/d6_dollar_pre_cpi.py)

## Calendar and event cells swept and empty (2026-08-07)

- **post-NFP equity DIRECTION** — distinct from the already-dead post-NFP vol cell, separately swept and empty. NFP close to next CPI is +0.129% (N=309) against an all-days control of +0.221% and an all-macro-event control of +0.146%; the sign flips between the 3 td and 5 td horizons; the overbought-NFP conditioner (rank5 >= 90) is NEGATIVE at -0.400%. (scratch/pitch_checks/2026-08-07/c9_nfp_into_cpi.py)
- **the run into August opex** — +0.342% over 26 non-overlapping years vs SPY's +0.374% unconditional h=10 drift, i.e. worse than a random 10 day long. August is not special (all-months +0.162%) and the whole effect lives in 2000-2004 (2010+ is -0.514%). (scratch/pitch_checks/2026-08-07/d2_nfp_to_opex_run.py)
- **VIX-expiry-week drift** — the raw cell (+0.175%, N=319) is mid-month position plus noise: within-month paired excess is +0.065% (t=0.67) and 2018+ paired excess is negative. The stated mechanism is falsified inside its own window, since the settle day itself is the worst day (-0.102%) and all the return lands before it. It is genuinely NOT the pre-opex week, and it still has no edge. (scratch/pitch_checks/2026-08-07/d4_vix_expiry_week.py, d4b_vix_week_vs_monthpos.py)
- **pre-expiry short-vol carry (long SVXY into VIX expiry)** — distinct from the event sleeve's V4 post-opex window (7.5% calendar overlap, so not a re-skin) and dead on its own numbers: the gate-matched control eats it (+0.46pp excess at the live horizon), 2018+ is +0.19% at t=0.18 on the -0.5x instrument, the mean is unstable to one session of horizon, and one gated 8 td window in 22 loses double digits (worst -24.8% post-2018). (scratch/pitch_checks/2026-08-07/c12_svxy_pre_expiry.py, c12b_gate_tail_and_concentration.py)
- **midterm mid-August seasonality** — N=6, carried entirely by 2002 (+8.68%); drop-two-best is negative. The midterm restriction anti-works at 21 td (SPY midterm +0.361% vs non-midterm +0.531%; IWM +0.269% vs +1.455%), so the seasonal board's midterm de-risk prior does not localize here. (scratch/pitch_checks/2026-08-07/d3_midterm_mid_august.py)
- **pre-event windows on the event's "own" instrument** — GLD into CPI underperforms GLD's unconditional drift at k=1..4, and conditioning on gold already rallying selects the crash tail (2013-04 -13.07%, 2020-08 -5.70%). The only cell with a pulse is the opposite one, gold ON the print day. DX into CPI is -3.5 bps, which pays neither the 1.5 bp futures nor the 6 bp UUP round trip. (scratch/pitch_checks/2026-08-07/d5_gold_pre_cpi.py, d6_dollar_pre_cpi.py)
- **weekend-risk discount at a stretched high (Friday to Monday)** — a pre-2013 fossil, exactly as the "famous calendar cells" entry predicts. Full-sample short is +0.115% (N=34) but 2013+ is -0.010% and 2018+ -0.007%, i.e. 0.0x cost. The weekday placebo is decisive: Tue/Wed/Thu entries on the same price cell are significantly POSITIVE (t=2.8/3.1/2.4) and Friday is not even the most extreme weekday. (scratch/pitch_checks/2026-08-07/e4_friday_weekend_high.py, e5_e4_fossil_test.py)

## Instruments and cost (2026-08-07 additions)

- **UNG long at a 52w low** — UNG's structural bleed is -0.90%/10 td and -28.65%/yr (buy-and-hold -99.85% over 19.3 years). The conditional cell is NEGATIVE in absolute terms (h=10 episodes -0.644%) and its apparent day-level edge was pure overlap inflation. Any relative edge is ~0.25pp inside a ~0.90%/10td drag, so it is unharvestable with an outright long. Worse than the USO roll-decay entry by roughly an order of magnitude. (scratch/pitch_checks/2026-08-07/c7_natgas_floor.py)

## Cross-asset event cells (2026-08-07 PM sweep, the first survey-then-select run)

The AM run's blind spot was that every calendar-anchored check ran on SPY. The
surface map opened the missing intersection and these are its results. The
headline lesson is that the cell was worth opening and still not tradeable, so
"we never looked" and "there was nothing there" are genuinely different
failures and only the first one is fixable by process.

- **NFP reaction with the long end at its 52w floor is a REAL cell, and it is a
  non-midterm cell.** TLT long from the NFP close to +3td, gated on TLT within
  3% of its 52w low: +0.543% vs +0.050% own drift, N=25, 76% hit, bootstrap
  P(mean<=0)=0.021, 27x cost. It passes every robustness test that killed other
  ideas: declustering IMPROVES it (t 1.92 -> 2.06), both eras are positive
  (pre-2018 t=2.12, 2018+ t=1.13), all nine LOYO t-stats are positive with a
  floor of 1.46, and dropping the two biggest years IMPROVES it to +0.926% at
  t=2.72. Then the cycle-year split ends it: midterm +0.071% (N=12, 58% hit,
  t=0.17) vs non-midterm +0.978% (N=13, 92% hit, t=2.72). Re-check this cell in
  a non-midterm year; it is parked, not dead.
  (scratch/pitch_checks/2026-08-07/n1_nfp_rates_floor.py, n2_redteam_and_open_cells.py)
- **The midterm split held across three independent instruments, which is what
  makes it a conditioner rather than noise.** Bonds go to zero, utilities go
  wrong-signed (-0.538%, 33% hit), and the dollar goes wrong-signed in BOTH
  vehicles (UUP -0.141% at 37.5% hit, DX -0.184% at 41.7%) while their
  non-midterm cells run t=4.85 and t=4.91. A conditioner that flips sign
  coherently across rates, rate-proxy equity and FX is structural. One that
  only shows up in a single instrument is usually a subsample.
  (scratch/pitch_checks/2026-08-07/n4_dollar_leg_final.py)
- **A rescuing sub-cell with N=1 is not a rescue.** The CPI-inside-the-hold
  split looked like it saved today (+1.614%, N=5, 100% hit, t=2.94), and today
  does have CPI inside the hold. Crossing it with the cycle year showed the
  midterm-and-CPI-inside cell has occurred exactly ONCE in twenty years, on
  2022-05-06, at the most violent point of the hiking cycle, and that single
  observation is also the best value in the entire midterm bucket. Always cross
  the rescuing conditioner with the killing one before believing the rescue.
  (scratch/pitch_checks/2026-08-07/n3_midterm_kill_confirm.py)
- **Count occurrences of a JOINT state before designing the trade.** The credit
  duration divergence was the most visually striking thing in the tape (HYG
  0.11% off its 52w high while LQD sat 0.98% off its 52w low). The joint state
  has occurred on 2 NFP days in 24 years. Unmeasurable is a kill, not a pass,
  and it is the second time in one day this trap fired (the AM run's dollar
  pre-CPI cell had N=0).
- **Utilities are now dead in all four expressions** and should not be
  re-opened without a new mechanism: outright washout (-0.123% vs +0.207%
  drift, and the SPY-near-high gate that fires today HURTS, +0.605% ungated vs
  -0.123% gated), the XLP pair (t 1.87 -> 0.12 declustered), the SPY spread
  (-0.311%, P(mean<=0)=0.774), and the rates channel (midterm-negative). A
  sector being the loudest thing in the cross-section is a reason to look, not
  evidence of an edge.

## Small N, mechanism, and when a multiplicity correction applies (2026-08-07)

McKinley asked directly whether long DX at a weak NFP close had been checked in
August midterm years. It had not, so it was measured. The check then killed it
on a family-wise p-value, and McKinley overruled that, correctly. **This entry
is the corrected version. Do not re-apply the reasoning it replaces.**

- **A multiplicity correction prices the cost of a SEARCH. It only applies to
  cells the search found.** The DX cell was pre-specified by McKinley before any
  code ran. The checker then built a 47-cell month x cycle-year grid of its own
  and scored his hypothesis against the best-of-47 null, reporting a family-wise
  p of 0.904 against a pre-specified p of 0.011. That is a category error: his
  idea was charged the search cost of the checker's scan. Correct for a grid you
  built. Never correct for a grid the other person never searched.
  (scratch/pitch_checks/2026-08-07/n6_dx_cell_vs_luck.py)
- **Small N is not a kill; it is a grade.** Markets produce small samples by
  construction, a cycle-year cell yields one observation every four years, and a
  rule demanding N=50 selects for stale regimes rather than safe ones. An idea
  with a plausible mechanism and N<15 is a grade C. Ship it, grade it honestly,
  let the scoreboard decide. The product's own premise is that false positives
  are acceptable because McKinley filters.
- **What still stands from that check, because none of it is about sample
  size.** The weak-close gate removed one observation of six, so the trade keys
  on August-midterm and not on the weak close, and the write-up must say so. The
  result holds under "down on the day" and "close below open" but collapses to
  N=2 under bottom-third-of-range, so the trigger definition is load-bearing and
  should be stated as chosen rather than discovered. Those are facts about the
  idea's construction and they survive the overrule.
- **The broad cell underneath, for reference rather than as a replacement.**
  Long DX after ANY weak NFP close, h=5: N=160, +0.1826%, 54.4% hit, t=1.964,
  bootstrap P(mean<=0)=0.0248, edge +0.1525pp over the all-NFP control, 12.2x
  the 1.5 bp futures round trip. Era-decayed: pre-2018 +0.2282% at t=1.89,
  2018+ +0.0931% at t=0.66. NOTE the contrast with the same day's
  TLT-floor-gated dollar work, where midterm was wrong-signed while here midterm
  is slightly better (+0.221% vs +0.169%). Different trigger, different
  conditioner behaviour, neither transfers to the other.
  (scratch/pitch_checks/2026-08-07/n5_dx_weak_close_august_midterm.py)

## Method traps (2026-08-10, from a 17-candidate sweep that killed 16)

- **A rates event cell measured against an all-days control is invalid.** TLT's
  own unconditional h=3 return swings from -0.202% at trading-day-of-month 2
  (t -2.19) to +0.215% at tdom 14 (t +2.58) with no event anywhere. CPI entries
  land at median tdom 9, in the good half, so the all-days control flatters them
  by construction: "long TLT into CPI" reads +0.178% raw and +6.7 bps against a
  tdom-matched control, at a 49.7% hit, +1.4 bps ex-2008. The whole
  "CPI/PPI/FOMC work on duration, NFP inverts" pattern is that calendar profile
  read back with event labels on it, because NFP sits at tdom 3 where TLT's own
  drift is negative. Build the tdom control first; it is reusable.
  (scratch/pitch_checks/2026-08-10/d5b_tdom_control.py, d5_cpi_tlt_parent.py)
- **A number quoted out of a kill report inherits that report's controls.** The
  CPI-TLT parent was published as +0.193% at sign p 0.001 inside a gate
  attribution whose author never questioned the all-days control, because the
  parent was not what he was grading. Re-derive a borrowed number from scratch
  before promoting it to a candidate, including the k the calendar can actually
  execute: the quoted cell was k=1 and the executable one was k=2, which pays
  30 bps less and starts by buying a session worth -2.6 bps.
- **Beta-neutralize a pair before crediting the spread.** Equal-dollar GDX minus
  GLD on a 5d thrust trigger pays +0.786% over 41 episodes; at the measured
  beta of 1.78 the same episodes pay -0.000% (t -0.00). Both legs are positive
  on trigger days. Report the regression beta, not just the correlation.
  (scratch/pitch_checks/2026-08-10/c6_gdx_gld_thrust.py)
- **Cluster depth is not a one-way objection.** The 2026-08-07 registry entry
  says a mid-cluster entry is not a fresh trigger, and that killed the crushed-
  skew cell outright (depth 4, which was that day's state, pays -1.459% at
  1-for-9 against +0.401% at depth 1). But the GDX pre-CPI cell runs the other
  way: depth > 2 pays +1.601% at a 76% hit against +0.296% for depth <= 2, and
  the trigger population's own median depth is 3. Measure the depth bucket and
  quote it; do not assume staleness in either direction.
- **A nested subset that reverses its parent's sign is a partition of noise.**
  DX "pullback inside an uptrend" pays -0.233% at z10 <= -1.25, -0.234% at
  <= -1.50, then +0.665% at <= -1.75. The last is a subset of the second, so the
  9 episodes between the cuts must average about -1.18%. Today's reading sitting
  inside the only positive slice is how a threshold-mined artifact presents.
  (scratch/pitch_checks/2026-08-10/c11_dx_pullback_uptrend.py)
- **The 14:00 ET placebo for any overnight/intraday decomposition.** The premise
  that an 08:30 release resolves in the opening auction predicts a large
  overnight premium on CPI/PPI/NFP and none on FOMC. The largest overnight
  premium in the whole study is SPY on FOMC day (+13.5 bps tdom-matched, hit
  64.2%, sign p 0.0000), and FOMC prints at 14:00. Dispersion agrees: 08:30
  prints raise overnight sd by 0-17% while the 14:00 print raises INTRADAY sd by
  up to 48% and lowers overnight sd. The overnight premium is a session-of-day
  effect, not an event effect.
  (scratch/pitch_checks/2026-08-10/e2b_placebo_teardown.py)

## Calendar and event cells swept and empty (2026-08-10)

- **PPI on equities.** First sweep of PPI in this repo (323 events, 2000+; the
  2026-08-06 event sweep never covered it). SPY on PPI day is -0.009% over 317
  episodes against SPY's own +0.039% same-span drift. The back-to-back
  CPI-then-PPI pair, one session apart, is +0.002% on N=55; the reverse ordering
  is -0.071% on N=133. Ordering carries no information.
  (scratch/pitch_checks/2026-08-10/c1_ppi_spy.py)
- **PPI on the curve is real but exactly one session wide.** The print session
  itself pays +0.115% (N=286, sign p 0.0105, +0.082pp tdom-matched, 2013+ only);
  every pre-print session is dead 2018+. Parked to the watchlist because it arms
  only on the eve of a release. A 52w-floor gate on it does nothing (+0.115% to
  +0.117% at a tenth the sample). (c2e_era_decider.py)
- **The quarterly Treasury refunding concession.** Mechanism falsified inside
  its own window: the predicted concession days (tdom 4-8) are where refunding
  months are MOST positive (+1.42 bps/day, 5 of 5 days), the cumulative
  difference grows monotonically to +0.741pp by tdom 16 with no kink at the
  auctions, and anchoring on the actual auction window inverts it to -0.249pp.
  February, a refunding month, is the worst of all twelve months at this entry
  and January, not one, is the best. The label does no work, and the tdom-6
  excess decays +41.6 bps pre-2009 to -7.7 bps 2018+, exactly as the 2008-09
  move to monthly 3y/10y/30y auctions predicts.
  (scratch/pitch_checks/2026-08-10/e1_refunding_concession.py, e1b)
- **The August big-box retail earnings cluster.** Cluster-anchored 4-session
  window +0.698% (N=26) against +0.681% for ALL August 4-session windows on the
  same equal-weight basket: the earnings anchor is worth 1.7 bps against 18 bps
  of cost. Ten sliding placebo anchors average +0.371% and two beat the real one
  with better records. Pooled across all 101 clusters since 2000 it is -0.041%.
  No era decay, which is the interesting part: it dies on the control, not on
  arbitrage. (scratch/pitch_checks/2026-08-10/d2_retail_cluster.py, d2b)

## Price-state cells swept and empty (2026-08-10)

- **Adding confirming legs to a momentum state does not create a state.**
  Synchronized 52w highs across SPY, EFA and HYG add +0.036pp over
  SPY-at-a-52w-high at h=10 and are NEGATIVE at h=2, 3 and 5; the triple
  underperforms unconditional SPY at every horizon 1 to 21. P(triple | SPY at
  high) is 0.319, so it is barely even a subset.
  (scratch/pitch_checks/2026-08-10/c9_sync_52w_high.py)
- **^SKEW bottom-decile at a 52w SPY high.** Reads +0.410% at sign p 0.0205
  (h=5, 35 episodes) and dies three ways: depth-4 entries pay -1.459%, the
  P/C-unconfirmed half (the live state) pays +0.088% against a +0.250% drift,
  and dropping Jan-2018 plus Apr/May-2026 leaves 1.3 bps. Complacency needs the
  two complacency measures to agree; SKEW alone is not a signal, and its h=5
  sign is UP, contradicting the no-put-wall mechanism it was built on.
  (c8_skew_crush_high.py, c8b_skew_teardown.py)
- **Silver thrust from deep inside a drawdown.** The drawdown conditioner
  inverts: deep-dd pays +1.378% at h=10 against +1.780% for the same thrust near
  a 52w high. Nudging the thrust from 8% to 10% flips h=5 to -4.229%.
  Distance-from-high is a U-shaped noise carve, not a conditioner.
  (c7_slv_drawdown_thrust.py)
- **A single market decoupling from a global risk-on thrust (EWZ form).** The
  short side's edge lives in its SHALLOWEST bucket (5d drop under 1% pays
  +1.237%) and reverses at the deep readings that make it interesting (5d below
  -3.5% pays -0.232%). Two years are 53% of the total; ex-2015 and ex-2026 it is
  3.5 bps against 8-14 bps of cost. The gate also sits in bear tape 0.0% of the
  time against an 18.9% base rate, the mirror of the SMH/QQQ over-selection.
  (d3_ewz_decoupler.py, d3b, d3c)
- **Energy's 5d washout into a CPI print.** The CPI anchor SUBTRACTS: XLE
  washout alone +0.464% at h=3, with the pre-CPI window -0.441%. All five energy
  instruments show negative edge against their own drift on the same days, and
  the XLE-minus-SPY form is -0.537% with the SPY leg positive, i.e. a
  short-energy bet rather than a snapback. (c5_energy_cpi_washout.py)
- **Bond vol at a floor as a conditioner, and the difference between a level and
  a return rank.** ^MOVE's 5-day return rank being bottom-decile does NOT mean
  its level is at a floor: the two coincide 30.7% of the time, and on
  2026-08-07 the return rank was 7.5 while the level sat at the 45.6th
  percentile. State which one the mechanism needs. (c3_movefloor_cpi.py)
- **Credit-quality divergence at joint 52w extremes (HYG high / LQD low).** Not
  a sample-size kill: at h=5 the LQD leg's residual against IEF is +0.000pp, so
  there is no credit-specific component at the tradeable horizon and the pair is
  a duration trade with a credit label. Separately the joint state has 4
  episodes since 2007, three in one 2018 summer. Parked with an arm condition.
  (d1_hyg_lqd_unanchored.py)
