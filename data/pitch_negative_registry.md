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

## Method traps (2026-08-11, from an 11-candidate sweep that killed 10)

- **An index effect that dies in translation to the vehicle.** This is not "no
  mechanism" and it is not "cost"; it is a third thing, and it took the whole
  morning to see. ^VIX really does fall across a CPI print: -1.81% at h=3 over
  317 events since 2000, at a 34.4% up-rate, and it is STRONGER after 2018
  (-2.63%, VIX/VIX3M t=-4.70). The tradeable expression collapses anyway,
  because SVXY's return over that window is mostly SPY beta: 2021+ raw +0.724%,
  beta-neutral residual **+0.036% at a 50.0% hit**. SPY did +0.432% on those
  days against +0.186% unconditionally. Before pitching an index phenomenon,
  regress the VEHICLE's cell return on the market over the same window and
  quote the residual. (a1_svxy_cpi_mechanism.py)
- **A sign test against a coin is the wrong null for a drifting instrument.**
  `sign_test(wins, n)` defaults to p=0.5, but an instrument with positive drift
  wins more than half of all windows by construction. Scored against each
  instrument's OWN unconditional hit rate, HYG's CPI cell moves from p=7.9e-05
  to **p=0.040** (144/230 against a 56.7% base) and SVXY's post-break cell from
  ~0 to **0.017** (70/100 against 59.2%). Pass the base rate as `p=` whenever
  the claim is a hit rate on a drifting asset. (a1_hyg_cpi.py)
- **Feeding `pct_rank` a return series.** `pitch_lab.pct_rank(s, n)` takes a
  PRICE series and computes `s.pct_change(n)` internally. Passing it a return
  series ranks `pct_change(n)` OF `pct_change(n)`, a second difference on a
  series that crosses zero constantly, and it fails silently because the output
  still looks like a 0-100 rank. On 2026-08-11 this corrupted six of eight
  price-state cells in the morning's own recon: the "GDX 5d rank >= 95" trigger
  it produced overlapped the real one on **8 of 272 days (7.6%)**, and for the
  XLV cell today's true rank was 100.0 while the broken statistic read 15.5, so
  today's state was not even inside the population being measured. Two of the
  surface map's dismissals rested on it. Sanity-check any new trigger by
  printing TODAY's value of it and confirming it matches the tape file.
  (a4_c4_c11_teardown.py, 02_price_state_recon.py)
- **The event inside your own hold window, as opposed to at your anchor.** A
  price-state trade entered the session before a print holds that print whether
  or not the thesis mentions it. XLE on a crude thrust pays +0.476% with no CPI
  in the window and **-1.204% with one** (Welch t -2.28), and that interaction
  is not the CPI main effect, which is only -0.084% for XLE across all days.
  Always split the historical trigger set by what lands inside the hold, not
  just by what the anchor is. (a2_c5_cpi_cross.py)
- **A future event date silently manufacturing a fake anchor.** Building
  "k sessions before event E" by `searchsorted` needs an explicit
  `if loc >= len(dates): continue`. Without it the next, unrealised event
  resolves to the end of the index and mints a spurious recent anchor; this
  produced a bogus 2026 row before it was caught. (a3_c7_iwm_jacksonhole.py,
  01_event_class_recon.py)
- **An anchor deserves an offset ladder before it deserves a check.** Slide the
  entry session from -5 to +3 around the event. A spike at one offset that
  decays either side is an event; a plateau is month position wearing an event
  label. SVXY's h=3 return peaks exactly on the CPI eve (+1.499%) and falls to
  -0.074% by +2, while SPY's ladder is flat across the whole range (+0.181% at
  the eve against +0.188% five sessions earlier) — which is how the morning
  learned the cell was about volatility rather than direction, before learning
  the vehicle could not harvest it. (03_cpi_offset_ladder.py)

## Cells swept and empty (2026-08-11)

- **Short vol held through a CPI print (long SVXY, eve to +3 td).** The headline
  cell is +1.130pp of excess at a 71.0% hit, and it is an artifact of two
  things: 44% of the sample predates the 2018-02 -1x to -0.5x leverage cut, and
  what remains is SPY beta (see the translation trap above). The registry's
  existing "post-CPI vol crush died after 2018" entry tested the print-session
  open to +2; that dead segment is **85% of this window's return** and decays on
  the same schedule (full +1.270%, 2011-17 +2.539%, 2018+ +0.338% at t 0.99).
  Distinct from the event sleeve's V4 (28.8% calendar overlap with V2/V4
  combined, zero this month). (a1_svxy_cpi_registry_and_leverage.py)
- **Long HYG into a CPI print.** Era sign flip, +17.8 bps pre-2018 to -2.8 bps
  from 2018 and -1.1 bps from 2021. Not credit and not duration: the residual
  against IEF carries a -0.10 loading, and against SPY it is +2.9 bps at t 0.42,
  which is 0.6x an HYG round trip. h=3 is a lone positive in an otherwise
  negative horizon profile (h=2 -8.6 bps, h=5 -1.7, h=10 -15.4). (a1_hyg_cpi.py)
- **Adding a second metals leg beside a live one.** Both SLV and GLD price well
  on a miner-led thrust and both fail the only test that mattered, which is what
  they add to a position the book already holds. SLV correlates +0.708 with the
  live GDX leg and paid **-2.716% at a 34.0% hit on the 50 episodes where that
  leg lost**; GLD correlates +0.724 and, added at 0.25x, 0.50x or 1.00x, leaves
  the book's hit rate at **58.3% at every weight** while widening the worst
  episode from -35.40% to -45.64%. That is size, not diversification. Check
  correlation against live exposure BEFORE pricing a second leg in the same
  complex. (a4_c4_slv_basket.py, a4_c4_c11_teardown.py)
- **Long IBB on healthcare 63d leadership.** Under the correct trigger the sign
  inverts: excess against its own drift is negative at every horizon from
  -0.267% (h=1) to -0.959% (h=10), bootstrap P(mean<=0) 0.985, record 41-47.
  Regressed on XLV the alpha is -0.126% at beta 1.04, so there is no biotech
  residual. The one positive slice is the rank>=100 bucket nested inside a
  monotonically negative sweep, whose complement (the 23 episodes between rank
  99 and 100) averages -1.156%. The 2013-15 biotech-bubble hypothesis is NOT the
  explanation and can be dropped: those 9 episodes pay -0.351% against -0.462%
  for everything else. (a4_c8_ibb_xlv.py)
- **Long XLE on a crude one-day thrust.** XLE's unconditional crude beta is
  0.479 (t 55.8); net of it the cell residual is +0.291% at a 49.3% hit and sign
  p 0.596, so the 67.2% headline is crude follow-through wearing an equity
  ticker, and no vehicle edge exists either (risk-adjusted 0.212 for XLE against
  0.194 for a vol-matched USO). 2009 and 2020 are 72% of the total. It is also a
  straight bet against the book: Overbot Vol Spike fired **47 SHORT signals on
  USO >= +5% days at avgR +0.29**, and 12 of 14 energy positions the book held
  across such a window were short. Parked with a band condition, see the
  watchlist. (a2_c3_round1.py, a2_c3_beta_and_book.py)
- **Short UNG through a CPI window.** A filter that does not filter, and the
  placebo is what proves it: shifting the identical short to anchors k=-8..+12
  sessions from the print gives excesses from -0.936% to +0.602%, and **two
  nonsense anchors beat the real one**. Excess over UNG's own bleed is +0.526%
  with a bootstrap CI of [-0.365%, +1.407%]; mid-month position accounts for
  what is left (tdoms 17/18/19, never CPI days, pay +0.386/+0.437/+0.329%).
  Worst 5-day short window -43.56%, seven months ago, and the edge halves in
  exactly today's near-52w-low state (+0.247% against +0.555%). (a2_c6_ung.py,
  a2_c6_placebo.py)
- **Short IWM on a Jackson Hole -13td anchor.** Wrong-signed in midterm years,
  where the short LOSES 29 bps at 3-of-6 down, against non-midterm -0.572% at
  16-of-20. Three sessions are the whole cell: dropping the best short year
  takes it from -0.374% to -0.194%, two years to -0.038%, three years to
  **+0.024% and a flipped sign**, with 2011 (-4.89%) and 2010 (-3.93%) supplying
  the bulk. Also not a small-cap story, since the IWM-SPY spread is only -0.110%
  at a 38.5% hit while SPY alone does -0.264%. And genuinely NOT a CPI cell in
  disguise: the August CPI lands inside the h=1 hold in only 4 of 26 years.
  (a3_c7_iwm_jacksonhole.py)
- **Nearest-neighbour tapes as a directional signal.** The usual kill for an
  analogue idea is that no analogue exists; here one does, and the idea dies
  anyway. Today's nearest neighbour is 0.57 sd-units away in a 6-dim
  standardised metric against a typical day's 0.39, and the neighbours do not
  cluster badly. But forward SPY excess over the unconditional baseline is
  **negative at every horizon** (-0.142 / -0.082 / -0.062 / -0.261 at
  h=1/3/5/10) with every sign p >= 0.41, so conditioning on the full joint state
  buys less than a random long. Swapping GLD for GDX in the feature vector keeps
  only 11 of 20 neighbour dates. A six-dimensional coincidence is not a
  mechanism. (a3_c10_nearest_neighbour.py)

## Method traps (2026-08-12, from a 12-candidate sweep that killed 11)

- **The rescue rule cuts both ways: cross the killing conditioner with the
  search that found it.** The 2026-08-07 entry says always cross a rescuing
  conditioner with the killing one before believing the rescue. Today the
  killing conditioner was itself discovered by looking, and it owes the same
  charge. The PPI-with-CPI-on-its-eve cell (N=55, +22.2 bps tdom-matched, 35-20)
  crossed with August is 0-for-4 at -0.85%, which reads as decisive until you
  price the search: the permutation probability that SOME month with N>=3 looks
  that bad is **0.087**, August ranks 6th of 12 in the parent's monthly means,
  and its own parent cell is a 12-12 null rather than a negative. Three of the
  four losses were -0.66%, -0.23% and -0.16%. A conditioner found by scanning
  twelve months is not the same object as one specified in advance, and the
  asymmetry is the trap: nobody would ship a cell on 4 observations, so nobody
  should kill a 55-observation cell on 4 either.
  (2026-08-12/a2b_c1_month_stability.py, r1_c1_august_adjudication.py,
  a8_composer_verify.py)
- **A control built from other instances of the treatment tests a different
  question than it is quoted for.** The month adjustment that produced this
  cell's scariest number (bootstrap P(mean<=0) 0.137, from 0.012) subtracted the
  mean of OTHER PPI print sessions in the same month, so it asks "does the CPI
  gate beat the parent" and not "is the cell positive". Rebuilt against
  non-event days the same cell reads +24.3 bps with bootstrap 0.022, and under a
  month x trading-day-of-month double control +24.7 bps at a 69.1% hit. Name the
  null a control implies before quoting the number it produces.
  (r4_c1_august_confound_and_verdict.py)
- **The gap-share test falsifies an 08:30-release mechanism in one line.** A
  release at 08:30 ET is fully contained in the prior-close to 09:30-open gap,
  so a cell claiming to harvest that release must earn its return there. Short
  USO across a PPI print earns **18%** of its excess in the gap and 81% between
  09:30 and 16:00, after the news is public, which kills the mechanism without
  any statistics. Run the decomposition before the battery, not after.
  (b1b_c5_ppi_mechanism.py)
- **A recon table's per-class hit column is the LONG side's hit rate.** The
  morning's event x class recon showed SVXY at "55%" on the PPI eve with a
  negative mean, and that 55% is the hit rate of a LONG. The short that was
  actually being considered wins 44.1% with a median of -0.218%. Flip the record
  before reading a hit rate as support for the side you are pitching.
  (b2_c6_ppi_svxy_short.py)
- **`close_panel` unions every member's dates, so a rolling 52-week window can
  be silently wrong.** Adding ^VIX to a panel injected three extra sessions into
  SVXY's index and moved its rolling 252-day max, which made a live 52w-high
  state read as not live. Compute distance-to-extreme on the single instrument's
  own series, never on a panel column.
  (b2b_c6_svxy_52wh_compose.py)

## Cells swept and empty (2026-08-12)

- **Short commodities or short vol across a PPI print session.** Both died on
  the placebo anchor ladder, which is now 3-for-3 as a killer in this repo. USO
  at the real k=2 anchor is -0.222% and ranks 2nd of 21 offsets (empirical
  p 0.095) with a nonsense anchor six sessions later more negative. Short SVXY
  produced the best statistic of the morning, a beta-neutral residual negative
  99 of 177 at sign p 0.0053 with beta explaining only 14% of variance, and then
  an anchor EIGHT SESSIONS AFTER the print scored 0.0043. Note for the record
  that the registry's SVXY beta objection was not what killed this one.
  (b1_c5_ppi_energy_short.py, b2c_c6_residual_placebo.py)
- **The PPI print session translated into IEF or LQD.** The edge is proportional
  to duration and nothing else: TLT/IEF excess ratio 2.25 against a daily-sd
  ratio of 2.10, so excess per unit of sd is 0.299 vs 0.280. After cost TLT
  strictly dominates (net 24.4 bps at 10.8x, IEF 10.0 at 6.0x, LQD 2.4 at 1.8x),
  and LQD's residual against IEF on the cell is -3.15 bps, so there is no credit
  component to translate. (a3_c2_vehicle_translation.py)
- **Long duration against short SPY on an inflation-print anchor.** Regressing
  the cell's TLT return on SPY leaves alpha +25.86 bps against a raw mean of
  +25.84, and the beta is NEGATIVE (-0.09), so the short-SPY leg is a
  long-duration proxy that doubles the bet: sd 0.896% -> 2.078%, hit 63.6% ->
  43.6%. A negative-beta hedge leg is not a hedge. (a4_c3_spread_vs_spy.py)
- **The utilities washout on a 21-day rank is the same corpse as the z10 form.**
  58.8% of rank21<=5 days sit inside the already-dead z10<=-1.5 cell and the
  corpse scores better (+0.226% vs +0.219%). It also inverts under the
  SPY-near-high gate exactly as the 2026-08-07 kill predicted: -0.651% at a
  33.3% hit (h=3) and -0.937% at 28.6% (h=5) with the gate on, against +0.260%
  and +0.083% ungated. **Utilities are now dead in five expressions.** The
  watchlist entry that asked for this check is closed by it. (c9_xlu_washout.py)
- **The semis laggard OUTRIGHT, not just the SMH/QQQ pair.** The trigger puts
  SPY below its 200-day on 59.4% of its days against a 24.2% base rate, so the
  registry's "regime bet, not relative value" kill transfers from the pair to
  the outright verbatim. Today's state was also outside the sample: SMH is below
  its own 200d on 78.1% of trigger days, and trigger days sitting >=15% ABOVE it
  number 4 of 347, declustering to one episode. (c10_smh_laggard.py)
- **A skew spike with a low-vol filter attached.** The filter subtracts: skew
  rank5>=95 ALONE pays +0.372% excess at h=5 over 185 episodes (sign p 0.026),
  adding the VIX rank<=35 leg discards 81 episodes and halves it to +0.175%, and
  the VIX leg alone is -0.075%. The "complacency" framing was also falsified by
  its own window, since VIX FALLS 2.33% across it against +0.66% all-days. The
  skew-alone cell is parked to the watchlist with a regime trigger.
  (c11_skew_vol_divergence.py, c11b_skew_alone_probe.py)
- **The Brazil five-day washout, long form.** Distinct from the registry's dead
  EWZ decoupler short, and dead on its own: top-2 episodes (2008-10-24,
  2020-03-20) are +60.6pp of a +85.8pp h=3 total and dropping them leaves
  +0.111%; tightening rank5 from 3 to 1 flips the sign to -0.337%. The
  shallow/deep split that killed the short cannot even be run here, because
  rank5<=3 fires on a 5-day drop deeper than -3.5% 100% of the time.
  (c12_ewz_washout.py)
- **Fading a live pitch position whose exit overlaps your own.** Not a
  statistical kill and worth stating as a rule. A short GDX entered at h=3
  against a live long GDX leg exiting on the same close is net zero exposure
  over the overlap, i.e. an early exit executed with two round trips and borrow
  instead of one cancellation. Position management is not a pitch. (The cell
  failed anyway: drop-2-best takes h=3 from +0.480% to -0.565%, and moving the
  GDX rank cut 99 -> 97 flips the sign.) (b4_c8_metals_thrust_fade.py)

**Correction owed to a published number.** The 2026-08-10 watchlist entry for
the PPI curve cell quoted "2018+ +0.133%". That is an average of two different
cells: +0.278% when a CPI printed on the eve and -0.017% when it did not. The
parent PPI cell has no modern-era edge outside the conditioner that happened to
be live on 2026-08-12. (a2_c1_gate_attribution.py)

## Method traps (2026-08-13, from a 12-direction sweep that killed all 12)

- **A single-ticker result has to be priced against its reference class, not just
  against its own bootstrap.** The morning's only survivor reached round 3 looking
  clean: long IHI on a 21d-rank-100 thrust out of a >=10% drawdown paid +1.499% at
  h=5 over 16 episodes (12-4), excess +1.267pp over its own drift, within-IHI
  bootstrap P(mean<=0) 0.0022, positive in 9 of 9 firing years, both eras
  positive, monotone in the rank gate and flat across the drawdown gate. Running
  the IDENTICAL rule on 27 sector ETFs ended it: Cochran Q 24.56 on 26 df (p
  0.544, I-squared 0.0%), fixed-effect common excess -0.035pp, observed
  cross-sectional sd 0.936pp against a mean sampling SE of 1.054pp (ratio 0.89, so
  the whole -2.96 to +1.33 spread is sampling noise), and a permutation of the
  same estimator produced a MAXIMUM of +1.92pp out of pure noise against IHI's
  observed +1.211pp. Family-wise p 0.9330: the result is a BELOW-median draw from
  the distribution of "best ticker under no effect". Cheap to run whenever a cell
  names one instrument out of a natural peer group, and stronger than any
  within-instrument robustness test, because it measures how much dispersion the
  estimator manufactures at N~16 when nothing is there. Note what it does NOT
  rely on: era stability, LOYO and concentration all PASSED here (LOYO floor
  +1.322%, drop-3-best +0.889%). (r1_cross_sector_placebo.py,
  r1b_multiplicity_max_of_k.py)
- **The percent/fraction double-scale trap fires in the direction that flatters
  the idea, too.** The first cross-sector permutation compared fraction-unit
  resampled means against a percent-unit observed excess and returned p=0.0000 IN
  FAVOUR of the survivor; corrected, the same test reads 0.9330 against it. The
  2026-08-07 entry described this trap as making a result look 100x too big. It
  equally makes a NULL look like a discovery. Assert units at the boundary of
  every resampling loop. (r1b_multiplicity_max_of_k.py)
- **A rank thrust can be a denominator roll rather than a move.** IHI's headline
  +13.94% 21-day return jumped 4.90pp on a session where price moved +0.18%,
  because the 21-day reference bar rolled from 51.34 to 49.22 (-4.13%). A
  percentile gate on a rolling return can fire on nothing happening today. Check
  the trigger day's own price change against the change in the lookback return,
  and prefer a magnitude gate when the mechanism is about a move.
  (r4_live_state_honesty.py)
- **Placebo ladders keep earning their place, and 4-for-4 became 6-for-6.** The
  Jackson Hole TLT cell (+1.162%, 24 events, 17-7, sign p 0.0320) ranks 14th of
  127 offsets, with off=-9 paying +1.369%; the 98th-percentile contango cell ranks
  10th of 21, with every offset from -4 to -10 paying +4.55% to +5.88% at a
  90-100% hit; the macro-vacuum cell ranks 6th of 17 at h=10 and 14th of 17 at
  h=5. In all three the ladder is a PLATEAU rather than a spike, which is the
  signature of month position rather than an event. (b1b_c4a_round2.py,
  a1_c1_termspread.py, a2_c5_macro_vacuum.py)
- **An "event" cell inside one month owes a MONTH-OF-YEAR control, not just a
  trading-day-of-month one.** TLT's 10td lag-1 forward return runs Nov +1.059%,
  Aug +0.494%, Jun +0.498%, Jul +0.451% against Oct -0.432%, Sep -0.220%, Apr
  -0.240%. That is large enough to swallow a +1% conditional mean whole, and it is
  what the Jackson Hole cell turned out to be: the unconditional Aug 6-16 window
  pays +1.025% at t=6.90 over 189 starts with no event involved, against the JH
  anchor's +1.162% over 24. This settles the tdom debt the W1 watchlist entry has
  owed since 2026-08-10 and raises its bar: a September NFP entry sits in TLT's
  second-worst month. (b1b_c4a_round2.py, b1c_c4a_midterm_control.py)
- **A vol-carry state can be a LAGGING marker of the move that created it.** On
  98th-percentile VIX3M/VIX contango triggers, SVXY's TRAILING 21-day return has a
  median of +10.46%. The gate does not identify carry about to be harvested, it
  identifies carry already harvested, which is why every negative offset in the
  placebo ladder beats the real anchor. Check the instrument's trailing return on
  trigger days before believing a premium story. (a1c_c1_intersection_cell.py)
- **"Positive in 9 of 9 years" is not evidence on its own.** Exactly 1 of the 27
  reference-class tickers achieved it and the global null expects ~0.3, so the line
  carries p~0.3. A perfect year record at ~2 episodes a year is a small number of
  coin flips wearing an impressive sentence. (r3_registry_failure_modes.py)

## Cells swept and empty (2026-08-13)

- **The ten sessions into Jackson Hole, all three cross-asset legs.** Rates: the
  anchor is decoration on an August seasonal (above), the mechanism loses in its
  own window (entering after the conference is -0.204%, +1 to +3 sessions -0.607%
  to -0.848%), 2018+ is +0.590% at 4-4, the duration-neutral residual against IEF
  is +0.122% at a 50.0% hit, and today's rung has no precedent at all (zero JH
  anchors in the sample had TLT within 1% of its 52w low). Gold: 10-11 at +0.577%,
  92% of it two episodes, midterm -1.213% at 1-4, and the independent Aug 6-16
  midterm control agrees at -0.859%, t=-2.53. Dollar: 13-13 at +0.090%, drop-best
  flips the sign, the midterm cell is entirely 2022's +3.00%, and 9 bps is 4.5x a
  DX round trip against the 5x bar. The JH anchor is now examined on rates, gold,
  FX and (2026-08-11) small caps, and is empty in all four. (b1_c4_jh.py,
  b1b_c4a_round2.py, b1c_c4a_midterm_control.py)
- **The macro vacuum, i.e. a hold with no 08:30 release inside it.** A gate that
  does not filter: it agrees with plain "no FOMC decision inside the hold" on 278
  of 318 anchors, and dropping FOMC from the release set collapses the h=10 excess
  from +0.352pp to +0.051pp. The mechanism is falsified inside its own window,
  since VIX's forward path in the vacuum is +1.000% at h=5 against +1.008% for all
  days, i.e. exactly the unconditional path; the only real effect is the complement
  (+3.659% when a release IS inside), which is event risk rather than premium
  decay. No dose response (Spearman of gap length against forward return +0.084),
  and 2018+ matched excess is -0.016pp. (a2_c5_macro_vacuum.py,
  a2b_c5_dose_response_and_fomc.py)
- **Term-structure percentile as a short-vol entry, in both directions.** Long
  SVXY: the best cell's 90% hit is a 1.63 beta to a tape 0.10% off its high, and
  the SPY residual hits 50.0% against a 52.4% base rate. Short SVXY: wrong-signed
  on the only legal vehicle, -1.82% at h=5 on 3-10, because VIX does rise (+5.73%,
  74% up-rate) and the spread does compress (-4.39pp) while SVXY still gains
  through futures roll. Neighbouring definitions are negative and carry -14.5% and
  -18.1% windows the winning cell excludes. (a1_c1_termspread.py,
  a1b_c1_definition_and_overlap.py, a1c_c1_intersection_cell.py)
  CORRECTION (2026-08-13, per McKinley): the journal kill's closing claim —
  "naked short UVXY is registry-dead, so there is no other expression to fall
  back on" — cited the wrong direction. Short UVXY is a SHORT-vol expression
  (UVXY is +1.5x long VIX futures) and can never be the fallback for a long-vol
  idea; the UVXY registry entry it grabbed belongs to the short-vol side of the
  sweep. The correct statement: the remaining long-vol vehicles are LONG
  UVXY/VXX, and they die on the same roll mechanism, only harder — SVXY +1.82%
  at h=5 implies the front futures basket FELL ~3.6% while spot VIX rose +5.73%,
  so long VXX (+1x) loses ~-3.6% and long UVXY (+1.5x) ~-5.5% plus decay over
  the same windows. No ETP delivers spot VIX; the futures curve is the only
  tradeable surface, and at 98th-pctile contango the roll drag swamps the spot
  rise in every direction-consistent vehicle. The kill verdict stands; the
  stated fallback reasoning was garbled and must not be reused as precedent.
- **The fragility dial's RATE OF CHANGE as a directional signal.** Flips sign on
  its own threshold (+0.716% at a 30-point 21d rise, +0.004% at 25, -1.132% at
  35), on its own lookback (-0.454% at 10d, -0.211% at 42d), and on its own
  VINTAGE (+0.716% on the sizing parquet, +0.061% on the research recompute, whose
  own last reading is ma10 41.64 against the sizing series' 72.24). Gate
  attribution: the SPY-near-high leg SUBTRACTS, and the level-only form is -1.034%
  at 2-4, the opposite sign and the dead book-wide throttle re-skinned. 87.3% of
  the cell's days are pre-2026-07-02 recompute vintage and the live state's
  PIT-only slice is ZERO days. Every defensive expression loses (long TLT / short
  SPY -1.244% at a 22.2% hit). The dial is a sizing input; it is not a direction.
  (b2_c9_dial_thrust.py)
- **Defensive dispersion, long XLU against short XLV.** Utilities are now dead in
  SIX expressions. What is new is that the escape hatch was tested and closed: the
  XLV short is the LOSING leg, -0.516% at h=3 against its own -0.117% short drift
  and -0.283% as a residual against SPY on 11-15, because XLV keeps rising after
  the trigger. The joint state is worse than either gate alone (pair -0.088%
  against +0.035% for the XLV gate by itself), and today's live SPY-near-high
  subset is the worst slice at -0.911% on a 22.2% hit. (c7_xlu_xlv_dispersion.py)
- **A single market breaking inside an intact thrust, FXI form.** The third country
  tested on this shape after EWZ (twice), failing the same two ways: the pair as
  pitched is wrong-signed (residual against EEM -0.277%, because EEM paid +1.084%
  against FXI's +0.834%), and the rank cut is a knife edge (20 gives +0.834%, 25
  gives -0.003%, 30 gives -0.426%). The EEM-positive gate also puts SPY below its
  200d on 0.0% of trigger days against a 19.7% base rate, the identical
  over-selection that killed EWZ and SMH/QQQ. Treat "one market decouples from a
  risk-on thrust" as a dead FAMILY, not three coincidences. (c8_fxi_eem.py,
  c8b_fxi_tight_teardown.py)
- **Index-pair mean reversion on a laggard index, DIA form.** A nested subset that
  reverses its parent's sign: without the SPY-positive gate the parent is -0.077%
  beta-neutral at h=3 on 26-34 with a day-level t of -2.13, every threshold
  neighbour is negative, and the loosest version with any observations pays 15.4
  bps against a 20 bp two-leg round trip (0.8x cost, top-2 episodes 79% of the
  total). The cell exactly as specified has occurred once in 26 years, which is
  today. Useful by-product: the DIA/SPY residual is NEGATIVELY correlated with the
  2026-08-11 QQQ/SPY pitch (-0.363 to -0.442 at h=1/3/5), so index pairs are not
  interchangeable re-skins of each other. (c10_dia_spy.py)
- **Fading a 1x sector ETF on a leveraged-ETF fade filter.** IHI passed 5 of the 6
  legs of the book's 3x ETF Overbot Fade on 2026-08-12 (r5 95.2, r10 94.4, r21
  100.0 all above 85; r126 50.4 and r252 41.3 both under 65), failing only r2.
  Measured on the 1x ticker the identical shape INVERTS: the short pays -0.953% at
  h=3 on 5-15, negative in all 9 firing years and at every threshold neighbour.
  The book confines that family to leveraged names by design, and this is the
  measurement showing the confinement is load-bearing rather than incidental.
  (c6_ihi_thrust.py)

## Method traps (2026-08-14, from an 11-candidate sweep that killed all 11)

- **A LEVEL percentile is still a rank trap when the window is a trailing year
  and the series has secular drift.** The 2026-08-10 entry said quote the LEVEL,
  not the return rank. This morning did exactly that and got trapped one layer
  down: ^SKEW at 134.37 is the **2.0th percentile of its trailing 252 days** and
  the **77.1st percentile of full history**, because SKEW's median has drifted
  from 114.04 (2000-04) to 143.11 (2026). Every historical trigger day sat at
  109.7-119.5. The premise "crash protection is being dumped" was false in
  absolute terms the whole time: tail insurance was priced ABOVE its 2018+
  median. On any series with a secular level drift, quote the level against full
  history AND the modern era, never against a trailing year alone.
  (a1_c1_skew_bottom_pole.py, a1b_c1_kill_confirm.py)
- **The alphabetical-selection placebo, a cheap and brutal test of any "pick the
  k names that did X" rule.** The insurance short's 14-name basket paid +1.055%
  at h=10 on 9-3, but the tradeable four-name forms gave **+0.905% for the four
  most washed against +1.568% for the four ALPHABETICALLY first**. A selection
  rule that ignores the signal entirely beat the one that uses it, so nothing
  could be attributed to which names broke. Run this whenever a basket idea has
  to be cut down to the grammar's 4-leg cap; it also catches the case where a
  research basket works and no tradeable subset does, which is a kill on
  tradeability. (b2c_short_reference_class.py)
- **An "intact trend" gate on a breadth washout is an INVERTER, not a filter.**
  Insurance breadth alone: **+0.917%** at h=5 over 233 episodes (t 2.589). Add
  the intact-63d-uptrend gate: **-0.789%** on 5-8. The split inside the parent is
  +0.885% not-intact against -0.789% intact, so the pitched interaction was
  precisely the losing half of a cell that only works without it. Gate
  attribution catches this in one run and it is the first thing to try on any
  "washout inside an uptrend" construction. (b2b_insurance_round2.py)
- **An instrument that changed leverage mid-sample is two instruments.** SVXY
  went -1x to -0.5x on **2018-02-28**: pre-break daily sd 4.60%, VIX beta -0.477,
  worst day **-88.41%**; post-break 2.41%, -0.256, -21.43%. Five of six episodes
  in the candidate cell sat on the security that no longer exists. Check the
  vehicle's own regime history before pooling a sample across it, the same way
  an era split is checked. (a2_c1b_svxy_translation.py)
- **A nearest-neighbour analogue whose answer moves with K is a construction,
  not a finding.** h=5 ran -0.224%, 0.000%, +0.150%, -0.227% across
  K=20/40/80/150 and h=3 flipped from -0.387% (K=20) to +0.224% (K=80). Dropping
  one feature from the distance metric changed 95% of the neighbour set. Scan K
  and the feature set BEFORE reading any forward number, and charge the cell for
  the grid: 24 cells scanned, exactly one clearing against 1.2 expected by
  chance. (a3_c7_nearest_neighbour.py)
- **An earnings anchor owes the same placebo ladder as a macro anchor, and the
  first one tested failed it.** The pre-print washout's true anchor ranked **3rd
  of 15** offsets (shift -13 at +0.396% and shift -4 at +0.381% both beat the
  real +0.331%), the whole ladder sitting in a +0.121% to +0.396% band, so true
  minus placebo mean was +0.084pp. Stripping the print left the same washout gate
  earning +0.200% on the same names. The earnings lane is not exempt from the
  ladder just because the event is company-specific. The ladder is now 7-for-7.
  (c3e_liquid_ladder.py, c3c_preprint_round2.py)

## Cells swept and empty (2026-08-14)

- **A cell that beats every control except the local one, and is LIVE today, so
  it will be re-found.** Long SPY with VIX's LEVEL in its bottom decile while SPY
  is within 0.5% of its 52w high, h=10: 88 episodes, **+0.637%, t 4.09, 67-21,
  sign p 0.0000**, era-stable (2018-2025 +0.939% at t 3.19), 31.9x cost. It dies
  on CTRL-c: the local +/-126td neighbourhood ex-trigger pays **+0.634%**, an
  edge of **+0.003pp**. It is 100% "these days live in a good regime", which is
  exactly what CTRL-c exists to catch, and it is the single best illustration in
  the registry of why the local control is not optional. Midterm h=5 is +0.027%
  (13 episodes) against non-midterm +0.181%. (a4_gateoff_byproduct.py)
- **The bottom pole of the skew distribution, in both directions.** Long SPY: the
  skew leg ALONE pays +0.217% (79 episodes) against a +0.382% unconditional, i.e.
  worse than doing nothing, and adding it to a VIX-plus-near-high cell discards 83
  of 88 episodes to add +0.31pp. Strict/core definitions flip sign (-0.876% on
  0-for-3 against +0.230%), and there are **zero episodes 2018-2025** at every
  definition. Long SVXY: gated h=5 is **-1.328%** against +0.208% gate-off, the
  sign flips by horizon (-0.130% / -1.328% / +3.990% at h=3/5/10), and h=10's top
  two episodes are 103% of the total. The skew SPIKE cell stays parked on the
  watchlist; the bottom pole is now dead in both vehicles.
  (a1_c1_skew_bottom_pole.py, a2_c1b_svxy_translation.py)
- **The insurance industry, in four expressions.** Long the basket (-0.789%,
  5-8, and the gate inverts, above), the basket against XLF (both legs lose:
  basket -0.789%, XLF -1.072%, so the +0.282% spread is a difference of two
  losses on 3-10 with top-2 episodes at 203% of total, and 1.1x cost), the single
  strongest name (h=5 -1.128% on 3-10, bootstrap P(mean<=0) 0.993), and the short
  (alphabetical placebo, above). The reference class closes it: the identical
  short on **10 industry groups** gives Cochran Q 6.49 on 9 df (p 0.690,
  I-squared 0.0%), a common excess of +0.819pp, and insurance ranking **2 of 10**.
  Any industry washing out inside an intact 63d trend does roughly the same
  thing; the insurance label carries nothing. (b2_insurance_breadth.py,
  b2b_insurance_round2.py, b2c_short_reference_class.py)
- **Producers against the barrel on a 63-day divergence, all three readings.**
  Every threshold at or below 17pp is negative (15pp: -0.289% on 12-12) and the
  sign flips at 18pp, against a live +18.69pp spread, so the cell was 0.7pp of
  luck away from being dead by inspection. The cells that pay have NEGATIVE
  medians (19pp: +2.37% mean, -0.26% median, **79% of the total in two April-2020
  episodes**). Era reverses (pre-2018 +1.363% on 11-4 against 2018+ -3.043% on
  1-8) and trigger days sit below SPY's 200d on **41.7% against a 20.3% base
  rate**, the same bear-tape over-selection that killed the FXI pair. Two
  by-products worth keeping: **USO's roll decay is NOT shortable at pitch
  horizons** (unconditional h=5 mean -0.003% with a POSITIVE median of +0.251%,
  so the -76.7% lifetime total return is variance drag rather than a harvestable
  drift, and a 5-day short starts 10 bps down), and a 63d relative-performance
  spread is a bear-tape selector by construction.
  (b5_xle_uso_divergence.py, b5b_xle_uso_round2.py)
- **The pre-print washout, i.e. the whole "buy the name that sold off into its
  own print" lane on liquid names.** Anchor ladder above; additionally the print
  premium is 15.3 bps = **1.5x** a 10 bps single-name round trip, and the two
  names the idea was about are negative in their own cell (TJX -0.543% over 13
  qualifying events, ROST -0.425%). The retail x August cell (N=38, +1.613%, sign
  p 0.0168) is a **$5-15 penny-bucket artifact** on a survivorship-biased panel
  (that bucket +0.729% against $40-100 at +0.207%) and does not replicate on
  liquid names (+0.167%, t 0.37); it is also priced out at ~70 searched cells.
  What survives is the null: a generic 5-day-washout reversal on liquid names
  (k=5/h=3, +0.534%, t 4.17, 2018+ +0.709%), which is the book's own dip-buy
  family and must not be re-dressed as a pitch.
  (c3_preprint_washout.py, c3c_preprint_round2.py, c3d_liquid_depth.py)
- **SMH into the NVDA print.** The mechanism is falsified inside its own window:
  by print MONTH, the August print that is live pays **-0.322%** against SMH's
  unconditional h=7 of +0.424% (edge -0.746pp), and 2020+ August prints are
  -1.339%. The laggard gate does not filter (rank63 <25 gives +1.125%, BELOW both
  gate-off +1.353% and the middle bucket +1.566%, on 12-12), and the anchor
  ladder is a plateau with the true anchor 6th of 13. The headline all-prints
  +1.353% is carried by November (+2.959%) and pre-2013 (+2.250% against
  2013-2019 +0.273%). (c4_nvda_print_runup.py, c4b_august_print.py)
- **The pre-opex WEEK entered on the Friday before**, which is the definition the
  2026-08-07 "run into August opex" kill did NOT measure. Also dead, and worse
  than doing nothing: +0.119% (N=319) against SPY's +0.192% all-days and +0.185%
  all-Fridays, hit rate 56.7% against 58.1%, anchor **14th of 21** offsets,
  2018+ -0.033%, top-2 episodes -25.82pp of a +37.98pp total, August subset
  +0.062% over 26 years. The opex window is now examined from the NFP close over
  10 td, from the Friday before over 5 td, and as VIX-expiry week, and is empty
  in all three. (c6_preopex_week.py, c6b_aug_midterm.py)
- **Bond vol compressed into a macro-quiet window, long duration.** Killed on the
  premise, which is the 2026-08-10 MOVE trap firing a second time: ^MOVE's LEVEL
  is at the **33.1st percentile** and 24.1% ABOVE its 52w low, only the 5d return
  rank (15.5) is low, and the two states overlap on 32.5% of days. As stated
  (level pctile <=10) the cell returns -0.100% over 200 episodes. The macro
  vacuum leg again does not filter (+0.074% vacuum against +0.094% print-inside),
  and TLT unconditional pays +0.455% at these trading days of month and +0.430%
  in August, so being long TLT on any mid-month August day beats the cell.
  (c8_bondvol_vacuum.py)

## Cells swept and empty (2026-08-17)

- **The bare August TLT month-position seasonal, i.e. the effect that killed the
  Jackson Hole anchor, tested as the trade itself.** It is real on the full
  sample and it is a BOND-BULL FOSSIL. At matched tdom 4-12, h=10 lag-1: all
  history +0.990% at 17 of 24 years (sign p 0.0320), ranking 2 of 12 months.
  By era: 2002-2012 **+1.989% at 10 of 11 years**, 2013-2017 +0.398%, 2018-2020
  +0.723%, **2021-2025 -0.455% at 2 of 5**, and 2018-2025 **-0.013% at 4 of 8**.
  The regime split names the mechanism and falsifies it live: the window pays
  +1.409% when yields are falling against +0.496% when rising, while
  unconditional TLT is regime-FLAT (+0.167 / +0.160) — so the seasonal is a
  proxy for the secular bull, not a calendar effect. August is also the only one
  of the three good duration months dead post-2018 (Nov +2.093%, Jun +1.208%).
  The IEF-neutral residual at beta 1.914 is +0.254% full but +0.024% for 2018+,
  so the TLT-specific component died with it. Do not re-open on a fresh event
  label; the anchor was already shown to be decoration (2026-08-13) and this
  closes the underlying cell. (a1_c1_tlt_lateaug.py, a1b, a1c,
  r1_verify_august_and_november.py)
- **NOVEMBER is the same cell alive**, kept as a watchlist park rather than a
  kill: +1.590% over 24 years at 20-4 (sign p 0.00077), rank 1 of 12,
  **2018-2025 +2.093% at 8 for 8** and 2021-2025 +2.325% at 5 for 5, Bonferroni
  0.0093 after charging the full 12-month scan, IEF-neutral residual +0.358% for
  2018+. Recorded here because the August kill above is only interpretable
  beside it: the month-of-year table is the control, and it says the duration
  seasonal migrated rather than vanished. (r1_verify_august_and_november.py)
- **The IG-complex 52w-low rung as a gate on the August window.** A filter that
  does not filter and then subtracts: the tight three-way rung crossed with the
  August window leaves **2 qualifying days of 241**. Loosened to TLT alone within
  1% of its 52w low there IS precedent, unlike the JH sample, and the gate moves
  the parent the WRONG way, -0.233pp at h=10 (6 days, 2 years) and -0.744pp at
  h=5. Also raises the bar on watchlist entry W6: its own freshness leg fails at
  depth 5, and pooled depth>1 entries pay **-0.629% at a 37.3% hit over N=59**.
  (a2_c1b_ig_rung_depth.py)
- **SPY at a 52w high while TLT sits at a 52w low, both directions.** The short
  side at h=5 is 9-2 (sign p 0.0327) and is the only pulse in the cross-asset
  conditioner lane, but top-2 episodes (2018-10-03, 2021-02-24) are **96% of the
  +7.75pp total**, the sign INVERTS one horizon earlier at h=3 (-0.242% on 3-7),
  the threshold grid decays monotonically as the TLT rung loosens (+1.157 /
  +0.775 / +0.345 / +0.031), and de-concentrated it is +3.9 bps against a 3 bp
  round trip. The two legs are near-independent (corr 0.176), so this is a
  coincidence cell rather than a confirmed state. Yields-up as a conditioner on
  SPY-near-high separately does nothing: TNX 21d rank >= 70 moves the parent
  -0.059pp at h=5 and the threshold ladder is a flat plateau.
  (a4_c12_yields_up_spy_high.py, a4b, a4c)
- **A 5-day complex-wide energy thrust into a 52w high, long.** The RANK form's
  +0.715% is an intersection artifact — thrust alone is -0.313% (85 episodes),
  near-high without thrust -0.298% — clearing its local control by only
  +0.234pp, and tightening to rank >= 99 (today printed 100.0) flips it to
  -0.094%. The honest MAGNITUDE form is negative at **every horizon 1-10**
  (today is +3.49 ATR; the >= 2.5 ATR cell pays -0.064% at h=5 over 25
  episodes). Two structural notes worth more than the kill: today's joint state
  has **no precedent** (0 of 11 thrust episodes ever carried an XLE-USO 63d
  divergence >= 18pp against today's +18.85pp), and **the "inversion" framing was
  false** — STRATEGY_BOOK carries two LONG 52w-high breakout strategies with XLE
  in universe (52wh Breakout, Sector BO), so long XLE at a 52w high is a book
  construction at a different horizon, not the opposite of the book's reflex.
  Check which side the book is actually on before claiming an inversion.
  (p2_c5_energy_thrust_high.py, p2b_c5_round2_state_and_book.py)
- **SMH against QQQ on a 63d-rank extreme, and the "today is not in the sample"
  test that killed it.** The spread is beta for the third time: both legs
  positive at h=5 (SMH +1.493%, QQQ +1.694%), equal-dollar -0.202%,
  beta-neutral residual **-0.519%** at the measured 1.19, negative in 23 of 24
  cells of the cut grid, and tightening to today's 1.6 rank DEEPENS it to
  -1.584%. The reusable method: historical trigger days averaged SMH **-4.3%
  against its 200d while today sits +27.3% above it**, and only 7 of 117 trigger
  days ever sat above +15%, all seven inside the live cluster. Check whether the
  live state is inside the historical support before reading any conditional
  mean. (p1_c4_smh_qqq_rank63.py)
- **EWZ against EEM, closing the country-decoupling family for the third
  country.** Wrong-signed at today's own depth (z10 <= -1.68): outright -1.787%,
  beta-neutral residual **-0.806%** at beta 1.13 over 21 episodes, and the depth
  grid is monotone the WRONG way (-0.825% at -1.25 through -3.001% at -2.0). The
  gate costs rather than filters (+0.085% ungated against -0.804% gated). The
  premise was also false on inspection: **EEM's own 63d rank is 2.8, LOWER than
  EWZ's 12.7**, so "one market sold while the complex is firm" was a 5-day
  statement dressed as a regime. Verify both legs' state, not just the one the
  story is about. (p3_c6_ewz_eem_family.py)
- **The VIX/VIX3M ratio at its 1st percentile, re-confirmed as a lagging
  marker.** The placebo offset ladder reproduces the 2026-08-13 finding almost
  exactly: offset -10 pays **+5.433% at an 85% hit** and -8 pays +4.407% at 95%
  against the true anchor's +1.672% at 75%, true anchor **12 of 16**. The fade
  direction is decisively wrong-signed (long UVXY -2.792% on 8-20, sign p
  0.9937). Today's SVXY trailing-21d is +4.59% against the +9.16% trigger median
  — a milder marker, not a different state, so the cell is the same one.
  (v1_c3_termstructure.py, v1b, v1c, v1d)
- **The VIX-expiry and opex anchors are ONE anchor, not two.** They share 189 of
  307 days; opex minus vix_expiry is +2 td in 196 months but **-3 td in 96**,
  where "vix_expiry-2" is actually the session AFTER opex, and the exclusive days
  carry the entire apparent disagreement (^VIX h5 excess +1.578pp vs -4.141pp).
  Any grid crossing both is double-counting one calendar fact. The vehicle leg
  separately inverts once the 2018-02-28 SVXY break is respected (the grid had
  pooled 77 of 178 observations on the -1x instrument): post-break long SVXY is
  negative against its local control at every anchor and horizon, worst -0.740%
  on 24-34. And 92% of opex-4 anchors hold into a live V4_POSTOPEX_VOL window.
  (v2_c8_expiry_pair.py)
- **IWM at a 52w high into opex week.** The opex gate is an INVERTER: the state
  with opex pays -0.250% at h=10 against +0.373% for the same state without it.
  The IWM/SPY pair at h=4 has top-2 episodes at **103% of total**, a LOYO floor
  of +0.004% and a window placebo ranking 4 of 15. Two method notes: today's
  signal bar sat at **opex-5, not opex-4** (the live cell pays +0.049% at 0.61x
  cost), so check where the SIGNAL bar sits rather than where today sits; and
  `searchsorted` on an event date past the end of the price index silently mints
  a fake anchor, which is the 2026-08-11 trap firing in reverse. The
  flow_mechanics half is **unfalsifiable in this repo** — option_surface_history
  holds 1 row and option_positioning_history 90, all dated 2026-08-05.
  (v3_c9_iwm_high_opex.py, v3b, v3c)
- **XRT washed out into the big-box cluster, and the intact-trend inverter
  replicating outside insurance.** XRT's 5d washout alone pays +0.290% at h=5;
  adding the intact-63d gate — **the state live today** (5d rank 18.7, 63d rank
  85.7) — takes it to **-0.443%**, while the broken-trend half pays +0.483%.
  Same sign at h=3 and h=10. The anchor placebo ladder is now **8 for 8**: across
  90 big-box clusters since 2006 the true anchor ranks **17 of 19** offsets, true
  minus placebo -0.333pp. The tradeable 3-name basket clears the alphabetical
  placebo (+0.118pp) but pays 0.7-1.3x its 15 bp round trip and is negative
  inside the live state. (e1_c7_xrt_retail_cluster.py, e1b)
- **GDX's maximal 21d thrust, closed by the reference class.** The identical rule
  on 15 names gives a cross-name excess of +0.025pp with observed sd 0.665pp
  against a sampling SE of 0.686pp — **dispersion ratio 0.97**, so the entire
  spread is noise; GDX ranks 2 of 15 and permutation gives **P(max >= GDX's
  +1.171pp) = 0.582**, a below-median draw from the null. The parent's edge lives
  only in the [20,26)% band and today's +26.01% sits in the losing half (h=5
  -0.858% on 2-8); magnitude-only h=10 runs +0.535% / **-2.224%** / +1.982% at
  the >=20 / >=26 / >=30 cuts, the same threshold instability that killed the
  silver version. The drawdown conditioner inverts exactly as on SLV (worth
  -2.5pp at h=5 against the near-high cell), and the h=10 headline is **97% two
  episodes**, leaving +0.039%. Rank-trap check PASSED for once (+26.01% is the
  98.2nd pctile of full-history 21d moves), so the kill is real rather than
  definitional. (e2_c10_gdx_thrust.py, e2b, e2c, e2d)

## Method traps (2026-08-18, from a 12-candidate sweep that killed all 12)

- **The loud state is usually the poisoning conditioner, not the edge. Four
  independent cells died this way in one morning**, which is enough to call it a
  pattern rather than four coincidences. The thing that made each candidate
  interesting was the thing that killed it: the bond-vol spike cell works only
  when TLT is NOT at its 52w low, and TLT is at its low today (spike-and-at-low
  N=22, **-1.152% at h=10, t=-2.06**, against spike-and-not-at-low +0.166%); the
  fresh 52w low SUBTRACTS 0.27-0.36pp from the deep-drawdown state it sits
  inside; the extension fade inverts exactly at the live magnitude (>=50% above
  the 200d pays **-0.290% at h=3 and -0.976% at h=10** to the short, against a
  pooled cell that is merely flat); and the utilities-versus-duration divergence
  is worse than either of its two legs alone. The generalisation for future
  mornings: when a tape extreme is what put a candidate on the map, the FIRST
  round-2 probe should be the cell's behaviour AT that extreme, not its pooled
  mean. Selecting on an extreme and then quoting the population mean is the
  error, and it is subtler than the mid-cluster-entry trap because nothing about
  the sample looks wrong.
- **The distance-from-the-extreme GRADIENT, which generalises the "is today
  inside historical support" test from a binary to a slope.** Support checks so
  far have asked whether today's reading falls inside the trigger population
  (2026-08-17 SMH, 2026-08-13 TLT). Better: regress the forward return WITHIN
  the trigger set on how far the instrument sits from the extreme, and read the
  fitted value at today's reading. On the month-end TLT cell the slope is
  **+0.126pp per 1% off the 52w low (t=+2.18) while the unconditional gradient
  runs the OTHER way (-0.009, t=-2.55)**, so the effect is specific to the cell;
  the bucket ladder is monotone (within 0.5% of the low: -0.581% on 6; >3% off:
  **+1.129% at a 72.5% hit, t=3.29** on 40), and today sits at the **0.0
  percentile** of the trigger population's distance distribution. That is a kill
  built on the 40-day complement at t=3.29, not on the 6-day bucket, which is
  what makes it a substantive kill rather than a small-N one. Cheap, and it says
  something a binary in-or-out test cannot: how much of the historical mean
  today's reading is entitled to.
- **Price the search on the grid the checker ACTUALLY walked, then let the cell
  die on something else anyway.** The month-end TLT residual was recovered from
  the corpse of the divergence cell, so the 2026-08-07 anti-rescue rule applied.
  Charged for the 2,415-cell grid (3 horizons x 13 exit offsets x 13 gate rungs x
  5 vehicles, n>=15) under a rotation permutation that preserves calendar and
  autocorrelation, it ranks **59th by t** with **P(grid max t >= 2.82) = 0.90**.
  Worth recording that the familywise number alone was NOT decisive enough to
  ship a kill on: the grid's best occupant sits at familywise p=0.060, which is
  suggestive rather than clean. The adjudication only became decisive on the
  live-state gradient above. Two lessons: charge the search, and do not let the
  multiplicity number be the whole verdict in either direction.
- **A cell can pass every robustness test in the book and still be untradeable
  today.** The gated month-end TLT cell PASSED the bond-bull fossil test (all 59
  triggers are in the rising-yield half by construction; the live-side splits pay
  +0.791% and +0.944%), the month-of-year control (triggers spread across all 12
  months, month-demeaned +0.738% at t=2.49, ex-August +0.682%), the era split
  (pre-2018 +0.893 / 2018+ +0.799 / 2021+ +0.504, decay without a sign flip),
  concentration (top-2 episodes = **-1% of total**), and LOYO (+0.658%). It is
  the first cell in this registry to clear the fossil test that killed the August
  TLT seasonal. It still does not trade, because of where TLT sits today. File
  the passes as well as the failures: the ungated parent is parked, not buried.

## Cells swept and empty (2026-08-18)

- **A one-day spike in bond volatility, on rates AND on equities.** Distinct from
  the two existing ^MOVE entries, which are both about MOVE at a FLOOR. The spike
  (+8.70%, 96.7th pctile of daily moves since 2002; note the LEVEL was only the
  43.2nd pctile of full history, so quote both) adds **+0.094pp at h=5 over the
  all-days control** (158 episodes +0.175% against 5,869 days +0.080%, Welch
  t=+0.56) and the exhaustion mechanism is falsified by its own horizon profile
  (+0.026 / +0.032 / -0.050 / +0.095 / -0.111pp at h=1/2/3/5/10 — non-monotone
  with negative neighbours around the one positive cell). The equity leg is the
  cleaner kill: forward ^VIX after a MOVE spike runs BELOW its all-day baseline at
  every horizon (-0.32 to -1.22pp), so rate vol is followed by CALMER equity vol,
  which falsifies "rate vol leads equity vol" inside its own window; short SPY
  pays -0.413% at h=5 on 65-93 against an all-days short control of -0.249%.
  (a2_c2_move_spike_rates.py, a3_c3_move_spike_equity.py)
- **A volatility pop on a session where spot barely moved ("fear without
  damage").** The no-damage leg is an INVERTER, not a filter, which is the
  2026-08-14 insurance finding replicating on a vol construction: the joint cell
  loses to all three of its gates at every horizon past h=1 (h=5: joint +0.006%
  against small-down-days-alone +0.150%, VIX-pops-alone +0.282%, local +/-126td
  +0.207%) and its edge against all days is negative at h=2 through h=10. The
  plain VIX pop is the only thing with content and the spot condition removes it.
  The 12-cell threshold grid splits 5-7 then 7-5 by sign, so both opposite trades
  the cell could support are equally unsupported. The three-condition variant
  that adds calm tape is parked (see the watchlist) and is dead as an INCREMENT:
  +0.874% at h=10 looks strong only against zero, and against calm-tape-alone
  (+0.479%, 300 episodes) the Welch t of the difference is +1.09.
  (b4_c4_vix_pop_no_damage.py, b4b, b4c)
- **Miner-versus-metal ratio reversion after a maximal thrust.** A different
  object from the 2026-08-17 GDX outright kill, and it dies harder: the
  beta-weighted short-GDX/long-GLD trade is wrong-signed at ALL TEN horizons in
  all three vehicle forms, **-0.576% at h=5 over 51 episodes against the same
  vehicle's +0.154% all-days control** (edge -0.730pp, bootstrap P(mean<=0)
  0.913). Conditional on a maximal miner-over-metal thrust the miner keeps
  outperforming, so the operating-leverage overshoot story is falsified by its
  own sign. Beta-neutralisation is NOT what breaks it (equal-dollar -0.909% and
  outright -1.153% are worse). The momentum mirror is separately dead on
  definition fragility (the 90th-pctile neighbour flips the sign to +0.340%) and
  cost (ex-2008 19.8 bps against a 10.2 bps two-leg round trip = 1.9x). Method
  note: the map's "97.9th percentile" was a FULL-HISTORY percentile, i.e.
  lookahead; the PIT trailing-252d rank is the tradeable statistic.
  (b5_c5_gdx_gld_ratio.py)
- **Jackson Hole on US large caps, completing the sweep to five asset classes.**
  The anchor is now examined on rates, gold, FX, small caps and large caps, and
  is empty in all five. The ladder is 9-for-9: across 26 anchors the true offset
  ranks **8 of 16 at h=10 and 12 of 16 at h=5**, a plateau with no spike anywhere
  from -10 to +5. The unconditional August window beats the event outright
  (tdom 6-16 all years **+0.234% over 286 starts** against the anchor's +0.102%
  over 26), reproducing the 2026-08-13 rates finding on a different asset.
  Midterm years invert it to **-1.485% at h=10** on the same split that killed the
  IWM version, with 2002 and 2022 the two worst anchors overall. Treat the JH
  anchor as closed. (b9_c9_jh_us_large.py)
- **International leadership into a joint 52w high, which closes the OTHER
  direction of the country family.** Every prior member (EWZ twice, FXI, SMH/QQQ)
  was one market BREAKING inside an intact thrust; this is sustained leadership,
  and it fails on leg attribution instead of on sign. At the horizon where the
  beta-neutral residual peaks (+0.202% at h=5, 53 episodes) the EFA leg beats its
  own base by **+0.013pp** while the SPY leg contributes **-0.202pp**, so 94% of
  the spread is a short-SPY bet; at h=3 the attribution flips, so the two adjacent
  horizons disagree about which leg carries it. Tape over-selection lands for the
  fourth time: SPY is above its 200d on **96.7% of trigger days against a 75.9%
  base rate**, and the trade is net short that tape. EWJ's era pattern REVERSES
  EFA's, so two near-identical constructions have opposite era signs.
  (c6_intl_leadership_pair.py)
- **Fading parabolic extension above the 200-day, and the magnitude inversion
  that makes the book's leverage confinement load-bearing.** Across 197 liquid
  names as a reference class (4,701 declustered episodes) the short loses at every
  horizon past h=3, and it gets WORSE the more parabolic the name: at >=50% above
  the 200d it pays **-0.290% at h=3 and -0.976% at h=10** to the short. Only
  **47.4% of 192 names** have a positive short mean at h=5, so the single-name
  framing was drawing from a population where the trade does not exist. This is
  the second measurement (after 2026-08-13's IHI) showing that the book confining
  its overbought fades to LEVERAGED names is load-bearing rather than incidental:
  ATR Extended Gap Up caps extension at 50% above the 50d and needs a gap trigger,
  and both fades exclude leaders by design. Survivorship runs the other way for a
  short and cannot explain it, since both cells carry the identical bias.
  (c7_extension_xsec.py)
- **Capitulation at a fresh 52-week low far below the 200-day.** Gate attribution
  reverses the thesis: the fresh-low condition, which IS the claimed mechanism,
  SUBTRACTS **-0.266 / -0.280 / -0.363pp at h=3/5/10** from names equally far
  below their 200d that are NOT at a new low (2,260 episodes against 4,065).
  Everything positive comes from the drawdown state and the capitulation trigger
  takes away from it. Modern era is significantly negative (**h=5 2018+ -1.006%,
  t=-2.74**, against +0.568% pre-2018). The 4-name cut fails the alphabetical
  placebo (never separating by more than noise, both sides with negative medians
  and 44-45% hit rates). Useful by-product, since it was checked explicitly: this
  state is OUTSIDE every book dip-buy rather than a re-dress of one — trigger rows
  have a median 252d perf rank of **0.4**, only 4.29% clear the >=50 floor and
  **0.00%** are above the 200d, so the book's yearly-uptrend gates exclude it by
  construction. (c8_new52wlow_below200_xsec.py, c8b)
- **Utilities, seventh expression, and the first one that was genuinely new.**
  Short XLU on utility STRENGTH while duration prints a 52w low. Day overlap with
  the six dead expressions was computed rather than asserted (0 of 30 joint days
  for the z10 washout, 1 of 30 for the rank21 form), so this died on its own
  numbers. The joint state is worse than both singles at every horizon: edge over
  unconditional short-XLU drift is +0.177pp for XLU-strength alone, +0.155pp for
  TLT-at-low alone, **-0.509pp joint**. The resolution runs opposite to the
  thesis, with XLU beating SPY by **+1.656% over the next 5 sessions at an 86.7%
  long-side hit rate**. Do NOT take the inverted long: it is a post-hoc sign flip
  recovered from a kill report AND the shape of the already-dead long-XLU-vs-SPY
  expression. (d1_c10_xlu_strength_tlt_low.py)
- **The 63-day-rank laggard cross-section.** The alphabetical placebo is now
  3-for-3 as a killer: the four deepest laggards pay **+0.017% excess at h=5
  against +0.102% for the four alphabetically-first** qualifiers, and in the
  live shape (SPY above its 200d, few names firing) **+0.105% against +0.416%**,
  a selection premium of -0.311pp. The live slice — all firing names above their
  own 200d — is +0.018% excess at a **50.2% hit** over 7,194 name-episodes. It is
  also the book's dip-buy family at a longer lookback, with 35.8% of laggard
  name-days satisfying an OLV-style gate against a 13.6% base rate (2.6x
  enrichment). The 2026-08-13 denominator-roll warning generalises: the t-63
  roll-off dominates the day's own move on **37.3%** of trigger days, and 25.8% of
  trigger names are ABOVE their 200d, which today's slate was 4-for-4.
  (d2_c11_laggard_xsec.py, d2b)
- **Month-end rebalance flow sized by the stock/bond divergence.** The divergence
  gate adds **+0.04pp** to the ungated month-end TLT cell (+0.586% on 31 anchors
  against +0.540% on 288), Spearman between divergence size and forward return is
  **+0.049**, and the bucket ladder is non-monotone, so the "sized by the
  divergence" mechanism has no dose response. The SPY half is a WRONG-SIGNED
  conditioner (short-SPY leg -0.223% on the SPY-strength half against +0.665% on
  the TLT-weakness half), so a story about two-sided rebalancing flow was one
  instrument being oversold. The UNGATED month-end anchor underneath it is real
  and is parked (+0.540% at t=3.88 over 288 anchors, month-demeaned +0.391% at
  t=2.85, exit-offset ladder monotone from +0.540% at the month-end close to
  -0.229% ten sessions past it). Also a calendar correction worth keeping: from
  the 2026-08-18 entry close to the 2026-08-31 month-end close is **9** sessions,
  not 8, and the encoding matters — the headline halves between h=8 and h=9.
  (a1_c1_monthend_divergence.py, a1b through a1h, r1, r2, r3)
