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

## Method traps (2026-08-19, from a 10-candidate sweep that killed all 10)

- **The gate that IS the mechanism subtracts from the plain state underneath
  it. FOUR cells died this way in one morning, which is the second consecutive
  morning the pattern has repeated** (2026-08-18 called it "the loud state is
  usually the poisoning conditioner"; this is the same thing stated as a gate
  rather than as an extreme). The tally: the crude-at-a-63-day-floor gate takes
  XLE at a fresh 52-week high from **+0.606% to -1.465%**, and every threshold
  from 5 to 25 subtracts; the 63-day-rank momentum-turn gate takes a deep
  drawdown from **+0.470% at h=21 (t=6.95) to -0.282%**; the index-near-a-high
  gate takes a megacap washout down by **-0.306pp**; and the two gates defining
  the semis-break cell move its number by **0.004pp** (+0.227% gate-off against
  +0.231% gate-on). The generalisation is now strong enough to be a first move
  rather than a round-2 probe: **run the cell gate-off BEFORE running it
  gate-on.** If the parent is positive and the gate is negative, the candidate
  is the parent minus a subtraction, and no story about the gate is admissible.
- **Run the near-neighbour lookback ladder, not the far one.** Both macro cells
  were maximal at exactly the pitched 21 sessions with every SHORTER neighbour
  wrong-signed, and the near neighbours are where it showed: gold h=1 pays
  10d **-0.059** / 13d **-0.178** / 15d **-0.044** / 18d **-0.134** / **21d
  +0.457** / 25d +0.182 / 30d +0.252; the dollar h=5 pays 10d **-0.182** / 13d
  **-0.080** / 15d **-0.079** / 18d **-0.013** / **21d +0.234** / 25d +0.168.
  A 10/42/63 ladder would have shown decay and passed both. The 15-to-18-session
  sign flip on the gold cell is the same rule stated two defensible ways, with
  the wrong-signed side better populated (N=52 against 47).
- **A magnitude story owes a dose response, and both of today's macro cells had
  one running backwards.** The thesis in each was the SIZE of an unconfirmed
  repricing, so the mechanism's own dial is testable directly: gold pays
  **+0.432 / +0.384 / +0.225%** at yield-rise floors of +0.05 / +0.10 / +0.20pt,
  i.e. LESS as the thing that is supposed to drive it gets bigger. That is a
  falsification inside the window, not a power problem, and it is cheaper to run
  than any robustness test.
- **A rank gate buys an unknown magnitude, so quote the level the rank bought.**
  Today's yield rank of 68.7 buys **+0.108pt** of 21-session yield thrust, the
  **32.3rd percentile of the trigger distribution** - the rank sounds like a
  strong state and the level is a weak one. Same trap on the other tail: ADBE's
  63-day rank of 96.0 buys a 63-day return of **+2.9%** against a trigger-set
  median of +21.9%. Pair every rank gate with a magnitude floor before believing
  the state is what its name says.
- **The denominator-roll warning is WORSE on the high tail than the low one.**
  The t-63 roll-off dominates the day's own move on **61.8% of 19,274** trigger
  name-days on the 63-day-rank-HIGH cross-section, against 37.3% measured on the
  rank-LOW version on 2026-08-18. A 63-day rank is not a 63-day move on either
  tail, and it is least a move where it looks most like a breakout.
- **A full-sample percentile is lookahead and it can be the whole candidate.**
  This morning's headline state, the one-day XLV-minus-XLK gap, is the **99.3rd
  percentile of 6,729 sessions since 2000 and the 97.2nd of the trailing year**,
  and the point-in-time rank does not clear the >=99 threshold the cell was built
  on. Separately, `pitch_lab.pct_rank` computes the trailing rank of an n-day
  PERCENT CHANGE, so calling it on a spread series that crosses zero is
  meaningless; a first-pass C1 built that way returned a consistently negative
  cell that the corrected version reversed at h=3. The reversal is itself the
  finding: the cell's sign is a function of how "99th percentile" is defined.
- **`close_panel` returns a UNION calendar, and `rolling(...).max()` silently
  poisons every window containing a foreign-calendar NaN.** CL=F trades some
  equity holidays, so the XLE column carried NaN rows and a 252-day rolling max
  returned NaN for any window touching one - it reported "XLE is not at a
  52-week high" on the session XLE closed exactly at one, and cut the
  52-week-high day count from **419 to 342**. Compute rolling statistics on the
  `.dropna()` series and reindex back. `pitch_lab.rolling_on_valid` now does
  this; the `pct_rank` callers were already safe by accident.

## Cells swept and empty (2026-08-19)

- **The maximal one-day sector rotation, in all four expressions.** A 4.07pp
  one-day XLV-minus-XLK gap is the largest fresh state on the tape and it is
  empty in every direction. CONTINUATION (long XLV / short XLK) dies on leg
  attribution: at its one positive horizon the long leg's excess over its own
  drift is **+0.007pp** against the short leg's **-0.387pp**, so the pair is a
  short-tech bet and 76% of that is short-beta, with the sign flipping
  -/+/-/- across h=1/3/5/10. SNAP-BACK (long XLK / short XLV) is the positive
  sign of the cell and dies on concentration: in the subclass matching today's
  calm near-high tape the **top 2 episodes are 96% of the h=3 total and both
  are 2026 prints**, ex-2026 is 7-7 at -0.599% and negative at 5 of 6 horizons,
  the by-year table runs -1.33 (2024) / -1.08 (2025) / +3.86 (2026), and the
  apparent midterm conditioner is that same cluster relabelled at 7 of 8
  episodes. Beta-neutralising collapses h=1 from +1.187% to +0.180%. The INDEX
  read dies on reference class: the pooled +0.498% at h=5 is a drawdown bounce
  whose trigger days sit a median **5.96% below the 52-week high** against
  today's 1.34%, and the near-high subclass excess is +0.016pp. Two things
  worth keeping: the naked long XLK BEATS the pair at h=5 and h=7, so if
  anything ever survives here it is not a pair trade; and the rotation trigger
  did beat an ignorant "any big tech-down day in calm near-high tape" placebo
  (-0.066 / +0.036 / -0.164 / +0.321 at h=1/2/3/5), which is the one test it
  passed. Parked on the watchlist with an arithmetic turn-on.
  (a1b_c1c2_fixed.py, a2_c3_round1.py, a4_round2_livedef.py,
  a5_c2_subclass_dev.py, a5b_c2_dropbest.py, a6_c3_nearhigh_concentration.py)
- **A megacap-growth complex breaking hard while the index holds.** The
  candidate is its two gates and they are worth **0.004pp**: break-only, with
  no index-holds and no falls-less-than-half condition, pays **+0.227% at h=5
  against the three-condition cell's +0.231%**. The best cell in the whole grid
  carries a Welch t of **+0.33** and inverts by era (pre-2018 -0.162%, 2018+
  +0.805%), and the holds gate is a dial that produces any answer asked of it
  (-1.0% gives +0.228% at h=3, -2.5% gives -0.059%). (a3_c5_round1.py)
- **Short the semis complex into the August NVDA print, i.e. the inverted sign
  of the 2026-08-14 long kill. The anti-rescue rule holds, and this is what it
  looks like when it is enforced rather than asserted.** Three independent
  kills. The placebo offset ladder puts the true anchor **14 of 16** at -0.134%
  against a ladder mean of +1.083% (best is k=+4 at +2.963%), so the print is
  decoration on late-August month position and the ladder is now **10-for-10**
  in this registry. The all-months version of the identical gated state pays
  the OPPOSITE sign over 25 episodes (**short -2.483%, t=-1.75, 7-18, sign p
  0.9927**), which says the 14-observation August slice is the anomaly and the
  08-14 long sign was right. And the gated cell's last instance is
  **2017-08-02**, so the 2020+ era in which NVDA drives the complex - the
  mechanism's own precondition - contributes **zero** observations. Minimum
  search family is 4 print months x 2 eras x 2 signs = 16, Bonferroni 0.003125,
  which no achievable record at N<=7 reaches. Replication note: the ungated
  August long reproduces at -0.030% here against the -0.322% quoted on 08-14,
  so the whole August-NVDA family is sensitive to the entry-lead definition.
  (b1_c4_semis_short.py, b1b_c4_placebo_ladder.py)
- **Long energy at a fresh 52-week high with crude at a 63-day floor, the
  outright form of the spread killed on 2026-08-14.** The gate is an inverter:
  plain XLE at a fresh 52-week high pays +0.606% over 70 episodes (only
  +0.135pp over the all-days baseline, so the momentum state is thin to begin
  with) and adding the crude floor takes it to **-1.465%**, with thresholds 5
  through 25 subtracting **-5.691 / -2.190 / -2.071 / -1.782 / -1.540pp**. The
  only bucket with a signature is crude **LEADING** at +1.551% and sign p 0.006,
  the opposite state. Across 19 energy vehicles the gate adds for **2**, and
  peer worst-episode windows run to -30.09%. Vehicle correction that matters
  book-wide: **CL=F and USO agree on a "63-day rank <= 15" gate on 95.7% of
  4,805 shared days and today sits in the disagreeing 4.3%** - CL=F reads 21.8
  against USO's 6.0, and the gap is USO roll decay, which is exactly what the
  barrel-at-a-floor state is supposed to mean. Read the front month before
  believing an ETF's 63-day rank. (b2_c9_xle_high_crude_floor.py, b2b)
- **A megacap at a 21-day return rank <= 5 while the index holds within 2% of
  its 52-week high.** The alphabetical placebo is now **5-for-5**: on the same
  868 trigger dates the four most-washed names pay **+0.122% market-relative at
  h=10 against +0.342% for the four alphabetically-first** with no rank
  condition, at a relative hit rate of 48.4%. The conditional name underperforms
  its own drift by -0.216pp, the index-near-a-high gate subtracts -0.306pp
  (rank<=5 while SPY is NOT near a high pays +0.774%), and the washout gradient
  is backwards across eight buckets with the deepest the weakest (+0.416%
  against +0.559% for the shallowest). Useful by-product: the state's historical
  population is low-beta defensives (MRK 27, TJX 26, MO 25, VZ 24, PG 23), not
  leaders breaking, so the cell does not contain the thing the thesis describes.
  (b3_c10_megacap_washout.py)
- **The rates-versus-dollar divergence, on the dollar and on gold. The first
  cross-asset macro cell in this registry to PASS the bond-bull fossil test and
  still die.** Both are magnitude stories with backwards dose responses (see
  method traps) sitting on a 21-session knife edge, and both are search-priced:
  rotation permutation over the 168-mask grid actually walked gives **P(grid max
  t >= 1.73) = 0.810** for the dollar and **P(grid max t >= 2.06) = 0.937** for
  gold. Three passes worth filing so the cell is parked rather than buried: the
  secular rising-yield half, which is today's, pays **+0.498% at a 70.6% hit**
  against +0.119% falling, so this is not a fossil; gate attribution is clean
  both ways on the gold cell at h=1 and h=3 (joint +0.570 against TNX-alone
  +0.216 and DX-alone -0.124); and day-1-of-run entries, which is today, pay
  +0.543% (t=2.89) against +0.077% for days 2-5 and -0.664% at day 16+, so the
  mid-cluster trap does not apply. **UUP-versus-DXY-spot is a COST problem and
  not a signal problem**: matched episodes differ by 1.3 bps (t=0.55) with 95.5%
  sign agreement and an all-days structural gap of 1.4 bps per 5td, so the
  standing "UUP is dead" entry stands on drag alone and must not also claim the
  vehicles disagree about the effect. Premise correction the map owed: "gold is
  hot" was a 21-day read - GLD's 63-day rank is 30.6, it closed BELOW its 200d
  and it is -19.6% off its 252-day high, i.e. a bounce inside a drawdown.
  (c6_round1.py, c6b, c6c, c6e_lookback_fullpanel.py, c7_round1.py, c7b, c7c,
  c7d_midcluster.py, c7f_h1_and_reopen.py, c67e_lookback_fine.py)
- **The base-breakout cross-section: a 63-day rank >= 95 while still 15% or more
  below the 52-week high.** Distinct from the 2026-08-18 laggard kill (that was
  rank LOW) and it dies on the mirror-image defect. Deep drawdown ALONE is
  positive at every horizon (+0.038 / +0.079 / +0.092 / +0.132 / +0.263 /
  **+0.470% at h=21, t=6.95**); adding the momentum-turn gate makes it negative
  at every one (**-0.076 / -0.132 / -0.160 / -0.080 / -0.176 / -0.282%**).
  Everything positive belongs to the drawdown and the signal removes it. The
  alphabetical placebo separates by +0.001 to +0.148pp with BOTH sides negative
  (h=5: deepest-4 -0.365%, alphabetical-4 -0.411%, all qualifiers -0.398% at
  t=-2.99), reference-class permutation gives **P(max name mean >= observed) =
  1.000** across 182 names, and today's live staples/food slate is the worst cut
  available at **-0.938% excess at h=5 over 148 episodes, 39.2% hit, sign p
  0.997**. Book-overlap by-product: the book touches this state at 2.38x
  enrichment and **72 of those 73 trades are Overbot Vol Spike, a SHORT** -
  where the book meets a deep-drawdown name thrusting, it fades it.
  (c8_round1.py, c8b_horizons_liveshape.py, c8c_book_overlap.py)

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

## Method traps (2026-08-21, from an 11-candidate sweep that killed all 11)

- **A watchlist entry can fire every condition it states and still be dead, on
  a state the entry never encoded.** The GLD miner-thrust cell was parked on
  2026-08-11 with three arming conditions and all three fired for the first
  time on 2026-08-21. The parent reproduced cleanly (75 episodes, +0.853% at
  h=5, excess +0.619pp, 48-27, sign p 0.0101, 28x cost, decluster-stable at
  min_gap 5/10/21/42) and the idea died anyway, because the entry had no rung
  for the TREND state of the instrument: 2018+ with GLD more than 10% below its
  52-week high pays **-0.641% over 10 episodes at a 50% hit against +0.844% at
  72% for the complement**, and the live reading was -16.26%. The rule this
  yields is general: before trusting any trigger, check where today's reading
  sits inside the episode support on the axes the trigger does NOT mention, not
  just the ones it does. A trigger is a claim about the conditions it names and
  says nothing about the ones it omits. (a1b_c1_gld_teardown.py)
- **Check the dose response on the conditioner's own axis, because it can run
  backwards while the cell still looks alive.** The same entry's thesis was
  "pays more the less the metal has joined". The gradient says otherwise: GLD 5d
  rank [85,95), which is where today sat at 86.9, pays **+0.892%** while [70,85)
  pays +0.585% and [50,70) +0.702%, and GDX one-week moves above +10% pay
  +0.483% at a 51.6% hit against +1.113% at 72.7% below that. So the cell is
  real and the stated mechanism is not what it keys on, which is the third
  occurrence of this pattern after the 2026-08-19 macro pair. A conditioner
  with no monotone dose response has not been shown to be the mechanism.
  (a1b_c1_gld_teardown.py)
- **Two anchors can be the same anchor.** The August opex close IS the
  Jackson-Hole-minus-5 anchor in **21 of 26 years** (offset -5 in 21, -10 in 5,
  0 once). Two candidate families that looked independent, the cross-asset
  post-opex sweep and JH-5 on the unswept classes, were one anchor family
  wearing two labels, and both labels were already closed. Before crossing a
  calendar anchor with anything, compute its offset distribution against every
  OTHER anchor in the window; a fixed-date event and a nth-weekday event will
  collide on a stable offset far more often than intuition suggests.
  (b1d_c3_impulse_state_collision.py)
- **Score a win record against the instrument's OWN conditional up-rate, not
  against a coin.** HYG at JH-5 is 17-2, which is p=0.0004 under a fair coin and
  looks decisive. Against HYG's own August-trading-day-of-month up-rate of
  **75.2%** the same record is **p = 0.1147**, and no vehicle in the six tested
  cleared 0.13 on the right null. This is the 2026-08-11 "sign test against a
  coin is the wrong null for a drifting instrument" entry, and the correct base
  rate is the CONDITIONAL one for the month and calendar position, not the
  instrument's all-days rate. (b2c_c9_hyg_signtest_basis.py)
- **The alphabetical placebo is now 7-for-8, and the one miss has been
  reversed.** 2026-08-20 recorded the placebo failing to kill on the SHORT side
  of the bank-breadth cell. Tested on the LONG side the next session, the four
  most-washed names **lose to the four alphabetically-first at every horizon
  from 1 to 10** (-0.117pp at h=5), and both the full 11-name basket (+0.826%)
  and the four STRONGEST names (+0.872%) also beat the selection rule. A
  placebo result on one side of a cell does not transfer to the other side.
  (c7_banks_broken_long.py)
- **A cell that peaks at the edge of the scanned horizon grid is exposure, not
  an impulse.** Every occupant of the 100-cell cross-asset opex grid peaked at
  h=10, the last horizon scanned, and the days 6-10 leg still carried excess
  (silver's August leg-1 +1.763pp, leg-2 +1.282pp). An expiry-flow mechanism
  predicts a front-loaded impulse that decays. When the profile instead rises to
  the boundary, the cell is measuring a calendar window the horizon happens to
  span, and extending the grid will move the "optimum" with it.
  (b1_c3_postopex_crossasset_round1.py, b1d_c3_impulse_state_collision.py)
- **Surviving concentration, era, tape-over-selection, beta-neutralisation and
  the obvious confound is still not enough for a single-name or single-country
  idea.** The Japan washout passed every one of those: 42 episodes, +1.564% at
  h=5, excess +1.460pp, 30-12, sign p 0.0040, 52x cost, era-stable both sides of
  2018, top-2 episodes are LOSERS so concentration runs the right way, SPY above
  its 200d on 66.7% of trigger episodes against a 71.6% base rate, EFA-hedged
  residual +0.674% at a 71.4% hit with a LOYO floor of +0.433%, and a daily
  EWJ/yen correlation of +0.020 that rules out the currency. The reference-class
  permutation is a separate and stricter test and it is what killed it. Run it
  BEFORE spending a round-3 development pass, not after.
  (a4c_c11_class_null_ownvol.py)

## Cells swept and empty (2026-08-21)

- **The opex anchor crossed with every non-equity class, which closes the
  anchor completely.** The US equity side died in both directions on
  2026-08-07 and 2026-08-20; this is the rest of the market. Ten vehicles (GLD,
  SLV, TLT, IEF, HYG, LQD, USO, XLE, UUP, FXI) by ten horizons, entry MOC on the
  opex close, excess against each vehicle's own trading-day-of-month-matched
  control: **grid excess sd 0.132pp and 0 of 100 cells clear 1.0pp**. Priced
  against cost, credit is +6.0 bps on HYG (1.5x) and +0.2 bps on LQD (0.1x),
  duration is wrong-signed at -0.105pp on TLT (-5.5x) and -0.038pp on IEF
  (-1.3x), and the dollar is -0.050pp (-0.8x). The four August subcells that
  looked alive each died on their own kill: **silver** (+3.661% over 20 anchors,
  ladder rank 1 of 17) gives back 73% of it to the adjacent plus-or-minus 2 and
  3 sessions, is 12-8 at sign p 0.2517 once GLD-hedged at beta 1.45, splits
  +7.858% pre-2013 (7-0) against +1.401% after with a drop-two of +0.363% versus
  an August all-sessions base of +0.584%, and its live-state reading of -41.6%
  from the 252d high is deeper than all 20 anchors (deepest -35.8%) in a deep
  half paying +1.77% against the shallow half's +5.55%; **XLE** is the parked
  2026-08-20 crude/JH-6 entry one session over and that entry's own condition
  blocks it with XLE at its 52-week high; **crude** has a drop-two of -0.512%;
  **China** shows the 2026-08-20 IWM signature, a gate worth +0.220pp of
  +1.746% and a ladder ranking 2 of 17 at h=10 but 9 of 17 at h=5.
  (b1_c3_postopex_crossasset_round1.py, b1b_c3_tdom_month_ladder.py,
  b1c_c3_slv_teardown.py, b1d_c3_impulse_state_collision.py)
- **Jackson Hole on credit and international, which completes that anchor to
  seven classes.** Six vehicles by ten horizons. Decisive kill is leg
  attribution: over the same 19 anchors HYG's August-tdom excess is +0.345pp at
  h=10 against **SPY's +0.700pp**, so credit SUBTRACTS 0.355pp from an equity
  leg the registry closed on 2026-08-18, and every SPY-beta-hedged residual is
  inside plus-or-minus 0.12pp (HYG +0.082, EEM +0.116, EFA +0.069, LQD -0.049,
  EWJ -0.171) except FXI's -0.738pp. The mechanism is falsified inside its own
  window: a premium-build story predicts a negative pre-speech leg and the
  h<=4 class mean is **+0.010pp**, with the ladder paying about the same
  entering 12 sessions early or 3 sessions AFTER the speech, so there is no
  release either. Midterm years are negative in 4 of 6 vehicles at h=10, a sixth
  independent reproduction of the JH midterm inversion. Placebo permutation over
  relocated anchors: P(max-of-6 >= observed) = 0.286 at h=5, 0.357 at h=10.
  Self-correction filed with it: the round-1 control pooled August AND September
  tdom-matched sessions, September dragged the control down and inflated every
  excess (HYG h=10 read +0.973pp against +0.345pp on the correct August-only
  control) - a month-matched control must match the month, singular.
  (b2_c9_jh5_credit_intl_round1.py, b2b_c9_jh5_round2.py)
- **An equity dip with credit refusing to confirm it.** The credit gate is worth
  **-0.022pp**: SPY 5d rank <= 10 alone pays +0.455% over 175 episodes and
  adding HYG within 0.5% of its 52-week high leaves +0.433% over 17, i.e. it
  discards 158 of 175 episodes to subtract 2.2 bps, while the complement pays
  +0.462%. **100.0% of gated trigger days sit above SPY's 200d against a 75.4%
  base rate**, so the gate is a bull-tape selector. Era +2.520% pre-2018 to
  -0.437% after. Conditioned on the book's own sizing statistic the cell pays
  +0.408% below dial 50 and **-1.511% above it with zero precedents above 70**,
  which independently reproduces the frag_risk_bands finding on a cell the book
  does not trade. Tightening the tolerance 0.25pp gives -0.715%. The one
  neighbour that worked, HYG's own 5d return >= -0.5% (+1.122%, 70 episodes,
  t 2.625, era-stable), dies to the asset-class reference class: run "SPY fell
  but X did not" across 14 vehicles and 9 of 14 are positive with a mean gate
  value of +0.214pp, HYG ranking **7 of 14 at h=3**, with IEF +0.317, LQD +0.348,
  XLU +0.648 and XLK +0.663 doing the same work. Duration wearing a credit
  label, for the third time. (c2_credit_unconfirmed_washout.py,
  c2b_dial_and_hygret.py, c2c_credit_reference_class.py)
- **The industry-breadth washout with the trend BROKEN, which closes the
  construction the 2026-08-14 insurance cell opened.** See the alphabetical
  placebo entry above for the deciding number. Also: the gate does not filter at
  the live reading (median-63d threshold walk gives +0.633pp excess below 50,
  +0.520 below 55, +0.517 below 60, +0.449 below 65, +0.344 below 70, and
  **+0.348pp with NO GATE AT ALL**, so the 70 line is worth -0.004pp); era runs
  +1.298% pre-2018 over 132 episodes to **-0.475pp excess at a 46.4% hit** after,
  with 2008 and 2009 carrying +92.3pp of a +150.6pp total; the basket correlates
  **0.926** with XLF and beats it by only +0.129pp against 8-10 bps of extra
  round trip, so the four-name form is XLF with tracking error at 1.9x cost
  full-history and 0x in the modern era; and tape over-selection runs the wrong
  way, 46.0% of trigger days below SPY's 200d against a 28.4% base rate.
  Reference class, third independent sample: Cochran Q 5.88 on 10 df,
  I-squared 0.0%, dispersion ratio 0.70, P(max group excess >= banks) = 0.891 at
  h=5 and 0.997 at h=10. (c7_banks_broken_long.py)
- **The beat that gets sold, i.e. post-earnings drift conditioned on an adverse
  reaction to a positive surprise.** First examination of the earnings-surprise
  columns in this repo: 78,672 events, 943 tickers, 2000-2026. Convention
  established rather than assumed - the parquet date is the announcement date
  and the reaction splits BMO/AMC almost evenly (|ret| over own 63d median is
  1.720 at offset 0 and 1.812 at offset +1, against ~1.0 at -1 and +2), so the
  reaction day is classified per event. **The adverse-move condition does no
  work**: against the matched cohort that beat and did NOT sell off, the gate
  adds +0.018pp at h=5 and subtracts -0.159pp at h=10, so continuation is
  wrong-signed (the cell is positive from h=2) and the snapback is smaller than
  doing nothing. On LIQUID_PLUS_COMMODITIES from 2013 it is **+0.015% over 601
  events at a clustered t of 0.12, 1.5 bps against a 10 bps single-name round
  trip**, with negative medians at h=1/3/5 and a sub-50% hit rate - a left-tail
  description a time exit cannot harvest. The signal lives in the illiquid names
  (+0.193%, clustered t 2.36), which is untradeable here. The percent and ATR
  parameterisations of the same gate disagree in SIGN (percent improves as the
  drop deepens, ATR goes negative at h=10 for -2/-3/-4 ATR), and 2022 and 2020
  carry 70% of the h=5 total while 2024 and 2025 are both negative.
  (c10_recon_convention.py, c10_beat_and_sold.py, c10b_liquid_teardown.py)
- **Silver against gold on a drawdown divergence, closing the metals-pair family
  for the third time.** The pitched cell is nothing: 45 episodes, **-0.014%
  equal-dollar and -0.019% beta-neutral, Welch -0.19, -0.2x cost**. Beta of SLV
  on GLD is 1.447 full-sample and 1.787 trailing-252d, so equal-dollar is a
  levered silver bet. The conditioner is U-shaped and points the wrong way: the
  state where silver is LESS deep than gold pays best at +0.601% while today's
  bucket pays -0.166%, and the intersection is worse than either gate alone
  (joint thrust -0.099%, gap <= -20 alone +0.270%). The single positive rung
  (<= -25pp, N=20, +1.006%, sign p 0.021) drops to +0.237% on a 1.99pp
  loosening, with the live reading 0.35pp inside it and 16 of 20 episodes in
  2008-2012 leaving two independent post-2013 instances.
  (a2_c8_slv_gld_drawdown.py, a2b_c8_gap25_probe.py, a2c_c8_era_and_fragility.py)
- **Japan, the fifth and last member of the country-decoupling family.** Prior
  members broke on decoupling (EWZ twice, FXI, SMH/QQQ), on sustained leadership
  (EFA) or on the funding side (KWEB). This one broke on nothing until the
  reference class: **P(max-of-10 >= EWJ) = 0.477 on excess and 0.620 on the
  beta-neutral residual**, where EWJ's +0.671% sits BELOW the null's median best
  draw of +0.806%, over 20,000 permutations preserving each name's own
  dispersion under an imposed common class mean, robust to dropping the two
  wildest peers (0.472 / 0.637). The rule is positive on 8 of the 10 peers with
  a median excess of +0.698%, so it is a class-wide effect and the country is
  simply the top draw of ten correlated names selected for being today's
  outlier. The decoupling leg that is the entire thesis adds +0.760pp over
  washout-alone at Welch t +1.24, and EWJ 5d rank <= 5 with no EFA gate at all
  already pays +0.516% excess over 154 episodes at sign p 0.0023. Treat the
  family as closed: a new country instance needs P(max-of-K) below 0.05 on the
  residual before it is worth a check.
  (a4_c11_ewj_washout.py, a4b_c11_refclass_null.py, a4c_c11_class_null_ownvol.py)

### Book findings, filed here because they are about the book rather than a pitch

- **`52wh Breakout` is substantially an earnings-reaction strategy and nothing
  documents it.** Found incidentally while measuring book overlap for the
  earnings candidate: **148 of its 250 ledger signals (59.2%) fall within one
  session of an earnings print**, avgR +0.494. By contrast Overbot Vol Spike's
  +/-10 trading-day blackout is real and airtight - 0 of 2,305 OVS signals fall
  within +/-10 td of a print, minimum |offset| 11. Not a defect and not a
  recommendation, but the 52wh Breakout entry rule is selecting earnings gaps
  far more often than its description implies, which matters for anyone sizing
  it or reasoning about its tail. (c10b_liquid_teardown.py, c0b_book_overlap.py)
- **The pitch state's book-overlap block was blind and is now fixed.**
  `scripts/build_pitch_state.py` read `Ticker` and `Strategy_Name` off the
  staging tabs, whose columns are `Symbol` and `Strategy_Ref`, so every staged
  row reached stage C with a null ticker and no morning could see WHICH names
  the book had staged. Flagged without a fix on 2026-08-17, fixed 2026-08-21.
  It mattered immediately: the book was staged short four gold miners (NEM, AGI,
  AU, CGAU) over exactly the window a live candidate would have held long gold,
  and the measured consequence was that adding the pitch to the book's own
  position cancels 82% of it.

## Method traps (2026-08-24, from a 10-candidate sweep that killed all 10)

- **Three candidates died because their PREMISE was false, not because the data
  was weak, and all three were checkable in one line at recon.** This is the
  cheapest class of kill in this registry and it went 3-for-10 in one morning,
  so it belongs at the top of stage C rather than inside round 2. (1) *Copper*:
  the candidate was "long copper on a five-day thrust to a fresh 52-week high"
  and **copper did not move** — FCX ran +15.30% while HG=F was **-0.30% and
  -1.84% off its own 52-week high**, XME -10.09% off, daily correlation 0.494.
  (2) *Breadth*: "20 of 218 names at a 52-week high" is **9.17%, the 54.2nd
  point-in-time percentile of its own trailing year**, below that year's 9.46%
  mean — an extreme in the prose, a median reading in the data, and the
  full-sample percentile (61.2) flattered it by +7.1pt. (3) *Yield level*: the
  "^TNX at a 52-week high" LEVEL trigger was selected precisely because it was
  NOT the killed return-rank trigger, and the two masks **coincide on 91% of
  days**, so it inherited the dead cell's search charge. The general rule:
  before any battery, print the thing the candidate is NAMED after and confirm
  it is doing what the name says. Name the underlying, the PIT percentile, and
  the overlap with the corpse you think you are avoiding.
  (b1_c3_copper_thrust_r1.py, c1_c6_newhigh_breadth.py, d2_c5_tnx_level_high.py)
- **A count-of-names breadth trigger owes an EFFECTIVE-N measurement, and
  "five names" can be one factor.** The energy cluster looked like breadth and
  was not: complex mean pairwise correlation 0.700, PC1 **73.5%** of variance,
  participation ratio **1.82 effective names of 11**; within today's five, PC1
  83.3% and **1.42 effective of 5**. The decisive consequence is conditional
  rather than descriptive: **P(XLE thrusting | 5 of 11 thrusting) = 0.920**, so
  92 of 100 trigger days were days the sector ETF itself was thrusting, and the
  count cost **-0.94pp** against the single-instrument version. Two of the five
  "names" were ETFs holding the other three. Compute PC1 share, the
  participation ratio, and P(flagship fires | cluster fires) before treating a
  count as breadth. (b2c_c7_refclass_fixed_and_factor.py)
- **A count gradient can be monotone in the wrong direction, and the pitched
  threshold can sit on the far side of the crossing.** Energy h=5 by count: 1
  +0.185%, 2 +0.275%, **3 +0.490%**, 4 +0.086%, **5 -0.779%**, 6 -0.354%, 8
  -0.470%. Episode level, count>=3 +0.393% against count>=5 -0.573%. The same
  shape appeared independently on new-high breadth the same morning (0 of 9
  sectors at a high +0.317%, 2 of 9 +0.310%, **3 of 9 -0.009%**, >=5 -0.136%),
  which is two unrelated constructions agreeing that a LITTLE of a thrust state
  is bullish and a LOT is not. Walk the count ladder in both directions before
  choosing k; the interesting cell is usually not the extreme one.
  (b2_c7_energy_z10_cluster_r1.py, c2_c6_breadth_attribution.py)
- **A cell can pass every robustness test and die because its MECHANISM decayed
  while its total return did not.** The month-end TLT parent is the strongest
  duration cell in this repo and it survived re-derivation: month-matched
  +0.346% at t=3.72 over 288 anchors, a clean exit-placebo **SPIKE** (ME+3
  +0.065 / ME+0 **+0.430** / ME-3 +0.205 / ME-9 +0.067) rather than the plateau
  that killed the Jackson Hole and August anchors, holdout 2014-2026 **+0.463%
  at t=3.56** which beats its own in-sample half, top-2 episodes 8% of total,
  and it passes the bond-bull fossil test in the modern era. It still does not
  trade, because the index-extension story predicts the LAST sessions carry the
  excess and they have stopped: TLT ME-1 -> ME-0 ran **+25.65 bp at t=3.09 and
  a 64.3% hit in 2002-2012** and pays **+3.99 bp at t=0.37 and a 48.1% hit
  against a 49.3% base rate in 2020-2026**, replicated on AGG (+10.87 -> +3.77),
  LQD (+23.38 -> +3.56) and IEF (+13.19 -> +8.19), with rolling 8-year t falling
  monotonically 3.02 -> 1.05. The five-session total holds up only because **a
  different session carries it in every era** (ME-1+ME-2 = 184% of the excess
  in 2002-2012, ME-3 in 2013-2019, ME-5+ME-2 after 2020). Decompose a
  multi-session window BY SESSION and check that the session the mechanism
  names is the one that pays; a window that sums positive out of a moving part
  is not an effect.
  (a1d_c1_tlt_mechanism_vs_modern.py, a1e_c1_decay_robustness.py)
- **The same test applied to the equity version of a flow story kills it
  outright, and it is one line.** SPY's ME-1 -> ME-0 session pays **-0.006% at
  a 47.6% hit** (post-2013 -0.040% at 45.4%) against an all-days +0.039% at
  54.5%, and it is negative on every equity vehicle (SPY -4.50 bp, IWM -11.76,
  QQQ -4.15, DIA -9.73) while TLT's is +16.61 bp at t +3.68. So **100% of the
  ME-5 window's equity return sits OUTSIDE the flow session it is named after**,
  and 88% of it is one scanned session the anchor does not name (ME-4 -> ME-3,
  +14.14 bp at t 2.45, 3 of 16 scanned sessions clearing |t|>=2 against 0.8
  expected). This also settles the turn-of-month overlap the registry asserts
  from memory: the classic last-1 leg IS dead post-2013, measured.
  (a2_c2_spy_me5.py, a2b_c2_spy_me5_round2.py)
- **A parked watchlist blocker is a claim and it can be measured on the wrong
  object. CORRECTION OWED TO W12.** Its stated turn-on is "forward return
  regresses POSITIVELY on distance from the 52w low at +0.126pp per 1% off
  (t=+2.18)". Re-derived on the UNGATED parent the entry actually parks, the
  slope is **-0.0082 (t=-0.51)** at ME-9 and **-0.0064 (t=-0.60)** at ME-5. The
  positive gradient exists only on the oversold-GATED subsample (TLT 21d
  <= -2.5%, N=50) that the entry's own second debt says to keep OFF, and that
  gate does not fire today (TLT 21d = -0.95%). The blocker was true of a cell
  the entry does not park. Today's reading is uninformative rather than adverse:
  N=7 comparables, 3-4, mean +0.533% but **median -0.105%**, and dropping the
  single best leaves -0.045% on 6. When a park cites a gradient, record WHICH
  subsample it was fitted on.
  (a1_c1_tlt_me_entry_ladder.py, a1b_c1_tlt_me5_round2.py)
- **`pitch_lab.sign_test` raised OverflowError on the `p != 0.5` branch above
  n ~ 1100, and that is the branch the doctrine sends you to.** The 2026-08-09
  fix moved the p=0.5 path to exact `Fraction` arithmetic and left the other
  one on floats, where `comb(n, k)` exceeds float range. Scoring a hit rate
  against the instrument's OWN conditional up-rate is the documented way to
  test a drifting asset (2026-08-11, -08-21), and the control cells it gets
  applied to run to thousands of days, so the crash fires on exactly the
  correct usage. Fixed 2026-08-24 by summing in LOG space on that branch
  (stable to any n, agrees with the exact form to 2e-13 relative, p=0.5 still
  exact); guard `tests/test_pitch_lab.py::test_sign_test_base_rate_survives_big_n`
  pins n=20000 plus the registry's own published HYG-at-JH-5 pair (17-2 is
  p=0.0004 against a coin and p=0.115 against a 75.2% base rate).

## Cells swept and empty (2026-08-24)

- **The month-end anchor on EQUITIES, first measurement in this repo.** See the
  session-decomposition entry above for the kill. Additionally: month-matching
  takes the raw +0.352% (t 2.647) to **+0.164% at t 1.25**, and dropping
  November (the only month with t>2) to +0.065% at t 0.53; the live August x
  midterm cell is **3-3 at -0.860%** (2002 -2.98, 2022 -4.47) against
  non-midterm August +0.707%; 27% of the total is two 2008 episodes; and the
  60-cell grid walked (15 offsets x 4 vehicles) gives Sidak familywise **0.877**
  with SPY ME-5 ranking 14th. The IWM rescue fails the same way (raw +0.513% at
  t 3.05, month-matched +0.301% at t 1.81, worst ME-1 session of any vehicle at
  -11.76 bp). (a2_c2_spy_me5.py, a2b)
- **The investment-grade complex pinned at 52-week lows, translated to IEF.**
  The anchor leg is wrong-signed: IEF within 1.0% of its 52-week low predicts
  IEF **-0.021% at a 49.2% hit** (excess -0.034pp), negative in every era and
  worst from 2022. No credit component to translate either, LQD's residual
  against IEF being -0.59 bp on the cell against +0.70 bp all-days, reproducing
  the 2026-08-12 PPI finding. Cost settles it: the join is **1.0x** a 3 bps
  round trip, today's exact rung 1.3x, freshness-filtered 2.5x. The excess ratio
  TLT/IEF is **1.22 against a daily-sd ratio of 2.13**, so translating W5 down
  the curve divides the edge by ~1.75 while the round trip stays flat — IEF is
  the wrong vehicle for this shape by construction. **Amendment to W5**: the
  shape belongs on TLT and only fresh — TLT on the drop-TLT rung, episode-first,
  ex-2022 is +0.289% at a 61.1% hit, t 2.19, **9.1x cost**, N=18. Do not widen
  the rung to reach a sample. (a3_c10_ief_ig_rungs.py, a3b)
- **Copper, the first metal examined here that is neither gold nor silver.**
  Premise false (above), and dead on its own numbers: the 52-week-high gate
  alone pays +0.691% (N=124) and the intersection with the thrust **-1.424%**
  (N=8, 3-5), negative at every horizon 2 through 10 (edge -0.637 to -2.486pp),
  reference-class rank **23 of 29** at P(random member >= FCX) = 0.793, and at
  the loose 10% rung FCX ranks 51 of 107 at P = 0.477. The one non-negative
  variant is 0.0x cost with a day-level mean of -0.293% and a sign that flips on
  the declustering gap (+0.479 / +0.705 / +0.529 / **-0.146** at gaps 5/10/21/63).
  Vehicle search fails across eight instruments (max +0.77pp, inside the noise
  band). Book overlap: on this exact state book-wide the ledger is **194 SHORT
  to 45 LONG** (4.3:1), and FCX's only ledger expression ever is 4 Overbot Vol
  Spike shorts at avgR +1.50. (b1_c3_copper_thrust_r1.py, b1b, b1c)
- **The energy z10 cluster, and it genuinely was NOT a re-skin of the
  2026-08-17 energy-thrust kill** — measured overlap 0 of 100 trigger days on
  the rank mask and 4 of 100 on the magnitude mask. It died on its own: see the
  effective-N and count-gradient entries above, plus definition fragility on the
  z rung (1.75 gives +0.390%, 2.25 gives -1.103%) and on membership (the only
  positive variants exclude the two ETFs and **do not fire today**). Reference
  class across 8 sectors with 25 random 11-member subsamples each: energy ranks
  **8 of 8** at -0.811pp, P(random sector >= energy) = 1.000. Alphabetical
  placebo goes to **8-for-9** (firing names -0.761% against alphabetical
  -0.727%, difference -0.034pp). Never produced an August episode.
  (b2_c7_energy_z10_cluster_r1.py, b2b, b2c, b3)
- **Long SPY against short QQQ, i.e. the 2026-08-11 pitch with the sign
  reversed.** Note first that `fingerprint()` keys on `TICKER:SIDE`, so the
  repeat block does NOT fire on a reversed pair — measured, the two hashes
  differ — while the structural object is the same one 9 sessions later. It
  dies on leg attribution: beta of SPY on QQQ is **0.617**, so equal-dollar is
  -0.383 units of QQQ beta; the long SPY leg pays -0.038% against SPY's own
  +0.192% drift (**excess -0.229pp**, i.e. negative alpha) and the whole spread
  comes from the short leg, which reverses at h=10. The beta-neutral residual is
  negative at six of seven horizons. The mechanism runs backwards where it
  matters: on days tech's 63-day rank is bottom-quintile while the index's is
  not, **QQQ LONG pays +0.508% at h=5**. Cost 2.1x at best, 0.3x on the literal
  form, and the naked long beats the pair at every horizon (the 2026-08-19 line
  reproducing). Confirms the 2026-08-13 note that index pairs are not
  interchangeable: corr with the DIA/SPY residual is **+0.351 to +0.429**, the
  mirror of the -0.363 to -0.442 recorded against the 08-11 side.
  (c3_c8_spy_qqq_pair.py, c4)
- **Gold strength with the 10-year yield at a 52-week LEVEL high.** Same object
  as the killed rank form (91% mask overlap, above) and dead independently: the
  gate-off parent is already negative against its own drift (h=10 -0.481%, edge
  -0.942pp, 40.8% hit, sign p 0.952) and the yield gate moves it **+0.010pp**.
  Every cell with N>=8 in the 180-cell grid is wrong-signed; rotation null
  P(grid max |t| >= 2.55) = 0.560. Today's state is the worst bucket, since the
  2026-08-21 GLD-drawdown kill reproduces here independently: thrust with GLD
  more than 10% below its 52-week high pays **-0.798% at h=5 (edge -1.012pp,
  35% hit)** and -1.307% at h=10, and GLD is -14.63% off. The joint cell has
  **3 days in 20 years and neither historical one carried today's drawdown**.
  Cost -2.9x. (d3_c9_gold_yield_level.py, d3b)
- **XLI washed out while a cyclical peer prints a 52-week high.** The pitched
  rung has **no history at all** (r5 rank <=3 with a peer at a high: 0 days
  ever) and today's literal state — XLI r5 rank <=5 with BOTH XLB and XLE at
  52-week highs — has occurred **once, today**. Loosening to a populated rung,
  the gate removes 3 of 150 episodes and moves the parent by **0.021pp**, the
  k-ladder is a one-rung spike with k=8 wrong-signed, and the rank ladder
  collapses as soon as it has observations (rung 7 pays +0.106% at a 33% hit,
  edge -0.109pp). The pair loses to the naked long at every horizon (h=5: XLI
  +2.284%, XLI-XLB +0.119%, XLI-XLE -0.869%) at 1.5x cost. Reduces to the
  book's own dip-buy family and **underperforms it**, ranking 3rd of 6 vehicles
  at h=3 and paying +0.074% at a 52.2% hit from 2018 against a family value of
  +0.534%. Book overlap: the morning's staged OLV longs (LUV, CHRW, CMI, WWD)
  correlate **0.803** with XLI at beta 1.16, a 64% variance duplicate.
  (d1_c4_xli_pair.py, d1b)
- **Cross-sectional new-high breadth with the index off its high.** Premise
  false (above) and the gate is a negative-value filter: breadth alone pays
  +0.104% at h=5 against an all-days +0.192%, i.e. **-0.086pp against doing
  nothing**, while the index-distance leg alone pays +0.266% and carries the
  cell. Tolerance walk costs 77% of the edge on a 0.75pp nudge (0.25% +0.452%,
  1.00% +0.106%), and the two universes disagree about the gate's worth by an
  order of magnitude. Bull-tape selector: **100.0%** of tape trigger days sit
  above SPY's 200d against a 71.6% base rate, with the trend split having N=1
  episode below it. And today's regime is outside the sample — median trigger
  ma10(63d) is **24.8** with an all-time max of 80.6 against today's **89.5**,
  and split on the live exposure-leg rule the edge is entirely in the
  complement (leg-OFF +0.008% at a 50.0% hit, leg-ON +0.754% at 81.0%, t=3.10).
  (c1_c6_newhigh_breadth.py, c2, c5)

### Calendar finding, filed because it is about the anchor set rather than one cell

- **On 2026-08-24 every macro anchor inside the 10 td horizon was already
  closed**, which is a first for this repo and worth recording as the reason a
  morning can be structurally empty. Jackson Hole (JH-4) is swept on seven
  asset classes with a pre-speech class mean of +0.010pp; the opex anchor
  (opex+1) is closed in both directions on equities and across ten non-equity
  vehicles by ten horizons; the two collide (the August opex close is JH-5 in
  21 of 26 years); NFP at +9 td is at the horizon cap and its one live cell is
  midterm-parked to 2027-01; and CPI, PPI, FOMC and quad witching are all
  beyond +12 td. That left month-end, which this morning closed on equities and
  suspended on rates. When the anchor set is empty the honest move is a
  price-state sweep and a stand-down, not an eighth class on a dead anchor.

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

## Method traps (2026-08-26, from a 12-candidate sweep that killed all 12)

- **A breadth COUNT can be 96% redundant with a single-name cell already
  closed.** The metals-complex count (members at a 21d rank >= 95) looked like
  a new object beside the closed GDX thrust cell; **116 of 121 count>=4 days
  (95.9%) ARE GDX-rank>=95 days**, and the residual "breadth without the
  headline name" cell is **5 days**. Before treating a count as a new object,
  compute its overlap with the largest member's own trigger. Companion to the
  2026-08-24 energy-count entry, whose ladder shape reproduced here and worse:
  long GDX h=5 by count runs +0.280 / +0.161 / -0.743 / -0.097 / **-0.861**
  (today's live count of 4) / -1.616 / +2.911% against an all-days +0.272%.
  (b3_c5_metals_breadth.py, b3b_c5_metals_round2.py)
- **The COMPOSITION behind a count is a free parameter, and enumerating it can
  produce three signs from one live state.** The same 2026-08-26 metals tape
  reads **+0.448%** (today's exact 3-equity 1-metal composition, N=12),
  **-0.861%** (the pitched count>=4 cell) and **+2.911%** (count=6) depending
  on which defensible reading is taken. Nine neighbouring definitions of the
  identical rule span -1.330% to +0.930%, five positive and four negative. A
  count trigger owes its membership list pre-declared, exactly like a grid.
  (b3b_c5_metals_round2.py)
- **A gate can be worth +0.6pp at one depth and NEGATIVE at the depth that is
  live.** The credit cell (HYG at a fresh 52w high while SPY is off its own)
  passed round 1 cleanly -- +0.530% over 62 episodes, Welch t +1.97, era-stable,
  8.8x cost -- and died to a depth-matched split: the gate is **+0.615pp at h=3
  when SPY is >=2% below its high and -0.042pp in the 1.0-2.0% band**, which is
  where today's -1.54% lands. Always split the conditioning variable at the LIVE
  value, not just at the threshold the cell was defined on. (a6b_c6_round2.py)
- **"Out of sample on the dial" is a kill in its own right, and this tape keeps
  triggering it.** Three independent cells died partly on it: the credit cell's
  **maximum historical dial reading is 68.0 across 100 trigger days** against
  today's 88.9; the dial >= 85 analogue has **16 resolved observations that are
  all one episode** (2021-12); and the QQQ cell killed on 2026-08-25 topped out
  at 80.4. When the live regime sits 20+ points outside everything a cell was
  measured on, say so as a kill rather than as a caveat.
  (a6b_c6_round2.py, a8_c8_dial85.py, a8b_c8_round2.py)
- **Charge a SESSION scan the same way you charge a grid.** The September
  month-turn's whole three-day window is one session: **ME-3 to ME-2 pays
  +83.12 bp at t=+3.11 on IWM, 93% of the window (100% on SPY)**, with every
  other session in ME-8..ME+7 inside +/-30 bp. Charged for the 16-session x
  12-month grid it was found in, a random scan on 26 draws returns
  **P(max |t| >= 3.11) = 0.063** (SPY 0.421). A one-session spike found by
  looking at sixteen is a scan result, and shifting the entry one session took
  h=5 from +0.86% to -0.27%. (c9c_scanned_session.py)
- **A sector-conditioner story is falsified when the gate pays BEST on the
  members with the wrong sign of exposure.** The XLRE duration-rally gate was
  pitched as rate repricing; XLRE's daily beta on TLT is **+0.060 (0.2% of
  variance)** while the biggest gate values at h=5 belong to **XLB (-0.219),
  XLF (-0.412) and XLE (-0.489)**, the three most negatively duration-correlated
  sectors. The gate is a market-timing selector. Check the exposure ordering
  before believing a channel. (c10b_loosen_refclass_mechanism.py, c10c_era_matched.py)
- **The midterm inversion is now book-wide rather than anchor-specific.** It
  killed or blocked five unrelated candidates in one morning: the dollar
  washout (**-0.479%, 36.4% hit, N=22**, both vehicles), the September month
  turn (**-0.366% on 3-3**), the payrolls run-up (**-0.676% on 3-3**), the
  bank pair (-0.233% vs +0.125%) and the two put/call cells (h=10 **-1.219%**
  vs +0.517%; h=3 **0-for-5**). Treat a midterm split as a REQUIRED round-2
  test in a midterm year, not an optional one.
- **A "virgin data surface" can have exactly one regime, and that is a
  structural fact to establish FIRST.** The CBOE **index and etp put/call
  series only begin 2019-10-07**, so the first usable trailing-252d
  point-in-time percentile is **2020-10-16** and no pre/post-2018 era split
  exists by construction. Establish a series' usable span before spending a
  morning's budget on it. (a0_pc_recon.py)

## Cells swept and empty (2026-08-26)

- **The CBOE index put/call at a trailing-year low with equity P/C mid-range,
  the first put/call cell ever tested in this repo.** The gate is
  anti-selective: index<=10 alone is 289 days, adding the mid-range equity leg
  discards **184 of them to move h=10 from +0.237% to -0.084%**, and the
  discarded complement pays **+0.503%**. The threshold ladder runs backwards
  (<=15 pays +0.762%, <=20 pays +0.862%). Long is negative against its own
  same-era drift at all six horizons (worst **-0.747pp** at h=10); short is
  8-18 at h=10 and one episode (drop-best-1 takes +0.084% to **-0.212%**). The
  pitched ma10/252 definition is the **worst of nine** MA-by-lookback
  neighbours. (a1_c1_index_pc.py, a1b_c1_round2.py)
- **The ETF-options put/call at a trailing-year low on small caps.** NOT a
  duplicate of the index cell -- the two share 38 days of 289/140, jaccard
  0.097 -- so it dies on its own numbers. Sign flips inside its own horizon
  profile (**-0.749% at h=3, +0.891% at h=10** on the same 21 episodes); five
  of nine definition neighbours are negative at h=10 (ma21/252 **-0.782%**);
  drop-best-2, both 2024, takes h=10 to **-0.082%**; the IWM-vs-SPY relative
  form covers **3.7x** a 12 bp round trip against a 5x bar and 0.6x on
  drop-best-1. (a2_c2_etp_pc.py, a2b_c2_round2.py)
- **The bare dollar 21d washout, the untested parent of two parked entries.**
  Full sample is flat: **+0.029% over 63 episodes, 31-32, sign p 0.599, 2.0x
  cost**, with a plateau ladder that turns negative at every looser rung.
  Midterm is wrong-signed on both vehicles. Two useful sub-findings filed with
  it: DX vs UUP agree on **95.8% of matched episodes with a 1.4 bp per 5td
  structural gap** (reproducing the registry's 1.3 bp figure), and the
  registered one-day weak-NFP-close cell does NOT transfer to the 21-day rank
  parent (the 22 episodes within 3 td of a print pay **-0.089%** against the
  away episodes' +0.093%). Parked to the first non-midterm instance.
  (b1_c3_dollar_washout.py, b1b_c3_dollar_round2.py, b1c_c3_nonmidterm_ladder.py)
- **The dollar washout translated to EEM, i.e. the EM funding trade.** Excess
  against its own all-days control is negative at all five horizons and the
  episode-vs-control difference is **-0.000% at Welch t -0.00 over 56
  episodes**. The dollar gate WITHOUT EEM's own washout pays +0.137% against
  the joint cell's +0.174% and all days' +0.243%. Beta-neutral residual
  **-0.093% against 1.13x SPY**, worse than the all-days residual, so it is a
  levered index long. Reference class: **3 of 6, P(max-of-6) = 0.938**. It
  dies with the country-decoupling family. (b2_c4_eem_dollar.py)
- **Long KRE against XLF, the mirror of the parked short.** The literal rung is
  **1 day in 4,818 sessions** and it is today; the nearest neighbour is 2
  episodes, both losers, at -3.199%. Naked long KRE pays +0.464% at h=5 while
  the short leg contributes **-0.414pp**, leaving **0.4x** cost, and the
  surviving long is not alpha (**-0.185% vs SPY**; at h=10 the pair is
  **-0.737% at t=-2.35**). Ten sub-industry-vs-parent pairs rank it **5 of 10**
  with a fixed-effect common excess of -0.071%, independently reproducing
  watchlist #19's blocker on a different construction. **Financials are now
  closed in both directions.** (c7_kre_xlf.py, c7b_loosen_and_legs.py)
- **The fragility dial at >= 85 as a directional index analogue.** All 16
  resolved h=5/h=10 observations come from the 2021-12 top, and the second
  episode is the one being traded, so it cannot also be its own evidence. The
  level ladder is monotone across the whole range (+1.035% bottom bucket to
  **-1.341%** top), i.e. the plain dial LEVEL as direction, the object CLAUDE.md
  records dead at PIT t = -0.23. The loosest form with spread (>=70, 12
  episodes) goes to **-0.103%** at h=10 on drop-best-3, and 2021, the majority
  year, pays -0.337%. **Vintage flips it**: +1.427% on the sizing parquet
  against **+0.245% at a 41.7% hit** on the research recompute, the two
  agreeing on 82 of 102 days with ma10 differing by up to 11.8 points. Long
  TLT as the defensive expression is -1.492% at a 12.5% hit.
  (a8_c8_dial85.py, a8b_c8_round2.py)
- **The turn INTO September entered at ME-3, on IWM and SPY.** One scanned
  session (see the method trap above). September is not distinguishable in its
  own family: **max-of-12 permutation P = 0.238** at its best horizon and
  0.987/1.000/1.000 at h=5/7/10, ranking 2nd/8th/10th on SPY, with June-end,
  May-end and Oct-end all beating it somewhere. The bare ME-3 entry pooled
  across twelve months is worth **1.3x cost**. Midterm h=3 is -0.366% on 3-3
  and h=7 hits **16.7%**; era runs +1.195% pre-2018 to **-0.500%** after.
  **The month-end anchor is now closed on equities in the month-of-year
  direction as well as the month-position one.**
  (c9_month_of_year.py, c9b_gate_and_scan.py, c9c_scanned_session.py)
- **XLRE out of a lagging base into a duration rally, the first real-estate
  cell tested here.** Pitched rung is 2 episodes; gate value swings
  +0.886 / -0.254 / **-2.422** / -0.640pp across h=3/5/7/10 and the
  "TLT NOT rallying" complement beats the joint cell at h=5 and h=7. Era-matched
  across ten sectors it ranks 2 of 10 then 4 of 10, family mean gate +0.476pp
  with **nine of ten sectors positive, Cochran Q 6.07 on 9 df, I-squared 0%**,
  random-date max-of-10 **P = 0.113**. Top-2 episodes are 61% of total
  (2020-04-02 alone +15.32%); drop-best-3 is **4.1x** cost at h=3 and 2.6x at
  h=5. The pooled family form is NEGATIVE at every horizon ex-2020, and deep
  history agrees (IYR **0.7x cost** over 2000+, VNQ 2.0x). **Real estate is
  closed and there is nothing to park.**
  (c10_xlre_rates.py, c10b_loosen_refclass_mechanism.py, c10c_era_matched.py, c10d_family_pooled.py)
- **MOVE/VIX at a trailing-year high, traded on duration.** The premise is
  false: **MOVE is at the 46.4th percentile of its own trailing year** while
  VIX is at the 17.5th, so the ratio is high because the denominator is cheap,
  and today's sub-population of 397 days has a median MOVE percentile of 28.6.
  Wrong-signed against its own control **15 of 15** across three vehicles and
  five horizons. Gate attribution: rich-ratio +0.073% vs not-rich +0.072% vs
  all days +0.080%, so the gate discards 77% of days to move the mean by
  **-0.001pp**, and the half matching the stated mechanism pays **-0.144%**.
  Cost 0.4x (TLT) and 0.3x (IEF). Data note for reuse: **^MOVE runs
  2002-11-12 to date, 5,881 observations, 94.8% business-day coverage** -- it
  is usable, it just says nothing here. (b4_c11_move_vix.py)
- **The run into NFP entered seven sessions early, four vehicles.** Ladder is a
  plateau on every vehicle (SPY's k=-2 beats the true anchor; **GLD peaks four
  sessions AFTER the print**, rank 7 of 16; TLT 9 of 16). The print session
  carries **6.5%** of the trade (+0.035% at t=+0.53 against a +0.500%
  six-session run-up; TLT's print session is -0.094%). Charged for the
  pre-declared 4x6 grid, the best cell returns a rotation-null
  **P(max >= observed) = 0.338**. September prints pay **-0.038%** against a
  same-month control of +0.156% while the other eleven months pay +0.588% at
  t=3.77, and September-in-a-midterm is **-0.676% on 3-3**. Only SPY clears a
  naked cost bar, at **4.5x** month-matched against 5x.
  (c12_nfp_ladder.py, c12b_sept_midterm_control.py)

### Calendar finding, filed because it is now three consecutive sessions

- **The anchor set was empty for a THIRD straight session, and the two
  additions that came into range both died.** NFP crossed from +8 td to +7 td,
  putting it inside the horizon cap for the first time, and the month turn came
  inside every horizon. Both were checked rather than dismissed, and both are
  now closed: NFP on the ladder plus a September-midterm cell of -0.676%, the
  month turn on a scanned session and a max-of-12 permutation P of 0.238. With
  Jackson Hole closed on seven classes, post-opex closed both ways, and
  CPI/PPI/FOMC/quad beyond the cap, **the late-August midterm calendar is
  exhausted in the strong sense: not merely swept, but swept including the
  cells that only became reachable this week.** The next genuinely new anchor
  is the September CPI/PPI pair on 2026-09-10/11, which enters the horizon
  around 2026-08-28.

## Method traps (2026-08-27, from a 10-candidate sweep that killed 9)

- **The PASS-THROUGH RATIO settles "is this vol effect harvestable" in one
  number, and it should be run before any statistics.** A spot-VIX effect only
  reaches a futures-based ETP to the extent the FRONT FUTURE moves with spot.
  Back out the implied front future from UVXY/SVXY, then compare the cell's
  pass-through against the unconditional baseline. Jackson Hole: on the 14
  SVXY-era JH+0 sessions spot VIX falls **-1.79%** while the implied front
  future moves **-0.17%**, a ratio of **0.09x**, against **0.55x** measured over
  all **2034** down-VIX sessions (spot -4.86% -> future -2.65%). A sixth of the
  normal rate means the spot drop is a same-day repricing of the near strip the
  front future has already discounted, so there is nothing in it for the
  vehicle. This is the general form of the registry's existing SVXY
  beta-translation objection and it is cheaper to run: it kills the trade
  without needing the beta regression, the ladder or the era split (all of
  which agreed here). Companion note: the JH spot effect itself is weaker in
  the tradeable era, -1.79% on 10/14 (sign p 0.0898) against -2.60% on 21/26
  (p 0.0010) on the full 2000+ sample. (a1_c1_jh_vol.py, a1b_c1_jh_vol_h1.py)
- **A one-session mechanism does not license a multi-session hold, and the
  check must test the horizon the mechanism CLAIMS.** The JH vol work measured
  the spot drop as exactly one session wide (JH-1 -> JH+0 -2.60%, 21/26;
  JH-1 -> JH+2 back to +1.04%) and then tested the vehicle only at h=2 and h=4.
  The h=1 vehicle form had to be requested separately. Match the hold to the
  measured width of the effect before running the battery, or the round-1
  verdict is about a different trade than the mechanism describes.
- **The ledger's columns are SPACE-separated and a wrong guess returns a FALSE
  ZERO overlap.** `data/backtest_trades_full.parquet` carries `Signal Date` and
  `Strategy`, not `Signal_Date`/`Strategy_Name`. A
  `dcol = "Signal_Date" if ... else led.columns[0]` fallback silently joins on
  `trade_id` and reports "0 ledger signals", which reads as "no book overlap"
  and is the most reassuring possible failure. Caught on the composer's own
  red-team pass; the true overlap was 5 signals on the trigger days and 66
  inside the holds. ASSERT the column names. (r1_redteam_c6.py)
- **A cross-sectional reference class can CONFIRM as well as kill, and the
  asymmetry is worth naming.** Three of this morning's candidates died to
  reference classes (132 sector pairs at family-wise p 1.0000, 23 sector ETFs
  at 0.8805, 205 single names at 1.0000) and the one survivor was confirmed by
  the same instrument: the GDX rule run independently on NEM/AEM/AU/KGC pools
  to +2.228% over 30 episodes at 22-8, sign p 0.0121 against those names' OWN
  up-rate. The class also supplies the honest MAGNITUDE, since GDX's +5.5pp is
  about 3x the family estimate, so the shipped expectation was shrunk toward
  the family rather than quoted at the headline. Build the class for every
  single-instrument cell, not only the ones you suspect. (c6c_gdx_replication.py)
- **Widening a rank rung can degrade the cell at the LIVE depth specifically,
  which is a different fact from the cell being fragile overall.** The GDX dose
  response grades correctly on every rung, but today's flush depth (-2.94%, the
  (-3,-2] bucket) pays +5.104% on 3-0 at rank>=99, +4.837% at >=97 and only
  **+2.206% on 5-3 at >=95**. The tight gate is load-bearing at this depth even
  though the ladder looks monotone in aggregate. Split the conditioning
  variable at the LIVE value, not just across its range.
  (c6d_gdx_dose.py, r1_redteam_c6.py)

## Cells swept and empty (2026-08-27)

- **Jackson Hole on VOLATILITY, which completes that anchor to EIGHT asset
  classes and closes it in the strong sense.** Long SVXY at h=1 (the only
  horizon the mechanism claims) pays +0.222% against an own-drift +0.099%,
  record 9-5, exact sign p 0.2120, and the **beta-hedged residual is +0.0010pp
  -- after SPY beta there is no cell at all**, because SPY's own h=1 on those
  anchors is -0.0162% against +0.0632% all-days, so the raw number is the -0.5x
  inverse of a mildly negative SPY day. The h=4 form is +0.332% against a
  +0.419% drift (excess -0.087pp) with **top-2 episodes = 115% of total**. The
  anchor ranks 11 of 25 at h=1 and 13 of 25 at h=4; k=+10 pays 2-13x more. The
  **live -0.5x leverage era is NEGATIVE** (-0.054%, N=8) while the entire
  positive mean is the pre-2018 -1x era (+0.591%, N=6). Midterm h=1 is -1.387%
  on N=3, the eighth independent JH midterm inversion. Do not reopen this
  anchor on a ninth class. (a1_c1_jh_vol.py, a1b_c1_jh_vol_h1.py)
- **The month turn on commodities and metals, the last unswept class of that
  anchor.** EW 8-name basket, ME-2 close to the ME-0 close: **+0.090% against
  an own drift of +0.090%, excess -0.000pp at Welch t +0.00** over 243
  observations, 128-115. Era split is the arbitraged-calendar signature,
  **pre-2013 +0.548% at a 65.8% hit -> 2013+ -0.131% at 46.3%**. Cost 0.2x (8
  legs x 5 bps = 40 bp against a 9.0 bp edge). The h=5 stretch form (+0.431% vs
  +0.223%, sign p 0.0412) dies three ways: beta-hedged residual +0.098pp on
  118-112, **August-only -0.136% with excess -0.359pp** and today IS an August
  turn, and the offset ladder is a plateau ranking m=3 5 of 21. Per-ticker grid
  best is XME h=5 at Sidak-over-16 p 0.2374. **The month-end anchor is now
  closed on equities, rates, FX and commodities.** (a2_c2_month_turn_cmdty.py)
- **A hold containing BOTH the PPI and the CPI of a back-to-back pair.** The
  gate is ON 39.7% of all days and agrees with the trivial "CPI in the hold"
  gate on **5045 of 5465 days (92.3%)**, reproducing the macro-vacuum kill
  (278/318). The gated cell is BELOW drift: today's exact configuration (PPI at
  entry+9, CPI at entry+10) pays +0.406% against a same-span +0.458%, and the
  **NEITHER cell (+0.538%) beats the BOTH cell outright**. Anchor ranks 21 of
  25. Gap-share: the two 08:30 release gaps contribute **+0.080% of the +0.406%
  hold, about 5%**, against an unconditional two-gap baseline of +0.0595%.
  TLT/IEF/GLD all below their own drift (-0.019 / -0.044 / -0.069pp). The
  September-only sub-cell (N=9, 8-1, sign p 0.0195) is the **equity month turn
  in disguise: 7 of its 9 dates are month-turn dates**, and today's instance is
  an August one where the cell is 50.0%. (a3_c3_ppi_cpi_containment.py,
  a3b_c3_september_probe.py)
- **Three-month implied vol at a one-year LEVEL floor.** SPY h=10 +0.592%
  against own drift +0.576% (edge +0.016pp, Welch t +0.08) and against a local
  +/-126td control of **+0.752%** -- the cell underperforms its own
  neighbourhood. **The threshold ladder is INVERTED**: lvlpct <=2 +0.076% /
  <=5 +0.092% / <=10 +0.256% / <=20 +0.373% / all days +0.388%, so the more
  extreme the floor the less it pays. h=3 and h=5 are wrong-signed against
  control (-0.167pp, -0.198pp). SVXY post-leverage-cut clears drift by 15 bp
  (1.9x) with two episodes at -25.04pp against a +28.02pp total. Note the level
  basis was chosen deliberately over the return-rank basis per the 2026-08-10
  level-vs-rank entry; both are now closed. (c4_vix3m_floor.py)
- **Sector-vs-sector pairs at a 63-day spread floor, which closes the
  relative-value pair form the way 2026-08-07 closed sector-vs-index.** Across
  **132 ordered pairs** of 12 sector names the identical rule is perfectly
  homogeneous: **Cochran Q 93.27 on 131 df, p 0.995, I-squared 0.0%**, common
  excess +0.179pp, and the permutation max-of-132 null has a **median of
  +1.440pp which is above every observed pair**. XLK-XLV (the "PIT 0.0" cell)
  ranks **75 of 132, family-wise p 1.0000**; SMH-XLV 0.9753; best pair in the
  grid 0.8560. Per-leg attribution kills the structure separately: **long SMH
  outright +1.282pp, short XLV outright -0.283pp, pair +1.010pp** -- the short
  leg subtracts and the pair is strictly worse than the outright long. Regime:
  SPY<200d +3.172% (t 2.89) against SPY>=200d **+0.033% at a 45.5% hit**, and
  the trigger over-selects SPY<200d by +12.6pp, which is the closed
  laggard-snapback mechanism. Magnitude-only form flips sign inside one family
  (-0.504 / -0.308 / -0.322 / +0.957pp). (b1_c5_sector_spread.py, b1b, b1c)
- **The single-name washout at a 21-day rank floor, on a 205-name reference
  class.** Common excess +0.226pp (z +1.45), **Cochran Q 219.33 on 204 df,
  p 0.220, I-squared 7.0%** -- homogeneous, nothing distinguishable. TJX's
  excess is **-3.554pp, rank 186 of 205, family-wise p 1.0000**, and its own
  h=10 cell is **-2.766% against a +0.753% drift on 3-3**. The h=5 form is one
  crisis: **drop 2008 and +4.233% becomes -0.792%** against a +0.400% drift.
  The POOLED cell flips sign after 2018 (name-matched excess h=5 +1.575pp at
  t 6.10 -> **-0.809pp at t -2.69**; h=10 +1.275 -> -1.081pp), and only returns
  by excluding 2008/09/2020, which is a post-hoc carve-out of exactly the tail
  the trigger over-selects. **There is also no basket form**: loosened to
  z10<=-1.5 / r21<=10 / <=5% off low, the trigger still returns 1 of 218 names.
  (b3_c10_washout_refclass.py, b3b_c10_round2.py)
- **The dollar washout CONFIRMED BY A BOUNCE, i.e. the conditioner that was
  supposed to rescue the midterm-parked parent.** It selects the wrong half.
  The bounce leg strips **93.5%** of parent days and the survivors underperform
  the discards: non-midterm h=5 pays **-0.129% over 8 episodes** while the
  anti-cell the filter throws away pays **+0.125% over 52**, against a parent of
  +0.051%. Every adjacent rung flips sign (r5>=75 +0.227% at a **25% hit**,
  r5>=50 -0.183%, r21<=2 & r5>=60 -0.147% at a **16.7% hit**). Midterm is N=2 at
  every horizon, so the parent's midterm block (watchlist 27) stands untouched
  and is NOT what killed this. (c9_dx_washout_bounce.py)

### Calendar finding, filed because it is now four consecutive sessions

- **The anchor set was empty for a FOURTH straight session, and the two cells
  that only became reachable this week both died the day they came into range.**
  The September PPI/CPI pair crossed inside the 10 td cap for the first time
  (+9 and +10 td) and is now closed as a containment object; Jackson Hole moved
  to JH-1 and its eighth and final class closed. That exhausts the anchor
  inventory in the strongest sense this repo has recorded: **every macro anchor
  reachable from late August of a midterm year has now been swept on every
  asset class it has a vehicle for.** The month-end anchor closed its fourth
  and last class the same morning. The next genuinely new anchor is the
  September FOMC on 2026-09-16, which enters the 10 td horizon around
  2026-09-02, and the pre-FOMC drift is already spoken for by the event
  sleeve's T1/T2 (midterm years take the T2 short form, gated on SPY's 21d rank
  being under 50). **The practical consequence for the next week is that a
  price-state sweep is the only honest search mode**, which is what produced
  today's single survivor.

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

## Method traps (2026-08-31, from a 22-candidate sweep that killed all 22)

- **The LAG PROFILE settles whether an effect has a shape, and it costs one
  line.** Run lag=0 / lag=1 / lag=2 at the same horizon before crediting any
  mechanism. The short-silver-after-a-complex-break cell reads **+0.039% /
  +0.516% / +0.035%** at h=1: one session wide, and it starts a session LATE.
  No forced-deleveraging continuation predicts that, and the direction of the
  anomaly is itself the tell -- the registry's standing worry is that the
  untradeable lag=0 look FLATTERS a cell, and here the tradeable lag=1 is
  **13x LARGER** than lag=0, which is backwards from every other cell measured
  in this repo. Nothing else caught it: gate attribution passed decisively
  (conjunction +0.531% against +0.031% for the single-name parent and -0.278%
  for the anti-cell), the local +/-126td control cleared at welch t +2.21, the
  record was 67-52 at sign p 0.019 scored against silver's own down-rate,
  declustering was stable at gap 5/10/21, and the six-family reference class
  CONFIRMED rather than killed. Era, decluster and reference-class work would
  all have shipped this. (b4c_c19_slv_short_teardown.py)
- **A continuation cell must be split by the ENTRY-DAY move, and the split is
  a mechanism test rather than a robustness test.** The same silver cell pays
  **+0.867% on 28-15 when silver BOUNCES more than 1% on the entry session**
  and +0.573% on 25-25 when it keeps falling, with an entry-day correlation of
  **-0.033**. "The highest-beta member keeps bleeding" is falsified by its own
  data. Any story about flow persisting into the next session owes this split.
  (b4c3_c19_entryday_and_2026.py)
- **`pitch_lab.cluster_note` ranks the top-k by ABSOLUTE value, so it NETS a
  large winner against a large loser and can report a concentrated cell as
  clean.** The silver cell's "top-2 episodes = 3% of a +41.66pp total" is a
  +16.86% and a -15.56% cancelling; ranked by VALUE on the side actually being
  traded, top-3 is **103% of total** and drop-best-3 is negative. Report
  concentration by value on the traded side, not by magnitude.
- **A percentile is two different statistics and this repo uses both.**
  `rolling(252).rank(pct=True)` (inclusive of the current observation) and the
  `w[:-1] <= w[-1]` form used by the morning recon (exclusive) differ by about
  0.4pp on a 252-day window. On the oil-services spread cell that is the whole
  result: the identical rung on identical data gives **+0.934% on 28-23 at
  15.6x cost** under one convention and **+0.005% on 28-29 at 0.08x** under the
  other, and today's live bar straddled the gate at **3.98 excl-self against
  4.37 rank**. Every parked cell must record which convention minted it.
  (b4_c6_oih_xop_ladder.py)
- **Charge the grid you SCANNED, not the axis you found it on.** The
  bond-vol band cell cleared a band-only permutation at P 0.029 and reads
  **P 0.857** once charged for the bands x horizons x vehicles it was actually
  walked over -- a below-median draw from the best-cell-under-no-effect
  distribution. The candidate spec itself named four sign and vehicle
  combinations, which is the disclosure that makes the wider charge mandatory.
  (b2c_c2_fullgrid_and_dose.py)
- **An inverted-U conditioner ladder falsifies a directional mechanism even
  when the live bucket is the maximum.** The tails are where the mechanism
  makes its strongest prediction, and for "an orderly repricing trends" the
  most compressed bond-vol bucket [0,20) is the **worst** long-duration bucket
  at -0.809%. A monotone ladder supports a dose response; a hump means the
  chosen band is mid-range wearing an extremity label.
- **A DEPTH BAND is instrument-specific and cannot be transplanted.** The
  2026-08-26 credit kill established that the index-distance gate is worth
  +0.615pp beyond 2% and -0.042pp in the 1-2% band -- measured on SPY.
  Substituting IWM because it sat 3.06% off its high assumed SPY's ladder;
  measured on IWM, the 2.0-5.0% band is IWM's **dead** band at -0.119% and
  **-6.0x cost**, and the open-ended >=2% form is positive only because it
  pools that dead band with the >5% tail. Same kill, different ticker.
  (b2f_c4_hyg_high_iwm_depth.py)
- **Print the distance-from-extreme across a calendar anchor's history before
  running any statistics.** One `print` of 27 August month-end anchors' yield
  distances ended the September duration candidate in a line: **^TNX has never
  been within 2% of its trailing-252 high at an August month end**, distances
  running -1.54% (today) to -62.48%, so the interaction cell has exactly one
  observation and it is the live one. This is the 2026-08-07 count-first rule
  applied to a conditioner rather than to a joint state.
  (b2e_c8_empty_cell_and_fallback.py)
- **The placebo offset ladder is now 12-for-12**, adding a second single-name
  earnings failure (the pre-print anchor ranks 5 of 16, with four neighbouring
  offsets beating it) after the 2026-08-25 case.
- **`data/earnings_calendar.parquet` is not usable as an anchor calendar before
  ~1993.** 82-87% of 1985-1992 rows land exactly on a quarter END and 1988-91
  carries up to 61% weekend dates: those are fiscal period ends masquerading as
  announcement dates. Restrict to 1996+ before any earnings anchor (prices
  bound it at 1999 anyway). First use of this file as a pitch anchor.
  (b3_c3_recon.py)
- **^GSPC is not a usable OVERNIGHT instrument before ~2013.** Yahoo's
  synthetic open gives a pre-2013 overnight series with a **median of exactly
  0.000 at a 25.0% up-rate** and an sd of 0.159%. A dividend-contamination
  hypothesis raised against the month-end overnight cell was WRONG for the
  same reason and is recorded here so it is not raised again: the RAW
  unadjusted overnight excess is LARGER than the adjusted one (SPY +9.92
  against +7.38 bp), so adjustment does not manufacture overnight returns.
  (b1_c1_me0_overnight.py, b1f_c1_single_roundtrip.py)

## Cells swept and empty (2026-08-31)

- **The month-end anchor's OVERNIGHT return, which is a genuinely new return
  object in this repo and closes the anchor's sixth form.** All five prior
  month-end closures measured close-to-close; nobody had measured
  `Open[ME+1]/Close[ME-0]`. The headline is real -- SPY +10.48 bp against a
  +2.98 bp unconditional overnight, 206-113, **sign p 0.0004** against SPY's
  own 55.1% overnight base rate; IWM +16.80 bp, sign p 0.0004 -- and the
  mechanism is false. The reversal regression that the auction story predicts
  **runs backwards on the one session that has the auction**: slope +0.194
  (t +2.03) on SPY's ME-0 sessions against **-0.131 (t -6.56)** on all
  sessions, with IWM +0.081 against -0.079 and QQQ +0.057 against -0.277. The
  15-vehicle reference class then names it: **EEM (+21.2 bp) and EFA (+15.0 bp)
  rank first and second**, two markets that are SHUT during the US closing
  auction and reopen overnight in Asia and Europe, and the family is
  homogeneous (Cochran Q p 0.6875, I-squared 0.0%) at a common excess of
  **+8.26 bp (t +7.24)** with SPY 9 of 15. One market-wide overnight drift
  wearing fifteen labels. August is the WORST of the twelve months on every
  vehicle (SPY **-9.87 bp** at a 46.2% hit, DIA -13.24 at 38.5%); the cell is
  Dec plus Oct-Nov, and December fails its own max-of-12 scan at P 0.476-0.931.
  The ladder does not isolate (ME-4 beats ME-0 on all five vehicles, true
  anchor **rank 2 of 9**), era decay is monotone (SPY 10.34 -> 4.78 -> 3.91 bp),
  cost never reaches 5x, and August-in-a-midterm is **3-3 and negative on all
  five vehicles**. (b1_c1_me0_overnight.py, b1d_round2_refclass.py)
- **The intraday shape of the month-end session, and the first use of the
  15-minute cache by a pitch check.** Data is deep and fine (SPY/IWM
  2003-09-10 onward, 5,708 sessions, 265 ME-0). The finding is genuine: the
  ME-0 last hour IS distinguishable on SPY (**-0.065% against +0.004%, welch
  t -2.52**) and IWM (**-0.128% against +0.009%, t -4.35**) on a last-hour
  volume share of 30.0% against 23.7%, and the offset ladder isolates the true
  anchor at **rank 1 of 9** on both, which is rare. As a trade it dies on era
  sign instability: SPY runs **+14.46 bp pre-2013 -> +1.75 (2013+) -> -2.03
  (2018+) -> -4.82 bp (2020+)**, wrong-signed in the modern era, and IWM
  decays to 0.43x cost. QQQ has the volume signature (25.1% against 21.2%) and
  **no return signature at all** (+0.003%, t +0.20), which is absorption rather
  than impact and is the cleanest single argument that the flow does not move
  price. The one-round-trip join (buy 15:00 ME-0, sell MOO ME+1) is worse than
  either leg at 0.67x / 0.22x / 1.12x cost, because the last hour eats half the
  overnight. (b1b_c15_intraday_shape.py, b1e_c15_lasthour_standalone.py)
- **Short silver after the whole metals complex breaks together.** See the lag
  profile and entry-day entries above; parked with two arming numbers. Filed
  here because of what did NOT kill it, so nobody re-runs them: the
  state-matched and depth-matched splits are **not distinguishable** (worst
  welch t -1.59 for "all four of today's states") and the multiplicity charge
  on those refinements is **p 0.822 over a 56-cell grid**, so the
  unconditional cell is the honest estimate and the state-matched h=3/h=5 forms
  (+1.355%, +1.334%) are not real. The GLD-beta residual also survives
  (+0.322pp edge, 61.1% hit), so this is NOT the closed "second metals leg is
  size, not diversification" objection. Per-leg attribution on the same
  trigger: short gold is **-2.6x cost** and short the miners **-1.1x**, both
  with top-2 concentration at 202% and -233% of total.
- **Fading the miners' 21-day outperformance of the metal at a 98th-percentile
  spread.** Wrong-signed at every horizon (-0.186% at h=1 to **-1.718% at h=10
  on 24-41**) with the percentile ladder monotone in the wrong direction, so
  the more extreme the spread the worse the fade. The beta-neutral form is also
  negative and the long-metal leg subtracts. The seven-pair miner-metal
  reference class is homogeneous with a **negative** common excess (-0.021% at
  h=1, -0.165% at h=3) and no family support for the fade in any name, while
  the CONTINUATION side pays **+1.718% at h=10 on 41-24, sign p 0.045**,
  independently re-confirming the 2026-08-27 finding. (b4d_c18_gdx_gld_fade.py)
- **Industrial metals thrusting while precious metals flush**, which is inside
  the sector-pair family closed on 2026-08-27 and reproduces that closure on a
  fresh enumeration: 131 ordered pairs from a 12-name pool give Cochran Q
  p 0.936, I-squared 0.0%, permutation max-of-131 **P 0.723**, with this pair
  ranking **39 of 131**; the pair form is worse still (common excess -0.144%,
  P 0.996). The short leg is the wrong side, since silver RISES after this
  trigger (**-0.474% at h=5 on 10-19** for the short), so the pair is strictly
  worse than the outright everywhere, and drop-best-2 at h=5 is -0.721% with
  the top two episodes at 356% of total. The trigger has also been ON since
  2026-08-11 with nothing happening. (b4b_c6_c10_pair_refclass.py)
- **Long oil services at a services-versus-exploration 63-day spread extreme**,
  the parked watchlist entry, now CLOSED rather than re-parked. Beyond the
  percentile-convention finding above: the ladder is non-monotone and inverted
  at the tight end (pit<=0.5 **-1.404% on 10-14**, <=1.0 -0.655%, <=2.5
  +0.934%, <=3.0 +0.005%, <=20 +0.875% on 90-63), so extremity is not what the
  cell keys on; the headline exists only at a declustering gap of exactly 10
  (gap 5 +0.306%, gap 21 +0.091%, gap 42 **-1.682%**); drop-best-3 is -0.126%;
  midterm is **-2.233% on 6-11**; and the 12-pair reference class is
  homogeneous with a **negative** common excess of -0.121% at a permutation
  max-of-12 P of 0.806, where this pair is not even the family maximum. The
  entry's "four wins away" arm was arithmetically wrong -- it converted losses
  to wins rather than adding episodes; the real answer at the 4.0 rung (31-32)
  is that no number under 15 consecutive wins arms it.
- **The tail-premium-to-at-the-money ratio (SKEW over VIX3M) at a trailing-year
  extreme.** A total re-skin, and the overlap statistic is the whole finding:
  `P(inside the closed VIX3M-floor OR SKEW-rank cells | ratio at a 95th
  percentile)` is **1.000 at day level (590 of 590) and 1.000 at episode level
  (95 of 95)**, and tightening the ratio rung makes it MORE redundant (0.989 at
  the 98th). The ratio moves on its denominator: corr(dlog ratio, -dlog VIX3M)
  **0.895** against corr(dlog ratio, dlog SKEW) 0.549, with VIX3M's daily sd
  1.9x SKEW's. The conjunction subtracts (SKEW alone +0.268pp of excess at
  sign p 0.035, the ratio -0.011pp, the VIX3M leg alone -0.090pp) and the
  ladder inverts exactly as the 2026-08-27 VIX3M floor did. Mechanism
  falsified in-window in BOTH directions: across the hold ^SKEW falls -0.251%
  but **^VIX RISES 3.816%** and ^VIX3M +1.956%, so the ratio reverts because
  the denominator rises, not because tail premium decays. Also out of sample:
  the live reading is the 98.4th trailing-252 percentile but only the **80.7th
  of full history**, the 2026-08-14 SKEW-median-drift trap live, and on the
  full-history basis the mechanism needs, today does not trigger at any rung.
  (b3_c5_overlap.py, b3_c5_battery.py, b3b_c5_livecell.py)
- **The dollar CONFIRMING a rate rise, the untested inversion of the parked
  unconfirmed form.** The premise is false on the live tape and that is the
  kill: the "rate rise" is **+5.7 bp over 21 sessions, the 6.9th percentile of
  1,001 trigger days** against a trigger-day median of +24.5 bp, and the dollar
  leg reads a 21-day rank of 42.9 against the rule's 65 floor. The ten-year is
  at a 52-week high **by level only**. Where the rule does fire the
  confirmation leg adds **+0.016pp at t +0.21** over the dollar alone, 45% of
  the total sits in two late-2008 episodes, the record is 136-145 at sign
  p 0.725, and the grid charge over 320 cells gives P 0.807. Restated to the
  state that actually fires today the sign INVERTS to **-0.536% at t -2.20**
  (32.4% hit, bootstrap 0.991), with short EURUSD, long USDJPY and the dollar
  ETF all agreeing. The offset ladder makes it a lagging label: **k=-5 pays
  +1.371% at a 100.0% hit, t 12.40** against the true anchor's +0.055%.
  No threshold arms it, so nothing is parked. (b3_c14_r1.py, b3b_c14_r2.py)
- **Pre-print drift in a deeply lagging mega-cap, the first use of the earnings
  calendar as a pitch anchor.** The pitched conditioner is what kills it: the
  "deeply lagging" gate is worth **-1.867pp on the very name pitched**, taking
  its pre-print session from +0.513% on 33-22 ungated to **-1.354% gated**, and
  the gate ladder is monotone against the pitch (<=5 -1.354% / <=10 -0.407% /
  <=25 -0.441% / **>25 +0.779%**), with today's reading on the worst rung. The
  horizon ladder falsifies the mechanism outright: holds beyond two sessions
  run -0.56pp at 3 td, -2.01pp at 5 td and **-4.52pp at 9 td**, so the lagging
  name keeps FALLING into its print and there is no run-up, only a 1-2 session
  tail. Reference class closes both forms -- gated, the name cannot even enter
  the 535-name class; ungated it ranks **116 of 934 at a permutation P of
  1.0000** against a near-homogeneous class (common excess +0.145pp, I-squared
  16.5%). Era flips pooled (+0.190pp at t 3.31 pre-2018 to **+0.002pp at t
  0.04** from 2018) at 1.49x cost, and liquid large caps specifically are
  **0.44x cost**. Half the raw number is beta (pooled beta 1.057, and the gate
  over-selects up-tape). Survivorship note: 97.7% of the calendar's tickers
  still report in 2026 and the price cache holds today's universe only, so a
  cell selecting names that just fell 17% in 63 days is exactly where the
  missing delistings sit -- the common excess is an UPPER BOUND.
  (b3_c3_preprint_r1.py, b3b_c3_preprint_r2.py, b3b_c3_ungated_refclass.py)
- **High yield at a 52-week high while the SMALL-CAP index sits below its
  own**, the depth-substituted re-ask of the 2026-08-26 credit kill. Dead as a
  re-skin (see the depth-band entry above) and independently on its reference
  class: six indices in the depth slot give a fixed-effect common excess of
  +0.143pp with **Cochran Q 4.32 on 5 df, I-squared 0.0%**, a cross-sectional
  sd of 0.152pp against a mean sampling SE of 0.162pp (**ratio 0.94**, so the
  whole spread is sampling noise), IWM ranking **5 of 6**, and a random-date
  max-of-6 **P of 0.954** -- the left tail of a null, not the right tail. The
  dial split finishes it: the entire edge is in dial [0,30) (+0.311 / +0.658 /
  +1.570%) while [50,70) pays **-0.453 / -0.794 / -1.749%** and the live
  dial>=80 slice is 2 episodes. It is a calm-tape effect and this tape is the
  opposite. (b2f_c4_hyg_high_iwm_depth.py, b2g_c4_freshhigh_repro_and_dial.py)
- **The small-cap laggard into the month turn, long IWM against short SPY.**
  A join that subtracts at every rung, on a parent that is already negative:
  the beta-neutral pair at ME-0 pays -0.038 / -0.108 / -0.170 / -0.293% at
  h=1/3/5/10 against its own all-days drift of -0.007 / -0.020 / -0.032 /
  -0.063%, so the anchor makes the pair WORSE at every horizon and the ladder
  ranks the true anchor **8 of 9**. The short leg is the better leg (SPY
  outright beats IWM outright at every horizon; IWM at ME-0 is below its own
  all-days drift), and short-leg attribution is negative at every rung and
  horizon (-0.116 to -0.284pp). Threshold-mined around the live reading: the
  h=5 gate ladder runs r5<20 **-0.200% on 9-18**, r5<30 +0.021%, r5<40 -0.039%,
  and the complement (IWM LEADING, r5>70) beats it. Confirmed NOT a re-skin of
  the closed index-pair cell (residual correlation +0.019 to +0.052), so it is
  a genuinely new object that dies on its own. Per the 2026-08-28 rule, a
  negative interaction is not parkable. (b1c_c12_iwm_spy_me0.py)
- **Long duration with the ten-year at a 52-week yield high and bond vol
  compressed.** The pitched conjunction beats neither parent; the surviving
  mid-range band is parked with an episode count. See the grid-charge and
  inverted-U entries above.
- **Short duration into September with the ten-year at a 52-week yield high.**
  The interaction cell is empty (one observation, the live one). The bare
  September parent fails on cost once short carry of ~1.79 bp/session is
  charged (**0.53x at h=5, 2.27x at h=10, -0.91x at h=21**) and on its own
  twelve-month scan (August ranks 4 of 12, **max-of-12 P 0.997**), with the top
  three episodes at 107% of total, 2021+ at -0.872% on a 40% hit, and the
  bond-bull objection INVERTED (falling-yield years +0.764% against
  rising-yield +0.551%). Reproduced the registry's month-of-year table exactly,
  with one correction: September is TLT's **third**-worst month on those
  numbers, not the second as the 2026-08-13 entry states (Oct -0.432%, Apr
  -0.240%, Sep -0.220%). (b2d_c8_september_duration.py)

### Calendar finding, filed because it is now six consecutive sessions

- **A sixth straight empty anchor set, and the month turn arrived and closed
  its sixth form the day it became reachable.** Jackson Hole moved to JH-1 and
  is closed on eight classes pre-speech and ten post; NFP at +4 td, PPI at +7
  and CPI at +8 are all closed on their own ladders; FOMC and VIX expiry at
  +11 and opex and quad witching at +13 remain beyond the 10 td cap. The month
  turn was the only live anchor, and its one never-measured return -- the
  overnight -- was measured today on daily bars AND at 15-minute resolution,
  and closed both ways. **The month-end anchor is now closed on six forms
  across five asset classes.** The next genuinely new anchor is still the
  September FOMC on 2026-09-16, which enters the horizon around 2026-09-02 and
  is spoken for by the event sleeve's T1/T2 -- and the midterm T2 short is
  gated on SPY's 21-day rank being under 50, which reads 91.3, so the sleeve's
  own rule remains off. The practical consequence is unchanged and now six
  sessions old: a price-state sweep is the only honest search mode. Today it
  produced twenty-two candidates across ten asset classes and no survivor, and
  the closest of them died on a lag profile that no other test in the battery
  would have caught.

## Method traps (2026-09-01, from a 12-candidate sweep that killed all 12)

- **Charge the grid the TRADED RUNG lives in, not the grid the original
  pre-registration declared.** The parked duration-neutral flattener declared
  `HS = (1,2,3,5,10)` and permuted 3 vehicles x 2 signs x 5 horizons to reach
  P 0.018 -- and then parked its arm on **h=8, which is not in that grid**.
  Charging the walk actually disclosed (3 vehicles x h=1..10 x 6 proximity
  rungs = 180 cells, a floor, since the 6 lookbacks make it 1080) gives the
  shipped cell **P 0.388** and the grid max **P 0.144**. A pre-registration
  does not immunise a cell that was later read off a horizon the
  pre-registration never charged. Re-read every parked arm against the grid
  its own headline number sits in before treating the park as protection.
  (a1_r2c_verdict.py)
- **A cost arm can CLEAR and the cell still die, so resolve the arm and keep
  going.** The flattener's stated arm was a two-leg round trip under 4.4 bps
  and the honest answer is **3.59 bps (6.18x)** on a half-spread MOC basis,
  **4.42 bps (5.01x)** with 0.50%/yr borrow on the short leg -- only the
  full-cross bound (5.31 bps) fails, and an MOC does not cross. Both stated
  blockers cleared (Jackson Hole passed on 2026-08-28) and the morning still
  ended in a kill. Recorded because the temptation on an armed watchlist entry
  is to treat the arm as the verdict.
- **^TNX is quoted in PERCENT, so an index point change x100 is basis points.**
  Two recon scripts multiplied by 10 and printed every yield change **10x low**,
  which produced the false framing "a one-year high at the top of a year of
  nothing" (the year did **+55.1 bp**, not +5.5) and was the entire premise of
  one of the twelve candidates. Caught in round 2, not round 1. Any rates cell
  that reasons about MAGNITUDE owes a units check against a known move before
  the statistics run.
- **The live magnitude bucket, not the pooled mean, is today's expectation.**
  The flattener's live 252-session yield change of **+55.1 bp sits below its
  episode median of +77.6**, and that low half pays **5.0 bps against 22.2**
  pooled = 1.38x cost. This is the 2026-08-07 "rank gates in a quiet tape buy a
  fraction of the historical force" trap arriving through a LEVEL trigger
  rather than a rank one: the trigger fired at 100.00% of the trailing-252
  maximum under both conventions and still bought a below-median thrust.
- **A class-level "inversion" read off a coarse map is a family statistic
  until proven otherwise.** The 15-class pre-FOMC map showed a near-universal
  midterm inversion with energy as the lone exception, which reads like an
  `inversion`-axis object. It is not: the cross-sectional spread is **1.03x its
  own sampling noise** (sd 0.717pp against a mean sampling SE of 0.699pp),
  Cochran Q p 0.8138, I-squared 0.0%, and the correlation-preserving
  permutation max-of-15 gives P 0.2447. The exception itself then lost its sign
  to drop-best-2 (+0.827pp -> **-0.493pp**, both top episodes from 2026) and
  ranked 8 of 12 on its own placebo ladder. (b1b_family_common_excess_r2.py,
  b2_crude_midterm_fomc_r1.py)
- **An inverse-variance common excess is not the family's answer.** The
  pre-FOMC family's fixed-effect common excess of **-0.274pp at z -2.51** is an
  artifact of up-weighting the low-volatility rates, dollar and credit legs;
  **equal-weighted it is -0.004pp** at a two-sided permutation P of 0.3754.
  Report both, or the weighting invents an effect the family does not have.
- **Two gate legs can each be positive alone and negative together, and the
  discarded complement can beat both.** Silver's post-parabolic cell: the
  drawdown leg alone +1.077%, the up-on-the-year leg alone +1.118%, the pitched
  conjunction **-1.032%**, and the OPPOSITE year leg (drawdown with a negative
  trailing year) **+1.563%**. "Still up huge on the year" cost 2.6pp against
  its own complement. A conjunction owes both single-leg cells AND the
  complement of each leg. (02_c1_silver_postparabolic.py)
- **Check which era the live half is in before reading an era split as
  reassurance.** The same silver cell is pre-2018 +7.676% on 4-for-4 and 2018+
  **-6.838% on 1-for-6**, and the 2018+ half IS 2026 -- 82% of all trigger days
  in history are the currently live episode. The cell has been paying -6.8% per
  10 td for the entire time it has been live.
- **A continuous futures series' seasonal edge can be its ROLL's seasonality.**
  NG=F prints **+0.1367 pp/day** more than UNG over 4,872 common sessions at
  0.850 return correlation, and the wedge is itself seasonal at **+0.3601
  pp/day in September**, which fully accounts for the contract's apparent
  September edge (+4.724% against the fund's +0.532% at h=10). Confirmed by an
  overnight-gap-by-day-of-month profile spiking to +1.33% on day 29. Before
  claiming a futures expression escapes an ETF's roll decay, measure the wedge
  and then measure it BY MONTH.
- **The placebo offset ladder is now 16-for-16**, adding four failures in one
  morning (crude, energy equity, the midterm index short, and the 15-class
  family average). In three of the four the extremum sat AFTER the decision
  date, which is the signature of a slow regime ramp wearing an event label.
- **The two z10 conventions in this repo disagree materially and a thrust cell
  must be measured on the one it was SELECTED on.** `build_pitch_state._metrics_for`
  (r10 over 21d vol scaled to 10d) read UNG at **+1.34** while `pitch_lab.zscore`
  (252d standardised) read **+0.713**, correlation 0.924. The natgas candidate
  was re-run under both so the kill could not be dismissed as the wrong cell.
  Same verdict either way, but the discrepancy is now on the record.

## Cells swept and empty (2026-09-01)

- **The pre-FOMC window across FIFTEEN asset classes, which closes the anchor
  cross-asset.** This is the anchor the registry had been naming as "the next
  genuinely new one" for six consecutive sessions, and both event-sleeve trades
  were off by their own gates (T1 is non-midterm only; T2 needs SPY's 21-day
  rank under 50 against a live 67.5), so the window was genuinely unclaimed and
  had never been swept off SPY. It is a homogeneous family with nothing
  distinguishable in it: **Cochran Q 9.26 on 14 df, p 0.8138, I-squared 0.0%**,
  cross-sectional sd 0.717pp against a mean sampling SE of 0.699pp (**ratio
  1.03**), permutation max-of-15 **P 0.2447** for the strongest class and a
  median per-class charge of 0.667. Equal-weight common excess **-0.004pp**
  (permutation P 0.3754). The placebo ladder over k=-20..0 ranks the true
  anchor **11 of 21** with the trough at **k=-6**, four sessions AFTER the
  decision. Declustering verified rather than assumed: consecutive entry gaps
  run min 23 / median 30 td against a 10 td window, **0 overlapping pairs of
  211**. (b1_fomc_family_r1.py, b1b_family_common_excess_r2.py)
- **Short the index at FOMC-10td in a midterm year**, i.e. the window the event
  sleeve's T2 declines. Overlap measured rather than asserted: leg correlation
  **0.679**, T2 owns 4 of the 10 held sessions, and the T2-free portion alone is
  **+0.079% on 27-26 at sign p 0.5000**. It dies because the live rung is
  wrong-signed -- the rank21 ladder pays +0.025% under 50, **-0.068% above 65
  on a 41.2% hit** and -0.535% above 80, with the live lag-1 input at 67.5 --
  which independently reproduces the sleeve's own frozen prereg cross-check at
  a different offset. Two episodes are **75% of the +23.49pp total** and
  drop-best-3 takes the short to -0.030%; pre-2018 -0.138% against 2018+
  +1.328%, so it is two bear years rather than midterms.
  (b3_spy_short_midterm_fomc_r1.py)
- **The FOMC decision that is ALSO the VIX settle, as a flow coincidence.**
  A third form of two already-dead parents and dead on its own terms. The
  "coincidence" is a calendar identity: **41 of 42 coincident decisions are
  March/June/September/December at trading-day-of-month 11-16**, the same
  mid-month confound that killed VIX-expiry-week drift. The 14:00 placebo
  falsifies the settle mechanism in its own window, since VRO settles off SPX
  opening prints and the coincidence **SUBTRACTS from both halves** of the
  session (overnight +9.1 bps against 17.7 for FOMC-only; intraday +0.2 against
  8.5). On the vol complex it runs outright the wrong way: SVXY **-0.121pp
  coincident against +1.904pp otherwise**. (b4_fomc_vix_expiry_coincidence_r1.py)
- **Long crude and energy equity into a midterm-year FOMC**, the map's lone
  counter-inversion. Placebo rank 8 of 12 (crude) and 10 of 12 (energy equity)
  with k=-14 paying 2.4x the anchor; concentration **157% of total in two 2026
  episodes**; nine-vehicle reference class at Cochran Q p 0.9892 with a spread
  **0.41x** its own sampling noise (the left tail of a null) and permutation
  P 0.2220 for the class max. The live entry state is the negative half: on
  index-down entry days crude pays **-0.458% on n=17** against +1.670% on up
  days. Vehicle note for reuse: **USO's unconditional 10-td drift is +0.9 bps
  against XLE's +47.5**, so "crude is the worst class into an FOMC on the full
  sample" is mostly wrapper rather than information.
- **Duration into an FOMC conditioned on the ten-year at a trailing-252
  maximum.** Not dead on its count -- a loosened, monotone form exists -- but
  dead on gate attribution applied to the CALENDAR leg: all 192 pre-FOMC
  anchors on the flattener pay **+0.017pp at t +0.38** over all days, so there
  is no pre-FOMC flattener effect to condition. Inside the flattener's own
  trigger the FOMC runs the wrong way at the traded horizon (**-0.320pp**,
  FOMC-in-hold against FOMC-out), and the candidate's 6 anchors are **6-for-6 a
  subset** of the parent's 183 trigger days. (a2b_fomc_gate_attribution.py)
- **A trailing-252 yield maximum reached with no rate thrust behind it.** The
  object does not exist: across 3 cuts x 2 vehicles x 2 horizons, all **24
  no-thrust-versus-with-thrust Welch t values lie in [-1.08, +1.37]**. The one
  standalone-alive cell is the flattener's own bottom-quintile sub-cell under a
  third label, measuring the same 10-12 observation population, own difference
  test +0.184pp at t +0.76. (a3_yield_high_no_thrust.py)
- **Bond volatility popping while equity volatility falls, at a yield level
  extreme.** The premise was FALSE on the live tape -- ^VIX rose 3.40% the same
  session ^MOVE rose 6.13% -- and the surface map had copied its own recon
  wrong. Granting it anyway, all eight vehicle-by-horizon attribution t values
  sit in **[-0.58, +0.75]** and today's ^MOVE move is the **48th percentile** of
  the interaction's own 25-episode support. (a4_movevix_at_level_extreme.py)
- **Long silver deep in a post-parabolic drawdown while still up on the year.**
  See the two-legs-anti-filter and live-era entries above. Reference class puts
  silver **last in its own family** (-1.538pp against a pooled +1.469pp at
  h=10, Welch t -0.98), so the depth band is not transplantable and silver does
  not even earn it. Cost at h=10 is **-12.9x**.
- **Energy closing at a fresh 52-week high on a session the index FELL.** The
  only candidate of the twelve to survive round 1 (+0.547% at h=5 against the
  up-gate's -0.008%) and the closest thing to a trade today. Killed by an
  inverted dose -- binned by the index's same-day move the **deepest down-bin
  is the worst cell in the table at -1.609%**, Spearman -0.066 over 435 at-high
  days, and today's -0.30% lands in the best INTERIOR bin -- plus definition
  fragility (within 1% of the high +0.146%, within 2% **-0.023%**) and a
  nine-sector family P of **0.5144** with XLE ranking 3 of 9. The h=21 boundary
  was settled separately: family P re-seeds at **0.0510-0.0616, never under
  0.05**, and ex-2022/2026 the excess collapses +0.762pp -> +0.264pp. Energy
  effective names **1.42 of 4 (PC1 83.0%)**. Parked as a near-miss.
- **The count of sectors simultaneously in the bottom fifteenth of their
  five-day rank.** Spearman between count and forward index return is **+0.034**
  (11-sector) and **+0.027** (9-sector); the live count of 5 is an interior
  bucket at **-0.043%** at h=21 while counts of 3 and 4 pay +2.265% and +2.109%;
  the gate is index drift with a breadth label at +0.201pp / Welch t +0.38.
  Note the literal 11-sector definition only has 2018+ history (XLC inception
  2018-06). (01_c3_sector_washout_count.py)
- **Natural gas thrusting into the September shoulder season**, which is a
  DIFFERENT state from the registry's closed "UNG long at a 52-week low" and
  dies anyway. Absolutely negative at today's depth (**-1.113% / -0.202% /
  -2.259%** at h=5/10/21 inside a -0.887%/10td bleed) while positive in excess,
  the thrust leg is a SHORT signal (**-1.256% at h=5, -3.074% at h=21** over 124
  episodes on 48-76 and 47-77), and the seasonal leg does not extend (September
  alone +1.051% against the conjunction's -0.120%; extending to Sep+Oct flips
  h=21 from +0.508% to -2.616%). Futures escape closed by the roll wedge above.

### Calendar finding, filed because the six-session prediction came true and closed

- **The anchor the registry had been pointing at for six consecutive sessions
  arrived, and it closed on the day it became reachable.** The September FOMC
  crossed inside the 10 td cap on 2026-09-01, carrying a VIX expiry on the same
  date, and both were swept: the pre-FOMC window on **fifteen asset classes**
  plus a nine-vehicle energy reference class, and the settle coincidence as a
  third form of two dead parents. The registry's standing note that the
  pre-FOMC drift was "already spoken for by the event sleeve's T1/T2" turned
  out to be the 2026-08-07 blind spot in miniature -- both sleeve trades were
  OFF by their own gates, so nothing was spoken for, and the cross-asset cells
  had never been opened. They are open and empty now. **What remains reachable
  is opex and quad witching on 2026-09-18, which enter the horizon around
  2026-09-04**, and September post-opex is the event sleeve's T3 territory with
  the registry already recording that September inverts the post-opex vol
  crush. The practical consequence is now seven sessions old and unchanged: a
  price-state sweep is the only honest search mode, and today it produced
  twelve candidates across nine asset classes and no survivor.

## 2026-09-02

- **The ATR conjunction on the crude thrust band, as a filter that does not
  filter.** Watchlist 4's arm fired on every stated leg for the first time
  since it was parked (USO +5.460% into [5,6), 1.65 ATR, no CPI/PPI in a
  three-session hold) and the parked cell reproduced to the decimal at 38
  episodes, +1.105pp excess, 73.7% hit, sign p 0.0025. The `>=1.50 ATR` leg,
  which was the whole reason the entry was an ARM rather than a state, moves
  the excess to **+1.121pp while discarding 22 of 38 episodes whose own excess
  is +1.094pp** - 1.6 bps of movement for 58% sample removal. The lesson
  generalises past this cell: **an arm condition minted from a bucket sweep has
  to be re-attributed when it fires, because a threshold chosen to separate a
  sample does not have to separate the effect.** Three further kills on the
  same cell: the bucket ladder is an interior spike with [4,5) at -0.186pp and
  a 4.8% lower edge halving it to +0.685pp; the top three of sixteen armed
  episodes are 72% of the total with 2026 wrong-signed at -0.451%; and net of a
  re-estimated 0.506 crude beta the residual is a 62.5% hit at sign p 0.227,
  so the registry's 2026-08-11 closure survives the band intact. The one attack
  that FAILED is worth recording so it is not re-run: a payrolls print inside
  the hold is +1.973% (n=7) against +1.088% out, Welch t +0.64, i.e. NFP is not
  the CPI/PPI containment effect. (c1_energy_band_r1.py, c1b_energy_band_r2.py)
- **The index-near-a-high gate is a bull-tape selector, now confirmed three
  times.** In the sector triple-rank-floor cell it SUBTRACTS 0.733pp (bare
  +1.646% on 65 episodes at t 2.236 -> gated +0.913% on 11), and 100.0% of its
  20 trigger days sit above SPY's 200-day against a 71.6% base with **zero**
  observations below. In the defense washout its apparent +1.595pp dose is the
  same thing wearing a positive sign, since the episodes it selects are ones
  where SPY itself runs +1.201%. Any future candidate whose novelty is "while
  the index holds near its high" starts from a negative prior and owes the
  above/below-200d base-rate split up front. (c5_xli_triple_floor_r1.py,
  c6_defense_washout_r1.py, and the 2026-08-25 watchlist-21 precedent)
- **"The range has been dead, then it broke" is monotone in the WRONG
  direction.** Bucketing a >=8% VIX pop by the VIX 21-day range percentile at
  h=10: bottom 5% pays **-0.479pp edge**, [5,15) +0.023pp, [15,30) +0.628pp,
  [30,50) +0.446pp, 50+ +0.209pp. The more compressed the prior range, the
  worse the subsequent index return, which falsifies the compression-release
  story inside its own window. The cell is also negative against every control
  it has (-0.239pp vs compression alone, -0.475pp vs the pop alone, -0.448pp vs
  local) at every horizon h=1..10. Distinct from watchlist 12 by construction -
  day-level Jaccard **0.008**, one shared day of a 120-day union - so this is a
  second, independent road to the same place. (c7_vix_range_pop_r1.py)
- **A conditioner that inverts its parent, on the metals complex break.** The
  parabolic-run gate (miner 21d rank >= 90) on the three-name break pays
  +0.185% on 4-8 against the bare parent's +0.454% on 87-90 and the DISCARDS'
  +0.521%, i.e. -0.270pp at h=1 and -1.344pp at h=5, monotone as the gate
  tightens (>=80 +0.435%, >=90 +0.185%, >=95 -0.162%). This closes the
  parabolic route to paying watchlist 29's lag-profile debt: the gated cell's
  lag profile is lag=0 -0.297%, lag=1 +0.185%, lag=2 -0.420%, still one session
  wide, still starting late, and smaller than the parent's. (a1_c3_slv_parabolic_break.py)
- **Miner-versus-metal, entered from the break side rather than the thrust
  side, fails the same two ways.** Beta-neutral at h=3 the long miner leg is
  +1.711pp and the short metal leg -1.365pp of a +0.458pp spread, so the long
  side is 495% of it; permutation across twelve miner/metal pairs gives
  family-wise P 1.0000 / 0.9947 / 0.8452 at h=1/3/5. The outright long miner
  form is separately a **class effect with dispersion ratio BELOW 1 at every
  horizon** (0.89 / 0.82 / 0.78 across 14 names), so the whole cross-name
  spread is smaller than sampling noise. (a2_c4_gdx_gld_pair.py,
  b1_c4_outright_miner_refclass.py)
- **September weakness does not exist at the month position everybody assumes
  it does.** Anchored at trading day 1, ^GSPC September pays -0.042% at h=3 on
  15-11, an excess of **-0.080pp** over all other months, and the midterm
  crossing is POSITIVE at +1.344% on 4-2 - wrong-signed for any short. Over the
  48-cell month x cycle grid the cell ranks 43 of 48 from the negative end with
  P(min-of-48 <= observed) = 1.0000. Any future calendar-month candidate owes
  the grid permutation before the anecdote. (a4_c11_september_midterm.py)
- **The repo holds two co-resident z10 definitions and they disagree about
  which states are live.** `build_pitch_state._metrics_for` (ret10 / vol21 x
  sqrt(10)) scored EWZ at **+2.03**, the tape's loudest extreme and the reason
  an international candidate was selected at all; `pitch_lab.zscore` scores the
  same name at **+1.46**, under which the candidate's state was never live.
  CLAUDE.md already records the drift for the context engine; this is the first
  time it decided a pitch candidate's existence. Print BOTH when a z10 extreme
  is the reason a candidate was chosen. (a3_c10_ewz_em_floor.py)
- **A commodity index at a fresh 52-week high into a CPI print.** The placebo
  anchor ladder is now four-for-four as a killer: sliding the at-a-high anchor
  k=-8..+8 from the print, the live k=+6 configuration pays +0.782pp while
  k=-7 pays **+3.365pp (5-0)** and k=+7 pays +2.921pp. The parent is dead
  underneath anyway - 90 episodes at +0.332% against a local +/-126td control
  of +0.224%, Welch t +0.78, top-2 episodes 60% of the total - and the
  conditioner flips sign across the vehicle class (XLE +0.992 CPI-in, GLD
  -0.312, SLV -0.049). (c8_dbc_high_cpi_r1.py)
- **Crude itself is the wrong vehicle for a crude-thrust follow-through.** On
  the same 38 band episodes USO pays +0.581% at a **47.4% hit, below a coin**,
  against the producers' +1.251% at 73.7%, and at h=1 crude is -0.576% at a
  31.25% hit because it mean-reverts the session after the pop. Measured roll
  decay is **-7.75pp/yr against the front contract over 20.4 years**. Where a
  producer/crude choice arises, XOP dominated XLE on both trigger sets
  (+1.749pp vs +1.121pp armed) - but the four-vehicle reference class is
  homogeneous, which is itself the confirmation that the whole object is a
  crude-complex move rather than producer alpha. (c2_vehicle_translation.py)

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

## 2026-09-04 (a 12-candidate sweep, all 7 checked candidates killed, stand-down)

- **A left-open threshold (`<= X`) is a CLAIM THAT THE EFFECT IS MONOTONE UP TO
  X, and the band the LIVE reading falls in has to be quoted on its own.** This
  is the second consecutive morning the trap fired and it is now the house
  rule. Post-NFP TLT conditioned on the PRIOR print's surprise looked like a
  survivor at `<= -50k` (h=3 +0.207%, n=31, gate attribution clean at parent
  -0.013% / gated +0.207% / discarded complement -0.066%). Decomposed: the
  moderate half `(-100,-50]` pays **+0.813% on 12** and the half today actually
  sits in, `<= -100k`, pays **-0.175% on 19 at 10-9**. The dose response runs
  BACKWARDS to the stated mechanism, so a bigger miss is a worse trade.
  Yesterday's bimodal VIX-range finding said the same thing about a `<= 15`
  arm. Decompose before believing, and never quote a threshold cell as today's
  expectation without locating today inside it.
  (b1_c3_nfp_prior_surprise_duration.py, b1b_c3_live_band_and_concentration.py)
- **`data/macro_release_history.parquet` exists, is 42,489 US rows of
  actual/consensus/surprise from 2013, and had never been opened by this
  product. Check the newest row PER SERIES, not per file.** The file is frozen
  at 2026-08-07. Monthly series survive that: NFP's last print (2026-08-07,
  -23k against +80k, surprise **-103k**) predates the cutoff and is readable
  today. Weekly series do not: **CFTC COT speculative net positions are five
  releases and 24 sessions stale**, and the staleness was priced rather than
  asserted, since the 104-week percentile moves a median 11.5 points (gold)
  over five releases at a p90 of 40.7, and the `>= 90` state flipped across
  such a gap **20 times in 39 hi readings**. A conditioner that cannot be read
  on the morning of the trade is a kill, and an unstable one is a kill twice.
  (b2_c4_cot_metals_positioning.py)
- **COT positioning is not a flow instrument on this data: both tails are
  long-positive, which means it is reading drift.** Gold above its 90th
  positioning percentile moves GLD h=5 by three basis points (parent +0.378%
  n=294, gated +0.521% n=32, discarded complement +0.491% n=203), and on GDX
  the gate is WORSE than its complement (+0.591% against +0.879%). The
  forced-unwind mechanism needs extreme length to be bearish; instead the
  crowded-SHORT tail pays as much or more (GDX at pctile `<= 15` gives
  +1.744%). The 32 episodes are one episode: 28 consecutive weeks from
  2024-04-05 to 2024-11-05, and dropping 2024 leaves four. The skill's honesty
  note about this repo lacking positioning data should now read "positioning
  data exists, is stale weekly, and did not survive gate attribution."
  (b2_c4_cot_metals_positioning.py)
- **The Dispersion signal does NOT invert into a short-correlation trade, and
  the book's fragility reading of it is the correct one.** Measured on the
  unlevered instrument, short ^VIX from the trigger pays **-6.62% at h=10 over
  104 episodes against ^VIX's own -1.82% drift, an excess of -4.81pp at Welch
  t -2.23**, same sign at h=5 and h=21. Index vol gets MORE expensive to
  realize after extreme dispersion, not less. On SVXY the cell is below the
  vehicle's own drift at every horizon and degrades monotonically toward
  today's reading (>85 +0.293%, >90 +0.290%, >92 -0.576%). Today's exact
  conjunction is the table's worst cell: dispersion with the dial `>= 50` pays
  **-4.17% at h=10 over 62 episodes at a 29.0% hit** against +1.78% with the
  dial below 50. The seven-signal reference class is homogeneous (Cochran Q
  1.49 on 7 df, I-squared 0.0%) with Dispersion 6 of 8. Definition fragility is
  severe: under a 252-day-lookback composite the signal does not fire at all
  today, reading 79.6. (c5_dispersion_short_vol.py, c5b_blockers.py)
- **A market HOLIDAY is a real anchor and its sign is the opposite of the folk
  trade.** First use of a closure anchor in this repo, derived from
  `master_prices` index gaps because there is no holiday list here. ^VIX RISES
  **+4.80% across a >= 3 calendar-day closure over 180 gaps at 136-44, sign p
  0.0000**, against +2.19% across an ordinary weekend and -0.28% on a plain
  overnight, monotone in the extra calendar day. The short-vol-into-the-long-
  weekend trade is therefore wrong-signed by about three VIX points, and SVXY
  across the closure pays -0.272% against +0.291% across a weekend. The
  mark-down is taken INTRADAY on the eve (short ^VIX on the eve session
  +1.257%, n=143, t 3.07) and reverses across the gap, so an eve MOC is one
  session late. (a1_svxy_closure.py, a1b_svxy_gap_decomp.py)
- **The Labor Day long is 0-for-8 since 2018 and the post-Labor-Day short is a
  fixed calendar date wearing a holiday label.** Long SPY entered MOC on the
  eve pays -0.290% at a 34.6% hit over 26 years, the placebo ladder ranks k=0
  **14 of 17**, gate-off across all 154 closures gives -0.076% so the Labor Day
  gate selects the parent's WORSE half, and 2018+ is **0-for-8 at -0.932%**
  (IWM 0-for-8 at -1.402%, t -4.34). Short from the first post-holiday close:
  SPY h=7 -0.551% against a FIXED September trading-day-4 anchor at -0.528%,
  IWM -0.617% against -0.617%, identical to three decimals, and the whole
  number is 2001's 9/11 week at +10.11% on SPY. The folk claim that September
  weakness begins after Labor Day fails its own calendar test on IWM, where
  forward-10 after the holiday is +0.713% against -1.452% before.
  (a2_labor_day_index.py, a2b_gate_attrib_and_inversion.py)
- **September quad witching is an FOMC anchor in costume.** The ungated run-in
  splits on whether an FOMC decision lands inside the window: **+2.382% over 11
  years at a 90.9% hit and t 3.50 with one, -0.834% over 15 at 46.7% without.**
  Every year has a quad and only 42% carry an FOMC. The laggard gate on the
  IWM/SPY pair is worth **+0.006pp** (+0.446% to +0.452%), and the same 63-day
  floor applied year-round pays -0.266% over 160 episodes at a 45.0% hit, which
  reproduces the 2026-08-31 kill exactly. The reference class over 16 index and
  industry ETFs is homogeneous (Q 13.50 on 15 df, I-squared 0.0%) with IWM 6 of
  16. Book overlap was checked and does NOT stick: 152 of 4,701 ledger rows
  in-window is 3.23% against a 3.49% calendar share, 0.93x.
  (c6_iwm_into_sep_quad.py, c6b_gate_vs_anchor.py)
- **The Labor Day driving-season boundary does nothing to energy, and the
  seasonal cannot be separated from the momentum state it arrives with.** Edge
  against each vehicle's own drift at h=10: USO -0.085pp, XLE -0.382pp, XOP
  +0.033pp, VLO -1.352pp, with sign tests undecided in BOTH directions on both
  crude vehicles. The placebo ladder ranks the true anchor 5, 6, 5 and 11 of
  17. The reference class over 7 energy plus 5 non-energy vehicles is
  homogeneous (Q 7.64 on 11 df, I-squared 0.0%, pooled +0.075%) and its only
  positive member is registry-dead UNG. Crossed with the live state, a
  pre-holiday 21-day rank at or above 80, XOP has **zero prior observations**
  against today's 94.8, XLE two averaging -1.42% and VLO four averaging
  -2.74%. (c10_labor_day_energy.py)

### Method finding, filed because it is the reason the morning shipped nothing

The strongest statistic produced all morning was a **corpse-recovered sign
flip** and it was not shipped. Long ^VIX and short the index ACROSS an extended
closure is coherent, monotone in the extra calendar day, era-stable and
strongest in the era we trade (short IWM 2018+ +0.424% at 40-19, sign p 0.0043,
14.1x cost; Labor Day 2018+ 8-for-8 on both vehicles). It surfaced INSIDE the
blockers run against two candidates it is the exact opposite of, so it carries
sign, era and horizon multiple comparisons before any threshold grid was
walked. That is the route the 2026-08-07 entry closed by name, after two such
inversions both died on re-examination. It is parked on the watchlist with a
forward arm instead. **A morning is allowed to end empty while holding a number
it likes; that is what "designed forward, not recovered from a corpse" costs,
and paying it is the point.**
