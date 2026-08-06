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
