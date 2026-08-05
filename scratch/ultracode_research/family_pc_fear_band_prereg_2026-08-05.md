# Pre-registration: P/C-fear exemption to the FAMILY4 frag band

Registered 2026-08-05, BEFORE the decisive study has run. Source: the
equity put/call deep-backfill session (scratch/putcall_dial_study.py,
putcall_5d_trade_sim.py, family_dipbuy_putcall_study.py,
family_dial_pc_interaction.py / family_dial_pc_ttest.py). No config
change ships until every gate below clears and the result is signed off.

## The candidate

The dip-buy family band carriers (FAMILY4: Weak Close Decent Sznls,
SPY QQQ MonFri Reversion, Monday Dip, Indices Oversold Bounce; plus
3x Bear ETF Overbot Fade and Monthly Weak Close) currently run
`frag_risk_bands = [[50, 999, 0.25]]`. Candidate: on signal dates where
the **equity P/C fear state is ON** — trailing-252d percentile of the
10d MA of the CBOE equity put/call ratio (data/cboe_putcall.parquet,
self-maintained nightly) **> 85 as of the signal-date close** — the
band multiplier reverts to **1.0x**. All other dial>=50 days keep 0.25x.
The ONLY pre-named alternative multiplier is 0.5x (if evidence lands
mid); no other value may be adopted.

Mechanism claim being tested: dial>=50 with complacent/neutral
positioning = falling knife (sellers remain); dial>=50 with washed-out
positioning = capitulation, which is the dip a mean-reversion entry
wants. The 2x2 that motivated this (research-recompute dial, 10d MA,
2016-06 -> 2026-05, 495 family trades):

|                | dial<50            | dial>=50               |
|----------------|--------------------|------------------------|
| no fear        | +0.59R, 73% (312)  | **-0.34R, 33% (70)**   |
| fear pct>85    | +0.79R, 78% (94)   | **+0.75R, 89% (19)**   |

The blended dial>=50 cell (-0.105R) reproduces the frag-band study's
hi-frag family figure exactly, so this is the band's own evidence base
re-partitioned, not a different sample.

## Why suspicion is warranted (stated up front, against the candidate)

- The fear-ON hi-frag cell is **19 trades / 11 signal dates / 3
  episodes** (2021, 2022, 2026). Three market moments.
- This split was found POST-HOC while exploring — the entire motivating
  sample is peeked. Date-clustered Welch t = 1.21 (p=.25); only the
  rank-based Mann-Whitney approaches significance (p=.057), and that is
  carried by the 89% win rate on 19 trades.
- The dial used was the RESEARCH RECOMPUTE (rd2_fragility_ts, my own
  10d MA), not the PIT vintage the band was validated on.
- The candidate would RAISE hi-frag exposure — structurally the same
  shape as the throttle-exemptions and >1.0x boosts this book has
  killed repeatedly. The 0.25x band survived its PIT gate at t=-1.96;
  weakening it re-litigates a validated live control on 19 trades.
- The mechanism story is satisfying, which is a reason for MORE
  suspicion, not less (the sector_loss_gate had a good story too).

## Gates (ALL must pass; failing any closes the candidacy negative)

1. **PIT re-bucket is THE evidentiary gate.** Re-bucket every family
   trade on the PIT-reweighted dial series (scratch/pit_reestimate.py
   machinery, vintage Y-1 weights scoring year Y), crossed with the P/C
   fear state (itself PIT-clean by construction: trailing-window
   percentile of as-published data). Required: (a) the no-fear hi-frag
   deficit survives PIT at clustered <= -1.5 sigma (the band's own case
   must still hold where the candidate keeps it); (b) the fear-ON
   hi-frag cell is >= +0.3R date-clustered under PIT vintages;
   (c) sensitivity shown at fear thresholds 80/85/90 x dial 45/50/55 —
   the result must not live on one knife-edge cell.
2. **Out-of-sample accumulation cures the peek.** At least **2 NEW
   hi-frag fear-ON episodes** (signal dates after 2026-08-05, on the
   live PIT dial parquet) must accrue with the split holding
   directionally (fear-ON episode avgR > 0 and > the contemporaneous
   no-fear hi-frag avgR) before anything ships. This naturally folds
   into the existing "re-examine FAMILY4 at +20 hi-frag trades (~2029)"
   trigger — same review, one more cut.
3. **Episode LOYO.** Dropping any single episode-year (2021, 2022,
   2026, plus any accrued under gate 2) must leave the fear-ON hi-frag
   cell positive. If one episode carries the sign, the candidacy dies.
4. **Live PIT availability + staleness convention decided BEFORE the
   study.** MEASURED 2026-08-05: the 21:30 UTC nightly scrape does NOT
   capture same-day — commit vintages show each run captures only D-1
   (CBOE's daily page for D populates sometime after ~22:30 UTC; a
   ~11:30 ET D+1 scrape does get D). So at the AM scan (~4:47 ET) the
   committed parquet ends at D-1 relative to a signal-date-D close.
   Consequence: the fear state MUST be pre-specified as **computed on
   data through D-1** (lag-1). The 10d MA + 252d percentile makes the
   one-day difference tiny, and gate 1's study must use the same lag-1
   stamping so live == studied. (Optional later upgrade: an AM capture
   run à la the risk-report correction, only if an overnight test shows
   the page populated by ~4:15 ET; NOT a gate.) Stale/missing rule:
   **fail-CLOSED to the incumbent 0.25x** (never fail-open to full
   size) when the newest row is older than 3 trading days. Add the
   composition order (2b band before/after the fear check) to
   tests/test_frag_risk_bands.py first.
5. **Parity + aligned sites.** Engine (strat_backtester 3b3
   frag_band_mult_at) and scan (daily_scan 2b frag_band_mult) must move
   together with a parity script per scratch/parity_check_frag_bands.py;
   the site fragility adjuster's single-band assumption (fragility.json
   / portfolio.js) budgeted into scope or explicitly excluded in the
   ship note.

## Explicitly out of scope

- Any multiplier beyond the pre-named 1.0x primary / 0.5x fallback.
- Extending the fear condition to OLV or any non-family strategy.
- Using the P/C fear state as a standalone signal, dial input, boost,
  or short anywhere (all graded and rejected this session: dial add,
  complacency shorts, standalone 5d long — see
  scratch/putcall_*_study.py outputs).
- Any re-weighting of the fragility composite itself (freeze policy A2).

## Revision 2 (2026-08-05, same day): McKinley's three-leg form

McKinley reviewed the 2x2 and chose the fuller rule, accepting the
appetite framing for the two legs the evidence does not independently
carry. The candidate becomes a FEAR-SELECTED BAND TABLE — same 2b/3b3
slot as today's band, so composition semantics with every other overlay
are byte-identical to the current machinery:

- fear ON  (lag-1 pct252 > 85):        `[[0, 50, 1.25], [50, 999, 1.0]]`
- fear OFF:                            `[[0, 50, 1.0],  [50, 999, 0.0]]`
- fear STALE/missing (> 3 td):         `[[0, 50, 1.0],  [50, 999, 0.25]]`
  (exactly today's behavior everywhere — the unambiguous fail-closed
  state; a dead P/C feed reproduces the incumbent book).

Per-leg evidentiary status and gates:

- **Leg A — dial>=50 & fear -> 1.0x** (the original candidate,
  exposure-raising): gates 1-3 unchanged, including gate 2's two new
  out-of-sample episodes.
- **Leg B — dial<50 & fear -> 1.25x boost** (APPETITE: weakest cell,
  +0.20R n.s.; three prior >1.0x boosts were killed in this book —
  stated, accepted). Gates: PIT non-inferiority (fear-ON dial<50 cell
  must not be WORSE than no-fear by more than 0.1R clustered), and it
  ships only together with leg A (never a standalone boost). Subject to
  gate 2 like leg A (it raises exposure). Multiplier is frozen at 1.25;
  any creep is a new prereg.
- **Leg C — dial>=50 & no fear -> 0x** (APPETITE tightening of a
  defense; may ship AHEAD of gates on McKinley's word alone, like the
  OLV ladder / gap derate appetite cuts, since it only reduces
  exposure). Hard requirements even so: (a) gate 1a — the no-fear
  deficit must survive the PIT re-bucket before the cell is judged
  permanent; (b) **shadow tracking is mandatory**: the engine books the
  zeroed trades into a counterfactual pass (build_trade_ledger nogate /
  gate_lab pattern) and daily_scan notes zeroed signals in the scan
  email, so the "+20 hi-frag trades" re-exam trigger keeps accruing
  evidence — without this the cell freezes at n=70 forever and the rule
  can never be falsified; (c) the staleness fallback above (0.25x, NOT
  0x) so a feed outage in a crash week cannot silently zero the family.

Pre-named multiplier set is now closed: {1.25, 1.0, 0.25 (stale
incumbent), 0.0}. Nothing else may be adopted under this registration.

## Revision 3 (2026-08-05, same day): SHIPPED ahead of gates

McKinley elected to ship the FULL three-leg rule live immediately
("yea do it ... a note in the signal cards for which it has a say now"),
overriding the gate-before-ship sequencing as an explicit appetite
decision. Recorded, not endorsed-by-evidence: the motivating cell remains
19 trades / 11 dates / 3 episodes, post-hoc. What this revision changes:

- Gates 1-3 convert from SHIP gates to the **post-ship review protocol** —
  same analyses, same thresholds, run when their inputs exist. A FAILED
  gate at review time is a presumptive ROLLBACK of the failing leg (the
  rule reverts to the incumbent 0.25x table), not a shrug.
- Gate 4 is CLOSED: lag-1 state + 3-bd fail-closed staleness + parity are
  implemented and tested (pc_fear.py; tests/test_pc_fear_bands.py,
  tests/test_frag_risk_bands.py state-matched parity).
- Leg-C shadow tracking is BUILT and mandatory to keep:
  build_trade_ledger.build_pcfear_shadow ->
  data/backtest_trades_pcfear_shadow.parquet (pc_fear_enabled=False pass).
  Removing it orphans the review protocol.
- Live surfaces: daily_scan 2b + 3b2, scan-email liveness footnote +
  per-signal Sizing notes; engine 3b3 replays lag-1 PIT so ledger == live.

## Status

- [x] Gate 4 availability check + staleness rule + composition test
      (2026-08-05: measured T+1 capture; lag-1 + fail-closed shipped
      with tests)
- [x] Leg C shadow tracking built (build_pcfear_shadow)
- [x] SHIPPED (rev 3, 2026-08-05) — all three legs live per McKinley
- [ ] Post-ship review, part 1: PIT re-bucket (gate 1 incl. leg B
      non-inferiority + leg C 1a) — runnable now, not yet run
- [ ] Post-ship review, part 2: 2+ new OOS hi-frag fear-ON episodes
      (gate 2) + LOYO on combined sample (gate 3) — folds into the
      FAMILY4 "+20 hi-frag trades (~2029)" re-exam
- [ ] Review outcome recorded here and in CLAUDE.md (rollback on fail)
