# Brief: liquid OVS 0.7x owner override

Date: 2026-09-05

## Decision

McKinley explicitly changed the Overbot Vol Spike liquid-tier multiplier from
0.5 to 0.7. This is an owner risk-appetite decision. It does not replace or
reinterpret the frozen D9/D12 research that originally selected 0.5.

## Exact contract

- OVS is the only `tier_risk_mults` carrier.
- `tier_risk_mults = {"Liquid": 0.7}`; missing tiers, including Overflow,
  resolve to 1.0.
- The multiplier applies once to unrounded risk for both P1 and P2 before
  shares and every hard/daily cap.
- The independent non-midterm rank-mean multiplier remains 0.7 below a strict
  rank mean of 94. Midterm years remain exempt from that overlay.
- The existing midterm multiplier remains 0.75.
- Path 1 remains 40 nominal / 60 effective bps. Path 2 remains 8 nominal / 12
  effective bps. The fixed P2 aggregate cap is not multiplied.
- Scanner and engine order of operations remain unchanged. The P2 pre-pass
  changes only because its staged-risk denominator uses the configured tier
  multiplier.

At GRM 1.5, before caps:

| State | P1 effective bps | P2 effective bps |
|---|---:|---:|
| Liquid normal | 42.0 | 8.4 |
| Liquid non-midterm, rank mean <94 | 29.4 | 5.88 |
| Liquid midterm | 31.5 | 6.3 |
| Overflow normal | 60.0 | 12.0 |

## Owned files

- `strategy_config.py`
- `pages/strat_backtester.py` (descriptive UI text only)
- `tests/test_ovs_risk_mults.py`
- `CLAUDE.md`
- `docs/plan_2026-09-04.md`
- `docs/running_list.md`
- this brief and `verify_ovs_liquid_070.md`

Do not change OVS signals, entry/exit rules, path bps, cap values, earnings
blackout, EOD-DD, scale-out, rank cutoff, cycle behavior, any other strategy,
or the external order runner.

## Proof

1. Focused contract and composition tests, including a binding mixed-tier P2
   cap.
2. Source scan for stale live-facing 0.5 descriptions; frozen historical
   briefs/evidence are excluded and retained.
3. Frozen-input engine replay comparing 0.5 with 0.7, with input hashes,
   position keys, row counts, risk/PnL reconciliation, and liquid/overflow,
   P1/P2, cycle and rank-cell breakouts.
4. Independent money-path verification and full test suite.

The replay measures implementation impact; it is not new evidence that 0.7
has an edge. Historical portfolio outputs remain subject to the documented
survivorship limitation.
