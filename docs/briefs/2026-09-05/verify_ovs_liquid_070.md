# Verify brief: liquid OVS 0.7x owner override

Independently verify the target worktree read-only. Do not run the live scanner,
broker executor, Sheets, R2, email, workflows, scheduler, or any deployment.

## Attacks

- Prove OVS alone carries `tier_risk_mults={"Liquid": 0.7}` and Overflow
  resolves to 1.0.
- Prove exact before-cap P1/P2 values for liquid normal, liquid bottom-rank
  non-midterm, liquid midterm, and overflow normal at GRM 1.5.
- Prove tier, rank and cycle each apply exactly once and the rank overlay is
  exempt in midterms.
- Prove both P1 and P2 use 0.7, including a binding mixed-tier P2 cap where
  staged-risk weights change but fixed cap dollars do not.
- Prove scanner and engine ordering and point-in-time inputs are unchanged.
- Prove no non-OVS carrier or unrelated execution policy changed.
- Audit the frozen replay: inputs/hashes, row/position identity, deltas,
  breakouts and reconciliation.
- Run focused tests, then the full suite. Reproduce any pre-existing failure
  on unchanged main before classifying it as baseline.

## Report

Return PASS or FAIL, exact checks, test counts, replay impact, external writes
(must be empty), limitations, and a concise handoff. Verifier artifacts belong
under `artifacts/verify_2026-09-05/ovs_liquid_070/`.
