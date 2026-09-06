# Brief: verify_sizing_d35_ovs_risk_mults (independent money-path verification)

Date: 2026-09-05. Mind session: Codex `/root`.

## Decision and why

D3.5 changes live scanner and ledger OVS sizing and requires an independent
verifier. Try to falsify point-in-time rank classification, the midterm
exemption, liquid/overflow isolation, and especially P2 aggregate-cap parity.

## Files you own

- `artifacts/verify_2026-09-05/d35_ovs_risk_mults/`

All source, tests, docs, data, and builder artifacts are read-only. Report an
exact reproduction for any defect; never patch it.

## Hard rules

Follow section 0 of `docs/plan_2026-09-04.md`. Do not run `daily_scan.py`,
`order_staging.py`, an order runner, or a workflow. Do not write Sheets, R2,
Cloudflare, IBKR, Task Scheduler, or production data. Do not commit, stash,
checkout, push, install dependencies, rename, move, or delete files.

## Intent

Independently establish all of the following:

1. Only OVS carries the exact rank-mean and tier configs; no field is GRM-scaled.
2. Non-midterm mean 93.999 is 0.7x; 94.0 is 1.0x; every midterm value is 1.0x;
   missing/non-finite rank input is 1.0x and cannot arise from a valid OVS mask.
3. Liquid OVS is 0.5x and overflow OVS 1.0x, on both P1 and P2. The two rules
   compose multiplicatively when both apply.
4. Scanner reads signal-close rank columns and `_scan_source`; engine snapshots
   the same four point-in-time columns and uses the liquid ticker set.
5. Engine's P2 denominator includes cycle, tier and extremity multipliers exactly
   once, while P2 cap dollars are unchanged. Attack days on both sides of the
   aggregate cap and mixed liquid/overflow P2 clusters.
6. Sizing happens before shares and every hard/daily cap. Existing D3.3/D3.4,
   cycle, blackout, gap-path, scale-out, precedence and non-OVS behavior remain.
7. Independently audit all four replay arms, changed-row attribution,
   midterm-zero extremity effect, reconciliation, frozen inputs and no writes.

## Recon first

Write an attack plan to
`artifacts/verify_2026-09-05/d35_ovs_risk_mults/00_attack_plan.md` before probes.

## Verification

- Inspect the complete cumulative diff and isolate D3.5 from verified D3.3+D3.4.
- Run focused tests plus verifier-owned synthetic scanner/engine probes.
- Run the full suite and reproduce any claimed baseline failure on main.
- Audit the frozen four-arm replay without external writes.
- Produce scripted `checks.json` with at least: `verdict`,
  `carrier_sets_exact`, `configs_exact`, `cutoff_strict`, `midterm_exempt`,
  `tier_isolation`, `p1_p2_both_scaled`, `p2_prepass_parity`,
  `scan_engine_parity`, `ordering_exact`, `noncarriers_unchanged`,
  `prior_sizing_invariants_unchanged`, `tests_failed`, `replay_audited`,
  `midterm_extremity_rows_affected`, `unrelated_diff_paths`, and
  `external_writes`.

## Report

Use section 6 of `docs/plan_2026-09-04.md` verbatim. Verdict must be exactly
PASS or FAIL; an unclosed gap is a FAIL.

