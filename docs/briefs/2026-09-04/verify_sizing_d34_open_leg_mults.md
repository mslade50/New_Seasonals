# Brief: verify_sizing_d34_open_leg_mults (independent money-path verification)

Date: 2026-09-05. Mind session: Codex `/root`.

## Decision and why

D3.4 changes scanner and ledger sizing and therefore requires an independent
verifier. Try to falsify the WCDS/LT Trend live-engine parity, especially the
same-day pre-fill re-key and the difference between an open leg and a working
limit.

## Files you own

- `artifacts/verify_2026-09-05/d34_open_leg_mults/`

All source, tests, docs, data, and builder artifacts are read-only. If a defect
is found, report an exact reproduction; do not patch it.

## Hard rules

Follow section 0 of `docs/plan_2026-09-04.md`. Do not run `daily_scan.py`,
`order_staging.py`, an order runner, or a workflow. Do not write Sheets, R2,
Cloudflare, IBKR, or Task Scheduler. Do not commit, stash, checkout, push,
install dependencies, rename, move, or delete files.

## Intent

Independently establish all of the following:

1. Only WCDS and LT Trend carry the exact 0.8/1.2 config.
2. One staged signal with no prior filled-open leg is 0.8x; two same-tier/day
   staged signals make every row 1.2x; one staged signal with any prior
   same-strategy filled-open leg (different ticker allowed) is 1.2x.
3. Working-but-unfilled candidates never trigger the open-leg branch. A failed
   candidate on a cluster day does not demote the other candidate from 1.2x.
4. Scanner counts same-day candidates per tier while prior open legs are
   strategy-wide. Engine duplicate liquid/overflow passes do not accidentally
   turn each other's one-signal days into a cluster.
5. The multiplier runs before cross-strategy absolute clamps, D3.3 same-day
   derates, gap sizing and daily caps. It composes with WCDS frag/PC state,
   D3.2 base tilt, and LT's overflow sizing without changing those inputs.
6. The dormant ticker-specific ladder behavior and every non-carrier are
   unchanged. No OneDrive or `daily_portfolio_report.py` edit is needed.
7. Independently audit the builder replay's scenario isolation, row keys,
   changed-row reconciliation and reported book/sleeve metrics.

## Recon first

Write an attack plan to
`artifacts/verify_2026-09-05/d34_open_leg_mults/00_attack_plan.md` before probes.

## Verification

- Inspect the complete cumulative diff and isolate the D3.4 paths from the
  already-PASS D3.3 change.
- Run focused tests plus independent synthetic engine/scanner-equivalent
  probes written only under the verifier artifact directory.
- Run the full test suite and reproduce any claimed baseline failure on main.
- Audit the full-history replay without external writes.
- Produce `artifacts/verify_2026-09-05/d34_open_leg_mults/checks.json` by script
  with at least: `verdict`, `carrier_set_exact`, `config_exact`, `solo_08`,
  `cluster_all_12`, `prior_open_any_ticker_12`, `working_not_counted`,
  `staged_not_filled`, `tier_local_cluster_count`, `scan_engine_parity`,
  `ordering_exact`, `noncarriers_unchanged`, `tests_failed`,
  `replay_audited`, `unrelated_diff_paths`, and `external_writes`.

### Round 2 after the required repair

Preserve round-1 FAIL evidence and write the fresh verifier artifacts under
`artifacts/verify_2026-09-05/d34_open_leg_mults/round2/`. Re-run both original
counterexamples directly: the $600 / $2.70 solo must be 177 shares in scanner
and engine, and a two-row LT overflow cluster already capped at 100 shares by
ADV must remain at or below 100. Also prove the scanner-only calculation keys
are absent after the post-pass. Repeat every round-1 semantic probe, focused
and full tests, baseline reproduction, replay audit, diff-scope audit, and
zero-external-write check; a repair that only makes the original two examples
pass while changing any prior invariant remains FAIL.

### Round 3 after the slack-ceiling repair

Preserve rounds 1 and 2; write only under
`artifacts/verify_2026-09-05/d34_open_leg_mults/round3/`. Re-run every prior
probe plus the round-2 boundary: a computed 110-share ADV ceiling above the
100-share base row must still cap the 1.2x target at 110. Independently test the
same boundary for the concurrent-notional ceiling. Repeat focused/full tests,
unchanged-main baseline reproduction, replay/diff/external-write audit, and
produce a fresh scripted checks.json and fixed-format PASS/FAIL report.

## Report

Use section 6 of `docs/plan_2026-09-04.md` verbatim. Verdict must be exactly
PASS or FAIL; gaps are a FAIL until the builder closes them.
