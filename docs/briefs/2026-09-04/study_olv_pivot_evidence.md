# Brief: study_olv_pivot_evidence (reproduce the evidence behind the live OLV pivot-aware entry policy)

Date 2026-09-04. Plan: `docs/plan_2026-09-04.md` (section 0, running list O21, section 6). Type: RESEARCH, read-only. You write only under `scratch/ultracode_research/olv_pivot_evidence_2026-09-04/`.

## Decision and why
Since 2026-08-31 (commits dec62f06, a3036527, 1119efae) Oversold Low Volume entries carry `pivot_entry_policy` (`olv_close_pivot_40_v2_20260901`): the limit offset below the close is widened or the signal skipped by how far the close sits above a 40/40 pivot high (default 0.25 ATR; 0.50 at 2-3 ATR above; 0.75 at 4-5; skip above 5). The 09-04 parity recon found it modeled in the engine and tested on all three sides, and found its only evidence is a comment in `strategy_config.py` ("359 completed-fill sample, +8.68R") that no script in the repo produces. It also found that 111 of 195 pivot sources on 2026-09-03 predate the price cache's 120-day re-adjust window, so those pivot highs sit on a pre-dividend basis (measured basis steps up to 1.75%, median 0.8%, on dividend payers), which can move a name across the 2/3, 4/5 ATR band edges. The mind's decision rule (closed set: KEEP the policy, or FLAG IT OFF live pending re-study): keep only if (a) the policy's effect on the OLV ledger reproduces as an improvement in PnL per unit risk with signal-date-clustered t >= 1.5 on 2010+ and on 2016-07+, and (b) band assignments are stable under basis correction (fewer than 10% of policy-affected signals change band or skip status when pivots are computed on a fully adjusted series). You compute; you do not decide.

## Files you own
`scratch/ultracode_research/olv_pivot_evidence_2026-09-04/` only.

## Hard rules
Section 0 of the plan. No config change, no ledger rebuild for production, no upload. One read-only yfinance pull is allowed for the basis-corrected series, capped at the OLV universe (about 200 tickers, daily bars, 3 years, `auto_adjust=True`); handle the MultiIndex per CLAUDE.md. Do not run the price updater.

## Intent
1. Reconstruct what the "359 completed-fill sample, +8.68R" could have been: read the policy code (`strategy_config.py` pivot block, the resolver function the scan and engine share, `tests/test_olv_pivot_entry_policy.py`), search `scratch/`, `research/`, `artifacts/`, git history (`git log --all -S "8.68"`, `-S "ClosePivot"`, `-S "pivot_entry_policy"`) and the codex branches for its origin. Report what you found or that it does not exist.
2. Engine replay with and without the policy: `process_signals_fast` exposes the policy through the execution dict; run OLV only (liquid and overflow tiers) 2010-01-01 to 2026-08-31 on the flat basis with all caps, once as configured and once with the policy disabled (set the field to None or the documented off value; state which). Report per era (2010+, 2016-07+, 2024+): N, fills, avgR, PnL per unit risk, total flat PnL, worst 21d, maxDD, and the signal-date-clustered t of the per-trade R difference on the affected signals only (those whose band is not default). Include the count of signals the policy skipped and what those would have earned without it.
3. Basis stability: compute the 40/40 pivot high and the distance in ATR for every OLV signal in the last 3 years on (a) the cache and (b) the fully adjusted yfinance series; report how many signals change band or skip status, and list the ten largest distance shifts with the dividend that explains them.
4. Live status: for 2026-09-03 and 09-04, list the OLV-universe names in a non-default band on the cache and which of them flip on the corrected series.

## Recon first
`scratch/ultracode_research/olv_pivot_evidence_2026-09-04/00_plan.md`.

## Verification
`checks.json` from your script: `{"evidence_found": bool, "evidence_location": "...", "with_policy": {...}, "without_policy": {...}, "affected_signals_n": int, "affected_diff_R": float, "affected_t_clustered_2010": float, "affected_t_clustered_2016": float, "skipped_n": int, "skipped_wouldbe_R": float, "basis_flip_share": float, "basis_flips_n": int, "live_nondefault_n": int, "live_flips_n": int}`.
No screenshots.

## Report
Section 6 format. Handoff: state which of the mind's two conditions hold; propose nothing outside KEEP / FLAG OFF.
