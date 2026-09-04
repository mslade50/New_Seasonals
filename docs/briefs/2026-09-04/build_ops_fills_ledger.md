# Brief: build_ops_fills_ledger (harvest_fills empty-ring gap guard; ledger provenance sha)

Date 2026-09-04. Plan: `docs/plan_2026-09-04.md` (section 0, D1, D11, section 6). Type: BUILD (small; a VERIFY agent covers it together with build_ops_supervisor).

## Decision and why
Two findings from the 09-04 parity recon. (1) `scripts/harvest_fills.py`'s `detect_gap` returns no gap when the broker ring is EMPTY, so "everything aged out of the 14-day ring" is indistinguishable from "no trades", which defeats the job's purpose (the harvester has never run in production and enters the pinned runtime at the v9 cutover; this must be in before it). (2) Both current ledger builds carry `ledger_git_sha=unknown`, so a vintage cannot be tied to the config that produced it. Decision: the gap detector treats an empty ring as a gap whenever the store's newest session is older than the ring's retention window minus one session, and prints it on the same loud GAP line (`--assert-no-gap` exits 3 as today); the ledger build reads `GITHUB_SHA` (fallback `git rev-parse HEAD`, fallback `unknown`) into the parquet metadata.

## Files you own
`scripts/harvest_fills.py`, `tests/test_fills_harvest.py`, `scripts/build_trade_ledger.py` (metadata only), `.github/workflows/deploy_site.yml` (only if the env var is not already exported to the step), and the ledger provenance test if one exists. Nothing else.

## Hard rules
Section 0 of the plan. Never call the broker DO or R2 outside a test double; never rebuild the production ledger; never upload.

## Intent
1. `detect_gap(ring, store, retention_days)`: empty ring AND store newest session older than `retention_days - 1` trading sessions before today (Eastern) -> gap=True with a message naming both dates; empty ring AND store newest session within that window -> no gap (a genuine no-trade fortnight); non-empty ring behaviour unchanged. Use the repo's trading calendar (`trading_calendar.py`) for session arithmetic, not calendar days. Tests for all three cases plus the existing ones.
2. `ledger_git_sha`: populated in `build_trade_ledger.py`'s metadata block from `GITHUB_SHA`, else `git rev-parse HEAD` via subprocess with a short timeout, else `unknown`; the provenance print at scan gate load already shows it. Confirm `deploy_site.yml` exposes `GITHUB_SHA` to the step (it is a default GitHub env var; verify the step does not scrub the environment). Add or extend a test that the metadata key is set from the env var when present.

## Recon first
`artifacts/build_2026-09-04/ops_fills_ledger/00_plan.md`, then build.

## Verification
`python -m pytest -q tests/test_fills_harvest.py -p no:cacheprovider` and the ledger provenance test; full suite once. `artifacts/build_2026-09-04/ops_fills_ledger/checks.json`: `{"tests_failed": int, "empty_ring_old_store_is_gap": bool, "empty_ring_recent_store_no_gap": bool, "nonempty_behaviour_unchanged": bool, "sha_from_env": bool, "sha_fallback_git": bool, "files_touched": [...]}`.
No screenshots.

## Report
Section 6 format. Handoff: the `git add` list.
