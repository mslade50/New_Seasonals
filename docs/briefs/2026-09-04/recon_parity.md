# Brief: recon_parity (engine / scan / staging parity, plus the 2026-08-25..09-03 change set)

Date 2026-09-04. Plan: `docs/plan_2026-09-04.md` (read section 0 first). Type: RECON. You write nothing outside your scratch folder.

## Decision and why
The book's ledger replays today's rules over history, and live orders are sized by `daily_scan.py` and shaped by `order_staging.py`. Every rule that exists on one side and not the other is a silent money error. CLAUDE.md lists the "aligned sites" contracts; several changes shipped since 2026-08-25 (OLV pivot-aware live entries, stale-pivot expiry, execution-tab position actions, delayed-fill backfill, the local-automation supervisor) with no parity audit and, for the OLV pivot entries, no CLAUDE.md entry. The sizing ship list (plan D3) will add five more overlays to exactly these files. The mind needs a contract-by-contract status before any of that is built.

## Files you own
None in the repo. Scratch only: `artifacts/recon_2026-09-04/parity/` (create it). Read-only on `C:\Users\McKinley Slade\OneDrive\trading_ibkr` (never run anything there).

## Hard rules
Section 0 of the plan. Running the repo's test suite is allowed (`python -m pytest -q tests/ -p no:cacheprovider`, it needs no network for most tests; if a test hangs on network, note it and exclude it). Running a read-only parity harness you write against the local `data/master_prices.parquet` cache is allowed; it must import scan/engine functions without side effects (look for how `scripts/` stub Streamlit with a `_NoOp` pattern). Do not rebuild the production ledger. Do not upload anything.

## Intent
1. Contract table. For every "aligned sites, change together" contract in CLAUDE.md (frag bands + P/C fear lag-1, OLV recency ladder, earnings size override composition, cross-strategy overlap clamp, same-day derate, gap-size derate, OVS 2-path + scale-out + Friday EOD-DD, OLV vol-confirm exit + T+3 fill window + ticker notional cap, stop arming day 2, stop gap-fill slippage, per-strategy 250 cap, GRM scaling of every bps constant, cycle-year mult, cross-strategy precedence ATR-Ext over OVS, Monthly Weak Close month-end detection, trend/event sleeve state conventions): read the code on every side and mark OK / DRIFT / UNMODELED-LIVE / UNMODELED-ENGINE, with the guard test named or "NO GUARD". Known suspect to confirm and quantify: the earnings override composes differently in scan (clobbers, applied last) vs engine (multiplied); state exactly which overlays each side keeps, and what diverges if OLV gains a tilt (plan D3.2 keeps OLV at 1.0, so say what would happen for the tilted strategies that carry an override: currently OLV and St OS Sznl).
2. The OLV pivot-aware entries (`git show dec62f06 a3036527 1119efae`). Exactly what changes in the staged `Limit_Price`, `Entry`, or any stamped column; whether the engine models it (it almost certainly does not); whether it introduces an ABSOLUTE dollar level into any engine path (that would break the dividend-adjustment scale-invariance rule in CLAUDE.md); whether it is tested; what CLAUDE.md should say. Give the mind a recommendation: model it in the engine, or gate it off live until modeled, with the evidence for the pivot rule if any exists in scratch/.
3. The change set 2026-08-25..09-03 on production paths: "Backfill delayed execution fill prices", "Give the execution tab three working ways to close a position", "Add quarter trim", the supervisor + `run_local_automation.ps1` + installer, "Recover stalled premarket scans safely" (two byte-identical commits), "Harvest broker fills", the EP and Discretionary Focus series. For each: production files touched, tests added, verdict SAFE / NEEDS-GUARD / NEEDS-FIX with the line. Specifically check: can a recovery routine re-run `daily_scan` and re-stage orders after `order_staging` already consumed the tab that morning; can a UI action on the execution tab send a live order in an unintended mode (stale book, double click, unknown mode); does the fills harvester's set-containment guard hold on an empty ring.
4. Parity harness. For every strategy in `STRATEGY_BOOK` and a sample of at least 40 liquid tickers, compare `daily_scan`'s live signal check against `filters.get_historical_mask(...).iloc[-1]` on the local cache for the latest session. Report divergences by (strategy, ticker) with the reason.
5. `pages/backtester.py` is a third engine (UI). List concretely where it diverges from `strat_backtester` (stop gap-fill, fill window, entry-day stop, scale-out, caps) so research on that page is known to be non-production.
6. Run the full test suite once and report pass/fail/skip counts and every failure verbatim. Do not fix anything.

## Recon first
Write `artifacts/recon_2026-09-04/parity/00_plan.md` with the contract list you will check and the harness design before writing code.

## Verification
`artifacts/recon_2026-09-04/parity/checks.json` produced by script:
`{"contracts_checked": int, "contracts_ok": int, "contracts_drift": [..], "contracts_unmodeled": [..], "no_guard": [..], "olv_pivot_modeled_in_engine": bool, "olv_pivot_absolute_level_in_engine_path": bool, "harness_pairs": int, "harness_divergences": [{"strategy":..., "ticker":..., "reason":...}], "pytest": {"passed":int, "failed":int, "skipped":int, "failures":[..]}, "changeset_verdicts": [{"commit":..., "verdict":..., "issue":...}]}`.
No screenshots required.

## Report
Section 6 format. Findings ranked by dollar relevance. Handoff: for each of plan D3.1-D3.5, name the exact functions on each side where the builder must add the overlay so that scan and engine compose in the same order, and name the test file that should pin it.
