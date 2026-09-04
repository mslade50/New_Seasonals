# Brief: verify_sizing_d31_d32 (independent verification of the WCDS tier retirement and the base-bps tilt)

Date 2026-09-04. Plan: `docs/plan_2026-09-04.md` (section 0 rule 5, D3 items 1-2, section 6). Type: VERIFY. You did not build this; your job is to break it.

## Decision and why
Plan rule 5: every money-path change gets an independent verifier before the mind reviews. The builder (`docs/briefs/2026-09-04/build_sizing_d31_d32.md`, read it) has left an uncommitted change in `strategy_config.py`, `daily_scan.py`, `pages/strat_backtester.py`, possibly `daily_portfolio_report.py`, and tests. You verify it against the decision as written, not against the builder's report.

## Files you own
None. Scratch only: `artifacts/verify_2026-09-04/sizing_d31_d32/`. You may not edit any repo file; if you find a defect you describe it with file and line and the mind sends it back to the builder.

## Hard rules
Section 0 of the plan. No ledger rebuild for production, no upload, no Sheets, nothing in OneDrive.

## Intent
Attack these, in order:
1. The decision table: read the diff (`git diff -- strategy_config.py daily_scan.py pages/strat_backtester.py daily_portfolio_report.py`) and confirm the tilt values and carriers match D3.2 exactly, every other strategy is 1.0, and nothing else in `strategy_config.py` changed (no bps, no bands, no windows).
2. Composition order on both sides: with the tilt applied at import, walk the scan's sizing steps (base -> frag band -> recency ladder -> cycle mult -> earnings override -> shares -> ADV cap -> ticker cap -> post-pass clamps/derates) and the engine's (3a..3b5) and confirm the tilt enters at the same point (the base) on both and is clobbered by the earnings override on both. Confirm the earnings override's `risk_bps` is NOT tilted. Confirm `path1_bps`/`path2_bps` are untouched. Confirm the per-strategy 250 cap in the engine is not scaled by the tilt.
3. Overflow path: for each of the three overflow-override consumers, confirm a tilted strategy in `OVERFLOW_RISK_OVERRIDES` would get the tilt (even if none is there today) and that OLV's override value is unchanged.
4. Idempotence: import `strategy_config` twice in one process (and via `importlib.reload`) and confirm the effective bps do not compound.
5. WCDS: grep both sides for any remaining seasonal-rank size multiplier; run the builder's new tests and your own fixture at ranks 70/55/40/20.
6. Replay: run the engine harness `scratch/ultracode_sizing_2026-09-02/dd_engine/engine_partial_replay.py`'s scenario mechanism (read it; it monkeypatches config per scenario and writes only under `dd_engine/`; ~14 min per full pass) for baseline-as-of-HEAD vs the working tree at GRM 1.5, or if that runner cannot express "tree as is", a full-ledger `process_signals_fast` pass of your own on the flat basis with all caps, written to your artifacts folder. Compare per-trade: the set of trade keys must be identical; only WCDS rows and rows of the six tilted strategies may change `Size_Mult`/`Shares`/`PnL`; report the 2010+ and 2016-07+ annual PnL, Sharpe, maxDD, worst-21d before/after. Expected direction: PnL roughly neutral, Sharpe up, maxDD narrower.
7. The scan email note: instantiate the note builder on a fixture signal for a tilted strategy and confirm the `tilt` fragment appears, and for WCDS confirm no tier text.
8. Full suite: `python -m pytest -q tests/ -p no:cacheprovider`; 0 failed expected (xfails from the tests-hygiene change are fine).

## Verification
`artifacts/verify_2026-09-04/sizing_d31_d32/checks.json` from your scripts: `{"table_matches_decision": bool, "other_config_changes": [...], "composition_same_both_sides": bool, "override_untilted": bool, "paths_untouched": bool, "cap_unscaled": bool, "overflow_sites_consistent": bool, "import_idempotent": bool, "wcds_tier_code_remaining": [...], "replay_keys_identical": bool, "rows_changed_outside_allowed": int, "metrics_before": {...}, "metrics_after": {...}, "note_fragment_ok": bool, "tests_failed": int}`.
No screenshots.

## Report
Section 6 format. Findings: every defect with file:line and a one-line fix, ranked. Handoff: PASS or FAIL for the mind, in one word, then the reasons.
