# Running list (fortnight 2026-09-04 to 2026-09-18)

Companion to `docs/plan_2026-09-04.md`. Three states: OPEN (needs a decision
or is not started), AGREED (decided, briefed or in progress; brief path
given), DONE (commit hash). Move items down, never delete them.

## OPEN

- O1. DONE 2026-09-04: McKinley confirmed D5 semantics (Y on every leg = placed at grammar size; partial approval refuses the basket).
- O2. D7 OLV EOD book cap: McKinley wants to understand usefulness first. Mind's assessment (2026-09-04): insurance with a known premium (ledger replay binds ~10 days / 3 episodes since 2015, ~$58k of $362k OLV PnL forgone, every clipped leg a winner; live would bind more often given pre-07-29 live sizes 1.15-2.85x ledger) against a tail the ledger never realized (OLV's worst drawdowns were idiosyncratic at dial 20-50, SPY flat). Not needed this fortnight (no OLV up-lever); D10's hedge work is the cheaper bound. DECIDED by OWNER 2026-09-04: leave the OLV EOD book cap disabled for now. Re-ask when the D10 hedge prereg lands or when any OLV up-lever is proposed.
- O3. DONE 2026-09-04: McKinley keeps the pitch cadence daily. D6's event-trigger change is withdrawn; watchlist expiry (20 td) and registry-by-cell index remain as quality items, low priority.
- O4. DONE 2026-09-04 (see A7): D8 review part 1 ran; all three legs STAND; recorded in the prereg Status block and CLAUDE.md.
- O21. OLV pivot-aware entry policy (live since 2026-08-31, modeled + tested on all three sides per recon_parity): its evidence ("359 completed fills, +8.68R", strategy_config comment) exists nowhere in the repo, and 111 of 195 pivot sources on 2026-09-03 are older than the cache's re-adjust window. Decision: keep live; a research brief must reproduce the evidence this fortnight or the policy flag goes off; CLAUDE.md entry owed; basis question waits on recon_data_window.
- O22. order_staging's REL_CLOSE close-gap guard (limit moved to open +/- 0.15 ATR on a > 0.5 ATR gap-down through it) is live and unmodeled in the engine across all 7 persistent-limit strategies (recon_parity finding 1). Decision: document in CLAUDE.md now; model it once the fills store can size the effect.
- O23. Mind action: resolve the 2026-09-02 `execution_report` receipt (disposition `failure` unless the email is found to have gone out) so the controller stops exiting red on it.
- O5. D9 study DONE (see A6); decision D12 (liquid OVS 0.5x) recorded; OWNER veto window open until the OVS build brief is written.
- O6. D10 hedge prereg rewrite + MES round trip (December contract).
- O7. D11 ops fixes: health task, task pruning, operational log, persistent runtime logs, 05:30 S4U re-run task, cutover cadence rule in docs.
- O8. D11: resolve the 2026-09-02 `execution_report` receipt (verify whether the email went out first).
- O9. D11: CI red on main since 2026-09-02 (`tests/test_runtime_unicode_guard.py` on `scratch/kelly_read_source_article_2026-09-02.md`).
- O10. D1: the one cutover, bringing `harvest_fills` into the pinned runtime; needs the 09-03 incident write-up first.
- O11. D3.1-D3.6 sizing ship list (each = build brief + verify brief + harness run).
- O12. D6 cadence changes: pitch event trigger + watchlist expiry + registry index; posts pause; EP pause + two fixes; Focus kill rule; fundamental weekly; cost logging in all `claude -p` runners.
- O13. D7 OLV reconciliation (discretion gap leg by leg; 28 unmatched overflow positions).
- O14. D5 pitch_moo verify agent + fixture dry-run; then McKinley registers tasks + flag via `!`.
- O15. `book_snapshot.py` margin tags + hedge-panel headroom line (display only).
- O16. `ledger_git_sha` from `GITHUB_SHA` in deploy_site.yml.
- O17. Deferred lenses (not this fortnight): docs drift / CLAUDE.md restructure (2,066 lines), site audit, regime-gap research, codex PR review beyond what recon_parity covers.
- O18. Standing idea: trading_ibkr under git in place (secrets ignored) before any edit there. Recon decides the how.
- O19. Standing idea: one 08:45 morning digest replacing the 4-5 morning emails.
- O20. Standing idea: monthly statement-to-ledger match as a series (ledger size understates live size before a rule ships).

## AGREED

- A1. Plan files (`docs/plan_2026-09-04.md`, `docs/running_list.md`, `docs/briefs/2026-09-04/`) committed. Mind.
- A2. Recon wave launched 2026-09-04: `docs/briefs/2026-09-04/recon_onedrive.md`, `recon_parity.md`, `recon_data_window.md`, `recon_worktree.md`. Read-only, scratch output only.
- A3. Freeze (D1) in force from 2026-09-04.
- A4. 2026-09-04 04:10 v8 premarket verified: cboe_am, master_prices_am, risk_am, event_sleeve_am, scan_am all `success local`; both AM site deploys `success github` (supervisor `status --date 2026-09-04`, read-only). First real test of the 09-03 stall fix passed.
- A5. D8 study brief `docs/briefs/2026-09-04/study_pcfear_review.md`; D9 study brief `docs/briefs/2026-09-04/study_ovs_liquid.md`; incident write-up brief `docs/briefs/2026-09-04/doc_incident_0903.md`. Launched 2026-09-04.

- A6. D9 liquid-OVS study ran 2026-09-04 under the frozen registration; all four decision inputs hold (t -3.05; cut i -1.83; top-cell -3.02; 2024/25/26 each below mean). Evidence committed under `scratch/ultracode_research/ovs_liquid_2026-09-04/`. Decision D12.

- A7. D8 review part 1 ran 2026-09-04 (`scratch/ultracode_research/pcfear_review_2026-09-04/`): gates 1a/1b/1c/legB/LOYO PASS on both vintages; gate 2 not runnable (0 new fear-ON episodes); outcome all legs STAND; Aug-2026 shadow +2.21R on 6 zeroed signals recorded as one episode.
- A8. recon_parity DONE 2026-09-04: 19 contracts, 0 drift, 1,292 live pairs 0 divergences; earnings-override suspect NOT confirmed; OLV pivot IS modeled; 7 red tests (6 disabled-feature, 1 network); close-gap guard unmodeled; harvest_fills gap detector blind on empty ring. Report in `artifacts/recon_2026-09-04/parity/`.
- A9. Incident write-up `docs/incidents/2026-09-03_scan_am_stall.md` written 2026-09-04 (three corrections to the ops audit recorded inside it). D1's precondition for the one cutover is met.
- A10. Builders launched 2026-09-04: `build_tests_hygiene`, `build_sizing_d31_d32` (verify brief `verify_sizing_d31_d32` ready), `build_ops_supervisor`, `build_ops_fills_ledger`.

## DONE

- (none yet)
