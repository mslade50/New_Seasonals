# Running list (fortnight 2026-09-04 to 2026-09-18)

Companion to `docs/plan_2026-09-04.md`. Three states: OPEN (needs a decision
or is not started), AGREED (decided, briefed or in progress; brief path
given), DONE (commit hash). Move items down, never delete them.

## OPEN

- O1. DONE 2026-09-04: McKinley confirmed D5 semantics (Y on every leg = placed at grammar size; partial approval refuses the basket).
- O2. D7 OLV EOD book cap: McKinley wants to understand usefulness first. Mind's assessment (2026-09-04): insurance with a known premium (ledger replay binds ~10 days / 3 episodes since 2015, ~$58k of $362k OLV PnL forgone, every clipped leg a winner; live would bind more often given pre-07-29 live sizes 1.15-2.85x ledger) against a tail the ledger never realized (OLV's worst drawdowns were idiosyncratic at dial 20-50, SPY flat). Not needed this fortnight (no OLV up-lever); D10's hedge work is the cheaper bound. DECIDED by OWNER 2026-09-04: leave the OLV EOD book cap disabled for now. Re-ask when the D10 hedge prereg lands or when any OLV up-lever is proposed.
- O3. DONE 2026-09-04: McKinley keeps the pitch cadence daily. D6's event-trigger change is withdrawn; watchlist expiry (20 td) and registry-by-cell index remain as quality items, low priority.
- O4. D8 decision rule: mind writes it after re-reading the 2026-08-05 prereg; study runs after.
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

## DONE

- (none yet)
