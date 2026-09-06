# Running list (fortnight 2026-09-04 to 2026-09-18)

Companion to `docs/plan_2026-09-04.md`. Three states: OPEN (needs a decision
or is not started), AGREED (decided, briefed or in progress; brief path
given), DONE (commit hash). Move items down, never delete them.

## OPEN

- O1. DONE 2026-09-04: McKinley confirmed D5 semantics (Y on every leg = placed at grammar size; partial approval refuses the basket).
- O2. D7 OLV EOD book cap: McKinley wants to understand usefulness first. Mind's assessment (2026-09-04): insurance with a known premium (ledger replay binds ~10 days / 3 episodes since 2015, ~$58k of $362k OLV PnL forgone, every clipped leg a winner; live would bind more often given pre-07-29 live sizes 1.15-2.85x ledger) against a tail the ledger never realized (OLV's worst drawdowns were idiosyncratic at dial 20-50, SPY flat). Not needed this fortnight (no OLV up-lever); D10's hedge work is the cheaper bound. DECIDED by OWNER 2026-09-04: leave the OLV EOD book cap disabled for now. Re-ask when the D10 hedge prereg lands or when any OLV up-lever is proposed.
- O3. DONE 2026-09-04: McKinley keeps the pitch cadence daily. D6's event-trigger change is withdrawn; watchlist expiry (20 td) and registry-by-cell index remain as quality items, low priority.
- O4. DONE 2026-09-04 (see A7): D8 review part 1 ran; all three legs STAND; recorded in the prereg Status block and CLAUDE.md.
- O21. DONE 2026-09-05 (see D-I): the pivot study reproduced and corrected the evidence, the basis audit found 1 of 19 policy assignments flipped, CLAUDE.md records the policy, and OWNER chose KEEP as a drawdown/appetite control rather than an edge rule.
- O22. order_staging's REL_CLOSE close-gap guard (limit moved to open +/- 0.15 ATR on a > 0.5 ATR gap-down through it) is live and unmodeled in the engine across all 7 persistent-limit strategies (recon_parity finding 1). Decision: document in CLAUDE.md now; model it once the fills store can size the effect.
- O24. OWNER, TODAY: sell CMI 54 sh (primary) by hand; confirm POWI 637 sh's time leg (15:59 today) is still working in TWS. See D14.
- O25. OWNER: put trading_ibkr under git in place per `artifacts/recon_2026-09-04/onedrive/git_plan.md` (8 PowerShell steps, object store outside OneDrive via --separate-git-dir, secrets/journals/flags ignored); pin the folder "Always keep on this device". Do this before the OLV exit fix is reviewed so the diff is recorded.
- O26. OWNER decision: `codex/legend-etf-prod-current` holds a ~10k-line IBKR execution sleeve (legend_etf/session.py, ibkr_adapter.py, reservations.py, databento_source.py) on a local branch only, no origin copy, not in CLAUDE.md. Push as backup, PR, or abandon.
- O27. OWNER decision: 27 local codex branches carry patch-novel commits with no origin copy (largest: research-OS / intraday / gap-reversal lineage, 21 commits; `critical-fixes-1-2-20260820`, 7 commits, status unknown). Backup-push all, the 27, or none. Then prune 17 abandoned worktrees and the two 6-8 GB checkouts (28.8 GB total).
- O28. `div_adjust.py` is LIVE and transmitting (modified two D target legs 2026-09-03) while its docstring and register script say it never transmits. Doc fix owed; kill switch is a source edit (`LIVE_ENABLED`).
- O29. `data/rd2_environment.json` written 05:10 today carries 6 NaN SPY fields and a "Downtrend" label; do not commit it; find the cause in the risk pipeline.
- O30. DONE 2026-09-04: safe-now plan executed as 8 commits 5089603e..86c5706d (see D-D). Remaining OWNER decisions from the same recon: D1 Legend-EMA/Databento scripts + requirements.txt deps (O31); D2 push/PR/abandon `codex/legend-etf-prod-current` (O26); D3 backup-push the 27 local-only branches (O27); D4 research-OS lineage + `critical-fixes-1-2-20260820` merge/park/drop; D5 `rd2_environment.json` NaN row not committed (O29); D6 prune worktrees after D3/D4. Discard list (X1-X4: `_old_*.py`, tool captures, both stashes, 33 patch-absorbed branches) awaits your nod; the mind will not delete without it.
- (superseded) Safe-now commit plan from recon_worktree (`artifacts/recon_2026-09-04/worktree/06_commit_plan.md`, ~45 min): .gitignore additions, pitch_lab.anchor_positions + its test, fundamental v2.1 + founder roster, the 51 check-date folders + context journal/flag state, data sync minus rd2_environment.json, the two research folders, cited scratch evidence. Mind executes after the current builders report (avoid mid-build commits of shared files).
- O31. Legend-EMA/Databento scripts (2,557 lines, tests 34/34, uncommitted, adds databento/keyring to requirements.txt): OWNER says commit with deps or move deps to a research requirements file.
- O32. Dynamic overflow universe: ALREADY ON MAIN since 2026-06-05 (cd4f83d5, 447a2dcf), gate OFF; the memory note saying "nothing committed" was stale and is corrected. Activation needs `OVERFLOW_UNIVERSE_ACTIVE=1` in the runtime env plus `data/overflow_universe.parquet`.
- O33. Scheduled full re-adjust of master_prices (monthly) to reset the 120-day-window basis steps (D15); brief owed after D13.
- O34. DONE 2026-09-05 (see D-I): OWNER chose KEEP. The policy is explicitly an appetite/fewer-fills/drawdown control, not an edge rule; the corrected study numbers govern future references.
- O37. DONE 2026-09-05: after the verified exit-runner PASS and an exact
  source-hash check, OWNER re-enabled `IBKR OLV Pre-Market Exits`. Task state
  is Ready; next scheduled run is 2026-09-07 09:10:10 ET; last result is 0.
- O36. `build_ops_supervisor` DONE 2026-09-04 (1,442 passed): premarket-retry task 05:30 (window 05:30-07:00 ET), invocation-scoped controller exit codes, stable state root under artifacts/automation with cutover copy-forward, -PruneSuperseded, two cron lines in the window, cadence doc. Verify brief `verify_ops_supervisor.md` launched. Mind's open calls from its handoff: widen the retry window to 08:30? add `--ignore-window` to resolve's dependent dispatch? (both deferred to after the verify).
- O44. OWNER, next week (Mon 09-07 or Tue 09-08, 10:00-16:00 ET, no other change that day): the v9 cutover. Full operator sequence with elevated steps marked is in the ops verifier's report (`artifacts/verify_2026-09-04/ops_supervisor/round2/`, section 7) and summarised in `docs/local_automation_task_scheduler.md`; the mind will paste it into this list on request. Order: tag main -> bump `AUTOMATION_RUNTIME_REF` + its test -> Prepare -> RegisterDisabled (-WhatIf first) -> `wevtutil sl Microsoft-Windows-TaskScheduler/Operational /e:true` (elevated) -> Cutover (-WhatIf first) -> Status; prune superseded generations only after one clean morning. THEN the SOXS upload (O40 command). Then `status premarket` after 05:45 and 07:30 the next morning.
- O23. DONE 2026-09-04 (see D-H). resolve the 2026-09-02 `execution_report` receipt (disposition `failure` unless the email is found to have gone out) so the controller stops exiting red on it.
- O5. DONE 2026-09-05 (see D-L): D9/D12's liquid OVS 0.5x and the D3.5
  non-midterm bottom-extremity 0.7x are built and independently verified.
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
- O45. DONE 2026-09-05: OWNER chose to keep GRM 1.5 after D3.6 failed its
  registered engine gate. The D3.6 source/test experiment was removed from the
  isolated worktree; D3.1-D3.5 and all failure evidence were preserved.
- O46. DONE 2026-09-05 (see D-N): OWNER superseded liquid OVS 0.5x with
  0.7x. Overflow, rank-mean, cycle, P1/P2 and cap policies are unchanged.
- O47. OWNER activation decisions for the post-close expected-flat control:
  name every account and signal producer; choose the authoritative broker
  position/fill feed, cutoff (recommended 16:35-16:40 ET), recipients,
  baseline/adjustment owner, correction policy, and whether a missing intent
  write blocks that producer. Recommendation: email one daily report for every
  scheduled run (including CLEAR), page ALERT/UNKNOWN, never auto-flatten, and
  require 10-20 consecutive shadow sessions with complete receipts and exact
  reconciliation before LIVE authority.
- O48. OWNER activation decisions for daily strategy discovery: approve the
  unattended X source (recommended official read-only API, otherwise a manual
  deterministic export), cost ceiling, source/query allowlist, collection and
  content-retention policy, weekday/weekend cadence, recipients, supported
  equities/ETF universe, and model/time budget. Keep X as discovery only;
  require independent data and reproducible after-cost validation plus 20
  shadow sessions before any automated owner-review email.
- O49. Code-audit remediation queue. First batch: F1 scanner requested/resolved
  universe coverage gate; F2 atomic or recoverable Sheets generation switch;
  F4 systemic fill-data outage as DATA_UNAVAILABLE + nonzero. Next batch: F5
  broker-truth OLV/D3.4 state, F6 version the OneDrive executor under Git/CI,
  F7 point-in-time universes/delistings, F10 locked dependencies, F11 retired
  scheduler fail-fast, and F12 dynamic-overflow ADV/cap ordering parity. Each
  money-path or operational change needs its own frozen contract and antagonist
  verification.
- O50. SECURITY, OWNER: revoke and rotate the plaintext Gmail application
  credential found in the external OneDrive reporting script, then store its
  replacement in approved secret storage. Do this before enabling any new SMTP
  sender. Removing the exposed value from backups or history is separate,
  destructive work and requires an explicit cleanup scope and approval.

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
- A11. recon_onedrive, recon_data_window, recon_worktree DONE 2026-09-04; reports under `artifacts/recon_2026-09-04/{onedrive,data_window,worktree}/`. OneDrive tests 421/421 pass; config drift clean except two order_staging fallback constants; no git; ExecAgent live on both accounts; three runners lack the verify-the-reject guard.
- A12. Builders launched 2026-09-04 (second wave): `build_olv_exit_fix` (verify brief ready), `build_tests_hygiene`, `build_sizing_d31_d32`, `build_ops_supervisor`, `build_ops_fills_ledger`; study `study_olv_pivot_evidence`. `build_soxs_repair` written, held until the sizing verify completes.
- A13. D3.5 OVS extremity + liquid-tier sizing DONE + VERIFIED PASS
  2026-09-05; see D-L. D3.6 is now the remaining sizing ship-list item.
- A14. D3.6 GRM 1.875 + overflow-long exclusion build and independent verify
  completed 2026-09-05 with formal FAIL; see D-M and O45. OWNER kept GRM 1.5,
  and the failed source/test experiment was removed. D3.1-D3.5 remain accepted.
- A10. Builders launched 2026-09-04: `build_tests_hygiene`, `build_sizing_d31_d32` (verify brief `verify_sizing_d31_d32` ready), `build_ops_supervisor`, `build_ops_fills_ledger`.

- O35. CONCURRENT SESSION HAZARD (2026-09-04 ~07:50 ET): another session has STAGED (git add) a 562-line overlay-free-Portfolio change in this working tree (pages/strat_backtester.py +39, scripts/build_trade_ledger.py +271, scripts/build_site.py, site/*, three tests) while this session's builders edit strat_backtester.py (sizing) and build_trade_ledger.py (ledger sha). The mind commits its own work by explicit path only and will not commit those files until that session lands its change. OWNER: which session is that, and can it commit or unstage?

- O42. SOXS repair verified PASS 2026-09-04 (Opus verifier: non-SOXS rows bit-identical, island mirror-consistent to 0.22%, guard battery 9/10 right; F1 false positive on a >2% mid-window ex-div, F2 `--only-tickers` uploads a partial refresh). Round 3 (length-rule tol 0.10; --only-tickers implies --no-upload) sent to the builder; commit follows; R2 upload still waits for the v9 cutover. R2 object confirmed untouched (161,963,528 bytes, 08:12 UTC).
- O43. Ops supervisor round 2 verified FAIL on three small items (cutover-state mirror below the missing-logs continue; marker read under config root not runtime root; health FAIL line only via RunLogger) + two false doc sentences; round 3 sent to the builder with the verifier's probes as acceptance. The verifier's full operator cutover sequence is in `artifacts/verify_2026-09-04/ops_supervisor/round2/` (report section 7).
- O41. OLV exit fix round 2 verified PASS 2026-09-04 (Opus verifier): F2/F3/F4 fixed and proven by the verifier's own sequences (34/34 builder tests, 18/18 attack sequences, no OCA join constructible); five narrower findings, two taken before Monday (recycled order id must also match ref+gat; SKIPPED_FLAT emails) plus three hardenings, sent as round 3 to the builder with the verifier's probe suite as acceptance. Monday operator check recorded in the verifier report (artifacts/verify_2026-09-04/olv_exit_fix/). Task stays DISABLED until round 3 lands; then OWNER runs `powershell -Command "Enable-ScheduledTask -TaskName 'IBKR OLV Pre-Market Exits'"`.
- O38. `verify_olv_exit_fix` returned FAIL 2026-09-04 (three code defects: stacked-leg retry steals the sibling bracket; late-run sell against a flat position; reject inside the verify pause misread as UNKNOWN). Round 2 build launched on Opus (brief appended). A second verify follows; the task stays DISABLED until PASS.
- O39. `verify_ops_supervisor` returned PASS with gaps (health/fallback-due crash on LockUnavailable; live-lease block counted as new failure; retry slot 05:30 vs 70-min lease; four doc contradictions). Round 2 build launched on Opus (brief appended). Round 1 stays uncommitted until round 2 verifies.
- O40. `build_soxs_repair` (D13) round 1 DONE 2026-09-04: local parquet repaired (46 SOXS rows / 15.0139, mirror-validated; vendor ratio 15.0139 confirms), backup `data/master_prices.parquet.bak_20260904_soxs`, `scripts/repair_price_island.py` + segment guard in the updater, 25 tests. Dry run: yfinance STILL serves the island; guard rejects it. Mind decisions: (1) segment-only row drop, not whole-ticker (round 2 sent to the builder), (2) R2 UPLOAD ONLY AFTER the v9 cutover puts the new updater in the pinned runtime (the old median guard would re-import the island at 17:10). Upload command: `python -c "from cache_io import upload_from_local; print(upload_from_local('data/master_prices.parquet','master_prices.parquet'))"`. The 126d window enters the island ~09-18; v9 must land before then.

## DONE

- D-P. Repository-wide quality audit and safe remediation slice completed
  2026-09-05. Audited 351 non-scratch Python files with zero AST syntax errors
  and clean fatal Ruff rules. Closed deterministic failure semantics in the
  portfolio report, intraday updater and local receipt health check, and made
  the price-island regression date-deterministic. Independent verifier PASS:
  52 focused plus 4 calendar tests. Integrated branch: 1,756 passed, 1 skipped,
  6 expected failures. High-severity decision work remains visible in O49.

- D-O. Offline daily-control foundations completed and antagonist-verified
  2026-09-05. Expected-flat reconciliation writes immutable structured local
  JSON/Markdown/HTML, uses exact quantities and fail-closed evidence authority,
  and has no order path (80 focused tests; round-two independent PASS).
  Strategy discovery v1.0.6 binds source, catalogs, exact research specs and
  lifecycle events into a reconciled run transaction; the final independent
  review passed exact revision `9ac98408b03072b961967b469f66e0913940de06`
  with 302 tests and 10/10 identity attacks rejected. Neither control has
  broker, X, email, scheduler, Sheets, R2 or production authority; activation
  gates are O47/O48/O50.

- D-N. Liquid-tier OVS 0.7x OWNER OVERRIDE implemented 2026-09-05. The
  D9/D12 0.5x study and D3.5 verification remain frozen history; this is an
  explicit risk-appetite change, not a post-hoc edge claim. At GRM 1.5 the
  before-cap contract is liquid normal P1/P2 42.0/8.4 effective bps, liquid
  non-midterm bottom-rank 29.4/5.88, liquid midterm 31.5/6.3, and overflow
  normal 60/12. Build/verify briefs:
  `docs/briefs/2026-09-05/{build,verify}_ovs_liquid_070.md`.

- D-M. D3.6 GRM 1.875 + overflow-long exclusion BUILT but VERIFIED FAIL
  2026-09-05 in `codex/fortnight-d33-clamps`. The implementation itself
  reconciled: 41 GRM-denominated
  fields scale once; all three consumers agree; four overflow-long base
  exemptions and both earnings-size exemptions are exact; fixed caps and
  D3.2-D3.5 dimensionless rules stay unchanged. Focused 54/54; full suite
  1,510 passed with zero D3.6 regressions (the lone price-island failure
  reproduces on unchanged main). Builder and independent frozen replays both
  found only $17,583.24/year of 2010+ improvement versus the required $30,000,
  a $12,416.76 shortfall. The other three gates pass: D3.5-control maxDD is
  $23,781.42 better than pre-D3, D3.6 worst-21d is 0.7386x pre-D3, and the
  2016+ drawdown trough is 2024-04-19. OWNER chose GRM 1.5; the failed D3.6
  source/test experiment was removed after verification, while D3.1-D3.5 and
  the full evidence were preserved. Evidence:
  `artifacts/{build,verify}_2026-09-05/d36_grm_step/`.

- D-L. D3.5 OVS rank-mean + liquid-tier sizing BUILT + VERIFIED PASS
  2026-09-05 in `codex/fortnight-d33-clamps` (uncommitted by fortnight rule):
  signal-close mean rank_2/5/10/21 below 94 sizes 0.7x outside midterms;
  year%4==2 is exempt while its existing 0.75 cycle cut remains; liquid OVS
  sizes 0.5x and overflow 1.0x on P1/P2. Independent attacks proved strict
  93.999/94, missing/non-finite fail-open, point-in-time snapshots, configured
  overflow 1.00x note stamping, and nonbinding/exact/binding mixed-tier P2
  cap arithmetic with cycle x tier x rank exactly once. Focused verifier
  74/74; full suite 1,499 passed with 0 D3.5 regressions (the lone price-island
  failure reproduces on unchanged main). Independent four-arm replay matched
  24,676 candidates and 4,700 rows/arm; 889 combined positions resized and
  zero midterm positions changed by the extremity-only arm. Combined versus
  D3.4: 2010+ annual PnL -$1.6k, Sharpe +0.027, maxDD unchanged; 2016-07+
  annual PnL -$1.75k, Sharpe +0.048, maxDD $0.66k better, worst day $9.0k
  better and worst-21d $2.8k worse. Evidence:
  `artifacts/{build,verify}_2026-09-05/d35_ovs_risk_mults/`.

- D-K. D3.4 WCDS/LT Trend solo-add sizing BUILT + VERIFIED PASS (round 3)
  2026-09-05 in `codex/fortnight-d33-clamps` (uncommitted by fortnight rule):
  true solos size 0.8x; 2+ same-tier/day staged rows or any strategy-wide
  filled-open leg size 1.2x; working limits do not count. Two verifier FAIL
  rounds closed scanner/engine one-share rounding drift and both binding/slack
  ADV + concurrent-notional ceiling escapes. Focused 44/44; full suite 1,488
  passed with 0 D3.4 regressions (the lone price-island failure reproduces on
  unchanged main). Frozen replay: 4,700 trades in both arms, 467 rows resized;
  versus D3.3, 2010+ annual PnL +$1.6k, Sharpe +0.008, maxDD -$3.6k worse;
  2016-07+ annual PnL +$1.5k, Sharpe +0.005, maxDD -$0.4k worse. Evidence:
  `artifacts/build_2026-09-05/d34_open_leg_mults/` and
  `artifacts/verify_2026-09-05/d34_open_leg_mults/round3/`.

- D-J. D3.3 clone clamps BUILT + VERIFIED PASS 2026-09-05 in `codex/fortnight-d33-clamps` (uncommitted by fortnight rule): IOB alone halves when both staged index signals fire; the five frozen 20-bps-nominal cross-strategy pairs join the incumbent IOB+MonFri pair; shared minimum-per-strategy resolution closes the engine's triple-collision last-pair-overwrite bug. Independent attacks proved aliases, absolute-then-clone order, staged-not-filled semantics, per-tier scan counting, pair-order invariance and no below-clamp raise. Focused 25/25; full suite 1,473 passed with 0 D3.3 regressions (the lone price-island failure reproduces unchanged on main). Full-history replay: 4,700 trades in both arms, 201 rows resized; versus the D3.2 control, 2010+ annual PnL -$7.8k, Sharpe -0.001, maxDD +$16.2k better, worst day/21d unchanged; 2016-07+ annual PnL -$12.0k, Sharpe -0.039, drawdown extrema unchanged. Evidence: `artifacts/{build,verify}_2026-09-05/d33_clone_clamps/`.

- D-I. OLV pivot policy KEEP (OWNER, 2026-09-05). The 2026-09-04 study (`scratch/ultracode_research/olv_pivot_evidence_2026-09-04/`) found no per-signal edge (affected-signal diff -4.6R, clustered t -0.33; total OLV PnL approximately unchanged). The old +8.68R citation was policy-v2 versus policy-v1, not policy versus no policy (the same sample versus no policy was -5.9R). KEEP is an explicit appetite choice: fewer fills and smaller June-2026 drawdowns (worst-21d -$37k versus -$60k; maxDD -$41k versus -$65k). Basis stability: 1 of 19 policy assignments flipped. `pivot_entry_policy.enabled` remains true; future docs must not call it an evidenced edge.

- D-H. 2026-09-02 `execution_report` receipt resolved by the mind 2026-09-04 as `failure operator` (reason recorded in the receipt); the controller stops exiting red on it. The 09-02 report email is unrecoverable (log destroyed); the 09-03 report succeeded.
- D-G. Local automation v9 contents committed b2c4da08 (D11; two verify rounds + round 3 fixes; 1,488 tests). Not in production until the v9 cutover (O44).
- D-F. SOXS island repair + segment-aware basis guard (D13) committed 2026-09-04 (verify PASS + round 3: length-rule tol 0.10, --only-tickers implies --no-upload; 39 tests). Local parquet repaired (backup `data/master_prices.parquet.bak_20260904_soxs`); R2 upload pending the v9 cutover (command in O40).
- D-E. OLV pre-market exit runner FIXED 2026-09-04 (D14): OneDrive `olv_exit_moo.py` sha256 3e8c1ad9..., `test_olv_exits.py` b393adc8... (40 tests); backups `_backup_20260904_olv_exit_{prepatch,round1,round2}/`; second verifier PASS + round-3 hardenings (recycled-id cross-check, SKIPPED_FLAT emails, positive orderId, anchored leg suffix, retry identity carried) with 26/26 verifier sequences. Mind's call: SKIPPED_FLAT stays exit 0 but always emails. OWNER re-enabled `IBKR OLV Pre-Market Exits` 2026-09-05 after the verified hashes matched; task Ready, next run 2026-09-07 09:10:10 ET, last result 0. Monday operator check in `artifacts/verify_2026-09-04/olv_exit_fix/` report. Not in git until O25 (trading_ibkr baseline).
- D-D. Worktree safe-now plan, 8 commits 2026-09-04: 5089603e .gitignore; 80ed495c pitch_lab anchor_positions; c98c44fb fundamental v2.1 + roster; 83e6e53a drill scripts + context journal/flag state; 7f408472 data sync (minus rd2_environment.json); c8ad7bfa two research folders; 0fb243d6 cited scratch evidence; 86c5706d root design records. Untracked entries 395 -> 17.
- D-C. Sizing D3.1 + D3.2 committed 1efcdf14 (verify PASS, full-history replay); ledger_git_sha 47168088; dead wcds overlay-lab control removed and CLAUDE.md sizing note added (next commit).
- D-B. harvest_fills empty-ring gap guard (`build_ops_fills_ledger`, half 1): committed by path 2026-09-04. Half 2 (ledger_git_sha from GITHUB_SHA in scripts/build_trade_ledger.py + tests/test_ledger_provenance.py) is built and tested in the worktree but HELD: that file carries another session's staged hunks (O35). Root cause of `unknown` confirmed: the deploy generator dir has no .git, so `git rev-parse` fails; reading GITHUB_SHA first fixes it with no workflow change.
- D-A. Tests hygiene (`build_tests_hygiene`): 11 failures -> 0; 6 strict xfails, 1 skip; committed by path 2026-09-04.
