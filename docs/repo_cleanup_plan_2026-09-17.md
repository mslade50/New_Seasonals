# Repo cleanup plan — New_Seasonals (written 2026-09-17, execute 2026-09-18+)

**STATUS 2026-09-18:** Phases 0-4 EXECUTED. Worktrees 107 -> 4, local branches
207 -> 38, stashes 3 -> 0, `.git` 598 -> 318 MB, ~42.4 GB reclaimed, scheduled
tasks byte-identical before/after. Open: section 8 decisions
(`artifacts/recon_2026-09-17/unmerged_branches_decision.md`: 30 codex branches
with unique work + 4 `wip/` branches), Phase 5 origin deletes (still GATED),
Repo Health Check task re-registration. Log:
`artifacts/recon_2026-09-17/cleanup_log.md`; recovery bundle
`pre_cleanup_all_refs.bundle` (keep 30 days).

Deliverable for a cleanup agent. Read all of section 0 and 1 before running a
single mutating command. Evidence tables are in `artifacts/recon_2026-09-17/`
(gitignored, local only):

| File | What it holds |
|---|---|
| `branches_report.md` | every local + remote branch, bucketed A-G, PR join, stash contents |
| `body.md` / `report_part1.md` | full worktree inventory (107 rows) + production evidence |
| `inventory_raw.json`, `sizes.json`, `wt_porcelain.txt` | raw worktree data (path, branch, sha, merged, ahead/behind, dirty, MB) |
| `tasks.json` | full `Get-ScheduledTask` dump (184 actions) — the BEFORE snapshot |
| `dirty_tree_report.md` | per-file table of main's 48 modified + 466 untracked files, provenance, recommended action |
| `hygiene_report.md` | .git size, packs, large blobs, disk breakdown, venvs |
| `superseded.json`, `provenance*.json`, `prs.json` | supporting joins |

## 0. State of the repo (measured 2026-09-17 evening)

| Metric | Value |
|---|---|
| Worktrees registered | 107 (0 prunable, 0 locked — every dir exists) |
| Local branches | 207 (203 `codex/*`) |
| Remote branches | 66 |
| Stashes | 3 |
| Local `main` vs `origin/main` | 0 ahead / **99 behind**, clean fast-forward available |
| Main working tree | 48 modified tracked + 466 untracked (68 not-ignored entries) |
| Disk, all worktrees | 71.7 GB (41.8 GB outside the 4 production trees) |
| `.git` | 619 MB, 5926 loose objects, 5 packs, 2 garbage tmp objects, fsck clean |

Root cause: `AGENTS.md` mandates one worktree per concurrent codex task via
`scripts/new_task_worktree.ps1`. Its default root is the sibling dir
`dev\New_Seasonals-worktrees\` (3 trees live there), but agents kept passing
`-WorktreeRoot` INSIDE the repo under `artifacts/` (96 trees across six
spellings: `worktrees`, `task-worktrees`, `task_worktrees`, `audit_worktrees`,
`agent-worktrees`, `agent_worktrees`). `.gitignore:44` is a bare `artifacts/`,
so `git status` never showed any of it. Nothing scheduled creates these; they
regrow only when an agent starts a task. Section 6 closes that hole.

## 1. DO NOT TOUCH — production, verified against Task Scheduler (all state Ready)

| Path (under `C:\Users\McKinley Slade\dev\`) | Branch | HEAD | Used by |
|---|---|---|---|
| `New_Seasonals` (main checkout) | `main` | — | `-ConfigRoot` for both automation runtimes; working dir for Daily Pitch, Daily Posts, Market Context Brief, Radar Weekly Sync, RadarPackExport, IBKR Option Surface, Strategy Research, Research S4U Readiness |
| `New_Seasonals-automation-runtime-v9` | `codex/local-primary-runtime-v9-20260912` | `f1dd7611` | 9 tasks `New Seasonals Local v9 - *` (premarket, premarket-retry, discretionary, execution, postclose, inventory-close, health, indicator, weekly-rundown). Tag `automation-runtime-2026-09-17.nyse-risk-v2` == HEAD |
| `new-seasonals-legend-runtime` | `codex/new-seasonals-legend-runtime` | `206e6777` | `NewSeasonals-LegendETF-Signals` / `-Session` / `-Watchdog` |
| `New_Seasonals\artifacts\runtimes\expected-exit-runtime-20260914` | `codex/expected-exit-runtime-20260914` | `bc6dc10e` | `New Seasonals Expected Exit Monitor` (`-PinnedSha bc6dc10e…`). **Lives inside the gitignored `artifacts/` next to 95 throwaways. Exclude by exact path.** |

Also live, NOT worktrees, also inside `artifacts/`:
- `New_Seasonals\artifacts\strategy_research_runtime\.venv` — interpreter for both research tasks
- `New_Seasonals\artifacts\scheduler_recovery_20260908\` — referenced by a Ready task
- `New_Seasonals\artifacts\automation\` — cutover log archive

Branches that must survive (production runs code that is NOT on `origin/main`):
- `codex/local-primary-runtime-v9-20260912` — 15 commits not on origin/main
- `codex/new-seasonals-legend-runtime` — 9 commits not on origin/main
- `codex/expected-exit-runtime-20260914` — on origin/main, keep anyway (pinned)
- all 30 `automation-runtime-*` tags (they are what makes runtime worktree deletion safe)

Merging those branches is NOT part of this cleanup. Do not rebase, merge, or
delete them. `A/worktrees/legend-etf-prod-current` is NAMED like production
but no task uses it; the real Legend runtime is the sibling dir above.

Hard rules for the whole job:
1. Never delete or revert a file under `data/` in the main checkout. `data/` there is
   the live canonical input set (`rd2_fragility.parquet`, `cboe_putcall.parquet`,
   journals). Only ever move it forward, and verify append-only (section 3).
2. Run mutating phases OUTSIDE the pipeline windows (ET): premarket 04:10,
   health 07:30, discretionary 08:35, execution 16:30, postclose 17:10, plus
   whatever `Get-ScheduledTaskInfo` reports as NextRunTime for the Legend and
   Expected Exit tasks. Before each phase: `Get-ScheduledTask | Where State -eq Running`
   must be empty for every New Seasonals task. Best window: 10:00-15:30 ET or after 19:00 ET.
3. Every deletion goes through `git worktree remove` / `git branch -d`; no `rm -rf`
   on anything under `dev\` until git has released it. `git worktree prune` is a
   no-op today (nothing is prunable) — run it only at the end.
4. Per McKinley's global rules: destructive ops (worktree removal, `branch -D`,
   remote deletes, gc) are listed for approval before running. This plan IS the
   approval request; phases 4 and 7 remain gated on an explicit go (marked GATED).
5. Log every command + result to `artifacts/recon_2026-09-17/cleanup_log.md`.

## 2. Phase 0 — preflight (read-only, ~10 min)

1. `git fetch origin` (no `--prune`). Confirm `git rev-list --left-right --count main...origin/main` is still `0 <N>` with N >= 99. If main has moved ahead, stop and re-read.
2. Re-dump scheduled tasks to `artifacts/recon_2026-09-17/tasks_before.json` and diff against `tasks.json`. Any new task referencing a worktree path changes section 1 — stop and re-classify.
3. Safety net, before anything else:
   ```
   git bundle create "artifacts/recon_2026-09-17/pre_cleanup_all_refs.bundle" --all
   git bundle verify "artifacts/recon_2026-09-17/pre_cleanup_all_refs.bundle"
   git stash show -p stash@{0} > artifacts/recon_2026-09-17/stash0_requirements_20260907.patch
   git stash show -p stash@{1} > artifacts/recon_2026-09-17/stash1_autostash_20260813.patch
   git stash show -p stash@{2} > artifacts/recon_2026-09-17/stash2_wip_risk_dials_20260716.patch
   ```
   The bundle holds every ref (~600 MB). Every later deletion is recoverable from it.
4. Confirm `artifacts/automation/` already holds the v7 and v8 cutover logs (the docs say Cutover copies the outgoing generation's logs forward before its worktree is deleted by hand).

## 3. Phase 1 — rescue main's uncommitted work, then fast-forward

The dirty tree is 9 days of edits stranded on a stale main while codex merged
99 commits upstream. Per-file verdicts are in `dirty_tree_report.md`. Summary:

| Change set | Files | Provenance | Action |
|---|---|---|---|
| CS-A Pre-FOMC Rally removal | 12 (fragility_core/_simple, daily_risk_report, weekly_market_rundown, risk_dashboard_v2, build_risk_json/atr_downside/horizon_stats/trade_console, risk.js, 2 tests) | merged upstream as `8b34c14c` on 2026-09-17 | **discard** (3 of these — risk_dashboard_v2.py, risk.js, test_pc_dial_signal.py — are BEHIND origin/main; committing them would regress PR #58) |
| CS-C Execution tab layout | execution.js/.html, style.css, 2 tests | merged upstream as `2f42987f` | **discard** |
| CS-F research pipeline (SSRN pagination, item_budget 250→500, 20-F/40-F, radar-pack projection, health sidecars) | 8 modified + 4 untracked tests/docs | committed on unmerged `codex/red-build-repairs` (tip `5dcd2758`) and 7 sibling branches | **discard the working copy; land the branch via PR** (section 5) |
| CS-B Daily Pitch "standalone quality" (build_pitch_state.py drops build_book/exposure/receipts, SKILL.md, docs/daily_pitch.md, daily_portfolio_report.py rename, 1 test) | 5 | on no ref anywhere; contradicts committed CLAUDE.md book-overlap text | **commit to a branch `wip/pitch-standalone-20260917`, do not merge — McKinley decides** |
| CS-D pitch_lab `filter_vs_reanchor` / `reanchor_null` (+141, +88 tests) | 2 | uncommitted only, tested | **commit on main** |
| CS-E pitch_grammar holiday `Execute_On` roll (+8, +26 tests) | 2 | uncommitted only, tested | **commit on main** |
| CS-G legend/databento prototype (3 scripts + 3 tests + requirements.txt) | 7 untracked | on no ref; origin/main ships a patch file against `scripts/databento_futures.py` instead | **commit to `wip/legend-databento-20260917`, reconcile later** |
| 13 modified data files (2 parquets append-only verified, 3 jsonl grew, journals/scoreboards) | 13 | scheduled jobs | **keep the LOCAL versions** (see step 5) |
| 424 untracked `scratch/{pitch,context,posts}_checks/` files | 424 | agent evidence trail, deliberately not ignored (CLAUDE.md) | **commit on main** in one commit |
| lock files, `*_receipts.jsonl`, `market_breadth.sqlite`, `latest_decision.json`, cursors | ~15 | job side-effects | **gitignore** (section 6), never commit |

Recipe (revised 2026-09-18: the working tree is NEVER moved backward — the
`main` ref is moved underneath it, so live `data/` files on disk are untouched
throughout and no job window applies):

1. `git switch -c wip/main-snapshot-20260918` and commit EVERYTHING that is
   modified or untracked-and-not-ignored (`git add -A`; the junk in the last row
   above is fine to include here, it gets ignored later). Record the sha (S).
   This is the recoverable snapshot; nothing is lost from here on.
2. `git switch -c tmp/ff` (still at S), then `git reset --soft origin/main`.
   HEAD is now origin/main; index + working tree still hold the full snapshot
   content. `git status` now shows exactly "snapshot minus origin/main".
3. `git branch -f main origin/main` then `git switch main`. Both refs point at
   the same commit, so the switch changes NO files. `git branch -D tmp/ff`.
   Verify: `git rev-parse main` == `git rev-parse origin/main`; `data/rd2_fragility.parquet`
   on disk is byte-identical to `wip/main-snapshot-20260918:data/rd2_fragility.parquet`.
4. Now curate the staged diff against the NEW main:
   - CS-A, CS-C, CS-F paths: `git checkout origin/main -- <paths>` (moves code
     FORWARD to the upstream version; these are the stale/duplicate sets).
   - CS-D (pitch_lab.py + test), CS-E (pitch_grammar.py + test): inspect the
     staged hunks against the new base — if origin/main also changed those
     files, re-apply by hand. Run
     `python -m pytest tests/test_pitch_lab.py tests/test_pitch_grammar.py -q`.
     Commit as two commits.
   - `scratch/{pitch,context,posts}_checks/`: commit as one commit.
   - Junk (locks, receipts, sqlite, cursors, latest_decision.json): `git restore --staged`
     them, then apply section 6's gitignore rules so they vanish from status.
5. Data files (13 paths in `dirty_tree_report.md` §data, all still on disk at the
   local vintage): compare disk vs origin/main: parquet row count and
   prefix-identity (snapshot rows >= main rows AND first min(rows) rows
   byte-equal), jsonl line count and prefix. Where disk is a superset, commit as
   "data: carry forward local job output through 2026-09-18". Where origin/main
   is NEWER, `git checkout origin/main -- <path>` and note it. Where neither is a
   prefix of the other, leave the disk version, do NOT commit it, and surface
   to McKinley — that is a forked live series.
6. CS-B and CS-G: `git switch -c wip/pitch-standalone-20260918 main`,
   `git checkout wip/main-snapshot-20260918 -- <5 CS-B paths>`, commit; same for
   CS-G (3 scripts + 3 tests + requirements.txt) on `wip/legend-databento-20260918`,
   folding in stash@{0}'s two requirement pins. Back to main. Both branches are
   decision items for McKinley (section 8). Then `git stash drop stash@{0}`.
7. stash@{1}: verdict DROPPABLE (Phase 0 `stash_analysis.md`) — `git stash drop`.
   stash@{2}: one additive line in `pages/risk_dashboard_v2.py`; save as
   `wip/da-elevated-history` branch or leave the patch file, then drop.
8. Verify `git status` on main is clean and `main` == `origin/main` + the new
   commits. Keep `wip/main-snapshot-20260918` until McKinley signs off on
   section 8, then delete it.

Stashes (all on main; inspect via the patches saved in Phase 0):
- `stash@{0}` 2026-09-07 "codex-preserve-requirements-before-main-update" — `requirements.txt` +2. Check whether both lines are on origin/main; if yes `git stash drop stash@{0}`, else surface.
- `stash@{1}` 2026-08-13 **autostash, 41 files +501/-227**, touches `strategy_config.py`, `pages/strat_backtester.py`, `daily_scan.py`, `pc_fear.py`, `filters.py`, `exposure_leg.py`, 8 workflows, 2 parquets. Nobody chose to save it. Do NOT drop blind: `git stash branch stash-autostash-20260813 stash@{1}` (creates a branch at its base and applies), then `git diff origin/main --stat` on the code files. If every hunk is already on origin/main (expected — it is a month old and the P/C fear band work shipped 08-05/08-12), delete the branch and the stash is gone with it. Any hunk NOT on origin/main goes into section 8.
- `stash@{2}` 2026-07-16 "wip-risk-dials" — `risk_dashboard_v2.py` +1. Same check, expected drop.

## 4. Phase 2 — worktrees (GATED: list the removal set in the log, then proceed)

Removing a worktree never deletes its branch, so every non-production worktree
can go once its uncommitted changes are accounted for. Target end state: exactly
5 entries in `git worktree list` (main + the 4 in section 1).

Before removing, account for the only real uncommitted content:
- `artifacts/worktrees/denali-technical-20260916` — **12 tracked modifications**
  (branch is merged into main). Diff them; if they are report output, note and
  discard; if code, commit them onto its branch first.
- Untracked-only dirt (1-8 files each): v9 (production, skip), v8 (3),
  `task_worktrees/whole-repo-site-20260906` (5), `task-worktrees/pead-feasibility-20260910` (4),
  `task-worktrees/sna-close-fill-snapshot-20260910` (3), `task_worktrees/codebase-review-20260906` (3),
  `worktrees/intraday-html-report-20260827` (3), and 10 trees with 1-2 files.
  For each: `git -C <path> status --porcelain` into the log, copy the files to
  `artifacts/recon_2026-09-17/untracked_rescue/<tree>/`, then remove with `--force`.
- Committed docs cite ~12 paths inside 6 task worktrees
  (`docs/trading_desk_working_plan.md`, `docs/olv_execution_parity_review_2026-09-09.md` →
  `task-worktrees/{olv-parity-review-20260909 [dir name differs, see below], ovs-sizing-parity-20260909,
  inventory-inputs-20260909, priority4-shortlist-20260909, tlt-validation-20260910, pead-feasibility-20260910}`).
  Grep both docs for `task-worktrees/`, copy each cited file (they are small
  evidence files) to `artifacts/evidence_archive/2026-09-09/<tree>/<relpath>`, and
  add one line at the top of each doc saying the trees were removed on the
  cleanup date and where the copies live. Commit the doc edit.

Removal order:
1. Retired runtimes, both fully merged into main and preserved by tags
   `automation-runtime-2026-09-02.1` (v7) and `automation-runtime-2026-09-03.1` (v8):
   `git worktree remove --force "C:/Users/McKinley Slade/dev/New_Seasonals-automation-runtime-v7"` and `-v8`. Reclaims 8.7 GB (v8 is 7.1 GB of `data/` cache).
2. The 3 trees in `dev\New_Seasonals-worktrees\` (daily-pitch-reliability, runtime-safety-tests, supervisor-runtime-audit).
3. All 95 non-production trees under `New_Seasonals\artifacts\{worktrees,task-worktrees,task_worktrees,audit_worktrees,agent-worktrees,agent_worktrees}`.
   Iterate from `git worktree list --porcelain`, NOT from directory listings, and
   skip by exact path: `artifacts/runtimes/expected-exit-runtime-20260914`.
   Five directory names differ from their branch (`olv-cap-putcall-status-20260910`→`codex/olv-closing-handoff-20260910`,
   `olv-inventory-runtime-20260909`→`codex/olv-receipt-runtime-20260909`,
   `unified-position-actions-20260908`→`codex/position-actions-rollout-notes`,
   `execution-order-repair-20260914`→`codex/site-deployment-current-main`,
   `single-risk-dial-20260917`→`codex/risk-forward-windows-20260917`) — never infer a branch from a dir name.
   `artifacts/worktrees/nyse-runtime-20260917` has HEAD == the production v9 SHA; safe to remove, the tag holds it.
4. `git worktree prune`, then `git worktree list` must show exactly 5. Then
   `Get-ChildItem artifacts | Where Name -like '*worktrees*'` must be empty
   (remove any empty leftover root dirs by hand).
5. Re-dump scheduled tasks and diff against `tasks_before.json`: identical.
   Run each production task's action path through `Test-Path` — all four must exist.

Expected reclaim: ~42 GB.

## 5. Phase 3 — local branches (after Phase 1 so `main` == origin/main)

`branches_report.md` classified everything against `origin/main`. Bucket
counts: A 68, B 29, C 3, D 106 (in worktrees; re-bucket after Phase 2), E 48,
F 18, G 0. Steps:

1. Bucket A (68 merged by ancestry): `git branch -d` each. `-d` refuses anything unmerged, so it is self-checking.
2. Bucket B (29 content-merged via squash; `git cherry origin/main <b>` all `-`): re-verify the cherry result for each in the log, then `git branch -D`.
3. Former bucket D (106) — after Phase 2 they are plain branches. Re-run the
   bucketing: merged by ancestry → delete; all-`-` cherry → delete; else keep.
   For the ~70 that carry unique commits, apply one more rule before keeping:
   if `git branch --contains <tip>` lists ANOTHER kept branch (the tip was
   superseded by a later runtime/feature branch that built on it), delete it —
   the commits survive on the descendant. Most of the "ahead 40-98, behind 0"
   September runtime-pin chains collapse this way. What remains is the real
   decision list for McKinley (section 8), written to
   `artifacts/recon_2026-09-17/unmerged_branches_decision.md` with date, unique
   commit count, diff file count, and subject.
4. Bucket C: delete `codex/local-primary-runtime-v9` and
   `codex/olv-inventory-runtime-20260909` (exact duplicates of branches still on
   origin — NOT the production `…-v9-20260912`, check the full name). Keep
   `codex/execution-fast-actions-local` (07-23, 5 files, exists nowhere else) → section 8.
5. Non-codex: `recovery/private-site-options-20260818` and `agent/fx-bracket-entry`
   are bucket A, delete. `overflow-universe` is content-absorbed (memory says it is
   on main gate-off) → delete after confirming `tests/test_overflow_universe.py`
   exists on main.
6. `wip/*` branches from Phase 1 and the `stash-autostash-20260813` branch: keep until section 8 is resolved.
7. Target: under ~20 local branches (main, 3 production, the wip/decision set).
   `.git/config` `branch.codex/*` sections are removed automatically by `branch -d`;
   verify with `git config --get-regexp '^branch\.' | wc -l`.

## 6. Phase 4 — stop the regrowth + gitignore + physical hygiene

1. `scripts/new_task_worktree.ps1`: resolve `$WorktreeRoot` to a full path and
   abort with a clear message if it is inside the repo root (`$repoRoot`). Add
   one sentence to `AGENTS.md` line 19: worktrees live ONLY under
   `dev\New_Seasonals-worktrees\`; never under `artifacts/`. Add a test-free
   smoke check to the log: run the script with `-WorktreeRoot artifacts/x` and
   confirm it refuses.
2. `.gitignore` additions (from `dirty_tree_report.md` §5): `*.lock`,
   `data/*_receipts.jsonl`, `data/strategy_research/*_receipts.jsonl`,
   `data/market_breadth.sqlite`, `data/strategy_research/latest_decision.json`,
   `data/strategy_research/strategy_source_cursors.json`, and the scratch-root
   `.txt`/`.html` strays the report lists. Confirm `git status` is clean afterward.
3. Large tracked CSVs: `seasonal_ranks.csv` (35.5 MB) and `sznl_ranks.csv`
   (16.4 MB) are tracked and rewritten wholesale (7+ vintages in the top-15
   blobs). Do NOT rewrite history (107 worktrees' worth of refs make that
   unsafe). Decision item for McKinley: move them to R2 like `master_prices.parquet`
   and ignore going forward. Not executed by this plan.
4. `.git` maintenance, LAST, after the bundle exists and phases 2-3 are done:
   `git gc` (default expiry; NOT `--prune=now`, NOT `--aggressive`). Removes the 2
   `tmp_obj_*` garbage entries and packs the 5926 loose objects. Report
   `git count-objects -vH` before/after.
5. Repo Health Check task is DISABLED and mis-registered (unquoted path split
   into `Execute = C:\Users\McKinley`, `Arguments = Slade\dev\...`). Re-register
   with a quoted path via its registration script (`scripts/` — find it with
   `grep -l run_repo_health_check scripts/*.ps1`), then enable. Same bug on the
   disabled `Github list updates` task; report it, leave it disabled.
6. ~40 LibreOffice `soffice_profile_*` temp dirs under `artifacts/denali-*`
   are permission-locked; delete from an elevated shell or leave, low value.

## 7. Phase 5 — origin (GATED: outward-facing, needs McKinley's explicit go)

1. Close draft PR #1 ("Document execution fast position actions",
   `codex/execution-fast-actions-handoff`, 2026-07-22, 1 file) — or merge it if the
   one file is still wanted.
2. Bucket E (48 remote branches merged into origin/main with merged PRs):
   `git push origin --delete <branch>` in batches of 10; list is in `branches_report.md`.
3. Bucket F (18 unmerged remote): 4 are content-landed and safe
   (`execution-fast-actions-handoff` after PR #1 closes, `ovs-sizing-parity-20260909`,
   `unified-position-actions-20260908`, `overflow-universe`); 12 have identical-sha
   local twins resolved in Phase 3 (delete remote only where the local was
   deleted as superseded); 2 pre-September stragglers (`codex/fix-daily-scan-fallback`
   08-17, `codex/execution-fast-actions-handoff`) → section 8.
4. After remote deletes: `git fetch --prune`, confirm `git branch -r | wc -l` matches.

## 8. Decisions for McKinley (surface at the end, do not resolve unilaterally)

1. CS-B Daily Pitch "standalone" change (`wip/pitch-standalone-20260917`): keep or drop? It removes book/exposure/receipt context from the pitch state and contradicts CLAUDE.md.
2. CS-G legend/databento prototype (`wip/legend-databento-20260917`) vs the patch-file approach origin/main shipped.
3. `codex/red-build-repairs` (CS-F, research pipeline fixes + 7 sibling branches): open a PR to land it, or let it die.
4. The surviving unmerged-branch list from Phase 3 step 3.
5. `codex/execution-fast-actions-local` (07-23) and the two straggler remotes.
6. Anything from `stash@{1}` not already on origin/main.
7. Production runs unmerged code: v9 runtime is 15 commits past origin/main, Legend 9. Merge plan is a separate task.
8. Move `seasonal_ranks.csv` / `sznl_ranks.csv` to R2.
9. The v9 runtime's own `data/` is 25 GB (production, untouched by this plan).

## 9. Verification checklist (end of run)

- `git worktree list` == 5 entries, all four production paths `Test-Path` true
- Scheduled-task dump identical to `tasks_before.json`; every New Seasonals task still Ready
- `git status` on main clean; `main` == `origin/main` (+ the new commits from Phase 1)
- `python -m pytest --collect-only -q` collects without import errors; `python -m pytest tests/test_pitch_lab.py tests/test_pitch_grammar.py -q` passes
- `python scripts/repo_health_check.py` runs clean (its default `-RuntimeRoot` fallback is the unversioned `New_Seasonals-automation-runtime` path — pass the v9 root explicitly)
- Disk: `du -sm` of `dev\` before vs after in the log (expect ~42 GB reclaimed)
- `pre_cleanup_all_refs.bundle` retained for 30 days
- Next-morning check: the 04:10 premarket pipeline receipt lands in R2 as usual
