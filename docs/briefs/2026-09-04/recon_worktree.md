# Brief: recon_worktree (uncommitted work, branches, worktrees, hygiene)

Date 2026-09-04. Plan: `docs/plan_2026-09-04.md` (read section 0 first). Type: RECON. You write nothing outside your scratch folder.

## Decision and why
The working tree carries 16 modified tracked files and roughly 387 untracked files; there are about 93 git worktrees and 98 branches, 21 GB under `artifacts/`, and the June 2026 dynamic-overflow-universe work has never been committed. Two byte-identical commits landed on 09-03 from a local main and a codex worktree. Before the fortnight's build agents start (they must not collide with in-flight work and must not commit someone else's half-finished change) the mind needs to know what every dirty and untracked item is, what is at risk of loss, and an ordered commit plan McKinley can execute in under two hours. This brief classifies; it does not stage, commit, stash, clean, or delete anything.

## Files you own
None. Scratch only: `artifacts/recon_2026-09-04/worktree/` (create it). Note the plan files just written (`docs/plan_2026-09-04.md`, `docs/running_list.md`, `docs/briefs/2026-09-04/*`) are the mind's and will be committed by the mind; classify them as such and leave them alone.

## Hard rules
Section 0 of the plan. Read-only git only: `status`, `diff`, `log`, `branch`, `stash list`, `worktree list`, `show`, `ls-files`, `check-ignore`. Never `add`, `commit`, `stash`, `checkout`, `reset`, `clean`, `worktree remove/prune`, `branch -d`.

## Intent
1. Every modified tracked file: diff stat, what the change does, complete or not, whether its tests pass (run only that file's tests), and which session likely made it (mtimes, handoff docs, commit messages on nearby branches). Classify: production-data refresh to commit (journals, scoreboards, PIT parquet, cboe), code in flight (fundamental/*, pitch_lab.py, tests/*, requirements.txt: what changed and why), accidental.
2. Every untracked file and directory: classify as (a) research evidence CLAUDE.md says is committed (`scratch/context_checks/<date>/` drill scripts, `scratch/pitch_checks/<date>/`), (b) generated or backup that should be gitignored (`*.bak_*`, `_old_*.py`, `_tape_*.py`, tool output), (c) real work never committed. Give special attention to: the dynamic overflow universe (`overflow_universe.py`, `scripts/build_overflow_universe.py`, `scripts/build_overflow_prices.py`, symbol master, `pages/backtester.py` changes, `data_provider.py` include_overflow, `tests/test_overflow_universe.py`; check the `overflow-universe` branch too), `research/settlement_cash_dash/`, `research/treasury_month_end/`, `research/letf_flow_monitor/`, `hedge_panel_build_brief_2026-08-25.md`, `agent_brief_phase0.txt`, the two July proposal HTMLs, `hh_hl.txt`, `data/universe_liquid.json`, `data/proxy_extra_ranks.csv`, `data/event_sleeve_state.json`, `data/trend_sleeve_state.json`, `data/context_flag_state.json`, `data/context_journal.jsonl`.
3. Branches, stashes, worktrees: every branch with commits not on main (`git log main..<branch> --oneline`), every stash with its date and touched files, every worktree with its path, HEAD, and dirty state. Which worktrees are abandoned (HEAD older than 7 days, no unique commits) and how much disk each holds. Note the pinned runtime worktrees (`New_Seasonals-runtime-v*`) as production, not prunable by this plan.
4. Risk of loss: rank uncommitted items by hours of work that a disk failure or careless `git clean` would destroy.
5. Hygiene versus the rules in AGENTS.md: is `artifacts/` used as intended, is `.gitignore` consistent with CLAUDE.md's statement that scratch `.py`/`.md` are tracked, any file over 5 MB not ignored, any secret-looking file not ignored (`.env`, `credentials*.json`, `*token*`).
6. Commit plan: an ordered list of commits (files, message, and any `.gitignore` additions) McKinley or the mind can execute in under two hours, separating (i) safe now, (ii) needs a decision from McKinley (name the decision), (iii) should be discarded (name why). The dynamic overflow universe must land with its activation gate OFF if it lands at all; state what that requires.

## Recon first
`artifacts/recon_2026-09-04/worktree/00_plan.md` first.

## Verification
`artifacts/recon_2026-09-04/worktree/checks.json` from a script:
`{"modified_tracked": int, "untracked": int, "branches_ahead_of_main": [{"branch":..., "commits":int}], "stashes": int, "worktrees": int, "worktrees_abandoned": int, "artifacts_gb": float, "large_unignored_files": [..], "secret_like_unignored": [..], "overflow_universe_in_tree": bool, "overflow_gate_default_off": bool, "commit_plan_steps": int}`.
No screenshots required.

## Report
Section 6 format. Findings ranked by risk of loss. Handoff: the commit plan as three numbered lists (safe now / needs McKinley / discard), plus the list of files any build agent this fortnight must NOT touch because another change is in flight there.
