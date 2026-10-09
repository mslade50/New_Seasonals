# Handoff: trading_ibkr repo cutover and remaining execution work (2026-10-09 17:00 ET)

Owner: McKinley. Read `CLAUDE.md` first. Both repos must be treated as live-money code.

## Goal

From Monday 2026-10-12, every IBKR script runs from a pinned worktree of the private repo
`mslade50/trading_ibkr`, with state in `C:\trading_state`, instead of from `OneDrive\trading_ibkr`.
The owner chose the FULL cutover (plan Phases 1+2+3 together) on **Sunday 2026-10-11 after 21:05 ET**.

## Where things are

| Item | Location / value |
|---|---|
| Plan (background) | `docs/trading_ibkr_onedrive_migration_plan_2026-09-23.md` |
| **Runbook (follow this)** | `deploy/cutover/RUNBOOK.md` on branch `cutover` of `mslade50/trading_ibkr` |
| Release tag | `ibkr-runtime-2026-10-11.2` = `5bee874a8cc8db81a6dcb9f62e8a67f23ce76423` (pushed) |
| Staged on desktop | `C:\trading\runtime\trading_ibkr-g1` (clean, at the tag); bundle in `C:\trading\bundles\` |
| Laptop clone | `C:\Users\mckin\dev\trading_ibkr` (branch `cutover`; worktrees `tibkr_exec`, `tibkr_options`, `tibkr_deploy`, `tibkr_live` can be removed) |
| Desktop repo clone | `C:\Users\McKinley Slade\dev\trading_ibkr` (no GitHub push credential over ssh: move commits as a git bundle + scp) |
| Code freeze | No direct edits to `OneDrive\trading_ibkr` since 2026-10-09 (`CLAUDE.md`, `AGENTS.md`). The OneDrive code equals snapshot `829096f` except the allowlisted `pa_nightly_report.py`. |

What the tag contains (all tested: 1460 passed / 0 failed / 1 skipped; cutover self-test 292/292):
- Live OneDrive code as of 2026-10-09, plus `runtime_paths` routing of every state and secret path (inert when the env vars are unset).
- Execution fixes:
  - Definitive IBKR rejects report `rejected`, not "VERIFY IN TWS".
  - Errors are captured per order, not in a global list.
  - A duplicate command reply carries the original's outcome.
  - The command expiry is re-checked after taking the execution lock.
  - 20 drifted tests resolved: `docs/test_drift_decisions_2026-10-09.md`; two lost safeties restored, in `close_resize` and in `event_moo`'s reject check.
- Options:
  - New `option_close` / `option_roll` command types (`option_position_orders.py`).
  - The chain carries last, OI and volume.
  - The agent advertises `capabilities`.
- Cutover tooling under `deploy/cutover/`: preflight, cutover, verify, rollback, `arm_new_types`, stage_worktree, self-test.

The New_Seasonals site side (Close/Roll on the Execution tab, roll mode, quote freshness badges, calendar prefill fix, TWS-style ladder) and the reader changes are merged on `main` (`c54b5f16`). Close/Roll buttons stay hidden until the new agent advertises the capability.

## Remaining tasks, in order

### Before Sunday 21:05 ET

1. **Two stuck `attention` records block preflight.** Get the owner's choice first: these are live order state. They are in `OneDrive\trading_ibkr\data\position_actions\`:
   - `4a2b57b6...json`: a `close_resize` on NOVT, created 2026-09-23;
   - `cf3e18da...json`: an `add_to_position` on RYAAY, created 2026-10-08.

   Both say "Automatic recovery paused by manual order edit; reconcile before resuming". There are two ways to resolve them:
   - the owner clicks Reconcile in the site Execution tab;
   - or, with explicit owner approval, you verify the IBKR position and exits match and mark the records terminal.
2. **The desktop New_Seasonals checkout must pull `main`.** `C:\Users\McKinley Slade\dev\New_Seasonals` is `ahead 4, behind 11`:
   - The 4 local commits are another session's unpushed OpenBreakout work (`2a1308e7`, `0fd57992`, `2fbb49d3`, `257b1b58`).
   - There are also ~37 dirty files.

   Confirm with the owner that the session is finished. Then push those commits (merge with origin first) and pull `main` there. Preflight checks for `run_radar_sync.bat` + `daily_pitch.py` from `wip/cutover-readers`.
3. If any code changes before Sunday:
   - commit to `cutover`;
   - rerun `python -m pytest -q -p no:cacheprovider --ignore=legend_ema_futures_staging` and `deploy\cutover\tests\run_selftest.ps1`;
   - run `python deploy/cutover/derive_state_manifest.py --write` if state paths moved;
   - cut a new tag `.3`, bundle it, scp it, re-stage the worktree (`git worktree remove` then `add --detach`), and update RUNBOOK tag/SHA.
4. Optional: rehearse task registration on a throwaway task (RUNBOOK section 1.5). The self-test cannot exercise the real Task Scheduler.

### Sunday 2026-10-11, 21:05 to ~21:45 ET (RUNBOOK section 2)

- Confirm ExecAgent exited at 21:00 (`exec_agent_last_run.log` tail). Never kill it: use the safe stop in RUNBOOK section 5.
- **The owner pauses OneDrive in the tray** (cannot be done over ssh).
- Run `preflight.ps1 -Tag ibkr-runtime-2026-10-11.2 -ExpectedSha 5bee874a8cc8db81a6dcb9f62e8a67f23ce76423`. It must say GO.
- Run `cutover.ps1 ... -WhatIf` and review it, then run the real `cutover.ps1`. Then `verify.ps1 -Sweep`.
- **`arm_new_types.ps1` runs only with explicit owner approval.** It adds `option_close,option_roll` to `LIVE_TYPES`.
- Resume OneDrive, then `verify.ps1 -Sweep` again.
- Rollback (`rollback.ps1 -RunDir ...`) must finish by 02:30 ET if needed.

### Monday 2026-10-12

- Watch list in RUNBOOK section 4:
  - 05:00 ExecAgent; 08:50 radar; 09:05 event; 09:10 OLV; 09:15 div; 09:29 Legend; 09:31 order chain; 09:36 risk agent open fill; 10:40 verify; 16:35 PA report; 17:00 div evening.
  - Every log must show `[runtime_paths] ... source=env`, and nothing new may appear in OneDrive.
- **11:00-15:00 staged live proof** (plan section 5.4): a 1-share far-limit order placed from the site, then cancelled. Requires the owner present.
- Check that the site shows Close/Roll on held options once the new agent is up. Test a Close preview on a real option position with the owner before any live close.

### Afterwards (not urgent)

- Rotate the Gmail app password, `EXEC_AGENT_TOKEN` and `STATUS_TOKEN`. Copies sat in OneDrive cloud storage, including the `*.moved-*` files.
- After 2 stable weeks, and with owner approval, rename `OneDrive\trading_ibkr` to `_RETIRED_<date>`.
- Known gaps left on purpose:
  - Multi-leg Close on SPY/QQQ/IWM/DIA is refused, because the reservation guard cannot prove a BAG reduces. Close those one leg at a time.
  - Roll risk ignores legs that stay held, so rolling the short leg of a covered call needs the unbounded ack.
  - There is no safe-restart/drain mode for ExecAgent: `docs/exec_agent_restarts_2026-10-09.md` in the repo.
  - SHADOW Input Archive and an orphan `.vbs` still point at OneDrive.
  - `refresh_intraday_replay.py` breaks once the futures staging data is renamed.
- Pre-existing New_Seasonals test failures, unrelated:
  - `test_execution_fill_reconcile.py` x2: the data: URL import.
  - `test_runtime_unicode_guard.py`.
  - `test_shared_risk_redaction.py`.

## Owner preferences that apply

- Do NOT build alerts for unprotected/stopless positions. Stopless positions are intentional. Exit-size mismatch notices are low priority, at most daily.
- Priorities: orders go through exactly as intended; options trading and visualization at parity with the IBKR desktop app; the repo cutover.
- Never place orders, connect to IBKR or change arming without explicit approval. Never print secrets.
