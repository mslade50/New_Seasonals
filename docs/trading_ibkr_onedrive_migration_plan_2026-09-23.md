# trading_ibkr: moving live IBKR code and state out of OneDrive

Survey date: 2026-09-23, read-only. Nothing was edited, stopped or re-registered.
The trigger: an `os.replace` onto a `data/position_actions` record failed with
WinError 5 on 2026-09-23 (the NOVT close, left at `phase=attention`) because
the OneDrive sync client had the destination open. `position_actions.replace_with_retry`
now retries with backoff. That is a mitigation. The fix is to get the state off
a synced volume.

## 1. The private repo already exists

| Item | Finding |
|---|---|
| GitHub | `mslade50/trading_ibkr`, PRIVATE, last push 2026-09-21 18:23 UTC |
| Local clone | `C:\Users\McKinley Slade\dev\trading_ibkr`, branch `main`, clean, in sync with `origin/main` (`1b7818e`) |
| History | 2 commits: `e66df8d` "Initial import of trading_ibkr from OneDrive" (2026-09-21 13:53 ET) and `1b7818e` "Add CUTOVER.md" (14:23 ET) |
| Tracked files | 128 (code, wrappers, task XML, docs, tests, `contract_reference.json`, `inventory_history_policy.json`, `legend_ema.env.example`) |
| `.gitignore` | Covers `credentials.json`, `*.env` (keeps `*.env.example`), `data/`, `artifacts/`, `auction_intents/`, `logs/`, `*.jsonl`, `*.lock`, `*.flag`, `*_placed*.json`, `*_pending.json`, `*_manifest.json`, `*_held_snapshot.json`, `*_last_result.json`, `staged_orders.csv`, `morning_orders.json`, `_backup_*/`. Verified with `git check-ignore`. No secret file appears in any commit. |
| `CUTOVER.md` | A 7-step checklist. Its own status line says "Nothing below has been done." The live system still runs from OneDrive. |
| Working tree | Also holds gitignored copies of state and secrets copied on 2026-09-21. They are STALE: 27 position-action records against 36 + 15 order-edits live, `exec_agent_seen.jsonl` 81,027 B against 85,483 B, and `exec_agent.env` differs by a byte. Never point a task at this tree without re-syncing state first. |

### Drift, OneDrive against `HEAD` (SHA-256, EOL-normalised)

119 of 128 tracked files are identical. 2 exist only in the repo (`.gitignore`, `CUTOVER.md`).

The 7 below differ. All but one were edited in OneDrive after the import:

| File | OneDrive mtime | Note |
|---|---|---|
| `exec_agent.py` | 09-23 12:25 | 2026-09-23 exec fixes (`7325f8e8` in New_Seasonals) |
| `execute_order.py` | 09-23 12:25 | same; OneDrive also holds `execute_order.py.bak_2026-09-23` |
| `manual_order_actions.py` | 09-23 12:25 | same |
| `position_actions.py` | 09-23 12:47 | WinError 5 retry (`replace_with_retry`) |
| `broker_reconciliation.py` | 09-23 12:47 | same fix set |
| `test_olv_exits.py` | 09-22 08:08 | |
| `pa_nightly_report.py` | 06-23 | Intentional divergence. The OneDrive copy HARDCODES `GMAIL_APP_PASSWORD` on line 40; the repo copy reads it through `_env_secret`. **Never copy OneDrive's version into git.** |

There are 8 code files in OneDrive that are not in the repo. All are new since the import except the first:
`_tmp_list_tasks.ps1` (ignored on purpose), `legend_ema.py`, `test_legend_ema.py`,
`register_legend_ema_task.ps1`, `run_legend_ema.bat`, `oca_revision_probe.py`,
`test_exec_fixes_20260923.py`, `test_exit_timing_fields.py`. The Legend EMA
runner went live on 2026-09-23 and is not in git at all.

## 2. Runtime inventory

### 2a. Scheduled tasks that execute OneDrive code

All run as Interactive logon under `DESKTOP-2KI41V6\McKinley Slade`.

| Task | Schedule (ET) | State | Action | Working dir |
|---|---|---|---|---|
| `ExecAgent` | daily trigger from 05:00, repeats every 5 min, plus logon; the wrapper self-gates to 05:00-21:00 | **Running** | `powershell -File ...\OneDrive\trading_ibkr\run_exec_agent.ps1` | none; the wrapper `Set-Location`s to the hardcoded `$RuntimeDirectory` |
| `IBKR Daily Order Chain` | Mon-Fri 09:31 | Ready | `...\OneDrive\trading_ibkr\run_order_staging.bat` (order_staging, then eq_order_entry, pa_order_entry, morning_order_summary) | none (`%~dp0`) |
| `IBKR Dividend Adjust (Evening)` | Mon-Fri 17:00 | Ready | `run_div_adjust.bat evening` | none |
| `IBKR Dividend Adjust (Pre-Open)` | Mon-Fri 09:15 | Ready | `run_div_adjust.bat 0915` | none |
| `IBKR Event Sleeve Auction Orders` | Mon-Fri 09:05 | Ready | `run_event_moo.bat` | OneDrive dir |
| `IBKR OLV Pre-Market Exits` | Mon-Fri 09:10 | Ready | `run_olv_exit_moo.bat` (primary runner, then PA legacy runner) | OneDrive dir |
| `IBKR Legend EMA` | Mon-Fri 09:29, long-lived to about 10:32 | Ready | `run_legend_ema.bat` | OneDrive dir |
| `IBKR Legend EMA Verify` | Mon-Fri 10:40 | Ready, never run | `run_legend_ema.bat --verify-only` | OneDrive dir |
| `IBKR IV History` | Mon-Fri 17:15 | Ready | `run_iv_history.bat` (hardcoded `cd`) | OneDrive dir |
| `IBKR OLV Book Cap (EOD)` | Mon-Fri 15:40 | **Disabled** | `run_olv_book_cap.bat` | OneDrive dir |
| `PA Nightly Portfolio Report` | Mon-Fri 16:35 | Ready | `run_pa_nightly_report.bat` | none |

These tasks do not live in OneDrive but read from it:

| Task | Schedule | Dependency on OneDrive |
|---|---|---|
| `New Seasonals Expected Exit Monitor` | every 1 min | `-ExecEnv "C:/Users/McKinley Slade/OneDrive/trading_ibkr/exec_agent.env"` in the task arguments |
| `Radar Weekly Sync` | Mon 08:50 | `scripts\run_radar_sync.bat` sets `IBKR=%USERPROFILE%\OneDrive\trading_ibkr` and runs `radar_trail_sync.py` from there. Arming flag `radar_trail_enabled.flag` (absent, so preview only). |
| `New Seasonals Local v9 - *` (premarket, postclose, execution, inventory-close, ...) | catalog | `automation_supervisor.resolve_credential_paths` defaults to `~/OneDrive/trading_ibkr/credentials.json` and `exec_agent.env`. The `LOCAL_AUTOMATION_GCP_JSON_PATH` / `LOCAL_AUTOMATION_EXEC_ENV_PATH` overrides exist, but none is set in `New_Seasonals\.env`. |
| `Daily Pitch (agent)` | Mon-Fri 05:10 | `daily_pitch.py:194` reads `~/OneDrive/trading_ibkr/credentials.json` |
| `NewSeasonals-LegendETF-Signals/Session/Watchdog` | 08:45 / 09:28 / 10:40 | `%LOCALAPPDATA%\NewSeasonals\legend_etf\runtime.env` has `LEGEND_ETF_EXECUTOR_ROOT="C:/Users/McKinley Slade/OneDrive/trading_ibkr"`. Memory says this Codex runner is abandoned, but all three tasks are still Ready and ran today. |

These tasks are registrable but not registered: pitch_moo (auction and open passes), trend_moo, intraday_store, option snapshot. `C:\Scripts` has no OneDrive references, and its four GHA triggers are disabled.

OneDrive-side tasks that make locking worse:
- `Python\one drive refresh` runs `OneDrive\onedrive_refresh.py`. It **`taskkill /f` kills and restarts OneDrive at 04:46 and 14:39 daily**. Both slots fall in or near trading windows, and each restart forces a full rescan that opens every file.
- `OneDriveWatchdog_Day` relaunches OneDrive every 5 min.

OneDrive cannot simply be switched off, because `Scott_stops` (SecuriSync) depends on it.

### 2b. Long-running processes (observed 14:37 ET)

| PID | Process | Note |
|---|---|---|
| 22168 / 37872 | `run_exec_agent.ps1` then `python -u ...\OneDrive\trading_ibkr\exec_agent.py` | Started 14:16. WS to the broker DO. Spawns `book_snapshot.py`, `execute_order.py`, `option_quote.py`, `option_workbench.py` and `futures_front.py` as subprocesses from `_DIR`. |
| 14472 | `cmd /c run_legend_ema.bat --test-window 14:25 --force-entry QQQ` | A manual Legend EMA test run was in progress during the survey. |
| 34544 | `C:\Jts\ibgateway\1047\ibgateway.exe`, listening on **7496** | Primary. The code labels it "TWS"; it is a Gateway. |
| 27796 | `ibgateway.exe`, listening on **4001** | PA |
| 19328 / 19716 / 43780 | OneDrive.exe, OneDrive.Sync.Service, FileCoAuth | |

`book_snapshot.py` is not its own task. `exec_agent` runs it on every book push.

### 2c. Files written at runtime (all resolve to the code dir through `__file__`)

Classification: **C** = critical order-state (a lost, rolled-back or forked copy can cause a duplicate or missing order), **S** = safety config (arming), **L** = log or status, **K** = config or reference data.

| Path (relative to the trading_ibkr root) | Writer, and how | Scope | Class |
|---|---|---|---|
| `data/position_actions/<sha>.json` (36) + `order_edits/<sha>.json` (15) + `operations.lock` | `position_actions.save` / `manual_order_actions` / `order_mutations`: `.pending`, fsync, `os.replace` with retry. Override `POSITION_ACTION_STATE_DIR` exists, but only as a namespace value, not an env var. | cross-day | **C**. This is where the WinError 5 hit. |
| `exec_agent_seen.jsonl` + `.lock` | `exec_agent` append under an exclusive lock. It is the command dedup record. | cross-day | **C**: rollback means replayed broker commands |
| `scheduled_option_intents.json` (absent now) | `exec_agent`: `.tmp` then `os.replace` (the second WinError 5 comment, line 195) | cross-day | **C** |
| `eq_placed_orders.json` | `eq_order_entry.journal_placed`: **non-atomic `open('w')`, and missing or corrupt reads as empty** | today only | **C**: a truncated write during a lock means a same-day re-run can double-place |
| `pa_placed_orders.json` | `pa_order_entry`, same pattern | today only | **C** |
| `olv_exit_primary_placed.json` | `olv_exit_primary`: mkstemp then `os.replace` | cross-day | **C** |
| `olv_exit_placed.json` + `.lock` | `olv_exit_pa_legacy`: mkstemp then `os.replace` | cross-day | **C** |
| `auction_intents/<sha>.json` (`event_migration.json`) | `auction_lifecycle.claim`: `O_EXCL` create, fsync. It is event_moo's durable pre-submission claim. | cross-day by design | **C** |
| `legend_ema_journal.jsonl` | `legend_ema` append | cross-day | **C** |
| `staged_orders.csv` | `order_staging` (rewritten 09:31), then read by `eq_order_entry` and by `book_snapshot` for entry metadata | intraday handoff | **C** during the 09:31 chain |
| `div_adjust_held_snapshot.json`, `div_adjust_pending.json`, `div_adjust_manifest.json` | `div_adjust` `write_text` (non-atomic). The 17:00 evening pass is read by the 09:15 pass, **including Fri to Mon**. | cross-day | **C** (a missed or double ex-div order adjustment) |
| `olv_book_cap_journal.json` (absent; task disabled) | `olv_book_cap`, atomic | cross-day | **C** if re-enabled |
| `event_moo_enabled.flag`, `legend_ema_enabled.flag` (present); `pitch_moo_`, `trend_moo_`, `radar_trail_enabled.flag` (absent) | hand-made | | **S**: a sync restore or delete silently arms or disarms live money |
| `exec_agent.env` (`AGENT_LIVE_ENABLED`, `LIVE_*` caps, tokens) | `arm_live.bat` / `disarm_live.bat` (`findstr`, `.tmp`, `move /y`) | | **S** + secret |
| `options_journal.jsonl` | `exec_agent` append | | L (audit) |
| `*_last_result.json` (event, pitch, trend, legend), `morning_orders.json`, `div_adjust_heartbeat.txt`, `intraday_store_heartbeat.txt`, `div_adjust_audit_*.csv` | status writers. `publish_sleeve_runtime_status.py` reads the flags. | | L |
| `*_last_run.log`, `logs/*.log`, `logs/oca_probe_*.jsonl` | wrapper redirection | | L |
| `data/iv_history.parquet`, `data/option_vol_state.parquet`, `data/universe_liquid.json`, `data/intraday/`, `data/option_position_snapshots.parquet` | `iv_history_update`, `option_workbench`, `intraday_store` (atomic `.tmp`), `option_snapshot_recorder` | | K (rebuildable, also mirrored to R2) |
| `contract_reference.json`, `inventory_history_policy.json`, `dividend_calendar.csv` | tracked / generated reference | | K |

**Count: 12 critical order-state stores live in OneDrive today.** Ten are C-class stores (`position_actions` with its 51 records, `exec_agent_seen`, `eq_placed`, `pa_placed`, `olv_exit_primary_placed`, `olv_exit_placed`, `auction_intents`, `legend_ema_journal`, `staged_orders.csv`, and the div-adjust trio as one store). The other two are the S-class flag and env stores. Two more C-class files (`scheduled_option_intents.json`, `olv_book_cap_journal.json`) appear on demand.

Outside OneDrive, for reference: `%LOCALAPPDATA%\NewSeasonals\legend_etf\` (Legend-ETF runtime state; `reservation_config.json` is absent, so the reservation guard is dormant) is already off the synced volume.

### 2d. Path coupling

Inside trading_ibkr (non-test):

| File | Coupling |
|---|---|
| `run_exec_agent.ps1:3` | `$RuntimeDirectory = 'C:\Users\McKinley Slade\OneDrive\trading_ibkr'`; Python path hardcoded |
| `register_exec_agent.ps1:3`, `ExecAgent.xml`, `ExecAgent_src.xml:43` | OneDrive path |
| `arm_live.bat:8`, `disarm_live.bat:4`, `run_iv_history.bat:3`, `run_option_snapshot.bat:3` | `cd /d "C:\Users\McKinley Slade\OneDrive\trading_ibkr"` |
| `run_legend_ema.bat:21` | Python path hardcoded (dir itself is `%~dp0`) |
| `trade_journal.py:60` | dead `C:\Users\mckin\OneDrive\trading_ibkr` |
| `intra_bt.py` | dead `C:\Users\mckin\Downloads`/`Dropbox` research paths |
| `r2_io.py:11` | `C:\Users\McKinley Slade\dev\New_Seasonals\.env` (R2 creds) |
| `morning_order_summary.py:63` | `SEASONALS_REPO` default `...\dev\New_Seasonals` |
| `radar_trail_sync.py:52` | `~/dev/radar-briefings` |
| `register_*_task.ps1` (10) | build actions from `$PSScriptRoot` / script dir, so re-running them from the new root repoints correctly |
| every other `run_*.bat` | `%~dp0`-relative, so it moves cleanly |
| **every state constant in 2c** | `Path(__file__).parent` / `SCRIPT_DIR`. Moving the code moves the state with it unless Phase 1 decouples them first. |

In New_Seasonals (plus each runtime worktree carries its own copy: `New_Seasonals-automation-runtime-v9`, `new-seasonals-legend-runtime`, `artifacts\runtimes\expected-exit-runtime-20260914`):

| File | Coupling |
|---|---|
| `scripts/automation_supervisor.py:1159` | default `Path.home()/"OneDrive"/"trading_ibkr"` for `credentials.json` + `exec_agent.env` |
| `scripts/publish_sleeve_runtime_status.py:80` | `--executor-root` default `~/OneDrive/trading_ibkr` (reads flags) |
| `daily_pitch.py:194` | `credentials.json` |
| `scripts/run_radar_sync.bat:29`, `scripts/register_radar_sync_task.ps1:50` | `%USERPROFILE%\OneDrive\trading_ibkr` |
| `scripts/run_exec_agent.ps1:3`, `scripts/register_exec_agent.ps1:3` | repo mirrors of the OneDrive launchers |
| `tests/*` (23 `IBKR_DIR` constants, e.g. `test_mongap_signal_date.py:44`, `test_trading_calendar.py`) | `~/OneDrive/trading_ibkr`; skip when absent. `IBKR_REVIEW_SOURCE`, `MANUAL_ORDER_TEST_SOURCE` and `EXEC_REPORTING_TEST_SOURCE` env overrides exist. |
| `broker_runtime/*_source_hashes.json` (5 files) | pin SHA-256 of OneDrive sources (`execute_order`, `exec_agent`, `book_snapshot`, `eq_order_entry`, `run_order_staging.bat`, `event_moo`, `trend_moo`, `olv_exit_moo`, `position_actions`, ...). `prepare*.py --source <checkout>` so the pins are path-agnostic, but they go stale as the repo moves. |
| `tests/fixtures/execution_runtime/` | frozen copies (`exec_agent_core.py`, `execute_order.py`, `execute_order_core.py`) |
| `CLAUDE.md`, `AGENTS.md`, docs, scratch | prose (about 60 docs/scratch files). Update by find-and-replace after cutover. |
| Task args | Expected Exit Monitor `-ExecEnv`; `%LOCALAPPDATA%\NewSeasonals\legend_etf\runtime.env` `LEGEND_ETF_EXECUTOR_ROOT` |

### 2e. Cross-deps and clientIds

Shared core imported by both the site executor and the book runners:
`legend_reservation_guard` (imported by nearly every placer: `eq_order_entry`, `pa_order_entry`, `event_moo`, `pitch_moo`, `trend_moo`, `olv_exit_*`, `div_adjust`, `execute_order`, `exec_agent`, `pa_*` utilities) -> `legend_portfolio_budget`; plus `sheets_client`, `trading_calendar_live`, `equity_sessions`, `execution_contracts`, `futures_sizing`, `pa_nightly_report` (imported for helpers by `div_adjust`, `olv_exit_*`, `olv_book_cap`, `morning_order_summary`).
`eq_order_entry` is itself a library for `event_moo`, `pitch_moo`, `trend_moo`, `sznl_*`, `futures_front`, `contract_reference`, `morning_order_summary`.
Site executor only: `exec_agent` -> `execute_order` / `book_snapshot` (subprocess) / `position_actions` / `position_action_agent` / `manual_order_actions` / `reconcile_position_exits` / `execution_lifecycle` / `order_edit_context` / `option_*`. `legend_ema.py` imports nothing local by design.
`legend_reservation_guard` hashes the executor source tree at import (`PROCESS_EXECUTOR_SOURCE_TREE_SHA256`) and validates `.legend_reservation_guard_required.json` against `Path(__file__).parent`. The marker is absent today, so this does not bind now. It WILL bind once the Legend-ETF guard is armed, and the executor root it pins then has to be the new root.

| clientId | Port | Owner | Mutates |
|---|---|---|---|
| **99** | 7496 | `eq_order_entry`, `order_staging` (primary), `olv_exit_primary`, `olv_book_cap`, `div_adjust` owner | **places and owns every book bracket on Primary** |
| **98** | 4001 | `pa_order_entry`, `pa_moc_entry`, `reconcile`, `pa_one_off`, `pa_positions`, `olv_exit_pa_legacy`, `div_adjust` owner | **owns PA brackets** |
| 123 / 147 | 7496 / 4001 | `execute_order` (site transmit) | site brackets; re-connects AS the owner id to modify legacy brackets |
| 122 / 146 | 7496 / 4001 | `book_snapshot` (read) | no |
| 147 / 148 / 146 | 7496 | `event_moo` / `pitch_moo` / `trend_moo` | yes (auction orders) |
| 161 | env | `legend_ema` | yes |
| 100 / 101 | 7496 | `sznl_entry`/`moc_entry` / `sznl_exit`, `contract_reference` | yes / read |
| 151 / 152 | | `div_adjust` probes | read |
| 130 | 4001 | `order_staging` PA read | read |
| 132, 133, 134, 135, 162, 144, 143, 142, 97, 96, 77, 105, 172 | | read-only probes, PA utilities, tests | mostly read |

Brackets bind to the **clientId**, not the file path. A code move does not orphan any live order, provided the new copy uses the same ids. The real hazard is **two copies running at once with the same id**. IBKR refuses the second connection (error 326), so a runner silently fails, or one copy mutates while the other holds stale journals.

### 2f. Secrets (paths only)

| Path | Holds | In the private repo |
|---|---|---|
| `OneDrive\trading_ibkr\credentials.json` | GCP service account (Sheets) | ignored, never committed |
| `OneDrive\trading_ibkr\exec_agent.env` | `EXEC_AGENT_TOKEN`, `STATUS_TOKEN` (HMAC), `EXEC_BROKER_WS`, the live arming switch and caps | ignored (`*.env`) |
| `OneDrive\trading_ibkr\legend_ema.env` | account, host, port, clientId and sizing | ignored (`*.env`); `.example` tracked |
| `OneDrive\trading_ibkr\pa_nightly_report.py:40` | **hardcoded Gmail app password, in OneDrive cloud storage** | the repo version is clean. Rotate the password. |
| `dev\New_Seasonals\.env` | R2 creds, read by `r2_io.py` | not in the trading repo |
| `%LOCALAPPDATA%\NewSeasonals\legend_etf\runtime.env` | Legend-ETF runtime config | n/a |
| `dev\trading_ibkr\credentials.json`, `exec_agent.env` | stale duplicates (ignored) | ignored |
| `HKCU\Environment\GH_PAT_NEW_SEASONALS` | GitHub PAT | n/a |

Everything in OneDrive, secrets included, is replicated to Microsoft's cloud and to any other device signed into the account.

## 3. Phased plan

Target layout, all on a local non-synced disk:

```
C:\trading\runtime\trading_ibkr-<gen>\   pinned worktree of mslade50/trading_ibkr (Phase 2)
C:\trading_state\trading_ibkr\           TRADING_IBKR_STATE_DIR: every C, S and L file from 2c, same relative layout
C:\trading_state\secrets\                credentials.json, exec_agent.env, legend_ema.env (ACL: owner only)
```

Use `C:\trading_state` rather than anything under `%USERPROFILE%`. Some users have Known-Folder backup redirect Desktop and Documents into OneDrive, and C:\ root is outside every sync scope. Confirm Defender/AV exclusions for the dir. AV was the co-cause named in the retry docstring.

**No-trade windows.** No step below runs in any of these:
- Weekdays 04:00-10:45 ET: premarket, 09:05 event, 09:10 OLV, 09:15 div, 09:29-10:40 Legend EMA, 09:31 chain.
- 15:30-17:30 ET: book cap slot, 16:05 inventory-close, 16:30 execution, 16:35 PA report, 17:00 div evening, 17:10 postclose, 17:15 IV.
- Any time ExecAgent is live and a site command could be in flight.

**Preferred window: Saturday, or a weekday after 21:05 ET** (ExecAgent self-exits at 21:00). Saturday also empties the today-scoped journals (`eq_`/`pa_placed_orders`), leaving only cross-day stores to carry.

### Phase 1: state out of OneDrive, behind one variable

Code change, made in `dev\trading_ibkr` first, then synced into OneDrive as the live copy:
1. Add `runtime_paths.py`: `STATE_DIR = Path(os.environ.get("TRADING_IBKR_STATE_DIR") or Path(__file__).parent)`, plus `SECRETS_DIR` (`TRADING_IBKR_SECRETS_DIR`, same fallback) and `state_path(name)`. The default equals today's behaviour, so the change can ship days before the switch.
2. Route every writer and reader in 2c through it: `position_actions.journal_root`, `manual_order_actions` (line 185), `order_mutations`, `position_action_agent` (line 61), `exec_agent` (`_SEEN_PATH`, `OPTIONS_JOURNAL_PATH`, `SCHEDULED_OPTIONS_PATH`), `eq_order_entry.PLACED_JOURNAL` and the `staged_orders.csv` path, `pa_order_entry`, `olv_exit_primary`, `olv_exit_pa_legacy`, `olv_book_cap`, `event_moo` (auction_intents at 325/332, `RESULT_PATH`, `ENABLE_FLAG`), `pitch_moo`, `trend_moo`, `legend_ema` (`JOURNAL_PATH`, `RESULT_PATH`, `ENABLE_FLAG`, `ENV_PATH` to `SECRETS_DIR`), `div_adjust` (lines 94-102), `order_staging.OUTPUT_FOLDER`, `morning_order_summary`, `book_snapshot` (the `staged_orders.csv` + `inventory_history_policy.json` reads), `reconcile`, `iv_history_update`/`option_*`/`intraday_store` data dirs, and every `CREDENTIALS_FILE = 'credentials.json'` to `SECRETS_DIR`.
3. While there, make `eq_order_entry.journal_placed` and `pa_order_entry`'s writer atomic (mkstemp, fsync, `os.replace`), and make a missing journal on a day that already has placements fail loud instead of reading as empty. (The prepared `broker_runtime` `entry_journal.py` already does the second part.)
4. Wrappers. Each `run_*.bat` calls one `trading_env.cmd` that sets `TRADING_IBKR_STATE_DIR` / `TRADING_IBKR_SECRETS_DIR` when they are not already defined. `run_exec_agent.ps1` reads `exec_agent.env` from `SECRETS_DIR` and redirects `exec_agent_last_run.log` there. `arm_live.bat` / `disarm_live.bat` edit the env file in `SECRETS_DIR`. The bats' `>> "%DIR%\*_last_run.log"` redirects move to the state dir.
5. New_Seasonals: set `LOCAL_AUTOMATION_GCP_JSON_PATH` and `LOCAL_AUTOMATION_EXEC_ENV_PATH` in `New_Seasonals\.env` (already supported). Then fix `daily_pitch.py:194`, `publish_sleeve_runtime_status.py --executor-root` (flags now live in the state dir, so add `--state-dir`), `run_radar_sync.bat` (flag path), and the Expected Exit Monitor `-ExecEnv` task argument. The v9 runtime reads `.env` from ConfigRoot, so it picks up the overrides without a runtime release.

Cutover order (Saturday):
1. `schtasks /End /TN ExecAgent`, then disable ExecAgent. Confirm no `python` process has a trading_ibkr path in its command line, and confirm the broker DO has no queued commands.
2. Zip the whole OneDrive tree to `E:\` and to `C:\trading_state_backup\<ts>\`. Write a SHA-256 manifest of every file in 2c.
3. Pause OneDrive sync for 2 h. This also pauses SecuriSync, so do it on a weekend.
4. `robocopy` the 2c files into `C:\trading_state\trading_ibkr\` with the same relative layout (`/COPY:DAT /DCOPY:T`), and the three secrets into `C:\trading_state\secrets\`. Re-hash, and require a 100% match against the manifest.
5. Rename the OneDrive originals to `*.moved-2026-MM-DD`. Do not delete them: a code path that still resolves to the old location then finds nothing, and fails or reads empty, rather than quietly forking a second live copy. For `data/position_actions` and `auction_intents`, rename the directory. Do the same for the secrets once step 7 passes.
6. Set the user env vars (`setx`) AND `trading_env.cmd`, so both scheduler-spawned and hand-run processes resolve the same root.
7. Resume OneDrive. Re-enable and start ExecAgent. Verify that `exec_agent_last_run.log` grows in the state dir, the book pushes to the site, and a dry-run site command writes its `position_actions` record under `C:\trading_state`.
8. Monday: watch 09:05, 09:10, 09:15, 09:29 and 09:31. Every `*_last_run.log`, journal and heartbeat must update in the state dir, and nothing new must appear in OneDrive.

Rollback: disable the tasks, unset the vars (the fallback is the code dir), rename the `*.moved` originals back, and copy back any state written since the switch. The newer copy wins; merge the `position_actions` and `auction_intents` records by file name, since they are content-addressed. The code change is inert without the var, so no code rollback is needed.

Tests:
- New `test_runtime_paths.py`: a grep-based test that fails if any module builds a state path from `__file__`/`SCRIPT_DIR` except through `runtime_paths`. It also runs each writer against a `tmp_path` state dir with the code dir read-only.
- The existing OneDrive suite (`test_olv_exits`, `test_eq_order_entry_dedup`, `test_event_moo`, `test_pitch_moo`, `test_trend_moo`, `test_legend_ema`, `test_exec_fixes_20260923`, `test_olv_book_cap`, `test_radar_trail_sync`, ...) run once with the var unset and once set.
- New_Seasonals: `tests/test_manual_order_actions.py`, `test_position_action_adapter.py`, `test_scheduled_options.py`, `test_order_entry_dedup.py`, `test_event_runner_identity.py`, pointed at the source through `IBKR_REVIEW_SOURCE`.
- `repo_health_check`.
- Live proof: on the first Monday, every file in 2c is newer in `C:\trading_state` than its `.moved` twin.

### Phase 2: code into the private repo, run from a pinned worktree

1. Reconcile drift. Commit the 6 changed OneDrive files and the 7 new code files (all but `_tmp_list_tasks.ps1`) plus the Phase 1 changes into `dev\trading_ibkr`. **Keep the repo's `pa_nightly_report.py`**, then rotate the Gmail app password, because the OneDrive copy has sat in cloud storage in plaintext. Delete `execute_order.py.bak_2026-09-23` and the `_backup_*` dirs from consideration (git replaces them). Run `git diff` against OneDrive until only ignored files differ.
2. Mirror New_Seasonals' local-runtime pattern (`docs/local_automation_task_scheduler.md`):
   - a dedicated branch `runtime`;
   - an immutable tag `ibkr-runtime-<YYYY-MM-DD>.<k>` on the exact tested SHA;
   - `git worktree add C:\trading\runtime\trading_ibkr-g1 <tag>`, clean with no local edits;
   - a marker file recording `{generation, sha, tag, runtime_root, state_dir}`;
   - optionally a dedicated venv (today everything uses the system `Python310`; pin `ib_insync`, `gspread`, `pandas`, `boto3` versions first, and ask before any `pip install`).
   Scheduled runs never `git pull`.
3. Rebase the pins: regenerate `broker_runtime/*_source_hashes.json` against the tagged SHA, and point the `IBKR_DIR` test constants at an env var (`TRADING_IBKR_SOURCE`, defaulting to the runtime root).
4. Nothing has moved yet at this point. OneDrive stays live. Validate the worktree with a dry run, pointed at the Phase 1 state dir but read-only: `olv_book_cap --dry-run`, `legend_ema --dry-run`, `book_snapshot` via the site book, and `morning_order_summary`.

Tests: the full OneDrive suite from inside the worktree; a byte-for-byte comparison of the worktree against the OneDrive tree (ignored files excluded) showing only intended diffs; `python -c "import legend_reservation_guard as g; print(g.PROCESS_EXECUTOR_ROOT)"` resolving to the worktree.

### Phase 3: repoint tasks, retire OneDrive

Order (one cutover per day at most; Saturday, or after 21:05 ET):
1. Disable every task in 2a. End ExecAgent, and confirm no runner is alive (clientIds 99/98/123/147 are all free: `Get-NetTCPConnection -RemotePort 7496,4001` shows only the Gateways' own listeners).
2. Run each `register_*_task.ps1` from the worktree (they derive paths from `$PSScriptRoot`), so tasks are re-registered **with the same names**. For ExecAgent, regenerate `ExecAgent.xml` from `register_exec_agent.ps1` with `$dir` pointing at the worktree. Re-registering Legend EMA must keep its long execution time limit. Update the disabled book-cap task too, so it cannot revive the OneDrive copy. `IBKR IV History` and the option snapshot wrapper need their hardcoded `cd` removed first (Phase 1).
3. Repoint the out-of-tree readers:
   - `scripts/run_radar_sync.bat` `IBKR=` goes to the worktree.
   - `scripts/run_exec_agent.ps1` and `register_exec_agent.ps1` in New_Seasonals.
   - `LEGEND_ETF_EXECUTOR_ROOT` in `%LOCALAPPDATA%\NewSeasonals\legend_etf\runtime.env`. Better still, retire the three LegendETF tasks if that runner really is abandoned.
   - `automation_supervisor` / `publish_sleeve_runtime_status` defaults.
   - `CLAUDE.md` / `AGENTS.md` / memory `trading-ibkr-local-path.md`.
4. Rename `OneDrive\trading_ibkr` to `OneDrive\trading_ibkr_RETIRED_<date>`, with state and secrets already gone. Any stale caller now fails loudly on a missing path. Keep it for 2 weeks, then delete it after explicit approval. Do the same for the stale `dev\trading_ibkr` working-tree state copies: they become the dev checkout, whose ignored state must never be used.
5. Re-enable the tasks, then watch a full trading day as in Phase 1 step 8.

Rollback: disable the tasks, rename the OneDrive dir back, and re-run the OneDrive copy's `register_*_task.ps1`. The state dir is shared by both code copies, so no state moves in either direction; that is why Phase 1 comes first. The one condition is that a rolled-back OneDrive copy must include the Phase 1 `runtime_paths` change, or it would write to the old location again.

Tests: the task action audit (the tasks.ps1 query this survey used) must show zero actions containing `OneDrive`; `Get-CimInstance Win32_Process` must show no command line containing `OneDrive\trading_ibkr`; `repo_health_check`; the first live day's journals.

## 4. Risks

- **Open positions and resting brackets.** Every book bracket is owned by clientId 99 (Primary) or 98 (PA), and TWS lets only the owning id modify them. The move is safe **only if clientIds are unchanged** and only one copy ever connects. Never run a test or dry run with 99/98 while the other copy's task is enabled. OLV exits, div adjust and book cap all connect as 99/98 on purpose.
- **OCA state.** OCA groups live at the broker, not in files, so a move cannot break them. What can break them is a runner that crashes mid-mutation (for example `execute_order` close_resize: shrink the exits, then send the close). Its only recovery record is `data/position_actions`. Carry that directory over byte-exact, and make sure no action record is in a non-terminal `phase` (grep for `attention`/`pending` before cutover; the NOVT record from 2026-09-23 must be resolved first).
- **Orders mid-flight.** `auction_intents` claims and `olv_exit_*_placed` entries are the idempotency proof for orders already sent. A rolled-back copy lets a runner send the same MOO/OPG again. Never cut over between an 09:05/09:10 placement and that session's open. A Friday 17:00 `div_adjust_held_snapshot.json` is consumed Monday 09:15, so it must carry over a weekend cutover.
- **Journals must carry exactly.** `exec_agent_seen.jsonl` (command dedup), `legend_ema_journal.jsonl`, `options_journal.jsonl` and `scheduled_option_intents.json` are append-only or cross-day. Copy them with a hash check and never merge by hand. Two live copies are worse than one stale copy.
- **Fail-open journals.** `eq_placed_orders.json` / `pa_placed_orders.json` read missing or corrupt as empty, and are written non-atomically. OneDrive can truncate them today. Fix this in Phase 1.
- **Arming drift.** Flags and `exec_agent.env` decide whether real money moves. After each phase, diff the `*_enabled.flag` set and the `AGENT_LIVE_ENABLED`/`LIVE_*` keys against the pre-cutover snapshot. An absent pitch or trend flag must stay absent.
- **OneDrive's own restarts** at 04:46 and 14:39 are the likeliest cause of the lock storms. They stay harmful to any file left behind until Phase 3 completes. Consider moving the 14:39 run outside market hours in the meantime; that is the user's call, and this survey did not do it.
- **Guard attestation.** Once `.legend_reservation_guard_required.json` and the reservation config are armed, the executor root and source-tree hash are pinned. Arm them only after Phase 3, against the worktree path.
- **Secrets.** The plaintext Gmail app password in `OneDrive\trading_ibkr\pa_nightly_report.py` and the executor tokens in `exec_agent.env` have been replicated to OneDrive cloud storage. Rotate `EXEC_AGENT_TOKEN`, `STATUS_TOKEN` and the Gmail app password after Phase 1.

## 5. Real-time test plan

Added 2026-09-23 from a read-only dry-run audit of every runner in 2a/2b. Nothing was run. Scope is the cutover only. Per the user's scope decisions, this plan does NOT include rotating the Gmail app password, moving the `one drive refresh` task, or hardening `eq_`/`pa_placed_orders.json` as a cutover prerequisite. Where earlier sections suggest those things, this section does not depend on them.

All `file:line` refs are to `OneDrive\trading_ibkr` unless prefixed.

### 5.0 Audit findings the plan has to design around

- **Every state path is `__file__`-relative today, and no runner can read state from another root.** 54 `Path(__file__)`/`dirname(__file__)`/`*_DIR =` lines sit in 32 non-test modules. Five modules also use a bare, cwd-relative `CREDENTIALS_FILE = 'credentials.json'`: `order_staging.py:160`, `pa_order_entry.py:17`, `olv_exit_primary.py:38`, `olv_exit_pa_legacy.py:117`, `moc_orders.py:10`. The only existing overrides are these:
  - `LEGEND_EMA_CLIENT_ID` and the env keys of `legend_ema.env` (`legend_ema.py:170-195`);
  - the csv argument to `eq_order_entry --dry-run` (`:880`);
  - `radar_trail_sync --recs`;
  - the Expected Exit Monitor's CLI paths;
  - `%LOCALAPPDATA%` for Legend-ETF.

  So **shadow runs need the Phase 1 `runtime_paths` change in the shadow's code.** Until then, the only way to isolate state is a throwaway code copy with its own state copy beside it.
- **Dry-runs that still write SHARED state (dangerous):**
  - `exec_agent` with `AGENT_LIVE_ENABLED=0` still sends on the broker WS: hello and heartbeat (`exec_agent.py:2095/2106`), a `book` push every 20 s tagged `mode:"dry-run"` (`:2116-2118`) that would overwrite the site's live book, `result` replies for every command it receives (`:1959`), and scheduled-intent re-sends (`:1780`). `position_action_agent.report_completed_edits` re-publishes `result`s from whatever `order_edits` it sees (`position_action_agent.py:88,117-134`). Dedup is local only (`exec_agent.py:120-138`). A second agent holding the same token is unsafe.
  - `order_staging` has no dry-run. It clears and rewrites the Sheets tabs `execution` (`order_staging.py:1625`) and **`execution_2` (`:1647`), which is `pa_order_entry`'s only input**. A shadow run would overwrite prod's PA orders.
  - `morning_order_summary` always sends two Gmail messages (`:855-857`) and uploads to R2 (`:804-806`).
  - `div_adjust` without `--confirm` still emails whenever anything is actionable (`:992-997`).
  - `pa_nightly_report` always emails (`:411`, `:383`).
  - `iv_history_update` (`:149`) and `option_snapshot_recorder` (`:155-157`) always upload to R2. An empty `R2_*` does not disable the upload: `r2_io.py:1-30` back-fills blanks from `dev\New_Seasonals\.env`, so only a non-empty bogus `R2_BUCKET` makes it fail (and it fails soft, `:62-64`).
  - `legend_ema --verify-only` and `--kill` ignore `--dry-run`: they cancel orders, send market exits and write the journal (`legend_ema.py:2125,2169,2180`).
- **A live money switch that the wrapper's own comment misdescribes.** `div_adjust.py:80` has `LIVE_ENABLED = True`, and `run_div_adjust.bat:41` passes `--confirm`, so both scheduled passes transmit modifies. The comment at `run_div_adjust.bat:36-40` ("harmless dry-runs") is stale. Treat `div_adjust` as a live placer in 5.5.
- **Shadow copies must hold no `*_enabled.flag`.** `event_moo_enabled.flag` and `legend_ema_enabled.flag` exist today. With a flag present, a run without `--check`/`--dry-run` places real orders against the copy's own journals, where prod's dedup cannot see them.
- **Undocumented clientIds, and three latent collisions.** Section 2e is missing several ids:
  - `morning_order_summary` 121 (7496) / 145 (4001) (`:66-67`)
  - `olv_book_cap` dry-run probes 148 / 149 (`olv_book_cap.py:100-101`)
  - `pa_nightly_report` 144 (`:34-35`)
  - `iv_history_update` 134 (`:38`)
  - `option_snapshot_recorder` 135 / 149 (`:42-43`)
  - `intraday_store` 132 (`:62-63`)
  - Legend-ETF: 154 feed, 155/156 exec, 156 signals, 157 clock (`dev\new-seasonals-legend-runtime`: `session.py:422`, `ibkr_adapter.py:228-231`, `prepare_legend_etf_signals.py:109`, `run_legend_etf_session.py:92`)

  The collisions:
  - **148** on 7496: `pitch_moo` and the book-cap probe.
  - **149** on 4001: the book-cap PA probe and `option_snapshot`.
  - **156**: Legend signals and the Legend PA exec default.

  Today they are separated only by schedule. Only `LEGEND_EMA_CLIENT_ID` and the Legend-ETF feed/account ids can be overridden without a code change.

**Shadow clientId rule: shadow id = prod id + 100.** 99→199, 98→198, 122/146→222/246, 130→230, 121/145→221/245, 151/152→251/252, 148/149→248/249, 144→244, 134→234, 161→261. None of the +100 ids appears in 2e or in the list above. Before each shadow session, run `Get-NetTCPConnection -RemotePort 7496,4001` and confirm the chosen ids are not connected. Shadows only ever connect read-only.

### 5.1 Phase 1 zero-change proof

Goal: show that routing every path through `runtime_paths` changes nothing, before any file moves.

1. **Ship with the variable pointed at the current folder.** Set `TRADING_IBKR_STATE_DIR` and `TRADING_IBKR_SECRETS_DIR` to `C:\Users\McKinley Slade\OneDrive\trading_ibkr`, both in `trading_env.cmd` and with `setx`. The resolved paths are byte-identical to today's, so behaviour cannot change. Any journal or order difference on the following day is therefore a bug in the change itself.
2. **Startup log line.** On import, `runtime_paths` prints one line, and every runner (including `exec_agent` and the `run_*.bat` headers) echoes it into its `*_last_run.log`:
   `[runtime_paths] code=<abs> state=<abs> secrets=<abs> source=env|fallback pid=<n>`
   Check: after one full session, `grep -h "\[runtime_paths\]" *_last_run.log logs\*.log` shows `source=env` and the OneDrive state path in every log that ran. Any `source=fallback` means a wrapper that did not source `trading_env.cmd`.
3. **Static proof (`test_runtime_paths.py`).** The test fails on:
   - any non-test `.py` other than `runtime_paths.py` that contains `__file__`, `SCRIPT_DIR`, `_DIR =` or `_THIS_DIR`, unless the line is allowlisted as code-relative rather than state (for example `legend_reservation_guard`'s source-tree hash, `sys.path` inserts, `book_snapshot`'s `inventory_history_policy.json` if it stays tracked config);
   - any bare state or secret filename literal (`'credentials.json'`, `*.jsonl`, `*_placed*.json`, `*.flag`, `staged_orders.csv`) that is not wrapped in `state_path()`/`secret_path()`. That covers the five cwd-relative `CREDENTIALS_FILE`s above;
   - any `.bat`/`.ps1` with a hardcoded `OneDrive\trading_ibkr` (`run_exec_agent.ps1:3`, `arm_live.bat:8`, `disarm_live.bat:4`, `run_iv_history.bat:3`, `run_option_snapshot.bat:3`).

   Baseline today: 54 hits in 32 modules plus 5 bare `CREDENTIALS_FILE`. The allowlist must be explicit and reviewed. The dynamic half (every writer against a `tmp_path` state dir, with the code dir read-only) is already in Phase 1's Tests list.
4. **Weekend copy with hash verification.** This expands Phase 1 cutover steps 2 and 4.
   - Before the copy, confirm that no `data/position_actions` record is in a non-terminal phase.
   - Build a manifest over the whole 2c set: `Get-ChildItem -Recurse -File | Get-FileHash -Algorithm SHA256`, plus per-directory file counts (for example 36 `position_actions` + 15 `order_edits`, and the `auction_intents` count).
   - `robocopy` the files. Re-hash the destination and require 100% equality of names, sizes and hashes, and equal counts. A single mismatch aborts the step; the original folder is still untouched at that point.
5. **Rename, never delete.** Rename the originals to `*.moved-<date>` (whole dirs for `data/position_actions` and `auction_intents`), then switch both variables to `C:\trading_state\...`.
6. **Monday check.**
   - At 09:12, 09:20, 09:40, 10:45 and 17:30, list every C/S/L file from 2c under `C:\trading_state\trading_ibkr` with `LastWriteTime`. Each runner that ran must have touched its files.
   - Missed-writer sweep: `Get-ChildItem "$env:USERPROFILE\OneDrive\trading_ibkr" -Recurse -File | ? { $_.LastWriteTime -gt $cutover -and $_.Extension -notin '.py','.bat','.ps1','.md','.xml' -and $_.Name -notlike '*.moved-*' }`.
   - **Any hit is a missed writer: a file recreated at its old path.** Identify the writer from the file name and 2c, and stop that runner's next run. The `.moved` copy stays as the reference. Merge the stray records into `C:\trading_state` by content-addressed name before the next run.

### 5.2 Code parity gate

Run this after Phase 2 step 1 has committed the drift (the 6 changed files, the 7 new code files and the Phase 1 change) and the worktree is created.
- `parity_gate.py` (new, lives in the repo):
  - It walks the worktree's tracked files and the OneDrive tree, and compares SHA-256 after CRLF→LF normalisation.
  - **PASS requires no content diff and no untracked OneDrive code file** (`*.py/*.bat/*.ps1/*.xml/*.json` that is not gitignored).
  - It has exactly two allowlist entries: `pa_nightly_report.py` (the repo copy is intentionally different, see §1) and `_tmp_list_tasks.ps1`.
  - It prints the gate time and the worktree SHA/tag.
- Code freeze: any OneDrive code file with an mtime newer than the last PASS invalidates the gate. OneDrive code was edited as recently as 09-23 12:47.
- Re-run the gate within 24 h before **every** task switch in 5.5, and again after any hotfix. A hotfix goes into the repo, then gets a new tag and generation, then a gate re-run. It never goes into OneDrive alone.

### 5.3 Shadow runs

**Setup** (once Phase 1 code is in the worktree):
- Shadow state dir: `C:\trading_shadow\<YYYY-MM-DD>\state`. About 2 minutes BEFORE each prod run, `robocopy` a fresh snapshot of `C:\trading_state\trading_ibkr` into it (or of the OneDrive folder, before Phase 1 cutover), with every `*_enabled.flag` deleted from the copy. Taking the snapshot pre-run matters: a post-run copy would make the shadow's dedup journals report "already placed" and the shadow would plan nothing.
- Shadow secrets dir: `C:\trading_shadow\secrets`. It holds `credentials.json` (Sheets reads) and a stripped `exec_agent.env` with only `STATUS_TOKEN` and `EXEC_BROKER_URL`: no `EXEC_AGENT_TOKEN`, `AGENT_LIVE_ENABLED=0`, and no `GMAIL_APP_PASSWORD`.
- `shadow_env.cmd` sets:
  - `TRADING_IBKR_STATE_DIR` / `TRADING_IBKR_SECRETS_DIR` to the dirs above. It must set them **explicitly**, because after Phase 1 `setx` would otherwise hand the shadow the prod state dir.
  - `R2_BUCKET=shadow-null` (non-empty on purpose, see 5.0);
  - `LEGEND_EMA_CLIENT_ID=261`;
  - the +100 ids for any runner that has gained a `--client-id`/env override.
- Shadow env vars (`*_CLIENT_ID` overrides etc.) are set per process in the shadow launcher, never with `setx`: prod code reads the same names, and a bad or leaked value can break prod imports (e.g. `PA_NIGHTLY_REPORT_CLIENT_ID` silently disables div_adjust email).
- `pa_order_entry.py --dry-run` exists only in the repo copy (branch `shadow-enablers`). The OneDrive copy ignores argv, so running it there with `--dry-run` places LIVE orders as clientId 98.
- Shadow ids 249 collide: option_snapshot PA and olv_book_cap PA probe (as prod 149 does). Keep them schedule-separated.
- Shadows run from the worktree, by hand or from separate `SHADOW <name>` tasks. They are never the prod task names.

**Schedule** (weekdays, ET). This covers only SHADOW-READY runners, plus NEEDS WORK runners once their minimal change has landed.

| Prod run | Shadow | Command (worktree, `shadow_env.cmd` sourced) | Diff against (prod side) |
|---|---|---|---|
| Mon 08:50 radar | 08:55 | `python radar_trail_sync.py` (never the bat, never `--apply`) | prod `radar` run stdout table (`radar_trail_sync.py:286-324`) |
| 09:05 event_moo | 09:07 | `python event_moo.py --check` | prod `event_moo_last_result.json` `orders`, today's `auction_intents/*.json`, IBKR open orders by orderRef |
| 09:10 OLV exits | 09:12 | read-only: import `olv_exit_primary.load_exit_rows(sh, today)` (`:1255`) | prod `olv_exit_primary_placed.json` / `olv_exit_placed.json` rows for today |
| 09:15 div 0915 | 09:20 | `python div_adjust.py --asof <today>` (probes need the +100 override first) | prod `div_adjust_audit_<date>.csv` rows |
| 09:29 Legend EMA | start 09:30:30 | `run_legend_ema.bat --dry-run` (id 261) | prod `legend_ema_journal.jsonl` entries for today and `legend_ema_last_result.json` `symbols` |
| 09:31 chain | 09:36 | (a) `eq_order_entry.py --dry-run <copy of PROD staged_orders.csv>`; (b) after the change, `order_staging.py --no-sheets --client-id 199 --aux-client-id 230 --out-csv <shadow dir>\staged_orders.csv` then `eq_order_entry.py --dry-run <shadow csv>`; (c) after the change, `pa_order_entry.py --dry-run <csv export of prod execution_2>` | (a) prod `eq_placed_orders.json`; (b) prod `staged_orders.csv`; (c) prod `pa_placed_orders.json` |
| 09:31 summary | 09:40 | after the change, `morning_order_summary.py --no-email --no-r2` (221/245) | prod `morning_orders.json` |
| ExecAgent book | any time 11:00-15:00 | after the change, `book_snapshot.py` on 222/246 | the site book push at the same minute (orders by `perm_id`/`order_ref`, positions by con_id) |
| 15:40 book cap (disabled) | 15:35, A/B | `olv_book_cap.py --dry-run` from OneDrive (148/149) and from the worktree (248/249), back to back | each other's `print_plan` block (`olv_book_cap.py:1562-1581`) |
| 16:35 PA report | 16:40 | after the change, `pa_nightly_report.py --out <file>` (244) | the prod email HTML |
| 17:00 div evening | 17:05 | `div_adjust.py --asof <today>` | prod evening audit CSV |
| 17:15 IV | 17:25 | after the change, `iv_history_update.py --no-upload` (234) | prod `data/iv_history.parquet` (today's rows) |
| Expected-exit monitor | any time | `run_expected_exit_monitor.py` with its own `--state`/`--artifacts`, `--exec-env` pointing at the new secrets path, and no `--upload`/`--send` | prod latest `runs/<uuid>/status.json` |
| pitch/trend (not registered) | any time that day | `pitch_moo.py --check`, `trend_moo.py --check`, from both copies, each with its own shadow state dir | each other |

Never shadow `legend_ema --verify-only`, `--kill` or `--force-entry`, or the `run_radar_sync.bat` / `run_div_adjust.bat` / `run_order_staging.bat` wrappers.

**Diff method.** Use one normaliser (`shadow_diff.py`, new, in the repo). It maps every source into rows `{runner, key, symbol, side, qty, order_type, tif, lmt, aux, target, stop, order_ref}`:
- the dry-run stdout lines: `eq_order_entry.py:856-872`, the `--check` lines at `event_moo.py:402-406` / `trend_moo.py:358-362`, the `legend_ema.py:1800-1809` "would place" lines;
- `staged_orders.csv` (`order_staging.py:1555-1571` columns);
- the placed journals, `auction_intents`, the div audit CSV and `book_snapshot` JSON.

The key is `order_ref`/`signal_ref` (symbol + side + strategy + date + tranche), falling back to symbol+side+strategy. The `--check`/dry-run printers do not emit orderRef today (`event_moo.py:274`, `trend_moo.py:248` compute it only on the live path; `eq_order_entry`'s preview skips `signal_ref`). Adding it is a one-line change per runner and makes keying exact.

**Pass criterion: 3 consecutive sessions per runner with zero FAIL rows, at least 2 of them non-empty.**
- An empty-vs-empty day counts only as "no regression".
- For rare runners, add replayed history: `div_adjust --asof` on past ex-dates against their saved audit CSVs, and event/pitch `--check` on historical `Execute_On` dates where the Sheet still holds the rows.

Classification of diff rows:
- **FAIL:** any missing or extra key; a side, order type, TIF or orderRef mismatch; a qty mismatch when the input was identical (prod's own `staged_orders.csv` / `execution_2`); any shadow write outside `C:\trading_shadow`. Check the last one with the same mtime sweep as 5.1 step 6, run over `C:\trading_state` and OneDrive.
- **ACCEPTABLE, timing-dependent:**
  - Live-anchored prices (open, gap clamp, REL_CLOSE, Legend bar OHLC, book-cap marks) within max(0.5%, 0.1 ATR).
  - For re-derived sizing (`order_staging` from a live NAV), qty within max(1 share, 1%).
  - Book-cap trims that change because prod already trimmed.
  - Timestamps, run ids and log order.
  - Sheet rows edited between the prod and shadow reads. Confirm these by re-reading the Sheet revision history.

### 5.4 Site agent (not shadowable, so a staged live proof)

Two agents cannot share the broker WS, and `execute_order` has no preview mode (`execute_order.py:3287-3288`). So switch ExecAgent first, **disarmed**, and prove it in stages. The prerequisite is 5.5 Tier 0 done, with the ExecAgent switch made after 21:05 ET.
1. **Echo.** With `AGENT_LIVE_ENABLED=0`, start the worktree ExecAgent.
   - The log shows the `[runtime_paths]` line with `C:\trading_state`, and the site banner shows dry-run with a fresh book.
   - Send a signed `echo` command (`exec_agent.py:338-339`). The `result` comes back on the site, one line is appended to `C:\trading_state\trading_ibkr\exec_agent_seen.jsonl`, and the 5.1 sweep finds nothing new in OneDrive.
2. **Preview.** Send a bracket-entry command with `dry_run:true` (`exec_agent.py:1906/1924`).
   - Expect `state="dry_run"` and preview legs (`position_action_agent.py:50-57`), with a `position_actions` record written under `C:\trading_state`.
   - Repeat for one `manual_order_actions` edit preview, to exercise the owner-id path without transmitting.
3. **1-share far-limit, placed and cancelled.** Do this the next weekday between 11:00 and 15:00 ET, outside every no-trade window, with no other site command in flight.
   - Arm with `arm_live.bat`, with the `LIVE_*` caps at their minimum.
   - Send a 1-share DAY BUY LMT at about 20% below last. Use a liquid, cheap symbol that is not in the book, not in Legend reservations, and has no OCA siblings.
   - In IBKR, confirm clientId 123 (primary) or 147 (PA) (`execute_order.py:171`), orderRef in the `SYMBOL|ACTION|Strategy_Ref|Date` form (`:674-782`), and the order Submitted.
   - Cancel it from the site. Confirm it is Cancelled at the broker, the `position_actions` record is in a terminal phase, and the next book push no longer shows the order.
   - Restore the arming state from the pre-cutover snapshot (flags and `AGENT_LIVE_ENABLED`/`LIVE_*`), and diff the result against that snapshot.

### 5.5 Task switch order, by risk

For every switch:
- The new task is registered from the worktree under **`<name> [wt]`**, created **disabled**. The OneDrive task stays registered. This supersedes Phase 3 step 2's "same names" for the cutover period; renaming or deleting the old tasks comes after 2 stable weeks, with explicit approval.
- The switch is `schtasks /Change /TN "<old>" /DISABLE`, then `schtasks /Change /TN "<name> [wt]" /ENABLE`. A `switch_runner.ps1 <runner> onedrive|worktree` does both, plus the guard marker below.
- **Rollback is the same single command in reverse:** re-enable the old task. The shared state dir means no state moves.
- Never switch inside a no-trade window, or between a placer's run and the moment its orders resolve. For example, not between 09:05/09:10 and the open, or between 17:00 div evening and 09:15 the next day.

Order, one tier per night at most, each tier needing a clean session before the next:
- **Tier 0: read-only / non-placing.**
  - Expected Exit Monitor `-ExecEnv`, Radar Weekly Sync (preview; the flag stays absent).
  - PA Nightly Report, IV History.
  - The Local v9 `LOCAL_AUTOMATION_*` overrides, `daily_pitch.py:194`, `publish_sleeve_runtime_status --state-dir`.
  - Legend-ETF `LEGEND_ETF_EXECUTOR_ROOT`.
  - The disabled OLV Book Cap task: repoint it and keep it disabled.
  - Proof: the next run's normal output, plus the 5.1 sweep.
- **Tier 1: ExecAgent** (with its `book_snapshot`/`execute_order` subprocesses), via 5.4.
- **Tier 2: scheduled placers, lowest blast radius first.**
  1. `div_adjust` (both passes; switch after the evening pass on a day with no pending snapshot to carry, or carry it explicitly).
  2. Event Sleeve.
  3. Legend EMA + Verify (keep the long execution limit).
  4. OLV Pre-Market Exits (owner ids 99/98).
  5. IBKR Daily Order Chain (99/98, the most orders) last.

  The first live run of each is watched in person, and its journal/log is diffed with `shadow_diff.py` against the last OneDrive-run day.

**Startup guard.** `runtime_paths.assert_active_root(runner)` runs first in every runner:
- It reads `C:\trading_state\trading_ibkr\active_roots.json` (`{"order_chain": "C:\\trading\\runtime\\trading_ibkr-g1", ...}`), and exits 3 with a loud log line and a non-zero task result if its own code dir is not the active root for that runner.
- It also refuses when `TRADING_IBKR_STATE_DIR` is unset and the code dir is not OneDrive. This blocks the fallback into a worktree's stale ignored state (§1 "Working tree").
- `switch_runner.ps1` flips the marker in the same call as the task enable, so a wrong-root copy that is accidentally enabled no-ops instead of placing.

IBKR's duplicate-clientId refusal (error 326) is only a **backstop**. It catches concurrent runs but not sequential double runs, and it fails silently. `grep "326" *_last_run.log` belongs in each post-switch check.

### 5.6 Per-runner verdicts

Counts: **8 SHADOW-READY, 10 NEEDS WORK, 3 NOT SHADOWABLE.** Local v9 tasks and Daily Pitch only read `credentials.json`/`exec_agent.env`; they are config-only and proven by their next run.

| Runner | Dry-run mode | Connects (prod ids) | Writes in dry-run | Verdict | Minimal change / proof used |
|---|---|---|---|---|---|
| `exec_agent` | `AGENT_LIVE_ENABLED!=1` (`:65,1665`); signed `dry_run` (`:1906,1924`); `echo` (`:338`) | no socket; spawns book_snapshot | seen.jsonl (`:120-138,1956`), options_journal (`:1957`), **WS book/result/hello** (`:2095-2118,1959,1780`), position_action_agent results (`:88`) | NOT SHADOWABLE | shared WS, local-only dedup. Proof: 5.4 echo, preview, 1-share. (Full shadow would need a `SHADOW=1` that suppresses every `ws.send` plus a staging broker DO.) |
| `execute_order` | none; exits "rejected" when unarmed (`:3287`) | 123/147 (`:171`), owner reconnect 99/98 (`:2661`) | n/a | NOT SHADOWABLE | 5.4 step 3 |
| `book_snapshot` | inherently read-only (`:101-102`) | 122/146 hardcoded (`:24-25`) | none (stdout) | NEEDS WORK | clientId env override (222/246); diff vs the site book |
| `order_staging` | none (`:1664-1668`) | 99 (`:295`), 130 (`:203`) | staged_orders.csv, **Sheets `execution` (`:1625`) + `execution_2` (`:1647`)** | NEEDS WORK | `--no-sheets` + `--client-id`; diff staged_orders.csv with timing tolerance |
| `eq_order_entry` | `--dry-run [csv]` (`:879-881`) | none in dry-run | none | SHADOW-READY | feed prod's csv, diff vs eq_placed_orders.json; add `signal_ref` to the preview print |
| `pa_order_entry` | none | 98 (`:25`), places | pa_placed_orders.json, broker | NEEDS WORK | `--dry-run` + a csv input instead of the `execution_2` tab (`:18-19,155-158`) |
| `morning_order_summary` | none (`:840`) | 121/145 read-only | **2 Gmail (`:855-857`), R2 (`:804-806`)**, morning_orders.json | NEEDS WORK | `--no-email --no-r2`, id override |
| `div_adjust` | omit `--confirm` (`:905`); `--asof` replay (`:907`) never transmits/emails | probes 151/152; apply as 99/98 (`:679`) | no-confirm: **email (`:992-997`)**, audit/pending/snapshot/heartbeat; `--asof`: REPLAY audit + heartbeat (local) | NEEDS WORK | probe-id override; `--asof` is the shadow; `LIVE_ENABLED=True` (`:80`) is live |
| `olv_exit_moo` (primary + pa_legacy) | none | 99/98 owners (`primary:43-44`, `legacy:122-123`) | journals, **Gmail on any unconfirmed leg** | NOT SHADOWABLE | the plan is computed inside `process_account`, which places. Proof: `load_exit_rows` read-only diff + first-live-run journal diff |
| `olv_book_cap` | `--dry-run` (`:1793`) | 148/149 read-only | bat log only | SHADOW-READY | A/B dry-run; avoid pitch's 148 window |
| `event_moo` | `--check` (`:376`) | no | `event_moo_last_result.json` (copy-local) | SHADOW-READY | add orderRef to the check print |
| `pitch_moo` | `--check` (`:560`) | no (148 live) | last_result (copy-local) | SHADOW-READY | A/B `--check` |
| `trend_moo` / trend rows path | `--check` (`:332`); staging `TREND_TRUE_MOO_FLAG` (`order_staging.py:184`) | no (146 live) | last_result (copy-local) | SHADOW-READY | A/B `--check`; staging part covered by order_staging |
| `legend_ema` | `--dry-run` / flag absent (`:1490-1494`) | 161, `LEGEND_EMA_CLIENT_ID` override (`:177-195`) | last_result (unguarded, copy-local); journal guarded (`:1498-1505`) | SHADOW-READY | id 261, pre-run journal copy; revisions/exit/verify: first live run at `MAX_SHARES=1` |
| `pa_nightly_report` | none | 144 (`:34-35`) | **Gmail** | NEEDS WORK | `--out FILE`, id override |
| `radar_trail_sync` | preview unless `--apply` (`:243,302`) | no (HTTPS relay GET /book) | none | SHADOW-READY | run the `.py`, not the bat (it hardcodes the flag path and does an R2 publish) |
| `iv_history_update` | none | 134 | parquet + **R2 upload (`:149`)** | NEEDS WORK | `--no-upload`, id override, drop the `cd` in `run_iv_history.bat:3` |
| `option_snapshot_recorder` (unregistered) | none | 135/149 | parquet + **R2 x4** | NEEDS WORK | same as IV |
| `intraday_store` (unregistered) | none | 132 hardcoded | local parquet + heartbeat | NEEDS WORK | id override only |
| Expected Exit Monitor | omit `--upload`/`--send` | no | own `--state`/`--artifacts` | SHADOW-READY | run the `.py` directly (the task's `.ps1` always passes `--upload`, `:48`) |
| Legend-ETF Signals/Session/Watchdog | dry by default (`LEGEND_ETF_LIVE_ENABLED=0`) | 154-157 read-only | `%LOCALAPPDATA%\NewSeasonals\legend_etf` only; **global mutex** | NEEDS WORK (minor) | no shadow needed (imports no trading_ibkr code; `EXECUTOR_ROOT` is checked only on the live path). Repoint `EXECUTOR_ROOT`, regenerate the guard manifest; never run a copy concurrently (shared mutex) |

### 5.7 Go / no-go checklist

Every item must be GO on the evening of a switch:
- [ ] The 5.1 zero-change day is complete: all `[runtime_paths]` lines show `source=env`, and no unexplained journal diff.
- [ ] `test_runtime_paths.py` is green. The OneDrive suite and the New_Seasonals suite are green with the variable set and unset.
- [ ] The 5.2 parity gate passed within 24 h, and no OneDrive code mtime is newer than the gate.
- [ ] Every runner in this tier has met its 5.3 pass criterion, or has its recorded alternative proof (5.4, first-live-run plan).
- [ ] No `data/position_actions` record is in `attention`/`pending`.
- [ ] Nothing is in flight: no auction order between placement and the open, no `div_adjust` evening snapshot awaiting 09:15 (unless explicitly carried), no Legend EMA position open, no ExecAgent command queued.
- [ ] The arming snapshot is taken: `*_enabled.flag` set, `AGENT_LIVE_ENABLED`/`LIVE_*` keys, `div_adjust.LIVE_ENABLED`. It is diffed again after the switch, and an absent flag stays absent.
- [ ] The clientId check (`Get-NetTCPConnection -RemotePort 7496,4001`) shows only the Gateways and the expected prod runners.
- [ ] The old task is registered and disabled, and `switch_runner.ps1` rollback has been rehearsed on a Tier 0 task.
- [ ] The Defender/AV exclusion for `C:\trading_state` and `C:\trading\runtime` is confirmed.
- [ ] The user is available for the next morning's 09:05-10:45 watch.

**No-trade windows** (for switches and state copies; read-only shadows on +100 ids are allowed inside them):
- Weekdays 04:00-10:45 ET and 15:30-17:30 ET (§3).
- Any time ExecAgent is armed and a command could be in flight.
- From 09:05/09:10 placement until the open resolves.
- From 17:00 div evening until the 09:15 pass has consumed its snapshot.
- Mon 08:45-09:00 (radar).
- Prefer Saturday, or weekdays after 21:05 ET. Each tier then gets its first live observation the following session.

## 6. Cutover config for step 5 (New_Seasonals readers)

Code: New_Seasonals branch `ibkr-state-dir-readers`, commit `d078b129` (not pushed). Every reader follows the `runtime_paths` rule: env var when set, else today's OneDrive path. With nothing set, all paths resolve as today. `tests/test_trading_ibkr_state_dir_readers.py` covers both cases.

Which runners pick the code up:
- Daily Pitch, Radar Weekly Sync and Sleeve Status run from the `dev\New_Seasonals` checkout, so they use the code as soon as it is on the checked-out branch.
- Local v9 and the Expected Exit Monitor run from pinned worktrees. They use config only: the `.env` keys and the launcher argument below. No runtime release is needed.

Set at cutover step 6, together with the Phase 1 vars:
1. **User env** (`setx`, plus `trading_env.cmd`):
   - `TRADING_IBKR_STATE_DIR=C:\trading_state\trading_ibkr`
   - `TRADING_IBKR_SECRETS_DIR=C:\trading_state\secrets`

   These drive:
   - `daily_pitch.credentials_path` (reads `credentials.json` from SECRETS_DIR, with no OneDrive fallback once set).
   - `publish_sleeve_runtime_status` (the `event_moo`/`trend_moo`/`legend_ema` `*_enabled.flag` files, `legend_ema_last_result.json` and `legend_ema_journal.jsonl` come from STATE_DIR; `--executor-root` stays the code dir). The Sleeve Status task args are unchanged.
   - `run_radar_sync.bat` (reads `radar_trail_enabled.flag` from STATE_DIR; the `radar_trail_sync.py` code path stays OneDrive until Phase 3).
2. **`dev\New_Seasonals\.env`** is the ConfigRoot of every Local v9 task and of the Expected Exit Monitor. The keys are already supported. Use forward slashes and no quotes:
   ```
   LOCAL_AUTOMATION_GCP_JSON_PATH=C:/trading_state/secrets/credentials.json
   LOCAL_AUTOMATION_EXEC_ENV_PATH=C:/trading_state/secrets/exec_agent.env
   ```
3. **Expected Exit Monitor.** The task runs `wscript` on `Documents\Codex\scheduled-task-launchers\c85d7b343750efae.vbs`, which passes `-ExecEnv "C:/Users/McKinley Slade/OneDrive/trading_ibkr/exec_agent.env"` to the pinned runtime `expected-exit-runtime-20260914` (SHA `bc6dc10`). The pinned `.ps1` still has `-ExecEnv` Mandatory, so change the value in the `.vbs` to `C:/trading_state/secrets/exec_agent.env`. The file is UTF-16LE, so keep that encoding. Do not drop the argument. No task re-registration is needed. The new `-ExecEnv` / `--exec-env` defaults only take effect after a runtime re-pin or a new task XML.
4. **Flags are state.** Every `*_enabled.flag` moves with the robocopy into `C:\trading_state\trading_ibkr\`. `radar_trail_enabled.flag` stays absent (Tier 0 preview).

Proof on the next run:
- `scripts\logs\radar_sync_last_run.log` shows `flag dir C:\trading_state\trading_ibkr`.
- The Sleeve Status payload shows the same flag booleans as before the switch. It refuses to publish if STATE_DIR is set but missing.
- The pitch writes the Sheets Pitch tab.
- The Local v9 receipts are green.
- The Expected Exit Monitor keeps publishing.

Scheduled tasks inherit `setx` values only for processes started after the change. Confirm the first run's log line; if a task still reads OneDrive, log off and back on, or restart the machine.

Rollback: unset both vars, remove the two `.env` keys, and restore the `.vbs` argument.

Left for Phase 3, or not live:
- Code paths: `LEGEND_ETF_EXECUTOR_ROOT` in `%LOCALAPPDATA%\NewSeasonals\legend_etf\runtime.env`, the Sleeve Status `-ExecutorRoot` arg, `run_radar_sync.bat` `%IBKR%`, and the test `IBKR_DIR` constants.
- `scripts/run_exec_agent.ps1` and `register_exec_agent.ps1` are unused repo mirrors. Re-sync them from the trading_ibkr step-4 versions rather than hand-editing them.
- `scratch/*` one-offs read `exec_agent.env`, `credentials.json` and `staged_orders.csv` from OneDrive. They are not production; after the rename they fail rather than fork.
