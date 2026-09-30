# Legend: one-share Primary live plan (owner decision 2026-09-22)

Owner decision (McKinley, 2026-09-22): the Legend EMA ETF sleeve goes live on
the PRIMARY account at a ONE-SHARE cap for its first sessions. Those sessions
are the broker drills; no DU paper account will be used. The 10:30 time exit
is confirmed. Full 40/30 NAV sizing follows once the proof file is produced
from the one-share sessions. This brief records what has to be built and the
facts found by read-only scoping on 2026-09-22. It supersedes
`legend_ungated_live_override.md` (same folder), which proposed skipping the
evidence gates outright and was refused by the Claude Code auto-mode classifier.

Checkouts: production runtime `C:/Users/McKinley Slade/dev/new-seasonals-legend-runtime`
(b46a56d9). External executor `C:/Users/McKinley Slade/OneDrive/trading_ibkr`.
Python `C:/Users/McKinley Slade/AppData/Local/Programs/Python/Python310/python.exe`.

## State on 2026-09-22

- First complete shadow session today: signals 08:45, session 09:28,
  decisions 09:31 (SPY/QQQ both SHORT, opened above EMA), revisions 09:46 /
  10:01 / 10:16 on the second, exit observed 10:30:15, watchdog success 10:40.
  Shadow count: 1 of the 5 the runbook asks for.
- Read-only probe (`scripts/check_legend_etf_readiness.py --entry-date
  2026-09-22`): account, history, ATR, dividend, position checks pass; missing
  live files = guard manifest, portfolio budget, paper proof.
- runtime.env: `LEGEND_ETF_PRIMARY_SIZING_MODE=nav_40_30`, LIVE_ENABLED 0,
  LIVE_DATE/ACCOUNTS blank, ALLOW_LONGS 0, SHA fields blank.

## Workstream 1: executor guard manifest (the hard one)

Scoping findings (reservations.py in the runtime checkout, executor root E:):

- `discover_broker_mutation_files` is a TEXT regex
  `\.\s*(?:placeOrder|cancelOrder|reqGlobalCancel)\s*\(` over every `.py`
  under the executor root (rglob; skips only `_backup*` path parts,
  `__pycache__`, `test_*.py`). `.runtime_backups/` and `artifacts/` ARE
  scanned. Comments and docstrings count. The builder requires the match set
  to be exactly `{legend_reservation_guard.py}`.
- Current raw matches: `event_contract.py:34` (a deadline proxy the guard
  itself calls via `guarded_place_order(AuctionClient(...))` in
  `event_moo.py:335`; already downstream of the guard, fails only the text
  regex), `manual_order_actions.py:132,152` (cancel/modify; agent-side these
  are in `DISABLED_UNSAFE_MUTATIONS` in exec_agent.py:68-70 but the
  execute_order.py handlers remain operator-reachable),
  `reconcile_position_exits.py:169,178` (live-reachable via the exec agent's
  `reconcile_exits` command).
- Guard wrapper API: `guarded_place_order(client, contract, order, *,
  account, mutation_kind, portfolio_direction, risk_bps, risk_usd, signal_id,
  existing_oca_group_orders, before_broker_call, now)`,
  `guarded_cancel_order(client, order, manual_cancel_order_time="", *,
  contract, account, oca_group_orders)`, `guarded_global_cancel(client)`.
  Config loads at CALL time; before the marker exists the wrappers fall
  through to the raw call (except a blank-tif block on mapped clusters).
- ARMING IS IRREVERSIBLE AND BOOK-WIDE: the builder writes
  `<executor-root>/.legend_reservation_guard_required.json` FIRST. After that,
  any missing/invalid config, or ANY edit to ANY executor `.py` that changes
  the source-tree hash, makes every guarded mutation raise
  (`legend_reservation_guard.py:290-324`) until the manifest is rebuilt and
  processes restarted. eq_order_entry, execute_order, event_moo, olv_book_cap
  all import the guard. This is a standing operational tax on a directory
  that changes weekly, and a real-money blast radius if a rebuild is missed.
- Integration receipt: `REQUIRED_INTEGRATION_TESTS` has 86 names; 57 exist
  as `def test_<name>` under the executor root; 29 DO NOT EXIST anywhere
  (list in the 2026-09-22 scoping report; several look like renames of
  test_olv_exits.py / test_close_resize.py cases). The receipt must claim
  the exact 86 set plus `reviewed_attestation ==
  "I_REVIEWED_SHARED_EXECUTOR_INTEGRATION_TESTS"`. No script produces it.
  An honest receipt therefore requires writing or renaming 29 tests first.
- Candidate-parity evidence: a passing file already exists and re-validates:
  `new-seasonals-legend-runtime/artifacts/legend-readiness-20260921/release_candidate_parity.json`.
- Builder CLI: `--reservation-dir --runtime-dir --executor-root
  --reservation-config --integration-receipt --candidate-parity-evidence
  --manifest --reviewed-attestation I_REVIEWED_SHARED_EXECUTOR_RESERVATIONS`.
  Also needs `contract_reference.json` in the executor root and the 7 pinned
  distributions importable.

Owner decisions needed before this workstream starts:
(a) accept the marker's book-wide arming and the rebuild-on-every-edit tax,
or ask Codex to redesign the gate (e.g. hash only the guard + the files that
import it, or make the marker per-sleeve); (b) authorize writing/renaming the
29 missing tests rather than attesting to tests that do not exist.

## Workstream 2: portfolio budget producer

No producer exists in the repo. Validator: `legend_etf/portfolio_guard.py`
(protocol `legend-equity-index-risk-budget-v3`, `risk_basis=stress_atr_bps`,
`entry_date` = today, `generated_at` 08:30-09:25 ET, `expires_at` next
midnight ET, `source_manifest_sha256` = guard manifest SHA, per-account
`{remaining_long_bps, remaining_short_bps, remaining_gross_bps}`,
`reservations` list). The debit at `portfolio_guard.py:214-301` rewrites the
file under `equity_index_cluster_budget.lock`. Build
`scripts/build_legend_portfolio_budget.py` and add it as an 08:35 step (or
inside the 08:45 signals task) writing Primary capacity from
`LEGEND_ETF_PRIMARY_LONG_BPS/SHORT_BPS/CLUSTER_BPS`. Depends on Workstream 1
for the manifest SHA. Not yet scoped in detail (classifier refused the
scoping pass in auto mode).

## Workstream 3: one-share drills on Primary + proof writer

- Validator `legend_etf/paper_proof.py` hardcodes `paper_account` starting
  `DU` and endpoint port 4002/7497. Change: accept the Primary account and
  port 7496 when runtime.env carries an explicit owner key naming that
  account for drills; everything else in the 15-key schema, the 9 drills,
  the revision-latency and 10:30-exit bounds stays as is.
- Drill runner: a standalone script that places the one-share IOC entry
  parent + OCA type-2 target + 10:30 GAT market sibling on Primary through
  the existing `ibkr_adapter` functions, exercises partial fill, target
  modify, restart-during-revision, disconnect/reconnect, records orderRef
  echoes from openOrder and execDetails, and writes `paper_proof.json` bound
  to the manifest SHA. Long drill AND short drill are both required by the
  schema (one orderRef echo each). Not yet scoped in detail (classifier
  refused the scoping pass).
- Alternative accepted by the owner: run the real session with
  `LEGEND_ETF_PRIMARY_MAX_SHARES_PER_ROOT=1` and derive the proof from the
  session's own audit journal. Requires the live gate to be open first, so it
  only works after Workstreams 1-2 and a validator that can accept a
  session-derived proof; otherwise circular.

## Workstream 4: arm

1. runtime.env: LIVE_ENABLED=1, LIVE_DATE=<entry date>,
   LIVE_ACCOUNTS=U16584234, ALLOW_LONGS=1 (ALLOW_SHORTS per owner; today's
   signals were short and nav_40_30 sizes shorts at 0), MAX_SHARES_PER_ROOT=1
   for the drill sessions, manifest/proof paths + SHA256s, budget path.
2. Re-register from the runtime checkout:
   `scripts/register_legend_etf_tasks.ps1 -Install -Replace -Live
   -LiveAcknowledgement REGISTER_DATED_GATED_LIVE_TASK -Accounts primary`
   (adds `-Live` to the session task, expands the watchdog to
   10:40/12:00/14:00/15:45/16:10).
3. Re-run `check_legend_etf_readiness.py` for the entry date; require
   `live_ready` true and `missing_live_files` empty.
4. Watch `%LOCALAPPDATA%\NewSeasonals\legend_etf\audit_live.jsonl` and
   `session_state_live.json`. Kill switch `LEGEND_ETF_LIVE_ENABLED=0`.
5. After the proof file exists and validates: raise MAX_SHARES_PER_ROOT to
   5000 for full 40/30.

## Open owner questions

- Shorts: enable or not, and at what size (nav_40_30 is long only).
- Workstream 1 design: accept Codex's marker semantics or redesign.
- The 29 missing integration tests: write them or drop them from the set.
