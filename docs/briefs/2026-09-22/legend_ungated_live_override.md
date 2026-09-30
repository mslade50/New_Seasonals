# Legend: owner ungated-live override (self-expiring, evidence gates only)

Owner decision 2026-09-22 (McKinley): run the Legend EMA ETF sleeve LIVE today
on Primary, long only, bypassing the four EVIDENCE artifacts the runner requires
because none of them can be produced today. Every BROKER-STATE safety check
stays intact. Concerns were raised and are on the record in
`docs/legend_readiness_2026-09-21.md` (branch `codex/legend-readiness-20260921`);
this brief is the owner's reaffirmed call.

## Setup

- Source: worktree `C:/Users/McKinley Slade/dev/new-seasonals-legend-runtime`
  at b46a56d9 (`codex/new-seasonals-legend-runtime`). Do NOT edit it; it is the
  scheduled production checkout.
- Work in a new worktree from the main repo:
  `git worktree add "C:/Users/McKinley Slade/dev/New_Seasonals-worktrees/legend-ungated-live-20260922" -b legend-ungated-live-20260922 b46a56d9`
- Python: `C:/Users/McKinley Slade/AppData/Local/Programs/Python/Python310/python.exe`
  (pinned deps installed). No pip installs.
- Hard deadline: code reviewed and the runtime checkout switched before the
  09:28 ET session task. Report by 08:10 ET.

## The switch

`LEGEND_ETF_OWNER_UNGATED_LIVE_DATE` in the machine-global `runtime.env`
(`%LOCALAPPDATA%\NewSeasonals\legend_etf\runtime.env`; loaded via
`load_dotenv(RUNTIME_ENV, override=True)` in `run_legend_etf_session.py:25` and
re-read with `dotenv_values` in `session.py` `_read_live_runtime`).

ACTIVE only when the value string-equals today's New York date, using the same
date source the live gate uses for `LEGEND_ETF_LIVE_DATE`
(`legend_etf/ibkr_adapter.py:114-152`). Blank, missing, mismatched, or
malformed = inactive = current behavior byte-for-byte. One helper,
`legend_etf/owner_override.py::ungated_live_active(env, today_ny) -> bool`,
called from every site.

## Skip when ACTIVE (and only then)

Line refs are from a read-only audit of b46a56d9; verify each.

1. `session.py:540-585` runtime.env completeness: `LEGEND_ETF_GUARD_MANIFEST`,
   `LEGEND_ETF_GUARD_MANIFEST_SHA256`, `LEGEND_ETF_PAPER_PROOF`,
   `LEGEND_ETF_PAPER_PROOF_SHA256`, `LEGEND_ETF_PORTFOLIO_BUDGET` may be blank.
   Every other key stays required.
2. `session.py:587-599` guard-manifest validation (`reservations.py:682-943`):
   skip entirely (integration receipt, executor inventory hash, marker file,
   candidate-parity binding, python_runtime pin).
3. `session.py:601-609` paper-proof validation (`legend_etf/paper_proof.py`): skip.
4. `session.py:658-677` portfolio-budget validation (`portfolio_guard.py:102-211`)
   and the capacity debit at `session.py:3169-3255` / `portfolio_guard.py:214-301`:
   skip; capacity unlimited for the session. Confirm `sizing.py:177-185` does
   not zero `nav_40_30`; Primary long must size to floor(NLV*0.40/price) SPY and
   floor(NLV*0.30/price) QQQ, capped by `LEGEND_ETF_PRIMARY_MAX_SHARES_PER_ROOT`.
5. `session.py:611-656` `_reconcile_attested_external_quarantines`: skip only
   the manifest-bound (attested) part. Keep any manifest-independent local
   quarantine check under the runtime dir.
6. `session.py:679-729` `_refresh_live_gate`: skip re-validating the three
   artifacts; KEEP the mid-session drift check on the live-gate keys and the
   five risk-sizing keys.
7. `session.py:1150-1218` startup correction audit: keep the live connection,
   `assert_server_clock`, `validate_correction_audit_gate`; skip only its
   guard-manifest step.
8. Watchdog / `--reconcile-only` path: honor the override wherever it would
   validate the same artifacts, so the 10:40+ sweeps do not fail the session
   after the fact.

## Do NOT touch

Clock calibration (`clock.py`, `ibkr_adapter.py:413-471`); late-start gate
(`session.py:4879-4899`); calendar (`session.py:1226`); signal-plan validation
(`session.py:1227-1246`); the live gate itself (`ibkr_adapter.py:114-152`,
`require_side`); reservation file locks (`session.py:1448-1476`);
`assert_symbol_clear`; `nlv()`; tape rechecks; `place_bracket` /
`final_transmit_check`; the 10:30 exit fail-safe; recovery; durable intent
writes; the PowerShell wrapper and registration scripts; anything under
`C:/Users/McKinley Slade/OneDrive/trading_ibkr`.

## Logging

Every skip prints one loud line, e.g.
`[OWNER-UNGATED-LIVE 2026-09-22] skipping guard-manifest validation`, and the
list of skips is written into the session's durable startup state/audit record
under `owner_ungated_live_skips` if such a record exists.

## Tests

`tests/test_legend_owner_override.py`:
(a) `ungated_live_active` truth table (match, mismatch, blank, missing, bad format);
(b) override INACTIVE: env missing GUARD_MANIFEST_SHA256 still fails completeness;
(c) override ACTIVE: same env passes, and manifest/paper-proof/budget validators
    are NOT called (mock, assert not called);
(d) override ACTIVE: `_refresh_live_gate` still refuses on a live-gate or
    risk-sizing key change;
(e) `nav_40_30` with unlimited capacity sizes floor(NLV*w/price) capped at
    MAX_SHARES_PER_ROOT.

Must pass with the Python310 exe:
`tests/test_legend_owner_override.py tests/test_legend_etf_native.py
tests/test_legend_etf_signal.py tests/test_legend_etf_execution.py
tests/test_legend_etf_recovery.py tests/test_legend_etf_parity.py
tests/test_legend_etf_hardening.py tests/test_legend_nav_execution_sizing.py
tests/test_legend_clock.py tests/test_legend_shadow_receipt.py
tests/test_legend_status.py`; existing tests unchanged and green.
`python -m ruff check legend_etf scripts/run_legend_etf_session.py` clean.

## Deliverable

Commit on `legend-ungated-live-20260922`, message starting
`Legend: owner ungated-live override (self-expiring, evidence gates only)`.
Do not push. Report: commit SHA, exact skip list with file:line, pytest summary
line, anything unsure, and any check you think is mis-classified.

## After the code lands (operator steps, not the agent's)

1. Review the diff. Switch the runtime checkout:
   `cd C:/Users/McKinley Slade/dev/new-seasonals-legend-runtime && git checkout legend-ungated-live-20260922`
2. Edit `runtime.env`: `LEGEND_ETF_LIVE_ENABLED="1"`,
   `LEGEND_ETF_LIVE_DATE="2026-09-22"`, `LEGEND_ETF_LIVE_ACCOUNTS="U16584234"`,
   `LEGEND_ETF_ALLOW_LONGS="1"`, `LEGEND_ETF_ALLOW_SHORTS="0"`,
   `LEGEND_ETF_OWNER_UNGATED_LIVE_DATE="2026-09-22"`. Back up first.
3. Re-register from the runtime checkout with `-Install -Replace -Live
   -LiveAcknowledgement REGISTER_DATED_GATED_LIVE_TASK -Accounts primary`
   (`scripts/register_legend_etf_tasks.ps1`). The current Session task runs
   `-ShadowThroughExit` with no `-Live` and will dry-run otherwise.
4. Re-run `scripts/check_legend_etf_readiness.py --entry-date 2026-09-22` and
   confirm `live_ready` and the three missing files are no longer reported.
5. Watch the 09:28 session log under `%LOCALAPPDATA%\NewSeasonals\legend_etf\logs`.
6. Kill switch: `LEGEND_ETF_LIVE_ENABLED=0`. Never delete runtime state or
   flatten by hand while the runner holds a position; use its recovery path.
