# Execution-tab reliability fix — brief (2026-09-21)

Owner request: "check all the execution types on the private site; there have
been a number of failures of late for orders placed there." Audit evidence
(gitignored, on the trading box): `artifacts/recon_2026-09-17/site_execution_audit.md`
(type matrix, live-code checks, root causes) and `site_execution_failures.md`
(59 failure rows 2026-08-25..09-21 from the agent log + broker command ring).

## What the 59 failures were

| Cause | n | Verdict |
|---|--:|---|
| RISK_ACK_REQUIRED (no-stop entry > 50 bps NLV) | 17 | Gate working; every one confirmed + filled. Wording only. |
| Position-action lock refusals (edit/cancel/close/flatten) | 15 | Lock correct; REPORTING wrong (`state=unknown` + "VERIFY IN TWS, DO NOT RETRY" for a clean no-op; reason never names the blocking action); no clear-lock control on the site. |
| "owning client cannot resolve exactly one matching open order" (modify/cancel) | 6 | Matching rule cannot identify the order uniquely. Needs analysis. |
| Trim button quantity mismatch | 3 | Sends a command every layer rejects. Retire. |
| flatten/modify "disabled until lifecycle rewrite" | 4 | Pre-2026-09-14; gone. |
| PA $30k notional cap on futures | 2 | Fixed 2026-09-21 (installed). |
| Futures contract mapping (exchange/expiry) | 2 | Verify the 2026-09-14 `futures_front.py` install covers it. |
| IBKR notices treated as failures (399 repriced, 2109 RTH) | 3 | Misclassification. |
| Real IBKR reject (short not available) | 1 | Nothing to fix. |
| Other single gates (live-proof check, out-of-hours) | 6 | Working as designed. |

Type matrix: WORKING with live evidence = LMT/MKT stock entries, close,
partial close, modify, attach-exits. DEFECTIVE = flatten (false "unknown"),
Trim. NEVER EXERCISED = MOO/MOC/STP_LMT entries, scale-outs, all futures, all
FX, all options (incl. the PA futures path just installed).

Also found: the OneDrive safety-test suite (18 `test_*.py`) has not been able
to start since early September — one broken import aborts collection — so no
execution guard has been verified since then. The repo's spent one-shot
preparer fragments (patches already live) fail their anchor tests locally and
skip on CI; retire them.

## Fix package (one live-agent candidate + one site change), in order

Agent side — build with a `broker_runtime/prepare_*.py` preparer in the
pattern of `prepare_entry_controls.py` (fragments + replace_once → candidate
dir with `.original` copies + `manifest.json` sha256s), never writing to
OneDrive; the owner installs from a runbook in the style of
`artifacts/recon_2026-09-17/pa_futures_install_runbook.md`:
1. Fix the test-suite import so the 18 OneDrive test files collect and run;
   report pass/fail/skip against a scratch overlay of the live dir.
2. Lock refusals RETURN `state="rejected"` with a reason naming symbol,
   action type, action id, created time and the discrepancy text. Catch the
   `ValueError` (position_actions.py ~441 and siblings on flatten/close/
   cancel/modify) at the command boundary; keep the `unknown` catch-all for
   post-transmit exceptions only. Test: pre-transmit → rejected; post-transmit
   → unknown.
3. IBKR codes 399, 2109 and the 2100-2199 warning range → notice, not failure.
   Test.
4. Analyse the 6 "cannot resolve exactly one matching open order" cases from
   `exec_agent_last_run.log` (orderRef ambiguity? scale-out pairs? missing
   permId?). Tighten only if safe (permId first, orderRef+qty fallback);
   otherwise write the design question for the owner.
5. New command `position_action_resolve` {account, symbol, action_id,
   operator_note}: re-checks the live book for the symbol, records positions +
   open orders into the state file, marks phase done / resolved_by operator.
   Refuses if the action is not unresolved. Fake-broker test.
6. Confirm the futures contract-mapping cases are covered by the 2026-09-14
   `futures_front.py` install; fix only if trivial.

Site side (`site/assets/execution.js` + schema doc + JS contract tests):
7. Retire the Trim control.
8. Add a clear-lock control that sends `position_action_resolve` with a
   required operator note, and render the new structured rejection reason
   (symbol / action / age) in the Activity panel. Reword the RISK_ACK prompt
   so it reads as a confirmation step, not an error.
9. Deploy via the cloud-only `deploy_site.yml` (never a local site build).

Repo hygiene:
10. Retire spent preparer fragments + their tests (list in
    `artifacts/recon_2026-09-17/reviewed_runtime_drift_review.md`).

Gates: repo `pytest -k "execution or broker or position or manual"` before/
after; OneDrive suite on the overlay; candidate diff saved and every hunk
mapped to an item above; runbook with backup + rollback. Two PRs (agent, site),
merged together; install the agent candidate outside 09:30-16:00 ET.

## Permission note
On 2026-09-21 the session's auto-mode permission layer refused to launch the
build ("Modify Shared Resources") because it builds against a copy of the
live trading folder. Run this brief in a session where that is permitted, or
grant the rule; the owner installs the candidate by hand either way.
