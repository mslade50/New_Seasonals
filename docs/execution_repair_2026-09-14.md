# Execution repair candidate — September 14, 2026

Status: implemented and tested in an isolated worktree; not installed in the broker runtime or deployed to the private site.

The live executor had an Add handler but omitted `add_to_position` from its dispatcher allowlist. Cancel and Modify remained disabled in both agent and executor. PA used the legacy position controls and handlers. Exit resizing could report failure while a fresh broker acknowledgement was available because it checked the stale Trade returned by submission.

This candidate aligns PA and Primary stock Close/Add/Re-add controls and routes both through the same position lifecycle. Cancel and Modify use exact account/contract/client/order/permanent identities, durable receipts and complete raw broker order fields. A bounded acknowledgement wait never resends. Missing fresh fill counters, fills during resizing and uncertain delivery require reconciliation. Unresolved edits and position actions block each other for the affected contract. Futures quantity increases include the contract multiplier in the notional check; closing orders cannot increase beyond available holdings.

Held futures are qualified by their existing contract ID before metadata validation. Front selection distinguishes the actual delivery month from the last trading date and never falls back to expired contracts. This addresses the MCL month mismatch.

Scheduled option buys use the user-selected capped limit policy. At the scheduled time, the existing expiry/delta selector resolves the option; a fresh live ask and contract tick set a rounded-up limit. Whole-contract sizing keeps `quantity × limit × multiplier` within the premium budget, excluding commissions. The order is BUY LMT DAY and may remain unfilled. Existing account, risk, quote, topology and time-window checks remain. Saved MKT instructions are rejected rather than silently converted. An uncertain submission is never retried automatically.

## Verification

- 108 tests passed across the new dispatcher/capped-option regressions, unified lifecycle, fast-action frontend, stop-limit and attach-exit suites. Actual reviewed agent/executor functions were extracted by AST and exercised with fake brokers; live modules were not imported or connected.
- Seven additional portable broker lifecycle tests passed, covering owner preflight, exact cancellation, ambiguous cancellation, modification, attached additions, auction claims and helper deadlines.
- All eight `tests/js/test_execution*` files and `test_expected_exits.mjs` passed.
- Local browser QA with synthetic positions and broker submissions blocked verified PA/Primary control parity, PA full-close routing, and the scheduled option premium-cap explanation. This is local UI verification, not production verification.
- Candidate preparation verifies installed source hashes and compiles every generated module. The new tests needing the external reviewed runtime skip when it is unavailable.

The older `test_broker_runtime_fixes.py` full suite has seven source-dependent failures/errors because its historical patcher expects a previous installed source version. Its seven relevant portable tests pass separately. The new repair tests use the currently installed source. No live broker submission or paper-broker acknowledgement was exercised.

## Controlled rollout

Prepare a new candidate directory with:

```powershell
python -m broker_runtime.prepare_execution_repairs --source 'C:/Users/McKinley Slade/OneDrive/trading_ibkr' --output artifacts/broker-candidate-release-v1
```

The package does not install or restart anything. Generated candidates retain private runtime configuration and stay in ignored artifacts. Verify the complete source and candidate manifest before installation; if any source changes, review and rebuild the candidate.

Install all nine candidate modules together while the Execution agent is stopped, preserving timestamped copies of the replaced modules. Preserve all order, schedule and position-action journals. Review pending work before restarting the armed agent. The exact target is the existing OneDrive `trading_ibkr` runtime for Primary and PA; no new service or paid infrastructure is required. Restarting can resume pending trading work, so the repository's financial-action approval rule requires explicit approval immediately before this operational step. Broker order changes remain an operator action.

Rollback: stop the agent, restore the saved module versions, and restore the previous frontend commit through the normal cloud pipeline. Keep all current journals and reconcile uncertain orders; a code rollback cannot undo fills.

Merge the reviewed site changes to `origin/main`, then dispatch `.github/workflows/deploy_site.yml` on main. Require its R2 pulls, build, freshness gate and Pages deploy to pass. Verify the Cloudflare production commit and the authenticated live Execution tab. Do not publish local `data/` or `dist/`.

## Outstanding operational issues

The earlier SNA close attempt left an attention journal after partially adjusting exits. The user is managing the live quantities. This candidate neither clears that receipt nor retries the close; broker/order reconciliation is still needed before subsequent SNA position actions.

The expected-exit status panel was unavailable during live inspection. Its frontend fixture test passes, but the live producer/R2 status has not been repaired or verified by this change. Stock Add/Re-add scope, option account restrictions and existing risk acknowledgements remain in force.
