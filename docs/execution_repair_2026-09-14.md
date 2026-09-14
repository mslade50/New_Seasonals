# Execution repair candidate — September 14, 2026

Status: the initial repair shipped in PRs 45 and 46. The September 14 follow-up audit below identified additional defects; its corrected runtime candidate is tested but installation is awaiting explicit approval after automatic approval review rejected the restart.

The user approved runtime installation/restart and cloud deployment, and requested direct Save for order edits. The editor now sends only the exact identity and changed quantity/prices, without purpose/direction/risk fields or a confirmation popup. The executor derives metadata from durable order provenance, broker-linked parents and current positions, and computes required risk from attached stops and uncovered quantity. Existing broker ownership/capacity checks remain. Five additional automatic-edit tests and the direct-Save frontend regression pass. The coordinated runtime candidate now contains ten modules, including `order_edit_context.py`.

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

Install all ten candidate modules together while the Execution agent is stopped, preserving timestamped copies of the replaced modules. Preserve all order, schedule and position-action journals. Review pending work before restarting the armed agent. The exact target is the existing OneDrive `trading_ibkr` runtime for Primary and PA; no new service or paid infrastructure is required. Restarting can resume pending trading work, so the repository's financial-action approval rule requires explicit approval immediately before this operational step. Broker order changes remain an operator action.

Rollback: stop the agent, restore the saved module versions, and restore the previous frontend commit through the normal cloud pipeline. Keep all current journals and reconcile uncertain orders; a code rollback cannot undo fills.

Merge the reviewed site changes to `origin/main`, then dispatch `.github/workflows/deploy_site.yml` on main. Require its R2 pulls, build, freshness gate and Pages deploy to pass. Verify the Cloudflare production commit and the authenticated live Execution tab. Do not publish local `data/` or `dist/`.

## Outstanding operational issues

The earlier SNA close attempt left an attention journal after partially adjusting exits. After deployment, this receipt incorrectly continued to block the user's manual quantity repairs. On September 14 it was explicitly reconciled and retired as rejected before close submission. The original executor persists a wire identity before submitting a close; this receipt had no wire and had stopped at resizing exits. Fresh broker evidence showed 700 shares and all eight exact exits acknowledged with no exit fills, at their original or intended reduced quantities. The original receipt was backed up with its SHA-256 before writing the audited resolution under the executor's operation lock. No broker order was submitted or changed by reconciliation. The user remains responsible for finishing the exit quantities.

`broker_runtime/position_action_recovery.py` provides this explicit operational recovery without a broker mutation API or automatic runtime hook. It requires fresh account evidence, exact unchanged exit identities and terms, no fills, original/intended remaining quantities, and a reviewed receipt hash. It refuses post-submission uncertainty, active entry parents, additions, re-adds, cancellations, or changed evidence. Regression tests reproduce the manual edit rejection before recovery and allow the same edit afterward through a simulated broker, while leaving the original close permanently terminal and rejected.

The expected-exit status panel was unavailable during live inspection. A direct read of its R2 key returned `NoSuchKey`; prior rollout notes say monitor scheduling was not registered. Its frontend fixture test passes, but the live producer has not been repaired by this change. Stock Add/Re-add scope, option account restrictions and existing risk acknowledgements remain in force.

## Follow-up runtime audit after the 17:07 SNA Modify

IBKR accepted the user's limit-order quantity change from 218 to 171, with zero fills. The reporting wrapper then failed: `execution_lifecycle.mutate_one` called the executable's `_out`, which prints JSON and returns exit code 0. `order_mutations` treated that integer as a result object, printed a second error result, and left the journal mutating. The agent correctly refused to trust the resulting two JSON documents. Earlier tests replaced `_out` with a dictionary-returning stub and therefore missed the integration defect.

The fix collects the inner result silently, saves it, then emits exactly one executable result. Completed-command replay emits the stored result without resubmitting. The agent reports completed edit receipts after reconnect, without resuming uncertain edits. The affected SNA receipt was separately backed up and reconciled using fresh exact broker identity, unchanged order terms, acknowledged quantity 171, and zero fills. No order was retried. The reporting correction reaches the site after the updated agent helper is installed and restarted.

Independent review also found and fixed:

- Add used incomplete held-contract routing metadata. Add and Close now qualify the exact held instrument before any exit mutation and retain the qualified copy.
- Add/re-add could accept a pending attached exit as success. Every expected child now needs broker acknowledgement; bounded waiting never resends, and incomplete acknowledgement remains unknown.
- Capped option limits used global `minTick`, which can differ from the routed exchange's premium-band increment. The corrected selector requests that exchange's market rule, rounds across its price bands before sizing, and preserves the premium cap.
- A live market-data flag alone did not establish quote freshness. Snapshot timestamp, exact contract identity, and a 30-second age limit are checked before sizing and again before submission. Missing rules or stale quotes reject before submission.

Verification: four actual CLI regressions failed on the old Modify/Cancel wrapper; six Add/acknowledgement regressions and six option regressions also reproduced defects before their fixes. The combined corrected suite passed 149 tests. All ten candidate modules compiled, and 17 CLI/routing tests passed against that exact candidate. A committed, source-checked CLI fixture makes the output-contract tests run in CI even without the external broker directory; Windows CI explicitly includes them. Broker interactions remained simulated, so this does not establish actual broker acceptance for every order type.

The follow-up install has not run. Automatic approval review rejected the armed runtime restart because it can resume financial commands and requires fresh approval for the expanded rollout. The approved next action must target the existing Primary/PA `ExecAgent` runtime, preserve timestamped backups, verify source/candidate hashes and absence of an active executor child, then restart. Rollback restores the saved code; it cannot undo broker fills.
