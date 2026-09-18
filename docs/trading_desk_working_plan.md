# Trading desk improvement plan

Updated: 2026-09-10. Owner: McKinley. Work proceeds one priority at a time.

## Purpose and decisions

Make the trading desk easier to operate, its answers more trustworthy, and its research more useful for deploying capital. Code cleanup is worthwhile only when it advances one of those outcomes.

- Execution is the primary workstation; Portfolio and Seasonal are next in importance.
- Portfolio's main view is theoretical. Actual trading belongs in the separate comparison and execution workflows.
- Daily Pitch judges standalone idea quality, excluding both holdings and the existing strategy mix (owner clarification, 2026-09-08).
- The separate strategy-discovery pipeline still evaluates fit with the algorithmic strategies we trade, not today's holdings.
- Preserve manual trade controls regardless of snapshot age; broker validation remains separate.
- Preserve the existing daily pitch cadence. New strategy-discovery email is conditional on something worth discussing.
- Work on Primary within the authorized scope. The owner explicitly approved the shared OVS fallback correction for Primary and PA on September 9; unrelated account and research workflows remain outside scope.
- No daily scan or order submission during this work. OVS fallback sizing and model rounding changes are explicitly approved; other account and sizing changes require their own scope decision.
- Do not start the next priority until McKinley reviews the completed current step.

## Priority order

| Priority | Outcome | Status | Completion criterion |
| --- | --- | --- | --- |
| 1 | Review and correct the trustworthiness of sampled existing answers | Complete: review and source corrections verified, 2026-09-08 | Trace a representative set of existing outputs to inputs and rules; document supported claims and errors; correct verified reasoning/source-contract defects within this scope; verify corrections with targeted evidence and regression checks; disclose remaining limits. |
| 2 | Repeated Execution actions require less arithmetic, re-entry, or cross-checking; clean up the tab as a secondary goal | Complete: bounded navigation/layout improvement live, 2026-09-08 | Identify actual friction in entry, trim, re-add, brackets, and hedging; implement only an evidenced improvement; demonstrate correct resulting behavior and steps removed. |
| 3 | Specific economically relevant differences between modeled and actual strategy behavior are resolved | Complete within the agreed OVS/OLV scope: model corrections plus actual-inventory, pending-cap, stop handoff and half-day cutover deployed and verified, 2026-09-09 | Start with a bounded OLV/OVS question; explain the difference, consequence, and decision needed using reproducible evidence. No wholesale engine rewrite. |
| 4 | Discovery finds useful additions to the algorithmic mix | In progress: TLT, baseline earnings drift and the authorized 52-week-high/beat-size follow-up reviewed. No implementation candidate established; awaiting review | Reuse existing research; assess redundancy and distinct sources of return; distinguish discussion-worthy evidence from readiness to trade; calibrate email criteria explicitly. |

## Priority 1 scope

This is an answer-quality review, not a certification of every strategy or a new production rollout. Review the September 8 pitch working note and its input contract, a completed recent pitch, the latest strategy-discovery decision, and existing theoretical/actual and overlay explanations. Use dated saved artifacts as evidence of those artifacts only, never as proof of live freshness.

Questions for each sample:

1. What decision is the explanation helping the trader make?
2. Do the cited data and their dates support the factual claims?
3. Does "book" mean actual inventory, theoretical positions, or the configured strategy mix?
4. Is the rule executable as described, and are costs, sample limits, and approximations visible?
5. Does the conclusion follow, or does it turn missing information into a confident claim?

Corrections should use existing workflow entry points. Preserve original research evidence; add corrections rather than silently rewriting history. Do not loosen discovery thresholds as part of priority 1 (that belongs to priority 4), run new candidate searches, send messages, change trading rules, or redesign the site.

## Review evidence and disposition

See [the completed review and corrections](answer_quality_review_2026-09-08.md).

- Pitch: corrected the inference that zero staging meant no holdings or algorithm overlap; identified stale Trend context; added explicit input meanings and independent sleeve dates.
- Candidate evidence: reproduced September SPY statistics and corrected the permutation comparison (0.7354 for September, rather than 0.1618 for the original maximum month).
- Grid research: reproduced 17 metrics; corrected interpretation of absolute versus incremental returns, in-sample control, capacity, and capital normalization. The sampled rejection decisions remain supported.
- Portfolio: email now labels theoretical positions/P&L and modeled date. Existing site comparison/overlay explanations were appropriate and preserved.
- Verification: 92 Python checks and the JavaScript comparison check passed, plus saved-artifact recomputations and workspace hygiene.

Original research and delivery records remain intact; the linked review is the correction to use when reading them. No daily scan, new research run, external message, deployment, or trading change was made.

## Handoff

Priority 2 authorized next, including Execution-tab cleanup. Live inspection
found that the full hedge panel separates positions from working orders and
pushes the ticket down the page. The first bounded improvement is a direct
position-to-orders link, with ticket access alongside the main trading view and
expandable secondary help/tools. Preserve all trading behavior, confirmations,
contract identities, and account selection. Validate without submitting orders.

Owner clarification after priority 1: removed Daily Pitch's portfolio-overlap
criterion and portfolio/staging/sleeve/exposure inputs. Retained market context,
quality checks, daily cadence and pitch-history repetition rules. Removed scan,
fill and portfolio-report receipts from pitch-specific dependency checks; no
scheduled jobs were changed. This supersedes priority 1's newly added pitch
book/sleeve interpretation machinery, which is no longer needed and was removed.
The grid-study corrections and theoretical email clarification still stand.
Validation for this clarification: 103 targeted checks passed, including tests
that first failed when pitch assembly still consulted the book or exposure
state. No pitch rerun, message, daily scan or scheduler change was made.

Priority 1 is complete within the bounded review scope. Source corrections are verified in the development checkout; the separately pinned portfolio runtime has not been promoted, and no new unattended run is claimed. McKinley authorized priority 2 next, with Execution-tab cleanup as a secondary goal.

## Priority 2 implementation and verification

- Positions and working orders now sit together. Each position has an Orders link that opens its existing symbol group, replacing the manual scroll/search through the hedge panel and collapsed order groups.
- The new-order ticket sits alongside the book on wide screens and above it on narrow screens, with section links for direct navigation.
- Detailed ticket help, hedge/exposure, futures sizing and activity are expandable. Mode, account, connection status and expected-exit status remain visible.
- All existing trade actions, payloads, confirmations, risk checks and contract/account selection remain unchanged. Navigation preserves ticket drafts and unfinished Modify edits through polling refreshes.
- Verified: 28 targeted Python checks and all eight execution JavaScript suites; isolated desktop/mobile browser checks covering navigation, all ticket types, expiry fields, outside-hours selection, account switching and draft preservation. Browser fixtures intercepted all requests; no command was submitted.
- CI: 1,870 tests passed (96 skipped) on Linux and 235 passed (9 skipped) in Windows contracts; browser/Worker contracts also passed. [CI run](https://github.com/mslade50/New_Seasonals/actions/runs/34272122549).
- Rollout: [PR #28](https://github.com/mslade50/New_Seasonals/pull/28) merged as `cec5b8cb174f6c863f1438495930ed6cf5638fcc`. [Cloud build](https://github.com/mslade50/New_Seasonals/actions/runs/34272373541) passed every required stage, including the R2 provenance/freshness gate. Cloudflare production deployment `d7129dcd-b3f3-4685-846b-0af70fdf72f0` reports source `cec5b8c`; published at 16:27 ET on September 8.
- Authenticated live inspection: Execution loaded current broker positions/orders, the new Orders links, ticket fields and expandable sections. Portfolio loaded its metrics, charts and trade log with this build's provenance; Seasonal loaded the sizer and its empty qualifying-setups state. Desktop/mobile interaction testing used isolated fixtures; no live order action was exercised.
- Existing limitation retained visibly: Expected exits still reports that verification is unavailable, as it did before this change. Treat that feed as unverified. Diagnose its data/service path separately before relying on its alerts; this layout change does not repair it or certify broker fills.

This step improves navigation and layout. It does not claim additional arithmetic automation or certify broker fills. Production verification is complete for this UI change. Pause for McKinley's review; priority 3 remains queued. No daily scan, order submission, email or scheduler change was performed during this step.

## Subsequent position controls and priority 3 handoff

The later owner-requested Close/Add/Re-add consolidation was implemented and deployed separately after the layout step above (PRs #29 and #30). Primary now uses Close and Add quantity/percentage dialogs, proportional exit adjustment, and compact Re-add with active color. Source, deployment, tests, and limitations are recorded in the merged [position-actions notes](https://github.com/mslade50/New_Seasonals/blob/main/docs/execution_position_actions.md). The earlier layout-only statements describe that earlier release.

McKinley authorized priority 3 after reviewing the deployment. The first [OVS model-versus-execution review](ovs_execution_parity_review_2026-09-08.md) is complete. The suspected P1-only/fixed-dollar divergence was historical; current installed code supports P2 and scanner sizing. The actual differences are entry-day target timing, Friday timed-stop versus closing-price simulation, missing-stamp fallback sizing, and cap/split rounding.

Owner decisions after the review: keep same-day profit targets live, and deliberately exclude them from the daily-bar Portfolio model to avoid lookahead/within-bar sequencing bias. Retain Friday's timed live loss stop as the feasible production implementation, with the model's closing-price test remaining an acknowledged approximation. Both exit questions are closed; do not change either behavior or launch an intraday study merely to force parity.

McKinley then authorized implementation of the remaining sizing cleanup. Both corrections are now implemented in isolated branch `codex/ovs-sizing-parity-20260909`: the prepared broker candidate derives 20% P2 sizing / 1.125% daily-cap defaults from current strategy configuration and logs fallback use; the model follows the live whole-share cap/split sequence and recomputes PnL from final quantities. Five regressions first failed on the old model; the final targeted suite passed 89 tests, plus all four standalone EOD-DD cases. Tiny positions, missing/non-finite metadata, both cap stages, unfilled-budget consumers, and preserved exit rules are covered.

Review: [PR #31](https://github.com/mslade50/New_Seasonals/pull/31), source commit `7109fe3a`, merged as `7633af144bddcf9ffeafb8971f6425e98b91f3bd`. [CI run 34344673552](https://github.com/mslade50/New_Seasonals/actions/runs/34344673552) passed Linux's full pytest suite and browser/Worker tests, plus Windows process/allocation contracts.

The owner explicitly approved the shared fallback correction for both Primary and PA. The reviewed broker candidate was installed at 07:36 ET after idle/source-hash checks and a byte-verified backup. Valid stamped sizes retain precedence; only missing or invalid settings use the corrected 20% P2 multiplier and 1.125% daily cap. No broker process or task was started. The pinned producer was promoted and passed its existing ValidateOnly guard; 64 model/related tests also passed with its actual Python environment. The final runtime commit is `90c30dd698244ef46ff9dc43a74d6bfb0ff5b7e3`, immutable fallback tag `automation-runtime-2026-09-09.2`. This includes the already-approved Close/Add/Re-add UI because scheduled site builds use the pinned snapshot; retaining its old site files would undo that simplification. The exact hashes, backups, rollback, and verification are documented in [the cleanup handoff](../artifacts/task-worktrees/ovs-sizing-parity-20260909/docs/ovs_sizing_cleanup_2026-09-09.md).

Cloud publication completed at 08:03 ET in [build 34346385829](https://github.com/mslade50/New_Seasonals/actions/runs/34346385829), using canonical R2 inputs and source `7633af14`. Every required stage passed: ledger and seasonal generation, immutable R2 bundle publication and exact readback, site assembly, freshness/provenance gate, and Pages deployment. Cloudflare production deployment `0b46eaef-f29d-4578-b851-5f642278d9da` reports source `7633af1` on main. [PR #32](https://github.com/mslade50/New_Seasonals/pull/32) merged as `8952f835066f90570ad0d0f57a433ca9c5b6b6bb`, aligning the fallback controller with the final runtime tag. That follow-up changes the controller pin, tests and dated release notes; it does not change the published site/model source. Its final CI passed 1,931 Python tests (102 skipped), all 25 JavaScript suites, and 235 Windows contracts (9 skipped).

Authenticated live checks after deployment: Portfolio loaded its metrics, charts and trade log, with ledger vintage `gha:34346385829`, source `7633af144bddcf9ffeafb8971f6425e98b91f3bd`, and 4,711 trades. Seasonal loaded September 8 inputs, its manual sizer and the valid no-qualifying-setups state. Execution loaded online broker data (book 18 seconds old at inspection), working exits and the consolidated Close/Add/Re-add controls. No live action button was exercised. The pre-existing Expected exits verification-unavailable warning remains; this release does not repair or certify that separate feed.

Brief status: priority 1 review complete within its recorded rollout limits; priority 2 Execution simplification live; priority 3's bounded OVS cleanup complete across the shared broker stager, pinned producer, fallback controller and published Portfolio. OLV is the next bounded review after McKinley accepts this step; priority 4 discovery improvements remain queued. No daily scan, trade, email, or exit-rule change was performed. Production Portfolio regeneration used only the approved cloud workflow. Installation and read-only checks establish the reviewed code is deployed; they do not claim a new live order/fill test.

## OLV review — September 9

McKinley authorized OLV next. The [bounded OLV review](olv_execution_parity_review_2026-09-09.md) is complete, with 101 targeted checks passing (one skip) and additional installed-function/model reproductions. No production changes were made during the review.

The first repair should restore volume-confirmed exit decisions: the runtime has no reviewed inventory seed, the September 8 PM production scan explicitly skipped OLV exit evaluation for unknown inventory, and the scheduled broker consumer still uses the older due-today/date-matching contract. A zero/no-exit task result does not certify that chain. Reconcile actual tagged tranches and install the matching producer/consumer together; preserve or separately approve the existing PA route rather than silently replacing a shared runner with a Primary-only one.

Remaining findings are gap-fill target anchoring (model uses fill, live uses submitted limit), pending orders omitted from the per-stock cap, fixed 15:59 time exits on early-close sessions, and OLV daily-cap share rounding. Recommendations: retain live target anchoring and align Portfolio; reserve pending-entry capacity if the cap is intended as a firm ceiling; derive near-close times from actual sessions; floor final OLV shares and recompute PnL. Target anchoring and pending-cap behavior need owner decisions before changing those rules. The accepted same-day target modeling boundary remains unchanged.

Current brief: priorities 1 and 2 retain their recorded completion status; priority 3 has OVS deployed and OLV review complete with the stop handoff first in its repair queue. Priority 4 discovery improvements remain queued. This review does not certify OLV production stops or change OLV entries, exits, sizing, tasks, broker orders, emails or site data.

## OLV implementation after owner decisions

McKinley approved both choices: preserve the submitted-limit profit target and
align Portfolio, and count pending entries toward the existing stock cap.
Both are implemented in isolated branch `codex/olv-parity-review-20260909`,
along with OLV share flooring and PnL correction. The live loss threshold and
accepted exclusion of same-day target credit in Portfolio remain unchanged.
Final local verification passed 2,010 tests, with 55 skipped and two expected
failures, using hash-verified historical sources for the older broker-patcher
fixtures. Workspace hygiene passed.
See the [implementation and rollout prerequisites](../artifacts/task-worktrees/olv-parity-review-20260909/docs/olv_implementation_2026-09-09.md).

Read-only production checks found further prerequisites for the stop/cap input:
the collector lacks remaining order quantities and per-account observation
times, and the fills endpoint/R2 status lack the required completeness
attestation. Older active tranches also need reviewed entry/ATR reconciliation;
the signal log alone does not establish all frozen inputs. Prepared broker
candidates are not installed. The Primary-only exit candidate must not silently
replace the shared PA route.

Priority 3 remains open. Next is the coordinated inventory/feed/stop repair,
then an approved production cutover and regeneration. The current optional
overlay fallback still lets valid trades proceed when inventory is unknown;
do not describe the pending cap as a hard guarantee in that state.
The early-close deadline issue remains open. No daily scan, trade, email,
scheduled-task change, production installation or deployment was performed
during this implementation.

## OLV deployment completion

The owner authorized deployment. [PR #33](https://github.com/mslade50/New_Seasonals/pull/33)
released the model/cap source; deployment QA then caught and corrected separate
target reconstruction in the site table/chart via
[PR #34](https://github.com/mslade50/New_Seasonals/pull/34).
The regression reproduced 102.00 versus 104.50 after a gap fill and passes after
the site uses the engine's recorded target. Corrected OLV charts use new keys.

The final [cloud build](https://github.com/mslade50/New_Seasonals/actions/runs/34367148325)
passed every stage and the freshness gate. Cloudflare production
`a0b8560e-197f-4cbc-aa09-d888bbe46166` serves `e9e568f`.
Runtime v9 uses `b17cd79d`, immutable tag `automation-runtime-2026-09-09.6`,
with matching source and a saved predecessor marker. ValidateOnly and 142
installed tests passed (one skip). Final CI passed Linux and Windows.

Independent canonical R2 verification confirmed all 316 OLV targets, including
76 gap-improved entries, follow the submitted-limit formula. Authenticated
Portfolio, Seasonal and Execution checks passed; the corrected RTX target and
new OLV chart were checked directly. See the [deployment record](olv_deployment_2026-09-09.md)
for evidence, rollback details and the intermediate build stopped before publication.

Priority 3 remains open specifically for the actual-inventory/fill-continuity
inputs, live pending-cap/volume-stop handoff, and early-close deadlines.
The source deployment preserves the explicit optional-overlay fallback;
it does not make the live cap available with unknown inputs.
No daily scan, trade, email, broker-runner invocation or scheduler cadence
change was performed. Priority 4 remains queued.

## Inventory inputs follow-up — September 9

The read-only broker collector now publishes remaining/filled order quantities
and per-account query timestamps. Both live account feeds were verified; all
24 Primary and 11 PA orders carried the new fields. No trading runner changed.
The owner clarified that untagged TWS trades stay discretionary unless assigned.
Reviewed execution allocations and an inventory reconciliation report are
implemented on `codex/inventory-inputs-20260909`; 148 focused tests pass.

Priority 3 remains open. The specific assignment of yesterday's 753-share D
sale is pending, the older D stop metadata remains unverified, and canonical
fill continuity still needs a coordinated relay/runtime repair. No seed was
activated and no daily scan, trade, email or scheduler change occurred.
Detailed record: [inventory inputs](../artifacts/task-worktrees/inventory-inputs-20260909/docs/inventory_inputs_2026-09-09.md).

## Priority 3 completion package — September 9, after close

The owner confirmed the D sale remains discretionary. The historical target
difference is verified in the existing dividend-adjustment audit. The older
520-share D tranche exited through its existing 15:59 order during the review;
five current OLV tranches now reconcile to broker holdings and tagged exits.
Their reviewed starting snapshot is prepared, not activated.

Implemented the shared R2 seed reader, live fills/order observation, verified
continuity across retained and canonical history, entry metadata capture,
Primary-only stop table/consumer with PA preserved, and NYSE half-day deadlines.
The backup scan receives the same live inputs. Final local verification passed
2,057 Python tests (55 skips, two expected failures) and all 26 JavaScript files.
Actual raw-bar validation produced zero OLV exit proposals, using inert writes.

Priority 3 remains open only for the deployment prerequisites and coordinated
activation: the active TWS history setting is still one day; the accumulated
site log does not itself prove missed executions can be recovered. Seven-day
history must be verified, then the seed, relay, pinned producer and broker
candidate activated together with the required financial approval. No daily
scan, order runner, test trade, email, seed upload or trading deployment occurred.
Priority 4 remains queued. See the [cutover package](../artifacts/task-worktrees/inventory-inputs-20260909/docs/olv_inventory_cutover_2026-09-09.md).

Source handoff: [PR #35](https://github.com/mslade50/New_Seasonals/pull/35),
commit `639bbcb7`, is a draft pending activation prerequisites. Its
[Linux and Windows CI](https://github.com/mslade50/New_Seasonals/actions/runs/34406599388)
both passed. The active TWS saved settings were rechecked and remain one day.
No merge or production activation is claimed.

## Approved activation preparation — September 9 evening

The owner explicitly approved the coordinated Primary activation. PR #35 is
now merged as `cf5ec7f2420aa71dab16a7eb24af44a744fd0797`; post-merge CI passed.
The exact runtime candidate `24d32662`, immutable tag
`automation-runtime-2026-09-09.7`, also passed
[Linux and Windows CI](https://github.com/mslade50/New_Seasonals/actions/runs/34423835601).
The final fallback pin is prepared in [draft PR #36](https://github.com/mslade50/New_Seasonals/pull/36), not merged.
Installed runtime and active fallback remain `.6`.

Read-only verification at 20:56 ET confirmed the collector was running with a
six-second-old completed query. The site's retained log contains executions
for every trading day from August 31 through September 9. No missing trading
day has been established, and this task did not turn the collector off.
The separate canonical archive remains stale at September 2. An explicit
historical API request starting September 3 returned only today's nine fills;
the active TWS settings remain one day. Seven-day history is needed for the
new continuity/recovery contract; the TWS interface is unavailable to this
session, so the owner must save that setting. Activation approval persists.

Rollback backups are verified at
`trading_ibkr/.runtime_backups/primary_olv_cutover_20260910T010027Z/`.
The shared R2 seed and new Primary exit table are absent; the existing exit
table is empty. No broker source, journal, seed, exit table, installed runtime,
relay or scheduler was changed. No scanner, trading runner or email was invoked.
The sole user-side prerequisite is the TWS history setting; after verification,
finish the already-approved coordinated cutover and live validation.

## Gateway correction — September 9 evening

The owner clarified that the running application is IB Gateway, not TWS.
The preceding request to change TWS Trade Log settings was incorrect. Saved
TWS XML files did not establish the identity of the running broker application.
There is no such Trade Log setting to change in Gateway. The observed legacy
API connection (server version 176) returned only the current day's executions,
consistent with IBKR's documented Gateway limitation. Newer API documentation
includes additional execution-history filters, but compatibility with this
installed connection has not been established and no upgrade is authorized
or required merely to follow the incorrect TWS instructions.

Do not activate the prepared `.7` cutover on the assumption that seven-day
TWS history will become available. Replace that prerequisite with a tested
Gateway-compatible persistence/continuity and recovery design. Retain the
site's collected execution history. No missing trading date has been found;
the stale canonical archive is a separate issue. No Flex reporting credentials
were found in the two configured environment files or named reporting files
in the broker directory; that is not proof none exist in Client Portal.

Activation approval remains valid; the backend history design is the remaining
work, not a user obligation to change an unavailable Gateway setting. The
installed runtime, broker files and seed remain unchanged. No daily scan ran.

## Priority 3 completed — September 9 evening

The owner approved deployment. The preceding Gateway hold is resolved; no TWS
setting change is required. The final installed runtime is `b5fa0093`, immutable
`automation-runtime-2026-09-09.9`, with a matching main fallback pin. Both
Linux/Windows exact-build CI and the installed ValidateOnly guard passed.

Primary now reads the reviewed shared opening inventory, fresh retained fills,
and remaining pending orders for OLV sizing and volume-stop decisions. The
strict Primary exit consumer uses its separate table and journal, preserving
PA's existing route. New Primary OLV deadlines respect NYSE half-days.
Gateway continuity is limited to the reviewed non-overnight OLV stock route;
it requires an observation after 20:00 ET on each trading day. Missing sessions
remain unknown, and the authorized optional-overlay fallback remains in place.

Production verification exercised the actual installed reader from a stale
feed with the command agent offline. It refreshed the read-only collector and
deployed relay, returned five known OLV tranches with zero pending OLV entries,
and left the UI book and agent state unchanged. Raw-bar stop evaluation used
inert writes and produced zero proposals. No scan or trading runner was used.

The stale archive was updated: 71 executions added to the previous 182, for
253 through September 9. R2 data/status hashes match, prior generations are
preserved, and no retained-window gap was found. The 753-share Primary D sale
remains discretionary; no historical allocation was changed.

Live QA caught and repaired a missing public Worker route and subsecond
Cloudflare receipt-clock skew. Regression tests cover both; final CI passed.
The deployed Worker is `95786eec`; its final deployment is
[run 34427623216](https://github.com/mslade50/New_Seasonals/actions/runs/34427623216).
Runtime CI is [run 34427970035](https://github.com/mslade50/New_Seasonals/actions/runs/34427970035).
See the [completed cutover record](../artifacts/task-worktrees/inventory-inputs-20260909/docs/olv_inventory_cutover_2026-09-09.md).

Completion is based on installed-code, live read-only, archive readback and
inert execution tests. It does not claim a new live fill or next-day scheduled
run. The separate expected-exit monitor's scheduling/UI warning and automatic
Flex recovery are not certified by this step. No daily scan, test trade,
trading-runner invocation, email or scheduler cadence change occurred.

Priorities 1–3 are complete within their recorded scopes. Priority 4 remains
queued; wait for McKinley to review this step before starting it.

## Priority 4 first step — September 9 evening

McKinley authorized the intake review and initial research shortlist. That step
is complete: [six ranked leads and the recommended first investigation](../artifacts/task-worktrees/priority4-shortlist-20260909/artifacts/priority4/strategy_shortlist.html),
with a [supporting working note](../artifacts/task-worktrees/priority4-shortlist-20260909/docs/discovery_shortlist_2026-09-09.md).

Start by validating the existing TLT month-end study. It has reusable work and
a plausible structural-flow mechanism, but its old graduation label is not
enough: recent results are weak, modeled gross edge/cost is 4.30x versus the
current 5x email requirement, and Trend already includes TLT. Its monthly
exit-P&L correlation must be replaced by daily algorithm-book fit before any
claim of complementarity. No current positions were consulted.

The accepted September 7–9 intake contains 134 distinct papers; two local
adaptations were tested and reasonably rejected. Broad deposit searches add
substantial irrelevant material. Keep a persistent curated research queue,
read the best lead's methodology, and distinguish source findings from our
experimental rules. Retain strict downstream email criteria during the first
study. The existing registry's September 6 Legend/hedge status is not current
enough to establish absence of algorithm overlap.

The next bounded deliverable is an unchanged-rule TLT validation with realistic
costs, recent evidence and daily algorithm-fit results, ending in a clear
reject/unresolved/advance decision. Review this shortlist before starting it.
Priority 4 remains open. No scan, discovery run, email, trade, production change,
threshold change or fresh market-data backtest occurred in this first step.

## Priority 4 TLT validation — September 10

The owner authorized the next investigation. The fixed TLT T-5-to-month-end
validation is complete with a **retain research-only / do not implement now**
recommendation. See the [full report](../artifacts/task-worktrees/tlt-validation-20260910/artifacts/tlt-validation/tlt_validation.html)
and [working record](../artifacts/task-worktrees/tlt-validation-20260910/docs/tlt_validation_2026-09-10.md).

Fresh prices closely reproduce the earlier event returns. Daily marked
drawdown is 17.60%, versus the old 12.10% at completed trades. Since 2020,
mean return after 10 bp trading costs and a cash proxy is +19.16 bp/event,
median -7.13 bp and conventional t-statistic 0.97. Recent uncertainty remains
substantial. The current gross/cost email condition also remains unmet.

A partial core+Trend diagnostic shows low correlation but effectively unchanged
Sharpe at fixed research allocations. This is not full-book validation:
Event, Legend and hedge histories are absent, baseline financing is not
harmonized, and no combined capital cap is enforced. No current positions
were used. Six incomplete 2020 flat-pass OVS ledger rows and isolated
repo/fresh-price daily-return discrepancies are recorded for separate review;
they were not silently repaired or filled with zeros.

Recommendation: park TLT and investigate post-earnings drift's methodology,
liquid-universe evidence and feasible entry timing next. Review this disposition
before starting that step. No source rule, allocation, email threshold,
production data, broker state, scheduler or private-site change was made.
No daily scan or email ran. Priority 4 remains open; no new strategy is
approved to trade.

## Priority 4 post-earnings drift — September 10

The owner authorized the first PEAD investigation. The methodology review,
input audit and declared delayed-entry pilot are complete. **Park the simple
price-reaction version; do not implement.** See the
[report](../artifacts/task-worktrees/pead-feasibility-20260910/artifacts/pead/pead_feasibility.html)
and [working note](../artifacts/task-worktrees/pead-feasibility-20260910/docs/pead_feasibility_2026-09-10.md).

The corrected current-liquid-universe sample has 6,771 completed events.
Buying the strongest reaction fifth beats SPY by +0.23% on average over the
following 20 sessions after 20 bp round-trip costs; since 2024 it is +0.10%,
falling to -0.10% at 40 bp costs. Shorting the weakest fifth loses outright
and trails its signed SPY benchmark overall after modeled costs and borrow.
Long-side uncertainty intervals include zero. These event averages do not
establish an investable portfolio, historical universe eligibility or capacity.
No algorithm-fit simulation or current-position analysis was warranted.

The calendar has no release times or historical estimate vintages. The audit
also verified that ADBE's September 27, 2017 10-Q filing was counted as an
additional earnings event after its September 19 results release. Excluded
only that source-confirmed record in the research copy and retained initial
results; production data is unchanged. Carry this into a separate calendar
quality review before expanding earnings-dependent production behavior.

Verification independently reconciled 20,313 event/delay rows, 6,951 signals,
42 threshold quarters, costs, timing and exclusions. No parameter tuning.
Potential next earnings branch: timestamped guidance/transcript surprise,
starting with historical data availability and a feasible test. This is a
distinct proposal, not approval to buy data or build another pipeline.

Wait for the owner to review before the next study. Priority 4 remains open;
no strategy is approved to trade. No daily scan, email, broker access, order,
scheduler change, production data edit or deployment occurred.

## Priority 4 earnings price/proximity follow-up — September 10

The owner questioned whether conditional setups were being dismissed too
early and authorized price reaction, 52-week-high proximity and beat-size
tests. That bounded follow-up is complete. See the
[interactive matrices and report](../artifacts/task-worktrees/pead-feasibility-20260910/artifacts/pead/proximity/pead_proximity.html)
and [working note](../artifacts/task-worktrees/pead-feasibility-20260910/docs/pead_proximity_results_2026-09-10.md).

Adding proximity within 5% of the high to the strongest-reaction fifth yields
+0.22% versus SPY after costs overall, compared with +0.23% unfiltered. The
fixed >=5% reaction / near-high group shows +0.45% overall but +0.07% since
2024. The stored >=25% EPS-beat / near-high group shows +0.84% overall and
-0.43% recently, but revised estimates make that appendix non-point-in-time.
Entry, horizon and costs were unchanged. All declared cells and eras are
visible; no best-performing combined rule was selected.

Measuring proximity at D-1 rather than D-10 weakens the primary strongest-
quintile result to +0.03% overall and -0.23% recently. Removing its five best
events leaves -0.06%. Only 22 of 429 near-high strongest-reaction events had
sufficient same-date non-earnings controls; no earnings-specific advantage or
algorithm-book fit is established. Full price matrices cover 6,771 events.

Recommendation remains research-only. The proposed conditional tests were
worth doing, but did not establish a reliable recent improvement. Further
EPS work first needs a trustworthy historical surprise measure. Independent
formula checks and all 12 interactive matrix views passed. No daily scan,
broker access, current holdings, email, order, production edit or deployment.
Wait for owner review before another study.
