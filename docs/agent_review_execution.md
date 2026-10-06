# Daily agent whole-idea execution adapter

The Daily Pitch and Daily Seasonal review inbox records a human research decision;
it does not submit orders. The new `execute-review.html` flow requires a separate
read-only broker preview and explicit confirmation of every leg. Both adapter
flags default to off. This source release does not install or activate a runner.

## Architecture and lifecycle

`functions/review-execution.js` verifies existing Cloudflare Access identity,
the current immutable proposal hash, current sent-delivery receipt, unexpired
review approval, product account binding, and a broker-generated preview hash.
Browser quantity, symbol, exits and proposal payload are never trusted. It reserves
each UUID with R2 compare-and-swap in the existing review object before forwarding
the exact command through the existing STATUS_TOKEN HMAC/broker connection.
Execution approval is separate from review approval. Multi-leg and unprotected
risk acknowledgements are unchecked by default and required when applicable.

The source-hash-pinned local adapter runs after the existing agent signature/expiry
checks, through its shared execution mutex and bounded executor subprocess.
The existing exact-account resolver, contract qualification, quantity/notional
caps, live risk checks, `build_bracket`, and `guarded_place_order` remain the
placement path. Preview and reconciliation connect to IB with `readonly=True`;
only a confirmed execute operation can request a writable connection, after
both new and existing gates pass independently in agent and executor.

All original legs are preflighted before the first placement. A permanent SQLite
claim is keyed by product, source idea ID and account, independently of proposal
version. FULL synchronous transactions persist `submitting` before any wire call.
An OS lock serializes the entire operation across processes. Retry of the same
intent returns its recorded state and never resumes unsent legs. A new UUID or
proposal version cannot re-enter an already claimed idea/account.

Delivery, working bracket, partial entry fill, full entry fill and closed position
are distinct states. Broker evidence must prove exact account, qualified conId,
quantities, parent/children, types/TIF, prices, arming/expiry, OCA and identities.
`Filled` status alone is insufficient without actual cumulative fill quantity.
Completed-order defaults are treated as missing quantities. Closed requires matched
explicit entry/exit quantities and terminal evidence for the parent and every owned
exit; working parent remainders or OCA siblings require reconciliation. This path
does not cancel or repair those orders automatically.
Fresh reconciliation is read-only and never sends a remainder, cancels a bracket,
rolls back a leg or recreates missing protection. Missing/corrupt local history,
unknown transport outcomes and incomplete evidence require reconciliation.

Independent multi-leg brackets are **not atomic**. After rejection or uncertainty,
the adapter stops. A partly completed idea requires human inventory review;
automatic retry, rollback and completion are intentionally unavailable.

## Supported instructions and exact limits

The shared route supports one to four exact US SMART/USD stock or ETF legs:
published CLOSE limits, broker-derived true-session OPEN limits, priced DAY/GTD
entries, stops, targets, future time exits, and source MOO/MOC entry instructions
that have no unresolved price exits. Quantities remain the published whole shares;
there is no resizing, proxy substitution, automatic front month or leg selection.

The executor's existing future time exit is a scheduled MKT/GTC child at 09:30
or 15:59 ET. It is **not a native future auction order**. The preview displays this
convention explicitly and confirmation acknowledges it. Stops arm next session
where the source instruction requires it. Native auction-price guarantees and
live fill behavior have not been verified by this offline release.

Conditional MFE trails, `Manual_Only` source legs, derivative contracts, proxy
instructions, repeated exact contracts, unsupported entry/exit grammars, and
missing source levels block the whole idea. They require separately implemented
native lifecycle support or a newly published exact supported instruction; this
adapter never silently drops an exit or substitutes an instrument.

Pitch binds to `primary` by default, matching its published proposals. Seasonal
has **no default account**. Its existing publisher marks the review manual because
integration/account selection were unavailable; an explicit execution binding is
separate and does not rewrite the original review payload. Individual source-leg
manual flags still block submission. The existing Pitch MOO runner must be known
disabled; a present `pitch_moo_enabled.flag` blocks the new adapter.

The generic `/exec-command` cannot sign `review_execution` commands. When the
new site live gate is explicitly enabled, tagged Pitch/Seasonal agent requests
through that old route are also blocked. Untagged unrelated manual execution is
outside this adapter. Existing schedules/settings are not changed by this source.

## Configuration and later user-controlled handoff

The following are **configuration requirements for a later handoff**, not actions
performed by this release:

1. Choose Daily Seasonal's exact account (`primary` or `pa`). Confirm published
   Pitch account and broker endpoint managed-account identity match. Site and
   local `REVIEW_EXECUTION_*_ACCOUNT` bindings must agree. No credential or grant
   change is introduced; existing Access and broker authentication are reused.
2. Verify the current four runtime code hashes against
   `broker_runtime/review_execution_source_hashes.json`. Prepare an isolated
   candidate with `prepare_review_execution.py --source <reviewed runtime>
   --output <new ignored artifact directory>`. Preparation only writes and
   compiles candidate files; it never installs, imports or runs broker entrypoints.
3. Review existing Pitch/Seasonal orders, fills, competing runners and inventory
   before selecting the persistent journal path. Initialize that new SQLite
   journal explicitly with `review_execution.Journal.initialize(path)` once.
   Existing or corrupt journals are never replaced/reset automatically.
4. A separate authorized user runtime promotion must install all five prepared
   files, including the corrected `broker_reconciliation.py`, compatibly with the
   existing reservation guard. No pinned runtime or Task Scheduler promotion is
   included here.
5. For preview-only verification, leave **all live flags off** and deliberately
   enable `REVIEW_EXECUTION_PREVIEW_ENABLED=1` on site and local runtime with the
   explicit persistent `REVIEW_EXECUTION_DB` path. Verify an authenticated whole-
   idea preview, exact account/contract/timing/caps, and durable history.
6. Submission additionally requires explicit `REVIEW_EXECUTION_LIVE_ENABLED=1`
   on site and runtime, existing `AGENT_LIVE_ENABLED`, correct live account scope,
   and both `review_execution` and `entry_bracket` in existing live types. A future
   user-controlled activation needs a separate live validation plan. This release
   does not enable any of these flags, deploy an order worker, start a service,
   dispatch an execution schedule, stage paper/live orders or change broker state.

Preview risk is limited by the existing account notional caps and a whole-idea
cap of at most 100 bps (`REVIEW_EXECUTION_MAX_RISK_BPS` can only lower it). Source
contract or risk/timing changes after preview require a fresh explicit preview.
No cap, quantity or unsafe account fallback is relaxed to make a proposal pass.

## Validation

Offline tests cover both products, exact account/version/review/delivery/expiry,
all-leg mapping and caps, unchecked confirmations, R2 publication/concurrency
races, permanent claims, crashes after receipt, explicit retry, uncertain delivery,
broker acknowledgements/rejection, partial/full entry and exit fills, ring eviction,
missing/corrupt journals, read-only reconciliation, and saved-intent restoration.
Local AST tests exercise the actual patched executor and native four-order builder
with fake order objects; no production module import or socket is used.

Authenticated production GUI QA and actual broker fills remain unverified. Cloud
production site deployment, if requested, must use the repository's cloud-only
private-site build skill; local test fixtures are not production freshness evidence.
