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

All original legs are sized and preflighted separately for Primary and PA.
The execution record is keyed by agent, source idea, immutable proposal version
and account. A permanent source/account guard additionally prevents a new version
from re-entering an already submitted idea. Existing receipts are retained; retries
never resume unsent legs. FULL synchronous transactions persist submitting before
any wire call, and an OS lock serializes the operation.

Delivery, working bracket, partial entry fill, full entry fill and closed position
are distinct states. Broker evidence must prove exact account, qualified conId,
quantities, parent/children, types/TIF, prices, arming/expiry, OCA and identities.
`Filled` status alone is insufficient without actual cumulative fill quantity.
Completed-order defaults are treated as missing quantities. Closed requires matched
explicit entry/exit quantities and terminal evidence for the parent and every owned
exit. A zero-fill cancellation also requires every owned child to be terminal;
working parent remainders or OCA siblings require reconciliation. This path
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
that have no unresolved price exits. Delivered quantities remain reference-only. Execution quantities are recalculated
from the delivered sizing instruction and fresh exact-account USD NLV. No proxy
substitution, automatic front month, leg selection, or cross-account reallocation
is permitted.

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

Both agents explicitly support Primary and PA, with separate per-account proposal
bindings, broker previews, confirmation dialogs, quantities, risk and saved intents.
The delivered sizing spec preserves risk_bps/NAV-percent mode, normalized leg
weights, ATR risk unit and multiplier. Pitch uses its established 30-bps default;
Seasonal uses its 15–50 bps band and explicit catastrophe-sizing distance. These
agents do not apply the systematic book's GRM, tilt, or older Seasonal-ticket
13-bps midterm rule.

Primary uses the agent's 1.0 risk multiplier. PA's agent multiplier is a required
explicit policy (`REVIEW_EXECUTION_PA_RISK_MULTIPLIER=1` or `1.3`); when absent,
the page and endpoint show PA blocked. The systematic staging book's 1.3 does not
establish this agent policy by itself. No environment setting is changed here.

Completed broker account-summary request results supply exact-account USD NLV,
buying power, available funds and excess liquidity, with an observed timestamp
no older than 60 seconds. Cached balances and the publisher's reference $750k basis
cannot substitute. Qualified stock/ETF contracts have multiplier 1 and lot 1;
unsupported instruments block the whole account idea. Any leg rounding to zero
blocks that account; no remaining budget moves to another leg or account.

Read-only margin/permission inquiries must match the exact account, conId and
sized quantity. Gross idea notional must fit fresh buying power and existing
hard account caps; summed positive initial/maintenance margin changes must fit
available funds/excess liquidity. Negative margin changes never subsidize another
leg. Existing positions or working orders in an idea's contract require a separate
inventory decision; this adapter does not add, net or reverse them automatically.

Publisher ATR-risk limits are 60 bps per idea and 150 bps per agent slate, applied
once under the configured account multiplier and existing adapter hard cap.
Durable staged-day claims count submitting, uncertain and closed ideas; only a
proved zero-fill whole-chain cancellation releases staged risk. Each agent/account
has its own ledger; no global pooled risk cap is invented. Confirmation rechecks
fresh equity/capacity and exact quantities; any changed plan requires a new preview.

Source-leg manual/trail flags still block submission. The legacy Pitch runner
must be known disabled; a present pitch_moo_enabled.flag blocks the new adapter.

The generic `/exec-command` cannot sign `review_execution` commands. When the
new site live gate is explicitly enabled, tagged Pitch/Seasonal agent requests
through that old route are also blocked. Untagged unrelated manual execution is
outside this adapter. Existing schedules/settings are not changed by this source.

## Configuration and later user-controlled handoff

The following are **configuration requirements for a later handoff**, not actions
performed by this release:

1. Confirm both agent products' Primary/PA endpoint identities and resolve the
   PA agent risk-multiplier policy. Keep site and local policy values consistent.
   Existing authentication is reused; no credential or grant change is introduced.
2. Verify the current four runtime code hashes against
   `broker_runtime/review_execution_source_hashes.json`. Prepare an isolated
   candidate with `prepare_review_execution.py --source <reviewed runtime>
   --output <new ignored artifact directory>`. Preparation only writes and
   compiles candidate files; it never installs, imports or runs broker entrypoints.
3. Review existing Pitch/Seasonal orders, fills, competing runners and inventory
   before selecting the persistent journal path. Initialize that new SQLite
   journal explicitly with `review_execution.Journal.initialize(path)` once.
   Existing or corrupt journals are never replaced/reset automatically.
4. A separate authorized user runtime promotion must install all six prepared
   files, including `review_sizing.py` and the corrected `broker_reconciliation.py`, compatibly with the
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
