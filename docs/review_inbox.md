# Daily Pitch and Seasonal human review inbox

Current implementation: [single review and automatic staging](review_and_stage.md).
The Review page now uses **Yes — stage orders** and **No — pass**. A new Yes
decision atomically reserves staging requests for the published accounts and
hands them to the existing broker process. No separate preview or confirmation
screen is required. The source is prepared; production activation is a separate
owner-approved handoff. Existing research-only approvals are never replayed.

## Historical research-only workflow

The inbox records human **review decisions** on exact delivered proposals. It
does not allocate capital, set Sheets `Approve=Y`, enable a runner, stage a ticket,
submit a broker command, or send messages. Silence never approves anything.

## Source and persistence

`daily_pitch.publish_review_payload` runs after a confirmed send and before a
Sheets write can fail. It requires the matching sent receipt/verdict. Dry run and
validation stop before this hook; `--no-send` cannot publish reviewable proposals.
Explicit receipts/custom journals stay local-only. Both products keep their
existing journals, receipt namespaces and execution integration settings.

`scripts/sync_review_inbox.py` is the cloud bridge for existing pinned publisher
runtimes. It downloads only the existing R2 journal and receipt for each product,
uses `verify_sent_receipt` to prove the complete delivered verdict, and reconciles
the same review ledger without running an agent or publisher. Missing receipts
remain explicit missing feeds. Ambiguous sends, journal mismatches, and incomplete
order rows fail; they are never converted to a stand-down. Directed amendments
retain the latest delivered record per source idea ID. Original quantities and
all order fields including Seasonal trails are retained; the execution-consumed
Sheet approval flag is excluded. No broker balance or inferred capital base is
used to resize a trade.

The canonical store is `review_inbox/v1/<pitch|seasonal>/<YYYY-MM-DD>.json` in the
existing private R2 bucket. Each daily object contains immutable version envelopes,
current version IDs, the confirmed delivery identity, and review events. Publisher
and endpoint use conditional writes to this **same object**: S3 If-Match/If-None-Match
and Worker `onlyIf.etagMatches`. A lost conditional write reloads and checks the
exact proposal/revision again. It never overwrites a competing decision or resurrects
a superseded version. Local publisher file locking also protects the staging file.

SHA-256 covers the exact canonical JSON string, including all leg fields, account
status, thesis/evidence, dates, and review deadlines. The full hash is checked on
read and decision. An unchanged current proposal retains its original publication
timestamp on reconciliation; changed content gets a new ID and requires fresh
review. An approved old version remains in history and never authorizes its replacement.

The ledger supports application-enforced append-only review history. It is not an
independent tamper-proof archive against someone with direct administrative bucket
write access. No such access is added by this implementation.

## Identity and review behavior

`/review-inbox` verifies the existing Cloudflare Access JWT in code. Signed subject,
issuer, audience and expiry establish the actor; client actor/email/time claims do
not. Missing existing Access configuration fails closed. POST requires same-origin
JSON, exact product/ID/full hash, expected revision, explicit confirmed decision,
and an idempotency UUID. GET history is authenticated and read-only. No credentials,
new Access application, policy edit, grant, or additional storage binding is required.

The endpoint also verifies that the daily ledger still matches the current sent
receipt. If a new delivery has not been reconciled or its send is ambiguous, the
old feed is marked stale and new decisions are blocked. Identical confirmed retries
can report their original event without writing another one. A network failure or
login interruption never triggers an automatic POST retry. The UI tells the user
to check central history, sign in, and explicitly confirm again if still pending.
Session storage holds only the unfinished decision intent/idempotency ID, not the
authoritative decision ledger or an authentication token.

The default view is pending proposals sorted by deadline. Both products show feed
health independently. Stand-down, missing feed, historical date, stale delivery,
superseded proposal, and failed state reads have distinct behavior. The date picker
opens historical review state with decisions disabled. A phone and desktop read
the same server ledger. The Review nav badge fetches only read-only state.

## Deadlines, accounts, and manual execution

Review deadlines use explicit `Execute_On` dates and the existing XNYS calendar:
MOO closes five minutes before session open; MOC closes thirty minutes before
session close (including early closes); LIMIT review closes at the first session
close. GTD expiry does not extend the first-session review/stage window. A holiday
or missing date is not silently rolled. A late delivered proposal is displayed as
expired, not extended. Receipt publication time and server time are timezone aware;
the UI displays ET. No intraday data request or order sizing occurs here.

Pitch retains the existing manual flow's `primary` account mapping. Seasonal has
no configured execution account in its publisher and says **Unassigned — manual
account selection required**. Seasonal, futures, proxies, and manual-only rows
have no asserted executable venue deadline or ticket prefill. A Seasonal review
approval is a review of the delivered idea, not approval of a broker account or
execution route.

After supported Pitch review approval, navigation opens the exact source idea in
the existing Pitch view with review ID/hash. That view rechecks current review
version/status and exact published order rows. Mismatch/auth failure blocks its
staging controls. Navigation itself creates no ticket and sends no command. Current
Pitch and Execution date/pass/instrument/confirmation gates remain in force. The
user must continue manually and avoid using both Sheet and site submission paths.

## Rollout boundary and supported cloud steps

Implementation and local tests may be completed without activation. The new sync
workflow runs every fifteen minutes on weekdays after activation and can also be
dispatched with a review date. It uses the existing R2 secret names with repository
contents read permission, and introduces no notification channel or execution
permission. GitHub scheduled runs can be delayed; the UI exposes missing/stale feeds.

**Pushing the new workflow to main enables that review-only cron.** Do not push,
dispatch the sync, publish review objects, or deploy until that cloud rollout is
authorized. Once authorized:

1. Push the reviewed commit to the existing main branch and require cloud CI.
2. Dispatch `Sync Human Review Inbox` for the current ET date and verify both
   product receipts were matched; a product with no delivery must remain missing.
3. Dispatch `.github/workflows/deploy_site.yml` on that commit. Follow
   `.agents/skills/build-private-site/SKILL.md`: R2-only cloud assembly, freshness
   gate, and Cloudflare deployment/source verification. Never deploy local dist.
4. Perform authenticated phone/desktop QA of `/review.html`, `/review-inbox`,
   history and supported manual handoff. Verify rejection/expiry/auth failure
   behavior using an explicitly approved test proposal, not a trading command.

No pinned-runtime change or local Scheduler operation is needed for the cloud
bridge. The prior denied Scheduler route is unrelated and must remain untouched.
Enabling trading, modifying Sheets approval consumers, implementing Seasonal
execution, granting access, or sending notifications requires separate authorization.

## Validation

`tests/test_review_publish.py` covers sent-receipt gates, original field preservation,
manual/unknown accounts, venue scope, holiday/early-close cutoffs, amendments,
stand-downs, dev isolation, local/remote locking, CAS merge and cloud/direct convergence.
`tests/js/test_review_inbox.mjs` covers signed Access identity, fail-closed configuration,
same-origin writes, concurrent decisions, retry/idempotency, source races, stale
deliveries, expiry, history and storage failure. Existing Pitch stage tests and
full site manifest tests remain required. Browser QA can use an isolated local
server with invented data and a mock R2 store; this is not proof of authenticated
production behavior or a live bucket write.

Official conditional-write reference:
https://developers.cloudflare.com/r2/api/workers/workers-api-reference/#conditional-operations
