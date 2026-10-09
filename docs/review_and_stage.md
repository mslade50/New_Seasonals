# One review, immediate staging

The Daily Pitch / Daily Seasonal Review page presents **Yes — stage orders**
and **No — pass**. Yes is the trading authorization. It immediately queues every
leg for the accounts shown on the proposal; there is no additional preview,
account-selection screen, checkbox or confirmation dialog. No records a pass
without requesting a note or staging an order.

The endpoint records the decision and all account requests in one conditional
write to the existing R2 review ledger, then forwards the saved commands through
the existing broker connection. The response can return while Cloudflare
`context.waitUntil` completes this short background handoff. The existing local
broker process sizes and submits the orders. No new service or Windows task is added.

The card shows each account as queued, staging, staged, blocked or needing checking.
A broker acknowledgement is not a fill. MOO/MOC orders are staged immediately
for their published auction; limits retain their published price rules. An
OPEN-derived limit approved premarket is saved in the local journal and stages
automatically at 09:32 ET, once the actual session-opening bar is available.
Pending jobs survive restart; a restart during submission reports uncertainty
instead of placing again. There is no automatic conversion to a market order,
next-day rollover or execution of an unsupported manual leg. OPEN-derived limits
may stage after recovery on the same day while their published approval and
entry deadlines remain valid; 09:32 is their earliest automatic staging time.

## Sizing

The delivered agent risk instruction and normalized leg weights are authoritative.
Publisher share counts use a reference equity value; the broker computes account
shares automatically from that same instruction and fresh exact-account equity.
Primary and PA retain the approved 1.0 multiplier: the same percentage risk on
each account's own equity. Systematic GRM and systematic PA multipliers do not apply.

Both accounts are included when both are explicitly published. No browser-supplied
symbol, quantity, account or price replaces the sealed agent instruction. Existing
account capacity, broker permission, risk and order checks run in the background.
Source instructions that cannot be executed appear blocked rather than silently
dropping a leg. Independent account orders and multi-leg brackets are not atomic;
a partial failure is shown and is not automatically undone or completed.

## Duplicate handling and failures

Each account receives one saved command UUID. Concurrent decisions and repeated
clicks cannot create a second account request. The broker's durable UUID record
and local source/account journal prevent a second placement after uncertain
delivery. Queuing, broker acceptance and fills remain separate recorded states.
Read-only refreshes never place an order. The existing execution-broker Worker
checks saved approvals once per minute and retries the same unexpired UUIDs.
The broker retries only queued or delivery-unknown handoffs; pushed/completed
commands are deduplicated. This recovers a Pages interruption or an offline
agent without a second click. An unavailable result remains needing checking,
never a proved zero-order outcome. Scheduled results are resent until the broker
acknowledges persistence; a repeated receipt cannot regress a later fill status.

Earlier events marked `human_review_only / not_submitted` retain their meaning.
They are not converted to trading approvals by deployment or page refresh. Only
a new Yes event marked `review_and_stage / queued` authorizes this flow.

## Activation handoff

This change is prepared source, not a live activation. The owner must approve
pushing/deploying and enabling the finished financial workflow before any live
runtime installation or setting changes.

1. Verify the four current runtime hashes against
   `broker_runtime/review_execution_source_hashes.json` and prepare the six-file
   candidate with `prepare_review_execution.py`. Preparation compiles files only.
2. After owner approval, back up and promote the six candidate files into the
   existing broker runtime, initialize a new persistent review journal once,
   and retain the legacy Pitch runner's disabled state.
3. Enable the existing review preview/live settings and journal path, and add
   `review_execution` alongside `entry_bracket` to the existing live type list.
   Preserve unrelated settings. Restart the normal existing broker process with
   the approved settings and verify both exact account identities.
   Verify `book.review_execution` reports the matching source SHA, v2 adapter,
   initialized journal, both accounts and both live types before enabling the site.
   The current runtime already has its general live gate enabled, so keep the
   new review live gate off until installation, journal and account checks pass.
   Existing v1 review journals receive an additive v2 migration preserving claims.
4. Deploy the approved commit using the cloud-only private-site workflow, then
   deploy the execution-broker Worker with its CHARTS R2 binding and minute
   trigger. Enable its review live flag and the same existing site preview/live
   settings after runtime readiness. Do not deploy local site data.
5. Verify authenticated page readiness without clicking Yes on a real proposal.
   The next new Yes authorizes its displayed accounts and orders. Do not replay
   today's already-recorded research-only decision as part of activation.

Rollback disables the review live flag on site, execution-broker and runtime. It stops new staging
requests; it does not cancel orders already placed. Preserve the journal and
review ledger so rollback cannot erase duplicate protection.

Offline validation covers the real endpoint and UI, account sizing, direct
staging without a prior preview, broker failures and partial submissions,
concurrent clicks, publication races and old research-only approvals.
