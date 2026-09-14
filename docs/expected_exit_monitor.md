# Primary expected-exit monitor

`scripts/monitor_expected_exits.py` evaluates reviewed actual tagged inventory against fresh exact Primary broker positions and execution evidence. It performs no broker or cloud operations. Its CLI accepts named local `--inventory`, `--book`, `--fills`, `--state` and `--output` JSON paths. `--asof` supplies a timezone-aware fixture clock; email requires explicit `--send`. Production source collection, publishing and scheduling are separate integration steps.

Inventory is the serialized `tagged_inventory` result (`status`, `asof_utc`, `tranches`). Each tranche must carry Primary account, conId, strategy, tranche identity, reference date, signed quantity and an explicit `exit_deadline_utc`. Reviewed opening inventory and continuous execution history remain prerequisites; theoretical targets or a missing position row never establish the starting allocation.

The broker book contains `at` and one `accounts` entry with `key: primary`, exact `broker_account`, `positions` and `orders`. Position rows carry exact account/conId. The `/fills` payload requires `completeness.accounts.primary` with exact matching `broker_account`, `complete: true`, `received_at` and `source_at` (or `complete_through`). Truncation, merge errors, unresolved incomplete days, missing account identity, naive timestamps or evidence older than 90 seconds produce an explicit unverified result. A PA outage alone does not invalidate Primary evidence.

Before the explicit deadline plus five minutes, an open obligation is pending. Afterward it is missed. Missing evidence produces `unable_to_verify`, never a fabricated zero. An obligation resolves only when trusted attributed inventory is flat with matched effective closing execution evidence. Corrections are deduplicated before contract/reference/side attribution. Aggregate broker net zero cannot erase an open algo tranche.

Execution corrections are deduplicated before account attribution too, so an account-changing revision revokes its earlier allocation. Conflicting same-revision identities and malformed or future execution timestamps are unverified. Accepted inventory timestamps are monotonic. An invalid, stale or older observation may produce a warning but cannot overwrite a prior deadline, resolved state or tranche episode.

The suggested local output is `data/expected_exit_status.json`; the integrated status publisher can publish it to `ops/expected_exit_status.json`. Schema 1 output contains `generated_at`, `account_key`, `status` (`ok`, `degraded`, `attention`), `counts`, `obligations`, `source_error`, `notifications` and `summary_due`. Each obligation exposes its identity, deadline, remaining tagged quantity, exact net broker quantity, confirmed closing quantity, status and explanation.

State preserves obligations after they leave the current inventory, deduplicates alerts, records resolution and distinguishes a reopened episode. At or after 16:10 ET on a trading session, one summary is queued only when missed or unverified obligations remain. A healthy day or a future pending deadline produces no routine summary email; a later exception remains eligible that day. The CLI holds the existing cross-process `GlobalFileLock` around the complete state/read/evaluate/send/write transaction. It writes a durable `sending` claim before calling the email adapter. Exceptions and `False` acknowledgements are `delivery_unknown`, because the existing SMTP adapter cannot distinguish a pre-send failure from failure after accepted DATA. These claims are not automatically retried. A crash after the claim likewise requires operator reconciliation before a resend.

The original monitor implementation did not register a schedule or publish a report. Fixtures cover the five-minute boundary, stale/naive timestamps, exact accounts, correction changes, netted holdings, dedup/resolution/reopening, the 16:10 summary and ambiguous SMTP delivery.

Observation-only runs may queue notifications, but each evaluation supersedes obsolete unsent events while preserving their history. A resolved issue observed only in shadow produces neither the obsolete warning nor a new resolution email when sending is later enabled. Resolution notices require a corresponding sent or ambiguously delivered alert. Repeated coverage outages use distinct episodes; old unsent daily summaries are superseded. The integration wrapper must publish reports monotonically so a delayed observer cannot overwrite a newer cloud projection.

## 14 September producer repair and deployment

The production read-only observation reproduced a report-writing failure after
16:10 ET: the trading-calendar predicate returned a NumPy boolean, which the
JSON encoder rejected. The summary predicate now returns a native boolean and
the CLI regression exercises report and durable-state serialization at 16:10.

The adapter now obtains one `/fills` payload and uses its matching book for
inventory and deadline evaluation. If the payload has no book, the existing
`/book` GET remains an explicit fallback. It supplies that same fill payload to
the inventory adapter, preventing the latter's local Gateway refresh and cloud
publication fallback. Source failures produce a current degraded report with
the inventory failure boundary; they cannot fabricate empty holdings.

`--exec-env` loads only `STATUS_TOKEN` and `EXEC_BROKER_URL` from the existing
broker configuration. The shared configuration root supplies R2 credentials.
The optional `--algorithm-catalog` remains supported; otherwise the adapter
builds the complete catalog from its pinned source. Deactivated/reference
strategies remain included. No email flag is enabled by the scheduled wrapper.

`scripts/expected_exit_task.py` creates an inert, disabled Task Scheduler XML.
`scripts/run_expected_exit_monitor.ps1` verifies the exact Git commit and clean
tracked source before running the producer. The task repeats every minute,
ignores overlapping invocations, runs hidden as the signed-in owner, and stops
an invocation after three minutes. It cannot launch the execution agent, a
scan, an inventory refresh, or an email. A logged-out owner or sleeping machine
will stop observations and the existing dashboard age check will show stale.

Deployment sequence, after the reviewed source is merged and the full SHA is
known:

1. Create a dedicated clean runtime worktree at that SHA; use the verified
   Python environment that passed the monitor tests. Keep the runtime worktree
   separate from active development and the command-agent directory.
2. Run `run_expected_exit_monitor.ps1 -ValidateOnly` with the exact
   `-RuntimeRoot`, `-ConfigRoot`, `-Python`, `-ExecEnv` and `-PinnedSha` values.
   This validates the pin and CLI without reading broker sources or publishing.
3. Run the Python adapter once with those configuration paths, an artifact
   state/output directory and no `--upload` or `--send`. Verify the actual
   source timestamps and truthful report. A degraded inventory result is a
   working producer reporting a dependency failure, not healthy exit coverage.
4. Generate XML with `expected_exit_task.py --runtime ... --config ...
   --python ... --exec-env ... --sha <full SHA> --user <Windows owner>
   --start <timezone-aware timestamp> --output <new artifact XML>`. Register it
   under `New Seasonals Expected Exit Monitor` using `schtasks /Create /TN ...
   /XML ...` without a force-replacement flag. Confirm its action and disabled
   state before enabling it. Existing definitions require comparison and a
   backed-up, explicit update instead of replacement by name alone.
5. Enable and run that exact task. Confirm two increasing report timestamps at
   `ops/expected_exit_status.json`, successful task results, and the private
   `/expected-exit-status` projection in Execution. Local generated status is
   not proof that the cloud projection is working. No site rebuild is required
   merely to refresh this live R2 endpoint.

The stable monitor state and per-run observations/logs are stored under
`<ConfigRoot>/artifacts/expected-exits/`. Rollback disables the new task and
preserves its state and observations. Do not delete history or restore an older
report over a newer generation. No broker order or financial state is changed
by this producer deployment.

At 21:55 UTC on 14 September, the status object did not exist and no matching
scheduled monitor was found. Fresh Primary broker receipts were available, but
the reviewed inventory seed was dated 10 September 01:51 UTC. Canonical/live
general continuity started on 14 September; OLV-only continuity started on 11
September. The archived 10 September receipt ended at 02:07 UTC and did not
overlap the 11 September receipt's 04:00 UTC start. The R2 archive contained
immutable fill rows without corresponding historical coverage receipts. These
records cannot establish uninterrupted history for the missing 10 September
session. The correct observed status remains `degraded`, with one explicit
unverified inventory-coverage obligation. Restoring that coverage requires
authoritative execution-history proof or a separately reviewed new opening
allocation; this repair does neither and does not narrow the monitored scope.
