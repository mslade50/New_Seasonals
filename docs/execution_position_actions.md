# Execution: consolidated position actions

Updated 2026-09-08. Follow-up to priority 2 of the trading desk working plan.
Rollout authorized 2026-09-08. Primary backend installed and verified;
the cloud publication record is linked below. Priority 3 remains queued.

## User-visible behavior

- Primary stock rows have Close…, Add…, Re-add, and Orders. Re-add keeps that
  label and turns green when selected; selection alone sends no command.
- Close accepts shares or a percentage, including 100%. It reads the exact
  broker account/contract and current exits. A successful empty order lookup
  permits a bare close; a failed lookup never counts as an empty book.
- Existing exit allocations are normalized to the remaining holding, preserving
  their relative weights, prices and timing. OCA stop/target alternatives count
  once. Largest-remainder rounding yields the exact whole-share total; zero
  allocations are cancelled. Full closes cancel the closing exits.
- Exits are adjusted before submitting the close. While a close is working, its
  unfilled quantity reserves closing capacity. A cancelled or partially filled
  close restores coverage to the actual remaining holding.
- Add accepts shares or a percentage. Existing exits first normalize to current
  holdings; each new allocation has its own entry parent and matching attached
  exits. Add/re-add require an existing price stop or scheduled time stop.
- Re-add stages only the confirmed closed quantity, after the close becomes
  terminal, at the original average cost using a DAY limit with inherited exits.
  It does not submit a new entry after the original trading session or with
  expired inherited exit timing.
- PA retains its existing behavior. Options retain their combo controls.
  Primary futures/FX use Close; Add/re-add are stock-only.
- Old snapshot age does not disable manual controls. Broker-side quantity,
  account, direction, exposure and identity validation remain authoritative.

## Broker behavior and limits

The narrow patcher modifies only Primary close_resize and add_to_position,
agent validation/preview, and background completion reporting. It retains the
original PA handlers and all existing account/type/live-mode gates. It exempts
Primary Add from its old hard-disabled lifecycle path; cancel, modify and legacy
trim_readd stay disabled.

The local operation journal records identity and attempted mutations before
transmission. Repeated commands cannot submit another close or re-add. An
uncertain acknowledgement, changed exit fill, unavailable owning client or
unreadable order identity stops further actions and reports reconciliation
required. Never erase a journal entry to retry an unresolved operation.

This is not an atomic transaction across TWS and the filesystem. A disconnect
after exit changes can leave reduced protection and require intervention.
Concurrent manual/TWS fills cannot be locked by the application. Resting limit
closes reserve inventory, so that quantity is not simultaneously covered by a
separate stop. Native broker acceptance, client ownership and partial-parent
callbacks still need broker-session verification; simulated tests are not
evidence of live fills.

Attached order release follows IBKR's documented parent/child transmit sequence.
Only OCA types with overfill blocking are supported for alternative exits.
Sources: [IBKR bracket orders](https://www.interactivebrokers.com/docs/general/order-types/complex-orders/bracket-orders)
and [IBKR order fields](https://www.interactivebrokers.com/docs/tws-api/ref/order).

## Verification

- 104 targeted Python checks passed; four existing tests remain expected failures
  because the installed legacy Add/trim handlers are still disabled.
- All 25 standalone JavaScript suites passed. Compact-control, Re-add and Add
  payload/confirmation tests are part of the existing fast-action suite.
- Simulations cover mismatched coverage, OCA alternatives, long/short and full
  closes, rounding to zero, delayed/partial/cancelled fills, owner failures,
  disappearing/filling exits, restart deduplication and expired re-add sessions.
- Five additional adapter checks parse the actual reviewed external source
  without importing it: native child sizes/release sequence, caps, stale Add
  inventory, uncertain release, and unchanged PA/live gates.
- Isolated Edge browser checks passed on desktop and mobile: toggle color,
  ticket navigation, full Close, short Add, quantity/percent validation,
  cancelled confirmation, PA preservation and stale-snapshot controls.
  All command calls were intercepted; no broker request was sent.
- Candidate generation verifies both source hashes and compiles all five
  candidate modules. Local candidate: artifacts/position-actions-candidate-v2.
  Browser evidence: artifacts/position-actions-browser.

## Coordinated rollout

1. Obtain immediate approval for activation under the global AGENTS.md
   financially consequential action rule. Target: Primary's OneDrive executor
   and agent, plus the matching private-site UI. No test trade is authorized.
2. Reverify the source hashes against position_action_source_hashes.json and
   check the running agent path. Preserve timestamped backups of every replaced
   file and its hash, including helper files if they already exist.
3. Install the reviewed five-file candidate together and restart only the
   execution agent. Preserve all environment gates, caps and account settings.
   Check connection/heartbeat and no-transmit previews; do not invoke a scan.
4. Merge the frontend source and deploy through the build-private-site skill:
   GitHub Actions deploy_site.yml on main, canonical R2 inputs, freshness gate,
   Cloudflare source SHA and authenticated Execution/Portfolio/Seasonal checks.
   Do not publish the unified UI before the compatible backend is installed.
5. Rollback: stop the changed agent, restore the saved files and restart with
   the prior configuration; restore the prior site commit through the same
   cloud build. Retain journals and reconcile any in-flight orders in TWS.
   Restoring code does not reverse broker orders or fills.

## Deployment record

User approved activation on 2026-09-08. PR #29 merged as
`80c299f40ee40ae55e07282d53e11796ec31d42e`. The five installed file hashes
matched the candidate manifest. ExecAgent restarted from PID 22140 to 41644;
its verified original files are in the broker checkout's
`.runtime_backups/position_actions_20260908T185604` directory.

The first TWS snapshot timed out; subsequent read-only checks confirmed the
agent online with fresh positions/orders for both accounts. Local deployed
validation/preview checks passed for partial Close, full Close with Re-add and
Add, with zero commands submitted.

A final integration check found that Add's nested parent receipts were not
recognized by the existing fill-price reconciler. The corrected receipt also
exposes only entry parent IDs in the existing top-level order_ids field. A
regression first reproduced the missing fill price, then passed with both
entry allocations combined and exit/other-account fills excluded. All 40
position-action, reviewed-adapter and fill-reconciliation checks passed.
The correction was installed at 19:06 ET; the agent reconnected and both
accounts again supplied fresh positions/orders. The final position_actions.py
SHA-256 is `262f760f1081c069be59e20cb8bb05e3eb05298d35a44f3283d8714f4e5112bd`.
Its immediate predecessor is preserved under
`.runtime_backups/add_receipt_20260908T190611`; all other runtime files retain
their original five-file candidate hashes.

Both repository-wide Linux and Windows CI jobs passed for
[PR #29](https://github.com/mslade50/New_Seasonals/actions/runs/34281791620)
and the [receipt correction](https://github.com/mslade50/New_Seasonals/actions/runs/34289098261).
For future local adapter tests against the original reviewed executor, set
`IBKR_REVIEW_SOURCE` to the initial verified backup directory above. These
tests parse the source without importing or running it.

The frontend source is commit `80c299f40ee40ae55e07282d53e11796ec31d42e`.
[Cloud publication and freshness record](https://github.com/mslade50/New_Seasonals/actions/runs/34288428684)
tracks the required R2-only build and production deployment.
No daily scan, trade or email was initiated during this rollout.
