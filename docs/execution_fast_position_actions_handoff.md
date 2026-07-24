# Handoff: execution-tab fast position actions

## Implementation status (2026-07-22)

- Execution-machine implementations are installed in `OneDrive/trading_ibkr`
  with rollback copies in `_backup_20260722_execution_fast_actions`.
- `book_snapshot.py` now publishes exact contract identity plus OCA/owner fields;
  `exec_agent.py` validates and previews both new command types; and
  `execute_order.py` contains independently gated live handlers.
- The private-site controls and payload tests are implemented on branch
  `codex/execution-fast-actions-local`, but the site has **not** been deployed.
- The new command types are **not** in `LIVE_TYPES`; the agent has not been
  restarted for them; no live or dry-run relay command has been sent.
- Broker-free new-action tests and the existing bracket, flatten/cancel, modify,
  option-spread, JavaScript, and full static-site build checks pass.

## Objective

Add two bracket-aware fast workflows to the private site's Execution tab:

1. **Trim with optional re-add** — close part of an existing position and, when
   enabled, stage that quantity back at the position's original entry price with
   the same price stop, target, and scheduled time stop as the existing position.
2. **Add to position** — add shares/contracts in the current direction and resize
   the existing protective bracket to cover the new total position.

Example: long 100 SMH with closing STP, LMT target, and scheduled MKT time-stop
orders. `Trim 1/2 + Re-add` should sell 50, protect the remaining 50, and stage a
50-share re-entry whose child exits reproduce the existing controls. `Add 1x`
should buy 100 and change the effective bracket coverage from 100 to 200.

## Current architecture

- Site UI: `New_Seasonals/site/assets/execution.js`
- Pages command proxy: `New_Seasonals/functions/exec-command.js`
- Cloudflare relay: `New_Seasonals/execution-broker/src/index.js`
- Execution-machine agent:
  - `OneDrive/trading_ibkr/exec_agent.py`
  - `OneDrive/trading_ibkr/execute_order.py`
  - `OneDrive/trading_ibkr/book_snapshot.py`
  - `OneDrive/trading_ibkr/exec_agent.env`

The Pages proxy and Cloudflare broker already relay arbitrary signed command
types and should not need a functional change. The local execution agent remains
the final validator and the only component that constructs or transmits IBKR
orders.

The existing `flatten` path is useful reference code: it matches the exact
instrument, captures close-direction exits, cancels them through their owning
client IDs, confirms cancellation, submits a fill-gated close, and rebuilds exits
for a partial remainder.

## Important command-contract decision

Use **new command types**, not `flatten` with an optional flag:

- `trim_readd`
- `add_to_position`

This is a rollout safety requirement. If a new site sent `flatten` with
`readd:true` to an old agent, the old agent could ignore the unknown field and
perform only the trim. A distinct command type makes an old agent reject or
dry-run the request instead of partially fulfilling it.

Add both command types to the site's `MUTATING_COMMANDS` set and to the execution
machine's `LIVE_TYPES` only after the agent implementation has been installed and
dry-run tested.

## Proposed payloads

### `trim_readd`

```json
{
  "symbol": "SMH",
  "sec_type": "STK",
  "expiry": null,
  "fraction": 0.5,
  "expected_position": 100,
  "close_order_type": "MKT",
  "readd": true,
  "readd_tif": "DAY"
}
```

- Absolute `qty` may be supported instead of `fraction`; reject quantities above
  the live holding and never silently clamp an explicitly entered quantity.
- `expected_position` is the quantity shown when the user confirmed. Reject if
  the live position has changed.
- Start with a fill-gated MKT trim and a DAY re-add. A resting close plus re-add is
  materially harder to reason about because both can remain working.
- The execution agent should determine the re-entry price from trusted live/local
  state rather than accept it from the browser.

### `add_to_position`

```json
{
  "symbol": "SMH",
  "sec_type": "STK",
  "expiry": null,
  "fraction": 1.0,
  "expected_position": 100,
  "order_type": "MKT"
}
```

- `fraction: 1.0` means add the current position quantity (100 -> 200).
- Useful fast buttons are Add 1/2 and Add 1x. An absolute quantity can live in a
  slower ticket later.
- The agent must independently apply the existing live quantity, notional, and
  account gates.

## Defining “original entry price”

Confirm the source before live rollout. IBKR's live book exposes `avg_cost`,
which is the practical default for a simple position but is not necessarily the
first fill price after prior adds, trims, commissions, or transfers.

Recommended initial behavior:

- Stock positions only.
- Use the live IBKR average cost and label the UI/confirmation **“re-add at Avg”**.
- If the true strategy entry is available from an authoritative ledger or order
  reference, it may replace average cost later, but do not call average cost the
  “original entry” without qualification.

## Capturing and cloning a bracket

Capture only working orders for the exact contract and in the position-closing
direction. Preserve, per leg:

- order type (`STP`, `LMT`, or scheduled `MKT`);
- trigger/limit price;
- total quantity;
- TIF;
- `goodAfterTime` and `goodTillDate`;
- `outsideRth`;
- OCA group topology and OCA type (create fresh OCA group names);
- `orderRef`/strategy tag;
- owning client ID and permanent order ID for cancellation/modification.

Fail before changing anything if a leg type or value cannot be represented
faithfully. Require at least one working protective exit for both new fast
actions: either a readable price-stop leg or a scheduled MKT time-stop leg.

For laddered brackets, do not set every cloned leg to the full position size.
Preserve each OCA rung's allocation ratio. Within one OCA group, all redundant
exit legs should carry the same quantity; the sum of the rung quantities should
equal the position quantity. Reject ambiguous bracket topologies.

## `trim_readd` execution sequence

1. Match exactly one live position by symbol, security type, and expiry/conId.
2. Reject options initially; a symbol-scoped action can tear apart a spread.
3. Verify `expected_position`, partial quantity, live gates, average cost, and a
   cloneable working bracket containing a price stop or scheduled time stop.
4. Capture the closing exit legs before cancelling anything.
5. Cancel captured closing exits through their owning client IDs and confirm
   they are gone. Leave unrelated same-direction entry orders alone only if that
   behavior is explicitly desired; otherwise reject the action while one exists.
6. Re-read the position so a stop fill during cancellation cannot turn the trim
   into a new opposite position.
7. Submit the partial MKT close and require a confirmed fill.
8. Rebuild the captured exits for the actual remaining live quantity.
9. If `readd` is enabled, stage a DAY limit parent at live average cost. Attach
   cloned exit children to that parent so they activate only after the re-add
   fills.
10. Return separate result fields for the trim fill, remaining-position exits,
    and re-add parent/child order IDs.

Failure behavior:

- Close did not fill: restore exits at the full live quantity; do not stage a
  re-add.
- Remaining-position exit rebuild failed: report failure prominently and state
  that the remainder needs manual protection.
- Re-add staging failed after a successful trim: keep the remainder protected,
  return failure/partial completion, and state explicitly that no re-add is
  working.

## `add_to_position` execution sequence

1. Match one exact non-option position and verify `expected_position`.
2. Reject if another same-direction entry/add order is already working.
3. Capture and validate the existing closing bracket before adding.
4. Apply quantity, notional, account, and security-type caps.
5. Submit the same-direction add and require a confirmed fill.
6. Re-read the live position and verify the expected direction and quantity.
7. Resize the captured bracket to the actual new live quantity. Modifying the
   existing legs through their owning client is acceptable; cancel/recreate is
   also acceptable if cancellation is confirmed and the failure path clearly
   reports any unprotected quantity.

There is an unavoidable transition risk between the add fill and a multi-leg
bracket resize. Keep that interval as short as possible and test every failure
branch. A safer but more complex implementation may initially attach cloned
children to the added lot, then consolidate the two brackets. Do not report
success until the resulting protection is verified from the live open-order
book.

## Site UI

In each eligible stock-position row, add compact controls:

- existing `Trim 1/2`;
- a persistent per-row `Re-add off/on` toggle that applies to the trim action;
- `Add 1/2`;
- `Add 1x`.

Disable re-add/add controls when neither a closing STP nor a scheduled closing
MKT time stop is visible. Keep the agent-side check authoritative because the
browser book can be stale.

Confirmations must state the exact account, side, quantity, order style, average
cost used for re-add, expected post-action quantity, and that the bracket will be
cloned/resized. Preserve the existing fail-dangerous execution-mode behavior:
unknown mode disables all mutating controls.

`book_snapshot.py` should add contract expiry/conId fields to working-order rows
if needed so the UI can associate exits with the exact position rather than only
the ticker.

## Agent validation and preview

Update `exec_agent.py` to:

- recognize and describe both new types;
- validate exact position identity, stale expected quantity, supported security
  type, existing stop bracket, working same-direction orders, and caps;
- produce a dry-run preview showing the trim/add and every bracket resize/clone;
- pass live commands only when the new types are explicitly present in
  `LIVE_TYPES`.

Update `execute_order.py` to add the two live handlers and include them in its
independent `SUPPORTED` gate. Both the parent agent and executor must validate
the contract independently.

## Tests

At minimum add:

- JS payload tests for both fast actions and the unknown-mode block;
- stale-position rejection;
- exact stock/future identity and option rejection;
- missing-stop rejection;
- unsupported or ambiguous bracket rejection before cancellation;
- one-group stop/target/time cloning with all fields preserved;
- multi-rung proportional quantity scaling and rounding;
- trim non-fill restores full exits and stages no re-add;
- trim fill protects the remainder before/while staging re-add;
- add non-fill leaves the old bracket untouched;
- add fill resizes all exit legs to the actual live total;
- cancellation/replace failures produce explicit manual-action warnings;
- idempotent retry does not submit a second trim/add.

Run both commands end-to-end in hard dry-run on the execution machine using a
representative three-leg SMH-style bracket before adding them to `LIVE_TYPES`.

## Rollout order

1. Implement and test the execution-machine agent.
2. Restart it in hard dry-run and confirm the broker book includes the required
   position/order identity fields.
3. Exercise both command types through the signed relay in dry-run.
4. Add the new command types to `LIVE_TYPES`, restart, and test with deliberately
   tiny size if desired.
5. Only then deploy the site UI.

Do not deploy the UI first. The distinct command types are intended to make an
old agent reject the actions, but agent-first rollout still keeps the behavior
easy to reason about.
