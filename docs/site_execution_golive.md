# Execution bridge — go-live runbook (Phase 2c)

Status: **NOT armed.** The platform is built, verified, and running in **hard
dry-run**: the agent validates and previews every command but contains **no order
transmission path** — there is nothing to "flip on" by accident. Going live is the
deliberate, watched procedure below. It is intentionally a human step.

Why it isn't automated: live transmission sends **real-money orders to your live
IBKR accounts**, and it cannot be verified without a real fill. So the only safe
way to turn it on is together, watching the first order, smallest size first —
the dry-run → tiny → full ramp.

---

## What's already done (the dry-run platform)
- Agent holds an outbound WebSocket to the broker (5 AM–9 PM ET, scheduled task).
- Read-only IBKR book (positions / orders / NLV) streamed to the **Positions /
  Open Orders** panels.
- Gatekeeper **validation** (position-must-exist, price ordering, notional + risk
  caps) and a full **order-chain preview** for every command.
- Idempotency, signed commands, behind Cloudflare Access.
- A green **DRY-RUN MODE** banner is shown whenever `book.mode !== "live"`.

The only thing missing is the order-transmission path — deliberately.

---

## The live-execution design (to be built together, during a watched session)
Add a **gated** transmit path to the agent, off by default, defence-in-depth:

1. **`AGENT_LIVE_ENABLED`** env (default unset → dry-run). Nothing transmits
   unless this is exactly `1`.
2. **Ramp caps** (start tight, widen later):
   - `LIVE_ACCOUNTS = {"pa"}`  — PA (small account) only at first.
   - `LIVE_TYPES = {"flatten"}` — closing an existing position only at first.
   - `LIVE_MAX_QTY` / `LIVE_MAX_NOTIONAL` — e.g. 1 share / a few hundred $.
3. **Dedicated `COMMAND_SECRET`** for signing (replaces the reused STATUS_TOKEN on
   the live path). The agent verifies with `COMMAND_SECRET` when set, else
   STATUS_TOKEN; the Pages `/exec-command` signs with the same.
4. **`execute_order.py`** — a separate subprocess (isolated like `book_snapshot.py`)
   that re-checks every gate independently, connects **non-readonly** only at the
   moment of transmit, places the order (reusing `sznl_entry` / `sznl_exit`
   construction — already proven by the preview), and returns the order/fill ids.
5. Agent `_handle_command`: after validation passes, if `LIVE_ENABLED` **and** all
   ramp gates pass → `execute_order.py` → `state="executed"` with the fill; else →
   dry-run (current behaviour). The book's `mode` flips to `"live"` → the UI banner
   turns amber.

This path is **unverified until a real fill** — building it is step 2 below, and
the first watched PA fill (step 4) is its test.

---

## Go-live steps (each is yours to authorize)
1. **Secret.** Generate `COMMAND_SECRET`; set on the Worker, the Pages project, and
   `exec_agent.env`. (Same mechanism as AGENT_TOKEN/STATUS_TOKEN.)
2. **Build the gated path** (the code above) — together, in a session, reviewed.
3. **Arm tiny.** On the trading box set `AGENT_LIVE_ENABLED=1` with
   `LIVE_ACCOUNTS=pa`, `LIVE_TYPES=flatten`, `LIVE_MAX_QTY=1`; restart the agent.
   Banner turns amber.
4. **First watched fill.** With a tiny PA position open, click **Flatten** on one
   small position. Watch the order appear and fill in **TWS** and in the Activity
   log. Confirm the fill matches the preview.
5. **Verify + ramp.** Once a few tiny fills are clean: widen `LIVE_TYPES`
   (entry_bracket, cancel), then add the primary account, then lift the caps.

## Kill switch / rollback
Set `AGENT_LIVE_ENABLED=0` (or blank) and restart the agent — instantly back to
hard dry-run. Stopping the `ExecAgent` task drops the socket entirely (the UI shows
offline). The broker and agent never store an order intent that survives a restart.

---

When you're ready to start, say so and we'll do step 1–2 together, then you arm and
watch the first fill. Until then the platform stays exactly where it is: complete,
validated, previewing everything, transmitting nothing.
