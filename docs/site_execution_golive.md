# Execution bridge — go-live runbook (Phase 2c)

Status: **migration fail-closed; not ready to arm for risk-increasing orders.**
The local agent contains an IBKR transmission path, but it must be upgraded to
the `COMMAND_SECRET`/policy-version contract and the broker still needs an
atomic aggregate-risk reservation before entries, adds, trim/re-adds, or option
orders can be enabled. Both server layers reject those command types today.
A stale browser book is never evidence of the current mode.

Why it isn't automated: live transmission sends **real-money orders to your live
IBKR accounts**, and it cannot be verified without a real fill. So the only safe
way to turn it on is together, watching the first order, smallest size first —
the dry-run → tiny → full ramp.

---

## What is implemented
- Agent holds an outbound WebSocket to the broker (5 AM–9 PM ET, scheduled task).
- Read-only IBKR book (positions / orders / NLV) streamed to the **Positions /
  Open Orders** panels.
- Browser validation and order-chain previews.
- Cloudflare Access JWT verification at the Pages boundary.
- A dedicated `COMMAND_SECRET`; `STATUS_TOKEN` is read-only and cannot authorize
  `/command`.
- Strict Pages schemas/allowlists, fresh-book and heartbeat checks, server-side
  risk/notional caps, and independent live type/account gates.
- A hard two-layer block on every risk-increasing live command until aggregate
  pending/working/open risk can be reserved atomically at the broker.
- A second broker-side gate requiring a fresh heartbeat, fresh mode-bearing book,
  a book bound to the sole active agent WebSocket session, and its own live
  switch/type/account allowlists. Reconnects clear the prior book.
- The local agent's validation, live allowlists/caps, and IBKR transmit subprocess.
- The UI reports **LIVE**, **DRY-RUN**, or **UNKNOWN**. Both LIVE and DRY-RUN are
  trusted only from a fresh book while the agent is online; UNKNOWN disables
  mutating controls.

---

## Live-execution gates

All three layers must agree before a command can transmit:

1. **Pages:** `EXEC_LIVE_ENABLED=1`, and the type/account must appear in
   `EXEC_LIVE_TYPES` / `EXEC_LIVE_ACCOUNTS`. New-risk commands are currently
   rejected regardless of arming; their per-command validation also stays below
   `EXEC_MAX_NEW_RISK_BPS` (default 500 bps) and
   `EXEC_MAX_NEW_NOTIONAL_PCT` (default 200% for non-futures entries). Live
   brackets default to `EXEC_LIVE_INSTRUMENTS=STK`; futures or FX must be
   deliberately added at both server layers as well as allowed by the agent.
   Live brackets also default to `EXEC_LIVE_ENTRY_TYPES=LMT,STP_LMT`; market,
   open, or close entries require deliberate arming at both server layers.
   Add/trim-readd actions separately default to
   `EXEC_LIVE_POSITION_INSTRUMENTS=STK`; live adds require a fresh matching
   price stop and are re-capped from current NLV, price, and stop.
   `close_only`, `cancel`, and `modify` are also rejected in live mode because
   an unchanged, cancelled, or downsized protective order can uncover or later
   reverse exposure. They remain preview-only until fresh-book order-role
   validation proves the complete action is risk-reducing.
2. **Broker Worker:** the same `EXEC_LIVE_ENABLED`, `EXEC_LIVE_TYPES`, and
   `EXEC_LIVE_ACCOUNTS` checks run independently inside the Durable Object.
3. **Local agent:** `AGENT_LIVE_ENABLED`/`LIVE_ENABLED`, `LIVE_TYPES`,
   `LIVE_ACCOUNTS`, and its quantity/notional/risk gates must also allow the
   command. The local agent must verify the same `COMMAND_SECRET`; there is no
   `STATUS_TOKEN` signing fallback on the Pages/broker path.

`dry_run` omitted or `true` is always a preview at the Pages gate. Live requires
an explicit `dry_run:false` minted by a browser that currently sees a fresh LIVE
book. The server then re-checks freshness and mode rather than trusting the client.

---

## Go-live steps (each is yours to authorize)
1. **Secret and agent contract.** Generate one strong `COMMAND_SECRET`; store it
   in GitHub Actions, the Worker, Pages, and `exec_agent.env`. Update/restart the
   external agent so its HMAC verifier uses it. Do not arm anything yet.
2. **Deploy fail-closed.** Run `Deploy Execution Broker` from `main`; versioned
   Worker vars atomically deploy with `EXEC_LIVE_ENABLED=0` and empty type/account
   allowlists, then the workflow wires and disarms Pages. Deploy the private site
   from `main` through its cloud-only workflow. Confirm the intended SHAs.
3. **Dry-run contract test.** With both server switches off, send an `echo` and a
   representative preview. Confirm the browser, Pages response, broker activity,
   and agent preview agree, and that stale/offline tests are blocked.
4. **Arm tiny at all three layers.** Start with the PA account and a risk-reducing
   type such as `flatten`; apply the smallest agent quantity/notional caps. Set
   matching Pages and Worker `EXEC_LIVE_TYPES=flatten`,
   `EXEC_LIVE_ACCOUNTS=pa`, then set `EXEC_LIVE_ENABLED=1` at each only during the
   watched session. Arm the agent last and confirm a fresh amber LIVE banner.
5. **First watched fill.** With a tiny PA position open, click **Flatten** on one
   small position. Watch the order appear and fill in **TWS** and in the Activity
   log. Confirm the fill matches the preview.
6. **Do not enable new risk yet.** Entry brackets, position adds/trim-readds,
   scheduled options, and option spreads remain blocked until an atomic broker
   reservation counts pending commands, working orders, and open risk. Only
   after that control and an agent-acknowledged delivery protocol are reviewed
   should the runbook gain a new-risk ramp.

## Kill switch / rollback
Set `EXEC_LIVE_ENABLED=0` on **either** Pages or the broker to stop new server-side
live delivery. Also set `AGENT_LIVE_ENABLED=0`/`LIVE_ENABLED=0` and restart the
agent. For an immediate connectivity stop, stop the ExecAgent task; the socket
drops and fresh-mode checks block the UI and both server gates.

---

Never describe the platform as transmitting nothing unless the current Pages,
broker, agent, and fresh book have all been checked. The code is live-capable.
