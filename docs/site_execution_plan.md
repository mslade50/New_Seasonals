# Site → execution platform — design (Phase 1 + Phase 2)

Status: **design for review.** Nothing built yet. Goal: move the morning
order-chain confirmation onto the site (Phase 1), then make the site able to
trade in/out of positions in either account via a local agent (Phase 2).

---

## The one hard constraint

TWS (primary, `127.0.0.1:7496`) and the PA Gateway (`127.0.0.1:4001`) are
**localhost-only on the trading machine** and must never be exposed to the
internet. So the invariant for everything below:

> The local machine always reaches **OUT** to Cloudflare. Nothing inbound ever
> touches the trading box. The local agent is the **final gatekeeper** — it
> independently validates every command and can refuse it.

This is just a generalization of what already exists: `order_staging.py` →
`eq_/pa_order_entry.py` → `morning_order_summary.py` already run locally, read a
plan (Sheets / CSV), talk to the broker, and report. We're adding a Cloudflare
broker + a UI, and turning the batch flow on-demand.

---

## Shared data path (both phases use it)

```
 trading machine (local)            Cloudflare                     browser (you)
 ────────────────────────           ───────────────────           ──────────────
 local agent  ──outbound wss──▶  Durable Object "ExecBroker"  ◀──  site (Access)
   • ib_insync to TWS/GW             • holds the agent WS (hibernation)
   • validates + executes            • command queue + order state
   • heartbeat                       • audit log
   • read-only summaries     ──▶  R2 (morning_orders.json, audit)  ──▶ Pages Fn ─▶ site
```

- **Local → Cloudflare is outbound only** (a WebSocket the agent dials out, plus
  R2 uploads). No firewall holes, no inbound exposure.
- **Site ↔ Cloudflare** through a Worker / Pages Functions, gated by Cloudflare
  Access (already configured: email OTP, allowlist).
- **R2** holds durable read-only artifacts (the morning summary, the audit log).
- A **Durable Object** holds the live agent connection + command queue + state
  (Cloudflare's WebSocket-hibernation DO is built for exactly this).

---

## Phase 1 — morning order summary on the site (READ-ONLY)

`morning_order_summary.py` already builds, per account, the structured result:
each staged order's state (working / filled / cancelled / missing), its exit-leg
confirmation (TARGET/STOP/EOD-DD/PROFIT_TAKER/TIME), risk in $ and bps, the
per-strategy rollup, and the dividend-adjustment banner. It only *renders* that
to HTML email today.

Change:
1. **Local**: serialize that structured summary to JSON and `upload_from_local`
   it to R2 (`morning_orders.json`). Keep the email too at first (belt + braces).
   Additive, low-risk — it doesn't touch the read-only broker logic.
2. **Cloudflare**: a Pages Function streams the R2 JSON with `no-store` — same
   pattern as `functions/chartimg/[[path]].js` streaming chart PNGs. Needs a
   second R2 binding in `wrangler.toml`.
3. **Site**: a new **Orders** tab fetches it live and renders the two account
   tables + banners (port the email's HTML to the site card/table style).

Why a Pages Function and not the build-time bake: the summary is generated
~9:31 AM ET, *after* the AM site deploy, so it must be fetched live (R2), not
baked into `dist/` at build. This live-fetch-from-R2 path is the **read-only
first slice of the Phase 2 pipe** — Phase 1 proves it before any execution.

---

## Phase 2 — execution bridge

### Components
- **Durable Object `ExecBroker`** — holds the local agent's hibernating
  WebSocket, a command queue, per-command state machine, and the append-only
  audit log. One instance per book.
- **Local agent** (new, trading machine) — dials OUT to the DO (`wss`),
  authenticates, receives commands, **validates**, executes via `ib_insync` into
  the right account, reports fills back, emits a heartbeat. Reuses the existing
  `eq_/pa_order_entry` submission code.
- **Site UI** — positions table with trim / exit / add actions → confirm modal →
  `POST` to a Worker (Access-auth'd + signed) → DO enqueues → agent executes →
  status streams back to the table.

### Connection model — outbound WebSocket (chosen)
The agent holds a persistent **outbound** WS to the DO (hibernation keeps it cheap).
Near-instant push, a natural heartbeat ("execution online/offline" on the site),
and zero inbound exposure. Falls back to reconnect-with-backoff if dropped;
commands queue in the DO while the agent is away.

### Security (the part to be paranoid about)
- **Access** gates the human (already in place).
- **Per-command auth**: short-lived **signed token** (HMAC with a secret shared
  only by the Worker and the local agent) + **idempotency key** + expiry. A stale
  browser tab or a replayed request can't fire an order.
- **Local agent is the final gatekeeper** — independent validation on every
  command: symbol allowlist, max size + max notional (per command and per day),
  "the position must actually exist" for a trim/exit, market-hours, sane price.
  Anything off-profile is refused and logged. The site is never trusted.
- **Kill switch**: a flag in the DO the agent checks before every action; flipping
  it halts execution. The agent also has a local hard-disable.
- **Audit log**: every intent → validation → submission → fill, append-only,
  surfaced on the site.
- **Idempotency / dedup**: unique command id; the agent dedups so a retry never
  double-submits.

### Ramp (dry-run → tiny → full)
1. **Dry-run** — agent receives + validates + logs "would submit X", places
   nothing; site shows the would-be order. Proves the full loop end-to-end.
2. **Tiny live** — PA account only, min size (1 share / tiny notional), one
   command type (flat a position). Watch fills + audit for days.
3. **Expand** — more command types (trim %, add), then the main account, then
   lift the size cap.

### Account routing
Command carries the target account (PA / primary); the agent connects to the
right port (4001 / 7496) and submits. The `execution` / `execution_2` split
already models this.

---

## Milestones
1. **P1** — morning summary → site (read-only data path).
2. **P2a** — DO broker + local agent skeleton + outbound WS + heartbeat
   ("execution online" indicator on the site). No commands yet.
3. **P2b** — dry-run command loop (site → DO → agent → "would do" → status back).
4. **P2c** — tiny live (PA, flat-position only, min size).
5. **P2d** — expand command types + main account + lift caps.

## Open questions / risks
- Trading machine must be up + TWS/Gateway logged in; the heartbeat surfaces
  "offline" so the UI never lies about execution availability.
- Reconciliation with the systematic batch flow — manual trims must not fight the
  staged book over the same position.
- Secret management for the HMAC key (local agent + Worker); rotation.
- Keep the morning email as a fallback initially (don't remove it until the site
  view is trusted).
- Cloudflare implementation (DO, WebSocket hibernation, Pages Functions, R2
  bindings) will be built against the `durable-objects` / `agents-sdk` /
  `cloudflare` skills for correctness.
