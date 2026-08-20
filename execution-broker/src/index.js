/* execution-broker — Cloudflare Worker + Durable Object.
 *
 * The cloud broker for the site's execution bridge. A single Durable Object
 * instance ("main") holds the local agent's OUTBOUND, hibernatable WebSocket,
 * tracks a heartbeat, and relays signed commands to the agent + collects results.
 *
 * The broker does not construct or transmit IBKR orders, but it is not a dumb
 * relay: it independently gates envelopes on heartbeat/book freshness, reported
 * mode, and server live type/account allowlists before pushing to the agent.
 *
 * Endpoints (all DO-routed):
 *   GET  /agent     agent WS upgrade            (Bearer AGENT_TOKEN)
 *   GET  /status    heartbeat / online state    (Bearer STATUS_TOKEN)
 *   POST /command   {signed, sig} -> push to agent  (Bearer COMMAND_SECRET)
 *   GET  /commands  recent commands + results   (Bearer STATUS_TOKEN)
 *   GET  /fills     accumulated executions ring (Bearer STATUS_TOKEN)
 *   GET  /health    plain liveness
 *
 * Deploy standalone (NOT part of the Pages site). See README.md.
 */
import { DurableObject } from "cloudflare:workers";
import { HEARTBEAT_STALE_MS, validHmacHex, validateBrokerCommand } from "./command_policy.mjs";

const BROKER_NAME = "main";          // single book -> single DO instance
const CMD_CAP = 50;                  // recent-command ring size (audit trail)
const SCHEDULED_CMD_CAP = 100;       // long-lived option schedules survive recent-ring churn
const FILLS_RETENTION_DAYS = 14;     // Trade Log trailing window
const FILLS_DAY_CAP = 500;           // per-day row cap (keeps each value < DO 128KiB limit)
const COMMAND_BODY_MAX = 32_768;
const PENDING_PREFIX = "pending_command:";

function stableJson(value) {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stableJson(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function sameCommandIntent(left, right) {
  const fields = ["id", "type", "account", "dry_run", "payload", "policy_version"];
  return stableJson(Object.fromEntries(fields.map((key) => [key, left && left[key]])))
    === stableJson(Object.fromEntries(fields.map((key) => [key, right && right[key]])));
}

export class ExecBroker extends DurableObject {
  _authed(request, token) {
    return (request.headers.get("Authorization") || "") === `Bearer ${token}`;
  }

  // Newest socket = most recently accepted or heartbeated (attachment stamps).
  // With >1 socket connected (zombie left by a restart/reconnect), commands go
  // to this one only — never fanned out — so a stale socket can't double-execute.
  _newestSocket(sockets) {
    let best = sockets[0], bestAt = -1;
    for (const s of sockets) {
      let att;
      try { att = s.deserializeAttachment() || {}; } catch { att = {}; }
      const at = Math.max(att.lastSeenAt || 0, att.connectedAt || 0);
      if (at > bestAt) { bestAt = at; best = s; }
    }
    return best;
  }

  _pendingKey(id) {
    return `${PENDING_PREFIX}${id}`;
  }

  async _updateCommandRecord(id, patch) {
    const recent = (await this.ctx.storage.get("recent_commands")) || [];
    const scheduled = (await this.ctx.storage.get("scheduled_commands")) || [];
    let changedRecent = false;
    let changedScheduled = false;
    const ri = recent.findIndex((r) => r.id === id);
    if (ri >= 0) {
      recent[ri] = { ...recent[ri], ...patch };
      changedRecent = true;
    }
    const si = scheduled.findIndex((r) => r.id === id);
    if (si >= 0) {
      scheduled[si] = { ...scheduled[si], ...patch };
      changedScheduled = true;
    }
    if (changedRecent) await this.ctx.storage.put("recent_commands", recent);
    if (changedScheduled) await this.ctx.storage.put("scheduled_commands", scheduled);
    return (ri >= 0 ? recent[ri] : null) || (si >= 0 ? scheduled[si] : null);
  }

  async _schedulePendingExpiry(expiresAt) {
    const expiry = Number(expiresAt || 0);
    if (!Number.isFinite(expiry) || expiry <= 0) return;
    const alarmAt = Math.max(Date.now() + 1_000, expiry + 1_000);
    const existing = await this.ctx.storage.getAlarm();
    if (existing == null || alarmAt < Number(existing)) {
      await this.ctx.storage.setAlarm(alarmAt);
    }
  }

  async _expirePending(now = Date.now()) {
    const rows = await this.ctx.storage.list({ prefix: PENDING_PREFIX });
    let nextExpiry = null;
    for (const [key, pending] of rows) {
      if (!pending || !pending.id) {
        await this.ctx.storage.delete(key);
        continue;
      }
      const expiry = Number(pending.expires_at || 0);
      if (Number.isFinite(expiry) && expiry > now) {
        nextExpiry = nextExpiry == null ? expiry : Math.min(nextExpiry, expiry);
        continue;
      }
      const prior = (await this._updateCommandRecord(pending.id, {})) || {};
      const uncertain = ["sending", "sent", "received"].includes(prior.state);
      await this._updateCommandRecord(pending.id, {
        state: uncertain ? "unknown" : "expired",
        result: { ok: false, detail: uncertain
          ? "delivery expired without a terminal result; verify TWS"
          : "command expired before delivery" },
      });
      await this.ctx.storage.delete(key);
    }
    if (nextExpiry != null) {
      await this.ctx.storage.setAlarm(Math.max(Date.now() + 1_000, nextExpiry + 1_000));
    }
  }

  async _deliverPending(pending, socket, sessionId) {
    const attemptedAt = Date.now();
    // Persist the attempt/session before the non-transactional WebSocket send.
    // If storage fails or the process dies after send, a later session sees a
    // possibly-delivered intent and marks it UNKNOWN instead of duplicating it.
    const started = {
      ...pending, attempts: Number(pending.attempts || 0) + 1,
      last_attempt_at: attemptedAt, last_session_id: sessionId,
      delivery_started_at: attemptedAt, last_delivery_error: null,
    };
    await this.ctx.storage.put(this._pendingKey(pending.id), started);
    await this._updateCommandRecord(pending.id, {
      state: "sending", last_attempt_at: attemptedAt,
      delivery_attempts: started.attempts, last_delivery_error: null,
    });
    try {
      socket.send(JSON.stringify({ type: "command", signed: pending.signed, sig: pending.sig }));
    } catch (error) {
      const detail = String((error && error.message) || error || "socket send failed");
      // A synchronous send exception means no frame was accepted by this
      // socket, so this one case remains safely retryable.
      const retryable = {
        ...started, last_session_id: null, last_delivery_error: detail,
      };
      await this.ctx.storage.put(this._pendingKey(pending.id), retryable);
      await this._updateCommandRecord(pending.id, {
        state: "queued", last_delivery_error: detail, last_attempt_at: attemptedAt,
      });
      return { ok: false, state: "queued", error: "agent delivery failed; command retained for retry" };
    }
    await this._updateCommandRecord(pending.id, {
      state: "sent", sent_at: attemptedAt, last_attempt_at: attemptedAt,
      delivery_attempts: started.attempts, last_delivery_error: null,
    });
    return { ok: true, state: "sent" };
  }

  async _redeliverPending(ws, sessionId, book) {
    const rows = await this.ctx.storage.list({ prefix: PENDING_PREFIX });
    const now = Date.now();
    const lastSeen = Number((await this.ctx.storage.get("last_seen")) || 0);
    for (const [key, pending] of rows) {
      if (!pending || !pending.id || pending.last_session_id === sessionId) continue;
      const prior = (await this._updateCommandRecord(pending.id, {})) || {};
      // A prior session may have delivered the frame even when its result was
      // lost. Never auto-redeliver across that uncertainty boundary.
      if (pending.last_session_id || ["sending", "sent", "received"].includes(prior.state)) {
        await this._updateCommandRecord(pending.id, {
          state: "unknown",
          result: { ok: false, detail: "prior-session delivery has no terminal result; verify TWS" },
        });
        await this.ctx.storage.delete(key);
        continue;
      }
      let cmd;
      try { cmd = JSON.parse(pending.signed); }
      catch {
        await this._updateCommandRecord(pending.id, { state: "rejected", result: { ok: false, detail: "stored command is invalid" } });
        await this.ctx.storage.delete(key);
        continue;
      }
      if (now > Number(cmd.expires_at || 0)) {
        const uncertain = ["sending", "sent", "received"].includes(prior.state);
        await this._updateCommandRecord(pending.id, {
          state: uncertain ? "unknown" : "expired",
          result: { ok: false, detail: uncertain
            ? "delivery expired without a terminal result; verify TWS"
            : "command expired before delivery" },
        });
        await this.ctx.storage.delete(key);
        continue;
      }
      const gate = validateBrokerCommand(cmd, {
        env: this.env, now, lastSeen, book, socketCount: 1, socketSession: sessionId,
      });
      if (!gate.ok) {
        await this._updateCommandRecord(pending.id, {
          state: "delivery_cancelled",
          result: { ok: false, detail: `redelivery rejected: ${gate.error}` },
        });
        await this.ctx.storage.delete(key);
        continue;
      }
      await this._deliverPending(pending, ws, sessionId);
    }
  }

  async fetch(request) {
    const url = new URL(request.url);

    // --- Agent WebSocket (outbound dial from the trading machine) ---
    if (url.pathname === "/agent") {
      if (!this._authed(request, this.env.AGENT_TOKEN)) return new Response("unauthorized", { status: 401 });
      if (request.headers.get("Upgrade") !== "websocket") return new Response("expected websocket upgrade", { status: 426 });
      const [client, server] = Object.values(new WebSocketPair());
      this.ctx.acceptWebSocket(server);                 // hibernatable accept
      const now = Date.now();
      const sessionId = crypto.randomUUID();
      server.serializeAttachment({ connectedAt: now, sessionId });
      await this.ctx.storage.put("connected_at", now);
      await this.ctx.storage.put("last_seen", now);
      // A heartbeat from a newly connected process must never re-authorize the
      // prior process's still-fresh LIVE book. Wait for this socket's first book.
      await this.ctx.storage.delete("book");
      await this.ctx.storage.delete("disconnected_at");
      return new Response(null, { status: 101, webSocket: client });
    }

    // --- Status read (from the site via the Pages proxy) ---
    if (url.pathname === "/status") {
      if (!this._authed(request, this.env.STATUS_TOKEN)) return new Response("unauthorized", { status: 401 });
      const lastSeen = (await this.ctx.storage.get("last_seen")) || 0;
      const connectedAt = (await this.ctx.storage.get("connected_at")) || null;
      const activeSockets = this.ctx.getWebSockets();
      const sockets = activeSockets.length;
      let sessionId = null;
      if (sockets === 1) {
        try { sessionId = (activeSockets[0].deserializeAttachment() || {}).sessionId || null; }
        catch (_) { sessionId = null; }
      }
      const now = Date.now();
      const age = lastSeen ? now - lastSeen : null;
      const online = sockets === 1 && !!sessionId && age != null && age < HEARTBEAT_STALE_MS;
      return Response.json({
        online, sockets, last_seen: lastSeen || null, connected_at: connectedAt,
        session_id: sessionId, heartbeat_age_ms: age,
        stale_after_ms: HEARTBEAT_STALE_MS, server_now: now,
      });
    }

    // --- Command in (from the site via the Pages /exec-command proxy) ---
    if (url.pathname === "/command" && request.method === "POST") {
      // STATUS_TOKEN is read-only. A separate command secret is required for
      // both this bearer check and the agent-verifiable HMAC envelope.
      if (!this._authed(request, this.env.COMMAND_SECRET)) return new Response("unauthorized", { status: 401 });
      let body;
      try {
        const declared = Number(request.headers.get("Content-Length") || 0);
        if (declared > COMMAND_BODY_MAX) return new Response("body too large", { status: 413 });
        const raw = await request.text();
        if (raw.length > COMMAND_BODY_MAX) return new Response("body too large", { status: 413 });
        body = JSON.parse(raw);
      } catch { return new Response("bad json", { status: 400 }); }
      const { signed, sig } = body || {};
      if (!signed || !sig) return Response.json({ ok: false, error: "missing signed/sig" }, { status: 400 });
      if (!(await validHmacHex(this.env.COMMAND_SECRET, signed, sig))) {
        return Response.json({ ok: false, error: "invalid command signature" }, { status: 401 });
      }
      let cmd;
      try { cmd = JSON.parse(signed); } catch { return Response.json({ ok: false, error: "bad signed payload" }, { status: 400 }); }
      const now = Date.now();
      // A connected zombie socket is not sufficient authority to deliver an
      // order. Require the same recent heartbeat and fresh mode-bearing book
      // that the Pages policy checked immediately before signing.
      const sockets = this.ctx.getWebSockets();
      const lastSeen = Number((await this.ctx.storage.get("last_seen")) || 0);
      const book = (await this.ctx.storage.get("book")) || null;
      const deliverySocket = sockets.length ? this._newestSocket(sockets) : null;
      let socketSession = null;
      try { socketSession = deliverySocket && (deliverySocket.deserializeAttachment() || {}).sessionId; }
      catch (_) { socketSession = null; }
      const gate = validateBrokerCommand(cmd, {
        env: this.env, now, lastSeen, book, socketCount: sockets.length, socketSession,
      });
      if (!gate.ok) {
        return Response.json({ ok: false, error: gate.error }, { status: gate.status });
      }
      // Idempotency: an id already in the ring is a resubmit of the same intent
      // (retry after a client-side timeout/error) — do NOT push it to the agent
      // again; return the existing record so the client can display it.
      const recent = (await this.ctx.storage.get("recent_commands")) || [];
      const scheduled = (await this.ctx.storage.get("scheduled_commands")) || [];
      const existing = recent.find((r) => r.id === cmd.id) || scheduled.find((r) => r.id === cmd.id);
      if (existing) {
        const pending = await this.ctx.storage.get(this._pendingKey(cmd.id));
        if (pending && !["done", "executed", "dry_run", "rejected", "cancelled", "expired"].includes(existing.state)) {
          let original;
          try { original = JSON.parse(pending.signed); }
          catch { return Response.json({ ok: false, error: "stored command is invalid" }, { status: 500 }); }
          if (!sameCommandIntent(original, cmd)) {
            return Response.json({ ok: false, error: "command id is already bound to a different intent" }, { status: 409 });
          }
          // Once a socket accepted a send, a retry cannot distinguish a lost
          // result from a lost command. Do not risk a duplicate order.
          if (pending.last_session_id || ["sending", "sent", "received"].includes(existing.state)) {
            return Response.json({
              ok: true, deduped: true, retried: false, id: cmd.id,
              state: existing.state, command: existing,
            });
          }
          // Pages issues a fresh short-lived envelope on a client retry. Keep
          // the durable id/intent but refresh the signed expiry before delivery.
          const refreshed = {
            ...pending, signed, sig, expires_at: cmd.expires_at, refreshed_at: now,
          };
          await this.ctx.storage.put(this._pendingKey(cmd.id), refreshed);
          await this._schedulePendingExpiry(cmd.expires_at);
          const delivery = await this._deliverPending(refreshed, deliverySocket, socketSession);
          return Response.json({
            ok: delivery.ok, deduped: true, retried: true, id: cmd.id,
            state: delivery.state, error: delivery.error, command: existing,
          }, { status: delivery.ok ? 200 : 503 });
        }
        return Response.json({ ok: true, deduped: true, id: cmd.id, state: existing.state, command: existing });
      }
      // Persist an outbox item before sending. It remains until an agent result,
      // so a send error or process/socket loss can be retried without losing the
      // intent; the agent durably deduplicates the command id before execution.
      const record = { id: cmd.id, type: cmd.type, account: cmd.account, dry_run: cmd.dry_run,
                       state: "queued", created_at: Date.now(), result: null };
      if (sockets.length > 1) record.sockets_at_delivery = sockets.length;
      recent.unshift(record);
      await this.ctx.storage.put("recent_commands", recent.slice(0, CMD_CAP));
      if (cmd.type === "scheduled_option") {
        scheduled.unshift({ ...record });
        await this.ctx.storage.put("scheduled_commands", scheduled.slice(0, SCHEDULED_CMD_CAP));
      }
      const pending = {
        id: cmd.id, signed, sig, created_at: Date.now(), expires_at: cmd.expires_at,
        attempts: 0, last_session_id: null,
      };
      await this.ctx.storage.put(this._pendingKey(cmd.id), pending);
      await this._schedulePendingExpiry(cmd.expires_at);
      const delivery = await this._deliverPending(pending, deliverySocket, socketSession);
      return Response.json({ ok: delivery.ok, id: cmd.id, state: delivery.state, error: delivery.error },
        { status: delivery.ok ? 200 : 503 });
    }

    // --- Recent commands + results (site polls this) ---
    if (url.pathname === "/commands") {
      if (!this._authed(request, this.env.STATUS_TOKEN)) return new Response("unauthorized", { status: 401 });
      const recent = (await this.ctx.storage.get("recent_commands")) || [];
      const scheduled = (await this.ctx.storage.get("scheduled_commands")) || [];
      const ids = new Set(recent.map((r) => r.id));
      const commands = recent.concat(scheduled.filter((r) => !ids.has(r.id)))
        .sort((a, b) => Number(b.created_at || 0) - Number(a.created_at || 0));
      return Response.json({ commands, server_now: Date.now() });
    }

    // --- Live book (positions / orders / NLV) the site polls ---
    if (url.pathname === "/book") {
      if (!this._authed(request, this.env.STATUS_TOKEN)) return new Response("unauthorized", { status: 401 });
      const book = (await this.ctx.storage.get("book")) || null;
      return Response.json({ book, server_now: Date.now() });
    }

    // --- Accumulated executions (Trade Log tab). IBKR only serves the current
    //     day's fills, so the ring built by _mergeFills IS the history. ---
    if (url.pathname === "/fills") {
      if (!this._authed(request, this.env.STATUS_TOKEN)) return new Response("unauthorized", { status: 401 });
      const days = await this.ctx.storage.list({ prefix: "fills:" });
      const fills = [];
      for (const v of days.values()) fills.push(...v);
      fills.sort((a, b) => String(b.time || "").localeCompare(String(a.time || "")));
      return Response.json({ fills, retention_days: FILLS_RETENTION_DAYS, server_now: Date.now() });
    }

    // --- Option spread query: POST kicks off a read-only chain fetch on the agent ---
    if (url.pathname === "/option" && request.method === "POST") {
      if (!this._authed(request, this.env.STATUS_TOKEN)) return new Response("unauthorized", { status: 401 });
      let body;
      try { body = await request.json(); } catch { return Response.json({ ok: false, error: "bad json" }, { status: 400 }); }
      const ticker = String((body && body.ticker) || "").toUpperCase().trim();
      if (!ticker) return Response.json({ ok: false, error: "ticker required" }, { status: 400 });
      const sockets = this.ctx.getWebSockets();
      if (!sockets.length) return Response.json({ ok: false, error: "agent offline" }, { status: 503 });
      const id = crypto.randomUUID();
      const expiry = (body && body.expiry) || null;
      await this.ctx.storage.put("option_query", { id, ticker, expiry, at: Date.now(), result: null });
      for (const s of sockets) s.send(JSON.stringify({ type: "option_query", id, ticker, expiry }));
      return Response.json({ ok: true, id, ticker });
    }
    if (url.pathname === "/option") {
      if (!this._authed(request, this.env.STATUS_TOKEN)) return new Response("unauthorized", { status: 401 });
      return Response.json({ query: (await this.ctx.storage.get("option_query")) || null, server_now: Date.now() });
    }

    // --- Workbench query: term structure + chain band for the options workbench.
    //     Small ring (not the /option single slot): expiry-change re-queries overlap
    //     the prior poll, so each query keeps its own entry addressed by id. ---
    if (url.pathname === "/workbench" && request.method === "POST") {
      if (!this._authed(request, this.env.STATUS_TOKEN)) return new Response("unauthorized", { status: 401 });
      let body;
      try { body = await request.json(); } catch { return Response.json({ ok: false, error: "bad json" }, { status: 400 }); }
      const ticker = String((body && body.ticker) || "").toUpperCase().trim();
      if (!ticker) return Response.json({ ok: false, error: "ticker required" }, { status: 400 });
      const sockets = this.ctx.getWebSockets();
      if (!sockets.length) return Response.json({ ok: false, error: "agent offline" }, { status: 503 });
      const id = crypto.randomUUID();
      const q = { id, ticker, mode: body.mode || "full", expiry: body.expiry || null,
        max_expiries: body.max_expiries || null, context: body.context || null,
        at: Date.now(), result: null };
      const ring = (await this.ctx.storage.get("workbench_queries")) || [];
      ring.unshift(q);
      await this.ctx.storage.put("workbench_queries", ring.slice(0, 8));
      this._newestSocket(sockets).send(JSON.stringify({ type: "workbench_query", ...q }));
      return Response.json({ ok: true, id, ticker });
    }
    if (url.pathname === "/workbench") {
      if (!this._authed(request, this.env.STATUS_TOKEN)) return new Response("unauthorized", { status: 401 });
      const ring = (await this.ctx.storage.get("workbench_queries")) || [];
      const id = url.searchParams.get("id");
      const query = id ? ring.find((r) => r.id === id) || null : ring[0] || null;
      return Response.json({ query, server_now: Date.now() });
    }

    // --- Futures sizing query: POST kicks off a pure read-only sizing calc on the agent ---
    if (url.pathname === "/futures_size" && request.method === "POST") {
      if (!this._authed(request, this.env.STATUS_TOKEN)) return new Response("unauthorized", { status: 401 });
      let body;
      try { body = await request.json(); } catch { return Response.json({ ok: false, error: "bad json" }, { status: 400 }); }
      const symbol = String((body && body.symbol) || "").toUpperCase().trim();
      if (!symbol) return Response.json({ ok: false, error: "symbol required" }, { status: 400 });
      const sockets = this.ctx.getWebSockets();
      if (!sockets.length) return Response.json({ ok: false, error: "agent offline" }, { status: 503 });
      const id = crypto.randomUUID();
      const q = { id, symbol, entry: body.entry ?? null, stop: body.stop ?? null,
        target: body.target ?? null, risk: body.risk ?? null, risk_pct: body.risk_pct ?? null,
        account_key: body.account_key || "primary", account_value: body.account_value ?? null,
        at: Date.now(), result: null };
      await this.ctx.storage.put("futures_size", q);
      for (const s of sockets) s.send(JSON.stringify({ type: "futures_size", ...q }));
      return Response.json({ ok: true, id, symbol });
    }
    if (url.pathname === "/futures_size") {
      if (!this._authed(request, this.env.STATUS_TOKEN)) return new Response("unauthorized", { status: 401 });
      return Response.json({ query: (await this.ctx.storage.get("futures_size")) || null, server_now: Date.now() });
    }

    // --- Futures front-month resolve: POST kicks off a read-only reqContractDetails on the agent ---
    if (url.pathname === "/futures_front" && request.method === "POST") {
      if (!this._authed(request, this.env.STATUS_TOKEN)) return new Response("unauthorized", { status: 401 });
      let body;
      try { body = await request.json(); } catch { return Response.json({ ok: false, error: "bad json" }, { status: 400 }); }
      const symbol = String((body && body.symbol) || "").toUpperCase().trim();
      if (!symbol) return Response.json({ ok: false, error: "symbol required" }, { status: 400 });
      const exchange = String((body && body.exchange) || "").toUpperCase().trim();
      if (exchange && !["CME", "CBOT", "NYMEX", "COMEX"].includes(exchange)) {
        return Response.json({ ok: false, error: "exchange must be CME, CBOT, NYMEX, or COMEX" }, { status: 400 });
      }
      const sockets = this.ctx.getWebSockets();
      if (!sockets.length) return Response.json({ ok: false, error: "agent offline" }, { status: 503 });
      const id = crypto.randomUUID();
      await this.ctx.storage.put("futures_front", { id, symbol, exchange: exchange || null, at: Date.now(), result: null });
      for (const s of sockets) s.send(JSON.stringify({ type: "futures_front", id, symbol, exchange: exchange || null }));
      return Response.json({ ok: true, id, symbol, exchange: exchange || null });
    }
    if (url.pathname === "/futures_front") {
      if (!this._authed(request, this.env.STATUS_TOKEN)) return new Response("unauthorized", { status: 401 });
      return Response.json({ query: (await this.ctx.storage.get("futures_front")) || null, server_now: Date.now() });
    }

    return new Response("not found", { status: 404 });
  }

  async webSocketMessage(ws, message) {
    let msg;
    try { msg = JSON.parse(typeof message === "string" ? message : ""); }
    catch { msg = { type: "raw" }; }

    if (msg.type === "hello" || msg.type === "heartbeat") {
      await this.ctx.storage.put("last_seen", Date.now());
      // stamp the socket so _newestSocket can prefer the live one over a zombie
      try { ws.serializeAttachment({ ...(ws.deserializeAttachment() || {}), lastSeenAt: Date.now() }); }
      catch (_) { /* best effort — connectedAt still breaks the tie */ }
      ws.send(JSON.stringify({ type: "ack", of: msg.type, server_now: Date.now() }));
      return;
    }

    // Live read-only book snapshot from the agent (positions / orders / NLV
    // + today's fills). Fills are folded into their own per-day ring and
    // stripped from the stored book (keeps the "book" value small).
    if (msg.type === "book") {
      let sessionId = null;
      try { sessionId = (ws.deserializeAttachment() || {}).sessionId; }
      catch (_) { sessionId = null; }
      if (!sessionId) {
        await this.ctx.storage.put("last_error", "book rejected: socket session missing; reconnect required");
        try { ws.close(1012, "reconnect required"); } catch (_) { /* best effort */ }
        return;
      }
      const book = { ...(msg.book || {}), at: msg.at || Date.now() };
      const accounts = (book.accounts || []).map(({ fills, ...rest }) => rest);
      await this.ctx.storage.put("book", { ...book, accounts, _broker_session_id: sessionId });
      await this.ctx.storage.put("last_seen", Date.now());
      try { await this._mergeFills(book); }
      catch (e) { await this.ctx.storage.put("last_error", `mergeFills: ${String((e && e.message) || e)}`); }
      // A reconnect gets a new session id. Only after its first fresh book is
      // stored may unresolved outbox commands be revalidated and redelivered.
      try { await this._redeliverPending(ws, sessionId, { ...book, accounts, _broker_session_id: sessionId }); }
      catch (e) { await this.ctx.storage.put("last_error", `redeliverPending: ${String((e && e.message) || e)}`); }
      return;
    }

    // Option-spread result from the agent -> attach to the pending query.
    if (msg.type === "option_result" && msg.id) {
      const q = await this.ctx.storage.get("option_query");
      if (q && q.id === msg.id) {
        q.result = msg.data; q.result_at = Date.now();
        await this.ctx.storage.put("option_query", q);
      }
      return;
    }

    // Workbench result from the agent -> attach to its ring entry by id.
    if (msg.type === "workbench_result" && msg.id) {
      const ring = (await this.ctx.storage.get("workbench_queries")) || [];
      const i = ring.findIndex((r) => r.id === msg.id);
      if (i >= 0) {
        ring[i].result = msg.data; ring[i].result_at = Date.now();
        await this.ctx.storage.put("workbench_queries", ring);
      }
      return;
    }

    // Futures-sizing result from the agent -> attach to the pending query.
    if (msg.type === "futures_result" && msg.id) {
      const q = await this.ctx.storage.get("futures_size");
      if (q && q.id === msg.id) {
        q.result = msg.data; q.result_at = Date.now();
        await this.ctx.storage.put("futures_size", q);
      }
      return;
    }

    // Futures front-month result from the agent -> attach to the pending query.
    if (msg.type === "futures_front_result" && msg.id) {
      const q = await this.ctx.storage.get("futures_front");
      if (q && q.id === msg.id) {
        q.result = msg.data; q.result_at = Date.now();
        await this.ctx.storage.put("futures_front", q);
      }
      return;
    }

    // Durable receipt means the agent journaled the id before any execution.
    // Keep the outbox item until a terminal result so an interrupted session
    // can replay the durable result (or surface UNKNOWN) on reconnect.
    if (msg.type === "command_receipt" && msg.id) {
      await this._updateCommandRecord(msg.id, {
        state: "received", received_at: Date.now(), agent_policy_version: msg.policy_version || null,
      });
      return;
    }

    // Command result from the agent -> attach to the recent-commands ring.
    if (msg.type === "result" && msg.id) {
      const recent = (await this.ctx.storage.get("recent_commands")) || [];
      const i = recent.findIndex((r) => r.id === msg.id);
      if (i >= 0) {
        recent[i].state = msg.state || "done";
        recent[i].result = { ok: msg.ok, detail: msg.detail, validation: msg.validation,
                             preview: msg.preview, fill: msg.fill, at: msg.at };
        await this.ctx.storage.put("recent_commands", recent);
      }
      const scheduled = (await this.ctx.storage.get("scheduled_commands")) || [];
      const si = scheduled.findIndex((r) => r.id === msg.id);
      if (si >= 0) {
        scheduled[si].state = msg.state || "done";
        scheduled[si].result = { ok: msg.ok, detail: msg.detail, validation: msg.validation,
                                 preview: msg.preview, fill: msg.fill, at: msg.at };
        await this.ctx.storage.put("scheduled_commands", scheduled);
      }
      await this.ctx.storage.delete(this._pendingKey(msg.id));
    }
  }

  // A lost receipt/result must not leave the UI claiming SENT forever.  The
  // alarm never retries a possibly executed command; it only converts an
  // expired sent/received intent to UNKNOWN so a human verifies TWS.
  async alarm() {
    await this._expirePending();
  }

  // Fold a book push's per-account fills into per-day storage keys
  // ("fills:YYYY-MM-DD", UTC day of the fill time). Upsert by exec_id — the
  // agent re-pushes the same day's fills every cycle, and commission reports
  // lag the execution by a beat, so later pushes fill in commission/PnL.
  // Day keys older than the retention window are pruned on every merge.
  async _mergeFills(book) {
    const incoming = [];
    for (const acc of (book && book.accounts) || []) {
      for (const f of acc.fills || []) {
        if (f && f.exec_id) incoming.push({ ...f, account_key: acc.key, account_label: acc.label });
      }
    }
    if (!incoming.length) return;
    const now = Date.now();
    const byDay = new Map();
    for (const f of incoming) {
      const t = Date.parse(f.time);
      const day = new Date(Number.isFinite(t) ? t : now).toISOString().slice(0, 10);
      if (!byDay.has(day)) byDay.set(day, []);
      byDay.get(day).push(f);
    }
    for (const [day, dayFills] of byDay) {
      const key = `fills:${day}`;
      const ring = (await this.ctx.storage.get(key)) || [];
      const byId = new Map(ring.map((f) => [f.exec_id, f]));
      for (const f of dayFills) {
        const prev = byId.get(f.exec_id);
        byId.set(f.exec_id, { ...(prev || {}), ...f, ingested_at: prev ? prev.ingested_at : now });
      }
      await this.ctx.storage.put(key, [...byId.values()].slice(0, FILLS_DAY_CAP));
    }
    const cutoffDay = new Date(now - FILLS_RETENTION_DAYS * 86_400_000).toISOString().slice(0, 10);
    const days = await this.ctx.storage.list({ prefix: "fills:" });
    for (const key of days.keys()) {
      if (key.slice("fills:".length) < cutoffDay) await this.ctx.storage.delete(key);
    }
  }

  async webSocketClose(ws, code, reason, wasClean) {
    await this.ctx.storage.put("disconnected_at", Date.now());
    try {
      const sessionId = (ws.deserializeAttachment() || {}).sessionId;
      const book = await this.ctx.storage.get("book");
      if (sessionId && book && book._broker_session_id === sessionId) {
        await this.ctx.storage.delete("book");
      }
    } catch (_) { /* fail closed via heartbeat/session checks */ }
    try { ws.close(code, reason); } catch (_) { /* already closing */ }
  }

  async webSocketError(ws, err) {
    await this.ctx.storage.put("last_error", String((err && err.message) || err));
  }
}

const DO_PATHS = new Set(["/agent", "/status", "/command", "/commands", "/book", "/fills", "/option", "/workbench", "/futures_size", "/futures_front"]);

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/" || url.pathname === "/health") {
      return new Response("execution-broker ok\n", { status: 200 });
    }
    if (DO_PATHS.has(url.pathname)) {
      const id = env.EXEC_BROKER.idFromName(BROKER_NAME);
      return env.EXEC_BROKER.get(id).fetch(request);
    }
    return new Response("not found", { status: 404 });
  },
};
