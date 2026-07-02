/* execution-broker — Cloudflare Worker + Durable Object.
 *
 * The cloud broker for the site's execution bridge. A single Durable Object
 * instance ("main") holds the local agent's OUTBOUND, hibernatable WebSocket,
 * tracks a heartbeat, and relays signed commands to the agent + collects results.
 *
 * Phase 2b adds the command loop — but the broker is still a DUMB RELAY: it does
 * not build, validate, or transmit orders. The site signs a command, the broker
 * pushes it down the open socket, and the LOCAL AGENT verifies the signature,
 * validates, and (in dry-run) only logs what it WOULD do. No order ever originates
 * here.
 *
 * Endpoints (all DO-routed):
 *   GET  /agent     agent WS upgrade            (Bearer AGENT_TOKEN)
 *   GET  /status    heartbeat / online state    (Bearer STATUS_TOKEN)
 *   POST /command   {signed, sig} -> push to agent  (Bearer STATUS_TOKEN)
 *   GET  /commands  recent commands + results   (Bearer STATUS_TOKEN)
 *   GET  /health    plain liveness
 *
 * Deploy standalone (NOT part of the Pages site). See README.md.
 */
import { DurableObject } from "cloudflare:workers";

const BROKER_NAME = "main";          // single book -> single DO instance
const HEARTBEAT_STALE_MS = 30_000;   // online iff a heartbeat landed within this
const CMD_CAP = 50;                  // recent-command ring size (audit trail)

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

  async fetch(request) {
    const url = new URL(request.url);

    // --- Agent WebSocket (outbound dial from the trading machine) ---
    if (url.pathname === "/agent") {
      if (!this._authed(request, this.env.AGENT_TOKEN)) return new Response("unauthorized", { status: 401 });
      if (request.headers.get("Upgrade") !== "websocket") return new Response("expected websocket upgrade", { status: 426 });
      const [client, server] = Object.values(new WebSocketPair());
      this.ctx.acceptWebSocket(server);                 // hibernatable accept
      const now = Date.now();
      server.serializeAttachment({ connectedAt: now });
      await this.ctx.storage.put("connected_at", now);
      await this.ctx.storage.put("last_seen", now);
      await this.ctx.storage.delete("disconnected_at");
      return new Response(null, { status: 101, webSocket: client });
    }

    // --- Status read (from the site via the Pages proxy) ---
    if (url.pathname === "/status") {
      if (!this._authed(request, this.env.STATUS_TOKEN)) return new Response("unauthorized", { status: 401 });
      const lastSeen = (await this.ctx.storage.get("last_seen")) || 0;
      const connectedAt = (await this.ctx.storage.get("connected_at")) || null;
      const sockets = this.ctx.getWebSockets().length;
      const now = Date.now();
      const age = lastSeen ? now - lastSeen : null;
      const online = sockets > 0 && age != null && age < HEARTBEAT_STALE_MS;
      return Response.json({
        online, sockets, last_seen: lastSeen || null, connected_at: connectedAt,
        heartbeat_age_ms: age, stale_after_ms: HEARTBEAT_STALE_MS, server_now: now,
      });
    }

    // --- Command in (from the site via the Pages /exec-command proxy) ---
    if (url.pathname === "/command" && request.method === "POST") {
      if (!this._authed(request, this.env.STATUS_TOKEN)) return new Response("unauthorized", { status: 401 });
      let body;
      try { body = await request.json(); } catch { return new Response("bad json", { status: 400 }); }
      const { signed, sig } = body || {};
      if (!signed || !sig) return Response.json({ ok: false, error: "missing signed/sig" }, { status: 400 });
      let cmd;
      try { cmd = JSON.parse(signed); } catch { return Response.json({ ok: false, error: "bad signed payload" }, { status: 400 }); }
      // Idempotency: an id already in the ring is a resubmit of the same intent
      // (retry after a client-side timeout/error) — do NOT push it to the agent
      // again; return the existing record so the client can display it.
      const recent = (await this.ctx.storage.get("recent_commands")) || [];
      const existing = recent.find((r) => r.id === cmd.id);
      if (existing) {
        return Response.json({ ok: true, deduped: true, id: cmd.id, state: existing.state, command: existing });
      }
      const sockets = this.ctx.getWebSockets();
      if (!sockets.length) return Response.json({ ok: false, error: "agent offline" }, { status: 503 });
      // record + push to the NEWEST agent socket only (it verifies the sig and
      // validates); >1 connected socket is an anomaly worth keeping in the audit trail
      const record = { id: cmd.id, type: cmd.type, account: cmd.account, dry_run: cmd.dry_run !== false,
                       state: "pushed", created_at: Date.now(), result: null };
      if (sockets.length > 1) record.sockets_at_delivery = sockets.length;
      recent.unshift(record);
      await this.ctx.storage.put("recent_commands", recent.slice(0, CMD_CAP));
      this._newestSocket(sockets).send(JSON.stringify({ type: "command", signed, sig }));
      return Response.json({ ok: true, id: cmd.id, state: "pushed" });
    }

    // --- Recent commands + results (site polls this) ---
    if (url.pathname === "/commands") {
      if (!this._authed(request, this.env.STATUS_TOKEN)) return new Response("unauthorized", { status: 401 });
      const recent = (await this.ctx.storage.get("recent_commands")) || [];
      return Response.json({ commands: recent, server_now: Date.now() });
    }

    // --- Live book (positions / orders / NLV) the site polls ---
    if (url.pathname === "/book") {
      if (!this._authed(request, this.env.STATUS_TOKEN)) return new Response("unauthorized", { status: 401 });
      const book = (await this.ctx.storage.get("book")) || null;
      return Response.json({ book, server_now: Date.now() });
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
      const sockets = this.ctx.getWebSockets();
      if (!sockets.length) return Response.json({ ok: false, error: "agent offline" }, { status: 503 });
      const id = crypto.randomUUID();
      await this.ctx.storage.put("futures_front", { id, symbol, at: Date.now(), result: null });
      for (const s of sockets) s.send(JSON.stringify({ type: "futures_front", id, symbol }));
      return Response.json({ ok: true, id, symbol });
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

    // Live read-only book snapshot from the agent (positions / orders / NLV).
    if (msg.type === "book") {
      await this.ctx.storage.put("book", { ...(msg.book || {}), at: msg.at || Date.now() });
      await this.ctx.storage.put("last_seen", Date.now());
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
    }
  }

  async webSocketClose(ws, code, reason, wasClean) {
    await this.ctx.storage.put("disconnected_at", Date.now());
    try { ws.close(code, reason); } catch (_) { /* already closing */ }
  }

  async webSocketError(ws, err) {
    await this.ctx.storage.put("last_error", String((err && err.message) || err));
  }
}

const DO_PATHS = new Set(["/agent", "/status", "/command", "/commands", "/book", "/option", "/futures_size", "/futures_front"]);

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
