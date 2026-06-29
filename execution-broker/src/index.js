/* execution-broker — Cloudflare Worker + Durable Object (Phase 2a).
 *
 * The cloud broker for the site's execution bridge. A single Durable Object
 * instance ("main") holds the local agent's OUTBOUND, hibernatable WebSocket and
 * tracks a heartbeat. The site reads /status (through a Pages proxy) to show an
 * "execution online" light. The agent dials /agent (wss) and sends hello +
 * periodic heartbeats.
 *
 * Phase 2a is the pipe-prover: NO commands, NO order logic. The DO only records
 * liveness and acks heartbeats. Command dispatch (behind a dry-run gate) is 2b.
 *
 * Deploy standalone (NOT part of the Pages site) so a broker change can never
 * break the live site. See README.md.
 */
import { DurableObject } from "cloudflare:workers";

const BROKER_NAME = "main";          // single book -> single DO instance
const HEARTBEAT_STALE_MS = 30_000;   // online iff a heartbeat landed within this

export class ExecBroker extends DurableObject {
  async fetch(request) {
    const url = new URL(request.url);

    // --- Agent WebSocket (outbound dial from the trading machine) ---
    if (url.pathname === "/agent") {
      if ((request.headers.get("Authorization") || "") !== `Bearer ${this.env.AGENT_TOKEN}`) {
        return new Response("unauthorized", { status: 401 });
      }
      if (request.headers.get("Upgrade") !== "websocket") {
        return new Response("expected websocket upgrade", { status: 426 });
      }
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
      if ((request.headers.get("Authorization") || "") !== `Bearer ${this.env.STATUS_TOKEN}`) {
        return new Response("unauthorized", { status: 401 });
      }
      const lastSeen = (await this.ctx.storage.get("last_seen")) || 0;
      const connectedAt = (await this.ctx.storage.get("connected_at")) || null;
      const sockets = this.ctx.getWebSockets().length;
      const now = Date.now();
      const age = lastSeen ? now - lastSeen : null;
      const online = sockets > 0 && age != null && age < HEARTBEAT_STALE_MS;
      return Response.json({
        online, sockets,
        last_seen: lastSeen || null, connected_at: connectedAt,
        heartbeat_age_ms: age, stale_after_ms: HEARTBEAT_STALE_MS, server_now: now,
      });
    }

    return new Response("not found", { status: 404 });
  }

  async webSocketMessage(ws, message) {
    let msg;
    try { msg = JSON.parse(typeof message === "string" ? message : ""); }
    catch { msg = { type: "raw" }; }
    // Phase 2a: only hello / heartbeat. Record liveness + ack. (2b dispatches here.)
    if (msg.type === "hello" || msg.type === "heartbeat") {
      await this.ctx.storage.put("last_seen", Date.now());
      ws.send(JSON.stringify({ type: "ack", of: msg.type, server_now: Date.now() }));
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

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/" || url.pathname === "/health") {
      return new Response("execution-broker ok\n", { status: 200 });
    }
    if (url.pathname === "/agent" || url.pathname === "/status") {
      const id = env.EXEC_BROKER.idFromName(BROKER_NAME);
      return env.EXEC_BROKER.get(id).fetch(request);
    }
    return new Response("not found", { status: 404 });
  },
};
