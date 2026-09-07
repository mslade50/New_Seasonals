/* Pages Function — accept a command from the site, sign it, forward to the broker.
 *
 * Route: POST /exec-command. Behind Cloudflare Access (human auth), plus an
 * in-code Access JWT check (_access.js) so a misconfigured Access wall doesn't
 * leave this endpoint open. Wraps the site's request into a command envelope,
 * HMAC-signs it with STATUS_TOKEN (shared with the local agent, which verifies
 * + is the final gatekeeper), and POSTs {signed, sig} to the broker. The broker
 * relays it down the agent's socket.
 *
 * The caller must supply an explicit dry_run boolean. The agent honors true as a
 * preview override layered on top of its own LIVE_* env gates, so the Pages
 * layer can request a no-transmit preview; with dry_run false the
 * agent's env decides. When the agent is armed and dry_run is false, a command
 * sent through here transmits a REAL order. Do not treat this endpoint as
 * preview-only.
 *
 * Idempotency: the client mints one UUID per user intent and reuses it on
 * retry-after-error; a well-formed body.id is forwarded unchanged so the broker
 * and agent can dedup resubmissions instead of double-executing.
 */
import { requireAccess } from "./_access.js";

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

async function hmacHex(key, msg) {
  const enc = new TextEncoder();
  const k = await crypto.subtle.importKey("raw", enc.encode(key), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const sig = await crypto.subtle.sign("HMAC", k, enc.encode(msg));
  return [...new Uint8Array(sig)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export async function onRequestPost({ request, env }) {
  const headers = { "Content-Type": "application/json", "Cache-Control": "no-store" };
  const denied = await requireAccess(request, env);
  if (denied) return denied;
  const base = env.EXEC_BROKER_URL, token = env.STATUS_TOKEN;
  if (!base || !token) {
    return new Response(JSON.stringify({ ok: false, error: "broker not configured" }), { status: 503, headers });
  }
  let body;
  try { body = await request.json(); } catch { return new Response(JSON.stringify({ ok: false, error: "bad json" }), { status: 400, headers }); }

  if (!body || !["primary", "pa"].includes(body.account)) {
    return new Response(JSON.stringify({ ok: false, error: "explicit valid account required" }), { status: 400, headers });
  }
  if (typeof body.dry_run !== "boolean") {
    return new Response(JSON.stringify({ ok: false, error: "explicit dry_run boolean required" }), { status: 400, headers });
  }

  const now = Date.now();
  const command = {
    // client-minted idempotency id (one per user intent, reused on retry) when
    // well-formed; otherwise minted fresh here. Broker + agent dedup on it.
    id: typeof body.id === "string" && UUID_RE.test(body.id) ? body.id : crypto.randomUUID(),
    type: String(body.type || ""),
    account: body.account,
    dry_run: body.dry_run,  // immutable no-transmit intent; downstream may only restrict further
    payload: body.payload || {},
    created_at: now,
    expires_at: now + 60_000,                       // 60s validity
  };
  const signed = JSON.stringify(command);
  const sig = await hmacHex(token, signed);

  try {
    const r = await fetch(`${base.replace(/\/$/, "")}/command`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
      body: JSON.stringify({ signed, sig }),
    });
    const data = await r.json().catch(() => ({}));
    return new Response(JSON.stringify({ ...data, id: command.id }), { status: r.status, headers });
  } catch (e) {
    return new Response(JSON.stringify({ ok: false, error: String(e) }), { status: 502, headers });
  }
}
