/* Pages Function — validate a command, sign it, and forward it to the broker.
 *
 * Route: POST /exec-command. Behind Cloudflare Access (human auth), plus an
 * in-code Access JWT check (_access.js) so a misconfigured Access wall doesn't
 * leave this endpoint open. Wraps the site's request into a command envelope,
 * HMAC-signs it with the dedicated COMMAND_SECRET (shared only with the broker
 * and local agent), and POSTs {signed, sig} to the broker. STATUS_TOKEN remains
 * read-only and is used only to fetch the broker's status/book snapshots.
 *
 * Live commands require all of: explicit dry_run:false, a fresh online book
 * that reports mode=live, Pages and broker kill switches, an armed command
 * type/account, strict payload validation, and server-side risk caps. Omitted
 * dry_run means preview. Unknown/stale state fails closed.
 *
 * Idempotency: the client mints one UUID per user intent and reuses it on
 * retry-after-error; a well-formed body.id is forwarded unchanged so the broker
 * and agent can dedup resubmissions instead of double-executing.
 */
import { requireAccess } from "./_access.js";
import { validateCommandRequest } from "./_execution_policy.mjs";

const MAX_BODY_BYTES = 32_768;

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
  const base = String(env.EXEC_BROKER_URL || "").replace(/\/$/, "");
  const statusToken = env.STATUS_TOKEN;
  const commandSecret = env.COMMAND_SECRET;
  if (!base || !statusToken || !commandSecret) {
    return new Response(JSON.stringify({ ok: false, error: "execution command bridge is not fully configured" }), { status: 503, headers });
  }

  const contentLength = Number(request.headers.get("Content-Length") || 0);
  if (contentLength > MAX_BODY_BYTES) {
    return new Response(JSON.stringify({ ok: false, error: "request body too large" }), { status: 413, headers });
  }

  let body;
  try {
    const raw = await request.text();
    if (raw.length > MAX_BODY_BYTES) {
      return new Response(JSON.stringify({ ok: false, error: "request body too large" }), { status: 413, headers });
    }
    body = JSON.parse(raw);
  } catch {
    return new Response(JSON.stringify({ ok: false, error: "bad json" }), { status: 400, headers });
  }

  try {
    const readHeaders = { Authorization: `Bearer ${statusToken}` };
    const [statusResponse, bookResponse] = await Promise.all([
      fetch(`${base}/status`, { headers: readHeaders }),
      fetch(`${base}/book`, { headers: readHeaders }),
    ]);
    if (!statusResponse.ok || !bookResponse.ok) {
      return new Response(JSON.stringify({ ok: false, error: "could not verify fresh execution state" }), { status: 503, headers });
    }
    const status = await statusResponse.json();
    const bookPayload = await bookResponse.json();
    const now = Date.now();
    const decision = validateCommandRequest(body, {
      env, status, book: bookPayload && bookPayload.book, now,
    });
    if (!decision.ok) {
      return new Response(JSON.stringify({ ok: false, error: decision.error }), { status: decision.status, headers });
    }

    const command = {
      ...decision.command,
      created_at: now,
      expires_at: now + 60_000,
      policy_version: decision.policy_version,
    };
    const signed = JSON.stringify(command);
    const sig = await hmacHex(commandSecret, signed);
    const r = await fetch(`${base}/command`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${commandSecret}` },
      body: JSON.stringify({ signed, sig }),
    });
    const data = await r.json().catch(() => ({}));
    return new Response(JSON.stringify({
      ...data,
      id: command.id,
      dry_run: command.dry_run,
      policy_version: decision.policy_version,
    }), { status: r.status, headers });
  } catch (e) {
    return new Response(JSON.stringify({ ok: false, error: String(e) }), { status: 502, headers });
  }
}
