/* Pages Function — accept a command from the site, sign it, forward to the broker.
 *
 * Route: POST /exec-command. Behind Cloudflare Access (human auth). Wraps the
 * site's request into a command envelope, HMAC-signs it with STATUS_TOKEN (shared
 * with the local agent, which verifies + is the final gatekeeper), and POSTs
 * {signed, sig} to the broker. The broker relays it down the agent's socket.
 *
 * Phase 2b: dry_run is FORCED true here regardless of input, and the agent only
 * ever logs what it WOULD do. No order is constructed or transmitted anywhere on
 * this path. (2c introduces a dedicated signing key + an explicit live-enable.)
 */
async function hmacHex(key, msg) {
  const enc = new TextEncoder();
  const k = await crypto.subtle.importKey("raw", enc.encode(key), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const sig = await crypto.subtle.sign("HMAC", k, enc.encode(msg));
  return [...new Uint8Array(sig)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export async function onRequestPost({ request, env }) {
  const headers = { "Content-Type": "application/json", "Cache-Control": "no-store" };
  const base = env.EXEC_BROKER_URL, token = env.STATUS_TOKEN;
  if (!base || !token) {
    return new Response(JSON.stringify({ ok: false, error: "broker not configured" }), { status: 503, headers });
  }
  let body;
  try { body = await request.json(); } catch { return new Response(JSON.stringify({ ok: false, error: "bad json" }), { status: 400, headers }); }

  const now = Date.now();
  const command = {
    id: crypto.randomUUID(),
    type: String(body.type || ""),
    account: body.account === "primary" ? "primary" : "pa",
    dry_run: true,                                  // forced server-side in 2b
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
