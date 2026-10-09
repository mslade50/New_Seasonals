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

// Shape check for the held-option commands (OPTION_COMBO_SPEC.md section 6).
// Structure only: leg identity, sides, ratios, price side, TIF. Unit counts,
// held-position checks and risk are the agent's and executor's job (no caps
// here). Same signing / expiry / idempotency path as option_spread below.
const SIDES = new Set(["BUY", "SELL"]);
function legsProblem(legs, label, max, opening) {
  if (!Array.isArray(legs) || legs.length < 1 || legs.length > max) return `${label}: 1 to ${max} legs required`;
  const seen = new Set();
  for (const [i, l] of legs.entries()) {
    if (!l || typeof l !== "object") return `${label} ${i + 1}: must be an object`;
    const ratio = l.ratio == null ? 1 : Number(l.ratio);
    if (!Number.isInteger(ratio) || ratio < 1) return `${label} ${i + 1}: ratio must be a whole number >= 1`;
    if (opening) {
      if (!SIDES.has(String(l.side || "").toUpperCase())) return `${label} ${i + 1}: side must be BUY or SELL`;
      if (!["C", "P"].includes(String(l.right || "").toUpperCase())) return `${label} ${i + 1}: right must be C or P`;
      if (!/^\d{8}$/.test(String(l.expiry || "").replace(/-/g, ""))) return `${label} ${i + 1}: expiry must be YYYYMMDD`;
      if (!(Number(l.strike) > 0)) return `${label} ${i + 1}: strike must be > 0`;
    } else {
      const cid = Number(l.con_id);
      if (!Number.isInteger(cid) || cid <= 0) return `${label} ${i + 1}: positive con_id required`;
      if (seen.has(cid)) return `${label} ${i + 1}: duplicate con_id`;
      seen.add(cid);
      if (!SIDES.has(String(l.action || "").toUpperCase())) return `${label} ${i + 1}: action must be BUY or SELL`;
    }
  }
  return null;
}
function optionPositionProblem(type, p) {
  if (type !== "option_close" && type !== "option_roll") return null;
  if (!p || typeof p !== "object") return "payload required";
  if (!/^[A-Z0-9.]{1,12}$/.test(String(p.symbol || ""))) return "symbol required";
  if (!SIDES.has(String(p.action || ""))) return "action must be BUY (net debit) or SELL (net credit)";
  const limit = Number(p.limit);
  if (!Number.isFinite(limit) || limit < 0 || (type === "option_close" && limit === 0)) return "limit must be a positive net price";
  if (!["DAY", "GTC"].includes(String(p.tif || "DAY"))) return "tif must be DAY or GTC";
  if (type === "option_close") return legsProblem(p.legs, "close leg", 4, false);
  const problem = legsProblem(p.close_legs, "close leg", 4, false) || legsProblem(p.open_legs, "new leg", 4, true);
  if (problem) return problem;
  if (p.close_legs.length + p.open_legs.length > 6) return "a roll has at most 6 legs";
  const risk = Number(p.debit_risk);
  if (!Number.isFinite(risk) || risk < 0) return "debit_risk must be a number >= 0";
  if (p.unbounded_ack != null && p.unbounded_ack !== true) return "unbounded_ack must be literal true when present";
  return null;
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

  // Whole-idea commands may be signed only after the dedicated route verifies
  // the current immutable proposal, delivery receipt and separate confirmation.
  if (body.type === "review_execution" ||
      (env.REVIEW_EXECUTION_LIVE_ENABLED === "1" &&
       /^(Pitch-|Seasonal_Agent-)/.test(String(body.payload?.strategy || "")))) {
    return new Response(JSON.stringify({ ok: false, error: "use the verified whole-idea review-execution route" }), { status: 409, headers });
  }

  const optionProblem = optionPositionProblem(String(body.type || ""), body.payload);
  if (optionProblem) {
    return new Response(JSON.stringify({ ok: false, error: `${body.type}: ${optionProblem}` }), { status: 400, headers });
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
