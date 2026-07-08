/* Pages Function — options-workbench query proxy (behind Access).
 *   POST /exec-workbench {ticker, mode, expiry?, max_expiries?, context?}
 *        -> kicks off a read-only term-structure + chain fetch, returns {id}
 *   GET  /exec-workbench?id=<id>  -> returns that query's {query:{id,ticker,result}}
 * Keeps the broker URL + token server-side. READ-ONLY (chain + greeks). */
import { requireAccess } from "./_access.js";

const base = (env) => (env.EXEC_BROKER_URL || "").replace(/\/$/, "");
const H = { "Content-Type": "application/json", "Cache-Control": "no-store" };

export async function onRequestPost({ request, env }) {
  const denied = await requireAccess(request, env);
  if (denied) return denied;
  if (!base(env) || !env.STATUS_TOKEN) return new Response(JSON.stringify({ ok: false, error: "broker not configured" }), { status: 503, headers: H });
  let body;
  try { body = await request.json(); } catch { return new Response(JSON.stringify({ ok: false, error: "bad json" }), { status: 400, headers: H }); }
  try {
    const r = await fetch(`${base(env)}/workbench`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${env.STATUS_TOKEN}` },
      body: JSON.stringify({
        ticker: body.ticker, mode: body.mode || "full", expiry: body.expiry || null,
        max_expiries: body.max_expiries || null, context: body.context || null,
      }),
    });
    return new Response(JSON.stringify(await r.json().catch(() => ({}))), { status: r.status, headers: H });
  } catch (e) {
    return new Response(JSON.stringify({ ok: false, error: String(e) }), { status: 502, headers: H });
  }
}

export async function onRequestGet({ request, env }) {
  const denied = await requireAccess(request, env);
  if (denied) return denied;
  if (!base(env) || !env.STATUS_TOKEN) return new Response(JSON.stringify({ query: null, configured: false }), { headers: H });
  try {
    const id = new URL(request.url).searchParams.get("id");
    const r = await fetch(`${base(env)}/workbench${id ? `?id=${encodeURIComponent(id)}` : ""}`,
      { headers: { Authorization: `Bearer ${env.STATUS_TOKEN}` } });
    return new Response(JSON.stringify(await r.json().catch(() => ({ query: null }))), { headers: H });
  } catch (e) {
    return new Response(JSON.stringify({ query: null, error: String(e) }), { headers: H });
  }
}
