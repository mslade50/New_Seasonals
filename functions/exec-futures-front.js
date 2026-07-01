/* Pages Function — futures front-month resolve proxy (behind Access).
 *   POST /exec-futures-front {symbol}  -> kicks off a read-only reqContractDetails on the agent, returns {id}
 *   GET  /exec-futures-front           -> returns the latest {query:{id,symbol,result}}
 * Keeps the broker URL + token server-side. READ-ONLY — no order is placed. */
const base = (env) => (env.EXEC_BROKER_URL || "").replace(/\/$/, "");
const H = { "Content-Type": "application/json", "Cache-Control": "no-store" };

export async function onRequestPost({ request, env }) {
  if (!base(env) || !env.STATUS_TOKEN) return new Response(JSON.stringify({ ok: false, error: "broker not configured" }), { status: 503, headers: H });
  let body;
  try { body = await request.json(); } catch { return new Response(JSON.stringify({ ok: false, error: "bad json" }), { status: 400, headers: H }); }
  try {
    const r = await fetch(`${base(env)}/futures_front`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${env.STATUS_TOKEN}` },
      body: JSON.stringify({ symbol: body.symbol }),
    });
    return new Response(JSON.stringify(await r.json().catch(() => ({}))), { status: r.status, headers: H });
  } catch (e) {
    return new Response(JSON.stringify({ ok: false, error: String(e) }), { status: 502, headers: H });
  }
}

export async function onRequestGet({ env }) {
  if (!base(env) || !env.STATUS_TOKEN) return new Response(JSON.stringify({ query: null, configured: false }), { headers: H });
  try {
    const r = await fetch(`${base(env)}/futures_front`, { headers: { Authorization: `Bearer ${env.STATUS_TOKEN}` } });
    return new Response(JSON.stringify(await r.json().catch(() => ({ query: null }))), { headers: H });
  } catch (e) {
    return new Response(JSON.stringify({ query: null, error: String(e) }), { headers: H });
  }
}
