/* Pages Function — option-spread query proxy (behind Access).
 *   POST /exec-option {ticker, expiry?}  -> kicks off a read-only chain fetch, returns {id}
 *   GET  /exec-option                    -> returns the latest {query:{id,ticker,result}}
 * Keeps the broker URL + token server-side. READ-ONLY (option chain + greeks). */
const base = (env) => (env.EXEC_BROKER_URL || "").replace(/\/$/, "");
const H = { "Content-Type": "application/json", "Cache-Control": "no-store" };

export async function onRequestPost({ request, env }) {
  if (!base(env) || !env.STATUS_TOKEN) return new Response(JSON.stringify({ ok: false, error: "broker not configured" }), { status: 503, headers: H });
  let body;
  try { body = await request.json(); } catch { return new Response(JSON.stringify({ ok: false, error: "bad json" }), { status: 400, headers: H }); }
  try {
    const r = await fetch(`${base(env)}/option`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${env.STATUS_TOKEN}` },
      body: JSON.stringify({ ticker: body.ticker, expiry: body.expiry || null }),
    });
    return new Response(JSON.stringify(await r.json().catch(() => ({}))), { status: r.status, headers: H });
  } catch (e) {
    return new Response(JSON.stringify({ ok: false, error: String(e) }), { status: 502, headers: H });
  }
}

export async function onRequestGet({ env }) {
  if (!base(env) || !env.STATUS_TOKEN) return new Response(JSON.stringify({ query: null, configured: false }), { headers: H });
  try {
    const r = await fetch(`${base(env)}/option`, { headers: { Authorization: `Bearer ${env.STATUS_TOKEN}` } });
    return new Response(JSON.stringify(await r.json().catch(() => ({ query: null }))), { headers: H });
  } catch (e) {
    return new Response(JSON.stringify({ query: null, error: String(e) }), { headers: H });
  }
}
