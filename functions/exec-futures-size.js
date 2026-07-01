/* Pages Function — futures position-sizing query proxy (behind Access).
 *   POST /exec-futures-size {symbol, entry, stop, target?, risk? | risk_pct?, account_key?, account_value?}
 *        -> kicks off a pure read-only sizing calc on the agent, returns {id}
 *   GET  /exec-futures-size  -> returns the latest {query:{id,symbol,result}}
 * Keeps the broker URL + token server-side. READ-ONLY — no order is placed. */
const base = (env) => (env.EXEC_BROKER_URL || "").replace(/\/$/, "");
const H = { "Content-Type": "application/json", "Cache-Control": "no-store" };

export async function onRequestPost({ request, env }) {
  if (!base(env) || !env.STATUS_TOKEN) return new Response(JSON.stringify({ ok: false, error: "broker not configured" }), { status: 503, headers: H });
  let body;
  try { body = await request.json(); } catch { return new Response(JSON.stringify({ ok: false, error: "bad json" }), { status: 400, headers: H }); }
  try {
    const r = await fetch(`${base(env)}/futures_size`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${env.STATUS_TOKEN}` },
      body: JSON.stringify({
        symbol: body.symbol, entry: body.entry, stop: body.stop, target: body.target ?? null,
        risk: body.risk ?? null, risk_pct: body.risk_pct ?? null,
        account_key: body.account_key || "primary", account_value: body.account_value ?? null,
      }),
    });
    return new Response(JSON.stringify(await r.json().catch(() => ({}))), { status: r.status, headers: H });
  } catch (e) {
    return new Response(JSON.stringify({ ok: false, error: String(e) }), { status: 502, headers: H });
  }
}

export async function onRequestGet({ env }) {
  if (!base(env) || !env.STATUS_TOKEN) return new Response(JSON.stringify({ query: null, configured: false }), { headers: H });
  try {
    const r = await fetch(`${base(env)}/futures_size`, { headers: { Authorization: `Bearer ${env.STATUS_TOKEN}` } });
    return new Response(JSON.stringify(await r.json().catch(() => ({ query: null }))), { headers: H });
  } catch (e) {
    return new Response(JSON.stringify({ query: null, error: String(e) }), { headers: H });
  }
}
