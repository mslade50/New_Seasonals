/* Pages Function - serve the latest Risk Agent payload live from R2.
 *
 * Route: /risk-agent-today  ->  R2 key "risk_agent/today.json".
 *
 * Not baked into dist/: daily_risk_agent.py publishes in the evening, after the
 * site deploy, so a baked copy would always be a day old. Same pattern as
 * pitch-today.js.
 *
 * Binding: reuses CHARTS (bound to the seasonals-cache bucket in wrangler.toml).
 * The site sits behind Cloudflare Access, so this inherits that auth wall.
 * READ-ONLY: never writes, and the paper sleeve has no execution path at all.
 */
export async function onRequestGet({ env }) {
  const jsonHeaders = { "Content-Type": "application/json", "Cache-Control": "no-store" };
  if (!env.CHARTS) {
    return new Response(JSON.stringify({ error: "store not bound (CHARTS R2 binding missing)" }),
      { status: 503, headers: jsonHeaders });
  }
  const obj = await env.CHARTS.get("risk_agent/today.json");
  if (!obj) {
    return new Response(JSON.stringify({ error: "no risk agent run published yet" }),
      { status: 404, headers: jsonHeaders });
  }
  const headers = new Headers(jsonHeaders);
  headers.set("etag", obj.httpEtag);
  return new Response(obj.body, { headers });
}
