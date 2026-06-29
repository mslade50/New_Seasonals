/* Pages Function — serve the morning order-chain summary JSON live from R2.
 *
 * Route: /morning-orders  ->  R2 key "morning_orders.json".
 *
 * The local morning_order_summary.py uploads this object after the 9:31 AM ET
 * order chain — AFTER the AM site deploy — so it can't be baked into dist/. The
 * Orders tab fetches it here, no-store, to always show the latest morning run.
 *
 * Binding: reuses CHARTS (bound to the seasonals-cache bucket in wrangler.toml);
 * the morning_orders.json key lives in that same bucket. The site sits behind
 * Cloudflare Access, so this inherits that auth wall. READ-ONLY — never writes.
 */
export async function onRequestGet({ env }) {
  const jsonHeaders = { "Content-Type": "application/json", "Cache-Control": "no-store" };
  if (!env.CHARTS) {
    return new Response(JSON.stringify({ error: "store not bound (CHARTS R2 binding missing)" }),
      { status: 503, headers: jsonHeaders });
  }
  const obj = await env.CHARTS.get("morning_orders.json");
  if (!obj) {
    return new Response(JSON.stringify({ error: "no morning summary published yet" }),
      { status: 404, headers: jsonHeaders });
  }
  const headers = new Headers(jsonHeaders);
  headers.set("etag", obj.httpEtag);
  return new Response(obj.body, { headers });
}
