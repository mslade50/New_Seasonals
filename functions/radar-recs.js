/* Pages Function — serve the momentum radar's weekly recs live from R2.
 *
 * Route: /radar-recs  ->  R2 key "radar_recs.json".
 *
 * Not baked into dist/ for the same reason morning_orders.json isn't: the radar
 * runs on its own weekend cadence (Sun/Mon 11:00 UTC), independent of the site's
 * 2x-daily deploy, and scripts/upload_radar_recs.py publishes from the local box
 * whenever that lands. Reading it live means the Radar tab shows this week's
 * plans without waiting for the next deploy.
 *
 * Binding: reuses CHARTS (bound to the seasonals-cache bucket in wrangler.toml);
 * radar_recs.json lives in that same bucket. The site sits behind Cloudflare
 * Access, so this inherits that auth wall. READ-ONLY — never writes.
 */
export async function onRequestGet({ env }) {
  const jsonHeaders = { "Content-Type": "application/json", "Cache-Control": "no-store" };
  if (!env.CHARTS) {
    return new Response(JSON.stringify({ error: "store not bound (CHARTS R2 binding missing)" }),
      { status: 503, headers: jsonHeaders });
  }
  const obj = await env.CHARTS.get("radar_recs.json");
  if (!obj) {
    return new Response(JSON.stringify({ error: "no radar recs published yet" }),
      { status: 404, headers: jsonHeaders });
  }
  const headers = new Headers(jsonHeaders);
  headers.set("etag", obj.httpEtag);
  return new Response(obj.body, { headers });
}
