/* Pages Function — serve today's Daily Pitch slate live from R2.
 *
 * Route: /pitch-today  ->  R2 key "pitch_today.json".
 *
 * Not baked into dist/ for the same reason radar_recs.json isn't: daily_pitch.py
 * publishes ~05:30-05:40 ET, AFTER the morning site deploy, so a baked copy
 * would always be yesterday's slate. Reading it live means the Pitch tab shows
 * this morning's ideas the moment the publisher lands them.
 *
 * Binding: reuses CHARTS (bound to the seasonals-cache bucket in wrangler.toml);
 * pitch_today.json lives in that same bucket. The site sits behind Cloudflare
 * Access, so this inherits that auth wall. READ-ONLY — never writes.
 */
export async function onRequestGet({ env }) {
  const jsonHeaders = { "Content-Type": "application/json", "Cache-Control": "no-store" };
  if (!env.CHARTS) {
    return new Response(JSON.stringify({ error: "store not bound (CHARTS R2 binding missing)" }),
      { status: 503, headers: jsonHeaders });
  }
  const obj = await env.CHARTS.get("pitch_today.json");
  if (!obj) {
    return new Response(JSON.stringify({ error: "no pitch published yet" }),
      { status: 404, headers: jsonHeaders });
  }
  const headers = new Headers(jsonHeaders);
  headers.set("etag", obj.httpEtag);
  return new Response(obj.body, { headers });
}
