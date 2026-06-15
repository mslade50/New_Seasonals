/* Pages Function — stream per-trade signal-chart PNGs from R2.
 *
 * Route: /chartimg/*  ->  R2 key "charts/<path>" in bucket binding CHARTS.
 *
 * The public route is /chartimg/ (NOT /charts/) on purpose: the gallery page
 * charts.html serves at the pretty URL /charts, and a /charts/* catch-all
 * function would shadow it. The R2 keys still live under charts/ (that's where
 * the renderer uploads), so the function just re-prefixes.
 *
 * Charts are immutable per (strategy, ticker, signal-date) so they cache hard.
 * The whole site sits behind Cloudflare Access, so this inherits that auth wall.
 */
export async function onRequestGet({ env, params }) {
  if (!env.CHARTS) {
    return new Response("Chart store not bound (CHARTS R2 binding missing)", { status: 503 });
  }
  const parts = Array.isArray(params.path) ? params.path : [params.path];
  // Keys are slugged ASCII (letters/digits/underscore + .png); reject empties
  // and anything that smells like traversal.
  if (!parts.length || parts.some((p) => !p || p === ".." || p.includes("\\") || p.includes("/"))) {
    return new Response("Bad request", { status: 400 });
  }
  const key = "charts/" + parts.join("/");
  const obj = await env.CHARTS.get(key);
  if (!obj) return new Response("Not found", { status: 404 });

  const headers = new Headers();
  obj.writeHttpMetadata(headers);
  headers.set("etag", obj.httpEtag);
  headers.set("Content-Type", "image/png");
  headers.set("Cache-Control", "public, max-age=86400, immutable");
  return new Response(obj.body, { headers });
}
