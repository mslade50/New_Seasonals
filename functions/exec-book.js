/* Pages Function — proxy the broker's live read-only book to the site.
 * Route: GET /exec-book. Behind Access; keeps the broker URL + token server-side.
 * Returns {book:{accounts:[...]}} (positions / orders / NLV), or {book:null} when
 * the broker isn't configured/reachable. READ-ONLY.
 */
export async function onRequestGet({ env }) {
  const headers = { "Content-Type": "application/json", "Cache-Control": "no-store" };
  const base = env.EXEC_BROKER_URL, token = env.STATUS_TOKEN;
  if (!base || !token) {
    return new Response(JSON.stringify({ book: null, configured: false }), { headers });
  }
  try {
    const r = await fetch(`${base.replace(/\/$/, "")}/book`, { headers: { Authorization: `Bearer ${token}` } });
    const data = await r.json().catch(() => ({ book: null }));
    return new Response(JSON.stringify(data), { headers });
  } catch (e) {
    return new Response(JSON.stringify({ book: null, error: String(e) }), { headers });
  }
}
