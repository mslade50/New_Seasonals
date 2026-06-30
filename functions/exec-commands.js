/* Pages Function — proxy the broker's recent commands + results to the site.
 * Route: GET /exec-commands. Behind Access; keeps the broker URL + token
 * server-side. Returns {commands:[...]} (newest first), or an empty list when the
 * broker isn't configured/reachable. READ-ONLY.
 */
export async function onRequestGet({ env }) {
  const headers = { "Content-Type": "application/json", "Cache-Control": "no-store" };
  const base = env.EXEC_BROKER_URL, token = env.STATUS_TOKEN;
  if (!base || !token) {
    return new Response(JSON.stringify({ commands: [], configured: false }), { headers });
  }
  try {
    const r = await fetch(`${base.replace(/\/$/, "")}/commands`, { headers: { Authorization: `Bearer ${token}` } });
    const data = await r.json().catch(() => ({ commands: [] }));
    return new Response(JSON.stringify(data), { headers });
  } catch (e) {
    return new Response(JSON.stringify({ commands: [], error: String(e) }), { headers });
  }
}
