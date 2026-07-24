/* Pages Function — proxy the broker's accumulated executions to the site.
 * Route: GET /exec-fills. Behind Access; keeps the broker URL + token server-side.
 * Returns {fills:[...], retention_days} (per-execution rows for both accounts,
 * merged across days by the broker DO), or {fills:null} when the broker isn't
 * configured/reachable. READ-ONLY.
 */
import { requireAccess } from "./_access.js";

export async function onRequestGet({ request, env }) {
  const headers = { "Content-Type": "application/json", "Cache-Control": "no-store" };
  const denied = await requireAccess(request, env);
  if (denied) return denied;
  const base = env.EXEC_BROKER_URL, token = env.STATUS_TOKEN;
  if (!base || !token) {
    return new Response(JSON.stringify({ fills: null, configured: false }), { headers });
  }
  try {
    const r = await fetch(`${base.replace(/\/$/, "")}/fills`, { headers: { Authorization: `Bearer ${token}` } });
    const data = await r.json().catch(() => ({ fills: null }));
    return new Response(JSON.stringify(data), { headers });
  } catch (e) {
    return new Response(JSON.stringify({ fills: null, error: String(e) }), { headers });
  }
}
