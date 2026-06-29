/* Pages Function — proxy the execution-broker /status to the site.
 *
 * Route: /exec-status. Keeps the broker URL + STATUS_TOKEN server-side and keeps
 * the call same-origin (so it stays behind Cloudflare Access). Returns
 * {online:false, configured:false} gracefully when the broker env vars aren't set
 * or the broker is unreachable, so the Execution tab shows "offline"/"not
 * configured" instead of erroring. READ-ONLY.
 *
 * Pages env vars (Settings -> Environment variables):
 *   EXEC_BROKER_URL  e.g. https://execution-broker.<subdomain>.workers.dev
 *   STATUS_TOKEN     matches the Worker's STATUS_TOKEN secret
 */
export async function onRequestGet({ env }) {
  const headers = { "Content-Type": "application/json", "Cache-Control": "no-store" };
  const base = env.EXEC_BROKER_URL;
  if (!base) {
    return new Response(JSON.stringify({ online: false, configured: false }), { headers });
  }
  try {
    const r = await fetch(`${base.replace(/\/$/, "")}/status`, {
      headers: { Authorization: `Bearer ${env.STATUS_TOKEN || ""}` },
    });
    if (!r.ok) {
      return new Response(JSON.stringify({ online: false, configured: true, error: `status ${r.status}` }), { headers });
    }
    const data = await r.json();
    return new Response(JSON.stringify({ ...data, configured: true }), { headers });
  } catch (e) {
    return new Response(JSON.stringify({ online: false, configured: true, error: String(e) }), { headers });
  }
}
