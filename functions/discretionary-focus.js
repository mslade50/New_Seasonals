/* Pages Function — serve the current discretionary research shortlist from R2.
 *
 * Route: /discretionary-focus
 * R2 key: discretionary_focus/current.json
 *
 * The feed updates independently of the private site's heavier build cadence.
 * This endpoint is read-only and the site remains behind Cloudflare Access.
 */
const FOCUS_KEY = "discretionary_focus/current.json";
const FOCUS_SCHEMA = "discretionary-focus.v1";

function jsonResponse(payload, status = 200, extraHeaders = null) {
  const headers = new Headers({
    "Content-Type": "application/json",
    "Cache-Control": "no-store",
  });
  if (extraHeaders) {
    for (const [key, value] of Object.entries(extraHeaders)) {
      if (value) headers.set(key, value);
    }
  }
  return new Response(JSON.stringify(payload), { status, headers });
}

function validEnvelope(payload) {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) return false;
  if (payload.schema_version !== FOCUS_SCHEMA || payload.research_only !== true) return false;
  if (!["READY", "NO_QUALIFIED_SETUP"].includes(payload.status)) return false;
  if (!Array.isArray(payload.focus) || payload.focus.length > 2) return false;
  if (payload.status === "READY" && payload.focus.length === 0) return false;
  if (payload.status === "NO_QUALIFIED_SETUP" && payload.focus.length !== 0) return false;
  return true;
}

export async function onRequestGet({ env }) {
  if (!env.CHARTS) {
    return jsonResponse({ error: "store not bound (CHARTS R2 binding missing)" }, 503);
  }
  const object = await env.CHARTS.get(FOCUS_KEY);
  if (!object) {
    return jsonResponse({ error: "no discretionary focus payload published yet" }, 404);
  }

  let payload;
  try {
    payload = JSON.parse(await object.text());
  } catch (_) {
    return jsonResponse({ error: "discretionary focus payload is not valid JSON" }, 502);
  }
  if (!validEnvelope(payload)) {
    return jsonResponse({ error: "discretionary focus payload failed its safety envelope" }, 502);
  }
  return jsonResponse(payload, 200, { etag: object.httpEtag });
}
