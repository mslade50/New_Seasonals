/* Pages Function — serve the current discretionary research shortlist from R2.
 *
 * Route: /discretionary-focus
 * R2 key: discretionary_focus/current.json
 *
 * The feed updates independently of the private site's heavier build cadence.
 * This endpoint is read-only and the site remains behind Cloudflare Access.
 */
import { validateFocusEnvelope } from "./_discretionary-focus-contract.js";

const FOCUS_KEY = "discretionary_focus/current.json";

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
  if (!validateFocusEnvelope(payload, new Date())) {
    return jsonResponse({ error: "discretionary focus payload failed strict validation" }, 502);
  }
  // The object text is parsed and reserialized above, so its original strong
  // ETag would not describe these response bytes. no-store is intentional.
  return jsonResponse(payload);
}
