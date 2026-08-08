/* Pages Function — private, reversible research-priority state.
 *
 * Route: /fundamental-state
 * R2 key: fundamental/site_state.json
 *
 * This endpoint can only store DEEPEN / WATCH / PASS choices. It has no order,
 * broker, portfolio, allocation, or messaging capability.
 */
import { requireAccess } from "./_access.js";

const STATE_KEY = "fundamental/site_state.json";
const ACTIONS = new Set(["DEEPEN", "WATCH", "PASS", "CLEAR"]);
const JSON_HEADERS = { "Content-Type": "application/json", "Cache-Control": "no-store" };

function defaultState() {
  return { version: 1, updated_at: null, actions: {} };
}

function response(body, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: JSON_HEADERS });
}

async function loadState(bucket) {
  const object = await bucket.get(STATE_KEY);
  if (!object) return defaultState();
  try {
    const parsed = JSON.parse(await object.text());
    if (!parsed || typeof parsed !== "object" || !parsed.actions || typeof parsed.actions !== "object") {
      return defaultState();
    }
    return { version: 1, updated_at: parsed.updated_at || null, actions: parsed.actions };
  } catch (_) {
    return defaultState();
  }
}

async function gate(request, env) {
  const denied = await requireAccess(request, env);
  if (denied) return denied;
  if (!env.CHARTS) return response({ ok: false, error: "research-state store is not bound" }, 503);
  return null;
}

export async function onRequestGet({ request, env }) {
  const denied = await gate(request, env);
  if (denied) return denied;
  return response(await loadState(env.CHARTS));
}

export async function onRequestPost({ request, env }) {
  const denied = await gate(request, env);
  if (denied) return denied;

  let text;
  try { text = await request.text(); }
  catch (_) { return response({ ok: false, error: "request body could not be read" }, 400); }
  if (text.length > 4096) return response({ ok: false, error: "request body is too large" }, 413);

  let body;
  try { body = JSON.parse(text); }
  catch (_) { return response({ ok: false, error: "request body must be JSON" }, 400); }

  const ticker = String((body && body.ticker) || "").trim().toUpperCase();
  const action = String((body && body.action) || "").trim().toUpperCase();
  const asOf = body && body.as_of != null ? String(body.as_of).slice(0, 32) : null;
  if (!/^[A-Z][A-Z0-9.-]{0,9}$/.test(ticker)) {
    return response({ ok: false, error: "invalid ticker" }, 400);
  }
  if (!ACTIONS.has(action)) return response({ ok: false, error: "invalid research action" }, 400);

  const state = await loadState(env.CHARTS);
  const now = new Date().toISOString();
  if (action === "CLEAR") delete state.actions[ticker];
  else state.actions[ticker] = { action, updated_at: now, as_of: asOf };
  state.version = 1;
  state.updated_at = now;

  await env.CHARTS.put(STATE_KEY, JSON.stringify(state), {
    httpMetadata: { contentType: "application/json", cacheControl: "no-store" },
  });
  return response(state);
}
