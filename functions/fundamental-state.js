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
  return { version: 1, updated_at: null, actions: {}, history: [] };
}

function response(body, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: JSON_HEADERS });
}

async function loadState(bucket) {
  const object = await bucket.get(STATE_KEY);
  if (!object) return { state: defaultState(), etag: null };
  const parsed = JSON.parse(await object.text());
  if (!parsed || Array.isArray(parsed) || parsed.version !== 1
      || !parsed.actions || typeof parsed.actions !== "object" || Array.isArray(parsed.actions)
      || (parsed.history != null && !Array.isArray(parsed.history)) || !object.etag) {
    throw new Error("Invalid stored research state");
  }
  for (const [ticker, record] of Object.entries(parsed.actions)) {
    if (!/^[A-Z][A-Z0-9.-]{0,9}$/.test(ticker) || !record || Array.isArray(record)
        || !ACTIONS.has(record.action)) throw new Error("Invalid stored research action");
  }
  // Preserve existing fields and evidence. A corrupt object is never replaced
  // with a synthetic empty state by a subsequent POST.
  return { state: { ...parsed, history: parsed.history || [] }, etag: object.etag };
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
  try { return response((await loadState(env.CHARTS)).state); }
  catch (_) { return response({ ok: false, error: "research-state history is unavailable; stored evidence was preserved" }, 503); }
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

  try {
    for (let attempt = 0; attempt < 5; attempt++) {
      const { state, etag } = await loadState(env.CHARTS);
      const now = new Date().toISOString();
      if (action === "CLEAR") delete state.actions[ticker];
      else state.actions[ticker] = { action, updated_at: now, as_of: asOf };
      state.history.push({ ticker, action, updated_at: now, as_of: asOf });
      state.updated_at = now;
      // R2 conditional put returns null on conflict. Refetch and merge, never
      // acknowledge a choice whose write was rejected. Protect first creation too.
      const written = await env.CHARTS.put(STATE_KEY, JSON.stringify(state), {
        onlyIf: etag ? { etagMatches: etag } : { etagDoesNotMatch: "*" },
        httpMetadata: { contentType: "application/json", cacheControl: "no-store" },
      });
      if (written) return response(state);
    }
    return response({ ok: false, error: "research state changed concurrently; retry this choice" }, 409);
  } catch (_) {
    return response({ ok: false, error: "research-state update failed; stored history was preserved" }, 503);
  }
}
