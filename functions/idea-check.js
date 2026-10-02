/* Pages Function — Idea Check queue: accept a trade idea, list recent verdicts.
 *
 * Route: /idea-check. POST {"text": "..."} appends a request to R2
 * "idea_check/queue.json" (newest 20 kept, oldest first). GET returns the newest
 * 10 requests, newest first, each joined to "idea_check/results/<id>.json" (null
 * while the local poller has not written one, i.e. "queued").
 *
 * Only this Function writes queue.json; only the local poller writes results.
 * Behind Cloudflare Access plus the in-code JWT check (_access.js), failing
 * closed. No broker, no HMAC: nothing here can place an order.
 *
 * Binding: CHARTS (the seasonals-cache bucket, same as pitch-today.js).
 */
import { requireAccess } from "./_access.js";

const QUEUE_KEY = "idea_check/queue.json";
const MAX_QUEUE = 20;
const LIST_N = 10;
const MAX_TEXT = 2000;
const ID_RE = /^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{6}$/;
const HEADERS = { "Content-Type": "application/json", "Cache-Control": "no-store" };

const reply = (obj, status = 200) => new Response(JSON.stringify(obj), { status, headers: HEADERS });

function newId(now = new Date()) {
  const stamp = now.toISOString().replace(/[-:]/g, "").replace(/\.\d+Z$/, "Z");
  const rnd = crypto.getRandomValues(new Uint8Array(3));
  return `${stamp}-${[...rnd].map((b) => b.toString(16).padStart(2, "0")).join("")}`;
}

async function readQueue(env) {
  const obj = await env.CHARTS.get(QUEUE_KEY);
  if (!obj) return [];
  try {
    const q = JSON.parse(await obj.text());
    return Array.isArray(q && q.requests) ? q.requests : [];
  } catch { return []; }
}

export async function onRequestPost({ request, env }) {
  const denied = await requireAccess(request, env);
  if (denied) return denied;
  if (!env.CHARTS) return reply({ error: "store not bound (CHARTS R2 binding missing)" }, 503);
  let body;
  try { body = await request.json(); } catch { return reply({ error: "bad json" }, 400); }
  const text = body && typeof body.text === "string" ? body.text.trim() : "";
  if (!text) return reply({ error: "text required" }, 400);
  if (text.length > MAX_TEXT) return reply({ error: `text over ${MAX_TEXT} characters` }, 400);

  const now = new Date();
  const entry = { id: newId(now), text, submitted_at: now.toISOString().replace(/\.\d+Z$/, "Z") };
  const requests = (await readQueue(env)).concat(entry).slice(-MAX_QUEUE);
  await env.CHARTS.put(QUEUE_KEY, JSON.stringify({ requests }),
    { httpMetadata: { contentType: "application/json" } });
  return reply({ id: entry.id });
}

export async function onRequestGet({ request, env }) {
  const denied = await requireAccess(request, env);
  if (denied) return denied;
  if (!env.CHARTS) return reply({ error: "store not bound (CHARTS R2 binding missing)" }, 503);
  const recent = (await readQueue(env)).slice(-LIST_N).reverse();
  const out = await Promise.all(recent.map(async (r) => {
    let result = null;
    if (typeof r.id === "string" && ID_RE.test(r.id)) {
      const obj = await env.CHARTS.get(`idea_check/results/${r.id}.json`);
      if (obj) { try { result = JSON.parse(await obj.text()); } catch { result = null; } }
    }
    return { id: r.id, text: r.text, submitted_at: r.submitted_at, result };
  }));
  return reply({ requests: out });
}
