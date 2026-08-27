/* Read-only R2 history projection for the private Focus page. */
import { validateFocusEnvelope } from "./_discretionary-focus-contract.js";

const HISTORY_PREFIX = "discretionary_focus/history/";

function jsonResponse(payload, status = 200) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
  });
}

export async function onRequestGet({ env }) {
  if (!env.CHARTS) return jsonResponse({ error: "store not bound" }, 503);
  let listed;
  try {
    listed = await env.CHARTS.list({ prefix: HISTORY_PREFIX, limit: 1000 });
  } catch (_) {
    return jsonResponse({ error: "focus history is unavailable" }, 503);
  }
  const keys = (listed.objects || []).map((object) => object.key)
    .filter((key) => key.endsWith(".json")).sort().reverse().slice(0, 60);
  const payloads = await Promise.all(keys.map(async (key) => {
    try {
      const object = await env.CHARTS.get(key);
      if (!object) return null;
      const payload = JSON.parse(await object.text());
      return validateFocusEnvelope(payload, new Date()) ? payload : null;
    } catch (_) { return null; }
  }));

  const byDate = new Map();
  for (const payload of payloads.filter(Boolean)) {
    const prior = byDate.get(payload.valid_for);
    if (!prior || (prior.phase !== "FINAL" && payload.phase === "FINAL"))
      byDate.set(payload.valid_for, payload);
  }
  const items = [...byDate.values()].sort((a, b) => b.valid_for.localeCompare(a.valid_for))
    .slice(0, 10).map((payload) => ({
      valid_for: payload.valid_for,
      status: payload.status,
      phase: payload.phase,
      generated_at: payload.generated_at,
      focus: payload.focus.map((card) => ({
        ticker: card.ticker,
        company_name: card.company_name,
      })),
    }));
  return jsonResponse({ schema_version: "discretionary-focus-history.v1", items });
}
