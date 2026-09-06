import { requireAccess } from "./_access.js";
import { readInputs, buildStatus } from "./_sleeve-status.js";

export async function onRequestGet({ request, env }) {
  const denied = await requireAccess(request, env);
  if (denied) return denied;
  const headers = { "Content-Type": "application/json", "Cache-Control": "no-store" };
  if (!env.CHARTS) return new Response(JSON.stringify({ error: "Status store unavailable" }), { status: 503, headers });
  return new Response(JSON.stringify(buildStatus(await readInputs(env.CHARTS))), { headers });
}
