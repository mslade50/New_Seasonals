import assert from "node:assert/strict";
import fs from "node:fs";

// Load the Function with _access.js swapped for a stub that records calls.
let denyWith = null;
globalThis.__access = async () => denyWith;
const src = fs.readFileSync(new URL("../../functions/idea-check.js", import.meta.url), "utf8")
  .replace(/import \{ requireAccess \} from "\.\/_access\.js";/, "const requireAccess = (...a) => globalThis.__access(...a);");
assert.ok(src.includes("requireAccess(request, env)"));
const { onRequestPost, onRequestGet } = await import(`data:text/javascript;base64,${Buffer.from(src).toString("base64")}`);

function fakeR2() {
  const store = new Map();
  return {
    store,
    async get(k) { return store.has(k) ? { text: async () => store.get(k) } : null; },
    async put(k, v) { store.set(k, v); },
  };
}
const post = (env, body) => onRequestPost({ request: { json: async () => { if (body === "BAD") throw new Error("x"); return body; }, headers: { get: () => null } }, env });
const get = (env) => onRequestGet({ request: { headers: { get: () => null } }, env });
const ID_RE = /^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{6}$/;

const env = { CHARTS: fakeR2() };
for (const bad of ["BAD", null, {}, { text: "   " }, { text: 5 }, { text: "x".repeat(2001) }]) {
  const r = await post(env, bad);
  assert.equal(r.status, 400, JSON.stringify(bad));
}
assert.equal(env.CHARTS.store.size, 0);

let r = await post(env, { text: "  long SPY after a gap  " });
assert.equal(r.status, 200);
assert.equal(r.headers.get("Cache-Control"), "no-store");
const first = (await r.json()).id;
assert.match(first, ID_RE);
assert.equal((await post(env, { text: "x".repeat(2000) })).status, 200);

const ids = [first];
for (let i = 0; i < 24; i++) ids.push((await (await post(env, { text: `idea ${i}` })).json()).id);
const q = JSON.parse(env.CHARTS.store.get("idea_check/queue.json")).requests;
assert.equal(q.length, 20);
assert.deepEqual(q.map((x) => x.id), ids.slice(-20).length === 20 ? q.map((x) => x.id) : []);
assert.equal(q[q.length - 1].text, "idea 23");
assert.ok(q.every((x) => ID_RE.test(x.id) && /Z$/.test(x.submitted_at)));
assert.equal(q[0].text, "idea 4");  // 27 posts in total, oldest 7 dropped

// GET: newest 10 first, result joined or null.
const newest = q[q.length - 1].id;
env.CHARTS.store.set(`idea_check/results/${newest}.json`, JSON.stringify({ id: newest, status: "done", verdict: "KILL" }));
r = await get(env);
const body = await r.json();
assert.equal(r.headers.get("Cache-Control"), "no-store");
assert.equal(body.requests.length, 10);
assert.equal(body.requests[0].id, newest);
assert.equal(body.requests[0].result.verdict, "KILL");
assert.equal(body.requests[1].result, null);
assert.deepEqual(Object.keys(body.requests[0]).sort(), ["id", "result", "submitted_at", "text"]);

// Empty queue and fail-closed paths.
assert.deepEqual(await (await get({ CHARTS: fakeR2() })).json(), { requests: [] });
assert.equal((await get({})).status, 503);
assert.equal((await post({}, { text: "a" })).status, 503);
denyWith = new Response("{}", { status: 401 });
assert.equal((await post(env, { text: "a" })).status, 401);
assert.equal((await get(env)).status, 401);
console.log("test_idea_check_function.mjs passed");
