import assert from "node:assert/strict";
import fs from "node:fs";

const source = fs.readFileSync(new URL("../../functions/_access.js", import.meta.url), "utf8");
const encoded = Buffer.from(source).toString("base64");
const { requireAccess } = await import(`data:text/javascript;base64,${encoded}`);

function requestWith(token = null) {
  return { headers: { get: (name) => name === "Cf-Access-Jwt-Assertion" ? token : null } };
}

async function expectDenied(name, env, token, status, errorText) {
  const response = await requireAccess(requestWith(token), env);
  assert.ok(response instanceof Response, `${name}: expected a Response`);
  assert.equal(response.status, status, `${name}: unexpected HTTP status`);
  assert.equal(response.headers.get("Cache-Control"), "no-store");
  const body = await response.json();
  assert.equal(body.ok, false);
  assert.match(body.error, errorText);
  console.log(`PASS ${name}: HTTP ${response.status} ${body.error}`);
}

await expectDenied("no Access config", {}, null, 503, /not configured/i);
await expectDenied("domain only", { ACCESS_TEAM_DOMAIN: "team.cloudflareaccess.com" }, null, 503, /not configured/i);
await expectDenied("audience only", { ACCESS_AUD: "aud" }, null, 503, /not configured/i);
await expectDenied(
  "configured without JWT",
  { ACCESS_TEAM_DOMAIN: "team.cloudflareaccess.com", ACCESS_AUD: "aud" },
  null,
  401,
  /missing Access JWT/i,
);
await expectDenied(
  "configured with malformed JWT",
  { ACCESS_TEAM_DOMAIN: "team.cloudflareaccess.com", ACCESS_AUD: "aud" },
  "not-a-jwt",
  401,
  /invalid Access JWT: malformed JWT/i,
);
