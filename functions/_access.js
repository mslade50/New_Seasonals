/* Shared helper — Cloudflare Access JWT verification for the exec endpoints.
 *
 * NOT a route: Pages Functions only creates routes for files that export
 * onRequest* handlers (wrangler filepath-routing; only `_middleware` gets
 * special name treatment). This file exports none, so it never serves traffic.
 *
 * Defense-in-depth for the exec-* Functions: they sit behind the Cloudflare
 * Access wall, but nothing at the code layer verified the caller's identity —
 * a single Access misconfig (policy edited, app deleted, route not covered)
 * would expose an armed live-order endpoint. This validates the
 * Cf-Access-Jwt-Assertion header Access injects on every authenticated
 * request: RS256 signature against the team's published signing keys
 * (WebCrypto, no npm deps), aud must contain ACCESS_AUD, exp must be in the
 * future. Verification failures fail CLOSED (401) when enforcement is on.
 *
 * Enforcement is opt-in: when ACCESS_TEAM_DOMAIN and ACCESS_AUD are BOTH set
 * on the Pages project, a missing/invalid JWT gets a 401. When either is
 * unset, requests pass through with a one-time console.warn so the site keeps
 * working before the values are configured.
 *
 * Pages env vars (Settings -> Environment variables):
 *   ACCESS_TEAM_DOMAIN  e.g. myteam.cloudflareaccess.com
 *   ACCESS_AUD          the Access application's Audience (AUD) tag
 */

let _warned = false;
let _certs = { domain: null, keys: null, fetchedAt: 0 };   // module-scope cache (per isolate)
const CERTS_TTL_MS = 6 * 60 * 60 * 1000;                   // refetch at most every 6h (+ on kid miss)

function b64urlToBytes(s) {
  s = s.replace(/-/g, "+").replace(/_/g, "/");
  while (s.length % 4) s += "=";
  const bin = atob(s);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

async function fetchKeys(domain, force) {
  const now = Date.now();
  if (!force && _certs.keys && _certs.domain === domain && now - _certs.fetchedAt < CERTS_TTL_MS) {
    return _certs.keys;
  }
  const r = await fetch(`https://${domain}/cdn-cgi/access/certs`);
  if (!r.ok) throw new Error(`certs fetch failed: ${r.status}`);
  const data = await r.json();
  const keys = (data && data.keys) || [];
  _certs = { domain, keys, fetchedAt: now };
  return keys;
}

/* Returns null when the token is valid, else a short reason string. */
async function verifyJwt(token, domain, aud) {
  const parts = token.split(".");
  if (parts.length !== 3) return "malformed JWT";
  let header, payload;
  try {
    header = JSON.parse(new TextDecoder().decode(b64urlToBytes(parts[0])));
    payload = JSON.parse(new TextDecoder().decode(b64urlToBytes(parts[1])));
  } catch { return "undecodable JWT"; }
  if (header.alg !== "RS256") return `unexpected alg ${header.alg}`;

  let keys = await fetchKeys(domain, false);
  let jwk = keys.find((k) => k.kid === header.kid);
  if (!jwk) {                                             // key rotation — refetch once
    keys = await fetchKeys(domain, true);
    jwk = keys.find((k) => k.kid === header.kid);
  }
  if (!jwk) return "no matching signing key";

  const key = await crypto.subtle.importKey(
    "jwk", jwk, { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" }, false, ["verify"]);
  const ok = await crypto.subtle.verify(
    "RSASSA-PKCS1-v1_5", key, b64urlToBytes(parts[2]),
    new TextEncoder().encode(`${parts[0]}.${parts[1]}`));
  if (!ok) return "bad signature";

  const auds = Array.isArray(payload.aud) ? payload.aud : [payload.aud];
  if (!auds.includes(aud)) return "aud mismatch";
  const nowSec = Math.floor(Date.now() / 1000);
  if (typeof payload.exp !== "number" || payload.exp <= nowSec) return "expired";
  return null;
}

/* Gate an exec endpoint on the caller's Access identity.
 * Returns null when the request may proceed, or a 401 JSON Response when not.
 * Enforced only when ACCESS_TEAM_DOMAIN + ACCESS_AUD are both set. */
export async function requireAccess(request, env) {
  const domain = String(env.ACCESS_TEAM_DOMAIN || "")
    .replace(/^https?:\/\//, "").replace(/\/.*$/, "").trim();
  const aud = env.ACCESS_AUD || "";
  if (!domain || !aud) {
    if (!_warned) {
      console.warn("exec endpoints: ACCESS_TEAM_DOMAIN/ACCESS_AUD not set — "
        + "Access JWT verification is OFF (relying on the Access wall alone)");
      _warned = true;
    }
    return null;
  }
  const headers = { "Content-Type": "application/json", "Cache-Control": "no-store" };
  const token = request.headers.get("Cf-Access-Jwt-Assertion");
  if (!token) {
    return new Response(JSON.stringify({ ok: false, error: "missing Access JWT" }), { status: 401, headers });
  }
  let reason;
  try { reason = await verifyJwt(token, domain, aud); }
  catch (e) { reason = `verification error: ${(e && e.message) || e}`; }   // fail closed
  if (reason) {
    return new Response(JSON.stringify({ ok: false, error: `invalid Access JWT: ${reason}` }), { status: 401, headers });
  }
  return null;
}
