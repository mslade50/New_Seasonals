"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const source = fs.readFileSync(
  path.join(__dirname, "..", "..", "site", "assets", "common.js"), "utf8");

function meta(build) {
  const health = {
    _site_build_id: build,
    build_id: build,
    built_at: "display-only and deliberately not parsed",
    artifacts: {},
  };
  return {
    _site_build_id: build,
    build_id: build,
    built_at: health.built_at,
    freshness: health,
    payloads: { health: true },
  };
}

function response(body) {
  return {
    ok: true,
    status: 200,
    headers: {
      get: name => name.toLowerCase() === "content-type" ? "application/json" : null,
    },
    json: async () => body,
  };
}

const context = {
  console,
  document: { addEventListener() {} },
  window: {},
  setTimeout: fn => fn(),
};
vm.createContext(context);
vm.runInContext(source, context, { filename: "common.js" });

(async () => {
  const anchors = [meta("A"), meta("B"), meta("B")];
  let payloadAttempts = 0;
  context.fetch = async url => {
    if (url.startsWith("data/meta.json")) return response(anchors.shift() || meta("B"));
    payloadAttempts += 1;
    return response({ _site_build_id: "B", candidates: [] });
  };

  const snapshot = await context.loadSiteSnapshot(async anchor => ({
    ideas: await context.fetchSitePayload(anchor, "data/ideas.json"),
  }), 3);
  assert.strictEqual(snapshot.meta.build_id, "B");
  assert.strictEqual(snapshot.ideas._site_build_id, "B");
  assert.strictEqual(payloadAttempts, 2);
  assert.strictEqual(
    context.sitePayloadPath("data/ideas.json", snapshot.meta),
    "data/ideas.json?site_build=B");

  let boundedAttempts = 0;
  context.fetch = async url => {
    if (url.startsWith("data/meta.json")) return response(meta("A"));
    boundedAttempts += 1;
    return response({ _site_build_id: "B", candidates: [] });
  };
  await assert.rejects(
    context.loadSiteSnapshot(async anchor => ({
      ideas: await context.fetchSitePayload(anchor, "data/ideas.json"),
    }), 2),
    /coherent site build/i,
  );
  assert.strictEqual(boundedAttempts, 2);

  console.log("PASS atomic site manifest");
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
