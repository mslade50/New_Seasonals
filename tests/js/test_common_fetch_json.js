"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const source = fs.readFileSync(
  path.join(__dirname, "..", "..", "site", "assets", "common.js"),
  "utf8",
);

let response;
const context = {
  fetch: async () => response,
  console,
};
vm.createContext(context);
vm.runInContext(source, context, { filename: "common.js" });

(async () => {
  response = {
    ok: true,
    headers: { get: () => "text/html; charset=UTF-8" },
    json: async () => ({ shouldNotParse: true }),
  };
  await assert.rejects(
    vm.runInContext('fetchJSON("data/fundamentals.json")', context),
    /expected JSON but received text\/html.*Refresh or sign in again/,
  );

  response = {
    ok: true,
    headers: { get: () => "application/json; charset=UTF-8" },
    json: async () => ({ ok: true }),
  };
  const payload = await vm.runInContext('fetchJSON("data/fundamentals.json")', context);
  assert.strictEqual(payload.ok, true);

  console.log("PASS JSON fetch content-type guard");
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
