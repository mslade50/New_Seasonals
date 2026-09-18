/* Smoke test for execution.js render() — checks the not-configured / online /
   offline states without a browser. */
const fs = require("fs"), vm = require("vm");
const sandbox = { console, Date };
sandbox.window = sandbox;
sandbox.document = { addEventListener() {}, getElementById() { return { innerHTML: "", textContent: "" }; } };
vm.createContext(sandbox);
vm.runInContext(
  fs.readFileSync("site/assets/common.js", "utf8") + "\n" +
  fs.readFileSync("site/assets/execution.js", "utf8"), sandbox);

const cases = [
  ["not-configured", { configured: false }, "Broker not configured"],
  ["online", { configured: true, online: true, sockets: 1, heartbeat_age_ms: 4000, stale_after_ms: 30000, connected_at: Date.now() }, "Execution online"],
  ["offline-noagent", { configured: true, online: false, sockets: 0 }, "No agent connected"],
  ["offline-stale", { configured: true, online: false, sockets: 1, heartbeat_age_ms: 60000 }, "heartbeat is stale"],
];
let ok = true;
for (const [name, s, want] of cases) {
  const html = sandbox.render(s);
  const hit = html.includes(want);
  if (!hit) ok = false;
  console.log((hit ? "OK   " : "MISS ") + name + " -> " + want);
}
process.exit(ok ? 0 : 1);
