import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RISK_JS = ROOT / "site" / "assets" / "risk.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_old_risk_payload_still_renders_without_signal_detail():
    script = r"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync(__RISK_JS__, "utf8");
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    id, innerHTML: "", textContent: "", on() {}, querySelectorAll() { return []; },
  });
  return elements.get(id);
}
let ready;
const payload = {
  asof: "2026-07-15", built_at: "2026-07-15 12:00 UTC", spy_last: 700,
  price_ctx: {}, fragility: {"21d": 20}, regime_mult: 1, n_active: 0,
  signals: [{name: "Legacy Signal", on: false, badge: "OFF", detail: ""}],
  forward_returns: {},
  spy_series: {dates: ["2026-07-14", "2026-07-15"], close: [699, 700]},
};
const plots = [];
const sandbox = {
  console,
  document: {
    addEventListener(name, fn) { if (name === "DOMContentLoaded") ready = fn; },
    getElementById: element,
  },
  renderNav() {}, setAsof() {}, fetchJSONOrNull: async () => payload,
  fmt: {num: v => String(v), pct: v => String(v), signed: v => String(v)},
  plotLayout: value => value, PLOT_CFG: {},
  Plotly: {
    newPlot(el, traces, layout) { plots.push({id: el.id, traces, layout}); },
    relayout() {},
  },
  Date, Math, Number, String, Object, Array,
};
vm.createContext(sandbox);
vm.runInContext(source, sandbox);
Promise.resolve(ready()).then(() => {
  const html = element("content").innerHTML;
  if (!html.includes("Legacy Signal")) throw new Error("legacy signal card missing");
  if (html.includes("signalOverlayChart")) throw new Error("unguarded signal overlay");
  if (plots.length !== 1 || plots[0].id !== "riskChart") {
    throw new Error("legacy SPY chart did not render");
  }
}).catch(error => { console.error(error); process.exitCode = 1; });
""".replace("__RISK_JS__", json.dumps(str(RISK_JS)))

    subprocess.run(
        [shutil.which("node"), "-e", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
