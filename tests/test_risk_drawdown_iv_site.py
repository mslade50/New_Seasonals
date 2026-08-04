import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RISK_JS = ROOT / "site" / "assets" / "risk.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_drawdown_iv_table_is_the_last_risk_page_section():
    script = r"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync(__RISK_JS__, "utf8");
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    id, innerHTML: "", textContent: "", value: "0.1",
    addEventListener() {}, querySelectorAll() { return []; },
  });
  return elements.get(id);
}
let ready;
const episode = {
  peak_date: "2025-02-19", trough_date: "2025-04-08", recovery_date: "2025-06-27",
  peak_spy: 610, trough_spy: 520, max_drawdown: -0.1475, days_to_trough: 33,
  iv_start_close: 15, iv_peak: 52, iv_peak_date: "2025-04-08",
  iv_change_points: 37, iv_change_pct: 2.4667,
};
const payload = {
  asof: "2026-08-03", built_at: "2026-08-03 22:00 UTC", spy_last: 700,
  price_ctx: {}, fragility: {}, regime_mult: 1, n_active: 0, signals: [],
  forward_returns: {"63d": {current_score: 40, n_episodes: 2, band_low: 37,
    band_high: 43, returns: {"63": {mean: 0.01, median: 0.01, pct_neg: 0.4, mean_z: 0.2}}}},
  drawdown_iv: {sample_from: "2016-08-03", sample_through: "2026-08-03",
    iv_basis: "VIX intraday high", default_threshold: 0.10,
    thresholds: [0.05, 0.10, 0.15], counts: {"0.05": 1, "0.1": 1, "0.15": 0},
    episodes: [episode]},
};
const sandbox = {
  console,
  document: {
    addEventListener(name, fn) { if (name === "DOMContentLoaded") ready = fn; },
    getElementById: element, querySelectorAll() { return []; },
  },
  renderNav() {}, setAsof() {}, fetchJSONOrNull: async () => payload,
  fmt: {
    num: (v, d = 1) => Number(v).toFixed(d),
    pct: (v, d = 1) => (Number(v) * 100).toFixed(d) + "%",
    signed: (v, d = 1) => (Number(v) >= 0 ? "+" : "") + Number(v).toFixed(d),
  },
  plotLayout: v => v, PLOT_CFG: {}, Plotly: {newPlot() {}, relayout() {}},
  Date, Math, Number, String, Object, Array, Set,
};
vm.createContext(sandbox);
vm.runInContext(source, sandbox);
Promise.resolve(ready()).then(() => {
  const html = element("content").innerHTML;
  const fwd = html.lastIndexOf("Forward returns at similar fragility readings");
  const dd = html.lastIndexOf("Peak IV during SPY drawdowns");
  if (fwd < 0 || dd <= fwd) throw new Error("drawdown IV table is not the last section");
  if (!html.includes("VIX close &rarr; peak")) throw new Error("close-to-peak column missing");
  if (!html.includes("52.0")) throw new Error("peak IV missing");
  if (!html.includes("+37.0 pts")) throw new Error("IV change missing");
  if (!html.includes("&ge;10% (1 episode)")) throw new Error("threshold selector missing");
}).catch(error => { console.error(error); process.exitCode = 1; });
""".replace("__RISK_JS__", json.dumps(str(RISK_JS)))

    subprocess.run(
        [shutil.which("node"), "-e", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
