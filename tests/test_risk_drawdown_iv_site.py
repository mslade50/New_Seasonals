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
    id, innerHTML: "", textContent: "",
    value: id === "ddIvHorizon" ? "63d" : id === "ddIvThreshold" ? "2" : "0.1",
    listeners: {},
    addEventListener(name, fn) { this.listeners[name] = fn; },
    querySelectorAll() { return []; },
  });
  return elements.get(id);
}
let ready;
const episode = {
  anchor_date: "2025-02-19", worst_low_date: "2025-04-08",
  anchor_spy_close: 610, anchor_atr: 8, worst_spy_low: 562,
  max_drawdown_atr: 6, max_drawdown_pct: -0.0787, sessions_to_low: 33,
  iv_start_close: 15, iv_peak: 52, iv_peak_date: "2025-04-08",
  iv_change_points: 37, iv_change_pct: 2.4667,
};
const episode2 = {
  anchor_date: "2022-01-03", worst_low_date: "2022-01-24",
  anchor_spy_close: 478, anchor_atr: 6, worst_spy_low: 454,
  max_drawdown_atr: 4, max_drawdown_pct: -0.0502, sessions_to_low: 14,
  iv_start_close: 16, iv_peak: 29, iv_peak_date: "2022-01-24",
  iv_change_points: 13, iv_change_pct: 0.8125,
};
const payload = {
  asof: "2026-08-03", built_at: "2026-08-03 22:00 UTC", spy_last: 700,
  price_ctx: {}, fragility: {}, regime_mult: 1, n_active: 0, signals: [],
  forward_returns: {"63d": {current_score: 40, n_episodes: 2, band_low: 37,
    band_high: 43, returns: {"63": {mean: 0.01, median: 0.01, pct_neg: 0.4, mean_z: 0.2}}}},
  drawdown_iv: {score_horizon: "63d", current_score: 40, band_low: 35, band_high: 45,
    n_episodes: 2, atr_period: 14, iv_basis: "VIX intraday high",
    default_horizon: "63d", default_threshold: 2,
    horizons: ["5d", "63d"], eligible_by_horizon: {"5d": 2, "63d": 1},
    thresholds: [1, 2, 3, 5], counts: {"5d": {"1": 1}, "63d": {"1": 2, "2": 2, "3": 2, "5": 1}},
    rows_by_horizon: {"5d": [episode], "63d": [episode, episode2]}},
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
  const dd = html.lastIndexOf("Peak IV after similar risk readings");
  if (fwd < 0 || dd <= fwd) throw new Error("drawdown IV table is not the last section");
  if (!html.includes("Same 2 declustered historical anchors")) throw new Error("sample linkage missing");
  if (!html.includes("Forward window")) throw new Error("window selector missing");
  if (!html.includes("6.00 ATR")) throw new Error("ATR drawdown missing");
  if (!html.includes("VIX close &rarr; peak")) throw new Error("close-to-peak column missing");
  if (!html.includes("52.0")) throw new Error("peak IV missing");
  if (!html.includes("+37.0 pts")) throw new Error("IV change missing");
  if (!html.includes("&ge;2 ATR (2 paths)")) throw new Error("threshold selector missing");
  const initialSummary = element("ddIvSummary").innerHTML;
  if (!initialSummary.includes("IV change") || !initialSummary.includes("ATR drawdown")) throw new Error("summary boxes missing");
  if (!initialSummary.includes("+25.0 pts") || !initialSummary.includes("5.00 ATR")) throw new Error("average/median summaries wrong");
  element("ddIvThreshold").value = "5";
  element("ddIvThreshold").listeners.change();
  const filteredSummary = element("ddIvSummary").innerHTML;
  if (!filteredSummary.includes("+37.0 pts") || !filteredSummary.includes("6.00 ATR")) throw new Error("summaries did not update with filters");
}).catch(error => { console.error(error); process.exitCode = 1; });
""".replace("__RISK_JS__", json.dumps(str(RISK_JS)))

    subprocess.run(
        [shutil.which("node"), "-e", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
