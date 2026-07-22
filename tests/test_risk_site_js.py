import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RISK_JS = ROOT / "site" / "assets" / "risk.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_risk_chart_keeps_ma_line_and_adds_gapless_daily_bar_panel():
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
const dates = ["2026-07-16", "2026-07-17", "2026-07-20", "2026-07-22"];
const payload = {
  asof: "2026-07-22", built_at: "2026-07-22 12:00 UTC", spy_last: 704,
  price_ctx: {}, fragility: {"63d": 74}, regime_mult: 1, n_active: 0,
  signals: [], forward_returns: {},
  spy_series: {dates, close: [700, 701, 703, 704]},
  sizing_state: {
    score: 48, threshold: 50, throttle_on: false, gap_to_threshold: 2,
    days_in_state: 3, banded_strategies: [], throttled: [],
    spark: {dates, ma: [43, 44, 46, 48], daily: [61, 68, 72, 74]},
  },
};
const plots = [];
const sandbox = {
  console,
  document: {
    addEventListener(name, fn) { if (name === "DOMContentLoaded") ready = fn; },
    getElementById: element, querySelectorAll() { return []; },
  },
  renderNav() {}, setAsof() {}, fetchJSONOrNull: async () => payload,
  fmt: {num: v => String(v), pct: v => String(v), signed: v => String(v)},
  plotLayout: value => value, PLOT_CFG: {},
  Plotly: {
    newPlot(el, traces, layout) { plots.push({id: el.id, traces, layout}); },
    relayout() {},
  },
  Date, Math, Number, String, Object, Array, Set,
};
vm.createContext(sandbox);
vm.runInContext(source, sandbox);
Promise.resolve(ready()).then(() => {
  const chart = plots.find(p => p.id === "riskChart");
  if (!chart) throw new Error("risk chart missing");
  const ma = chart.traces.find(t => t.yaxis === "y2");
  const daily = chart.traces.find(t => t.yaxis === "y3");
  if (!ma || ma.mode !== "lines") throw new Error("10d MA is not an upper-panel line");
  if (!daily || daily.type !== "bar") throw new Error("daily 63d bars are not in panel 2");
  if (daily.y.join(",") !== "61,68,72,74") throw new Error("bars do not use daily readings");
  if (chart.layout.yaxis.domain[0] <= chart.layout.yaxis3.domain[1]) {
    throw new Error("chart domains overlap");
  }
  if (chart.layout.bargap !== 0) throw new Error("bar gap must be zero");
  const breaks = chart.layout.xaxis.rangebreaks;
  if (!breaks.some(b => b.bounds && b.bounds.join(",") === "sat,mon")) {
    throw new Error("weekend range break missing");
  }
  if (!breaks.some(b => b.values && b.values.includes("2026-07-21"))) {
    throw new Error("missing-session range break missing");
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


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_atr_downside_tables_render():
    """A payload carrying atr_downside renders the dial-band table under the hero
    and a per-signal table under EACH firing signal (and none under off ones)."""
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
const H = ["5d", "10d", "21d", "42d", "63d"];
const mk = o => Object.fromEntries(H.map(h => [h, {"1": o + 15, "2": o + 5, "3": o - 5, "5": o - 25}]));
const atr = {
  measure: "low_touch", atr_period: 14, mults: [1, 2, 3, 5], horizons: H,
  data_from: "2001-01-02", data_through: "2026-06-02",
  baseline: mk(45),
  signals: { "Seasonal Rank Divergence": { n_events: 139, n_episodes: 55, episode: mk(55), day: mk(50) } },
  dial: { value: 42.9, band: 3, lo: 39.9, hi: 45.9, table: mk(58),
          n_by_h: Object.fromEntries(H.map(h => [h, 150])), band_from: "2017-07-25", band_through: "2026-06-02" },
};
const payload = {
  asof: "2026-07-22", built_at: "2026-07-22 11:00 UTC", spy_last: 748,
  price_ctx: {}, fragility: {"63d": 48.6}, regime_mult: 1, n_active: 1,
  sizing_state: { score: 42.9, threshold: 50, throttle_on: false, gap_to_threshold: 7.1,
                  days_in_state: 12, banded_strategies: [], throttled: [], spark: { dates: [], ma: [] } },
  signals: [
    { name: "Seasonal Rank Divergence", on: true, badge: "FIRING", detail: "risk-off leads" },
    { name: "Dispersion", on: false, badge: "OFF", detail: "" },
  ],
  forward_returns: {}, atr_downside: atr,
};
const sandbox = {
  console,
  document: { addEventListener(name, fn) { if (name === "DOMContentLoaded") ready = fn; }, getElementById: element },
  renderNav() {}, setAsof() {}, fetchJSONOrNull: async () => payload,
  fmt: { num: (v, d) => Number(v).toFixed(d == null ? 0 : d), pct: v => String(v), signed: v => String(v) },
  plotLayout: v => v, PLOT_CFG: {}, Plotly: { newPlot() {}, relayout() {} },
  Date, Math, Number, String, Object, Array,
};
vm.createContext(sandbox);
vm.runInContext(source, sandbox);
Promise.resolve(ready()).then(() => {
  const html = element("content").innerHTML;
  const fail = m => { throw new Error(m); };
  if (!html.includes("Downside when the dial sits here")) fail("dial table missing");
  if (!html.includes("42.9")) fail("dial value missing");
  if (!html.includes("fresh Seasonal Rank Divergence trigger")) fail("firing-signal table missing");
  if (!html.includes("55 episodes")) fail("episode count missing");
  if (!html.includes("&ge;2 ATR")) fail("ATR column header missing");
  if (html.includes("fresh Dispersion trigger")) fail("off-signal must not get a table");
  if ((html.split("atr-card").length - 1) !== 2) fail("expected exactly 2 atr-cards (dial + 1 firing)");
}).catch(error => { console.error(error); process.exitCode = 1; });
""".replace("__RISK_JS__", json.dumps(str(RISK_JS)))

    subprocess.run(
        [shutil.which("node"), "-e", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
