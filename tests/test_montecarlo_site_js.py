import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MC_JS = ROOT / "site" / "assets" / "montecarlo.js"

_HARNESS = r"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync(__MC_JS__, "utf8");

function fakeNode(tag) {
  return {
    tag, id: "", innerHTML: "", textContent: "", style: {}, children: [],
    listeners: {},
    classList: {
      _set: new Set(),
      add(c) { this._set.add(c); },
      remove(c) { this._set.delete(c); },
      toggle() {},
      contains(c) { return this._set.has(c); },
    },
    appendChild(ch) { this.children.push(ch); return ch; },
    addEventListener(name, fn) { this.listeners[name] = fn; },
    querySelectorAll(sel) {
      return sel === "button" ? this.children.filter(c => c.tag === "button") : [];
    },
  };
}
const elements = new Map();
function element(id) {
  if (!elements.has(id)) { const n = fakeNode("div"); n.id = id; elements.set(id, n); }
  return elements.get(id);
}
let ready;
const plots = [];
const sandbox = {
  console,
  document: {
    addEventListener(name, fn) { if (name === "DOMContentLoaded") ready = fn; },
    getElementById: element,
    createElement: fakeNode,
    querySelectorAll() { return []; },
  },
  renderNav() {}, setAsof() {},
  fetchJSONOrNull: async () => payload,
  fmt: { money: v => "$" + Math.round(v) },
  plotLayout: v => v, PLOT_CFG: {},
  Plotly: {
    newPlot(el, traces, layout) {
      plots.push({ id: typeof el === "string" ? el : el.id, traces, layout });
    },
    relayout() {},
  },
  Date, Math, Number, String, Object, Array, Set, JSON, parseInt,
};
"""


def _run(script):
    subprocess.run(
        [shutil.which("node"), "-e", script],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_daily_pnl_chart_defaults_to_1y_and_range_control_rewindows():
    script = (_HARNESS + r"""
// ~2 years of weekday dates ending 2026-07-28
const dates = [];
let d = new Date("2024-08-01T00:00:00Z");
const end = new Date("2026-07-28T00:00:00Z");
while (d <= end) {
  const wd = d.getUTCDay();
  if (wd !== 0 && wd !== 6) dates.push(d.toISOString().slice(0, 10));
  d = new Date(d.getTime() + 86400e3);
}
const pnl = dates.map((_, i) => (i % 3) - 1);  // mix of -1/0/+1
const payload = {
  asof: dates[dates.length - 1], date_min: dates[0], basis_nav: 750000,
  n_sims: 10000, mean_block_td: 10,
  daily_series: { dates, pnl },
  empirical: {
    n_days: dates.length, pct_active: 50, p_up_all: 30, p_up_active: 55, p_flat: 40,
    mean_day: 100, ann_pnl: 25200, std_day: 900, sharpe: 1.8, var95: 1000,
    var99: 2000, cvar99: 3000,
    day_bands: { "5": -1000, "25": 0, "50": 0, "75": 200, "95": 1200 },
    thresholds: [], worst_days: [],
  },
  modern: {
    p_up_active: 55, mean_day: 100, ann_pnl: 25200, sharpe: 1.8,
    var99: 2000, cvar99: 3000,
  },
  month: { bands: { "5": -5000, "25": 0, "50": 2000, "75": 5000, "95": 12000 },
           p_neg: 30, p_lt_2pct: 5, p_lt_5pct: 1, p_bad_day: 10,
           dd_p50: -3000, dd_p95: -9000, dd_worst: -20000,
           hist: { edges: [0, 1, 2], counts: [1, 1] } },
  year: { bands: { "5": -10000, "25": 20000, "50": 60000, "75": 90000, "95": 150000 },
          p_neg: 8, p_lt_2pct: 4, p_lt_5pct: 1, p_bad_day: 60,
          dd_p50: -10000, dd_p95: -30000, dd_worst: -60000,
          hist: { edges: [0, 1, 2], counts: [1, 1] } },
  calendar: {
    months: { n: 24, p_neg: 30, median: 2000, worst: -9000, worst_when: "2025-04" },
    years: { n: 2, p_neg: 0, median: 30000, worst: 10000, worst_when: "2025" },
  },
  intraday: {
    n_days: 100, cal_years: 2.0, table: [], hist: { edges: [0, 1], counts: [1] },
    scatter: { dates: [], trough_pct: [], finish_pct: [] }, deepest: [],
    series: { dates: [dates[0], dates[dates.length - 2]], trough: [-500, -700] },
  },
};
vm.createContext(sandbox);
vm.runInContext(source, sandbox);
Promise.resolve(ready()).then(() => {
  const chart = plots.find(p => p.id === "dailyChart");
  if (!chart) throw new Error("daily PnL chart missing");
  const pts = chart.traces[0];
  if (pts.x.length >= dates.length) throw new Error("default window is not trailing 1y");
  if (pts.x[0] < "2025-07-28") throw new Error("1y window starts too early: " + pts.x[0]);
  const trough = chart.traces.find(t => t.name === "intraday trough");
  if (!trough) throw new Error("trough trace missing");
  if (trough.x.length !== 1 || trough.x[0] !== dates[dates.length - 2]) {
    throw new Error("trough dots not clipped to window");
  }
  if (!element("dailyStats").innerHTML.includes("maxDD")) {
    throw new Error("window stats line missing maxDD");
  }
  // simulate picking "All" on the range control
  const seg = element("dailyRangeSeg");
  const allBtn = seg.children.find(b => b.textContent === "All");
  if (!allBtn) throw new Error("range control missing All preset");
  const onBtn = seg.children.find(b => b.classList.contains("on"));
  if (!onBtn || onBtn.textContent !== "1Y") throw new Error("default preset is not 1Y");
  allBtn.listeners["click"]();
  const chart2 = plots.filter(p => p.id === "dailyChart").pop();
  if (chart2.traces[0].x.length !== dates.length) {
    throw new Error("All preset did not expand to full history");
  }
  if (chart2.traces.find(t => t.name === "intraday trough").x.length !== 2) {
    throw new Error("All preset did not restore both trough dots");
  }
}).catch(error => { console.error(error); process.exitCode = 1; });
""").replace("__MC_JS__", json.dumps(str(MC_JS)))
    _run(script)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_old_payload_without_daily_series_still_renders():
    script = (_HARNESS + r"""
const payload = {
  asof: "2026-07-28", date_min: "2003-01-17", basis_nav: 750000,
  n_sims: 10000, mean_block_td: 10,
  empirical: {
    n_days: 5000, pct_active: 50, p_up_all: 30, p_up_active: 55, p_flat: 40,
    mean_day: 100, ann_pnl: 25200, std_day: 900, sharpe: 1.8, var95: 1000,
    var99: 2000, cvar99: 3000,
    day_bands: { "5": -1000, "25": 0, "50": 0, "75": 200, "95": 1200 },
    thresholds: [], worst_days: [],
  },
  modern: {
    p_up_active: 55, mean_day: 100, ann_pnl: 25200, sharpe: 1.8,
    var99: 2000, cvar99: 3000,
  },
  month: { bands: { "5": -5000, "25": 0, "50": 2000, "75": 5000, "95": 12000 },
           p_neg: 30, p_lt_2pct: 5, p_lt_5pct: 1, p_bad_day: 10,
           dd_p50: -3000, dd_p95: -9000, dd_worst: -20000,
           hist: { edges: [0, 1, 2], counts: [1, 1] } },
  year: { bands: { "5": -10000, "25": 20000, "50": 60000, "75": 90000, "95": 150000 },
          p_neg: 8, p_lt_2pct: 4, p_lt_5pct: 1, p_bad_day: 60,
          dd_p50: -10000, dd_p95: -30000, dd_worst: -60000,
          hist: { edges: [0, 1, 2], counts: [1, 1] } },
  calendar: {
    months: { n: 280, p_neg: 30, median: 2000, worst: -9000, worst_when: "2008-10" },
    years: { n: 23, p_neg: 0, median: 30000, worst: 10000, worst_when: "2004" },
  },
};
vm.createContext(sandbox);
vm.runInContext(source, sandbox);
Promise.resolve(ready()).then(() => {
  if (plots.find(p => p.id === "dailyChart")) {
    throw new Error("daily chart rendered without daily_series payload");
  }
  if (!plots.find(p => p.id === "histMonth") || !plots.find(p => p.id === "histYear")) {
    throw new Error("distribution histograms missing on legacy payload");
  }
  if (element("content").innerHTML.includes("Daily PnL (actual)")) {
    throw new Error("daily card markup emitted without payload support");
  }
}).catch(error => { console.error(error); process.exitCode = 1; });
""").replace("__MC_JS__", json.dumps(str(MC_JS)))
    _run(script)
