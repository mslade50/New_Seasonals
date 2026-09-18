import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RISK_JS = ROOT / "site" / "assets" / "risk.js"

# The shared teammate build serves risk.js against a payload with these
# sizing_state keys stripped (plus the "Book posture" nugget).  Kept here so a
# change to the redaction contract fails this guard rather than leaking.
REDACTED_SIZING_KEYS = (
    "banded_strategies",
    "throttled",
    "threshold",
    "throttle_on",
    "gap_to_threshold",
    "days_in_state",
    "episodes",
    "exposure",
    "sleeve",
)
BANDED_STRATEGY_NAMES = (
    "Monday Dip",
    "Weak Close Decent Sznls",
    "SPY QQQ MonFri Reversion",
    "Monthly Weak Close",
)


def _private_risk_payload() -> dict:
    """A full private payload shaped like data/site_risk.json."""
    dates = [
        "2026-07-20", "2026-07-21", "2026-07-22", "2026-07-23", "2026-07-24",
        "2026-07-27", "2026-07-28", "2026-07-29", "2026-07-30", "2026-07-31",
    ]
    ma = [44.1, 45.0, 46.2, 47.4, 48.1, 49.0, 50.3, 51.2, 52.0, 52.9]
    horizons = ["5d", "10d", "21d", "42d", "63d"]

    def table(offset):
        return {
            h: {"1": offset + 15, "2": offset + 5, "3": offset - 5, "5": offset - 25}
            for h in horizons
        }

    return {
        "asof": "2026-07-31",
        "built_at": "2026-07-31 22:05 UTC",
        "spy_last": 771.95,
        "price_ctx": {"regime_label": "Extended uptrend", "drawdown": -0.012},
        "fragility": {"63d": 52.9},
        "regime_mult": 1.45,
        "n_active": 1,
        "dates": dates,
        "spy_series": {"dates": dates, "close": [760 + i for i in range(len(dates))]},
        "sizing_state": {
            "asof": "2026-07-31",
            "basis": "10d MA of 63d dial, append-only PIT parquet",
            "score": 52.9,
            "raw_63d": 62.2,
            "threshold": 50.0,
            "throttle_on": True,
            "gap_to_threshold": -2.9,
            "days_in_state": 4,
            "banded_strategies": [
                {"strategy": name, "bands": [[50.0, 999.0, 0.25]]}
                for name in BANDED_STRATEGY_NAMES
            ],
            "throttled": [
                {"strategy": name, "mult": 0.25} for name in BANDED_STRATEGY_NAMES
            ],
            "pit_start": "2026-07-02",
            "spark": {"dates": dates, "ma": ma, "daily": [v + 8 for v in ma]},
            "episodes": [["2026-07-28", "2026-07-31"]],
            "exposure": {"mult": 1.0, "active_rule": "raw 21d > 50", "asof": "2026-07-31"},
            "sleeve": {"position": "LONG", "since": "2026-06-02", "n_transitions": 3},
        },
        "signals": [
            {"name": "Seasonal Rank Divergence", "on": True, "badge": "FIRING",
             "detail": "risk-off leads"},
            {"name": "Dispersion", "on": False, "badge": "OFF", "detail": ""},
        ],
        "signal_detail": {
            "Seasonal Rank Divergence": {
                "periods": [["2026-07-29", "2026-07-31"]],
                "current": {"value": 12.0, "summary": "defensives lead"},
                "metric": {"label": "Rank gap", "unit": "pp", "decimals": 1,
                           "values": [1.0] * len(dates)},
            },
        },
        "forward_returns": {},
        "vol_kpi": {"vix": 15.2, "vix3m": 17.1, "term_ratio": 0.89},
        "atr_downside": {
            "measure": "low_touch", "atr_period": 14, "mults": [1, 2, 3, 5],
            "horizons": horizons, "data_from": "2001-01-02",
            "data_through": "2026-07-31", "baseline": table(45),
            "signals": {"Seasonal Rank Divergence": {
                "n_events": 139, "n_episodes": 55,
                "episode": table(55), "day": table(50)}},
            "dial": {"value": 52.9, "band": 3, "lo": 49.9, "hi": 55.9,
                     "table": table(58), "band_from": "2017-07-25",
                     "n_by_h": {h: 150 for h in horizons}},
        },
        "nuggets": [
            {"title": "Fragility: neutral and building", "tone": "info",
             "lines": ["The 63d dial sits at 52.9."]},
            {"title": "Book posture: regime multiplier 1.45x", "tone": "good",
             "lines": ["The core-exposure dial says run full size."]},
        ],
    }


def _redacted_risk_payload() -> dict:
    payload = copy.deepcopy(_private_risk_payload())
    for key in REDACTED_SIZING_KEYS:
        payload["sizing_state"].pop(key, None)
    payload["nuggets"] = [
        n for n in payload["nuggets"] if not n["title"].startswith("Book posture")
    ]
    return payload


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_insufficient_sample_is_visible_without_fabricated_return_table():
    script = r'''
const fs = require("fs"), vm = require("vm");
const sandbox = {document: {addEventListener() {}}};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(__RISK_JS__, "utf8"), sandbox);
const html = sandbox.fwdTable("63d", {
  status: "insufficient_sample", min_samples: 5, n_episodes: 4,
  current_score: 87, band_low: 82, band_high: 92,
  returns: {5: null, 10: null, 21: null, 42: null, 63: null},
});
if (!html.includes("Insufficient sample") || !html.includes("4 episodes") ||
    !html.includes("5 completed observations") ||
    ![5, 10, 21, 42, 63].every(w => html.includes(`>${w}d</td>`))) {
  throw new Error("small-sample card is missing or invents return statistics");
}
const partial = sandbox.fwdTable("63d", {
  current_score: 83, n_episodes: 7, band_low: 78, band_high: 88,
  returns: {42: null, 63: null},
});
if (!partial.includes(">42d</td>") || !partial.includes(">63d</td>") ||
    partial.includes("undefined")) throw new Error("sparse windows are hidden or malformed");
'''.replace("__RISK_JS__", json.dumps(str(RISK_JS)))
    subprocess.run([shutil.which("node"), "-e", script], check=True, capture_output=True, text=True)


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
  signals: [{name: "Distribution Dominance", on: false, badge: "OFF", detail: ""}],
  forward_returns: {},
  spy_series: {dates: ["2026-07-14", "2026-07-15"], close: [699, 700]},
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
  Date, Math, Number, String, Object, Array,
};
vm.createContext(sandbox);
vm.runInContext(source, sandbox);
Promise.resolve(ready()).then(() => {
  const html = element("content").innerHTML;
  if (!html.includes("Distribution Dominance")) throw new Error("legacy signal card missing");
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


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_full_history_spark_clips_hero_and_ships_full_main_chart():
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
// ~2y of weekday dates ending 2026-07-28
const dates = [];
let dt = new Date("2024-08-01T00:00:00Z");
const end = new Date("2026-07-28T00:00:00Z");
while (dt <= end) {
  const wd = dt.getUTCDay();
  if (wd !== 0 && wd !== 6) dates.push(dt.toISOString().slice(0, 10));
  dt = new Date(dt.getTime() + 86400e3);
}
const vals = dates.map((_, i) => 40 + (i % 20));
const payload = {
  asof: dates[dates.length - 1], built_at: "2026-07-28 12:00 UTC", spy_last: 700,
  price_ctx: {}, fragility: {"63d": 55}, regime_mult: 1, n_active: 0,
  signals: [], forward_returns: {},
  spy_series: {dates, close: vals.map(v => 600 + v)},
  sizing_state: {
    score: 49, threshold: 50, throttle_on: false, gap_to_threshold: 1,
    days_in_state: 3, banded_strategies: [], throttled: [],
    spark: {dates, ma: vals, daily: vals},
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
  Date, Math, Number, String, Object, Array, Set, parseInt,
};
vm.createContext(sandbox);
vm.runInContext(source, sandbox);
Promise.resolve(ready()).then(() => {
  const main = plots.find(p => p.id === "riskChart");
  if (!main) throw new Error("main chart missing");
  const ma = main.traces.find(t => t.yaxis === "y2");
  if (!ma || ma.x.length !== dates.length) {
    throw new Error("main chart must plot the FULL dial history");
  }
  const spark = plots.find(p => p.id === "sizingSpark");
  if (!spark) throw new Error("hero spark missing");
  if (spark.traces[0].x.length !== 252) {
    throw new Error("hero spark must clip to trailing 252 sessions, got " +
                    spark.traces[0].x.length);
  }
  if (!element("content").innerHTML.includes("dialRangeSeg")) {
    throw new Error("range control markup missing");
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
def test_shared_risk_page_renders_redacted_payload_without_desk_state():
    """The shared teammate build (body data-page="shared-risk") renders the dial
    and its downside tables from a redacted payload and leaks no desk state."""
    script = r"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync(__RISK_JS__, "utf8");
const FIXTURES = __FIXTURES__;

function run(payload, page) {
  const elements = new Map();
  function element(id) {
    if (!elements.has(id)) elements.set(id, {
      id, innerHTML: "", textContent: "",
      on() {}, addEventListener() {}, querySelectorAll() { return []; },
    });
    return elements.get(id);
  }
  let ready;
  const plots = [];
  const asof = [];
  const sandbox = {
    console,
    document: {
      addEventListener(name, fn) { if (name === "DOMContentLoaded") ready = fn; },
      getElementById: element,
      querySelectorAll() { return []; },
      body: { dataset: page ? { page } : {} },
    },
    renderNav() {},
    setAsof(text) { asof.push(text); },
    fetchJSONOrNull: async () => payload,
    fmt: {
      num: (v, d) => v == null ? "" : Number(v).toFixed(d == null ? 2 : d),
      pct: (v, d) => v == null ? "" : (Number(v) * 100).toFixed(d == null ? 1 : d) + "%",
      signed: (v, d) => v == null ? "" : (v >= 0 ? "+" : "") + Number(v).toFixed(d == null ? 2 : d),
    },
    plotLayout: v => v, PLOT_CFG: {},
    Plotly: {
      newPlot(el, traces, layout) { plots.push({id: el.id, traces, layout}); },
      relayout() {},
    },
    Date, Math, Number, String, Object, Array, Set, parseInt,
  };
  vm.createContext(sandbox);
  vm.runInContext(source, sandbox);
  return Promise.resolve(ready()).then(() => ({
    html: element("content").innerHTML, plots, asof,
  }));
}

const fail = m => { throw new Error(m); };

Promise.all([
  run(FIXTURES.redacted, "shared-risk"),
  run(FIXTURES.private, "shared-risk"),
  run(FIXTURES.redacted, null),
  run(FIXTURES.private, null),
  run(null, "shared-risk"),
]).then(([shared, sharedFull, privateRedacted, privateFull, missing]) => {
  const lower = shared.html.toLowerCase();

  // (b) no desk state, no strategy names
  for (const banned of ["throttle", "exposure", "sleeve", "book posture",
                        "sizes live orders", "days in state", "d in state"]) {
    if (lower.includes(banned)) fail(`shared risk page leaked "${banned}"`);
  }
  for (const name of FIXTURES.strategies) {
    if (shared.html.includes(name)) fail(`shared risk page leaked strategy "${name}"`);
  }

  // (c) the dial reading and both ATR tables still render
  if (!shared.html.includes("Market Risk Dial")) fail("shared hero title missing");
  if (!shared.html.includes("52.9")) fail("dial score missing from shared hero");
  if (!shared.html.includes("Main risk dial")) fail("dial KPI tile missing");
  if (!shared.html.includes("atr-dial-card")) fail("ATR dial-band table missing");
  if (!shared.html.includes("Downside when the dial sits here")) fail("dial table caption missing");
  if (!shared.html.includes("fresh Seasonal Rank Divergence trigger")) {
    fail("per-signal ATR table missing under a firing signal");
  }
  if (!shared.asof.length || !String(shared.asof[0]).includes("2026-07-31")) {
    fail("shared page never received an as-of string");
  }
  if (!shared.plots.some(p => p.id === "sizingSpark")) fail("dial spark did not render");
  const spark = shared.plots.find(p => p.id === "sizingSpark");
  if ((spark.layout.shapes || []).length) fail("shared spark must carry no threshold/episode shapes");

  // the SHARED flag alone suppresses desk state even on an unredacted payload
  const fullLower = sharedFull.html.toLowerCase();
  for (const banned of ["throttle", "exposure", "sleeve", "book posture"]) {
    if (fullLower.includes(banned)) fail(`shared mode rendered "${banned}" from a full payload`);
  }

  // a redacted payload must also render cleanly in the PRIVATE build
  if (!privateRedacted.html.includes("52.9")) fail("private build broke on a redacted payload");
  if (privateRedacted.html.toLowerCase().includes("throttle on")) {
    fail("private build invented throttle state from a redacted payload");
  }

  // regression: the private build still shows its desk state in full
  if (!privateFull.html.includes("THROTTLE ON")) fail("private throttle badge regressed");
  if (!privateFull.html.includes("Monday Dip @ 0.25x")) fail("private throttle badges regressed");
  if (!privateFull.html.includes("Exposure leg")) fail("private exposure line regressed");
  if (!privateFull.html.includes("Clean-air SPY sleeve")) fail("private sleeve line regressed");
  if (!privateFull.html.includes("Book posture")) fail("private nuggets regressed");

  // (4) missing payload renders a clear panel, not a blank page
  if (!missing.html.includes("Risk payload unavailable")) fail("missing-payload panel absent");
  if (!missing.html.includes("fetchfail")) fail("missing-payload panel is not styled as a failure");
}).catch(error => { console.error(error); process.exitCode = 1; });
""".replace("__RISK_JS__", json.dumps(str(RISK_JS))).replace(
        "__FIXTURES__",
        json.dumps({
            "private": _private_risk_payload(),
            "redacted": _redacted_risk_payload(),
            "strategies": list(BANDED_STRATEGY_NAMES),
        }),
    )

    subprocess.run(
        [shutil.which("node"), "-e", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
