"""Private-site guards for the similar-fragility candlestick explorer."""

import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_risk_json import build_price_explorer


ROOT = Path(__file__).parents[1]
RISK_JS = ROOT / "site" / "assets" / "risk.js"


def test_price_payload_aligns_cached_assets_to_spy_calendar(tmp_path):
    dates = pd.bdate_range("2026-01-02", periods=24)
    spy = pd.DataFrame(
        {
            "Open": range(100, 124),
            "High": range(102, 126),
            "Low": range(99, 123),
            "Close": range(101, 125),
        },
        index=dates,
    )
    cached = []
    for ticker, offset in [("QQQ", 200), ("IWM", 300)]:
        for i, date in enumerate(dates):
            cached.append({
                "ticker": ticker, "date": date,
                "Open": offset + i, "High": offset + i + 2,
                "Low": offset + i - 1, "Close": offset + i + 1,
            })
    cache_path = tmp_path / "master_prices.parquet"
    pd.DataFrame(cached).to_parquet(cache_path)

    payload = build_price_explorer(spy, dates, cache_path)

    assert payload["assets"][:3] == ["SPY", "QQQ", "IWM"]
    assert len(payload["series"]["SPY"]["open"]) == len(dates)
    assert payload["series"]["QQQ"]["close"][0] == 201.0
    assert payload["series"]["IWM"]["high"][-1] == 325.0


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_private_risk_page_renders_fixed_match_lines_on_switchable_candles():
    script = r"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync(__RISK_JS__, "utf8");
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    id, innerHTML: "", textContent: "",
    value: id === "matchHorizon" ? "63d" : id === "matchAsset" ? "SPY" : "",
    listeners: {}, plotListeners: {},
    addEventListener(name, fn) { this.listeners[name] = fn; },
    on(name, fn) { this.plotListeners[name] = fn; },
    querySelectorAll() { return []; },
  });
  return elements.get(id);
}
let ready;
const dates = ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"];
const candles = base => ({
  open: [base, base + 1, base + 2, base + 3],
  high: [base + 2, base + 3, base + 4, base + 5],
  low: [base - 1, base, base + 1, base + 2],
  close: [base + 1, base + 2, base + 3, base + 4],
});
const payload = {
  asof: "2026-01-07", built_at: "2026-01-07 22:00 UTC", spy_last: 104,
  price_ctx: {}, fragility: {}, regime_mult: 1, n_active: 0, signals: [], dates,
  forward_returns: {
    "63d": {current_score: 50, band_low: 45, band_high: 55,
      n_episodes: 2, episode_dates: [dates[1], dates[3]], returns: {}},
    "21d": {current_score: 40, band_low: 35, band_high: 45,
      n_episodes: 1, episode_dates: [dates[2]], returns: {}},
  },
  price_explorer: {
    assets: ["SPY", "QQQ"],
    series: {SPY: candles(100), QQQ: candles(200)},
  },
  drawdown_iv: {rows_by_horizon: {}, horizons: [], n_episodes: 2},
};
const plots = [];
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
  plotLayout: value => value, PLOT_CFG: {},
  Plotly: {
    react(el, traces, layout, config) {
      plots.push({id: el.id, traces, layout, config});
      return Promise.resolve();
    },
    newPlot() { return Promise.resolve(); },
    relayout() { return Promise.resolve(); },
  },
  Date, Math, Number, String, Object, Array, Set, Map, Promise,
};
vm.createContext(sandbox);
vm.runInContext(source, sandbox);
Promise.resolve(ready()).then(() => {
  const html = element("content").innerHTML;
  const fwd = html.lastIndexOf("Forward returns at similar fragility readings");
  const explorer = html.lastIndexOf("Similar-fragility match explorer");
  if (explorer <= fwd) throw new Error("match explorer is not the final risk section");

  const initial = plots[plots.length - 1];
  if (!initial || initial.id !== "matchPriceChart") throw new Error("match chart missing");
  if (initial.traces[0].type !== "candlestick") throw new Error("price trace is not candlestick");
  const initialLines = initial.layout.shapes.map(shape => shape.x0).join(",");
  if (initialLines !== `${dates[1]},${dates[3]}`) throw new Error("wrong 63d overlays");
  if (!initial.config.displayModeBar || !initial.config.scrollZoom) throw new Error("zoom controls disabled");

  element("matchAsset").value = "QQQ";
  element("matchAsset").listeners.change();
  const switched = plots[plots.length - 1];
  if (switched.traces[0].open[0] !== 200) throw new Error("asset did not switch");
  if (switched.layout.shapes.map(shape => shape.x0).join(",") !== initialLines) {
    throw new Error("asset switch changed match overlays");
  }

  element("matchHorizon").value = "21d";
  element("matchHorizon").listeners.change();
  const horizon = plots[plots.length - 1];
  if (horizon.layout.shapes.length !== 1 || horizon.layout.shapes[0].x0 !== dates[2]) {
    throw new Error("horizon switch did not use its match set");
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
