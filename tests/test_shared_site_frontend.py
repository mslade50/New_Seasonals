"""Guards for the shared teammate site's page shells.

Covers the pieces the shared build owns on the frontend side: an identical
overflow nav on every page, the Seasonality/Macro sub-tabs on index.html, and
the fact that no page links a private surface.  The JS behaviour that those
shells depend on (the sub-tab switcher in seasonality.js and the macro renderer)
is exercised under the same Node stub harness the risk-page guards use.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "shared_site"
ASSETS = ROOT / "site" / "assets"

SHARED_PAGES = ("index.html", "risk.html", "heatmaps.html", "correlations.html")
# Primary links stay visible; everything else lives in the .nav-more dropdown,
# mirroring renderNav() on the private site.
PRIMARY_NAV = (("index.html", "Seasonality"), ("risk.html", "Risk"))
OVERFLOW_NAV = (("heatmaps.html", "Heatmaps"), ("correlations.html", "Correlation Heatmaps"))
# Same list build_shared_seasonals.validate_shared_output scans for.
FORBIDDEN = ("execution.html", "orders.html", "signals.html", "portfolio", "/exec-book")

SUMMARY_LABEL = {
    "index.html": "More",
    "risk.html": "More",
    "heatmaps.html": "Heatmaps",
    "correlations.html": "Correlation Heatmaps",
}


def _page(name: str) -> str:
    return (SHARED / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("page", SHARED_PAGES)
def test_every_shared_page_exists_and_avoids_private_surfaces(page):
    html = _page(page).lower()
    assert [v for v in FORBIDDEN if v in html] == []


@pytest.mark.parametrize("page", SHARED_PAGES)
def test_shared_nav_is_identical_and_overflows_like_the_private_site(page):
    html = _page(page)
    nav = re.search(r"<nav>(.*?)</nav>", html, re.S)
    assert nav, f"{page} has no <nav>"
    nav = nav.group(1)

    head, _, tail = nav.partition('<details class="nav-more">')
    assert tail, f"{page} is missing the nav-more dropdown"

    # Primary links, in order, outside the dropdown.
    for href, label in PRIMARY_NAV:
        assert f'href="{href}"' in head and f">{label}</a>" in head
    assert head.index('href="index.html"') < head.index('href="risk.html"')
    for href, _ in OVERFLOW_NAV:
        assert f'href="{href}"' not in head, f"{page} leaves {href} outside the dropdown"

    summary = re.search(r"<summary>(.*?)</summary>", tail).group(1)
    assert summary == SUMMARY_LABEL[page]
    for href, label in OVERFLOW_NAV:
        assert f'href="{href}"' in tail and f">{label}</a>" in tail

    # Exactly one active link, and it is this page.
    active = re.findall(r'<a href="([^"]+)" class="active">', nav)
    assert active == [page], f"{page} marks {active} active"


def test_shared_index_has_lab_and_macro_subtabs_only():
    html = _page("index.html")
    assert 'src="assets/macro_seasonal.js"' in html
    assert 'data-page="shared-seasonality"' in html
    views = re.findall(r'data-seasonal-view="([^"]+)"', html)
    assert views == ["lab", "macro"], views
    assert 'id="seasonal-panel-lab" role="tabpanel"' in html
    assert 'id="seasonal-panel-macro"' in html
    assert re.search(r'id="seasonal-panel-macro"[^>]*hidden', html), "macro panel must start hidden"
    assert 'id="seasonality-lab"' in html
    assert 'id="macro-seasonality"' in html
    # The private Signals & Sizer board is deliberately not shared.
    assert "seasonal-panel-signals" not in html


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_macro_subtab_switches_and_renders_without_private_payloads():
    macro_payload = {
        "asof": "2026-09-17",
        "sznl_asof": "2026-09-16",
        "sznl_available": True,
        "rows": [
            {"ticker": "GLD", "name": "Gold ETF", "ibkr": "GC", "file": "GLD.bin",
             "chart_label": "GLD (GC futures)", "price": 402.5,
             "r5": 91.0, "s5": 88.0, "s10": 62.0, "r20": 44.0, "s21": 12.0,
             "r50": 58.0, "s63": 51.0, "s126": 70.0, "r200": 33.0, "s252": 49.0},
            {"ticker": "VIX", "name": "Volatility Index", "ibkr": "VIX", "file": None,
             "price": 15.4, "r5": None, "s5": None, "s10": None, "r20": None,
             "s21": None, "r50": None, "s63": None, "s126": None, "r200": None,
             "s252": None},
        ],
    }
    script = r"""
const fs = require("fs");
const vm = require("vm");

function stubElement(id, extra) {
  return Object.assign({
    id, innerHTML: "", textContent: "", hidden: false, dataset: {},
    classList: { toggle() {}, add() {}, remove() {} },
    setAttribute() {}, addEventListener() {},
    querySelector() { return { addEventListener() {} }; },
    querySelectorAll() { return []; },
  }, extra || {});
}

const panels = {
  "seasonal-panel-lab": stubElement("seasonal-panel-lab"),
  "seasonal-panel-macro": stubElement("seasonal-panel-macro", { hidden: true }),
};
const macroRoot = stubElement("macro-seasonality");
// Enough of querySelectorAll for the lazy-chart wiring: the renderer looks up
// the chart cards it just wrote into innerHTML.
macroRoot.querySelectorAll = sel => (
  sel === ".macro-chart-card" && macroRoot.innerHTML.includes('id="macro-chart-0"')
    ? [stubElement("macro-chart-0", { dataset: { file: "GLD.bin", ticker: "GLD", label: "GLD" } })]
    : []);
const byId = { ...panels, "macro-seasonality": macroRoot };
const buttons = ["lab", "macro"].map(view => stubElement(`seasonal-tab-${view}`, {
  dataset: { seasonalView: view },
}));
const observed = [];
let macroPayload = __MACRO__;

const sandbox = {
  console,
  document: {
    addEventListener() {},
    dispatchEvent() {},
    getElementById: id => byId[id] || null,
    querySelectorAll: sel => (sel === "[data-seasonal-view]" ? buttons : []),
  },
  window: {
    history: { replaceState() {} },
    location: { hash: "", pathname: "/index.html", search: "" },
  },
  CustomEvent: class { constructor(name, init) { this.type = name; this.detail = init && init.detail; } },
  IntersectionObserver: class {
    constructor(cb) { this.cb = cb; }
    observe(target) { observed.push(target); }
    unobserve() {}
  },
  fetchJSONOrNull: async () => macroPayload,
  fetch: async () => { throw new Error("no network in this harness"); },
  plotLayout: v => v, PLOT_CFG: {}, Plotly: { newPlot() {} },
  Date, Math, Number, String, Object, Array, Set, Map, JSON, parseInt, parseFloat,
  isNaN, Promise, Error, TextDecoder: globalThis.TextDecoder,
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(__SEASONALITY_JS__, "utf8"), sandbox);
vm.runInContext(fs.readFileSync(__MACRO_JS__, "utf8"), sandbox);

const fail = m => { throw new Error(m); };

// 1. The two-button shared tab set must not fall back to the private
//    "signals" view (setSeasonalView defaults to it when the view is absent).
sandbox.setSeasonalView("macro");
if (panels["seasonal-panel-macro"].hidden) fail("macro panel stayed hidden after switching to it");
if (!panels["seasonal-panel-lab"].hidden) fail("lab panel stayed visible after switching away");
sandbox.setSeasonalView("lab");
if (panels["seasonal-panel-lab"].hidden) fail("lab panel did not come back");
if (!panels["seasonal-panel-macro"].hidden) fail("macro panel stayed visible");

// 2. The macro renderer fills the shared panel from data/seasonality/macro.json.
Promise.resolve(sandbox.initMacroSeasonality()).then(() => {
  const html = macroRoot.innerHTML;
  if (!html.includes("GLD")) fail("macro table is missing its rows");
  if (!html.includes("Gold ETF")) fail("macro table is missing instrument names");
  if (!html.includes("Sznl_21")) fail("macro table header is missing");
  if (!html.includes("2026-09-17")) fail("macro asof is missing");
  if (!html.includes("macro-chart-0")) fail("chart card for a priced row is missing");
  if (!html.includes("VIX")) fail("row without price history should still be listed");
  if (observed.length !== 1) fail(`expected 1 lazy chart card, got ${observed.length}`);

  // 3. A build without the macro payload degrades to a visible notice.
  macroPayload = null;
  macroRoot.innerHTML = "";
  return Promise.resolve(sandbox.initMacroSeasonality()).then(() => {
    if (!macroRoot.innerHTML.includes("fetchfail")) fail("missing macro payload renders blank");
  });
}).catch(error => { console.error(error); process.exitCode = 1; });
""".replace("__SEASONALITY_JS__", json.dumps(str(ASSETS / "seasonality.js"))) \
   .replace("__MACRO_JS__", json.dumps(str(ASSETS / "macro_seasonal.js"))) \
   .replace("__MACRO__", json.dumps(macro_payload))

    subprocess.run(
        [shutil.which("node"), "-e", script],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
