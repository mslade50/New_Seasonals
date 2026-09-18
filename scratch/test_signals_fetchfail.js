/* Smoke test for signals.js init() — failed-tab banner vs empty-tab caption
   vs populated cards, without a browser. */
const fs = require("fs"), vm = require("vm");

function fakeEl() {
  return {
    className: "", textContent: "", innerHTML: "", children: [],
    appendChild(c) { this.children.push(c); },
    querySelector() { return fakeEl(); },
  };
}

function collectHTML(el) {
  let out = "";
  for (const c of el.children) {
    out += (c.textContent || "") + "|" + (c.innerHTML || "") + "@class=" + (c.className || "") + "\n";
    if (c.children && c.children.length) out += collectHTML(c);
  }
  return out;
}

async function run(payload) {
  const content = fakeEl();
  const sandbox = { console, Date, JSON, Math };
  sandbox.window = sandbox;
  sandbox.document = {
    addEventListener() {},
    getElementById() { return content; },
    createElement() { return fakeEl(); },
  };
  sandbox.renderNav = () => {};
  sandbox.setAsof = () => {};
  sandbox.makeTable = () => {};
  sandbox.fmt = { money: v => "$" + Number(v).toFixed(2), signed: v => String(v) };
  sandbox.fetchJSONOrNull = url => Promise.resolve(url.includes("signals.json") ? payload : null);
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync("site/assets/signals.js", "utf8"), sandbox);
  await sandbox.init();
  return content.innerHTML + "\n" + collectHTML(content);
}

const row = { Symbol: "AAPL", Strategy_Ref: "OLV", Trade_Direction: "LONG",
              Signal_Close: "200", Frozen_ATR: "4", Limit_Price: "199",
              Quantity: "100", Risk_Amt: "1000", Order_Type: "LMT" };

(async () => {
  let ok = true;
  const check = (name, html, wants, rejects = []) => {
    for (const w of wants) {
      const hit = html.includes(w);
      if (!hit) ok = false;
      console.log((hit ? "OK   " : "MISS ") + name + " -> contains " + JSON.stringify(w));
    }
    for (const r of rejects) {
      const absent = !html.includes(r);
      if (!absent) ok = false;
      console.log((absent ? "OK   " : "FAIL ") + name + " -> must NOT contain " + JSON.stringify(r));
    }
  };

  // 1. failed tab: null + errors -> banner, no calm caption for that tab
  let html = await run({
    fetched_at: "2026-07-02",
    tabs: { Order_Staging: null, Overflow: [row] },
    errors: { Order_Staging: "APIError: quota exceeded" },
  });
  check("failed-tab", html,
    ["Liquid — fetch FAILED", "FETCH FAILED", "quota exceeded", "@class=fetchfail", "AAPL"]);

  // 2. genuinely empty tabs -> calm caption, no banner
  html = await run({ fetched_at: "2026-07-02", tabs: { Order_Staging: [], Overflow: [] } });
  check("empty-tabs", html, ["No staged orders on this tab.", "Liquid (0)"], ["FETCH FAILED"]);

  // 3. no payload at all -> existing skip message
  html = await run(null);
  check("no-payload", html, ["No signals payload"]);

  process.exit(ok ? 0 : 1);
})();
