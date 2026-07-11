/* portfolio.js — client-side filterable analytics over the trade ledger.

   Sizing basis: FLAT $750k (PnL_flat). Per-trade dollars are additive, so any
   filtered subset yields an exact equity curve / Sharpe / DD on this basis.

   Daily-curve resolution:
   - strategy / tier / date filters  -> exact daily MTM (sum of per-strategy
     daily series computed server-side)
   - direction / ticker filters      -> fallback: realized PnL booked on exit
     date (step curve; vol metrics approximate). A notice badge appears.
*/
"use strict";

const S = {
  meta: null, trades: [], sd: null, positions: null, exposure: null, corr: null,
  dateIdx: [],            // sd.dates as strings
  dateToI: new Map(),
  f: { strategies: null, tier: "All", dir: "All", preset: "All", from: null, to: null, tickerQ: "" },
  sizing: "scaled",       // "scaled" (compounds with equity) | "flat" (fixed base)
  lev: 1.0,               // portfolio leverage multiplier
  mult: new Map(),        // strategy -> risk multiplier (default 1)
  midScalar: 1.0,         // midterm-year (year%4==2) risk scalar; OVS exempt
                          // (its 0.75x midterm tilt is already in the ledger)
  midMask: null,          // Uint8Array aligned to dateIdx: 1 = midterm year
  nativeBps: new Map(),   // strategy -> median Risk_bps from the ledger
  rangeTouched: false,    // user has interacted with the Range control
  tradeLogTable: null,
  fragility: null,        // fragility.json payload (dial series, 5d-smoothed basis)
  stopfills: null,        // stopfills.json (stop-exit fill quality, flat $750k)
  sfRows: null,           // stopfills.trades unpacked to row objects
  sfTable: null,          // makeTable handle for the stop-tail table
  drawdowns: null,        // drawdowns.json (full-book flat episodes)
  ddaUW: null,            // cached full-book underwater curve (from S.sd)
  sectorRisk: null,       // sector_risk.json (exposure timeline + gate telemetry)
  gateLab: null,          // gate_lab.json (sector-gate counterfactual)
  gateBlockedRows: [],    // blocked trades as trade-log-shaped rows (GateBlocked=true)
  showBlocked: false,     // merge gate-blocked trades into the filtered analytics
  extLab: null,           // ext_lab.json (OVS hold-extension counterfactual)
  extById: new Map(),     // trade_id -> rebooked row (OvsExt=true)
  showOvsExt: false,      // swap rebooked OVS exits into the filtered analytics
  mtmDates: null,         // trade_mtm.json calendar (B-days, payload-local)
  mtmDateToI: new Map(),  // date -> index into mtmDates
  mtmMain: new Map(),     // trade_id -> [startIdx, pnlVector] (flat $750k)
  mtmExt: new Map(),      // trade_id -> vector for the rebooked T+5 exit
  mtmGate: new Map(),     // Strategy|Tier|Ticker|SignalDate -> vector (blocked rows)
  // fragility sizing adjuster (off by default = today's exact-curve behavior).
  // step:  mult = boost below thr, floor at/above thr.
  // ramp:  live-style — boost at 0 -> 1.0 at thr, then linear down to floor at 100
  //        (dial 63d / ma 10 / thr 25 / floor 0.10 / boost 1.25 = the live ramp).
  frag: { dial: "off", ma: 10, thr: 50, floor: 0.5, boost: 1.0, shape: "step" },
};
const MID_EXEMPT = new Set(["Overbot Vol Spike"]);
const FRAG_EXEMPT = MID_EXEMPT;   // live frag mult applies to non-OVS orders only

function isMidYearStr(d) { return d ? (parseInt(d.slice(0, 4), 10) % 4) === 2 : false; }

/* full dollar multiplier for a trade under the current dials */
function tradeMult(t) {
  let m = multFor(t.Strategy);
  if (S.midScalar !== 1 && !MID_EXEMPT.has(t.Strategy) && isMidYearStr(t.Entry_Date || t.Signal_Date)) {
    m *= S.midScalar;
  }
  if (fragActive() && !FRAG_EXEMPT.has(t.Strategy)) m *= fragMultForTrade(t);
  return m;
}

/* ---- fragility sizing adjuster ---- */
function fragActive() { return !!(S.fragility && S.frag.dial !== "off"); }

let _fragCache = { key: "", vals: null };
/* dial series with the trailing MA applied (nulls skipped, live parity:
   dropna().rolling(ma, min_periods=1).mean()) */
function fragSeries() {
  const { dial, ma } = S.frag;
  const key = dial + "|" + ma;
  if (_fragCache.key === key) return _fragCache.vals;
  const raw = S.fragility.dials[dial];
  if (!raw) return null;
  const n = raw.length, vals = new Array(n).fill(null);
  const win = [];
  let sum = 0;
  for (let i = 0; i < n; i++) {
    if (raw[i] != null) {
      win.push(raw[i]); sum += raw[i];
      if (win.length > ma) sum -= win.shift();
    }
    if (win.length) vals[i] = sum / win.length;
  }
  _fragCache = { key, vals };
  return vals;
}

/* score as of a signal date: last reading on/before it, none if > 7 days stale */
function fragScoreFor(dateStr) {
  const dates = S.fragility.dates;
  const i = upperBound(dates, dateStr) - 1;
  if (i < 0) return null;
  if ((Date.parse(dateStr) - Date.parse(dates[i])) > 7 * 86400e3) return null;
  const vals = fragSeries();
  return vals ? vals[i] : null;
}

function fragMultOf(score) {
  if (score == null) return 1.0;          // no data -> native size (live parity)
  const { thr, floor, boost, shape } = S.frag;
  if (shape === "step") return score >= thr ? floor : boost;
  // ramp: boost at 0 -> 1.0 at thr, then linear 1.0 -> floor at 100
  if (score <= thr) return thr > 0 ? boost - (score / thr) * (boost - 1) : boost;
  return Math.max(floor, 1 - ((score - thr) / (100 - thr)) * (1 - floor));
}

function fragMultForTrade(t) {
  const d = t.Signal_Date || t.Entry_Date;
  return d ? fragMultOf(fragScoreFor(d)) : 1.0;
}
// Portfolio-level math runs on the ledger's $750k flat allocation; equity
// curves and portfolio dollars are DISPLAYED from a $10k start (returns,
// Sharpe, DD are scale-invariant). Trade-level dollars (trade log, open
// positions, expectancy, per-strategy PnL) stay at the $750k allocation.
const START_EQ = 750000;
const DISPLAY_EQ = 10000;
const DSCALE = DISPLAY_EQ / START_EQ;

function multFor(strat) {
  const m = S.mult.get(strat);
  return (m == null ? 1 : m) * S.lev;
}
function anyMultActive() {
  if (S.lev !== 1 || S.midScalar !== 1) return true;
  for (const v of S.mult.values()) if (v !== 1) return true;
  return false;
}

document.addEventListener("DOMContentLoaded", init);

async function init() {
  renderNav("index.html");
  try {
    const [meta, trades] = await Promise.all([
      fetchJSON("data/meta.json"), fetchJSON("data/trades.json")]);
    S.meta = meta;
    S.trades = rowsFromColumnar(trades);
    const [sd, pos, exp, corr, frag, sf, dda, sr, gl, xl, tmm] = await Promise.all([
      meta.payloads.strategy_daily ? fetchJSONOrNull("data/strategy_daily.json") : null,
      meta.payloads.positions ? fetchJSONOrNull("data/positions.json") : null,
      meta.payloads.exposure ? fetchJSONOrNull("data/exposure.json") : null,
      meta.payloads.correlation ? fetchJSONOrNull("data/correlation.json") : null,
      meta.payloads.fragility ? fetchJSONOrNull("data/fragility.json") : null,
      meta.payloads.stopfills ? fetchJSONOrNull("data/stopfills.json") : null,
      meta.payloads.drawdowns ? fetchJSONOrNull("data/drawdowns.json") : null,
      meta.payloads.sector_risk ? fetchJSONOrNull("data/sector_risk.json") : null,
      meta.payloads.gate_lab ? fetchJSONOrNull("data/gate_lab.json") : null,
      meta.payloads.ext_lab ? fetchJSONOrNull("data/ext_lab.json") : null,
      meta.payloads.trade_mtm ? fetchJSONOrNull("data/trade_mtm.json") : null,
    ]);
    S.sd = sd; S.positions = pos; S.exposure = exp; S.corr = corr; S.fragility = frag;
    S.stopfills = sf; S.drawdowns = dda; S.sectorRisk = sr; S.gateLab = gl; S.extLab = xl;
    if (sf && sf.trades) S.sfRows = rowsFromColumnar(sf.trades);
    if (gl && gl.strategies) {
      for (const st of gl.strategies) {
        if (!st.blocked_trades) continue;
        for (const r of rowsFromColumnar(st.blocked_trades)) {
          r.GateBlocked = true;
          S.gateBlockedRows.push(r);
        }
      }
    }
    if (xl && xl.modified_trades) {
      for (const r of rowsFromColumnar(xl.modified_trades)) {
        r.OvsExt = true;
        S.extById.set(r.trade_id, r);
      }
    }
    if (tmm && tmm.main && tmm.dates) {
      S.mtmDates = tmm.dates;
      tmm.dates.forEach((d, i) => S.mtmDateToI.set(d, i));
      const g = tmm.main;
      for (let i = 0; i < g.trade_id.length; i++)
        S.mtmMain.set(g.trade_id[i], [g.start[i], g.pnl[i]]);
      if (tmm.ext) for (let i = 0; i < tmm.ext.trade_id.length; i++)
        S.mtmExt.set(tmm.ext.trade_id[i], [tmm.ext.start[i], tmm.ext.pnl[i]]);
      if (tmm.gate) for (let i = 0; i < tmm.gate.key.length; i++)
        S.mtmGate.set(tmm.gate.key[i], [tmm.gate.start[i], tmm.gate.pnl[i]]);
    }
    if (sd) {
      S.dateIdx = sd.dates;
      sd.dates.forEach((d, i) => S.dateToI.set(d, i));
      S.midMask = new Uint8Array(sd.dates.length);
      sd.dates.forEach((d, i) => { if (isMidYearStr(d)) S.midMask[i] = 1; });
    }
    setAsof(`ledger thru ${meta.ledger_last_signal} · built ${meta.built_at}`);
    document.getElementById("subtitle").textContent =
      `${meta.n_trades.toLocaleString()} trades · ${meta.n_tickers} tickers · ` +
      `${meta.date_min} to ${meta.date_max} · $750k base, filter-exact recompute`;
    computeNativeBps();
    buildFilterBar();
    buildSizingSeg();
    buildRiskPanel();
    renderStatic();
    apply();
  } catch (e) {
    document.getElementById("kpis").innerHTML = `<div class="err">Failed to load data: ${e.message}</div>`;
    console.error(e);
  }
}

/* ================= filters ================= */
function allStrategyNames() {
  return [...new Set(S.meta.strategies.map(s => s.Strategy))].sort();
}

function buildFilterBar() {
  const el = document.getElementById("filters");
  el.innerHTML = "";

  // strategy multiselect
  const msel = document.createElement("div");
  msel.className = "msel";
  const mbtn = document.createElement("button");
  mbtn.className = "btn"; mbtn.id = "stratBtn";
  msel.appendChild(mbtn);
  const panel = document.createElement("div");
  panel.className = "panel"; panel.style.display = "none";
  msel.appendChild(panel);
  el.appendChild(msel);

  const names = allStrategyNames();
  S.f.strategies = new Set(names);
  const acts = document.createElement("div");
  acts.className = "acts";
  acts.innerHTML = `<button class="btn ghost" data-a="all">All</button>
                    <button class="btn ghost" data-a="none">None</button>
                    <span class="cnt"></span>`;
  panel.appendChild(acts);
  const boxes = [];
  for (const n of names) {
    const row = document.createElement("label");
    row.className = "row";
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.checked = true; cb.value = n;
    cb.addEventListener("change", () => {
      cb.checked ? S.f.strategies.add(n) : S.f.strategies.delete(n);
      syncStratBtn(); apply();
    });
    boxes.push(cb);
    row.appendChild(cb);
    row.appendChild(document.createTextNode(" " + n));
    panel.appendChild(row);
  }
  acts.querySelector('[data-a="all"]').addEventListener("click", () => {
    S.f.strategies = new Set(names); boxes.forEach(b => b.checked = true); syncStratBtn(); apply();
  });
  acts.querySelector('[data-a="none"]').addEventListener("click", () => {
    S.f.strategies.clear(); boxes.forEach(b => b.checked = false); syncStratBtn(); apply();
  });
  mbtn.addEventListener("click", () => {
    panel.style.display = panel.style.display === "none" ? "block" : "none";
  });
  document.addEventListener("click", ev => {
    if (!msel.contains(ev.target)) panel.style.display = "none";
  });
  function syncStratBtn() {
    mbtn.textContent = `Strategies (${S.f.strategies.size}/${names.length})`;
  }
  syncStratBtn();

  // tier + direction segments
  el.appendChild(makeSeg("Tier", ["All", "Liquid", "Overflow"], v => { S.f.tier = v; apply(); }));
  el.appendChild(makeSeg("Direction", ["All", "Long", "Short"], v => { S.f.dir = v; apply(); }));

  // date presets
  const presets = ["All", "10Y", "5Y", "3Y", "1Y", "YTD"];
  const presetSeg = makeSeg("Range", presets, v => {
    S.f.preset = v;
    S.rangeTouched = true;
    const max = S.meta.date_max;
    const maxD = new Date(max + "T00:00:00Z");
    let from = null;
    if (v === "YTD") from = max.slice(0, 4) + "-01-01";
    else if (v !== "All") {
      const yrs = parseInt(v);
      const d = new Date(maxD); d.setUTCFullYear(d.getUTCFullYear() - yrs);
      from = d.toISOString().slice(0, 10);
    }
    S.f.from = from; S.f.to = null;
    fromInp.value = from || ""; toInp.value = "";
    apply();
  });
  el.appendChild(presetSeg);

  const fromInp = document.createElement("input");
  fromInp.type = "date"; fromInp.title = "From";
  const toInp = document.createElement("input");
  toInp.type = "date"; toInp.title = "To";
  fromInp.addEventListener("change", () => { S.f.from = fromInp.value || null; S.f.preset = "Custom"; S.rangeTouched = true; markSeg(presetSeg, null); apply(); });
  toInp.addEventListener("change", () => { S.f.to = toInp.value || null; S.f.preset = "Custom"; S.rangeTouched = true; markSeg(presetSeg, null); apply(); });
  el.appendChild(fromInp); el.appendChild(toInp);

  // ticker search
  const tInp = document.createElement("input");
  tInp.type = "text"; tInp.placeholder = "Tickers (e.g. NVDA, XLE)";
  tInp.style.width = "170px";
  let tmr = null;
  tInp.addEventListener("input", () => {
    clearTimeout(tmr);
    tmr = setTimeout(() => { S.f.tickerQ = tInp.value.trim(); apply(); }, 250);
  });
  el.appendChild(tInp);

  // gate-blocked counterfactual toggle (only when gate_lab.json shipped)
  let gateCb = null;
  if (S.gateBlockedRows.length) {
    const gLbl = document.createElement("label");
    gLbl.style.cssText = "display:inline-flex;align-items:center;gap:5px;cursor:pointer";
    gLbl.title = "Counterfactual: include the trades the sector-loss gate blocked " +
      "(outcomes from a no-gate engine rerun). " +
      (S.mtmDates ? "Curve stays on the exact per-trade MTM basis. "
                  : "Equity curve drops to the realized-at-exit basis while on. ") +
      "Off = the book as traded.";
    gateCb = document.createElement("input");
    gateCb.type = "checkbox";
    gateCb.addEventListener("change", () => { S.showBlocked = gateCb.checked; apply(); });
    gLbl.appendChild(gateCb);
    gLbl.appendChild(document.createTextNode(
      ` All trades (+${S.gateBlockedRows.length} gate-blocked)`));
    el.appendChild(gLbl);
  }

  // OVS hold-extension counterfactual toggle (only when ext_lab.json shipped)
  let extCb = null;
  if (S.extById.size) {
    const xLbl = document.createElement("label");
    xLbl.style.cssText = "display:inline-flex;align-items:center;gap:5px;cursor:pointer";
    xLbl.title = "Counterfactual: OVS trades losing at the T+2 time exit hold to T+5 " +
      "(2-ATR target stays live). Swaps the rebooked exits into every filtered metric. " +
      (S.mtmDates ? "Curve stays on the exact per-trade MTM basis. "
                  : "Equity curve drops to the realized-at-exit basis while on. ") +
      "Off = the book as traded.";
    extCb = document.createElement("input");
    extCb.type = "checkbox";
    extCb.addEventListener("change", () => { S.showOvsExt = extCb.checked; apply(); });
    xLbl.appendChild(extCb);
    xLbl.appendChild(document.createTextNode(
      ` OVS losers to T+5 (${S.extById.size} rebooked)`));
    el.appendChild(xLbl);
  }

  // reset
  const rb = document.createElement("button");
  rb.className = "btn ghost"; rb.textContent = "Reset";
  rb.addEventListener("click", () => {
    S.f = { strategies: new Set(names), tier: "All", dir: "All", preset: "All", from: null, to: null, tickerQ: "" };
    S.rangeTouched = false;
    S.showBlocked = false;
    if (gateCb) gateCb.checked = false;
    S.showOvsExt = false;
    if (extCb) extCb.checked = false;
    boxes.forEach(b => b.checked = true); syncStratBtn();
    fromInp.value = ""; toInp.value = ""; tInp.value = "";
    el.querySelectorAll(".seg").forEach(seg => {
      seg.querySelectorAll("button").forEach((b, i) => b.classList.toggle("on", i === 0));
    });
    apply();
  });
  el.appendChild(rb);
}

function makeSeg(label, values, onpick) {
  const box = document.createElement("span");
  box.appendChild(Object.assign(document.createElement("label"), { textContent: label }));
  const seg = document.createElement("span");
  seg.className = "seg";
  values.forEach((v, i) => {
    const b = document.createElement("button");
    b.textContent = v;
    if (i === 0) b.classList.add("on");
    b.addEventListener("click", () => {
      seg.querySelectorAll("button").forEach(x => x.classList.remove("on"));
      b.classList.add("on");
      onpick(v);
    });
    seg.appendChild(b);
  });
  box.appendChild(seg);
  box.seg = seg;
  return box;
}
function markSeg(segBox, value) {
  segBox.seg.querySelectorAll("button").forEach(b =>
    b.classList.toggle("on", b.textContent === value));
}

/* ================= sizing toggle + risk panel ================= */
function buildSizingSeg() {
  const host = document.getElementById("sizingSeg");
  const seg = document.createElement("span");
  seg.className = "seg";
  for (const [val, label] of [["scaled", "Scaled with portfolio"], ["flat", "Flat $750k"]]) {
    const b = document.createElement("button");
    b.textContent = label;
    if (val === S.sizing) b.classList.add("on");
    b.addEventListener("click", () => {
      S.sizing = val;
      seg.querySelectorAll("button").forEach(x => x.classList.remove("on"));
      b.classList.add("on");
      apply();
    });
    seg.appendChild(b);
  }
  host.appendChild(seg);
}

function computeNativeBps() {
  const byStrat = new Map();
  for (const t of S.trades) {
    if (t.Risk_bps == null) continue;
    if (!byStrat.has(t.Strategy)) byStrat.set(t.Strategy, []);
    byStrat.get(t.Strategy).push(t.Risk_bps);
  }
  for (const [s, arr] of byStrat) {
    arr.sort((a, b) => a - b);
    S.nativeBps.set(s, arr[Math.floor(arr.length / 2)]);
  }
}

function buildRiskPanel() {
  const names = allStrategyNames();
  for (const n of names) if (!S.mult.has(n)) S.mult.set(n, 1);

  // leverage + midterm-scalar slider row
  const levRow = document.getElementById("levRow");
  levRow.className = "levrow";
  levRow.innerHTML = `<label>Portfolio leverage</label>
    <input type="range" id="levSlider" min="0" max="3" step="0.05" value="1">
    <span class="lv" id="levVal">1.00x</span>
    <label title="Scales risk only in presidential midterm years (year%4==2: 2002, 2006, ... 2026). OVS exempt — its 0.75x midterm tilt is already baked into the ledger.">Midterm-yr scalar</label>
    <input type="range" id="midSlider" min="0.5" max="1.25" step="0.05" value="1">
    <span class="lv" id="midVal">1.00x</span>
    <button class="btn ghost" id="riskReset">Reset all</button>`;
  const slider = levRow.querySelector("#levSlider");
  const levVal = levRow.querySelector("#levVal");
  const midSlider = levRow.querySelector("#midSlider");
  const midVal = levRow.querySelector("#midVal");
  let tmr = null;
  slider.addEventListener("input", () => {
    S.lev = +slider.value;
    levVal.textContent = S.lev.toFixed(2) + "x";
    syncRiskUI();
    clearTimeout(tmr);
    tmr = setTimeout(apply, 160);
  });
  midSlider.addEventListener("input", () => {
    S.midScalar = +midSlider.value;
    midVal.textContent = S.midScalar.toFixed(2) + "x";
    syncRiskUI();
    clearTimeout(tmr);
    tmr = setTimeout(apply, 160);
  });

  // fragility sizing adjuster row
  const fragRow = document.getElementById("fragRow");
  let fragEls = null;
  if (S.fragility) {
    fragRow.className = "levrow";
    fragRow.innerHTML = `
      <label title="Sizes each trade by the risk-dial fragility score on its signal date (10d-MA basis lives in daily_scan). OVS exempt, matching live. Trades before ${S.fragility.dates[0]} keep native size (no score data).">Fragility sizing</label>
      <span class="seg" id="fragDialSeg"></span>
      <span class="seg" id="fragShapeSeg"></span>
      <label>MA</label>
      <input type="number" id="fragMa" min="1" max="42" step="1" value="10" style="width:52px">
      <label>Threshold</label>
      <input type="range" id="fragThr" min="0" max="100" step="1" value="50">
      <span class="lv" id="fragThrVal">50</span>
      <label>Floor</label>
      <input type="range" id="fragFloor" min="0" max="1" step="0.05" value="0.5">
      <span class="lv" id="fragFloorVal">0.50x</span>
      <label>Boost</label>
      <input type="range" id="fragBoost" min="1" max="1.5" step="0.05" value="1">
      <span class="lv" id="fragBoostVal">1.00x</span>
      <span class="cap" id="fragInfo"></span>`;
    const mkSeg = (hostId, values, cur, onpick) => {
      const seg = fragRow.querySelector(hostId);
      for (const v of values) {
        const b = document.createElement("button");
        b.textContent = v;
        if (v === cur) b.classList.add("on");
        b.addEventListener("click", () => {
          seg.querySelectorAll("button").forEach(x => x.classList.remove("on"));
          b.classList.add("on");
          onpick(v);
          syncRiskUI();
          apply();
        });
        seg.appendChild(b);
      }
      return seg;
    };
    mkSeg("#fragDialSeg", ["Off", ...Object.keys(S.fragility.dials)], "Off",
          v => { S.frag.dial = v === "Off" ? "off" : v; });
    mkSeg("#fragShapeSeg", ["Step", "Ramp"], "Step",
          v => { S.frag.shape = v.toLowerCase(); });
    fragEls = {
      ma: fragRow.querySelector("#fragMa"),
      thr: fragRow.querySelector("#fragThr"), thrVal: fragRow.querySelector("#fragThrVal"),
      floor: fragRow.querySelector("#fragFloor"), floorVal: fragRow.querySelector("#fragFloorVal"),
      boost: fragRow.querySelector("#fragBoost"), boostVal: fragRow.querySelector("#fragBoostVal"),
      info: fragRow.querySelector("#fragInfo"),
    };
    fragEls.ma.addEventListener("change", () => {
      let v = parseInt(fragEls.ma.value, 10);
      if (!isFinite(v) || v < 1) v = 1;
      if (v > 42) v = 42;
      fragEls.ma.value = v;
      S.frag.ma = v;
      syncRiskUI();
      if (fragActive()) apply();
    });
    const wireSlider = (el, valEl, field, fmtV) => {
      el.addEventListener("input", () => {
        S.frag[field] = +el.value;
        valEl.textContent = fmtV(+el.value);
        syncRiskUI();
        if (!fragActive()) return;
        clearTimeout(tmr);
        tmr = setTimeout(apply, 160);
      });
    };
    wireSlider(fragEls.thr, fragEls.thrVal, "thr", v => String(v));
    wireSlider(fragEls.floor, fragEls.floorVal, "floor", v => v.toFixed(2) + "x");
    wireSlider(fragEls.boost, fragEls.boostVal, "boost", v => v.toFixed(2) + "x");
  }

  // per-strategy multiplier grid
  const grid = document.getElementById("multGrid");
  grid.innerHTML = "";
  const rows = [];
  for (const n of names) {
    const row = document.createElement("div");
    row.className = "multrow";
    const native = S.nativeBps.get(n);
    row.innerHTML = `<span class="nm" title="${n}">${n}</span>
      <input type="number" min="0" max="5" step="0.1" value="1">
      <span class="bps"></span>`;
    const inp = row.querySelector("input");
    inp.addEventListener("change", () => {
      let v = parseFloat(inp.value);
      if (!isFinite(v) || v < 0) v = 0;
      if (v > 5) v = 5;
      inp.value = v;
      S.mult.set(n, v);
      syncRiskUI();
      apply();
    });
    rows.push({ name: n, inp, bpsEl: row.querySelector(".bps"), native });
    grid.appendChild(row);
  }

  levRow.querySelector("#riskReset").addEventListener("click", () => {
    S.lev = 1; slider.value = "1"; levVal.textContent = "1.00x";
    S.midScalar = 1; midSlider.value = "1"; midVal.textContent = "1.00x";
    for (const r of rows) { r.inp.value = "1"; S.mult.set(r.name, 1); }
    if (fragEls) {
      S.frag = { dial: "off", ma: 10, thr: 50, floor: 0.5, boost: 1.0, shape: "step" };
      fragEls.ma.value = "10";
      fragEls.thr.value = "50"; fragEls.thrVal.textContent = "50";
      fragEls.floor.value = "0.5"; fragEls.floorVal.textContent = "0.50x";
      fragEls.boost.value = "1"; fragEls.boostVal.textContent = "1.00x";
      for (const segId of ["#fragDialSeg", "#fragShapeSeg"]) {
        const seg = fragRow.querySelector(segId);
        seg.querySelectorAll("button").forEach((b, i) =>
          b.classList.toggle("on", i === 0));
      }
    }
    syncRiskUI();
    apply();
  });

  function syncRiskUI() {
    for (const r of rows) {
      const m = S.mult.get(r.name);
      r.inp.classList.toggle("tweaked", m !== 1);
      if (r.native != null) {
        const eff = r.native * m * S.lev;
        r.bpsEl.textContent = `${r.native.toFixed(0)} -> ${eff.toFixed(0)} bps`;
      } else r.bpsEl.textContent = "";
    }
    const bits = [];
    if (S.lev !== 1) bits.push(`${S.lev.toFixed(2)}x leverage`);
    if (S.midScalar !== 1) bits.push(`midterm ${S.midScalar.toFixed(2)}x (ex-OVS)`);
    if ([...S.mult.values()].some(v => v !== 1)) bits.push("per-strategy overrides");
    if (fragActive()) {
      const f = S.frag;
      bits.push(`frag ${f.dial}/MA${f.ma} ${f.shape} thr${f.thr} floor${f.floor.toFixed(2)}` +
                (f.boost !== 1 ? ` boost${f.boost.toFixed(2)}` : ""));
    }
    if (fragEls) {
      if (fragActive()) {
        const vals = fragSeries();
        const last = vals ? vals[vals.length - 1] : null;
        // coverage + average multiplier over the frag-eligible trades
        let nCov = 0, nAll = 0, mSum = 0, nThrot = 0;
        for (const t of S.trades) {
          if (FRAG_EXEMPT.has(t.Strategy)) continue;
          nAll++;
          const sc = (t.Signal_Date || t.Entry_Date) ? fragScoreFor(t.Signal_Date || t.Entry_Date) : null;
          if (sc == null) continue;
          nCov++;
          const m = fragMultOf(sc);
          mSum += m;
          if (m < 1) nThrot++;
        }
        fragEls.info.textContent =
          `today ${last == null ? "n/a" : last.toFixed(1)} -> ${fragMultOf(last).toFixed(2)}x · ` +
          `covers ${nCov}/${nAll} non-OVS trades, avg ${nCov ? (mSum / nCov).toFixed(2) : "-"}x, ` +
          `${nCov ? Math.round(100 * nThrot / nCov) : 0}% throttled`;
      } else {
        fragEls.info.textContent = "";
      }
    }
    const summary = document.getElementById("riskSummary");
    summary.textContent = bits.length ? `(ACTIVE: ${bits.join(", ")})` : "(all at native risk)";
  }
  S.syncRiskUI = syncRiskUI;
  syncRiskUI();
}

/* ================= filtering ================= */
function tickerTokens() {
  if (!S.f.tickerQ) return null;
  return S.f.tickerQ.toUpperCase().split(",").map(s => s.trim()).filter(Boolean);
}

function filteredTrades(ignoreDates) {
  const toks = tickerTokens();
  const { strategies, tier, dir, from, to } = S.f;
  let src = S.showBlocked && S.gateBlockedRows.length
    ? S.trades.concat(S.gateBlockedRows) : S.trades;
  if (S.showOvsExt && S.extById.size) {
    src = src.map(t => S.extById.get(t.trade_id) || t);
  }
  return src.filter(t => {
    if (!strategies.has(t.Strategy)) return false;
    if (tier !== "All" && t.Tier !== tier) return false;
    if (dir !== "All" && t.Direction !== dir) return false;
    if (!ignoreDates) {
      const d = t.Entry_Date || t.Signal_Date;
      if (from && d < from) return false;
      if (to && d > to) return false;
    }
    if (toks && !toks.some(tok => (t.Ticker || "").toUpperCase().startsWith(tok))) return false;
    return true;
  });
}

function curveExact() {
  // per-trade fragility multipliers, gate-blocked counterfactual rows, and
  // rebooked OVS-extension exits can't be applied to the per-strategy
  // aggregated daily series -> fall back to the realized step curve
  return S.sd && S.f.dir === "All" && !tickerTokens() && !fragActive()
    && !S.showBlocked && !S.showOvsExt;
}

/* daily pnl array (risk multipliers + leverage applied) for current filters */
function dailySeries(trades, ignoreDates) {
  if (curveExact()) {
    const keys = Object.keys(S.sd.series).filter(k => {
      const [strat, tier] = k.split("||");
      if (!S.f.strategies.has(strat)) return false;
      if (S.f.tier !== "All" && tier !== S.f.tier) return false;
      return true;
    });
    let i0 = 0, i1 = S.dateIdx.length - 1;
    if (!ignoreDates && S.f.from) i0 = lowerBound(S.dateIdx, S.f.from);
    if (!ignoreDates && S.f.to) i1 = upperBound(S.dateIdx, S.f.to) - 1;
    if (i1 < i0) return { dates: [], pnl: [], exact: true };
    const n = i1 - i0 + 1;
    const pnl = new Float64Array(n);
    for (const k of keys) {
      const strat = k.split("||")[0];
      const m = multFor(strat);
      if (m === 0) continue;
      const arr = S.sd.series[k];
      const applyMid = S.midScalar !== 1 && S.midMask && !MID_EXEMPT.has(strat);
      for (let i = 0; i < n; i++) {
        const mm = (applyMid && S.midMask[i0 + i]) ? S.midScalar : 1;
        pnl[i] += arr[i0 + i] * m * mm;
      }
    }
    return { dates: S.dateIdx.slice(i0, i1 + 1), pnl: Array.from(pnl), exact: true };
  }
  // per-trade MTM vector path (trade_mtm.json): exact daily MTM for ANY
  // per-trade selection — direction/ticker filters, gate + extension
  // toggles, fragility multipliers. Marks are clipped to the filter window,
  // matching the aggregated exact path's semantics.
  if (S.mtmDates) {
    let i0 = 0, i1 = S.mtmDates.length - 1;
    if (!ignoreDates && S.f.from) i0 = lowerBound(S.mtmDates, S.f.from);
    if (!ignoreDates && S.f.to) i1 = upperBound(S.mtmDates, S.f.to) - 1;
    if (i1 < i0) return { dates: [], pnl: [], exact: true };
    const pnl = new Float64Array(i1 - i0 + 1);
    for (const t of trades) {
      const m = tradeMult(t);
      if (m === 0) continue;
      const rec = t.OvsExt ? S.mtmExt.get(t.trade_id)
        : t.GateBlocked ? S.mtmGate.get(`${t.Strategy}|${t.Tier}|${t.Ticker}|${t.Signal_Date}`)
        : S.mtmMain.get(t.trade_id);
      if (rec) {
        const [s, v] = rec;
        for (let k = 0; k < v.length; k++) {
          const j = s + k;
          if (j < i0) continue;
          if (j > i1) break;
          pnl[j - i0] += v[k] * m;
        }
      } else if (t.Exit_Date && t.PnL_flat != null) {
        // no vector shipped for this row — book realized PnL at exit
        const j = S.mtmDateToI.get(t.Exit_Date);
        if (j != null && j >= i0 && j <= i1) pnl[j - i0] += t.PnL_flat * m;
      }
    }
    return { dates: S.mtmDates.slice(i0, i1 + 1), pnl: Array.from(pnl), exact: true };
  }
  // last-resort fallback (no trade_mtm payload): realized PnL on exit dates
  const map = new Map();
  for (const t of trades) {
    const d = t.Exit_Date;
    if (!d || t.PnL_flat == null) continue;
    map.set(d, (map.get(d) || 0) + t.PnL_flat * tradeMult(t));
  }
  const exitDates = [...map.keys()].sort();
  // Densify onto the trading calendar: day-count metrics (Sharpe, CAGR,
  // vol, years) must see the flat days between exits, or a sparse subset
  // (single strategy, counterfactual toggle) inflates them by the density
  // ratio. Pad from the window start / to the window end when set, so the
  // basis matches the exact path; never clip realized PnL (a trade entered
  // in-window can exit past S.f.to).
  if (S.dateIdx && S.dateIdx.length && exitDates.length) {
    let lo = exitDates[0], hi = exitDates[exitDates.length - 1];
    if (!ignoreDates && S.f.from && S.f.from < lo) lo = S.f.from;
    if (!ignoreDates && S.f.to && S.f.to > hi) hi = S.f.to;
    const cal = S.dateIdx.slice(lowerBound(S.dateIdx, lo), upperBound(S.dateIdx, hi));
    const dates = [...new Set([...cal, ...exitDates])].sort();
    return { dates, pnl: dates.map(d => map.get(d) || 0), exact: false };
  }
  return { dates: exitDates, pnl: exitDates.map(d => map.get(d)), exact: false };
}

function lowerBound(arr, x) {
  let lo = 0, hi = arr.length;
  while (lo < hi) { const m = (lo + hi) >> 1; arr[m] < x ? lo = m + 1 : hi = m; }
  return lo;
}
function upperBound(arr, x) {
  let lo = 0, hi = arr.length;
  while (lo < hi) { const m = (lo + hi) >> 1; arr[m] <= x ? lo = m + 1 : hi = m; }
  return lo;
}

/* ================= metrics ================= */
function tradeMetrics(tr) {
  const n = tr.length;
  const rs = tr.map(t => t.R).filter(v => v != null);
  // Dollar metrics carry the risk multipliers + leverage + midterm scalar;
  // R stats stay raw (R is per unit of risk by definition).
  const pnls = tr.filter(t => t.PnL_flat != null)
                 .map(t => t.PnL_flat * tradeMult(t));
  const wins = pnls.filter(v => v > 0), losses = pnls.filter(v => v < 0);
  const sum = a => a.reduce((x, y) => x + y, 0);
  const mean = a => a.length ? sum(a) / a.length : null;
  const std = a => {
    if (a.length < 2) return null;
    const m = mean(a);
    return Math.sqrt(sum(a.map(v => (v - m) ** 2)) / (a.length - 1));
  };
  const totR = sum(rs), avgR = mean(rs), stdR = std(rs);
  const sortedR = rs.slice().sort((a, b) => a - b);
  const q = p => sortedR.length ? sortedR[Math.min(sortedR.length - 1, Math.floor(p * sortedR.length))] : null;
  let maxConsecL = 0, run = 0;
  const chron = tr.slice().sort((a, b) => (a.Exit_Date || "").localeCompare(b.Exit_Date || ""));
  for (const t of chron) {
    if (t.PnL_flat != null && t.PnL_flat < 0) { run++; maxConsecL = Math.max(maxConsecL, run); }
    else if (t.PnL_flat != null) run = 0;
  }
  const holds = tr.map(t => t.Hold_Days).filter(v => v != null);
  return {
    n, winRate: pnls.length ? wins.length / pnls.length : null,
    totR, avgR, stdR,
    sqn: (avgR != null && stdR) ? Math.sqrt(Math.min(n, 100)) * avgR / stdR : null,
    pf: losses.length ? sum(wins) / Math.abs(sum(losses)) : null,
    expectancy: mean(pnls),
    payoff: (wins.length && losses.length) ? mean(wins) / Math.abs(mean(losses)) : null,
    totPnl: sum(pnls),
    tail: (q(0.95) != null && q(0.05) != null && q(0.05) !== 0) ? Math.abs(q(0.95) / q(0.05)) : null,
    maxConsecL,
    avgHold: mean(holds),
  };
}

function dailyMetrics(ds) {
  const { pnl } = ds;
  const n = pnl.length;
  if (!n) return {};
  const scaled = S.sizing === "scaled";
  // Arithmetic daily returns on the $750k base drive Sharpe / Sortino / vol
  // in both modes (standard convention). In scaled mode the equity path
  // compounds them geometrically: sizes grow/shrink with equity each day.
  const rets = pnl.map(v => v / START_EQ);
  const sum = a => a.reduce((x, y) => x + y, 0);
  const m = sum(rets) / n;
  const sd = n > 1 ? Math.sqrt(sum(rets.map(v => (v - m) ** 2)) / (n - 1)) : 0;
  const downside = rets.filter(v => v < 0);
  const dsd = downside.length > 1 ?
    Math.sqrt(sum(downside.map(v => v * v)) / downside.length) : 0;
  // equity path + max drawdown vs running peak (per active sizing mode),
  // displayed from a $10k start (scale-invariant in % terms)
  let eq = DISPLAY_EQ, peak = DISPLAY_EQ, maxDD = 0;
  let bestDay = -Infinity, worstDay = Infinity;
  const equity = new Array(n);
  for (let i = 0; i < n; i++) {
    const dayPnl = scaled ? eq * rets[i] : pnl[i] * DSCALE;
    eq += dayPnl;
    if (eq <= 0) eq = 0.01; // leverage blow-up floor; keeps log axis sane
    equity[i] = eq;
    if (dayPnl > bestDay) bestDay = dayPnl;
    if (dayPnl < worstDay) worstDay = dayPnl;
    if (eq > peak) peak = eq;
    const dd = eq / peak - 1;
    if (dd < maxDD) maxDD = dd;
  }
  const years = n / 252;
  const annRet = scaled
    ? (years > 0 ? Math.pow(eq / DISPLAY_EQ, 1 / years) - 1 : null)  // CAGR
    : m * 252;                                                        // arithmetic
  const annVol = sd * Math.sqrt(252);
  return {
    equity, annRet, annVol,
    sharpe: sd ? (m / sd) * Math.sqrt(252) : null,
    sortino: dsd ? (m / dsd) * Math.sqrt(252) : null,
    maxDD,
    mar: (maxDD && annRet != null) ? annRet / Math.abs(maxDD) : null,
    totPnl: eq - DISPLAY_EQ,
    bestDay, worstDay,
  };
}

/* ================= render ================= */
function apply() {
  const tr = filteredTrades();
  const ds = dailySeries(tr);
  const tm = tradeMetrics(tr);
  const dm = dailyMetrics(ds);
  renderKPIs(tm, dm, ds.exact);
  renderEquity(ds, dm);
  renderCumR(tr);
  renderMonthly(ds);
  // rolling Sharpe needs 252d of warmup, so it computes on the date-unbounded
  // stream (same strategy/tier/dir/ticker filters) and only displays the window
  renderRolling(dailySeries(filteredTrades(true), true));
  renderHist(tr);
  renderSeasonality(tr);
  renderHoldBuckets(tr);
  renderStratTable(tr);
  renderYearTable(tr, ds);
  renderTradeLog(tr);
  renderStopFills();
}

function kpiCard(label, value, cls, sub) {
  return `<div class="kpi"><div class="l">${label}</div>
    <div class="v ${cls || ""}">${value}</div>${sub ? `<div class="s">${sub}</div>` : ""}</div>`;
}

function renderKPIs(tm, dm, exact) {
  const el = document.getElementById("kpis");
  const basisNote = document.getElementById("basisNote");
  basisNote.style.display = exact ? "none" : "inline-block";
  const scaled = S.sizing === "scaled";
  el.innerHTML = [
    kpiCard("Trades", tm.n.toLocaleString()),
    kpiCard("Win Rate", tm.winRate == null ? "-" : fmt.pct(tm.winRate, 1)),
    kpiCard("Total R", fmt.num(tm.totR, 0), clsSign(tm.totR)),
    kpiCard("Avg R", tm.avgR == null ? "-" : fmt.num(tm.avgR, 3), clsSign(tm.avgR)),
    kpiCard("Profit Factor", tm.pf == null ? "-" : fmt.num(tm.pf, 2)),
    kpiCard("Expectancy", tm.expectancy == null ? "-" : fmt.money(tm.expectancy), clsSign(tm.expectancy), "per trade @ $750k alloc"),
    kpiCard("SQN", tm.sqn == null ? "-" : fmt.num(tm.sqn, 2)),
    kpiCard("Payoff", tm.payoff == null ? "-" : fmt.num(tm.payoff, 2), null, "avg win / avg loss"),
    kpiCard("Total PnL", dm.totPnl == null ? "-" : fmt.money(dm.totPnl), clsSign(dm.totPnl),
            scaled ? "compounded, $10k start" : "flat, $10k base"),
    kpiCard(scaled ? "CAGR" : "Ann Return",
            dm.annRet == null ? "-" : fmt.pct(dm.annRet, 1), clsSign(dm.annRet),
            scaled ? "geometric" : "of $750k"),
    kpiCard("Ann Vol", dm.annVol == null ? "-" : fmt.pct(dm.annVol, 1)),
    kpiCard("Sharpe", dm.sharpe == null ? "-" : fmt.num(dm.sharpe, 2), clsSign(dm.sharpe)),
    kpiCard("Sortino", dm.sortino == null ? "-" : fmt.num(dm.sortino, 2)),
    kpiCard("Max Drawdown", dm.maxDD == null ? "-" : fmt.pct(dm.maxDD, 1), "neg",
            scaled ? "compounded path" : "flat path"),
    kpiCard("MAR", dm.mar == null ? "-" : fmt.num(dm.mar, 2)),
    kpiCard("Tail Ratio", tm.tail == null ? "-" : fmt.num(tm.tail, 2), null, "|p95 / p5| of R"),
    kpiCard("Max Consec Losses", tm.maxConsecL),
    kpiCard("Avg Hold", tm.avgHold == null ? "-" : fmt.num(tm.avgHold, 1), null, "trading days"),
  ].join("");
}

function renderEquity(ds, dm) {
  const eqEl = document.getElementById("eqChart");
  const ddEl = document.getElementById("ddChart");
  const cap = document.getElementById("eqCaption");
  if (!ds.dates.length) { Plotly.purge(eqEl); Plotly.purge(ddEl); cap.textContent = ""; return; }
  const scaled = S.sizing === "scaled";
  cap.textContent = (scaled
    ? "Scaled with portfolio size: daily returns compound geometrically (risk grows/shrinks with equity). Log scale, $10k start."
    : "Flat sizing: every trade risks bps of a fixed allocation. Era-comparable; best for judging raw edge. $10k display base.")
    + (S.rangeTouched ? "" : " Showing trailing 1y by default — pick a Range or double-click to zoom out.");
  const equity = dm.equity;
  const lastDate = ds.dates[ds.dates.length - 1];

  // Default (untouched Range): trailing-1y viewport over full-history data.
  // Once the user picks a Range, show exactly the filtered span (autorange).
  let xRange = null, yRange = null, ddRangeY = null, w0 = 0;
  let peak = -Infinity;
  const dd = equity.map(v => { peak = Math.max(peak, v); return (v / peak - 1) * 100; });
  if (!S.rangeTouched) {
    const cut = new Date(lastDate + "T00:00:00Z");
    cut.setUTCFullYear(cut.getUTCFullYear() - 1);
    const cutStr = cut.toISOString().slice(0, 10);
    w0 = lowerBound(ds.dates, cutStr);
    const winEq = equity.slice(w0);
    if (winEq.length) {
      xRange = [cutStr, lastDate];
      const lo = Math.min(...winEq), hi = Math.max(...winEq);
      yRange = scaled
        ? [Math.log10(Math.max(lo, 0.01) * 0.98), Math.log10(hi * 1.02)]
        : [lo - (hi - lo) * 0.06 - 1, hi + (hi - lo) * 0.06 + 1];
      const winDD = dd.slice(w0);
      ddRangeY = [Math.min(...winDD) * 1.15 - 0.1, 0.5];
    }
  }

  Plotly.react(eqEl, [{
    x: ds.dates, y: equity, mode: "lines", name: "Equity",
    line: { color: "#00d18f", width: 1.8 },
  }], plotLayout({
    height: 330,
    xaxis: xRange ? { range: xRange } : {},
    yaxis: { type: scaled ? "log" : "linear", tickformat: "$,.4~s",
             range: yRange,
             title: { text: scaled ? "Equity (compounded, log, $10k start)" : "Equity (flat, $10k base)", font: { size: 11 } } },
    shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: DISPLAY_EQ, y1: DISPLAY_EQ,
               line: { color: "#444c5c", width: 1, dash: "dot" } }],
  }), PLOT_CFG);

  Plotly.react(ddEl, [{
    x: ds.dates, y: dd, mode: "lines", name: "Drawdown",
    fill: "tozeroy", line: { color: "#ff5d5d", width: 1 }, fillcolor: "rgba(255,93,93,.18)",
  }], plotLayout({
    height: 150, margin: { t: 8 },
    xaxis: xRange ? { range: xRange } : {},
    yaxis: { ticksuffix: "%", range: ddRangeY },
  }), PLOT_CFG);
}

function renderCumR(tr) {
  const el = document.getElementById("cumRChart");
  const byStrat = new Map();
  for (const t of tr) {
    if (t.R == null || !t.Exit_Date) continue;
    if (!byStrat.has(t.Strategy)) byStrat.set(t.Strategy, []);
    byStrat.get(t.Strategy).push(t);
  }
  let entries = [...byStrat.entries()].map(([s, arr]) => {
    return [s, arr, Math.abs(arr.reduce((x, t) => x + t.R, 0))];
  }).sort((a, b) => b[2] - a[2]);
  const traces = [];
  const top = entries.slice(0, 12), rest = entries.slice(12);
  top.forEach(([s, arr], i) => {
    arr.sort((a, b) => a.Exit_Date.localeCompare(b.Exit_Date));
    let c = 0;
    traces.push({
      x: arr.map(t => t.Exit_Date), y: arr.map(t => (c += t.R)),
      mode: "lines", name: s, line: { width: 1.4, color: PALETTE[i % PALETTE.length] },
    });
  });
  if (rest.length) {
    const all = rest.flatMap(([, arr]) => arr)
      .sort((a, b) => a.Exit_Date.localeCompare(b.Exit_Date));
    let c = 0;
    traces.push({
      x: all.map(t => t.Exit_Date), y: all.map(t => (c += t.R)),
      mode: "lines", name: `Other (${rest.length})`, line: { width: 1.2, color: "#5a6272", dash: "dot" },
    });
  }
  Plotly.react(el, traces, plotLayout({
    height: 330, yaxis: { title: { text: "Cumulative R", font: { size: 11 } } },
    legend: { font: { size: 10 } },
  }), PLOT_CFG);
}

function renderMonthly(ds) {
  const el = document.getElementById("monthlyChart");
  if (!ds.dates.length) { Plotly.purge(el); return; }
  const agg = new Map();
  for (let i = 0; i < ds.dates.length; i++) {
    const y = ds.dates[i].slice(0, 4), m = +ds.dates[i].slice(5, 7);
    const k = y + "-" + m;
    agg.set(k, (agg.get(k) || 0) + ds.pnl[i]);
  }
  const years = [...new Set(ds.dates.map(d => d.slice(0, 4)))].sort().reverse();
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const z = years.map(y => months.map((_, mi) => {
    const v = agg.get(y + "-" + (mi + 1));
    return v == null ? null : +(v / START_EQ * 100).toFixed(2);
  }));
  Plotly.react(el, [{
    z, x: months, y: years, type: "heatmap",
    colorscale: [[0, "#c0392b"], [0.5, "#10141d"], [1, "#0e9f6e"]],
    zmid: 0, showscale: false,
    texttemplate: "%{z:.1f}", textfont: { size: 9 },
    hovertemplate: "%{y} %{x}: %{z:.2f}%<extra></extra>",
  }], plotLayout({
    height: Math.max(280, 17 * years.length + 70),
    margin: { l: 46, t: 8 }, xaxis: { side: "top", gridcolor: "rgba(0,0,0,0)" },
    yaxis: { gridcolor: "rgba(0,0,0,0)", autorange: "reversed", type: "category" },
    hovermode: "closest",
  }), PLOT_CFG);
}

function renderRolling(dsAll) {
  const el = document.getElementById("rollSharpeChart");
  const W = 252;
  if (dsAll.pnl.length < W + 10) {
    Plotly.purge(el);
    el.innerHTML = '<p class="cap">Fewer than ~262 trading days of history for these filters — rolling 252d Sharpe unavailable.</p>';
    return;
  }
  // Only clear the container when it holds the fallback caption (no live plot).
  // Wiping innerHTML on an element Plotly has already initialized breaks
  // Plotly.react on every subsequent render.
  if (!el._fullLayout) el.innerHTML = "";
  const rets = dsAll.pnl.map(v => v / START_EQ);
  // prefix sums for O(n) rolling mean/std
  const n = rets.length, ps = new Float64Array(n + 1), ps2 = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) { ps[i + 1] = ps[i] + rets[i]; ps2[i + 1] = ps2[i] + rets[i] * rets[i]; }
  const xs = [], ys = [];
  for (let i = W; i <= n; i++) {
    const s = ps[i] - ps[i - W], s2 = ps2[i] - ps2[i - W];
    const m = s / W, varr = (s2 - W * m * m) / (W - 1);
    const sd = varr > 0 ? Math.sqrt(varr) : 0;
    xs.push(dsAll.dates[i - 1]);
    ys.push(sd ? +(m / sd * Math.sqrt(252)).toFixed(3) : null);
  }
  // display only the active date window (computed with full-history warmup)
  let xRange = null;
  if (S.f.from || S.f.to) {
    xRange = [S.f.from || xs[0], S.f.to || xs[xs.length - 1]];
  }
  Plotly.react(el, [{
    x: xs, y: ys, mode: "lines", name: "Sharpe (252d)",
    line: { color: "#4da3ff", width: 1.4 },
  }], plotLayout({
    height: 280,
    xaxis: xRange ? { range: xRange } : {},
    shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0, y1: 0,
               line: { color: "#444c5c", width: 1, dash: "dot" } }],
  }), PLOT_CFG);
}

function renderHist(tr) {
  const el = document.getElementById("histChart");
  const rs = tr.map(t => t.R).filter(v => v != null).map(v => Math.max(-3, Math.min(5, v)));
  Plotly.react(el, [{
    x: rs, type: "histogram", xbins: { start: -3, end: 5, size: 0.2 },
    marker: { color: "#4da3ff", line: { color: "#0b0e14", width: 0.5 } },
  }], plotLayout({
    height: 280, hovermode: "closest", bargap: 0.02,
    xaxis: { title: { text: "R multiple (clipped at -3 / +5)", font: { size: 11 } } },
    yaxis: { title: { text: "Trades", font: { size: 11 } } },
  }), PLOT_CFG);
}

function renderSeasonality(tr) {
  const monthEl = document.getElementById("monthSeasChart");
  const wdEl = document.getElementById("weekdaySeasChart");
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const mAgg = new Array(12).fill(0), mCnt = new Array(12).fill(0);
  const wAgg = new Array(7).fill(0), wCnt = new Array(7).fill(0);
  for (const t of tr) {
    if (t.R == null || !t.Entry_Date) continue;
    const d = new Date(t.Entry_Date + "T00:00:00Z");
    mAgg[d.getUTCMonth()] += t.R; mCnt[d.getUTCMonth()]++;
    wAgg[d.getUTCDay()] += t.R; wCnt[d.getUTCDay()]++;
  }
  Plotly.react(monthEl, [{
    x: months, y: mAgg.map(v => +v.toFixed(1)), type: "bar",
    marker: { color: mAgg.map(v => v >= 0 ? "#00d18f" : "#ff5d5d") },
    customdata: mCnt, hovertemplate: "%{x}: %{y:.1f}R (%{customdata} trades)<extra></extra>",
  }], plotLayout({
    height: 250, hovermode: "closest",
    yaxis: { title: { text: "Total R by entry month", font: { size: 11 } } },
  }), PLOT_CFG);
  const wdNames = ["Mon", "Tue", "Wed", "Thu", "Fri"];
  const wy = [1, 2, 3, 4, 5].map(i => +wAgg[i].toFixed(1));
  Plotly.react(wdEl, [{
    x: wdNames, y: wy, type: "bar",
    marker: { color: wy.map(v => v >= 0 ? "#00d18f" : "#ff5d5d") },
    customdata: [1, 2, 3, 4, 5].map(i => wCnt[i]),
    hovertemplate: "%{x}: %{y:.1f}R (%{customdata} trades)<extra></extra>",
  }], plotLayout({
    height: 250, hovermode: "closest",
    yaxis: { title: { text: "Total R by entry weekday", font: { size: 11 } } },
  }), PLOT_CFG);
}

function renderHoldBuckets(tr) {
  const el = document.getElementById("holdChart");
  const buckets = [[0, 2], [3, 5], [6, 10], [11, 21], [22, 42], [43, 9999]];
  const labels = ["0-2d", "3-5d", "6-10d", "11-21d", "22-42d", "43d+"];
  const agg = buckets.map(() => ({ r: 0, n: 0 }));
  for (const t of tr) {
    if (t.R == null || t.Hold_Days == null) continue;
    const i = buckets.findIndex(([a, b]) => t.Hold_Days >= a && t.Hold_Days <= b);
    if (i >= 0) { agg[i].r += t.R; agg[i].n++; }
  }
  Plotly.react(el, [{
    x: labels, y: agg.map(a => +a.r.toFixed(1)), type: "bar",
    marker: { color: agg.map(a => a.r >= 0 ? "#00d18f" : "#ff5d5d") },
    customdata: agg.map(a => a.n),
    hovertemplate: "%{x}: %{y:.1f}R (%{customdata} trades)<extra></extra>",
  }], plotLayout({
    height: 250, hovermode: "closest",
    yaxis: { title: { text: "Total R by holding period", font: { size: 11 } } },
  }), PLOT_CFG);
}

function renderStratTable(tr) {
  const el = document.getElementById("stratTable");
  const groups = new Map();
  for (const t of tr) {
    const k = t.Strategy + "||" + t.Tier;
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(t);
  }
  const totalR = tr.reduce((x, t) => x + (t.R || 0), 0);
  const rows = [...groups.entries()].map(([k, arr]) => {
    const [s, tier] = k.split("||");
    const m = tradeMetrics(arr);
    return {
      Strategy: s, Tier: tier, Trades: m.n,
      Win: m.winRate, TotR: m.totR, AvgR: m.avgR, PF: m.pf,
      PnL: m.totPnl, Share: totalR ? m.totR / totalR : null,
      AvgHold: m.avgHold,
    };
  });
  makeTable(el, {
    columns: [
      { key: "Strategy", label: "Strategy", align: "l" },
      { key: "Tier", label: "Tier", align: "l" },
      { key: "Trades", label: "Trades", fmt: v => v.toLocaleString() },
      { key: "Win", label: "Win %", fmt: v => fmt.pct(v, 1) },
      { key: "TotR", label: "Total R", fmt: v => fmt.num(v, 1), cls: clsSign },
      { key: "AvgR", label: "Avg R", fmt: v => fmt.num(v, 3), cls: clsSign },
      { key: "PF", label: "PF", fmt: v => v == null ? "" : fmt.num(v, 2) },
      { key: "PnL", label: "PnL ($)", fmt: v => fmt.money(v), cls: clsSign },
      { key: "Share", label: "% of R", fmt: v => v == null ? "" : fmt.pct(v, 1) },
      { key: "AvgHold", label: "Avg Hold", fmt: v => v == null ? "" : fmt.num(v, 1) },
    ],
    rows, defaultSort: { key: "TotR", dir: -1 },
  });
}

function renderYearTable(tr, ds) {
  const el = document.getElementById("yearTable");
  const years = new Map();
  for (let i = 0; i < ds.dates.length; i++) {
    const y = ds.dates[i].slice(0, 4);
    if (!years.has(y)) years.set(y, { pnl: [], trades: 0, r: 0, wins: 0, closed: 0 });
    years.get(y).pnl.push(ds.pnl[i]);
  }
  for (const t of tr) {
    const y = (t.Entry_Date || "").slice(0, 4);
    if (!years.has(y)) continue;
    const g = years.get(y);
    g.trades++;
    if (t.R != null) g.r += t.R;
    if (t.PnL_flat != null) { g.closed++; if (t.PnL_flat > 0) g.wins++; }
  }
  const rows = [...years.entries()].sort((a, b) => b[0].localeCompare(a[0])).map(([y, g]) => {
    const n = g.pnl.length;
    const tot = g.pnl.reduce((x, v) => x + v, 0);
    const rets = g.pnl.map(v => v / START_EQ);
    const m = n ? rets.reduce((x, v) => x + v, 0) / n : 0;
    const sd = n > 1 ? Math.sqrt(rets.reduce((x, v) => x + (v - m) ** 2, 0) / (n - 1)) : 0;
    let eq = START_EQ, peak = START_EQ, maxDD = 0;
    for (const v of g.pnl) { eq += v; peak = Math.max(peak, eq); maxDD = Math.min(maxDD, eq / peak - 1); }
    return {
      Year: y, Trades: g.trades, Win: g.closed ? g.wins / g.closed : null,
      TotR: g.r, PnL: tot * DSCALE, Ret: tot / START_EQ,
      MaxDD: maxDD, Sharpe: sd ? m / sd * Math.sqrt(252) : null,
    };
  });
  makeTable(el, {
    columns: [
      { key: "Year", label: "Year", align: "l" },
      { key: "Trades", label: "Trades" },
      { key: "Win", label: "Win %", fmt: v => fmt.pct(v, 1) },
      { key: "TotR", label: "Total R", fmt: v => fmt.num(v, 1), cls: clsSign },
      { key: "PnL", label: "PnL ($10k base)", fmt: v => fmt.money(v), cls: clsSign },
      { key: "Ret", label: "Return %", fmt: v => fmt.pct(v, 1), cls: clsSign },
      { key: "MaxDD", label: "Max DD", fmt: v => fmt.pct(v, 1), cls: () => "neg" },
      { key: "Sharpe", label: "Sharpe", fmt: v => v == null ? "" : fmt.num(v, 2) },
    ],
    rows,
  });
}

function renderTradeLog(tr) {
  const el = document.getElementById("tradeLog");
  const columns = [
    { key: "Entry_Date", label: "Entry", align: "l" },
    { key: "Exit_Date", label: "Exit", align: "l" },
    { key: "Strategy", label: "Strategy", align: "l" },
    { key: "Tier", label: "Tier", align: "l" },
    { key: "Ticker", label: "Ticker", align: "l",
      fmt: (v, r) => {
        if (r.GateBlocked) return `${v || ""} <span class="badge warn" title="Sector-loss gate blocked this signal live — outcome is counterfactual">GATE</span>`;
        if (r.OvsExt) return `${v || ""} <span class="badge warn" title="Losing T+2 exit rebooked to T+5 — outcome is counterfactual">T+5</span>`;
        return v || "";
      } },
    { key: "Direction", label: "Dir", align: "l",
      fmt: v => `<span class="badge ${v === "Short" ? "dirS" : "dirL"}">${v || ""}</span>` },
    { key: "Entry_Price", label: "Entry $", fmt: v => fmt.num(v, 2) },
    { key: "Exit_Price", label: "Exit $", fmt: v => fmt.num(v, 2) },
    { key: "Return_Pct", label: "Ret %", fmt: v => v == null ? "" : fmt.signed(v, 2), cls: clsSign },
    { key: "R", label: "R", fmt: v => v == null ? "" : fmt.signed(v, 2), cls: clsSign },
    { key: "PnL_flat", label: "PnL ($)", fmt: v => fmt.money(v), cls: clsSign },
    { key: "Hold_Days", label: "Hold", fmt: v => v == null ? "" : v + "d" },
    { key: "Exit_Type", label: "Exit Type", align: "l" },
    { key: "Entry_Criteria", label: "Criteria", align: "l" },
  ];
  // closed trades only — open positions (time stop not reached) live in the
  // Open Positions section above
  const rows = tr.filter(t => !t.Open)
    .sort((a, b) => (b.Entry_Date || "").localeCompare(a.Entry_Date || ""));
  if (S.tradeLogTable) S.tradeLogTable.setRows(rows);
  else S.tradeLogTable = makeTable(el, {
    columns, rows, pageSize: 25, search: true, csvName: "trades_filtered.csv",
  });
}

/* ---------- static (full-book) sections ---------- */
function renderStatic() {
  // open positions + overnight gap stress
  const posEl = document.getElementById("positionsTable");
  const stressNote = document.getElementById("posStressNote");
  if (S.positions && S.positions.positions.length) {
    const rows = S.positions.positions;
    // flatten Gap_Stress into sortable columns (older payloads without the
    // field just render blanks)
    for (const r of rows) {
      for (const k of [1, 2, 3]) {
        const g = Array.isArray(r.Gap_Stress) ? r.Gap_Stress.find(x => x.gap_atr === k) : null;
        r["Gap" + k] = g ? g.impact : null;
        r["GapB" + k] = g ? g.stop_blown : null;
        r["GapC" + k] = g && g.stop_cap != null ? g.stop_cap : null;
      }
    }
    const totLong = rows.filter(r => r.Direction === "Long").reduce((x, r) => x + (r.Mkt_Value || 0), 0);
    const totShort = rows.filter(r => r.Direction === "Short").reduce((x, r) => x + (r.Mkt_Value || 0), 0);
    const opnl = rows.reduce((x, r) => x + (r.Open_PnL || 0), 0);
    const hasStress = rows.some(r => r.Gap1 != null);
    const gapSum = k => rows.reduce((x, r) => x + (r["Gap" + k] || 0), 0);
    const cards = [
      kpiCard("Open Positions", rows.length),
      kpiCard("Long Mkt Value", fmt.money(totLong)),
      kpiCard("Short Mkt Value", fmt.money(totShort)),
      kpiCard("Net", fmt.money(totLong - totShort), clsSign(totLong - totShort)),
      kpiCard("Open PnL", fmt.money(opnl), clsSign(opnl)),
    ];
    if (hasStress) {
      for (const k of [1, 2, 3]) {
        const s = gapSum(k);
        cards.push(kpiCard(`Gap −${k} ATR`, fmt.money(s), "neg",
          `book gap stress · w/ open PnL ${fmt.money(opnl + s)}`));
      }
    }
    document.getElementById("posCards").innerHTML = cards.join("");
    const noStop = '<span class="cap" title="No live stop for this strategy — level shown is only the R denominator">—</span>';
    const gapCol = k => ({
      key: "Gap" + k, label: `Gap −${k} ATR`,
      fmt: (v, r) => {
        if (v == null) return "";
        const b = r["GapB" + k];
        const cap = r["GapC" + k];
        const tag = b === false
          ? ` <span class="cap" title="Stop survives this gap — further intraday slide bounded at the stop${cap != null ? " (" + fmt.money(cap) + ")" : ""}">s</span>`
          : b === true ? ' <span class="cap" title="Gap opens beyond the stop — fills at the gapped price">g</span>' : "";
        return fmt.money(v) + tag;
      },
      cls: clsSign,
    });
    makeTable(posEl, {
      columns: [
        { key: "Entry_Date", label: "Entry", align: "l" },
        { key: "Time_Stop", label: "Time Stop", align: "l" },
        { key: "Strategy", label: "Strategy", align: "l" },
        { key: "Tier", label: "Tier", align: "l" },
        { key: "Ticker", label: "Ticker", align: "l" },
        { key: "Sector", label: "Sector", align: "l" },
        { key: "Direction", label: "Dir", align: "l",
          fmt: v => `<span class="badge ${v === "Short" ? "dirS" : "dirL"}">${v || ""}</span>` },
        { key: "Entry_Price", label: "Entry $", fmt: v => fmt.num(v, 2) },
        { key: "Current_Price", label: "Last $", fmt: v => v == null ? "" : fmt.num(v, 2) },
        { key: "Stop_Price", label: "Stop $",
          fmt: (v, r) => r.Use_Stop === false ? noStop : (v == null ? "" : fmt.num(v, 2)),
          cls: (v, r) => {
            if (r.Use_Stop === false || v == null || r.Current_Price == null) return "";
            const long = r.Direction !== "Short";
            const distAtr = Math.abs(r.Current_Price - v) /
              Math.max(1e-9, Math.abs(r.Entry_Price - v));
            return (long ? r.Current_Price <= v : r.Current_Price >= v) ? "neg"
              : distAtr < 0.35 ? "neg" : "";
          } },
        { key: "Stop_Dist_ATR", label: "Stop dist",
          fmt: (v, r) => (r.Use_Stop === false || v == null) ? "" :
            fmt.num(v, 2) + (r.Stop_Armed === false ? ' <span class="cap" title="Stop arms next session (day-2 arming)">u</span>' : ""),
          cls: (v, r) => (r.Use_Stop === false || v == null) ? "" :
            v <= 0.25 ? "neg" : v < 0.75 ? "neu" : "pos" },
        { key: "Tgt_Price", label: "Target $", fmt: v => v == null ? "" : fmt.num(v, 2) },
        { key: "Days_Held", label: "Held", fmt: v => v == null ? "" : v + "d" },
        { key: "Days_To_Time_Stop", label: "T-stop in",
          fmt: v => v == null ? "" : v + "d", cls: v => (v != null && v <= 1) ? "neu" : "" },
        { key: "Shares", label: "Shares", fmt: v => fmt.num(v, 1) },
        { key: "Mkt_Value", label: "Mkt Value", fmt: v => fmt.money(v) },
        { key: "Open_PnL", label: "Open PnL", fmt: v => fmt.money(v), cls: clsSign },
        gapCol(1), gapCol(2), gapCol(3),
      ],
      rows, defaultSort: { key: "Entry_Date", dir: -1 },
    });
    stressNote.textContent = hasStress
      ? "Gap stress: adverse overnight gap of 1/2/3 ATR applied to every open position at flat-$750k shares, " +
        "slippage excluded. Every dollar figure is the pure mark-to-gap at the next open, so the book KPIs sum " +
        "one consistent model. Tags describe what the stop does next: (s) stop survives the gap and bounds " +
        "further intraday slide at the stop level; (g) gap opens beyond the stop and fills at the gapped price; " +
        "no tag = no armed stop. Day-2 arming means even today's entries have live stops at the open being " +
        "stressed. Stop dist = ATRs of room left before the stop. Positions are backtest-modeled opens (not " +
        "broker fills); dollars ignore the risk-panel dials."
      : "";
  } else {
    posEl.innerHTML = '<p class="cap">No open positions.</p>';
    if (stressNote) stressNote.textContent = "";
  }

  // compounded reference curve
  const refEl = document.getElementById("refChart");
  if (S.sd && S.sd.equity_compounded) {
    Plotly.react(refEl, [{
      x: S.sd.dates, y: S.sd.equity_compounded, mode: "lines",
      name: "Compounded", line: { color: "#b07cff", width: 1.5 },
    }], plotLayout({
      height: 300, yaxis: { type: "log", tickformat: "$,.0s" },
    }), PLOT_CFG);
  } else refEl.innerHTML = '<p class="cap">No compounded series in this build.</p>';

  // exposure
  const expEl = document.getElementById("expChart");
  if (S.exposure) {
    const e = S.exposure;
    const mk = (y, name, color) => ({ x: e.dates, y, mode: "lines", name, line: { color, width: 1 } });
    Plotly.react(expEl, [
      mk(e.long, "Long", "#00d18f"), mk(e.short, "Short", "#ff5d5d"),
      mk(e.net, "Net", "#4da3ff"), mk(e.gross, "Gross", "#ffc14d"),
    ], plotLayout({ height: 300, yaxis: { ticksuffix: "%" } }), PLOT_CFG);
  } else expEl.innerHTML = '<p class="cap">No exposure series in this build.</p>';

  // correlation
  const corrEl = document.getElementById("corrChart");
  const divEl = document.getElementById("divTable");
  if (S.corr) {
    const c = S.corr;
    const z = c.matrix.map((row, i) => row.map((v, j) => i === j ? null : v));
    Plotly.react(corrEl, [{
      z, x: c.strategies, y: c.strategies, type: "heatmap",
      colorscale: "RdBu", reversescale: true, zmin: -1, zmax: 1,
      texttemplate: "%{z:.2f}", textfont: { size: 8.5 },
      hovertemplate: "%{y} vs %{x}: %{z:.2f}<extra></extra>", showscale: false,
    }], plotLayout({
      height: Math.max(380, 30 * c.strategies.length + 120),
      margin: { l: 140, b: 120 }, hovermode: "closest",
      xaxis: { tickangle: 45, gridcolor: "rgba(0,0,0,0)", tickfont: { size: 9.5 } },
      yaxis: { gridcolor: "rgba(0,0,0,0)", tickfont: { size: 9.5 } },
    }), PLOT_CFG);
    makeTable(divEl, {
      columns: [
        { key: "strategy", label: "Strategy", align: "l" },
        { key: "avg_corr", label: "Avg Corr", fmt: v => fmt.num(v, 2),
          cls: v => v < 0.2 ? "pos" : v < 0.45 ? "neu" : "neg" },
        { key: "max_corr", label: "Max Corr", fmt: v => fmt.num(v, 2) },
        { key: "max_with", label: "Most Correlated With", align: "l" },
      ],
      rows: c.diversification,
    });
  } else {
    corrEl.innerHTML = '<p class="cap">No correlation matrix in this build.</p>';
    divEl.innerHTML = "";
  }

  renderDrawdownAnatomy();
  renderSectorRisk();
  renderGateLab();
  renderExtLab();
}

/* ---------- sector-gate history: with vs without (gate_lab.json) ---------- */
function renderGateLab() {
  const section = document.getElementById("glSection");
  const gl = S.gateLab;
  if (!gl || !gl.strategies || !gl.strategies.length) {
    section.innerHTML = '<p class="cap">No gate counterfactual payload in this build ' +
      "(the ledger build writes data/backtest_trades_nogate.parquet; rerun it to populate).</p>";
    return;
  }

  // KPIs: per gated strategy (OLV today). Impact = baseline − nogate:
  // positive means the gate ADDED that much by blocking.
  const cards = [];
  for (const st of gl.strategies) {
    const s = st.summary || {};
    const b = s.blocked || {}, base = s.baseline || {}, ng = s.nogate || {};
    const impR = (base.tot_r != null && ng.tot_r != null) ? base.tot_r - ng.tot_r : null;
    const imp$ = (base.pnl_flat != null && ng.pnl_flat != null) ? base.pnl_flat - ng.pnl_flat : null;
    const pfx = gl.strategies.length > 1 ? st.strategy + " · " : "";
    cards.push(
      kpiCard(pfx + "Blocked trades", b.n ?? 0,
              null, `gate: ${st.window_td}td / ${fmt.num(st.max_realized_r, 1)}R`),
      kpiCard(pfx + "Blocked R", b.tot_r == null ? "-" : fmt.signed(b.tot_r, 1), clsSign(b.tot_r),
              b.avg_r == null ? "" : `avg ${fmt.signed(b.avg_r, 3)}R · win ${fmt.pctRaw(b.win_pct, 0)}`),
      kpiCard(pfx + "Gate impact (R)", impR == null ? "-" : fmt.signed(impR, 1), clsSign(impR),
              "baseline − no-gate; + = gate helped"),
      kpiCard(pfx + "Gate impact ($)", imp$ == null ? "-" : fmt.money(imp$), clsSign(imp$),
              "flat $750k, realized"),
    );
    if (st.n_gone) {
      cards.push(kpiCard(pfx + "Displaced", st.n_gone, "neu",
        "baseline trades absent in the no-gate run (cap/ladder interaction)"));
    }
  }
  document.getElementById("glKpis").innerHTML = cards.join("");

  // cumulative realized PnL curves, gate on (solid) vs off (dashed)
  const traces = [];
  gl.strategies.forEach((st, i) => {
    const c = st.curve;
    if (!c || !c.dates || !c.dates.length) return;
    const cum = arr => { let s = 0; return arr.map(v => +(s += v).toFixed(0)); };
    const color = PALETTE[i % PALETTE.length];
    traces.push(
      { x: c.dates, y: cum(c.base_pnl), mode: "lines", name: `${st.strategy} — gate on`,
        line: { color, width: 1.6 } },
      { x: c.dates, y: cum(c.nogate_pnl), mode: "lines", name: `${st.strategy} — gate off`,
        line: { color: "#ffc14d", width: 1.3, dash: "dot" } },
    );
  });
  const curveEl = document.getElementById("glCurve");
  if (traces.length) {
    Plotly.react(curveEl, traces, plotLayout({
      height: 340, yaxis: { tickformat: "$,.3~s" },
    }), PLOT_CFG);
  } else {
    curveEl.innerHTML = '<p class="cap">No curve data in this payload.</p>';
  }

  // blocked-trade table (all gated strategies merged; Strategy column present)
  makeTable(document.getElementById("glTable"), {
    columns: [
      { key: "Signal_Date", label: "Signal", align: "l" },
      { key: "Ticker", label: "Ticker", align: "l" },
      { key: "Tier", label: "Tier", align: "l" },
      { key: "Direction", label: "Dir", align: "l",
        fmt: v => `<span class="badge ${v === "Short" ? "dirS" : "dirL"}">${v || ""}</span>` },
      { key: "Entry_Date", label: "Entry", align: "l" },
      { key: "Exit_Date", label: "Exit", align: "l" },
      { key: "R", label: "R", fmt: v => v == null ? "" : fmt.signed(v, 2), cls: clsSign },
      { key: "PnL_flat", label: "PnL ($)", fmt: v => fmt.money(v), cls: clsSign },
      { key: "Exit_Type", label: "Exit Type", align: "l",
        fmt: (v, r) => (r.Open ? '<span class="badge warn">OPEN</span> ' : "") + (v || "") },
    ],
    rows: S.gateBlockedRows,
    pageSize: 25, csvName: "gate_blocked_trades.csv",
    defaultSort: { key: "Signal_Date", dir: -1 },
  });

  const prov = (gl.provenance || {}).nogate || {};
  document.getElementById("glCaption").textContent =
    (gl.note ? gl.note + " " : "") +
    "The trade log's 'All trades' filter toggle merges these same blocked rows into every " +
    "filtered metric above. " +
    (prov.build_utc ? `Counterfactual vintage: built ${prov.build_utc} (${prov.source || "?"}).` : "");
}

/* ---------- OVS hold-extension lab: with vs without (ext_lab.json) ---------- */
function renderExtLab() {
  const section = document.getElementById("xlSection");
  const xl = S.extLab;
  if (!xl || !xl.curve || !S.extById.size) {
    section.innerHTML = '<p class="cap">No hold-extension counterfactual payload in this build ' +
      "(the ledger build writes data/backtest_trades_ovsext.parquet; rerun it to populate).</p>";
    return;
  }

  // KPIs — impact = extended − baseline: positive means holding losers to
  // T+5 would have ADDED that much (opposite sign convention to the gate,
  // where the counterfactual is the rule turned OFF).
  const s = xl.summary || {};
  const base = s.baseline || {}, extd = s.extended || {};
  const mb = s.modified_before || {}, ma = s.modified_after || {};
  const impR = (extd.tot_r != null && base.tot_r != null) ? extd.tot_r - base.tot_r : null;
  const imp$ = (extd.pnl_flat != null && base.pnl_flat != null) ? extd.pnl_flat - base.pnl_flat : null;
  document.getElementById("xlKpis").innerHTML = [
    kpiCard("Rebooked trades", mb.n ?? 0, null,
            `losing T+2 time exits · ${xl.n_hit_target ?? 0} hit target by T+5`),
    kpiCard("Rebooked R", mb.tot_r == null || ma.tot_r == null ? "-"
            : `${fmt.signed(mb.tot_r, 1)} → ${fmt.signed(ma.tot_r, 1)}`, clsSign(ma.tot_r),
            ma.win_pct == null ? "" : `win ${fmt.pctRaw(mb.win_pct, 0)} → ${fmt.pctRaw(ma.win_pct, 0)}`),
    kpiCard("Extension impact (R)", impR == null ? "-" : fmt.signed(impR, 1), clsSign(impR),
            "extended − baseline; + = holding helped"),
    kpiCard("Extension impact ($)", imp$ == null ? "-" : fmt.money(imp$), clsSign(imp$),
            "flat $750k, realized"),
  ].join("");

  // cumulative realized PnL, current exit (solid) vs extended (dashed)
  const c = xl.curve;
  const curveEl = document.getElementById("xlCurve");
  if (c.dates && c.dates.length) {
    const cum = arr => { let t = 0; return arr.map(v => +(t += v).toFixed(0)); };
    Plotly.react(curveEl, [
      { x: c.dates, y: cum(c.base_pnl), mode: "lines", name: `${xl.strategy} — current exit`,
        line: { color: PALETTE[0], width: 1.6 } },
      { x: c.dates, y: cum(c.ext_pnl), mode: "lines", name: `${xl.strategy} — losers to T+5`,
        line: { color: "#ffc14d", width: 1.3, dash: "dot" } },
    ], plotLayout({ height: 340, yaxis: { tickformat: "$,.3~s" } }), PLOT_CFG);
  } else {
    curveEl.innerHTML = '<p class="cap">No curve data in this payload.</p>';
  }

  // rebooked-trade table (would-have-been T+5 outcomes)
  makeTable(document.getElementById("xlTable"), {
    columns: [
      { key: "Signal_Date", label: "Signal", align: "l" },
      { key: "Ticker", label: "Ticker", align: "l" },
      { key: "Tier", label: "Tier", align: "l" },
      { key: "Entry_Date", label: "Entry", align: "l" },
      { key: "Exit_Date", label: "New Exit", align: "l" },
      { key: "R", label: "R @T+5", fmt: v => v == null ? "" : fmt.signed(v, 2), cls: clsSign },
      { key: "PnL_flat", label: "PnL ($)", fmt: v => fmt.money(v), cls: clsSign },
      { key: "Exit_Type", label: "Exit Type", align: "l" },
    ],
    rows: [...S.extById.values()],
    pageSize: 25, csvName: "ovs_ext_rebooked_trades.csv",
    defaultSort: { key: "Signal_Date", dir: -1 },
  });

  const prov = (xl.provenance || {}).ovsext || {};
  document.getElementById("xlCaption").textContent =
    (xl.rule ? "Rule: " + xl.rule + " " : "") + (xl.note ? xl.note + " " : "") +
    "The filter bar's 'OVS losers to T+5' toggle swaps these same rebooked exits into every " +
    "filtered metric above. " +
    (prov.build_utc ? `Counterfactual vintage: built ${prov.build_utc} (${prov.source || "?"}).` : "");
}

/* ---------- stop-fill quality (filter-reactive; stopfills.json) ---------- */
function renderStopFills() {
  const section = document.getElementById("sfSection");
  if (!S.stopfills || !S.sfRows) {
    section.innerHTML = '<p class="cap">No stop-fill payload in this build.</p>';
    return;
  }
  const kEl = document.getElementById("sfKpis");
  const chEl = document.getElementById("sfChart");
  const capEl = document.getElementById("sfCaption");
  const toks = tickerTokens();
  const { strategies, dir, from, to } = S.f;
  // Date range matches the trade log's basis (entry date) so the two adjacent
  // sections agree on which trades are in scope; older payloads without
  // entry_date fall back to exit_date.
  const rows = S.sfRows.filter(r => {
    const d = r.entry_date || r.exit_date;
    return strategies.has(r.strategy) &&
      (dir === "All" || r.direction === dir) &&
      (!from || d >= from) && (!to || d <= to) &&
      (!toks || toks.some(tk => (r.ticker || "").toUpperCase().startsWith(tk)));
  });

  const cl = S.stopfills.classifier || {};
  capEl.textContent =
    `Engine fill model: ${cl.slip_bps ?? 3} bps slippage on every stop fill, +${cl.gap_extra_bps ?? 10} bps when ` +
    `the bar gaps through (fill at the open); classifier threshold ${cl.gap_threshold_bps ?? 8} bps beyond the ` +
    "reconstructed stop. Dollars are flat $750k and ignore the risk-panel dials. The Tier filter is not applied " +
    "here (stop rows carry no tier). Ledger is a full backtest rebuild — marginal fills can flicker between vintages.";

  if (!rows.length) {
    kEl.innerHTML = '<p class="cap">No stop exits under the current filters.</p>';
    Plotly.purge(chEl);
    if (!chEl._fullLayout) chEl.innerHTML = "";
    if (S.sfTable) S.sfTable.setRows([]);
    else document.getElementById("sfTable").innerHTML = "";
    return;
  }
  if (chEl.firstChild && !chEl._fullLayout) chEl.innerHTML = "";

  const sum = a => a.reduce((x, y) => x + y, 0);
  const nGap = rows.filter(r => r.gapped).length;
  const slips = rows.map(r => r.slip_r).filter(v => v != null);
  const rs = rows.map(r => r.r).filter(v => v != null);
  kEl.innerHTML = [
    kpiCard("Stop Exits", rows.length.toLocaleString()),
    kpiCard("Gap-Through Rate", fmt.pct(nGap / rows.length, 1), nGap / rows.length > 0.25 ? "neg" : "neu",
            `${nGap} of ${rows.length} gapped`),
    kpiCard("Avg Slip (R)", slips.length ? fmt.num(sum(slips) / slips.length, 3) : "-", "neg",
            "beyond the stop level"),
    kpiCard("Avg Stop R", rs.length ? fmt.num(sum(rs) / rs.length, 2) : "-", "neg"),
    kpiCard("Worst Stop R", rs.length ? fmt.num(Math.min(...rs), 2) : "-", "neg"),
    kpiCard("Cum Slip+Gap Cost", fmt.money(sum(rows.map(r => r.cost_flat || 0))), "neg", "flat $750k, vs fill-at-stop"),
  ].join("");

  // per-strategy aggregates from the filtered rows
  const byStrat = new Map();
  for (const r of rows) {
    if (!byStrat.has(r.strategy)) byStrat.set(r.strategy, { n: 0, gap: 0, slip: 0 });
    const a = byStrat.get(r.strategy);
    a.n++; if (r.gapped) a.gap++; a.slip += (r.slip_r || 0);
  }
  const stats = [...byStrat.entries()]
    .map(([s, a]) => ({ s, n: a.n, rate: a.gap / a.n, slip: a.slip / a.n }))
    .sort((a, b) => b.n - a.n);
  Plotly.react(chEl, [
    { x: stats.map(d => d.s), y: stats.map(d => +(d.rate * 100).toFixed(1)), type: "bar",
      name: "Gap-through %", marker: { color: "#ffc14d" },
      customdata: stats.map(d => d.n),
      hovertemplate: "%{x}: %{y:.1f}% of %{customdata} stops<extra></extra>" },
    { x: stats.map(d => d.s), y: stats.map(d => +d.slip.toFixed(3)), type: "scatter",
      mode: "markers", name: "Avg slip (R)", yaxis: "y2",
      marker: { color: "#4da3ff", size: 9, symbol: "diamond" },
      hovertemplate: "%{x}: %{y:.3f}R avg slip<extra></extra>" },
  ], plotLayout({
    height: 320, hovermode: "closest", bargap: 0.35,
    margin: { b: 90, r: 46 },
    xaxis: { tickangle: 30, tickfont: { size: 9.5 } },
    yaxis: { ticksuffix: "%", title: { text: "Gap-through rate", font: { size: 11 } }, rangemode: "tozero" },
    yaxis2: { overlaying: "y", side: "right", title: { text: "Avg slip (R)", font: { size: 11 } },
              gridcolor: "rgba(0,0,0,0)", rangemode: "tozero" },
  }), PLOT_CFG);

  const tail = rows.slice().sort((a, b) => (a.r ?? 0) - (b.r ?? 0)).slice(0, 12);
  const columns = [
    { key: "exit_date", label: "Exit", align: "l" },
    { key: "strategy", label: "Strategy", align: "l" },
    { key: "ticker", label: "Ticker", align: "l" },
    { key: "direction", label: "Dir", align: "l",
      fmt: v => `<span class="badge ${v === "Short" ? "dirS" : "dirL"}">${v || ""}</span>` },
    { key: "r", label: "R", fmt: v => v == null ? "" : fmt.signed(v, 2), cls: clsSign },
    { key: "slip_r", label: "Slip (R)", fmt: v => v == null ? "" : fmt.num(v, 2), cls: () => "neg" },
    { key: "gapped", label: "Fill", align: "l",
      fmt: v => v ? '<span class="badge warn">GAP</span>' : '<span class="badge off">AT STOP</span>' },
    { key: "cost_flat", label: "Cost ($)", fmt: v => fmt.money(v), cls: () => "neg" },
  ];
  if (S.sfTable) S.sfTable.setRows(tail);
  else S.sfTable = makeTable(document.getElementById("sfTable"), {
    columns, rows: tail, defaultSort: { key: "r", dir: 1 },
  });
}

/* ---------- drawdown anatomy (static full-book; drawdowns.json) ---------- */
function renderDrawdownAnatomy() {
  const section = document.getElementById("ddaSection");
  const dd = S.drawdowns;
  if (!dd || !dd.episodes || !dd.episodes.length) {
    section.innerHTML = '<p class="cap">No drawdowns payload in this build (skipped under --no-mtm).</p>';
    return;
  }
  // episode selector
  const segHost = document.getElementById("ddaSeg");
  segHost.innerHTML = "";
  const lbl = document.createElement("label");
  lbl.textContent = "Episode";
  segHost.appendChild(lbl);
  const sel = document.createElement("select");
  sel.className = "btn";
  dd.episodes.forEach((e, i) => {
    const o = document.createElement("option");
    o.value = String(i);
    o.textContent = `#${i + 1}  -${e.depth_pct.toFixed(1)}%  ${e.peak_date} → ${e.trough_date}` +
                    (e.recovery_date ? "" : "  (unrecovered)");
    sel.appendChild(o);
  });
  sel.addEventListener("change", () => renderDDEpisode(+sel.value));

  // full-book flat underwater curve from the per-strategy daily series
  if (S.sd && S.sd.total_flat && !S.ddaUW) {
    const pnl = S.sd.total_flat, n = pnl.length;
    const uw = new Array(n);
    let eq = START_EQ, peak = START_EQ;
    for (let i = 0; i < n; i++) {
      eq += pnl[i];
      if (eq > peak) peak = eq;
      uw[i] = +((eq - peak) / START_EQ * 100).toFixed(3);
    }
    S.ddaUW = { dates: S.sd.dates, uw };
  }
  segHost.appendChild(sel);
  renderDDEpisode(0);
}

function renderDDEpisode(i) {
  const dd = S.drawdowns;
  const e = dd.episodes[i];
  document.getElementById("ddaKpis").innerHTML = [
    kpiCard("Depth", fmt.money(-Math.abs(e.depth_dollars)), "neg", "flat $750k dollars"),
    kpiCard("Depth %", "-" + fmt.num(e.depth_pct, 1) + "%", "neg", "of the fixed $750k base"),
    kpiCard("Peak → Trough", e.length_td + " td", null, `${e.peak_date} → ${e.trough_date}`),
    kpiCard("Recovery", e.recovery_td == null ? "not yet" : e.recovery_td + " td",
            e.recovery_td == null ? "neg" : null,
            e.recovery_date ? `recovered ${e.recovery_date}` : "still underwater"),
  ].join("");

  // underwater curve with the episode window shaded
  const curveEl = document.getElementById("ddaCurve");
  if (S.ddaUW) {
    const endX = e.recovery_date || S.ddaUW.dates[S.ddaUW.dates.length - 1];
    Plotly.react(curveEl, [{
      x: S.ddaUW.dates, y: S.ddaUW.uw, mode: "lines", name: "Underwater",
      fill: "tozeroy", line: { color: "#ff5d5d", width: 1 }, fillcolor: "rgba(255,93,93,.15)",
    }], plotLayout({
      height: 220, margin: { t: 10 },
      yaxis: { ticksuffix: "%", title: { text: "Underwater (% of $750k)", font: { size: 11 } } },
      shapes: [{ type: "rect", xref: "x", yref: "paper",
                 x0: e.peak_date, x1: endX, y0: 0, y1: 1,
                 fillcolor: "rgba(255,193,77,.12)", line: { color: "rgba(255,193,77,.5)", width: 1 } }],
    }), PLOT_CFG);
  } else {
    curveEl.innerHTML = '<p class="cap">No daily book series in this build — underwater context unavailable.</p>';
  }

  const hbar = (el, items, labelKey, valKey, height) => {
    const node = document.getElementById(el);
    if (!items || !items.length) {
      Plotly.purge(node);
      node.innerHTML = '<p class="cap">No attribution rows for this episode.</p>';
      return;
    }
    if (node.firstChild && !node._fullLayout) node.innerHTML = "";
    const ys = items.map(x => x[labelKey]).reverse();
    const xs = items.map(x => x[valKey]).reverse();
    Plotly.react(node, [{
      y: ys, x: xs, type: "bar", orientation: "h",
      marker: { color: xs.map(v => v >= 0 ? "#00d18f" : "#ff5d5d") },
      hovertemplate: "%{y}: %{x:$,.0f}<extra></extra>",
    }], plotLayout({
      height: height || Math.max(180, 24 * items.length + 70),
      margin: { l: 190, t: 8 }, hovermode: "closest",
      xaxis: { tickformat: "$,.3~s" },
      yaxis: { tickfont: { size: 9.5 }, gridcolor: "rgba(0,0,0,0)" },
    }), PLOT_CFG);
  };
  hbar("ddaStrat", e.strategies, "key", "pnl");
  hbar("ddaSector", e.sectors, "sector", "pnl");

  makeTable(document.getElementById("ddaTrades"), {
    columns: [
      { key: "exit_date", label: "Exit", align: "l" },
      { key: "ticker", label: "Ticker", align: "l" },
      { key: "strategy", label: "Strategy", align: "l" },
      { key: "exit_type", label: "Exit Type", align: "l" },
      { key: "r", label: "R", fmt: v => v == null ? "" : fmt.signed(v, 2), cls: clsSign },
      { key: "pnl_flat", label: "PnL ($)", fmt: v => fmt.money(v), cls: clsSign },
    ],
    rows: e.worst_trades || [],
  });

  document.getElementById("ddaNote").textContent =
    (dd.note ? dd.note + " " : "") +
    "Episodes are detected on the full-book flat-$750k curve at native sizing; they do not respond to page " +
    "filters, the risk-panel dials, or the fragility adjuster.";
}

/* ---------- sector concentration + sector-loss gate (sector_risk.json) ---------- */
function renderSectorRisk() {
  const section = document.getElementById("srSection");
  const sr = S.sectorRisk;
  if (!sr) {
    section.innerHTML = '<p class="cap">No sector-risk payload in this build.</p>';
    return;
  }

  // stacked-area gross exposure timeline (sector keys pre-ordered by lifetime exposure)
  const tlEl = document.getElementById("srTimeline");
  if (sr.exposure && sr.exposure.dates && sr.exposure.dates.length) {
    const secs = Object.keys(sr.exposure.sectors);
    const traces = secs.map((s, i) => ({
      x: sr.exposure.dates, y: sr.exposure.sectors[s],
      stackgroup: "one", mode: "lines", name: s,
      line: { width: 0.5, color: PALETTE[i % PALETTE.length] },
      hovertemplate: `${s}: %{y:.1f}%<extra></extra>`,
    }));
    Plotly.react(tlEl, traces, plotLayout({
      height: 340, yaxis: { ticksuffix: "%" },
      legend: { font: { size: 9.5 } },
    }), PLOT_CFG);
  } else {
    tlEl.innerHTML = '<p class="cap">No sector exposure series in this build.</p>';
  }

  // current open concentration
  const openEl = document.getElementById("srOpen");
  if (sr.open_concentration && sr.open_concentration.length) {
    const oc = sr.open_concentration.slice().reverse();  // desc -> bottom-up for hbar
    Plotly.react(openEl, [{
      y: oc.map(o => o.sector), x: oc.map(o => o.pct), type: "bar", orientation: "h",
      marker: { color: oc.map((_, i) => PALETTE[(oc.length - 1 - i) % PALETTE.length]) },
      customdata: oc.map(o => [o.notional, o.n]),
      hovertemplate: "%{y}: %{x:.1f}% · %{customdata[0]:$,.0f} · %{customdata[1]} pos<extra></extra>",
    }], plotLayout({
      height: Math.max(180, 30 * oc.length + 70), margin: { l: 150, t: 8 },
      hovermode: "closest",
      xaxis: { ticksuffix: "%" }, yaxis: { gridcolor: "rgba(0,0,0,0)" },
    }), PLOT_CFG);
  } else {
    openEl.innerHTML = '<p class="cap">No open positions.</p>';
  }

  // gate telemetry
  const gateEl = document.getElementById("srGate");
  gateEl.innerHTML = "";
  const gate = sr.gate;
  if (gate && gate.strategies && gate.strategies.length) {
    for (const st of gate.strategies) {
      const head = document.createElement("p");
      head.className = "cap";
      head.textContent = `${st.strategy} — trailing ${st.window_td}td realized R by sector; ` +
        `blocks at ${fmt.num(st.max_realized_r, 1)}R or worse (asof ${gate.asof}, next trading day)`;
      gateEl.appendChild(head);
      const tblHost = document.createElement("div");
      gateEl.appendChild(tblHost);
      if (st.sectors && st.sectors.length) {
        makeTable(tblHost, {
          columns: [
            { key: "sector", label: "Sector", align: "l" },
            { key: "r_sum", label: `Trailing ${st.window_td}td R`,
              fmt: v => fmt.signed(v, 2), cls: clsSign },
            { key: "n_exits", label: "Exits" },
            { key: "distance_r", label: "Margin to block",
              fmt: v => v == null ? "" : fmt.num(v, 2) + "R",
              cls: v => v == null ? "" : v <= 0 ? "neg" : v < 0.5 ? "neu" : "pos" },
            { key: "blocked", label: "Gate", align: "l",
              fmt: v => v ? '<span class="badge on">BLOCKED</span>' : '<span class="badge off">CLEAR</span>' },
            // pre-stringified so makeTable's click-sort compares the visible
            // text instead of '[object Object]' arrays
            { key: "exits_str", label: "Contributing exits", align: "l" },
          ],
          rows: st.sectors.map(s => ({
            ...s,
            exits_str: (s.exits || []).map(x => `${x.ticker} ${x.date} ${fmt.signed(x.r, 2)}R`).join(", "),
          })),
        });
      } else {
        tblHost.innerHTML = '<p class="cap">No sector exits in the trailing window — gate clear everywhere.</p>';
      }
      if (st.unknown_exits && st.unknown_exits.length) {
        const u = document.createElement("p");
        u.className = "cap";
        u.textContent = "UNKNOWN-sector exits (pass through, never pooled or gated): " +
          st.unknown_exits.map(x => `${x.ticker} ${x.date} ${fmt.signed(x.r, 2)}R`).join(", ");
        gateEl.appendChild(u);
      }
    }
  } else {
    gateEl.innerHTML = '<p class="cap">No gate telemetry in this build.</p>';
  }

  const prov = sr.provenance || {};
  document.getElementById("srCaption").textContent =
    "Exposure is gross notional at entry, held to exit, as % of the flat $750k base (weekly samples). " +
    "UNKNOWN-sector names are shown as their own bucket and never pooled. " +
    (prov.build_utc
      ? `Ledger vintage: built ${prov.build_utc} (${prov.source || "unknown source"}, ` +
        `${prov.git_sha || "no sha"}, ${prov.rows || "?"} trades). The ledger is a full backtest rebuild — ` +
        "near-threshold gate values can differ from the vintage that gated the morning scan."
      : "Ledger provenance unavailable in this build.");
}
