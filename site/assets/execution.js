/* execution.js — execution-bridge dashboard (TWS-style).

   Layout, top to bottom:
     - connection bar: agent online light + account tabs (Primary / PA) + NLV
     - Positions panel  (live read-only book from the agent) + row actions
     - Open Orders panel (live working orders) + Cancel
     - New Order ticket: entry bracket (+ optional time stop) / flatten / echo
     - Activity: recent commands + results

   Commands execute LIVE when the agent is armed (mode banner amber) and DRY-RUN
   otherwise — the agent decides by AGENT_LIVE_ENABLED + LIVE_TYPES, and every
   mutating action confirms with a LIVE/Dry-run dialog. Positions/orders come from
   book_snapshot.py over the agent's read-only IBKR connection. Static parts render once; the data
   panels refresh every 4s. */
"use strict";

document.addEventListener("DOMContentLoaded", initExecution);

const state = { account: "primary", book: null, status: null };
let pollTimer = null;
let FUT_SPECS = {};   // symbol/alias -> {exchange,multiplier,min_tick,...}; drives the FUT readout
const frontState = { id: null, timer: null, manual: false };   // FUT contract-month auto-resolve

async function initExecution() {
  renderNav("execution.html");
  const el = document.getElementById("content");
  el.innerHTML = shell();
  FUT_SPECS = (await fetchJSONOrNull("assets/futures_specs.json")) || {};   // for the FUT ticket readout
  document.querySelectorAll("[data-acct]").forEach((b) =>
    b.addEventListener("click", () => setAccount(b.dataset.acct)));
  document.getElementById("cmdType").addEventListener("change", syncFields);
  document.getElementById("cmdSend").addEventListener("click", sendTicket);
  document.getElementById("cmdFields").addEventListener("input", updateReadout);
  document.getElementById("fs_go").addEventListener("click", sizeFutures);
  ["fs_symbol", "fs_entry", "fs_stop", "fs_target", "fs_risk", "fs_riskpct"].forEach((id) => {
    const e = document.getElementById(id);
    if (e) e.addEventListener("keydown", (ev) => { if (ev.key === "Enter") sizeFutures(); });
  });
  syncFields();
  await poll();
  pollTimer = setInterval(poll, 4000);
}

function shell() {
  return `
    <div id="modeBanner"></div>
    <div id="connBar"></div>
    <div class="exec-tabs" style="margin:10px 0 4px">
      <button class="btn" data-acct="primary">Primary</button>
      <button class="btn ghost" data-acct="pa">PA</button>
    </div>
    <div id="positions"></div>
    <div id="orders" style="margin-top:14px"></div>

    <div class="card" style="max-width:760px;margin-top:18px">
      <div style="font:700 14px inherit;margin-bottom:4px">New order</div>
      <p class="cap" style="margin:0 0 10px">Bracket: entry limit + stop + target, plus an optional <b>time stop</b> (closes at market 15:59 ET on that date) and <b>entry expiry</b> (the entry order is DAY unless you set a date &mdash; then GTD to that date's close). Submits per the mode banner above &mdash; live when armed.</p>
      <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:8px">
        <label class="cap">Type</label>
        <select id="cmdType">
          <option value="entry_bracket">entry bracket</option>
          <option value="flatten">flatten</option>
          <option value="echo">echo (ping)</option>
        </select>
        <span class="cap">Account: <b id="ticketAcct">primary</b></span>
      </div>
      <div id="cmdFields" style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:8px"></div>
      <div id="ticketReadout" style="font:12px inherit;margin:0 0 10px;min-height:16px"></div>
      <button class="btn" id="cmdSend">Send order</button>
      <span id="cmdMsg" class="cap" style="margin-left:10px"></span>
    </div>

    <div class="card" style="max-width:760px;margin-top:18px">
      <div style="font:700 14px inherit;margin-bottom:4px">Futures sizing <span class="cap" style="display:inline;font-weight:400">&mdash; risk &rarr; contracts + notional (read-only)</span></div>
      <p class="cap" style="margin:0 0 10px">Enter a futures symbol with entry/stop and a risk budget; the agent sizes the contract count off the live multiplier and shows the notional exposure. Places nothing. Risk % uses the selected account's NLV.</p>
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px">
        <label class="cap">Symbol</label><input id="fs_symbol" value="ES" style="width:70px;text-transform:uppercase">
        <label class="cap">Entry</label><input id="fs_entry" style="width:78px">
        <label class="cap">Stop</label><input id="fs_stop" style="width:78px">
        <label class="cap">Target</label><input id="fs_target" style="width:78px">
        <label class="cap">Risk $</label><input id="fs_risk" style="width:78px">
        <label class="cap">or %</label><input id="fs_riskpct" style="width:52px">
        <button class="btn" id="fs_go">Size</button>
        <span id="fs_msg" class="cap"></span>
      </div>
      <div id="fs_result"></div>
    </div>

    <div id="activity" style="margin-top:18px"></div>`;
}

function setAccount(acct) {
  state.account = acct;
  document.querySelectorAll("[data-acct]").forEach((b) =>
    b.className = "btn" + (b.dataset.acct === acct ? "" : " ghost"));
  const ta = document.getElementById("ticketAcct");
  if (ta) ta.textContent = acct;
  renderPanels();
}

function acctBook() {
  const accs = (state.book && state.book.accounts) || [];
  return accs.find((a) => a.key === state.account) || null;
}

async function poll() {
  const [s, b, c] = await Promise.all([
    fetchJSONOrNull("/exec-status"),
    fetchJSONOrNull("/exec-book"),
    fetchJSONOrNull("/exec-commands"),
  ]);
  state.status = s || { online: false, configured: false };
  state.book = (b && b.book) || null;
  state.commands = (c && c.commands) || [];
  setAsof(state.status.online ? "execution online" : state.status.configured ? "execution offline" : "broker not configured");
  renderPanels();
}

function renderPanels() {
  set("modeBanner", renderModeBanner());
  set("connBar", renderConnBar());
  set("positions", renderPositions());
  set("orders", renderOrders());
  set("activity", renderActivity());
}
function set(id, html) { const el = document.getElementById(id); if (el) el.innerHTML = html; }

/* ---------- mode banner (dry-run vs live) ---------- */
function renderModeBanner() {
  const mode = (state.book && state.book.mode) || "dry-run";
  if (mode === "live") {
    return `<div class="card" style="border-color:#a8852f;background:rgba(255,193,77,.10);padding:9px 14px;font:700 13px inherit;color:#ffc14d">
      &#9888;&#65039; LIVE ARMED &mdash; orders ARE transmitted to IBKR.</div>`;
  }
  return `<div class="card" style="border-color:#2c8f63;background:rgba(61,219,143,.08);padding:9px 14px;font:700 13px inherit;color:#3ddb8f">
    &#9679; DRY-RUN MODE &mdash; actions are validated and previewed, but <u>nothing is transmitted</u> to IBKR.</div>`;
}

/* ---------- connection bar ---------- */
function dot(c) { return `<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${c};box-shadow:0 0 8px ${c}"></span>`; }
function renderConnBar() {
  const s = state.status || {};
  const tone = !s.configured ? "#9aa3b2" : s.online ? "#3ddb8f" : "#ff6b6b";
  const label = !s.configured ? "Broker not configured" : s.online ? "Execution online" : "Execution offline";
  const ab = acctBook();
  const nlv = ab && ab.nlv != null ? `NLV ${fmt.money(ab.nlv)}` : "";
  const age = state.book && state.book.at ? `· book ${Math.max(0, Math.round((Date.now() - state.book.at) / 1000))}s ago` : "";
  return `<div class="card" style="display:flex;align-items:center;gap:12px;padding:10px 14px">
    <span style="font:700 15px inherit;display:flex;align-items:center;gap:8px">${dot(tone)} ${label}</span>
    <span class="cap" style="margin-left:auto">${nlv} ${age}</span></div>`;
}

/* ---------- positions ---------- */
function pnlPct(p) {
  const cost = p.avg_cost != null && p.position ? Math.abs(p.avg_cost * p.position) : null;
  return cost && p.unrealized_pnl != null ? p.unrealized_pnl / cost : null;
}
function renderPositions() {
  const ab = acctBook();
  const head = `<div style="font:700 14px inherit;margin:0 0 6px">Positions <span class="cap" style="display:inline;font-weight:400">${ab && ab.label ? "· " + esc(ab.label) : ""}</span></div>`;
  if (!ab) return head + panelNote("No book yet — the agent publishes positions when it's online and TWS is up.");
  if (ab.error) return head + panelNote(`${esc(ab.label)}: ${esc(ab.error)}.`);
  const pos = ab.positions || [];
  if (!pos.length) return head + panelNote("Flat — no open positions.");
  const rows = pos.map((p) => {
    const long = (p.position || 0) > 0;
    const pct = pnlPct(p);
    const sym = p.sec_type === "FUT" && p.expiry ? `${esc(p.symbol)} <span class="cap" style="display:inline">${esc(p.expiry)}</span>` : esc(p.symbol);
    return `<tr>
      <td class="l" style="font-weight:600">${sym}</td>
      <td class="${long ? "pos" : "neg"}" style="font-weight:600">${fmt.num(p.position, 0)}</td>
      <td>${fmt.num(p.avg_cost, 2)}</td>
      <td>${p.market_price != null ? fmt.num(p.market_price, 2) : "&mdash;"}</td>
      <td>${p.market_value != null ? fmt.money(p.market_value) : "&mdash;"}</td>
      <td class="${clsSign(p.unrealized_pnl)}" style="font-weight:600">${p.unrealized_pnl != null ? fmt.money(p.unrealized_pnl) : "&mdash;"}</td>
      <td class="${clsSign(pct)}">${pct != null ? fmt.pct(pct, 1) : "&mdash;"}</td>
      <td class="l" style="white-space:nowrap">
        <button class="btn xs" onclick='execFlatten(${posJson(p)},1)'>Flatten</button>
        <button class="btn xs ghost" onclick='execFlatten(${posJson(p)},0.5)'>Trim&frac12;</button>
      </td></tr>`;
  }).join("");
  return head + `<div class="tblwrap"><table class="tbl"><thead><tr>
    <th class="l">Symbol</th><th>Pos</th><th>Avg</th><th>Last</th><th>Mkt Val</th><th>uP&amp;L $</th><th>uP&amp;L %</th><th class="l">Actions</th>
    </tr></thead><tbody>${rows}</tbody></table></div>`;
}

/* ---------- open orders ---------- */
function orderPx(o) {
  const t = String(o.order_type || "").toUpperCase();
  if (t.startsWith("STP")) return o.aux != null ? o.aux : o.lmt;   // stop trigger, not the 0.00 lmt
  if (o.lmt) return o.lmt;
  if (o.aux) return o.aux;
  return null;                                                      // MKT / MOC / MOO — no price
}
function fmtOrderTime(s) {
  if (!s) return "";
  const m = String(s).match(/(\d{4})(\d{2})(\d{2})[ -](\d{2}):(\d{2})/);
  return m ? `${m[2]}/${m[3]} ${m[4]}:${m[5]}` : esc(String(s));    // MM/DD HH:MM
}
function orderRow(o) {
  const buy = String(o.action).toUpperCase() === "BUY";
  const px = orderPx(o);
  return `<tr>
    <td class="l" style="font-weight:600">${esc(o.symbol)}</td>
    <td class="l ${buy ? "pos" : "neg"}" style="font-weight:600">${esc(o.action)}</td>
    <td>${fmt.num(o.qty, 0)}</td>
    <td class="l">${esc(o.order_type)}</td>
    <td>${px != null ? fmt.num(px, 2) : "&mdash;"}</td>
    <td class="l">${esc(o.tif || "")}</td>
    <td class="l" style="color:#8c95a2">${fmtOrderTime(o.good_after) || "&mdash;"}</td>
    <td class="l" style="color:#8c95a2">${fmtOrderTime(o.good_till) || "&mdash;"}</td>
    <td class="l" style="color:#8c95a2">${esc(o.status || "")}</td>
    <td class="l"><button class="btn xs ghost" onclick='execCancel(${o.perm_id || 0},${o.order_id || 0},"${esc(o.symbol)}")'>Cancel</button></td>
  </tr>`;
}
const expandedTickers = new Set();   // Open Orders: which tickers are expanded (persists across 4s polls)
function toggleOrderGroup(sym) {
  const k = String(sym).toUpperCase();
  if (expandedTickers.has(k)) expandedTickers.delete(k); else expandedTickers.add(k);
  set("orders", renderOrders());
}
window.toggleOrderGroup = toggleOrderGroup;
function typeRank(o) {                                    // sort order within a ticker
  const t = String(o.order_type || "").toUpperCase();
  if (t.startsWith("STP")) return 1;                     // stops after limit legs
  if (t === "MKT" || t.startsWith("MO")) return 2;       // MKT / MOC / MOO time-stops last
  return 0;                                              // LMT (entry / target) first
}
function orderLegPreview(o) {
  const px = orderPx(o);
  return px != null ? `${esc(o.order_type || "")} ${fmt.num(px, 2)}` : esc(o.order_type || "");
}
function pnlSpan(label, v) {
  return v == null ? `${label} &mdash;` : `${label} <b class="${clsSign(v)}">${fmt.money(v)}</b>`;
}
// Best case (all profit-target LMTs fill) / worst case (all stops fill), per ticker.
// entryPrice is derived from live PnL (avoids the futures averageCost-includes-multiplier
// ambiguity); pending groups reference the parent entry limit. mult from sec_type (FUT only).
function groupPnl(sym, legs, ab) {
  const pos = (ab.positions || []).find((p) => String(p.symbol).toUpperCase() === sym && p.position);
  let entryPrice, sign, mult, exits;
  if (pos) {
    mult = pos.sec_type === "FUT" ? ((futSpec(sym) || {}).multiplier || 1) : 1;
    sign = pos.position > 0 ? 1 : -1;
    if (pos.market_price != null && pos.unrealized_pnl != null) {
      entryPrice = pos.market_price - pos.unrealized_pnl / (pos.position * mult);
    } else if (pos.avg_cost != null) {
      entryPrice = pos.avg_cost / mult;
    } else return { best: null, worst: null };
    exits = legs;
  } else {
    const entry = legs.find((o) => (o.parent_id || 0) === 0);
    const ep = entry ? orderPx(entry) : null;
    if (ep == null) return { best: null, worst: null };
    mult = entry.sec_type === "FUT" ? ((futSpec(sym) || {}).multiplier || 1) : 1;
    sign = String(entry.action).toUpperCase() === "BUY" ? 1 : -1;
    entryPrice = ep;
    exits = legs.filter((o) => o !== entry);
  }
  let best = null, worst = null;
  for (const o of exits) {
    const t = String(o.order_type || "").toUpperCase();
    const q = o.qty || 0;
    if (t.startsWith("STP")) {
      const px = o.aux != null ? o.aux : o.lmt;
      if (px != null) worst = (worst || 0) + (px - entryPrice) * mult * sign * q;
    } else if (t === "LMT" && o.lmt != null) {
      best = (best || 0) + (o.lmt - entryPrice) * mult * sign * q;
    }
  }
  return { best, worst };
}
function ordersSection(title, list, ab) {
  const h = `<div class="cap" style="font-weight:700;margin:10px 0 4px">${title} <span style="font-weight:400">&middot; ${list.length}</span></div>`;
  if (!list.length) return h + `<div class="cap" style="color:#8c95a2;margin-bottom:6px">(none)</div>`;
  const groups = new Map();
  for (const o of list) {
    const k = String(o.symbol).toUpperCase();
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(o);
  }
  let body = "";
  for (const sym of [...groups.keys()].sort()) {
    const legs = groups.get(sym).slice()
      .sort((a, b) => typeRank(a) - typeRank(b) || (a.order_id || 0) - (b.order_id || 0));
    const open = expandedTickers.has(sym);
    const caret = open ? "&#9662;" : "&#9656;";          // triangle down / right
    const preview = open ? "" : legs.map(orderLegPreview).join(" &middot; ");
    const bw = groupPnl(sym, legs, ab);
    const bwFrag = (bw.best != null || bw.worst != null)
      ? ` &nbsp;&middot;&nbsp; ${pnlSpan("best", bw.best)} &middot; ${pnlSpan("worst", bw.worst)}`
      : "";
    body += `<tr style="cursor:pointer;background:rgba(255,255,255,.03)" onclick="toggleOrderGroup('${esc(sym)}')">
      <td class="l" colspan="10" style="font-weight:600">${caret} ${esc(sym)}
        <span class="cap" style="font-weight:400;display:inline">&nbsp;(${legs.length})${preview ? " &nbsp;&middot;&nbsp; " + preview : ""}${bwFrag}</span></td></tr>`;
    if (open) body += legs.map(orderRow).join("");
  }
  return h + `<div class="tblwrap"><table class="tbl"><thead><tr>
    <th class="l">Symbol</th><th class="l">Side</th><th>Qty</th><th class="l">Type</th><th>Price</th><th class="l">TIF</th><th class="l">Start</th><th class="l">End</th><th class="l">Status</th><th class="l"></th>
    </tr></thead><tbody>${body}</tbody></table></div>`;
}
function renderOrders() {
  const ab = acctBook();
  const head = `<div style="font:700 14px inherit;margin:0 0 6px">Open Orders</div>`;
  if (!ab || ab.error) return head + panelNote("&mdash;");
  const ords = ab.orders || [];
  if (!ords.length) return head + panelNote("No working orders.");
  // Split by whether the symbol has an open position: exits-on-positions vs pending entries.
  const posSyms = new Set((ab.positions || []).map((p) => String(p.symbol).toUpperCase()));
  const onPos = ords.filter((o) => posSyms.has(String(o.symbol).toUpperCase()));
  const pending = ords.filter((o) => !posSyms.has(String(o.symbol).toUpperCase()));
  return head
    + ordersSection("On open positions (working exits)", onPos, ab)
    + ordersSection("Pending entries — not filled yet", pending, ab);
}

function panelNote(html) { return `<div class="card" style="padding:12px 14px"><span class="cap">${html}</span></div>`; }
function posJson(p) {
  return JSON.stringify({ symbol: p.symbol, sec_type: p.sec_type, expiry: p.expiry }).replace(/'/g, "&#39;");
}

/* ---------- row actions (dry-run commands) ---------- */
function isLive() { return !!(state.book && state.book.mode === "live"); }
function actionLead(verb) { return isLive() ? `⚠️ LIVE — ${verb}` : `Dry-run: ${verb}`; }

function execFlatten(pos, fraction) {
  const pct = fraction === 1 ? "100%" : "50%";
  if (!confirm(`${actionLead("flatten")} ${pct} of ${pos.symbol} (${state.account})? Cancels its working orders first.`)) return;
  sendCommand("flatten", { symbol: pos.symbol, sec_type: pos.sec_type, expiry: pos.expiry, fraction, order_type: "MKT" });
}
function execCancel(permId, orderId, symbol) {
  if (!confirm(`${actionLead("cancel")} order ${orderId} (${symbol}, ${state.account})?`)) return;
  sendCommand("cancel", permId ? { scope: "order", perm_id: permId, order_id: orderId } : { scope: "symbol", symbol });
}
window.execFlatten = execFlatten;
window.execCancel = execCancel;

/* ---------- new-order ticket ---------- */
function inp(id, val, w) { return `<input id="${id}" value="${val}" style="width:${w || 90}px">`; }
function syncFields() {
  const t = document.getElementById("cmdType").value;
  const f = document.getElementById("cmdFields");
  if (t === "echo") {
    f.innerHTML = `<label class="cap">Note</label>${inp("f_note", "ping from site", 200)}`;
  } else if (t === "flatten") {
    f.innerHTML = `<label class="cap">Symbol</label>${inp("f_symbol", "USO", 90)}
      <label class="cap">Fraction</label><select id="f_frac"><option value="1">100%</option><option value="0.5">50%</option></select>`;
  } else {
    f.innerHTML = `<label class="cap">Instr</label><select id="f_sectype"><option value="STK">Stock</option><option value="FUT">Future</option></select>
      <label class="cap">Symbol</label>${inp("f_symbol", "USO", 80)}
      <label class="cap">Side</label><select id="f_action"><option>BUY</option><option>SELL</option></select>
      <label class="cap">Qty</label>${inp("f_qty", "692", 70)}
      <label class="cap">Entry</label>${inp("f_entry", "104.80", 80)}
      <label class="cap">Stop</label>${inp("f_stop", "103.29", 80)}
      <label class="cap">Target</label>${inp("f_target", "123.21", 80)}
      <span id="f_futrow"></span>
      <label class="cap">Entry exp</label><input type="date" id="f_expiry" style="width:140px">
      <label class="cap">Time stop</label><input type="date" id="f_timestop" style="width:140px">`;
    const st = document.getElementById("f_sectype");
    if (st) st.addEventListener("change", () => {
      if (val("f_sectype") === "FUT" && !futSpec(val("f_symbol"))) {
        const s = document.getElementById("f_symbol"); if (s) s.value = "ES";   // sensible FUT default
      }
      frontState.manual = false;
      renderFutRow(); updateReadout(); scheduleFrontResolve();
    });
    const sym = document.getElementById("f_symbol");
    if (sym) sym.addEventListener("input", () => { frontState.manual = false; scheduleFrontResolve(); });
    renderFutRow();
  }
  updateReadout();
}

function futSpec(sym) { return FUT_SPECS[String(sym || "").toUpperCase().trim()] || null; }
function renderFutRow() {
  const row = document.getElementById("f_futrow");
  if (!row) return;
  if (val("f_sectype") !== "FUT") { row.innerHTML = ""; return; }
  row.innerHTML = `<label class="cap">Contract</label><input id="f_futexp" placeholder="auto" style="width:78px">
    <span id="f_futhint" class="cap" style="display:inline"></span>`;
  const exp = document.getElementById("f_futexp");
  if (exp) exp.addEventListener("input", () => { frontState.manual = true; });   // stop auto-fill once typed
}

function updateReadout() {
  const t = document.getElementById("cmdType").value;
  const el = document.getElementById("ticketReadout");
  if (!el) return;
  if (t === "entry_bracket") {
    const isFut = (val("f_sectype") === "FUT");
    const sym = String(val("f_symbol") || "").toUpperCase().trim();
    const spec = isFut ? futSpec(sym) : null;
    const mult = spec ? spec.multiplier : 1;
    const hint = document.getElementById("f_futhint");
    if (hint) hint.innerHTML = spec
      ? `${esc(spec.exchange)} · mult ${spec.multiplier} · tick ${spec.min_tick}`
      : (sym ? `<span style="color:#ff6b6b">not in spec table</span>` : "");
    const qty = Number(val("f_qty")), entry = Number(val("f_entry")), stop = Number(val("f_stop")), target = Number(val("f_target"));
    const action = val("f_action"), dist = Math.abs(entry - stop);
    let warn = "";
    if (action === "BUY" && !(stop < entry && entry < target)) warn = "BUY needs stop &lt; entry &lt; target";
    if (action === "SELL" && !(target < entry && entry < stop)) warn = "SELL needs target &lt; entry &lt; stop";
    if (isFut && sym && !spec) warn = "unknown futures symbol — not in the spec table";
    if (isFut && !val("f_futexp")) warn = warn || "enter the contract month (e.g. 202609)";
    if (warn) { el.innerHTML = `<span style="color:#ff6b6b">${warn}</span>`; return; }
    const parts = [];
    if (isFut && qty) parts.push(`<b>${qty} contract${qty === 1 ? "" : "s"}</b>`);
    if (qty && dist) parts.push(`Risk <b>${fmt.money(qty * dist * mult)}</b>`);
    if (dist && target) parts.push(`R:R <b>${(Math.abs(target - entry) / dist).toFixed(2)}:1</b>`);
    if (qty && entry) parts.push(`Notional <b>${fmt.money(qty * entry * mult)}</b>`);
    const ts = val("f_timestop");
    if (ts) parts.push(`Time-exit <b>${ts}</b>`);
    const ex = val("f_expiry");
    parts.push(`Entry <b>${ex ? "GTD " + ex : "DAY"}</b>`);
    el.innerHTML = `<span style="color:#9aa3b2">${parts.join(" &nbsp;·&nbsp; ")}</span>`;
  } else if (t === "flatten") {
    const sym = String(val("f_symbol") || "").toUpperCase();
    const ab = acctBook();
    const pos = ((ab && ab.positions) || []).find((p) => String(p.symbol).toUpperCase() === sym);
    el.innerHTML = pos
      ? `<span style="color:#9aa3b2">Current position: <b>${fmt.num(pos.position, 0)}</b> ${esc(sym)} (${state.account})</span>`
      : `<span style="color:#ffc14d">No ${esc(sym || "?")} position in ${state.account}</span>`;
  } else {
    el.innerHTML = "";
  }
}
function val(id) { const e = document.getElementById(id); return e ? e.value : undefined; }
function ticketPayload(t) {
  if (t === "echo") return { note: val("f_note") };
  if (t === "flatten") return { symbol: val("f_symbol"), fraction: Number(val("f_frac")), order_type: "MKT" };
  const sec_type = val("f_sectype") || "STK";
  const fut_expiry = sec_type === "FUT" ? String(val("f_futexp") || "").replace(/\D/g, "") : null;
  return { symbol: val("f_symbol"), sec_type, fut_expiry, action: val("f_action"), quantity: Number(val("f_qty")),
    entry: Number(val("f_entry")), stop: Number(val("f_stop")), target: Number(val("f_target")),
    time_stop: val("f_timestop") || null, expiry: val("f_expiry") || null };
}
function sendTicket() {
  const t = document.getElementById("cmdType").value;
  const p = ticketPayload(t);
  if (t === "entry_bracket" && p.sec_type === "FUT" && !p.fut_expiry) {
    document.getElementById("cmdMsg").textContent = "enter the contract month (e.g. 202609)";
    return;
  }
  if (t !== "echo") {
    const inst = p.sec_type === "FUT" ? `${p.symbol} FUT ${p.fut_expiry}` : p.symbol;
    const summary = t === "entry_bracket"
      ? `${p.action} ${p.quantity} ${inst} @ ${p.entry} [${p.expiry ? "GTD " + p.expiry : "DAY"}] (stop ${p.stop}, target ${p.target}${p.time_stop ? ", time " + p.time_stop : ""})`
      : `flatten ${Math.round((p.fraction || 1) * 100)}% of ${p.symbol}`;
    if (!confirm(`${actionLead(t === "entry_bracket" ? "place" : "flatten")} ${summary} on ${state.account}?`)) return;
  }
  sendCommand(t, p, "cmdMsg");
}

async function sendCommand(type, payload, msgId) {
  const msg = msgId ? document.getElementById(msgId) : null;
  if (msg) msg.textContent = "sending...";
  try {
    const r = await fetch("/exec-command", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type, account: state.account, payload }),
    });
    const d = await r.json();
    if (msg) msg.textContent = d.ok ? `queued ${(d.id || "").slice(0, 8)}` : `error: ${d.error || ("HTTP " + r.status)}`;
  } catch (e) {
    if (msg) msg.textContent = "error: " + e;
  }
  setTimeout(poll, 600);
}

/* ---------- futures sizing (read-only: risk -> contracts + notional) ---------- */
const sizeState = { id: null, timer: null };
async function sizeFutures() {
  const msg = document.getElementById("fs_msg");
  const symbol = String(val("fs_symbol") || "").toUpperCase().trim();
  if (!symbol) { msg.textContent = "symbol required"; return; }
  const entry = Number(val("fs_entry")), stop = Number(val("fs_stop"));
  if (!entry || !stop) { msg.textContent = "entry and stop required"; return; }
  const target = val("fs_target") ? Number(val("fs_target")) : null;
  const risk = val("fs_risk") ? Number(val("fs_risk")) : null;
  const risk_pct = val("fs_riskpct") ? Number(val("fs_riskpct")) : null;
  if (risk == null && risk_pct == null) { msg.textContent = "enter risk $ or %"; return; }
  msg.textContent = "sizing…";
  clearTimeout(sizeState.timer);
  try {
    const r = await fetch("/exec-futures-size", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbol, entry, stop, target, risk, risk_pct, account_key: state.account }),
    });
    const d = await r.json();
    if (!d.ok) { msg.textContent = "error: " + (d.error || ("HTTP " + r.status)); return; }
    sizeState.id = d.id;
    pollSize(0);
  } catch (e) { msg.textContent = "error: " + e; }
}
async function pollSize(n) {
  if (n > 30) { document.getElementById("fs_msg").textContent = "timed out — is the agent online?"; return; }
  const d = (await fetchJSONOrNull("/exec-futures-size")) || {};
  const q = d.query;
  if (q && q.id === sizeState.id && q.result) {
    document.getElementById("fs_msg").textContent = "";
    renderSize(q.result);
    return;
  }
  sizeState.timer = setTimeout(() => pollSize(n + 1), 1500);
}
function renderSize(data) {
  const el = document.getElementById("fs_result");
  if (!data || data.error) {
    el.innerHTML = `<div class="card" style="padding:10px 14px"><span class="neg">${esc((data && data.error) || "no result")}</span></div>`;
    return;
  }
  const notPct = data.notional_pct != null ? ` · ${fmt.num(data.notional_pct, 1)}% of acct (${fmt.num(data.leverage, 2)}x)` : "";
  const rr = data.rr != null ? ` · R:R ${fmt.num(data.rr, 2)}:1` : "";
  const note = data.note ? `<div class="cap" style="color:#ffc14d;margin-top:8px">${esc(data.note)}</div>` : "";
  el.innerHTML = `<div class="card" style="padding:12px 14px">
    <span style="font:700 16px inherit">${esc(data.symbol)} ${esc(data.action)} &mdash;
      <span style="color:#4da3ff">${data.contracts} contract${data.contracts === 1 ? "" : "s"}</span></span>
    <div class="kv" style="margin-top:10px">
      <div class="k">Risk / contract</div><div class="v">${fmt.money(data.risk_per_contract)} (${fmt.num(data.stop_ticks, 0)} ticks)</div>
      <div class="k">Total risk</div><div class="v">${fmt.money(data.total_risk)} <span class="cap" style="display:inline">/ budget ${fmt.money(data.risk_budget)}</span></div>
      <div class="k">Total notional</div><div class="v">${fmt.money(data.total_notional)}${notPct}</div>
      <div class="k">Multiplier</div><div class="v">${fmt.num(data.multiplier, 2)} <span class="cap" style="display:inline">· tick ${data.min_tick}${rr}</span></div>
    </div>${note}</div>`;
}

/* ---------- futures contract-month auto-resolve (read-only reqContractDetails) ---------- */
function scheduleFrontResolve() {
  if (val("f_sectype") !== "FUT") return;
  clearTimeout(frontState.timer);
  frontState.timer = setTimeout(resolveFront, 500);   // debounce while the symbol is typed
}
async function resolveFront() {
  const symbol = String(val("f_symbol") || "").toUpperCase().trim();
  if (!symbol || !futSpec(symbol)) return;            // unknown symbol: the readout already warns
  const exp = document.getElementById("f_futexp");
  try {
    const r = await fetch("/exec-futures-front", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbol }),
    });
    const d = await r.json();
    if (!d.ok) return;
    frontState.id = d.id;
    if (exp && !frontState.manual && !exp.value) exp.placeholder = "resolving…";
    pollFront(0);
  } catch (e) { /* silent — the field stays manual/editable */ }
}
async function pollFront(n) {
  const exp = document.getElementById("f_futexp");
  if (n > 20) { if (exp) exp.placeholder = "202609"; return; }
  const d = (await fetchJSONOrNull("/exec-futures-front")) || {};
  const q = d.query;
  if (q && q.id === frontState.id && q.result) {
    const res = q.result;
    const cur = String(val("f_symbol") || "").toUpperCase().trim();
    if (exp && res.expiry && !res.error && !frontState.manual && cur === res.symbol) {
      exp.value = res.expiry; exp.placeholder = "auto"; updateReadout();
    } else if (exp) { exp.placeholder = "202609"; }
    return;
  }
  setTimeout(() => pollFront(n + 1), 1200);
}

/* ---------- activity ---------- */
function stateBadge(state) {
  const map = { dry_run: ["#3ddb8f", "DRY-RUN OK"], rejected: ["#ff6b6b", "REJECTED"],
                executed: ["#ffc14d", "EXECUTED"], duplicate: ["#9aa3b2", "duplicate"],
                pushed: ["#9aa3b2", "pushed"], error: ["#ffc14d", "ERROR"] };
  const [c, t] = map[state] || ["#9aa3b2", state || ""];
  return `<span style="color:${c};font-weight:600">${esc(t)}</span>`;
}
function resultCell(c) {
  const res = c.result || {};
  const tone = res.ok === true ? "pos" : res.ok === false ? "neg" : "neu";
  let html = `<span class="${tone}">${esc(res.detail || c.state || "pending")}</span>`;
  const pv = res.preview || {};
  if (pv.legs && pv.legs.length) {
    html += `<div class="exec-legs">${pv.legs.map((l) => esc(l)).join("<br>")}` +
      (pv.summary ? `<br><span style="color:#c7ccd6;font-weight:600">${esc(pv.summary)}</span>` : "") + `</div>`;
  }
  if (res.fill) {
    const f = res.fill;
    html += `<div class="exec-legs" style="color:#ffc14d">filled ${esc(String(f.filled ?? "?"))} @ ${esc(String(f.avg_fill ?? "—"))} · #${esc(String(f.order_id ?? ""))} (${esc(String(f.status ?? ""))})</div>`;
  }
  return html;
}
function clockTime(ms) {
  if (!ms) return "";
  try { return new Date(ms).toLocaleTimeString("en-US", { hour12: false }); } catch (e) { return ""; }
}
function renderActivity() {
  const cmds = state.commands || [];
  const liveTrail = (state.book && state.book.mode === "live");
  if (!cmds.length) return "";
  const rows = cmds.map((c) => `<tr style="vertical-align:top">
      <td class="l" style="color:#8c95a2">${esc(clockTime(c.created_at))}</td>
      <td class="l" style="font-weight:600">${esc(c.type || "")}</td>
      <td class="l">${esc(c.account || "")}</td>
      <td class="l">${stateBadge(c.state)}</td>
      <td class="l">${resultCell(c)}</td></tr>`).join("");
  return `<div style="font:700 14px inherit;margin-bottom:6px">Activity / audit
      <span class="cap" style="display:inline;font-weight:400">· ${liveTrail ? "LIVE" : "dry-run, places nothing"} · last ${cmds.length}</span></div>
    <div class="tblwrap"><table class="tbl"><thead><tr>
    <th class="l">time</th><th class="l">type</th><th class="l">acct</th><th class="l">state</th><th class="l">result / order preview</th>
    </tr></thead><tbody>${rows}</tbody></table></div>`;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
