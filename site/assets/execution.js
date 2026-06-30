/* execution.js — execution-bridge dashboard (TWS-style, read-only / dry-run).

   Layout, top to bottom:
     - connection bar: agent online light + account tabs (Primary / PA) + NLV
     - Positions panel  (live read-only book from the agent) + row actions
     - Open Orders panel (live working orders) + Cancel
     - New Order ticket (dry-run): echo / flatten / entry_bracket
     - Activity: recent commands + the agent's "would do" results

   Every action sends a DRY-RUN command (the agent logs what it would do and
   places nothing). Positions/orders come from book_snapshot.py over the agent's
   read-only IBKR connection. Static parts (tabs, ticket) render once; the data
   panels refresh every 4s. */
"use strict";

document.addEventListener("DOMContentLoaded", initExecution);

const state = { account: "primary", book: null, status: null };
let pollTimer = null;

async function initExecution() {
  renderNav("execution.html");
  const el = document.getElementById("content");
  el.innerHTML = shell();
  document.querySelectorAll("[data-acct]").forEach((b) =>
    b.addEventListener("click", () => setAccount(b.dataset.acct)));
  document.getElementById("cmdType").addEventListener("change", syncFields);
  document.getElementById("cmdSend").addEventListener("click", sendTicket);
  syncFields();
  await poll();
  pollTimer = setInterval(poll, 4000);
}

function shell() {
  return `
    <div id="connBar"></div>
    <div class="exec-tabs" style="margin:10px 0 4px">
      <button class="btn" data-acct="primary">Primary</button>
      <button class="btn ghost" data-acct="pa">PA</button>
    </div>
    <div id="positions"></div>
    <div id="orders" style="margin-top:14px"></div>

    <div class="card" style="max-width:760px;margin-top:18px">
      <div style="font:700 14px inherit;margin-bottom:4px">New order &mdash; DRY-RUN</div>
      <p class="cap" style="margin:0 0 10px">The agent validates and reports what it <b>would</b> submit. Nothing reaches IBKR.</p>
      <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:8px">
        <label class="cap">Type</label>
        <select id="cmdType">
          <option value="entry_bracket">entry bracket</option>
          <option value="flatten">flatten</option>
          <option value="echo">echo (ping)</option>
        </select>
        <span class="cap">Account: <b id="ticketAcct">primary</b></span>
      </div>
      <div id="cmdFields" style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:10px"></div>
      <button class="btn" id="cmdSend">Send dry-run</button>
      <span id="cmdMsg" class="cap" style="margin-left:10px"></span>
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
  set("connBar", renderConnBar());
  set("positions", renderPositions());
  set("orders", renderOrders());
  set("activity", renderActivity());
}
function set(id, html) { const el = document.getElementById(id); if (el) el.innerHTML = html; }

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
function renderOrders() {
  const ab = acctBook();
  const head = `<div style="font:700 14px inherit;margin:0 0 6px">Open Orders</div>`;
  if (!ab || ab.error) return head + panelNote("&mdash;");
  const ords = ab.orders || [];
  if (!ords.length) return head + panelNote("No working orders.");
  const rows = ords.map((o) => {
    const buy = String(o.action).toUpperCase() === "BUY";
    const px = o.lmt != null ? o.lmt : (o.aux != null ? o.aux : null);
    return `<tr>
      <td class="l" style="font-weight:600">${esc(o.symbol)}</td>
      <td class="l ${buy ? "pos" : "neg"}" style="font-weight:600">${esc(o.action)}</td>
      <td>${fmt.num(o.qty, 0)}</td>
      <td class="l">${esc(o.order_type)}</td>
      <td>${px != null ? fmt.num(px, 2) : "&mdash;"}</td>
      <td class="l">${esc(o.tif || "")}</td>
      <td class="l" style="color:#8c95a2">${esc(o.status || "")}</td>
      <td class="l"><button class="btn xs ghost" onclick='execCancel(${o.order_id || 0},"${esc(o.symbol)}")'>Cancel</button></td>
    </tr>`;
  }).join("");
  return head + `<div class="tblwrap"><table class="tbl"><thead><tr>
    <th class="l">Symbol</th><th class="l">Side</th><th>Qty</th><th class="l">Type</th><th>Price</th><th class="l">TIF</th><th class="l">Status</th><th class="l"></th>
    </tr></thead><tbody>${rows}</tbody></table></div>`;
}

function panelNote(html) { return `<div class="card" style="padding:12px 14px"><span class="cap">${html}</span></div>`; }
function posJson(p) {
  return JSON.stringify({ symbol: p.symbol, sec_type: p.sec_type, expiry: p.expiry }).replace(/'/g, "&#39;");
}

/* ---------- row actions (dry-run commands) ---------- */
function execFlatten(pos, fraction) {
  const pct = fraction === 1 ? "100%" : "50%";
  if (!confirm(`Dry-run: flatten ${pct} of ${pos.symbol} (${state.account})?`)) return;
  sendDryRun("flatten", { symbol: pos.symbol, sec_type: pos.sec_type, expiry: pos.expiry, fraction, order_type: "MKT" });
}
function execCancel(orderId, symbol) {
  if (!confirm(`Dry-run: cancel order ${orderId} (${symbol}, ${state.account})?`)) return;
  sendDryRun("cancel", orderId ? { scope: "order", order_id: orderId } : { scope: "symbol", symbol });
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
    f.innerHTML = `<label class="cap">Symbol</label>${inp("f_symbol", "USO", 80)}
      <label class="cap">Side</label><select id="f_action"><option>BUY</option><option>SELL</option></select>
      <label class="cap">Qty</label>${inp("f_qty", "692", 70)}
      <label class="cap">Entry</label>${inp("f_entry", "104.80", 80)}
      <label class="cap">Stop</label>${inp("f_stop", "103.29", 80)}
      <label class="cap">Target</label>${inp("f_target", "123.21", 80)}`;
  }
}
function val(id) { const e = document.getElementById(id); return e ? e.value : undefined; }
function ticketPayload(t) {
  if (t === "echo") return { note: val("f_note") };
  if (t === "flatten") return { symbol: val("f_symbol"), fraction: Number(val("f_frac")), order_type: "MKT" };
  return { symbol: val("f_symbol"), action: val("f_action"), quantity: Number(val("f_qty")),
    entry: Number(val("f_entry")), stop: Number(val("f_stop")), target: Number(val("f_target")) };
}
function sendTicket() { sendDryRun(document.getElementById("cmdType").value, ticketPayload(document.getElementById("cmdType").value), "cmdMsg"); }

async function sendDryRun(type, payload, msgId) {
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

/* ---------- activity ---------- */
function stateBadge(state) {
  const map = { dry_run: ["#3ddb8f", "DRY-RUN OK"], rejected: ["#ff6b6b", "REJECTED"],
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
  return html;
}
function renderActivity() {
  const cmds = state.commands || [];
  if (!cmds.length) return "";
  const rows = cmds.map((c) => {
    const age = c.created_at ? Math.round((Date.now() - c.created_at) / 1000) + "s" : "";
    return `<tr style="vertical-align:top">
      <td class="l" style="font-weight:600">${esc(c.type || "")}</td>
      <td class="l">${esc(c.account || "")}</td>
      <td class="l">${stateBadge(c.state)}</td>
      <td class="l">${resultCell(c)}</td>
      <td class="l" style="color:#8c95a2">${esc(age)}</td></tr>`;
  }).join("");
  return `<div style="font:700 14px inherit;margin-bottom:6px">Activity <span class="cap" style="display:inline;font-weight:400">· dry-run, places nothing</span></div>
    <div class="tblwrap"><table class="tbl"><thead><tr>
    <th class="l">type</th><th class="l">acct</th><th class="l">state</th><th class="l">result / order preview</th><th class="l">ago</th>
    </tr></thead><tbody>${rows}</tbody></table></div>`;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
