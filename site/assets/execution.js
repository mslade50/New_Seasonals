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

async function initExecution() {
  renderNav("execution.html");
  const el = document.getElementById("content");
  el.innerHTML = shell();
  document.querySelectorAll("[data-acct]").forEach((b) =>
    b.addEventListener("click", () => setAccount(b.dataset.acct)));
  document.getElementById("cmdType").addEventListener("change", syncFields);
  document.getElementById("cmdSend").addEventListener("click", sendTicket);
  document.getElementById("cmdFields").addEventListener("input", updateReadout);
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
      <p class="cap" style="margin:0 0 10px">Bracket: entry limit + stop + target, plus an optional <b>time stop</b> (closes at market 15:59 ET on that date if neither exit hit). Submits per the mode banner above &mdash; live when armed.</p>
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
      <td class="l"><button class="btn xs ghost" onclick='execCancel(${o.perm_id || 0},${o.order_id || 0},"${esc(o.symbol)}")'>Cancel</button></td>
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
    f.innerHTML = `<label class="cap">Symbol</label>${inp("f_symbol", "USO", 80)}
      <label class="cap">Side</label><select id="f_action"><option>BUY</option><option>SELL</option></select>
      <label class="cap">Qty</label>${inp("f_qty", "692", 70)}
      <label class="cap">Entry</label>${inp("f_entry", "104.80", 80)}
      <label class="cap">Stop</label>${inp("f_stop", "103.29", 80)}
      <label class="cap">Target</label>${inp("f_target", "123.21", 80)}
      <label class="cap">Time stop</label><input type="date" id="f_timestop" style="width:140px">`;
  }
  updateReadout();
}

function updateReadout() {
  const t = document.getElementById("cmdType").value;
  const el = document.getElementById("ticketReadout");
  if (!el) return;
  if (t === "entry_bracket") {
    const qty = Number(val("f_qty")), entry = Number(val("f_entry")), stop = Number(val("f_stop")), target = Number(val("f_target"));
    const action = val("f_action"), dist = Math.abs(entry - stop);
    let warn = "";
    if (action === "BUY" && !(stop < entry && entry < target)) warn = "BUY needs stop &lt; entry &lt; target";
    if (action === "SELL" && !(target < entry && entry < stop)) warn = "SELL needs target &lt; entry &lt; stop";
    if (warn) { el.innerHTML = `<span style="color:#ff6b6b">${warn}</span>`; return; }
    const parts = [];
    if (qty && dist) parts.push(`Risk <b>${fmt.money(qty * dist)}</b>`);
    if (dist && target) parts.push(`R:R <b>${(Math.abs(target - entry) / dist).toFixed(2)}:1</b>`);
    if (qty && entry) parts.push(`Notional <b>${fmt.money(qty * entry)}</b>`);
    const ts = val("f_timestop");
    if (ts) parts.push(`Time-exit <b>${ts}</b>`);
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
  return { symbol: val("f_symbol"), action: val("f_action"), quantity: Number(val("f_qty")),
    entry: Number(val("f_entry")), stop: Number(val("f_stop")), target: Number(val("f_target")),
    time_stop: val("f_timestop") || null };
}
function sendTicket() {
  const t = document.getElementById("cmdType").value;
  const p = ticketPayload(t);
  if (t !== "echo") {
    const summary = t === "entry_bracket"
      ? `${p.action} ${p.quantity} ${p.symbol} @ ${p.entry} (stop ${p.stop}, target ${p.target}${p.time_stop ? ", time " + p.time_stop : ""})`
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
