/* execution.js — execution-bridge status + dry-run command loop (Phase 2b).

   Top: the agent online/offline heartbeat (polls /exec-status). Middle: a small
   form to send a command — which is signed by the Pages Function and dry-run
   validated by the local agent; it PLACES NOTHING. Bottom: recent commands + the
   agent's "would do" results (polls /exec-commands). The form is rendered once so
   the 5s polls don't reset it. Read-only with respect to real orders. */
"use strict";

document.addEventListener("DOMContentLoaded", initExecution);

let pollTimer = null;

async function initExecution() {
  renderNav("execution.html");
  const el = document.getElementById("content");
  el.innerHTML = shell();
  document.getElementById("cmdType").addEventListener("change", syncFields);
  document.getElementById("cmdSend").addEventListener("click", sendCommand);
  syncFields();
  await poll();
  pollTimer = setInterval(poll, 5000);
}

function shell() {
  return `
    <div id="execStatus"><div class="spin">Loading...</div></div>

    <div class="card" style="max-width:620px;margin-top:18px">
      <div style="font:700 14px inherit;margin-bottom:4px">Send command &mdash; DRY-RUN only</div>
      <p class="cap" style="margin:0 0 10px">The Pages Function signs it and the local agent validates it and logs
        what it <b>would</b> do. Nothing is built or sent to IBKR (Phase 2b).</p>
      <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:8px">
        <label class="cap">Type</label>
        <select id="cmdType">
          <option value="echo">echo</option>
          <option value="flatten">flatten</option>
          <option value="entry_bracket">entry_bracket</option>
        </select>
        <label class="cap">Account</label>
        <select id="cmdAccount"><option value="pa">pa</option><option value="primary">primary</option></select>
      </div>
      <div id="cmdFields" style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:10px"></div>
      <button class="btn" id="cmdSend">Send dry-run</button>
      <span id="cmdMsg" class="cap" style="margin-left:10px"></span>
    </div>

    <div id="execCommands" style="margin-top:18px"></div>`;
}

function inp(id, val, w) { return `<input id="${id}" value="${val}" style="width:${w || 90}px">`; }

function syncFields() {
  const t = document.getElementById("cmdType").value;
  const f = document.getElementById("cmdFields");
  if (t === "echo") {
    f.innerHTML = `<label class="cap">Note</label>${inp("f_note", "hello from the site", 220)}`;
  } else if (t === "flatten") {
    f.innerHTML = `<label class="cap">Symbol</label>${inp("f_symbol", "USO", 90)}
      <label class="cap">Order</label><select id="f_otype"><option>MKT</option><option>LMT</option></select>`;
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

function payloadFromForm(t) {
  if (t === "echo") return { note: val("f_note") };
  if (t === "flatten") return { symbol: val("f_symbol"), order_type: val("f_otype"), fraction: 1.0 };
  return {
    symbol: val("f_symbol"), action: val("f_action"), quantity: Number(val("f_qty")),
    entry: Number(val("f_entry")), stop: Number(val("f_stop")), target: Number(val("f_target")),
  };
}

async function sendCommand() {
  const t = document.getElementById("cmdType").value;
  const account = document.getElementById("cmdAccount").value;
  const msg = document.getElementById("cmdMsg");
  msg.textContent = "sending...";
  try {
    const r = await fetch("/exec-command", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type: t, account, payload: payloadFromForm(t) }),
    });
    const d = await r.json();
    msg.textContent = d.ok ? `queued ${(d.id || "").slice(0, 8)}` : `error: ${d.error || ("HTTP " + r.status)}`;
  } catch (e) {
    msg.textContent = "error: " + e;
  }
  setTimeout(poll, 700);
}

async function poll() {
  const s = await fetchJSONOrNull("/exec-status") || { online: false, configured: false };
  const st = document.getElementById("execStatus");
  if (st) st.innerHTML = renderStatus(s);
  setAsof(s.online ? "execution online" : s.configured ? "execution offline" : "broker not configured");

  const c = await fetchJSONOrNull("/exec-commands") || { commands: [] };
  const cc = document.getElementById("execCommands");
  if (cc) cc.innerHTML = renderCommands(c.commands || []);
}

function dot(color) {
  return `<span style="display:inline-block;width:11px;height:11px;border-radius:50%;
    background:${color};box-shadow:0 0 9px ${color}"></span>`;
}
function secs(ms) { return ms == null ? "&mdash;" : Math.round(ms / 1000) + "s"; }

function renderStatus(s) {
  let tone, label, sub;
  if (!s.configured) {
    tone = "#9aa3b2"; label = "Broker not configured";
    sub = "Deploy the execution-broker Worker and set EXEC_BROKER_URL + STATUS_TOKEN on the site.";
  } else if (s.online) {
    tone = "#3ddb8f"; label = "Execution online";
    sub = `Local agent connected and beating &middot; last heartbeat ${secs(s.heartbeat_age_ms)} ago.`;
  } else {
    tone = "#ff6b6b"; label = "Execution offline";
    sub = s.sockets ? "Agent socket open but heartbeat stale." : "No agent connected (idle outside 5 AM&ndash;9 PM ET).";
  }
  if (s.error) sub += ` <span style="color:#ffc14d">(${esc(s.error)})</span>`;
  const since = s.connected_at ? new Date(s.connected_at).toLocaleString() : "&mdash;";
  return `
    <div class="card" style="max-width:620px">
      <div style="display:flex;align-items:center;gap:10px;font:700 18px inherit">${dot(tone)} ${label}</div>
      <p class="cap" style="margin:7px 0 0">${sub}</p>
      <div class="kv" style="margin-top:14px">
        <div class="k">Agent sockets</div><div class="v">${s.sockets ?? 0}</div>
        <div class="k">Connected since</div><div class="v">${since}</div>
        <div class="k">Heartbeat age</div><div class="v">${secs(s.heartbeat_age_ms)}</div>
      </div>
    </div>`;
}

function renderCommands(cmds) {
  if (!cmds.length) return "";
  const rows = cmds.map((c) => {
    const res = c.result || {};
    const tone = res.ok === true ? "#3ddb8f" : res.ok === false ? "#ff6b6b" : "#9aa3b2";
    const age = c.created_at ? Math.round((Date.now() - c.created_at) / 1000) + "s ago" : "";
    return `<tr>
      <td class="l" style="color:#8c95a2">${esc((c.id || "").slice(0, 8))}</td>
      <td class="l" style="font-weight:600">${esc(c.type || "")}</td>
      <td class="l">${esc(c.account || "")}</td>
      <td class="l">${esc(c.state || "")}</td>
      <td class="l" style="color:${tone}">${esc(res.detail || "")}</td>
      <td class="l" style="color:#8c95a2">${esc(age)}</td>
    </tr>`;
  }).join("");
  return `
    <div style="font:700 14px inherit;margin-bottom:6px">Recent commands</div>
    <div class="tblwrap"><table class="tbl">
      <thead><tr><th class="l">id</th><th class="l">type</th><th class="l">acct</th>
        <th class="l">state</th><th class="l">result</th><th class="l">when</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>`;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
