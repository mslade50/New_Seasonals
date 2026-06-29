/* execution.js — execution-bridge status (Phase 2a).

   Polls /exec-status (a Pages Function proxy to the execution-broker Worker) every
   few seconds and shows the local agent's online/offline heartbeat. No commands,
   no positions, no click-to-trade yet — those land in later phases. Read-only. */
"use strict";

document.addEventListener("DOMContentLoaded", initExecution);

let pollTimer = null;

async function initExecution() {
  renderNav("execution.html");
  await poll();
  pollTimer = setInterval(poll, 5000);
}

async function poll() {
  const el = document.getElementById("content");
  if (!el) return;
  const s = await fetchJSONOrNull("/exec-status") || { online: false, configured: false };
  el.innerHTML = render(s);
  setAsof(s.online ? "execution online" : s.configured ? "execution offline" : "broker not configured");
}

function dot(color) {
  return `<span style="display:inline-block;width:11px;height:11px;border-radius:50%;
    background:${color};box-shadow:0 0 9px ${color}"></span>`;
}

function secs(ms) { return ms == null ? "&mdash;" : Math.round(ms / 1000) + "s"; }

function render(s) {
  let tone, label, sub;
  if (!s.configured) {
    tone = "#9aa3b2";
    label = "Broker not configured";
    sub = "Deploy the execution-broker Worker and set EXEC_BROKER_URL + STATUS_TOKEN on the site (see execution-broker/README.md).";
  } else if (s.online) {
    tone = "#3ddb8f";
    label = "Execution online";
    sub = `Local agent connected and beating &middot; last heartbeat ${secs(s.heartbeat_age_ms)} ago.`;
  } else {
    tone = "#ff6b6b";
    label = "Execution offline";
    sub = s.sockets
      ? "Agent socket is open but the heartbeat is stale &mdash; check exec_agent.py on the trading machine."
      : "No agent connected. Start exec_agent.py on the trading machine.";
  }
  if (s.error) sub += ` <span style="color:#ffc14d">(${esc(s.error)})</span>`;
  const since = s.connected_at ? new Date(s.connected_at).toLocaleString() : "&mdash;";

  return `
    <div class="card" style="max-width:580px">
      <div style="display:flex;align-items:center;gap:10px;font:700 18px inherit">${dot(tone)} ${label}</div>
      <p class="cap" style="margin:7px 0 0">${sub}</p>
      <div class="kv" style="margin-top:14px">
        <div class="k">Agent sockets</div><div class="v">${s.sockets ?? 0}</div>
        <div class="k">Connected since</div><div class="v">${since}</div>
        <div class="k">Heartbeat age</div><div class="v">${secs(s.heartbeat_age_ms)}</div>
        <div class="k">Stale after</div><div class="v">${secs(s.stale_after_ms)}</div>
      </div>
    </div>
    <p class="cap" style="margin-top:16px">Phase 2a &mdash; heartbeat only. Command dispatch (dry-run first),
      positions, and click-to-trade arrive in later phases. Nothing on this page places or modifies an order.</p>`;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
