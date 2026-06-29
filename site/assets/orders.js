/* orders.js — morning order-chain confirmation (read-only).

   Renders the JSON that the LOCAL morning_order_summary.py uploads to R2 after
   the 9:31 AM order chain (per-account order states + exit-leg confirmation +
   risk bps + per-strategy rollup + div-adjust banner). Fetched LIVE from the
   /morning-orders Pages Function (the summary is generated after the AM deploy,
   so it can't be baked into the build); falls back to a local data file in dev.

   This page is display-only — it never sends a command or touches an order. It
   stands up the read-only half of the execution data path (see
   docs/site_execution_plan.md, Phase 1). */
"use strict";

document.addEventListener("DOMContentLoaded", initOrders);

const TONE = {  // status/banner colours on the dark theme
  ok:   { fg: "#3ddb8f", bg: "rgba(61,219,143,.10)", bd: "#2c8f63" },
  bad:  { fg: "#ff6b6b", bg: "rgba(255,107,107,.10)", bd: "#a33" },
  info: { fg: "#4da3ff", bg: "rgba(77,163,255,.10)", bd: "#3a6ea5" },
  warn: { fg: "#ffc14d", bg: "rgba(255,193,77,.10)", bd: "#a8852f" },
  grey: { fg: "#9aa3b2", bg: "rgba(154,163,178,.08)", bd: "#3a4252" },
};

function esc(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function banner(tone, html) {
  const t = TONE[tone] || TONE.grey;
  return `<div style="font:600 13px/1.4 inherit;background:${t.bg};border:1px solid ${t.bd};
    color:${t.fg};border-radius:8px;padding:10px 14px;margin:8px 0 14px">${html}</div>`;
}

async function initOrders() {
  renderNav("orders.html");
  const el = document.getElementById("content");
  // live first (Pages Function streaming R2), then a baked dev fallback
  const data = await fetchJSONOrNull("/morning-orders")
            || await fetchJSONOrNull("data/morning_orders.json");
  if (!data) {
    el.innerHTML = banner("grey", "No morning order summary available yet. " +
      "It publishes after the 9:31 AM ET order chain runs.");
    return;
  }
  setAsof(`morning run ${esc(data.generated_at || "?")}`);

  const topTone = data.status === "all_confirmed" ? "ok"
                : data.status === "no_orders" ? "grey" : "bad";
  let html = banner(topTone, esc(data.headline || ""));
  if (data.div_adjust && data.div_adjust.title) {
    html += banner(data.div_adjust.tone || "grey",
      `${esc(data.div_adjust.title)}<div style="font:400 12px inherit;color:#c7ccd6;margin-top:3px">${esc(data.div_adjust.body || "")}</div>`);
  }
  for (const acc of (data.accounts || [])) html += accountSection(acc);
  el.innerHTML = html;
}

function verdict(o) {
  const st = o.entry_state;
  const miss = (o.missing || []).join(", ");
  if (st === "missing")   return ["bad",  "&#9940; ENTRY NOT WORKING"];
  if (st === "cancelled") return ["bad",  `&#9940; ENTRY ${esc(String(o.entry_status || "").toUpperCase())}`];
  if (st === "filled")    return (o.missing || []).length
                                 ? ["bad",  `&#9888; FILLED &mdash; missing: ${esc(miss)}`]
                                 : ["info", `&#9989; FILLED &middot; ${o.n_exp} exits live`];
  if ((o.missing || []).length) return ["bad", `&#9888; missing: ${esc(miss)}`];
  return ["ok", `&#9989; all ${o.n_exp} legs`];
}

function accountSection(acc) {
  if (acc.error && !(acc.orders || []).length) {
    return `<h2>${esc(acc.label)}</h2>` + banner("warn", `&#9888; ${esc(acc.error)}`);
  }
  const orders = acc.orders || [];
  const tone = acc.problems === 0 ? "ok" : "bad";
  const filledNote = acc.filled ? ` (${acc.filled} already filled)` : "";
  const bnr = acc.problems === 0
    ? `&#9989; All ${acc.n} order(s) placed with every expected exit leg${filledNote}.`
    : `&#9888; ${acc.problems} of ${acc.n} order(s) need attention.`;

  const rows = orders.map(o => {
    const [vt, vtxt] = verdict(o);
    const legs = (o.found || []).length ? (o.found).join("+") : "&mdash;";
    return `<tr>
      <td class="l" style="font-weight:600">${esc(o.symbol)}</td>
      <td class="l">${esc(o.action)}</td>
      <td>${fmt.num(o.qty, 0)}</td>
      <td>${fmt.num(o.lmt, 2)}</td>
      <td class="l">${esc(o.strat || "&mdash;")}</td>
      <td>${fmt.money(o.risk)}</td>
      <td style="font-weight:600">${fmt.num(o.bps, 1)}</td>
      <td class="l" style="color:#9aa3b2">${esc(legs)}</td>
      <td class="l" style="color:${TONE[vt].fg};font-weight:600;white-space:nowrap">${vtxt}</td>
    </tr>`;
  }).join("");

  const stratRows = (acc.by_strategy || []).map(s =>
    `<tr><td class="l" style="font-weight:600">${esc(s.strat)}</td>
      <td>${fmt.money(s.risk)}</td><td style="font-weight:600">${fmt.num(s.bps, 1)}</td></tr>`).join("");

  return `
    <h2>${esc(acc.label)} <span class="cap" style="display:inline;font-weight:400">&middot; ${esc(acc.base_note || "")}</span></h2>
    ${banner(tone, bnr)}
    ${acc.error ? `<p class="cap" style="color:${TONE.warn.fg}">Note: ${esc(acc.error)}</p>` : ""}
    <div class="tblwrap"><table class="tbl">
      <thead><tr>
        <th class="l">Symbol</th><th class="l">Side</th><th>Qty</th><th>Entry</th>
        <th class="l">Strategy</th><th>Risk $</th><th>Risk bps</th>
        <th class="l">Exit legs</th><th class="l">Confirmation</th>
      </tr></thead>
      <tbody>${rows}</tbody>
      <tfoot><tr style="font-weight:700;border-top:2px solid #2a3242">
        <td class="l" colspan="5">Total &middot; ${acc.working + acc.filled}/${acc.n} placed (${acc.working} working &middot; ${acc.filled} filled)</td>
        <td>${fmt.money(acc.total_risk)}</td><td>${fmt.num(acc.total_bps, 1)}</td><td colspan="2"></td>
      </tr></tfoot>
    </table></div>
    <h3 class="cap" style="margin:14px 0 4px">Risk by strategy &middot; ${esc(acc.base_note || "")}</h3>
    <div class="tblwrap" style="max-width:420px"><table class="tbl">
      <thead><tr><th class="l">Strategy</th><th>Risk $</th><th>Risk bps</th></tr></thead>
      <tbody>${stratRows}</tbody>
    </table></div>`;
}
