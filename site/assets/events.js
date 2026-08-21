/* events.js — the event sleeve: per-trade status cards, open positions,
 * realized history vs the prereg expectancy.
 *
 * Reads data/event_sleeve.json (build_site.build_event_sleeve): the sleeve's
 * live state + status cards, and the append-only journal graded against
 * master_prices. Realized rows are modeled from daily bars (MOC = close,
 * MOO = open), not broker fills — an auction fill differs by noise and the
 * page says so rather than pretending precision. Sizing basis is the fixed
 * $750k ACCOUNT_VALUE constant, NOT live NLV; the subtitle states it.
 */
"use strict";

const KIND_COLOR = {
  staged: "#4da3ff", open: "#00d18f", skipped: "#ffc14d",
  armed: "#8a93a5", error: "#ff5d5d",
};
const KIND_LABEL = {
  staged: "STAGED TODAY", open: "OPEN", skipped: "SKIPPED TODAY",
  armed: "ARMED", error: "ERROR",
};

const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function tradeName(t) {
  return String(t || "").replace(/_/g, " ")
    .replace(/\b\w+/g, (w) => w.charAt(0) + w.slice(1).toLowerCase())
    .replace(/\bFomc\b/g, "FOMC");
}

function kpi(label, value, sub) {
  return `<div class="kpi"><div class="l">${label}</div>
    <div class="v">${value}</div>${sub ? `<div class="s">${sub}</div>` : ""}</div>`;
}

function cardHtml(c, summaryByTrade) {
  const col = KIND_COLOR[c.kind] || "#8a93a5";
  const s = summaryByTrade[c.trade];
  const realized = s
    ? `<div class="cap" style="margin-top:6px">Realized: ${s.n} trade${s.n === 1 ? "" : "s"},
        ${s.wins}/${s.n} up, avg <b class="${clsSign(s.avg_ret_pct)}">${fmt.signed(s.avg_ret_pct)}%</b>/window,
        <b class="${clsSign(s.total_pnl)}">${fmt.money(s.total_pnl)}</b> (${fmt.signed(s.total_nav_bps, 1)} bps NAV)</div>`
    : `<div class="cap" style="margin-top:6px">Realized: nothing graded yet.</div>`;
  return `<div class="card" style="border-left:3px solid ${col}">
    <div style="display:flex;justify-content:space-between;gap:8px;align-items:baseline">
      <b>${esc(tradeName(c.trade))}</b>
      <span style="color:${col};font-size:11px;letter-spacing:.4px">${KIND_LABEL[c.kind] || esc(c.kind)}</span>
    </div>
    <div style="margin-top:4px;font-size:13px;color:${col}">${esc(c.status)}</div>
    <div class="cap" style="margin-top:6px">${esc(c.rule)}</div>
    <div class="cap" style="margin-top:2px"><b>Prereg:</b> ${esc(c.evidence)}</div>
    ${realized}
  </div>`;
}

function openTable(open) {
  if (!open.length) return "";
  const rows = open.map((r) => `<tr>
    <td class="l">${esc(tradeName(r.trade))}</td>
    <td class="l">${esc(r.ticker)}</td>
    <td class="l">${esc(r.side)}</td>
    <td>${fmt.num(r.qty, 0)}</td>
    <td>${fmt.date(r.entry_date)}</td>
    <td>${r.entry_px == null ? "&mdash;" : fmt.num(r.entry_px)}</td>
    <td>${r.mark_px == null ? "&mdash;" : fmt.num(r.mark_px)}</td>
    <td class="${clsSign(r.ret_pct)}">${r.ret_pct == null ? "&mdash;" : fmt.signed(r.ret_pct) + "%"}</td>
    <td class="${clsSign(r.pnl)}">${r.pnl == null ? "&mdash;" : fmt.money(r.pnl)}</td>
    <td>${fmt.date(r.exit_on)} ${esc(r.exit_order_type || "")}</td>
  </tr>`).join("");
  return `<div class="card" style="margin-top:14px"><b>Open positions</b>
    <div class="cap">Marked to the latest cached close; an entry staged today has no bar yet.</div>
    <div class="tblwrap"><table class="tbl">
      <thead><tr><th class="l">Trade</th><th class="l">Ticker</th><th class="l">Side</th>
        <th>Qty</th><th>Entry</th><th>Entry px</th><th>Mark</th><th>Ret</th><th>PnL</th>
        <th>Scheduled exit</th></tr></thead>
      <tbody>${rows}</tbody></table></div></div>`;
}

function historySection(closed) {
  const el = document.createElement("div");
  el.style.marginTop = "14px";
  const head = document.createElement("div");
  head.innerHTML = `<b>Closed trades</b>
    <div class="cap">Every journaled round trip, graded MOC=close / MOO=open from the price cache.
      Rows missing a bar (cache gap) show no return.</div>`;
  el.appendChild(head);
  const tblEl = document.createElement("div");
  el.appendChild(tblEl);
  makeTable(tblEl, {
    columns: [
      { key: "entry_date", label: "Entry", fmt: fmt.date },
      { key: "trade", label: "Trade", align: "l", fmt: (v) => esc(tradeName(v)) },
      { key: "ticker", label: "Ticker", align: "l", fmt: esc },
      { key: "side", label: "Side", align: "l", fmt: esc },
      { key: "qty", label: "Qty", fmt: (v) => fmt.num(v, 0) },
      { key: "entry_px", label: "Entry px", fmt: (v) => v == null ? "" : fmt.num(v) },
      { key: "exit_date", label: "Exit", fmt: fmt.date },
      { key: "exit_px", label: "Exit px", fmt: (v) => v == null ? "" : fmt.num(v) },
      { key: "ret_pct", label: "Ret %", fmt: (v) => v == null ? "" : fmt.signed(v) + "%", cls: clsSign },
      { key: "pnl", label: "PnL", fmt: (v) => v == null ? "" : fmt.money(v), cls: clsSign },
      { key: "nav_bps", label: "NAV bps", fmt: (v) => v == null ? "" : fmt.signed(v, 1), cls: clsSign },
      { key: "late", label: "Late", fmt: (v) => v ? "LATE" : "" },
    ],
    rows: closed,
    pageSize: 25,
    search: true,
    csvName: "event_sleeve_trades.csv",
    defaultSort: { key: "entry_date", dir: -1 },
  });
  return el;
}

async function init() {
  renderNav("events.html");
  const p = await fetchJSONOrNull("data/event_sleeve.json");
  const el = document.getElementById("content");
  if (!p) {
    el.innerHTML = '<p class="cap">No event_sleeve.json in this build (payload is best-effort; check the deploy log).</p>';
    return;
  }
  setAsof(`built ${p.generated}`);

  const summaryByTrade = {};
  for (const s of p.summary || []) summaryByTrade[s.trade] = s;
  const graded = (p.history?.closed || []).filter((r) => r.ret_pct != null);
  const totPnl = graded.reduce((a, r) => a + (r.pnl || 0), 0);
  const wins = graded.filter((r) => r.ret_pct > 0).length;
  const openRows = p.history?.open || [];
  const openPnl = openRows.reduce((a, r) => a + (r.pnl || 0), 0);

  el.innerHTML = `
    <div class="kpis">
      ${kpi("Open positions", openRows.length,
            openRows.length ? `<span class="${clsSign(openPnl)}">${fmt.money(openPnl)} marked</span>` : "flat")}
      ${kpi("Realized trades", graded.length, graded.length ? `${wins}/${graded.length} up` : "none graded yet")}
      ${kpi("Realized PnL", `<span class="${clsSign(totPnl)}">${fmt.money(totPnl)}</span>`,
            graded.length ? fmt.signed(totPnl / p.account_value * 1e4, 1) + " bps of basis" : "")}
      ${kpi("Sizing basis", fmt.money(p.account_value), "fixed ACCOUNT_VALUE, not live NLV")}
      ${kpi("Journal", `${p.journal_n} record${p.journal_n === 1 ? "" : "s"}`, "append-only, R2")}
    </div>
    <div class="grid2" id="cards"></div>`;
  const cardsEl = document.getElementById("cards");
  cardsEl.innerHTML = (p.cards || []).map((c) => cardHtml(c, summaryByTrade)).join("");

  const openHtml = openTable(openRows);
  if (openHtml) el.insertAdjacentHTML("beforeend", openHtml);
  if ((p.history?.closed || []).length) el.appendChild(historySection(p.history.closed));
  else el.insertAdjacentHTML("beforeend",
    '<p class="cap" style="margin-top:14px">No closed trades yet — the first exits will appear here once the journal pairs them.</p>');
}

if (typeof document !== "undefined") document.addEventListener("DOMContentLoaded", init);
