/* tradelog.js — executed-trades log (both accounts).

   Reads /exec-fills (Pages proxy -> broker DO /fills): per-execution rows the
   agent's book pushes accumulated across days, deduped by IBKR exec_id. The
   page aggregates executions into one row per order (account + perm_id + side)
   with a share-weighted average price, or shows raw executions via the toggle.
   Filters: trailing window (today / 7d / 14d) + account (all / primary / pa).
   Times display in ET; polling refreshes in place every 30s. */
"use strict";

document.addEventListener("DOMContentLoaded", initTradeLog);

const TL_REFRESH_MS = 30_000;
const tlState = { days: 7, account: "all", fills: [], raw: false, error: null };
let tlTable = null;

function stratFromRef(ref) {
  // orderRef 'SYMBOL|ACTION|Strategy_Ref|Staged_Date[|tranche]' -> strategy
  if (!ref) return "";
  const parts = String(ref).split("|");
  return parts.length >= 4 ? (parts[2] || "").trim() : String(ref);
}

function etParts(iso) {
  if (!iso) return { date: "", time: "" };
  const d = new Date(iso);
  if (isNaN(d)) return { date: String(iso).slice(0, 10), time: "" };
  return {
    date: d.toLocaleDateString("en-CA", { timeZone: "America/New_York" }),
    time: d.toLocaleTimeString("en-GB", { timeZone: "America/New_York", hour12: false }),
  };
}

/* one row per order: account + perm_id + side (perm_id 0 falls back to the
   exec_id so unmatched rows never merge). Share-weighted avg price. */
function aggregateOrders(fills) {
  const groups = new Map();
  for (const f of fills || []) {
    const pid = f.perm_id ? String(f.perm_id) : "x" + f.exec_id;
    const key = `${f.account_key || f.account || ""}|${pid}|${f.side || ""}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(f);
  }
  const rows = [];
  for (const g of groups.values()) {
    g.sort((a, b) => String(a.time || "").localeCompare(String(b.time || "")));
    const first = g[0];
    const qty = g.reduce((s, f) => s + (f.qty || 0), 0);
    const notional = g.reduce((s, f) => s + (f.qty || 0) * (f.price || 0), 0);
    const comm = g.reduce((s, f) => s + (f.commission || 0), 0);
    const pnl = g.filter((f) => f.realized_pnl != null);
    rows.push({
      time: first.time,
      account_key: first.account_key || "",
      account: first.account_label || first.account_key || first.account || "",
      symbol: first.local_symbol || first.symbol || "",
      sec_type: first.sec_type || "",
      side: first.side === "BOT" ? "BUY" : first.side === "SLD" ? "SELL" : (first.side || ""),
      qty,
      avg_price: qty ? notional / qty : null,
      notional,
      strategy: stratFromRef(first.order_ref),
      order_ref: first.order_ref || null,
      n_fills: g.length,
      commission: comm || null,
      realized_pnl: pnl.length ? pnl.reduce((s, f) => s + f.realized_pnl, 0) : null,
    });
  }
  rows.sort((a, b) => String(b.time || "").localeCompare(String(a.time || "")));
  return rows;
}

function rawRows(fills) {
  return (fills || []).map((f) => ({
    time: f.time,
    account_key: f.account_key || "",
    account: f.account_label || f.account_key || f.account || "",
    symbol: f.local_symbol || f.symbol || "",
    sec_type: f.sec_type || "",
    side: f.side === "BOT" ? "BUY" : f.side === "SLD" ? "SELL" : (f.side || ""),
    qty: f.qty,
    avg_price: f.price,
    notional: (f.qty || 0) * (f.price || 0),
    strategy: stratFromRef(f.order_ref),
    order_ref: f.order_ref || null,
    n_fills: 1,
    commission: f.commission,
    realized_pnl: f.realized_pnl,
  })).sort((a, b) => String(b.time || "").localeCompare(String(a.time || "")));
}

function tlCutoffDate(days) {
  // trailing window in ET calendar days; days=1 -> today only
  const today = new Date().toLocaleDateString("en-CA", { timeZone: "America/New_York" });
  const d = new Date(today + "T00:00:00Z");
  d.setUTCDate(d.getUTCDate() - (days - 1));
  return d.toISOString().slice(0, 10);
}

function filteredRows() {
  const base = tlState.raw ? rawRows(tlState.fills) : aggregateOrders(tlState.fills);
  const cutoff = tlCutoffDate(tlState.days);
  return base
    .map((r) => { const p = etParts(r.time); return { ...r, date: p.date, timeET: p.time }; })
    .filter((r) => r.date >= cutoff)
    .filter((r) => tlState.account === "all" || r.account_key === tlState.account);
}

function kpiHtml(rows) {
  const buys = rows.filter((r) => r.side === "BUY");
  const sells = rows.filter((r) => r.side === "SELL");
  const sum = (a, k) => a.reduce((s, r) => s + (r[k] || 0), 0);
  const pnlRows = rows.filter((r) => r.realized_pnl != null);
  const pnl = sum(pnlRows, "realized_pnl");
  const kpi = (l, v, s) => `<div class="kpi"><div class="l">${l}</div><div class="v">${v}</div>` +
    (s ? `<div class="s">${s}</div>` : "") + `</div>`;
  return `<div class="kpis">` +
    kpi("Orders", String(rows.length), `${sum(rows, "n_fills")} executions`) +
    kpi("Bought", fmt.money(sum(buys, "notional")), `${fmt.num(sum(buys, "qty"), 0)} sh / ${buys.length} orders`) +
    kpi("Sold", fmt.money(sum(sells, "notional")), `${fmt.num(sum(sells, "qty"), 0)} sh / ${sells.length} orders`) +
    kpi("Commissions", fmt.money(sum(rows, "commission"), 2)) +
    (pnlRows.length
      ? kpi("Realized PnL", `<span class="${clsSign(pnl)}">${fmt.money(pnl, 0)}</span>`, "closing executions only")
      : "") +
    `</div>`;
}

const TL_COLUMNS = [
  { key: "date", label: "Date", align: "l" },
  { key: "timeET", label: "Time (ET)", align: "l" },
  { key: "account", label: "Account", align: "l" },
  { key: "symbol", label: "Symbol", align: "l" },
  { key: "side", label: "Side", align: "l", cls: (v) => (v === "BUY" ? "pos" : v === "SELL" ? "neg" : "") },
  { key: "qty", label: "Qty", fmt: (v) => fmt.num(v, 0) },
  { key: "avg_price", label: "Avg Px", fmt: (v) => fmt.num(v, 2) },
  { key: "notional", label: "Notional", fmt: (v) => fmt.money(v) },
  { key: "strategy", label: "Strategy", align: "l" },
  { key: "n_fills", label: "Fills" },
  { key: "commission", label: "Comm", fmt: (v) => (v == null ? "" : fmt.money(v, 2)) },
  { key: "realized_pnl", label: "Realized", fmt: (v) => (v == null ? "" : fmt.money(v, 0)), cls: clsSign },
];

function renderShell() {
  const el = document.getElementById("content");
  el.innerHTML = `
    <div class="tbl-controls" style="gap:10px; flex-wrap:wrap; align-items:center;">
      <div class="seg" id="tlWin">
        <button data-d="1">Today</button>
        <button data-d="7" class="on">7 days</button>
        <button data-d="14">14 days</button>
      </div>
      <div class="seg" id="tlAcct">
        <button data-a="all" class="on">All</button>
        <button data-a="primary">Primary</button>
        <button data-a="pa">PA</button>
      </div>
      <div class="seg" id="tlMode">
        <button data-r="0" class="on">By order</button>
        <button data-r="1">Raw fills</button>
      </div>
      <div class="info" id="tlNote"></div>
    </div>
    <div id="tlKpis"></div>
    <div class="card" id="tlTable"></div>`;
  const wire = (boxId, attr, apply) => {
    document.querySelectorAll(`#${boxId} button`).forEach((b) => b.addEventListener("click", () => {
      document.querySelectorAll(`#${boxId} button`).forEach((x) => x.classList.remove("on"));
      b.classList.add("on");
      apply(b.dataset[attr]);
      renderData();
    }));
  };
  wire("tlWin", "d", (v) => { tlState.days = +v; });
  wire("tlAcct", "a", (v) => { tlState.account = v; });
  wire("tlMode", "r", (v) => { tlState.raw = v === "1"; });
}

function renderData() {
  const rows = filteredRows();
  document.getElementById("tlKpis").innerHTML = rows.length ? kpiHtml(rows) : "";
  const note = document.getElementById("tlNote");
  if (tlState.error) note.textContent = tlState.error;
  else if (!tlState.fills.length) note.textContent =
    "No executions recorded yet — the ring accumulates from the agent's next trading session.";
  else note.textContent = "";
  if (!tlTable) {
    tlTable = makeTable(document.getElementById("tlTable"), {
      columns: TL_COLUMNS, rows, pageSize: 50, search: true,
      csvName: "trade_log.csv", defaultSort: { key: "time", dir: -1 },
    });
  } else tlTable.setRows(rows);
}

async function tlLoad() {
  const data = await fetchJSONOrNull("/exec-fills");
  if (data && Array.isArray(data.fills)) {
    tlState.fills = data.fills;
    tlState.error = null;
  } else if (data && data.configured === false) {
    tlState.error = "Execution broker not configured for this deploy.";
  } else if (!tlState.fills.length) {
    tlState.error = "Could not reach the execution broker.";
  }
  setAsof(`fills as of ${new Date().toLocaleTimeString("en-GB", { timeZone: "America/New_York", hour12: false })} ET`);
  renderData();
}

async function initTradeLog() {
  renderNav("tradelog.html");
  renderShell();
  await tlLoad();
  setInterval(tlLoad, TL_REFRESH_MS);
}
