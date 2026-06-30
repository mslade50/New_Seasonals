/* common.js — nav, fetch, formatting, Plotly defaults, table component */
"use strict";

const PAGES = [
  { href: "index.html",    label: "Portfolio" },
  { href: "ideas.html",    label: "Ideas" },
  { href: "signals.html",  label: "Signals" },
  { href: "orders.html",   label: "Orders" },
  { href: "execution.html", label: "Execution" },
  { href: "options.html",  label: "Options" },
  { href: "seasonal.html", label: "Seasonal" },
  { href: "charts.html",   label: "Charts" },
  { href: "risk.html",     label: "Risk" },
];

function renderNav(active) {
  const el = document.getElementById("topbar");
  if (!el) return;
  const links = PAGES.map(p =>
    `<a href="${p.href}" class="${p.href === active ? "active" : ""}">${p.label}</a>`).join("");
  el.innerHTML = `<div class="brand">Seasonals <span>/</span> Private</div>
    <nav>${links}</nav><div class="asof" id="navAsof"></div>`;
}

async function fetchJSON(path) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.json();
}

async function fetchJSONOrNull(path) {
  try { return await fetchJSON(path); } catch (e) { return null; }
}

function setAsof(text) {
  const el = document.getElementById("navAsof");
  if (el) el.textContent = text || "";
}

/* ---------- formatting ---------- */
const fmt = {
  money: (v, d = 0) => v == null ? "" : (v < 0 ? "-$" : "$") +
    Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }),
  num: (v, d = 2) => v == null ? "" :
    Number(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }),
  pct: (v, d = 1) => v == null ? "" : (v * 100).toFixed(d) + "%",
  pctRaw: (v, d = 1) => v == null ? "" : Number(v).toFixed(d) + "%",
  signed: (v, d = 2) => v == null ? "" : (v >= 0 ? "+" : "") + Number(v).toFixed(d),
  date: v => v == null ? "" : String(v).slice(0, 10),
};

function clsSign(v) { return v == null ? "neu" : v > 0 ? "pos" : v < 0 ? "neg" : "neu"; }

/* ---------- Plotly defaults ---------- */
const PLOT_CFG = { displayModeBar: false, responsive: true };

function plotLayout(overrides) {
  const base = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "#0f131c",
    font: { color: "#c7ccd6", size: 11.5,
            family: '-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif' },
    margin: { l: 52, r: 14, t: 26, b: 36 },
    xaxis: { gridcolor: "#1c2230", zerolinecolor: "#2a3242", linecolor: "#232a38" },
    yaxis: { gridcolor: "#1c2230", zerolinecolor: "#2a3242", linecolor: "#232a38" },
    hovermode: "x unified",
    hoverlabel: { bgcolor: "#1a2030", bordercolor: "#2a3242", font: { color: "#e8eaf0" } },
    legend: { orientation: "h", y: 1.08, x: 1, xanchor: "right",
              bgcolor: "rgba(0,0,0,0)" },
  };
  return deepMerge(base, overrides || {});
}

function deepMerge(a, b) {
  const out = Array.isArray(a) ? a.slice() : Object.assign({}, a);
  for (const k of Object.keys(b)) {
    if (b[k] && typeof b[k] === "object" && !Array.isArray(b[k]) &&
        a[k] && typeof a[k] === "object" && !Array.isArray(a[k])) {
      out[k] = deepMerge(a[k], b[k]);
    } else out[k] = b[k];
  }
  return out;
}

const PALETTE = ["#4da3ff", "#00d18f", "#ff5d5d", "#ffc14d", "#b07cff", "#3ddbd9",
                 "#ff8fab", "#9ee06d", "#ffa94d", "#6db8ff", "#e06da8", "#8fd3ff",
                 "#d4c45a", "#7ce0b8", "#ff7d5d", "#a8a4ff"];

/* ---------- table component ----------
   makeTable(el, {
     columns: [{key, label, align:'l'|'r', fmt: fn(v,row), cls: fn(v,row)}],
     rows: [...], pageSize: 25|0 (0 = all), search: true|false,
     csvName: "trades.csv"|null, defaultSort: {key, dir} })
*/
function makeTable(el, opts) {
  const state = {
    rows: opts.rows.slice(),
    view: opts.rows.slice(),
    page: 0,
    pageSize: opts.pageSize || 0,
    sortKey: opts.defaultSort ? opts.defaultSort.key : null,
    sortDir: opts.defaultSort ? opts.defaultSort.dir : -1,
    q: "",
  };

  el.innerHTML = "";
  let controls = null, info = null, searchBox = null;
  if (opts.search || opts.csvName || state.pageSize) {
    controls = document.createElement("div");
    controls.className = "tbl-controls";
    if (opts.search) {
      searchBox = document.createElement("input");
      searchBox.placeholder = "Filter rows (any column)...";
      searchBox.addEventListener("input", () => { state.q = searchBox.value.toLowerCase(); state.page = 0; refresh(); });
      controls.appendChild(searchBox);
    }
    if (opts.csvName) {
      const b = document.createElement("button");
      b.className = "btn ghost"; b.textContent = "Export CSV";
      b.addEventListener("click", () => downloadCSV(state.view, opts.columns, opts.csvName));
      controls.appendChild(b);
    }
    info = document.createElement("div");
    info.className = "info";
    controls.appendChild(info);
    el.appendChild(controls);
  }

  const wrap = document.createElement("div");
  wrap.className = "tblwrap";
  const table = document.createElement("table");
  table.className = "tbl";
  wrap.appendChild(table);
  el.appendChild(wrap);

  let pager = null;
  if (state.pageSize) {
    pager = document.createElement("div");
    pager.className = "tbl-controls";
    pager.innerHTML = `<div class="pager">
      <button data-a="prev">Prev</button><span class="pg"></span><button data-a="next">Next</button>
      <select class="btn" data-a="size">
        <option value="25">25</option><option value="50">50</option>
        <option value="100">100</option><option value="0">All</option></select>
    </div>`;
    pager.querySelector('[data-a="prev"]').addEventListener("click", () => { state.page--; refresh(); });
    pager.querySelector('[data-a="next"]').addEventListener("click", () => { state.page++; refresh(); });
    const sizeSel = pager.querySelector('[data-a="size"]');
    sizeSel.value = String(state.pageSize);
    sizeSel.addEventListener("change", () => { state.pageSize = +sizeSel.value; state.page = 0; refresh(); });
    el.appendChild(pager);
  }

  function applySort() {
    if (!state.sortKey) return;
    const k = state.sortKey, dir = state.sortDir;
    state.view.sort((a, b) => {
      const va = a[k], vb = b[k];
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      if (typeof va === "number" && typeof vb === "number") return (va - vb) * dir;
      return String(va).localeCompare(String(vb)) * dir;
    });
  }

  function refresh() {
    state.view = !state.q ? state.rows.slice() :
      state.rows.filter(r => opts.columns.some(c => {
        const v = r[c.key];
        return v != null && String(v).toLowerCase().includes(state.q);
      }));
    applySort();

    const n = state.view.length;
    const ps = state.pageSize || n || 1;
    const npages = Math.max(1, Math.ceil(n / ps));
    state.page = Math.min(Math.max(0, state.page), npages - 1);
    const rows = state.pageSize ? state.view.slice(state.page * ps, (state.page + 1) * ps) : state.view;

    const head = "<thead><tr>" + opts.columns.map(c => {
      const arr = state.sortKey === c.key ? `<span class="arr">${state.sortDir > 0 ? "▲" : "▼"}</span>` : "";
      return `<th class="${c.align === "l" ? "l" : ""}" data-k="${c.key}">${c.label}${arr}</th>`;
    }).join("") + "</tr></thead>";

    const body = "<tbody>" + rows.map(r => "<tr>" + opts.columns.map(c => {
      const v = r[c.key];
      const txt = c.fmt ? c.fmt(v, r) : (v == null ? "" : v);
      const cls = (c.align === "l" ? "l " : "") + (c.cls ? c.cls(v, r) : "");
      return `<td class="${cls}">${txt}</td>`;
    }).join("") + "</tr>").join("") + "</tbody>";

    table.innerHTML = head + body;
    table.querySelectorAll("th").forEach(th => th.addEventListener("click", () => {
      const k = th.dataset.k;
      if (state.sortKey === k) state.sortDir *= -1;
      else { state.sortKey = k; state.sortDir = -1; }
      refresh();
    }));

    if (info) info.textContent = `${n.toLocaleString()} rows`;
    if (pager) {
      pager.querySelector(".pg").textContent = `page ${state.page + 1} / ${npages}`;
      pager.querySelector('[data-a="prev"]').disabled = state.page === 0;
      pager.querySelector('[data-a="next"]').disabled = state.page >= npages - 1;
    }
  }

  refresh();
  return {
    setRows(rows) { state.rows = rows.slice(); state.page = 0; refresh(); },
  };
}

function downloadCSV(rows, columns, name) {
  const esc = v => {
    if (v == null) return "";
    const s = String(v);
    return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  };
  const head = columns.map(c => esc(c.label)).join(",");
  const body = rows.map(r => columns.map(c => esc(r[c.key])).join(",")).join("\n");
  const blob = new Blob([head + "\n" + body], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

/* columnar payload -> array of row objects */
function rowsFromColumnar(payload) {
  const cols = payload.columns;
  const keys = Object.keys(cols);
  const n = payload.n != null ? payload.n : (cols[keys[0]] || []).length;
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const r = {};
    for (const k of keys) r[k] = cols[k][i];
    out[i] = r;
  }
  return out;
}
