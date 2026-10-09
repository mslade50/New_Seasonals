/* risk-agent.js - the Risk Agent's latest decision, display only.
 *
 * Reads /risk-agent-today (Pages Function -> R2 key risk_agent/today.json,
 * written by daily_risk_agent.py after the evening email). The sleeve is a
 * $200k PAPER book: this page has no staging, no execution link and no button
 * that does anything. Every value is the publisher's own, copied verbatim; the
 * only thing computed here is the NAV sparkline geometry. All text is escaped.
 */
"use strict";

const RA_ENDPOINT = "/risk-agent-today";

const raEsc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const raNum = (v) => {
  if (typeof v === "number") return isFinite(v) ? v : null;
  if (typeof v === "string" && v.trim() !== "" && isFinite(Number(v))) return Number(v);
  return null;
};
const raMoney = (v, signed) => {
  const n = raNum(v);
  if (n == null) return "-";
  const s = Math.abs(n).toLocaleString("en-US", { maximumFractionDigits: 0 });
  return (n < 0 ? "-" : signed && n > 0 ? "+" : "") + s;
};
const raFrac = (v, d) => {
  const n = raNum(v);
  return n == null ? "-" : `${n >= 0 ? "+" : ""}${(n * 100).toFixed(d == null ? 1 : d)}%`;
};
const raPct = (v) => {
  const n = raNum(v);
  return n == null ? "-" : `${n >= 0 ? "+" : ""}${n.toFixed(1)}%`;
};
const raPlain = (v) => (v == null || v === "" ? "-" : raEsc(v));
const raCls = (v) => { const n = raNum(v); return n == null || n === 0 ? "" : n > 0 ? "pos" : "neg"; };

function raExit(ex) {
  const e = ex || {};
  const parts = [];
  if (e.stop != null) parts.push(`stop ${e.stop}`);
  if (e.target != null) parts.push(`target ${e.target}`);
  if (e.time_td != null) parts.push(`time ${e.time_td} td`);
  return parts.join(", ") || "-";
}

function raEntry(en) {
  const e = en || {};
  return e.type === "LIMIT" ? `LIMIT ${e.limit} (${e.fill_window_td || 1} td)` : (e.type || "-");
}

function raTable(headers, rows, leftCols) {
  if (!rows.length) return '<p class="note">None.</p>';
  const left = new Set(leftCols || []);
  const th = headers.map((h, i) => `<th class="${left.has(i) ? "l" : ""}">${raEsc(h)}</th>`).join("");
  const body = rows.map((r) => "<tr>" + r.map((c, i) =>
    `<td class="${left.has(i) ? "l" : ""}${c.cls ? " " + c.cls : ""}"${c.wrap ? ' style="white-space:normal"' : ""}>${c.html}</td>`).join("") + "</tr>").join("");
  return `<div class="tblwrap"><table class="tbl"><thead><tr>${th}</tr></thead><tbody>${body}</tbody></table></div>`;
}
const cell = (html, extra) => ({ html, ...(extra || {}) });

/* NAV curve points from scoreboard.nav_curve: [n, ...] or [{nav|value, date}, ...]. */
function raCurve(sb) {
  const raw = sb && (sb.nav_curve || sb.nav_series);
  if (!Array.isArray(raw)) return [];
  /* grader emits [[date, nav], ...]; tolerate bare numbers and {nav} objects. */
  return raw.map((p) => (Array.isArray(p) ? raNum(p[1])
    : typeof p === "object" && p ? raNum(p.nav != null ? p.nav : p.value) : raNum(p)))
    .filter((v) => v != null);
}

function raSparkline(values) {
  if (values.length < 2) return "";
  const w = 320, h = 70, pad = 4;
  const lo = Math.min(...values), hi = Math.max(...values), span = hi - lo || 1;
  const pts = values.map((v, i) => {
    const x = pad + (i * (w - 2 * pad)) / (values.length - 1);
    const y = h - pad - ((v - lo) * (h - 2 * pad)) / span;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  const up = values[values.length - 1] >= values[0];
  return `<svg class="ra-spark" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-label="NAV curve">` +
    `<polyline fill="none" stroke="${up ? "var(--green, #00d18f)" : "var(--red, #ff5d5d)"}" stroke-width="2" points="${pts}"/></svg>`;
}

function raKpi(label, value, sub) {
  return `<div class="kpi"><div class="l">${raEsc(label)}</div><div class="v">${raEsc(value)}</div>` +
    (sub ? `<div class="s">${raEsc(sub)}</div>` : "") + "</div>";
}

function raCard(o) {
  const opt = o.option;
  const lines = [];
  lines.push(`<div><b>${raEsc(o.instrument)}</b> <span class="badge info">${raEsc(o.side || o.kind || "")}</span> ` +
    `qty ${raPlain(o.qty)} @ ${o.ref_price != null ? raEsc(o.ref_price) : "chain"}</div>`);
  lines.push(`<div class="note">Entry ${raEsc(raEntry(o.entry))} | Exit ${raEsc(raExit(o.exit))} | ` +
    `Risk ${raPlain(o.risk_bps)} bps | Notional ${raMoney(o.notional)}</div>`);
  if (opt) {
    const stress = opt.status === "PASS_STRESS" ? "near" : "survives";
    lines.push(`<div><span class="badge ${stress}">${raEsc(opt.status)}</span> certified max loss $${raMoney(opt.max_loss)}` +
      `${raNum(opt.stress_move) != null ? ` (stress +${(opt.stress_move * 100).toFixed(0)}%)` : ""}</div>`);
    const legs = (opt.legs || []).map((l) => `${raPlain(l.qty)} ${raPlain(l.right)} ${raPlain(l.strike)} ${raPlain(l.expiry)}`).join(" / ");
    if (legs) lines.push(`<div class="note">Legs: ${legs}</div>`);
  }
  [["Thesis", o.thesis], ["Evidence", o.evidence], ["Survived", o.survived], ["What kills it", o.what_kills_it]]
    .forEach(([k, v]) => { if (v) lines.push(`<div><b>${k}:</b> ${raEsc(v)}</div>`); });
  return `<div class="card" style="margin:8px 0">${lines.join("")}</div>`;
}

function raScoreboard(sb) {
  const s = sb || {};
  /* grade_risk_agent.scoreboard().headline: percent units, fixed names. */
  const h = s.headline || {};
  const k = [];
  if (raNum(h.nav) != null) k.push(raKpi("NAV", raMoney(h.nav)));
  if (raNum(h.total_return_pct) != null) k.push(raKpi("Return", raPct(h.total_return_pct)));
  if (raNum(h.max_drawdown_pct) != null) k.push(raKpi("Max drawdown", raPct(h.max_drawdown_pct)));
  if (raNum(h.vs_spy_pct) != null) k.push(raKpi("Vs SPY", raPct(h.vs_spy_pct)));
  if (raNum(h.sharpe) != null) k.push(raKpi("Sharpe", raNum(h.sharpe).toFixed(2)));
  [["Brier 5d", h.brier_5], ["Brier 21d", h.brier_21]].forEach(([l, v]) => {
    if (raNum(v) != null) k.push(raKpi(l, raNum(v).toFixed(3)));
  });
  const spark = raSparkline(raCurve(s));
  if (!k.length && !spark) return '<p class="note">No scoreboard yet.</p>';
  return `<div class="kpis">${k.join("")}</div>${spark}`;
}

function renderRiskAgent(p) {
  const out = [];
  const posture = p.posture || {};
  const standDown = p.mode === "stand_down";
  out.push(`<p class="note">As of <b>${raEsc(p.asof)}</b> | ${raEsc(p.decision_id)} | model ${raPlain(p.model)} / ${raPlain(p.effort)} | ` +
    `published ${raPlain(p.published_at)}</p>`);
  if (standDown) out.push(`<div class="radar-warn"><b>DATA HOLD.</b> ${raEsc(p.reason)}</div>`);
  if ((p.warnings || []).length) {
    out.push(`<div class="radar-warn"><b>Data warnings</b><ul>${p.warnings.map((w) => `<li>${raEsc(w)}</li>`).join("")}</ul></div>`);
  }
  out.push("<h2>Posture</h2>");
  out.push(`<div class="card"><div>${raEsc(posture.summary)}</div>` +
    `<div class="note">Net beta ${raPlain(posture.net_beta)} | cash ${raPlain(posture.cash_pct)}%</div></div>`);

  out.push("<h2>Scoreboard</h2>" + raScoreboard(p.scoreboard));

  out.push("<h2>SPY forecasts</h2>");
  out.push(raTable(["Horizon", "P(up)", "q10", "q90", "Basis"],
    (p.forecasts || []).slice().sort((a, b) => (a.horizon_td || 0) - (b.horizon_td || 0)).map((f) => [
      cell(`${raPlain(f.horizon_td)} td`), cell(raPlain(f.p_up)), cell(raPct(f.q10_pct)), cell(raPct(f.q90_pct)),
      cell(raPlain(f.basis), { wrap: true }),
    ]), [4]));

  out.push("<h2>Today's new orders</h2>");
  out.push((p.new_orders || []).map(raCard).join("") || '<p class="note">None.</p>');

  out.push("<h2>Verdicts on held positions</h2>");
  out.push(raTable(["Position", "Symbol", "Verdict", "Reason"],
    (p.verdicts || []).map((v) => [cell(raPlain(v.id)), cell(raPlain(v.symbol)),
      cell(`<b>${raEsc(v.action)}</b>`), cell(raPlain(v.reason), { wrap: true })]), [0, 1, 2, 3]));

  const book = p.book || {};
  out.push("<h2>Paper book</h2>");
  out.push(`<div class="kpis">${raKpi("NAV", raMoney(book.nav))}${raKpi("Cash", raMoney(book.cash))}` +
    `${raKpi("Realised P&L", raMoney(book.realized_pnl, true))}${raKpi("Pending orders", raPlain(book.pending))}</div>`);
  out.push(raTable(["ID", "Symbol", "Side", "Qty", "Entry", "Mark", "P&L $", "Risk bps", "Stop", "Target", "Time exit"],
    (book.positions || []).map((x) => [
      cell(raPlain(x.id)), cell(raPlain(x.symbol)), cell(raPlain(x.side)), cell(raPlain(x.qty)),
      cell(raPlain(x.entry)), cell(raPlain(x.mark) + (x.stale_mark ? " (stale)" : "")),
      cell(raMoney(x.pnl, true), { cls: raCls(x.pnl) }), cell(raPlain(x.risk_bps)),
      cell(raPlain(x.stop)), cell(raPlain(x.target)), cell(raPlain(x.time_exit)),
    ]), [0, 1, 2]));
  const after = p.book_after || {};
  if (after.positions_after != null) {
    out.push(`<p class="note">After today's orders: ${raPlain(after.positions_after)} positions | ` +
      `${raPlain(after.risk_bps_after)} bps risk | gross ${raMoney(after.gross_notional_after)} = ${raPlain(after.gross_x_nav)}x NAV</p>`);
  }

  const rej = p.considered_and_rejected || [];
  if (rej.length) {
    out.push("<h2>Considered and rejected</h2><ul>" + rej.map((r) =>
      `<li><b>${raEsc(r.idea)}</b>: ${raEsc(r.reason)}</li>`).join("") + "</ul>");
  }
  const wl = p.watchlist || [];
  if (wl.length) {
    out.push("<h2>Watchlist</h2><ul>" + wl.map((w) =>
      `<li><b>${raEsc(w.idea)}</b>: ${raEsc(w.trigger)} (expires ${raEsc(w.expires)})</li>`).join("") + "</ul>");
  }
  return out.join("");
}

function renderMissing() {
  return '<div class="radar-warn">No Risk Agent run has been published yet. It publishes after the evening ' +
    'run of <code>daily_risk_agent.py</code>.</div>';
}

function renderFailure(message) {
  return `<div class="radar-warn">Could not load the Risk Agent: ${raEsc(message)}.</div>`;
}

async function loadRiskAgent(el, fetchImpl) {
  try {
    const r = await (fetchImpl || fetch)(RA_ENDPOINT, { cache: "no-store" });
    if (r.status === 404) { el.innerHTML = renderMissing(); return null; }
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const payload = await r.json();
    el.innerHTML = renderRiskAgent(payload);
    return payload;
  } catch (e) {
    el.innerHTML = renderFailure((e && e.message) || e);
    return null;
  }
}

function main() {
  if (typeof renderNav === "function") renderNav("risk-agent.html");
  const el = document.getElementById("content");
  if (el) loadRiskAgent(el);
}

if (typeof document !== "undefined") document.addEventListener("DOMContentLoaded", main);
if (typeof module !== "undefined") module.exports = {
  renderRiskAgent, renderMissing, renderFailure, loadRiskAgent, raEsc, raSparkline, raCurve,
  RA_ENDPOINT,
};
