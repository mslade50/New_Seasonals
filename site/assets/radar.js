/* radar.js — the momentum radar's weekly plans, and a deep link that stages one.
 *
 * Reads /radar-recs (Pages Function -> R2 key radar_recs.json, published by
 * scripts/upload_radar_recs.py). VERBATIM RULE: the radar's book engine already
 * did every calculation — entry trigger, limit cap, stop, target, share count,
 * fill window, time exit. This page formats those numbers and hands them to the
 * Execution ticket unchanged. It never derives one from another. If a value is
 * missing it renders as "-" and the Stage button refuses, rather than guessing.
 *
 * Stage flow: execution.html?stage=radar&... -> execution.js applyRadarPrefill()
 * fills the entry-bracket ticket. Nothing is sent; the operator still reviews
 * the readout, hits Send, and passes the agent's own gates.
 */
"use strict";

const RADAR_ENDPOINT = "/radar-recs";
const RADAR_STRATEGY = "Momentum_Radar";   // must match radar_trail_sync.py --strategy
// The engine's entry vocabulary -> the execution ticket's entry_type.
const ORDER_TYPE_MAP = { BUY_STOP_LIMIT: "STP_LMT", BUY_LIMIT: "LMT", SELL_LIMIT: "LMT" };

const num = (v) => (typeof v === "number" && isFinite(v) ? v : null);
const px = (v) => (num(v) == null ? "-" : v.toFixed(2));
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function sideOf(rec) {
  const order = String((rec.entry || {}).order || "").toUpperCase();
  return order.startsWith("SELL") ? "SELL" : "BUY";
}

function entryTypeOf(rec) {
  return ORDER_TYPE_MAP[String((rec.entry || {}).order || "").toUpperCase()] || null;
}

/* The stage link carries EXPLICIT levels, unlike the Seasonal tab's link which
   passes an ATR and lets the ticket derive stop/target from the manual-seasonal
   convention. The radar already decided all of them, so deriving anything here
   would be inventing a second opinion. */
function stageHref(rec, recsDate) {
  const e = rec.entry || {}, s = rec.stop || {}, t = rec.targets || {},
        tm = rec.time || {}, z = rec.sizing || {};
  const type = entryTypeOf(rec);
  const entry = num(e.trigger) != null ? e.trigger : num(e.limit);
  const q = new URLSearchParams({
    stage: "radar", sym: String(rec.ticker || "").toUpperCase(), side: sideOf(rec),
    type: type || "", entry: entry == null ? "" : String(entry),
    strat: RADAR_STRATEGY, refdate: recsDate || "",
  });
  if (num(e.limit_cap) != null) q.set("cap", String(e.limit_cap));
  if (num(s.price) != null) q.set("stop", String(s.price));
  if (num(t.t1) != null) q.set("target", String(t.t1));
  if (num(z.shares) != null) q.set("qty", String(z.shares));
  if (tm.valid_through) q.set("exp", String(tm.valid_through));
  if (tm.time_exit_date) q.set("ts", String(tm.time_exit_date));
  return `execution.html?${q.toString()}`;
}

/* A plan is stageable only if the ticket can express it exactly. Anything else
   is shown with the reason, never a half-filled ticket. */
function stageBlockers(rec, ageDays) {
  const e = rec.entry || {}, s = rec.stop || {}, z = rec.sizing || {}, tm = rec.time || {};
  const out = [];
  const type = entryTypeOf(rec);
  if (!type) out.push(`entry order ${e.order || "?"} has no ticket equivalent`);
  if (type === "STP_LMT" && num(e.limit_cap) == null) out.push("stop-limit with no limit cap");
  if (num(e.trigger) == null && num(e.limit) == null) out.push("no entry price");
  if (num(s.price) == null) out.push("no stop price");
  if (!(num(z.shares) > 0)) out.push("no share count");
  const today = new Date().toLocaleDateString("en-CA");
  if (tm.valid_through && String(tm.valid_through) < today)
    out.push(`fill window closed ${tm.valid_through}`);
  if (tm.valid_from && String(tm.valid_from) > today)
    out.push(`not live until ${tm.valid_from}`);
  if (ageDays != null && ageDays > 10) out.push(`recs are ${ageDays}d old`);
  return out;
}

function recCard(rec, recsDate, ageDays) {
  const e = rec.entry || {}, s = rec.stop || {}, t = rec.targets || {},
        tm = rec.time || {}, z = rec.sizing || {};
  const blockers = stageBlockers(rec, ageDays);
  const type = entryTypeOf(rec);
  const entryTxt = type === "STP_LMT"
    ? `STP LMT ${px(e.trigger)} <span class="cap" style="display:inline">cap ${px(e.limit_cap)}</span>`
    : `LMT ${px(num(e.limit) != null ? e.limit : e.trigger)}`;
  const caps = (z.caps_hit || []).length
    ? ` <span class="cap" style="display:inline">(${(z.caps_hit || []).map(esc).join(", ")})</span>` : "";
  const btn = blockers.length
    ? `<span class="cap" style="color:#ffc14d">cannot stage: ${blockers.map(esc).join("; ")}</span>`
    : `<a class="btn" href="${esc(stageHref(rec, recsDate))}">Stage &rarr;</a>`;
  return `<div class="card radar-card">
    <div class="radar-head">
      <b>${esc(rec.ticker)}</b>
      <span class="radar-pill">${esc(rec.setup_grade || "?")}</span>
      <span class="cap">${esc(rec.plan_type || "")}</span>
      <span class="cap">${esc(rec.sector || "")}</span>
    </div>
    <div class="radar-grid">
      <div><span class="cap">Entry</span> ${entryTxt}</div>
      <div><span class="cap">Stop</span> ${px(s.price)} <span class="cap" style="display:inline">${
        num(s.atr_mult) == null ? "" : s.atr_mult + " ATR"}</span></div>
      <div><span class="cap">T1</span> ${px(t.t1)} <span class="cap" style="display:inline">${
        num(t.t1_frac) == null ? "" : Math.round(t.t1_frac * 100) + "%"}</span></div>
      <div><span class="cap">Size</span> ${num(z.shares) == null ? "-" : z.shares} sh${caps}</div>
      <div><span class="cap">Window</span> ${esc(tm.valid_from || "?")} &rarr; ${esc(tm.valid_through || "?")}</div>
      <div><span class="cap">Time exit</span> ${esc(tm.time_exit_date || "-")}</div>
    </div>
    <div class="radar-ticket">${esc(rec.ticket || "")}</div>
    ${e.gap_rule ? `<p class="cap">Engine note: ${esc(e.gap_rule)}. A native IBKR stop-limit does
      NOT die on that gap &mdash; the trigger fires and a resting limit stays at the cap, fillable on a
      pullback. Check the open before staging if you want the rule literally.</p>` : ""}
    <div class="radar-actions">${btn}</div>
  </div>`;
}

function positionRow(p) {
  return `<tr><td class="l">${esc(p.ticker)}</td><td class="l">${esc(p.status)}</td>
    <td>${num(p.shares_remaining) == null ? "-" : p.shares_remaining}</td>
    <td>${px(p.current_stop)}</td><td class="l">${esc(p.stop_kind || "-")}</td>
    <td>${num(p.unrealized_r) == null ? "-" : p.unrealized_r.toFixed(2)}</td>
    <td class="l">${esc(p.time_exit_date || "-")}</td><td class="l">${esc(p.next_action || "-")}</td></tr>`;
}

function render(payload) {
  const el = document.getElementById("content");
  const age = num(payload.age_days);
  const stale = age != null && age > 10;
  const bits = [`recs <b>${esc(payload.date)}</b>`,
                age == null ? "" : `${age}d old`,
                `published ${esc(payload.generated_at)}`,
                payload.regime && payload.regime.active_rule
                  ? `regime ${esc(payload.regime.active_rule)}` : ""].filter(Boolean);
  const banners = [];
  if (stale) banners.push(`<div class="radar-warn">These recs are ${age} days old. The radar runs weekly;
    a gap this long means a run was missed. Staging off a stale vintage is on you.</div>`);
  if (payload.mint_blocked) banners.push(`<div class="radar-warn">The radar did NOT mint new plans this week
    (<code>${esc(payload.mint_blocked)}</code>). Existing positions stand; there are no new entries.</div>`);
  if (payload.pull && payload.pull !== "pulled" && payload.pull !== "skipped")
    banners.push(`<div class="radar-warn">Publisher could not refresh the radar clone: ${esc(payload.pull)}</div>`);

  const recs = payload.new_recs || [];
  const positions = payload.open_positions || [];
  const c = payload.counts || {};
  el.innerHTML = `
    <p class="cap">${bits.join(" &nbsp;&middot;&nbsp; ")}</p>
    ${banners.join("")}
    <h2>New recommendations (${recs.length})</h2>
    ${recs.length ? recs.map((r) => recCard(r, payload.date, age)).join("")
                  : `<p class="cap">No new plans this week.</p>`}
    <p class="cap">Also in the engine's output, not shown here:
      ${["plan_only", "watch_only", "budget_cut", "zeroed"]
        .map((k) => `${c[k] || 0} ${k.replace("_", " ")}`).join(" &middot; ")}.</p>
    <h2>Open positions (${positions.length})</h2>
    ${positions.length ? `<table class="tbl"><thead><tr>
        <th class="l">Ticker</th><th class="l">Status</th><th>Shares</th><th>Stop</th>
        <th class="l">Kind</th><th>Unreal R</th><th class="l">Time exit</th><th class="l">Next action</th>
      </tr></thead><tbody>${positions.map(positionRow).join("")}</tbody></table>
      <p class="cap">Stops shown are what the engine stepped to this week. Applying them to the live
      book is <code>radar_trail_sync.py</code> (never lowers a stop, only touches
      <code>${RADAR_STRATEGY}</code>-tagged legs).</p>`
      : `<p class="cap">No open radar positions.</p>`}`;
}

async function main() {
  renderNav("radar.html");
  const el = document.getElementById("content");
  try {
    const payload = await fetchJSON(RADAR_ENDPOINT);
    if (payload && payload.error) throw new Error(payload.error);
    render(payload);
  } catch (e) {
    el.innerHTML = `<div class="radar-warn">Could not load the radar recs: ${esc(e.message || e)}.
      Publish them with <code>python scripts/upload_radar_recs.py</code> on the trading box.</div>`;
  }
}

if (typeof document !== "undefined") document.addEventListener("DOMContentLoaded", main);
if (typeof module !== "undefined") module.exports = { stageHref, stageBlockers, entryTypeOf, sideOf };
