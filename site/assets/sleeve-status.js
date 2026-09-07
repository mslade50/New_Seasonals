/* Independent of the static site-build snapshot; reads current canonical reports. */
"use strict";
(() => {
  const safe = value => String(value ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const stamp = value => {
    const d = new Date(value);
    return value && Number.isFinite(d.valueOf()) ? d.toLocaleString("en-US", { timeZone: "America/New_York", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }) + " ET" : "Not reported";
  };
  const links = new Set(["events.html", "execution.html", "execution.html#hedge", "risk.html"]);
  function card(row) {
    const needs = row.health === "unavailable" || row.health === "aging" || row.runtime_health === "unavailable";
    const observed = row.kind === "Research pending" ? "Scope reviewed" : "State reported";
    return `<article class="card sleeve-card">
      <div class="sleeve-heading"><h3>${safe(row.name)}</h3><span class="sleeve-kind">${safe(row.kind)}</span></div>
      <div class="sleeve-state ${needs ? "sleeve-attention" : ""}">${safe(row.deployment)}</div>
      <p>${safe(row.summary)}</p>
      <dl class="sleeve-dates"><dt>${observed}</dt><dd>${safe(row.report_date || "Not reported")}${row.health === "aging" ? " · report aging" : row.health === "unavailable" ? " · unavailable" : ""}</dd>
      <dt>Machine checked</dt><dd>${safe(row.checked_at ? stamp(row.checked_at) : row.runtime_note)}${row.checked_at && row.runtime_health === "unavailable" ? " · refresh needed" : ""}</dd></dl>
      <div class="sleeve-next"><span>Next</span> ${safe(row.next)}</div>
      ${links.has(row.href) ? `<a class="sleeve-link" href="${row.href}">Open ${row.href.startsWith("events") ? "Events" : row.href.startsWith("risk") ? "Risk" : "Execution"} →</a>` : ""}
    </article>`;
  }
  let busy = false;
  async function refresh() {
    if (busy || !document.getElementById("sleeve-cards")) return;
    busy = true;
    const message = document.getElementById("sleeve-updated");
    const button = document.getElementById("sleeve-refresh");
    button.disabled = true;
    try {
      const r = await fetch("/sleeve-status", { cache: "no-store", signal: AbortSignal.timeout(15000) });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const body = await r.json();
      if (body.schema !== "sleeve-status.v1" || !Array.isArray(body.sleeves)) throw new Error("Invalid status response");
      document.getElementById("sleeve-cards").innerHTML = body.sleeves.map(card).join("");
      message.textContent = `Fetched ${stamp(body.fetched_at)} · source dates below show when each system reported`;
      message.classList.remove("sleeve-attention");
    } catch {
      message.textContent = "Status refresh failed. Any cards below are from the previous fetch; refresh or sign in again.";
      message.classList.add("sleeve-attention");
      if (!document.getElementById("sleeve-cards").children.length) document.getElementById("sleeve-cards").textContent = "Status is unavailable. This does not change trading controls.";
    } finally {
      busy = false;
      button.disabled = false;
    }
  }
  document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("sleeve-refresh")?.addEventListener("click", refresh);
    refresh();
    setInterval(() => { if (!document.hidden) refresh(); }, 60000);
    document.addEventListener("visibilitychange", () => { if (!document.hidden) refresh(); });
  });
})();
