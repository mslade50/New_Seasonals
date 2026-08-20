/* fundamentals.js — one-screen research inbox for the fundamental sleeve. */
"use strict";

const FUND_STATE_KEY = "fundamentalResearchState.v1";
const FUND_ACTIONS = new Set(["DEEPEN", "WATCH", "PASS", "CLEAR"]);

function esc(value) {
  return String(value == null ? "" : value)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

function safeURL(value) {
  try {
    const url = new URL(String(value || ""), window.location.href);
    return ["http:", "https:"].includes(url.protocol) ? url.href : "";
  } catch (_) { return ""; }
}

function actionRecord(state, ticker) {
  return (state && state.actions && state.actions[ticker]) || null;
}

function actionName(state, ticker) {
  const record = actionRecord(state, ticker);
  return record && FUND_ACTIONS.has(record.action) ? record.action : "";
}

function localState() {
  try {
    const parsed = JSON.parse(localStorage.getItem(FUND_STATE_KEY) || "null");
    if (parsed && typeof parsed === "object" && parsed.actions && typeof parsed.actions === "object") {
      return parsed;
    }
  } catch (_) { /* ignore corrupt local state */ }
  return { version: 1, updated_at: null, actions: {} };
}

function saveLocalState(state) {
  try { localStorage.setItem(FUND_STATE_KEY, JSON.stringify(state)); } catch (_) { /* best effort */ }
}

async function loadResearchState() {
  try {
    const response = await fetch("/fundamental-state", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const state = await response.json();
    if (!state || typeof state.actions !== "object") throw new Error("invalid state");
    saveLocalState(state);
    return { state, mode: "cloud" };
  } catch (_) {
    return { state: localState(), mode: "local" };
  }
}

function updateLocalAction(state, ticker, action) {
  const next = {
    version: 1,
    updated_at: new Date().toISOString(),
    actions: Object.assign({}, (state && state.actions) || {}),
  };
  if (action === "CLEAR") delete next.actions[ticker];
  else next.actions[ticker] = { action, updated_at: next.updated_at };
  saveLocalState(next);
  return next;
}

async function persistAction(state, ticker, action, asOf) {
  try {
    const response = await fetch("/fundamental-state", {
      method: "POST",
      cache: "no-store",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker, action, as_of: asOf || null }),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const next = await response.json();
    if (!next || typeof next.actions !== "object") throw new Error("invalid state");
    saveLocalState(next);
    return { state: next, mode: "cloud" };
  } catch (_) {
    return { state: updateLocalAction(state, ticker, action), mode: "local" };
  }
}

function tagHTML(row) {
  const tags = [];
  const trend = String(row.trend_state || "UNAVAILABLE").toUpperCase();
  tags.push(`<span class="fund-tag trend-${esc(trend.toLowerCase())}">200d ${esc(trend)}</span>`);
  if (row.product_circle) {
    const score = row.product_fit_score == null ? "" : ` · ${esc(row.product_fit_score)}/10`;
    tags.push(`<span class="fund-tag">Product fit${score}</span>`);
  }
  if (row.founder_led) tags.push('<span class="fund-tag">Founder-led</span>');
  return tags.join("");
}

function actionBadge(action) {
  if (action === "DEEPEN") return '<span class="fund-state deepen">Deeper work requested</span>';
  if (action === "WATCH") return '<span class="fund-state watch">Watching for proof</span>';
  if (action === "PASS") return '<span class="fund-state pass">Passed</span>';
  return "";
}

function actionButtons(ticker, current) {
  const options = [
    ["DEEPEN", "Deepen"],
    ["WATCH", "Watch"],
    ["PASS", "Pass"],
  ];
  return `<div class="fund-actions" aria-label="Research action for ${esc(ticker)}">
    ${options.map(([action, label]) =>
      `<button class="fund-action ${current === action ? "selected" : ""}" type="button"
        data-ticker="${esc(ticker)}" data-action="${action}">${label}</button>`).join("")}
  </div>`;
}

function sourceHTML(sources) {
  const rows = (sources || []).filter(source => source && source.label);
  if (!rows.length) return "";
  return `<details class="fund-inline-details"><summary>Sources</summary><ul class="fund-source-list">
    ${rows.map(source => {
      const url = safeURL(source.url);
      const label = url
        ? `<a href="${esc(url)}" target="_blank" rel="noopener">${esc(source.label)}</a>`
        : esc(source.label);
      const date = source.as_of ? ` · ${esc(source.as_of)}` : "";
      return `<li>${label}${date}</li>`;
    }).join("")}
  </ul></details>`;
}

function reviewCard(review, state) {
  const current = actionName(state, review.ticker);
  return `<article class="fund-review-card">
    <div class="fund-card-top">
      <div>
        <div class="fund-ticker-line"><strong>${esc(review.ticker)}</strong><span>${esc(review.company_name)}</span></div>
        <div class="fund-tags">${tagHTML(review)}</div>
      </div>
      ${actionBadge(current)}
    </div>
    <p class="fund-verdict">${esc(review.verdict)}</p>
    <div class="fund-thesis-grid">
      <div><span>What may be mispriced</span><p>${esc(review.mispricing)}</p></div>
      <div><span>What is priced in</span><p>${esc(review.priced_in)}</p></div>
      <div><span>Valuation support</span><p>${esc(review.valuation)}</p></div>
      <div><span>Downside mechanism</span><p>${esc(review.downside)}</p></div>
      <div><span>Proof trigger</span><p>${esc(review.proof_trigger)}</p></div>
      <div><span>Kill condition</span><p>${esc((review.kill_conditions || [])[0] || "Not yet defined")}</p></div>
    </div>
    <div class="fund-decision-row">
      <div><span class="fund-micro-label">Your decision</span><p>${esc(review.exact_decision)}</p></div>
      ${actionButtons(review.ticker, current)}
    </div>
    ${sourceHTML(review.sources)}
  </article>`;
}

function activeCard(row, state) {
  const current = actionName(state, row.ticker);
  return `<article class="fund-active-card">
    <div class="fund-card-top">
      <div>
        <div class="fund-ticker-line"><strong>${esc(row.ticker)}</strong><span>${esc(row.company_name)}</span></div>
        <div class="fund-tags">${tagHTML(row)}</div>
      </div>
      ${actionBadge(current)}
    </div>
    <p>${esc(row.verdict)}</p>
    <div class="fund-next"><span>Next proof point</span>${esc(row.next_review || "No review trigger recorded.")}</div>
    ${actionButtons(row.ticker, current)}
  </article>`;
}

function suppressedHTML(payload, state) {
  const rows = [...(payload.reviews || []), ...(payload.active_research || [])]
    .filter(row => actionName(state, row.ticker) === "PASS");
  if (!rows.length) return "";
  return `<details class="fund-drawer"><summary>Suppressed <span>${rows.length}</span></summary>
    <div class="fund-suppressed">
      ${rows.map(row => `<div><span><strong>${esc(row.ticker)}</strong> ${esc(row.company_name)}</span>
        <button class="fund-action" type="button" data-ticker="${esc(row.ticker)}" data-action="CLEAR">Undo</button></div>`).join("")}
    </div>
  </details>`;
}

function passesHTML(audit) {
  const summary = audit && audit.pass_summary;
  const reasons = (summary && summary.reasons) || [];
  if (!summary || !reasons.length) return "";
  const trend = summary.trend_overlay || {};
  return `<details class="fund-drawer fund-pass-drawer"><summary>Why ${Number(summary.background_count || 0).toLocaleString()} companies are not in front of you</summary>
    <p class="cap">${esc(summary.reason_method || "")}</p>
    <div class="fund-pass-list">
      ${reasons.map(reason => {
        const pct = Math.max(0, Math.min(100, Number(reason.pct || 0)));
        return `<div class="fund-pass-row">
          <div class="fund-pass-head"><strong>${esc(reason.label)}</strong><span>${Number(reason.count || 0).toLocaleString()} · ${pct.toFixed(1)}%</span></div>
          <div class="fund-pass-meter" aria-hidden="true"><i style="width:${pct}%"></i></div>
          <p>${esc(reason.explanation)}</p>
        </div>`;
      }).join("")}
    </div>
    <div class="fund-trend-overlay"><strong>Trend overlay</strong>
      <span>${Number(trend.without_full_confirmation || 0).toLocaleString()} lack full confirmation · ${Number(trend.amber || 0).toLocaleString()} amber · ${Number(trend.red || 0).toLocaleString()} red</span>
      <small>${esc(trend.method || "")}</small>
    </div>
  </details>`;
}

function lensesHTML(lenses) {
  const l = lenses || {};
  return `<details class="fund-drawer"><summary>Research lenses</summary>
    <div class="fund-lens-grid">
      <div><strong>${esc(l.product_circle_count || 0)}</strong><span>product-centric companies</span></div>
      <div><strong>${esc(l.founder_led_count || 0)}</strong><span>current founder-CEOs</span></div>
      <div><strong>${esc(l.founder_product_overlap_count || 0)}</strong><span>overlap both lists</span></div>
    </div>
    <p class="cap">${esc(l.trend_rule || "")}</p>
  </details>`;
}

function auditHTML(audit) {
  const a = audit || {};
  const sources = (a.sources || []).filter(source => source && source.label);
  const routes = Object.entries((a.research_funnel && a.research_funnel.routes) || {});
  const stateHealth = a.state_health || {};
  return `<details class="fund-drawer"><summary>Research audit</summary>
    <div class="fund-audit-grid">
      <div><span>Discovered</span><strong>${Number(a.discovered || 0).toLocaleString()}</strong></div>
      <div><span>Eligible</span><strong>${Number(a.research_eligible || 0).toLocaleString()}</strong></div>
      <div><span>Baseline-ready</span><strong>${Number(a.baseline_ready || a.fundamental_covered || 0).toLocaleString()}</strong></div>
      <div><span>Deep-ready</span><strong>${Number(a.deep_ready || 0).toLocaleString()}</strong></div>
      <div><span>Decision-ready</span><strong>${Number(a.decision_ready || 0).toLocaleString()}</strong></div>
      <div><span>SEC packages</span><strong>${Number(a.sec_covered || 0).toLocaleString()}</strong></div>
    </div>
    ${sources.length ? `<p class="cap">Source vintages: ${sources.map(source => `${esc(source.label)} (${esc(source.as_of || "undated")})`).join(" · ")}</p>` : ""}
    ${routes.length ? `<p class="cap">Research routes: ${routes.map(([name, count]) => `${esc(name.replaceAll("_", " ").toLowerCase())} ${Number(count || 0).toLocaleString()}`).join(" / ")}</p>` : ""}
    <p class="cap">State: controls ${esc(stateHealth.controls || "MISSING")} · triggers ${esc(stateHealth.triggers || "MISSING")} · evidence ${esc(stateHealth.evidence || "MISSING")} · portfolio ${esc(stateHealth.portfolio || "MISSING")}</p>
    <p class="cap">SEC package presence is not a line-by-line filed-fact tie-out. The full candidate queue stays out of this page by design.</p>
  </details>`;
}

function render(payload, state, mode, message) {
  const root = document.getElementById("fundamentalContent");
  const passed = ticker => actionName(state, ticker) === "PASS";
  const reviews = (payload.reviews || []).filter(row => !passed(row.ticker)).slice(0, 3);
  const active = (payload.active_research || []).filter(row => !passed(row.ticker)).slice(0, 3);
  const needsReview = reviews.length > 0;
  const portfolio = payload.portfolio || {};
  const modeText = mode === "cloud" ? "Synced to private research state" : "Saved on this device";
  const syncText = message || modeText;

  root.innerHTML = `
    <section class="fund-hero ${needsReview ? "needs-review" : "clear"}">
      <div class="fund-hero-copy">
        <div class="fund-eyebrow">Fundamental inbox · ${esc(payload.as_of || "as-of unavailable")}</div>
        <h1>${needsReview ? `${reviews.length} quick review${reviews.length === 1 ? "" : "s"}` : "Nothing needs your attention"}</h1>
        <p>${needsReview
          ? "These names cleared the research bar far enough to require a judgment."
          : `${active.length ? `${active.length} ${active.length === 1 ? "name remains" : "names remain"} in active research.` : "No names are waiting in active research."} None currently has a proven mispricing, adequate valuation support, and observable thesis trigger.`}</p>
        <div class="fund-safety">Research controls only. Your clicks change what gets researched next; they never allocate capital or create orders.</div>
      </div>
      <div class="fund-cap-card" aria-label="Fundamental sleeve limits">
        <div><strong>${portfolio.position_count == null ? "—" : esc(portfolio.position_count)} / ${esc(portfolio.max_positions ?? 10)}</strong><span>positions</span></div>
        <div><strong>${portfolio.capital_allocated_pct == null ? "—" : `${esc(portfolio.capital_allocated_pct)}%`} / ${esc(portfolio.capital_cap_pct ?? 30)}%</strong><span>capital</span></div>
        <small>${esc(portfolio.tracking_posture || "Allocation stays manual")}</small>
      </div>
    </section>

    ${needsReview ? `<section class="fund-section"><div class="fund-section-head"><div><h2>Quick review</h2><p>Only the decision-relevant evidence.</p></div></div>
      <div class="fund-review-list">${reviews.map(row => reviewCard(row, state)).join("")}</div></section>` : ""}

    ${active.length ? `<section class="fund-section fund-active-section"><div class="fund-section-head"><div><h2>Active research</h2><p>Optional. Nothing here requires a decision today.</p></div></div>
      <div class="fund-active-grid">${active.map(row => activeCard(row, state)).join("")}</div></section>` : ""}

    <div class="fund-drawers">
      ${suppressedHTML(payload, state)}
      ${passesHTML(payload.audit)}
      ${lensesHTML(payload.lenses)}
      ${auditHTML(payload.audit)}
    </div>
    <div class="fund-sync" id="fundSync" role="status" aria-live="polite">${esc(syncText)}</div>`;
}

document.addEventListener("DOMContentLoaded", async () => {
  renderNav("fundamentals.html");
  const root = document.getElementById("fundamentalContent");
  try {
    const [payload, stateResult] = await Promise.all([
      fetchJSON("data/fundamentals.json"),
      loadResearchState(),
    ]);
    let state = stateResult.state;
    let mode = stateResult.mode;
    setAsof(payload.as_of ? `Research ${payload.as_of}` : "Fundamentals");
    render(payload, state, mode);

    root.addEventListener("click", async event => {
      const button = event.target.closest("button[data-action][data-ticker]");
      if (!button || button.disabled) return;
      const ticker = String(button.dataset.ticker || "").toUpperCase();
      const action = String(button.dataset.action || "").toUpperCase();
      if (!/^[A-Z][A-Z0-9.-]{0,9}$/.test(ticker) || !FUND_ACTIONS.has(action)) return;
      root.querySelectorAll(`button[data-ticker="${CSS.escape(ticker)}"]`).forEach(el => { el.disabled = true; });
      const result = await persistAction(state, ticker, action, payload.as_of);
      state = result.state;
      mode = result.mode;
      const verb = action === "CLEAR" ? "Cleared" : `${action.charAt(0)}${action.slice(1).toLowerCase()} selected for`;
      render(payload, state, mode, `${verb} ${ticker}. ${mode === "cloud" ? "Synced privately." : "Saved on this device."}`);
    });
  } catch (error) {
    root.innerHTML = `<div class="fetchfail"><strong>Fundamental research is unavailable.</strong><br>
      <span class="mono">${esc(error && error.message ? error.message : error)}</span></div>`;
    setAsof("Fundamentals unavailable");
  }
});
