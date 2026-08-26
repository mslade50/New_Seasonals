/* focus.js — fail-closed daily discretionary research shortlist. */
"use strict";

const FOCUS_ENDPOINT = "/discretionary-focus";
const FOCUS_HISTORY_ENDPOINT = "/discretionary-focus-history";
const FOCUS_SCHEMA = "discretionary-focus.v1";
const FOCUS_STATUSES = new Set(["READY", "NO_QUALIFIED_SETUP"]);
const FOCUS_PHASES = new Set(["PROVISIONAL", "FINAL"]);
const FOCUS_REQUIRED_CARD_TEXT = [
  "company_name", "why_now", "setup", "catalyst", "priced_in", "next_proof",
];
const FOCUS_TOP_LEVEL_KEYS = new Set([
  "schema_version", "research_only", "quick_review_created", "live_actions_enabled",
  "order_staging_enabled", "status", "phase", "as_of", "valid_for", "generated_at",
  "expires_at", "focus", "screen_summary", "provenance", "no_setup_reason",
]);
const FOCUS_CARD_KEYS = new Set([
  "rank", "ticker", "company_name", "why_now", "setup", "trigger", "invalidation",
  "catalyst", "priced_in", "next_proof", "event_date", "earnings_td", "technical",
  "sources",
]);
const FOCUS_FORBIDDEN_KEYS = new Set([
  "action", "action_id", "allocation", "approval_status", "approved_for_capital",
  "broker", "decision", "dry_run_required", "limit_order", "notional", "order",
  "order_id", "order_type", "position_size", "position_size_pct",
  "proposed_weight_pct", "quantity", "quick_review", "risk_amt", "risk_bps",
  "shares", "side", "tif",
]);
const FOCUS_PRICE_KEYS = [
  "price", "pivot", "level", "trigger_price", "stop_price", "invalidation_price",
];

function focusEsc(value) {
  return String(value == null ? "" : value)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

function focusSafeURL(value) {
  try {
    const text = String(value || "").trim();
    if (!/^https?:\/\//i.test(text)) return "";
    const url = new URL(text);
    return ["http:", "https:"].includes(url.protocol) ? url.href : "";
  } catch (_) { return ""; }
}

function focusIsRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function focusISODate(value) {
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const parsed = new Date(`${value}T00:00:00Z`);
  return !Number.isNaN(parsed.getTime()) && parsed.toISOString().slice(0, 10) === value;
}

function focusZonedTime(value) {
  if (typeof value !== "string" || !/(?:Z|[+-]\d{2}:\d{2})$/.test(value)) return null;
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function focusText(value) {
  return typeof value === "string" && value.trim().length > 0;
}

function focusNewYorkParts(value) {
  const parsed = value instanceof Date ? value : focusZonedTime(value);
  if (!parsed) return null;
  const parts = Object.fromEntries(new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23",
  }).formatToParts(parsed).filter((part) => part.type !== "literal")
    .map((part) => [part.type, part.value]));
  return {
    date: `${parts.year}-${parts.month}-${parts.day}`,
    hour: Number(parts.hour), minute: Number(parts.minute), second: Number(parts.second),
  };
}

function focusHasForbiddenKey(value) {
  if (Array.isArray(value)) return value.some(focusHasForbiddenKey);
  if (!focusIsRecord(value)) return false;
  return Object.entries(value).some(([key, nested]) =>
    FOCUS_FORBIDDEN_KEYS.has(String(key).trim().toLowerCase()) || focusHasForbiddenKey(nested));
}

function focusHasOnlyKeys(value, allowed) {
  return Object.keys(value).every((key) => allowed.has(key));
}

function focusPriceExpression(value) {
  if (focusText(value)) return true;
  if (!focusIsRecord(value) || !focusText(value.condition)) return false;
  let hasNumeric = false;
  for (const key of FOCUS_PRICE_KEYS) {
    if (value[key] == null) continue;
    if (typeof value[key] !== "number" || !Number.isFinite(value[key])) return false;
    hasNumeric = true;
  }
  return !hasNumeric || value.price_basis === "RAW_AS_TRADED";
}

function focusSourceValid(source, generatedAt) {
  if (!focusIsRecord(source) || !focusText(source.source_id) || !focusText(source.label))
    return false;
  if (!focusText(source.url) || !focusSafeURL(source.url)) return false;
  const generatedLocal = focusNewYorkParts(generatedAt);
  if (!generatedLocal) return false;
  let sourceTime;
  if (focusISODate(source.as_of)) {
    if (source.as_of > generatedLocal.date) return false;
    sourceTime = new Date(`${source.as_of}T12:00:00Z`);
  } else {
    sourceTime = focusZonedTime(source.as_of);
    if (!sourceTime || sourceTime > new Date(generatedAt.getTime() + 5 * 60 * 1000)) return false;
  }
  if (generatedAt - sourceTime > 550 * 24 * 60 * 60 * 1000) return false;
  return typeof source.primary === "boolean";
}

function focusScreenSummaryValid(summary, selectedCount) {
  if (!focusIsRecord(summary)) return false;
  const fields = ["input_count", "technical_pass_count", "research_pass_count", "selected_count"];
  if (!fields.every((key) => Number.isInteger(summary[key]) && summary[key] >= 0)) return false;
  if (summary.selected_count !== selectedCount) return false;
  if (summary.technical_pass_count > summary.input_count) return false;
  if (summary.research_pass_count > summary.technical_pass_count) return false;
  if (summary.selected_count > summary.research_pass_count) return false;
  if (!focusIsRecord(summary.rejected_counts)) return false;
  const rejected = Object.values(summary.rejected_counts);
  if (!rejected.every((count) => Number.isInteger(count) && count >= 0)) return false;
  return rejected.reduce((total, count) => total + count, 0) + selectedCount === summary.input_count;
}

function focusDigestValid(value) {
  return typeof value === "string" && /^[0-9a-f]{64}$/.test(value);
}

function focusProvenanceValid(provenance, payload, generatedAt) {
  if (!focusIsRecord(provenance) ||
      !focusText(provenance.screen_snapshot_id) ||
      !focusText(provenance.research_snapshot_id) ||
      !focusText(provenance.policy_version)) return false;
  const screenAt = focusZonedTime(provenance.screen_captured_at);
  const researchAt = focusZonedTime(provenance.research_as_of);
  if (!screenAt || !researchAt) return false;
  const futureTolerance = 5 * 60 * 1000;
  if (screenAt > new Date(generatedAt.getTime() + futureTolerance) ||
      researchAt > new Date(generatedAt.getTime() + futureTolerance)) return false;
  const screenMaxAge = 96 * 60 * 60 * 1000;
  if (generatedAt - screenAt > screenMaxAge || generatedAt - researchAt > 36 * 60 * 60 * 1000)
    return false;
  if (Object.prototype.hasOwnProperty.call(provenance, "screen_digest") &&
      !focusDigestValid(provenance.screen_digest)) return false;
  if (Object.prototype.hasOwnProperty.call(provenance, "research_digest") &&
      !focusDigestValid(provenance.research_digest)) return false;
  return true;
}

function validateFocusCard(card, index, payload, generatedAt) {
  if (!focusIsRecord(card)) return `focus[${index}] must be an object`;
  if (!focusHasOnlyKeys(card, FOCUS_CARD_KEYS)) return `focus[${index}] has unexpected fields`;
  if (card.rank !== index + 1) return `focus[${index}].rank must equal ${index + 1}`;
  if (!focusText(card.ticker) || !/^[A-Z0-9][A-Z0-9.^\/-]{0,19}$/.test(card.ticker))
    return `focus[${index}].ticker is invalid`;
  for (const key of FOCUS_REQUIRED_CARD_TEXT) {
    if (!focusText(card[key])) return `focus[${index}].${key} is required`;
  }
  if (!focusPriceExpression(card.trigger)) return `focus[${index}].trigger is invalid`;
  if (!focusIsRecord(card.invalidation) ||
      !focusPriceExpression(card.invalidation.technical) ||
      !focusText(card.invalidation.thesis_kill))
    return `focus[${index}].invalidation is invalid`;
  if (!focusISODate(card.event_date))
    return `focus[${index}].event_date must be YYYY-MM-DD`;
  if (card.event_date < payload.valid_for)
    return `focus[${index}].event_date cannot precede valid_for`;
  if (!Number.isInteger(card.earnings_td) || card.earnings_td <= 5)
    return `focus[${index}].earnings_td must be greater than five`;
  if (!focusIsRecord(card.technical)) return `focus[${index}].technical is required`;
  const observedAt = focusZonedTime(card.technical.observed_at);
  if (!observedAt || observedAt > new Date(generatedAt.getTime() + 5 * 60 * 1000))
    return `focus[${index}].technical.observed_at is invalid`;
  const maxAgeMs = 96 * 60 * 60 * 1000;
  if (generatedAt - observedAt > maxAgeMs)
    return `focus[${index}].technical.observed_at is stale`;
  if (card.technical.setup_gate !== "PASS" || card.technical.liquidity_gate !== "PASS")
    return `focus[${index}].technical gates must pass`;
  if (typeof card.technical.setup_quality !== "number" ||
      !Number.isFinite(card.technical.setup_quality) ||
      card.technical.setup_quality < 0 || card.technical.setup_quality > 100)
    return `focus[${index}].technical.setup_quality is invalid`;
  if (!Array.isArray(card.sources) || card.sources.length === 0)
    return `focus[${index}].sources is required`;
  if (!card.sources.every((source) => focusSourceValid(source, generatedAt)))
    return `focus[${index}].sources is invalid`;
  if (!card.sources.some((source) => source.primary === true))
    return `focus[${index}].sources needs a primary source`;
  return null;
}

function focusUnavailable(reason) {
  return { state: "UNAVAILABLE", payload: null, reason: String(reason || "invalid payload") };
}

function validateFocusPayload(payload, now = new Date()) {
  if (!focusIsRecord(payload)) return focusUnavailable("payload must be an object");
  if (payload.schema_version !== FOCUS_SCHEMA)
    return focusUnavailable(`schema_version must be ${FOCUS_SCHEMA}`);
  if (!focusHasOnlyKeys(payload, FOCUS_TOP_LEVEL_KEYS))
    return focusUnavailable("payload has unexpected fields");
  if (focusHasForbiddenKey(payload)) return focusUnavailable("payload contains execution fields");
  if (payload.research_only !== true || payload.quick_review_created !== false ||
      payload.live_actions_enabled !== false || payload.order_staging_enabled !== false)
    return focusUnavailable("research-only safety envelope failed");
  if (!FOCUS_PHASES.has(payload.phase)) return focusUnavailable("phase is invalid");
  if (!FOCUS_STATUSES.has(payload.status))
    return focusUnavailable("status must be READY or NO_QUALIFIED_SETUP");
  if (!focusISODate(payload.as_of)) return focusUnavailable("as_of must be YYYY-MM-DD");
  if (!focusISODate(payload.valid_for)) return focusUnavailable("valid_for must be YYYY-MM-DD");
  if (payload.valid_for < payload.as_of)
    return focusUnavailable("valid_for cannot precede as_of");

  const generatedAt = focusZonedTime(payload.generated_at);
  const expiresAt = focusZonedTime(payload.expires_at);
  if (!generatedAt) return focusUnavailable("generated_at needs an explicit timezone");
  if (!expiresAt) return focusUnavailable("expires_at needs an explicit timezone");
  if (expiresAt <= generatedAt)
    return focusUnavailable("expires_at must be later than generated_at");
  const generatedLocal = focusNewYorkParts(generatedAt);
  const expiresLocal = focusNewYorkParts(expiresAt);
  if (!generatedLocal || (payload.phase === "FINAL" && generatedLocal.date !== payload.valid_for))
    return focusUnavailable("generated_at must be on valid_for in New York");
  const closeExpiry = expiresLocal && expiresLocal.date === payload.valid_for &&
    expiresLocal.second === 0 && expiresLocal.minute === 15 &&
    [13, 16].includes(expiresLocal.hour);
  if (!closeExpiry)
    return focusUnavailable("expires_at must be 15 minutes after the XNYS close");
  if (!(now instanceof Date) || Number.isNaN(now.getTime()))
    return focusUnavailable("validation clock is invalid");
  if (generatedAt > new Date(now.getTime() + 5 * 60 * 1000))
    return focusUnavailable("generated_at is implausibly in the future");

  if (!Array.isArray(payload.focus)) return focusUnavailable("focus must be an array");
  if (payload.focus.length > 2) return focusUnavailable("focus contains more than two names");
  if (payload.status === "READY" && payload.focus.length === 0)
    return focusUnavailable("READY requires one or two names");
  if (payload.status === "NO_QUALIFIED_SETUP" && payload.focus.length !== 0)
    return focusUnavailable("NO_QUALIFIED_SETUP requires an empty focus list");
  if (payload.status === "NO_QUALIFIED_SETUP" && !focusText(payload.no_setup_reason))
    return focusUnavailable("NO_QUALIFIED_SETUP requires a reason");
  if (payload.status === "READY" && payload.no_setup_reason)
    return focusUnavailable("READY cannot include a no-setup reason");

  if (!focusScreenSummaryValid(payload.screen_summary, payload.focus.length))
    return focusUnavailable("screen_summary is invalid");
  if (!focusProvenanceValid(payload.provenance, payload, generatedAt))
    return focusUnavailable("provenance is invalid");

  const tickers = new Set();
  for (let index = 0; index < payload.focus.length; index++) {
    const error = validateFocusCard(payload.focus[index], index, payload, generatedAt);
    if (error) return focusUnavailable(error);
    if (tickers.has(payload.focus[index].ticker))
      return focusUnavailable("focus contains duplicate tickers");
    tickers.add(payload.focus[index].ticker);
  }

  if (now >= expiresAt) {
    return { state: "EXPIRED", payload, reason: `expired at ${payload.expires_at}` };
  }
  if (payload.status === "NO_QUALIFIED_SETUP") {
    return { state: "NO_QUALIFIED_SETUP", payload, reason: "" };
  }
  return { state: "READY", payload, reason: "" };
}

function focusTimeHTML(value) {
  const parsed = focusZonedTime(value);
  const label = parsed
    ? parsed.toLocaleString("en-US", {
        month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZoneName: "short",
      })
    : String(value || "");
  return `<time datetime="${focusEsc(value)}">${focusEsc(label)}</time>`;
}

function focusSourcesHTML(sources) {
  const items = (sources || []).map((source) => {
    const url = focusSafeURL(source.url);
    const label = focusEsc(source.label);
    const meta = `${source.primary ? "primary" : "secondary"} · ${focusEsc(source.as_of)}`;
    return `<li>${url ? `<a href="${focusEsc(url)}" target="_blank" rel="noopener">${label}</a>` : label}<small>${meta}</small></li>`;
  });
  if (!items.length) return "";
  return `<details class="focus-sources"><summary>Sources</summary><ul>${items.join("")}</ul></details>`;
}

function focusPlainValue(value) {
  if (focusText(value)) return value.trim();
  if (!focusIsRecord(value)) return String(value == null ? "" : value);
  const populated = Object.keys(value).filter((key) => value[key] != null && value[key] !== "");
  if (populated.length === 1 && populated[0] === "condition") return focusPlainValue(value.condition);
  const preferred = ["condition", "technical", "thesis_kill", "live_state", "price_basis"];
  const keys = [...new Set([...preferred, ...Object.keys(value).sort()])];
  return keys.filter((key) => value[key] != null && value[key] !== "")
    .map((key) => `${key.replace(/_/g, " ")}: ${focusPlainValue(value[key])}`).join(" | ");
}

function focusCardHTML(card) {
  const event = `Next earnings · ${focusEsc(card.event_date)} · ${focusEsc(card.earnings_td)} trading days`;
  return `<article class="focus-card">
    <div class="focus-card-head">
      <div>
        <div class="focus-rank">FOCUS ${card.rank}</div>
        <div class="focus-ticker"><strong>${focusEsc(card.ticker)}</strong>
          ${card.company_name ? `<span>${focusEsc(card.company_name)}</span>` : ""}</div>
      </div>
      <span class="focus-live-badge">RESEARCH PRIORITY</span>
    </div>
    <p class="focus-why">${focusEsc(card.why_now)}</p>
    <div class="focus-setup"><span>Setup</span><p>${focusEsc(card.setup)}</p></div>
    <div class="focus-detail-grid">
      <div><span>Required trigger</span><p>${focusEsc(focusPlainValue(card.trigger))}</p></div>
      <div><span>Immediate invalidation</span><p>${focusEsc(focusPlainValue(card.invalidation))}</p></div>
      <div><span>Catalyst</span><p>${focusEsc(card.catalyst)}</p></div>
      <div><span>What is priced in</span><p>${focusEsc(card.priced_in)}</p></div>
      <div><span>Next proof</span><p>${focusEsc(card.next_proof)}</p></div>
      <div><span>Event clock</span><p>${event}</p></div>
    </div>
    ${focusSourcesHTML(card.sources)}
  </article>`;
}

function focusHeroHTML(payload, count) {
  const armedURL = focusSafeURL(payload.provenance.tradingview_armed_url);
  const liveURL = focusSafeURL(payload.provenance.tradingview_live_url);
  const screenLinks = [
    armedURL ? `<a href="${focusEsc(armedURL)}" target="_blank" rel="noopener">Open Armed screen</a>` : "",
    liveURL ? `<a href="${focusEsc(liveURL)}" target="_blank" rel="noopener">Open Live RVOL screen</a>` : "",
  ].filter(Boolean).join("");
  return `<section class="focus-hero">
    <div>
      <div class="focus-eyebrow">Discretionary Focus · ${focusEsc(payload.phase)} · valid for ${focusEsc(payload.valid_for)}</div>
      <h1>${count} name${count === 1 ? "" : "s"} deserve attention</h1>
      <p>The screen allocated research time; price and live relative-volume confirmation are still required.</p>
      <div class="focus-safety">Research only · maximum two names · zero is valid · no recommendation or capital action.</div>
      ${screenLinks ? `<div class="focus-screen-links">${screenLinks}</div>` : ""}
    </div>
    <div class="focus-clock">
      <div><span>Evidence through</span><strong>${focusEsc(payload.as_of)}</strong></div>
      <div><span>Generated</span><strong>${focusTimeHTML(payload.generated_at)}</strong></div>
      <div><span>Expires</span><strong>${focusTimeHTML(payload.expires_at)}</strong></div>
    </div>
  </section>`;
}

function focusStateHTML(result) {
  if (!result || result.state === "UNAVAILABLE") {
    const reason = result && result.reason ? result.reason : "the current payload could not be loaded";
    return `<section class="focus-empty unavailable">
      <div class="focus-state-label">UNAVAILABLE</div>
      <h1>Focus list unavailable</h1>
      <p>No names are shown because the current research payload failed validation.</p>
      <small>${focusEsc(reason)}</small>
    </section>`;
  }

  const payload = result.payload;
  if (result.state === "EXPIRED") {
    return `<section class="focus-empty expired">
      <div class="focus-state-label">EXPIRED</div>
      <h1>Focus list expired</h1>
      <p>The ${focusEsc(payload.valid_for)} list is no longer valid. No prior names are carried forward.</p>
      <small>Generated ${focusTimeHTML(payload.generated_at)} · expired ${focusTimeHTML(payload.expires_at)}</small>
    </section>`;
  }

  if (result.state === "NO_QUALIFIED_SETUP") {
    return `<section class="focus-empty clear">
      <div class="focus-state-label">PROCESS COMPLETE</div>
      <h1>No qualified setup today</h1>
      <p>The funnel completed for ${focusEsc(payload.valid_for)} and correctly returned zero names.</p>
      <p>${focusEsc(payload.no_setup_reason)}</p>
      <small>Evidence through ${focusEsc(payload.as_of)} · generated ${focusTimeHTML(payload.generated_at)}</small>
    </section>`;
  }

  return `${focusHeroHTML(payload, payload.focus.length)}
    <section class="focus-grid">${payload.focus.map(focusCardHTML).join("")}</section>`;
}

function validateFocusHistory(payload) {
  if (!focusIsRecord(payload) || payload.schema_version !== "discretionary-focus-history.v1" ||
      !Array.isArray(payload.items) || payload.items.length > 10) return [];
  const seen = new Set();
  const valid = [];
  for (const item of payload.items) {
    if (!focusIsRecord(item) || !focusISODate(item.valid_for) || seen.has(item.valid_for) ||
        !FOCUS_STATUSES.has(item.status) || !FOCUS_PHASES.has(item.phase) ||
        !focusZonedTime(item.generated_at) || !Array.isArray(item.focus) || item.focus.length > 2)
      return [];
    if ((item.status === "READY" && item.focus.length === 0) ||
        (item.status === "NO_QUALIFIED_SETUP" && item.focus.length !== 0)) return [];
    if (!item.focus.every((row) => focusIsRecord(row) && focusText(row.ticker) &&
        /^[A-Z0-9][A-Z0-9.^\/-]{0,19}$/.test(row.ticker) && focusText(row.company_name)))
      return [];
    seen.add(item.valid_for);
    valid.push(item);
  }
  return valid;
}

function focusHistoryHTML(items) {
  if (!items.length) return `<section class="focus-history"><h2>Recent history</h2>
    <p class="muted">No validated archived sessions are available yet.</p></section>`;
  const rows = items.map((item) => {
    const names = item.focus.length
      ? item.focus.map((row) => `<strong>${focusEsc(row.ticker)}</strong>`).join(" · ")
      : "No qualified setup";
    return `<tr><td>${focusEsc(item.valid_for)}</td><td>${focusEsc(item.phase)}</td>
      <td>${focusEsc(item.status.replace(/_/g, " "))}</td><td>${names}</td></tr>`;
  }).join("");
  return `<section class="focus-history"><h2>Recent history</h2>
    <p>Validated archived outcomes only; prior names are never carried into today's list.</p>
    <div class="focus-history-scroll"><table><thead><tr><th>Session</th><th>Phase</th>
      <th>Outcome</th><th>Names</th></tr></thead><tbody>${rows}</tbody></table></div></section>`;
}

async function initFocus() {
  renderNav("focus.html");
  const root = document.getElementById("focusContent");
  const historyRoot = document.getElementById("focusHistory");
  if (!root) return;
  let result;
  try {
    const payload = await fetchJSON(FOCUS_ENDPOINT);
    result = validateFocusPayload(payload, new Date());
  } catch (error) {
    result = focusUnavailable(error && error.message ? error.message : "request failed");
  }
  root.innerHTML = focusStateHTML(result);
  if (result.payload) {
    setAsof(`Focus ${result.payload.valid_for} · generated ${result.payload.generated_at}`);
  } else {
    setAsof("Focus unavailable");
  }
  if (historyRoot) {
    try {
      const history = await fetchJSON(FOCUS_HISTORY_ENDPOINT);
      historyRoot.innerHTML = focusHistoryHTML(validateFocusHistory(history));
    } catch (_) {
      historyRoot.innerHTML = `<section class="focus-history"><h2>Recent history</h2>
        <p class="muted">History is temporarily unavailable; the current list above is unaffected.</p></section>`;
    }
  }
}

if (typeof document !== "undefined") document.addEventListener("DOMContentLoaded", initFocus);
