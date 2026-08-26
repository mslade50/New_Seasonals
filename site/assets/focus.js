/* focus.js — fail-closed daily discretionary research shortlist. */
"use strict";

const FOCUS_ENDPOINT = "/discretionary-focus";
const FOCUS_SCHEMA = "discretionary-focus.v1";
const FOCUS_STATUSES = new Set(["READY", "NO_QUALIFIED_SETUP"]);
const FOCUS_REQUIRED_CARD_TEXT = [
  "why_now", "setup", "trigger", "invalidation", "catalyst", "priced_in", "next_proof",
];

function focusEsc(value) {
  return String(value == null ? "" : value)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

function focusSafeURL(value) {
  try {
    const url = new URL(String(value || ""), window.location.href);
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

function validateFocusCard(card, index) {
  if (!focusIsRecord(card)) return `focus[${index}] must be an object`;
  if (card.rank !== index + 1) return `focus[${index}].rank must equal ${index + 1}`;
  if (!focusText(card.ticker) || !/^[A-Z0-9.^\/-]{1,15}$/.test(card.ticker))
    return `focus[${index}].ticker is invalid`;
  if (card.company_name != null && typeof card.company_name !== "string")
    return `focus[${index}].company_name must be text`;
  for (const key of FOCUS_REQUIRED_CARD_TEXT) {
    if (!focusText(card[key])) return `focus[${index}].${key} is required`;
  }
  if (card.event_date != null && !focusISODate(card.event_date))
    return `focus[${index}].event_date must be YYYY-MM-DD`;
  if (card.event_label != null && typeof card.event_label !== "string")
    return `focus[${index}].event_label must be text`;
  if (card.sources != null) {
    if (!Array.isArray(card.sources) || card.sources.length > 10)
      return `focus[${index}].sources must contain at most 10 items`;
    for (let sourceIndex = 0; sourceIndex < card.sources.length; sourceIndex++) {
      const source = card.sources[sourceIndex];
      if (!focusIsRecord(source) || !focusText(source.label))
        return `focus[${index}].sources[${sourceIndex}] needs a label`;
      if (source.url != null && typeof source.url !== "string")
        return `focus[${index}].sources[${sourceIndex}].url must be text`;
    }
  }
  return null;
}

function focusUnavailable(reason) {
  return { state: "UNAVAILABLE", payload: null, reason: String(reason || "invalid payload") };
}

function validateFocusPayload(payload, now = new Date()) {
  if (!focusIsRecord(payload)) return focusUnavailable("payload must be an object");
  if (payload.schema_version !== FOCUS_SCHEMA)
    return focusUnavailable(`schema_version must be ${FOCUS_SCHEMA}`);
  if (payload.research_only !== true)
    return focusUnavailable("research_only must be true");
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
  if (!(now instanceof Date) || Number.isNaN(now.getTime()))
    return focusUnavailable("validation clock is invalid");

  if (!Array.isArray(payload.focus)) return focusUnavailable("focus must be an array");
  if (payload.focus.length > 2) return focusUnavailable("focus contains more than two names");
  if (payload.status === "READY" && payload.focus.length === 0)
    return focusUnavailable("READY requires one or two names");
  if (payload.status === "NO_QUALIFIED_SETUP" && payload.focus.length !== 0)
    return focusUnavailable("NO_QUALIFIED_SETUP requires an empty focus list");

  for (let index = 0; index < payload.focus.length; index++) {
    const error = validateFocusCard(payload.focus[index], index);
    if (error) return focusUnavailable(error);
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
    return `<li>${url ? `<a href="${focusEsc(url)}" target="_blank" rel="noopener">${label}</a>` : label}</li>`;
  });
  if (!items.length) return "";
  return `<details class="focus-sources"><summary>Sources</summary><ul>${items.join("")}</ul></details>`;
}

function focusCardHTML(card) {
  const event = card.event_date
    ? `${focusEsc(card.event_label || "Event")} · ${focusEsc(card.event_date)}`
    : "No dated event supplied";
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
      <div><span>Required trigger</span><p>${focusEsc(card.trigger)}</p></div>
      <div><span>Immediate invalidation</span><p>${focusEsc(card.invalidation)}</p></div>
      <div><span>Catalyst</span><p>${focusEsc(card.catalyst)}</p></div>
      <div><span>What is priced in</span><p>${focusEsc(card.priced_in)}</p></div>
      <div><span>Next proof</span><p>${focusEsc(card.next_proof)}</p></div>
      <div><span>Event clock</span><p>${event}</p></div>
    </div>
    ${focusSourcesHTML(card.sources)}
  </article>`;
}

function focusHeroHTML(payload, count) {
  return `<section class="focus-hero">
    <div>
      <div class="focus-eyebrow">Discretionary Focus · valid for ${focusEsc(payload.valid_for)}</div>
      <h1>${count} name${count === 1 ? "" : "s"} deserve attention</h1>
      <p>The screen allocated research time; price and live relative-volume confirmation are still required.</p>
      <div class="focus-safety">Research only · maximum two names · zero is valid · no recommendation or capital action.</div>
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
      <small>Evidence through ${focusEsc(payload.as_of)} · generated ${focusTimeHTML(payload.generated_at)}</small>
    </section>`;
  }

  return `${focusHeroHTML(payload, payload.focus.length)}
    <section class="focus-grid">${payload.focus.map(focusCardHTML).join("")}</section>`;
}

async function initFocus() {
  renderNav("focus.html");
  const root = document.getElementById("focusContent");
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
}

if (typeof document !== "undefined") document.addEventListener("DOMContentLoaded", initFocus);
