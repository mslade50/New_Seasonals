/* idea.js — Idea Check tab: submit a trade idea, watch the verdict arrive.
 *
 * POST /idea-check queues the idea in R2 (Pages Function); a local poller runs
 * the reviewer and writes idea_check/results/<id>.json. GET /idea-check returns
 * the newest 10 requests joined to their results. Every string from an idea or
 * a result is set with textContent, never innerHTML.
 */
"use strict";

const IDEA_ENDPOINT = "/idea-check";
const IDEA_POLL_MS = 15000;
const IDEA_MAX = 2000;
const IDEA_BADGE = {
  KILL: "kill", SURVIVES: "survives", "NEAR-MISS": "near", "NEEDS-INFO": "info",
};

// { label, cls } for one GET item; in-flight means queued or running.
function ideaBadge(item) {
  const r = item && item.result;
  if (!r) return { label: "queued", cls: "queued", inflight: true };
  if (r.status === "running") return { label: "running", cls: "warn", inflight: true };
  if (r.status === "error") return { label: "error", cls: "on", inflight: false };
  if (r.status === "done" && IDEA_BADGE[r.verdict])
    return { label: r.verdict, cls: IDEA_BADGE[r.verdict], inflight: false };
  return { label: "error", cls: "on", inflight: false };
}

function anyInflight(items) { return (items || []).some((i) => ideaBadge(i).inflight); }

function ideaEl(doc, tag, cls, text) {
  const el = doc.createElement(tag);
  if (cls) el.className = cls;
  if (text != null) el.textContent = String(text);
  return el;
}

function ideaList(doc, label, rows) {
  const box = ideaEl(doc, "div", "");
  box.appendChild(ideaEl(doc, "div", "cap", label));
  const ul = ideaEl(doc, "ul", "");
  rows.forEach((t) => ul.appendChild(ideaEl(doc, "li", "", t)));
  box.appendChild(ul);
  return box;
}

function ideaCard(doc, item) {
  const b = ideaBadge(item);
  const r = item.result;
  const card = ideaEl(doc, "div", "card idea-item");
  const head = ideaEl(doc, "div", "");
  head.appendChild(ideaEl(doc, "span", `badge ${b.cls}`, b.label));
  head.appendChild(ideaEl(doc, "span", "cap-inline", ` ${item.submitted_at || ""}`));
  card.appendChild(head);
  card.appendChild(ideaEl(doc, "div", "idea-text", item.text));
  if (r) {
    if (r.headline) card.appendChild(ideaEl(doc, "div", "", r.headline));
    if (r.status === "error" || (r.status !== "running" && b.label === "error"))
      card.appendChild(ideaEl(doc, "div", "radar-warn", r.error || "The reviewer reported an error."));
    if (Array.isArray(r.numbers) && r.numbers.length) card.appendChild(ideaList(doc, "Numbers", r.numbers));
    if (Array.isArray(r.tweaks) && r.tweaks.length) card.appendChild(ideaList(doc, "Tweaks", r.tweaks));
    if (r.body_md) {
      const d = ideaEl(doc, "details", "");
      d.appendChild(ideaEl(doc, "summary", "", "Full write-up"));
      d.appendChild(ideaEl(doc, "pre", "", r.body_md));
      card.appendChild(d);
    }
  }
  return card;
}

function renderIdeas(doc, container, items) {
  container.textContent = "";
  if (!items.length) { container.appendChild(ideaEl(doc, "p", "cap", "No checks yet.")); return; }
  items.forEach((i) => container.appendChild(ideaCard(doc, i)));
}

let ideaTimer = null;

async function loadIdeas() {
  const el = document.getElementById("content");
  let items = [];
  try {
    const data = await fetchJSON(IDEA_ENDPOINT);
    if (data && data.error) throw new Error(data.error);
    items = data.requests || [];
    renderIdeas(document, el, items);
  } catch (e) {
    el.textContent = "";
    el.appendChild(ideaEl(document, "div", "radar-warn", `Could not load checks: ${e.message || e}`));
    return;
  }
  clearTimeout(ideaTimer);
  if (anyInflight(items)) ideaTimer = setTimeout(loadIdeas, IDEA_POLL_MS);
}

async function submitIdea() {
  const box = document.getElementById("ideaText");
  const btn = document.getElementById("ideaSubmit");
  const msg = document.getElementById("ideaMsg");
  const text = box.value.trim();
  if (!text) { msg.textContent = "Type an idea first."; return; }
  if (text.length > IDEA_MAX) { msg.textContent = `Keep it under ${IDEA_MAX} characters.`; return; }
  btn.disabled = true;
  msg.textContent = "Submitting...";
  try {
    const r = await fetch(IDEA_ENDPOINT, { method: "POST", cache: "no-store",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
    box.value = "";
    msg.textContent = "Queued.";
    await loadIdeas();
  } catch (e) {
    msg.textContent = `Submit failed: ${e.message || e}`;
  } finally {
    btn.disabled = false;
  }
}

function main() {
  renderNav("idea.html");
  document.getElementById("ideaSubmit").addEventListener("click", submitIdea);
  loadIdeas();
}

if (typeof document !== "undefined") document.addEventListener("DOMContentLoaded", main);
if (typeof module !== "undefined") module.exports = { ideaBadge, anyInflight, renderIdeas, IDEA_POLL_MS };
