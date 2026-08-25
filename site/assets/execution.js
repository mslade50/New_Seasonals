/* execution.js — execution-bridge dashboard (TWS-style).

   Layout, top to bottom:
     - connection bar: agent online light + account tabs (Primary / PA) + NLV
     - Positions panel  (live read-only book from the agent) + row actions
     - Open Orders panel (live working orders) + Cancel
     - Scheduled closing orders (legs that fire at today's close)
     - New Order ticket: entry bracket / scheduled option buy / close-only / flatten / echo
     - Activity: recent commands + results

   Commands execute LIVE when the agent is armed (mode banner amber) and DRY-RUN
   otherwise — the agent decides by AGENT_LIVE_ENABLED + LIVE_TYPES, and every
   mutating action confirms with a LIVE/Dry-run dialog. Mode is only trusted from
   a FRESH book while the agent is online; a null/stale book or offline agent is
   UNKNOWN and treated as LIVE (fail dangerous). Positions/orders come from
   book_snapshot.py over the agent's read-only IBKR connection. Static parts render once; the data
   panels refresh every 4s. */
"use strict";

document.addEventListener("DOMContentLoaded", initExecution);

const state = { account: "primary", book: null, status: null };
let pollTimer = null;
let FUT_SPECS = {};   // symbol/alias -> {exchange,multiplier,min_tick,...}; drives the FUT readout
const frontState = { id: null, timer: null, manual: false };   // FUT live-contract discovery + front month

/* Deep-link prefill from the Seasonal tab (execution.html?stage=1&sym=&side=&win=&atr=&px=):
   fills the entry-bracket ticket per the manual-seasonal conventions —
   stop 1.0 / 1.3 / 1.6 ATR for a 5 / 10 / 21 td window, target 2:1, time stop
   at the window end (weekends skipped; holidays are NOT — nudge the date if it
   lands on one), qty = 30 bps of the SELECTED account's NLV once the book
   loads. Everything lands in editable fields — nothing is sent. */
const STAGE_MULTS = { 5: 1.0, 10: 1.3, 21: 1.6 };
const STAGE_RISK_BPS = 30;
const stage = (() => {
  const q = new URLSearchParams(location.search);
  if (q.get("stage") !== "1") return null;
  const sym = String(q.get("sym") || "").toUpperCase().trim();
  const side = q.get("side") === "SELL" ? "SELL" : "BUY";
  const win = parseInt(q.get("win") || "", 10);
  const atr = parseFloat(q.get("atr") || "");
  const px = parseFloat(q.get("px") || "");
  const mult = STAGE_MULTS[win];
  if (!sym || !mult || !(atr > 0) || !(px > 0)) return null;
  return { sym, side, win, atr, px, mult, qtyPending: true };
})();

function addTradingDays(from, n) {
  const d = new Date(from);
  let left = n;
  while (left > 0) {
    d.setDate(d.getDate() + 1);
    const wd = d.getDay();
    if (wd !== 0 && wd !== 6) left--;
  }
  return d.toLocaleDateString("en-CA");
}

/* Deep-link prefill from the Radar tab (execution.html?stage=radar&...).
   Unlike the Seasonal link above, which passes an ATR and lets this file DERIVE
   stop/target from the manual-seasonal convention, the radar's book engine has
   already decided every level. So these params are explicit and are copied in
   verbatim — deriving anything here would be a second opinion competing with
   the engine's. Everything lands in editable fields; nothing is sent. */
const radarStage = (() => {
  const q = new URLSearchParams(location.search);
  if (q.get("stage") !== "radar") return null;
  const sym = String(q.get("sym") || "").toUpperCase().trim();
  const entry = parseFloat(q.get("entry") || "");
  if (!sym || !(entry > 0)) return null;
  const n = (k) => { const v = parseFloat(q.get(k) || ""); return v > 0 ? v : null; };
  return {
    sym, entry,
    side: q.get("side") === "SELL" ? "SELL" : "BUY",
    type: String(q.get("type") || "LMT").toUpperCase(),
    cap: n("cap"), stop: n("stop"), target: n("target"), qty: n("qty"),
    exp: q.get("exp") || "", ts: q.get("ts") || "",
    soFrac: n("sofrac"), soTarget: n("sotarget"),
    acct: q.get("acct") === "primary" ? "primary" : null,
    strat: String(q.get("strat") || "").trim(), refdate: q.get("refdate") || "",
  };
})();

function applyRadarPrefill() {
  if (!radarStage) return;
  const r = radarStage;
  const setv = (id, v) => {
    const e = document.getElementById(id);
    if (e) e.value = String(v);
    ticketDraft[id] = String(v);
  };
  document.getElementById("cmdType").value = "entry_bracket";
  ticketDraft.f_entry_type = r.type;
  syncFields();
  const act = document.getElementById("f_action");
  if (act) act.value = r.side;
  const sel = document.getElementById("f_entry_type");
  if (sel) sel.value = r.type;
  syncEntryTypeFields();
  setv("f_symbol", r.sym);
  setv("f_entry", r.entry);
  if (r.cap != null) setv("f_entry_cap", r.cap);
  if (r.stop != null) setv("f_stop", r.stop);
  if (r.target != null) setv("f_target", r.target);
  if (r.qty != null) setv("f_qty", r.qty);
  if (r.exp) setv("f_expiry", r.exp);
  if (r.ts) setv("f_timestop", r.ts);
  if (r.strat) setv("f_strategy", r.strat);
  if (r.soFrac != null) setv("f_so_frac", r.soFrac);
  if (r.soTarget != null) setv("f_so_target", r.soTarget);
  // Radar plans are a PRIMARY-account sleeve. They are sized off the radar's own
  // $250k book, which is unrelated to PA's NLV, and a single plan's notional can
  // approach PA's whole live cap. Pin the account rather than inherit whichever
  // tab happened to be selected. radar_trail_sync.py defaults to primary to match.
  if (r.acct === "primary" && state.account !== "primary") setAccount("primary");
  updateReadout();
  const msg = document.getElementById("cmdMsg");
  if (msg) msg.textContent = `prefilled from Radar — levels and size copied verbatim from the ` +
    `book engine's plan${r.refdate ? ` (${r.refdate})` : ""}; review and send`;
}

function applyStagePrefill() {
  if (!stage) return;
  const sgn = stage.side === "BUY" ? 1 : -1;
  const dist = stage.mult * stage.atr;
  const setv = (id, v) => {
    const e = document.getElementById(id);
    if (e) e.value = String(v);
    ticketDraft[id] = String(v);
  };
  document.getElementById("cmdType").value = "entry_bracket";
  syncFields();
  const act = document.getElementById("f_action");
  if (act) act.value = stage.side;
  setv("f_symbol", stage.sym);
  setv("f_entry", stage.px.toFixed(2));
  setv("f_stop", (stage.px - sgn * dist).toFixed(2));
  setv("f_target", (stage.px + sgn * 2 * dist).toFixed(2));
  setv("f_timestop", addTradingDays(new Date(), stage.win));
  updateReadout();
  const msg = document.getElementById("cmdMsg");
  if (msg) msg.textContent = `prefilled from Seasonal — ${stage.win}d window, ${stage.mult} ATR stop, 2:1 target; qty fills from the ${state.account} NLV`;
}

function fillStageQty() {
  if (!stage || !stage.qtyPending) return;
  const ab = acctBook();
  if (!ab || ab.nlv == null) return;
  stage.qtyPending = false;
  const q = document.getElementById("f_qty");
  if (q && !q.value) {
    const n = Math.floor((ab.nlv * STAGE_RISK_BPS / 10000) / (stage.mult * stage.atr));
    if (n > 0) { q.value = String(n); ticketDraft.f_qty = String(n); }
  }
  updateReadout();
}

async function initExecution() {
  renderNav("execution.html");
  const el = document.getElementById("content");
  el.innerHTML = shell();
  FUT_SPECS = (await fetchJSONOrNull("assets/futures_specs.json")) || {};   // for the FUT ticket readout
  document.querySelectorAll("[data-acct]").forEach((b) =>
    b.addEventListener("click", () => setAccount(b.dataset.acct)));
  document.getElementById("cmdType").addEventListener("change", syncFields);
  document.getElementById("cmdSend").addEventListener("click", sendTicket);
  document.getElementById("cmdFields").addEventListener("input", updateReadout);
  document.getElementById("fs_go").addEventListener("click", sizeFutures);
  ["fs_symbol", "fs_entry", "fs_stop", "fs_target", "fs_risk", "fs_riskpct"].forEach((id) => {
    const e = document.getElementById(id);
    if (e) e.addEventListener("keydown", (ev) => { if (ev.key === "Enter") sizeFutures(); });
  });
  syncFields();
  applyStagePrefill();               // seasonal deep link: prefill the bracket ticket
  applyRadarPrefill();               // radar deep link: verbatim levels from the book engine
  await poll();
  pollTimer = setInterval(poll, 4000);
}

function shell() {
  return `
    <div id="modeBanner"></div>
    <div id="connBar"></div>
    <div class="exec-tabs" style="margin:10px 0 4px">
      <button class="btn" data-acct="primary">Primary</button>
      <button class="btn ghost" data-acct="pa">PA</button>
    </div>
    <div id="positions"></div>
    <div id="orders" style="margin-top:14px"></div>
    <div id="closers" style="margin-top:14px"></div>

    <div class="card" style="max-width:760px;margin-top:18px">
      <div style="font:700 14px inherit;margin-bottom:4px">New order</div>
      <p class="cap" style="margin:0 0 10px">Bracket: stock, futures, or USD-pair FX entry as <b>limit</b> or <b>market</b>; stock entries also support <b>market-on-close</b> and <b>stop-limit</b> (a breakout trigger plus the worst fill you will take &mdash; risk, R:R and notional are all shown and gated at that cap, not the trigger). <b>Scheduled option buy</b> waits until the specified ET time, then resolves the live chain, chooses the nearest target-delta call or put, sizes from the current ask, and submits a SMART market order. Its premium budget is approximate because the market fill can slip. Stop, target, <b>time stop</b> (closes at market 15:59 ET on that date), and limit-entry expiry are optional. <b>Primary futures are uncapped</b>: IBKR buying power and exchange limits are the hard constraints; large stopped risk and unprotected entries require a secondary approval. PA keeps its $30k futures ceiling. <b>Attach exits</b> adds a stop / target / time-stop OCA group. <b>Close only</b> leaves working orders untouched; <b>Flatten</b> cancels them before closing. Submits per the mode banner above &mdash; live when armed.</p>
      <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:8px">
        <label class="cap">Type</label>
        <select id="cmdType">
          <option value="entry_bracket">entry bracket</option>
          <option value="scheduled_option">scheduled option buy</option>
          <option value="exit_attach">attach exits</option>
          <option value="close_only">close only (leave orders)</option>
          <option value="flatten">flatten</option>
          <option value="echo">echo (ping)</option>
        </select>
        <span class="cap">Account: <b id="ticketAcct">primary</b></span>
      </div>
      <div id="cmdFields" style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:8px"></div>
      <div id="ticketReadout" style="font:12px inherit;margin:0 0 10px;min-height:16px"></div>
      <button class="btn" id="cmdSend" data-mutation disabled>Send order</button>
      <span id="cmdMsg" class="cap" style="margin-left:10px"></span>
    </div>

    <div class="card" style="max-width:760px;margin-top:18px">
      <div style="font:700 14px inherit;margin-bottom:4px">Futures sizing <span class="cap" style="display:inline;font-weight:400">&mdash; risk &rarr; contracts + notional (read-only)</span></div>
      <p class="cap" style="margin:0 0 10px">Enter a futures symbol with entry/stop and a risk budget; the agent sizes the contract count off the live multiplier and shows the notional exposure. Places nothing. Risk % uses the selected account's NLV.</p>
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px">
        <label class="cap">Symbol</label><input id="fs_symbol" value="ES" style="width:70px;text-transform:uppercase">
        <label class="cap">Entry</label><input id="fs_entry" style="width:78px">
        <label class="cap">Stop</label><input id="fs_stop" style="width:78px">
        <label class="cap">Target</label><input id="fs_target" style="width:78px">
        <label class="cap">Risk $</label><input id="fs_risk" style="width:78px">
        <label class="cap">or %</label><input id="fs_riskpct" style="width:52px">
        <button class="btn" id="fs_go">Size</button>
        <span id="fs_msg" class="cap"></span>
      </div>
      <div id="fs_result"></div>
    </div>

    <div id="activity" style="margin-top:18px"></div>`;
}

function setAccount(acct) {
  state.account = acct;
  document.querySelectorAll("[data-acct]").forEach((b) =>
    b.className = "btn" + (b.dataset.acct === acct ? "" : " ghost"));
  const ta = document.getElementById("ticketAcct");
  if (ta) ta.textContent = acct;
  renderPanels();
}

function acctBook() {
  const accs = (state.book && state.book.accounts) || [];
  return accs.find((a) => a.key === state.account) || null;
}

async function poll() {
  const [s, b, c] = await Promise.all([
    fetchJSONOrNull("/exec-status"),
    fetchJSONOrNull("/exec-book"),
    fetchJSONOrNull("/exec-commands"),
  ]);
  state.status = s || { online: false, configured: false };
  state.book = (b && b.book) || null;
  state.commands = (c && c.commands) || [];
  setAsof(state.status.online ? "execution online" : state.status.configured ? "execution offline" : "broker not configured");
  renderPanels();
  fillStageQty();                    // seasonal deep link: qty needs the book's NLV
  checkRiskAck();                    // unprotected-entry bounce -> secondary-approval prompt
}

function renderPanels() {
  set("modeBanner", renderModeBanner());
  set("connBar", renderConnBar());
  set("positions", renderPositions());
  // an open inline Modify must survive the 4s poll — don't redraw under the inputs
  if (!orderEdit.key) set("orders", renderOrders());
  set("closers", renderClosers());
  set("activity", renderActivity());
  syncMutationControls();
}
function set(id, html) { const el = document.getElementById(id); if (el) el.innerHTML = html; }

/* ---------- mode banner (live vs dry-run vs unknown) ---------- */
const BOOK_STALE_MS = 90000;   // ~2 agent book-push cycles; older than this the reported mode is stale
function epochMs(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return null;
  // The local agent historically emitted Python time.time() seconds, while
  // the broker's own fallback uses Date.now() milliseconds. Accept both so
  // display age and the fail-safe execution-mode freshness check agree.
  return n < 1e12 ? n * 1000 : n;
}
function bookAgeMs(book = state.book, now = Date.now()) {
  const at = epochMs(book && book.at);
  return at == null ? null : Math.max(0, now - at);
}
function bookFresh(book = state.book, now = Date.now()) {
  const age = bookAgeMs(book, now);
  return age != null && age <= BOOK_STALE_MS;
}
// Tri-state: "live" | "dry-run" | "unknown". Dry-run is only believed when a FRESH book
// explicitly reports it while the agent is online; a null/stale book or an offline agent
// means UNKNOWN, which is treated as live everywhere (fail dangerous, never fail open).
function deriveExecMode(book, status, now = Date.now()) {
  if (book && book.mode === "live") return "live";
  const online = !!(status && status.online);
  if (online && bookFresh(book, now) && book.mode === "dry-run") return "dry-run";
  return "unknown";
}
function execMode() { return deriveExecMode(state.book, state.status); }
const MUTATING_COMMANDS = new Set([
  "entry_bracket", "close_only", "flatten", "cancel", "modify", "trim_readd", "add_to_position",
  "exit_attach", "scheduled_option", "scheduled_option_cancel",
]);
function mutationBlocked(type) {
  return execMode() === "unknown" && (!type || MUTATING_COMMANDS.has(type));
}
function syncMutationControls() {
  const blocked = execMode() === "unknown";
  document.querySelectorAll("[data-mutation]").forEach((control) => {
    const staticBlocked = control.dataset.staticDisabled === "true";
    control.disabled = blocked || staticBlocked;
    control.setAttribute("aria-disabled", String(blocked || staticBlocked));
    if (blocked) control.title = "Disabled until the agent is online and a fresh book confirms execution mode";
    else if (staticBlocked) control.title = "Requires a visible price stop or scheduled time stop";
    else if (!staticBlocked && control.title && control.title.startsWith("Disabled until")) control.removeAttribute("title");
  });
}
function rejectUnknownMutation(msgId) {
  if (execMode() !== "unknown") return false;
  const msg = msgId ? document.getElementById(msgId) : null;
  if (msg) msg.textContent = "BLOCKED: execution mode unknown — reconnect the agent and wait for a fresh book";
  return true;
}
function renderModeBanner() {
  const mode = execMode();
  if (mode === "live") {
    return `<div class="card" style="border-color:#a8852f;background:rgba(255,193,77,.10);padding:9px 14px;font:700 13px inherit;color:#ffc14d">
      &#9888;&#65039; LIVE ARMED &mdash; orders ARE transmitted to IBKR.</div>`;
  }
  if (mode === "unknown") {
    return `<div class="card" style="border-color:#a8852f;background:rgba(255,193,77,.10);padding:9px 14px;font:700 13px inherit;color:#ffc14d">
      &#9888;&#65039; MODE UNKNOWN &mdash; assume LIVE. No fresh book confirms dry-run (book missing/stale or agent offline).
      Mutating controls are disabled until the agent is online and publishes a fresh book confirming execution mode.</div>`;
  }
  return `<div class="card" style="border-color:#2c8f63;background:rgba(61,219,143,.08);padding:9px 14px;font:700 13px inherit;color:#3ddb8f">
    &#9679; DRY-RUN MODE &mdash; actions are validated and previewed, but <u>nothing is transmitted</u> to IBKR.</div>`;
}

/* ---------- connection bar ---------- */
function dot(c) { return `<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${c};box-shadow:0 0 8px ${c}"></span>`; }
function renderConnBar() {
  const s = state.status || {};
  const tone = !s.configured ? "#9aa3b2" : s.online ? "#3ddb8f" : "#ff6b6b";
  const label = !s.configured ? "Broker not configured" : s.online ? "Execution online" : "Execution offline";
  const ab = acctBook();
  const nlv = ab && ab.nlv != null ? `NLV ${fmt.money(ab.nlv)}` : "";
  const ageMs = bookAgeMs();
  const age = ageMs != null ? `· book ${Math.round(ageMs / 1000)}s ago` : "";
  return `<div class="card" style="display:flex;align-items:center;gap:12px;padding:10px 14px">
    <span style="font:700 15px inherit;display:flex;align-items:center;gap:8px">${dot(tone)} ${label}</span>
    <span class="cap" style="margin-left:auto">${nlv} ${age}</span></div>`;
}

/* ---------- positions ---------- */
function pnlPct(p) {
  const cost = p.avg_cost != null && p.position ? Math.abs(p.avg_cost * p.position) : null;
  return cost && p.unrealized_pnl != null ? p.unrealized_pnl / cost : null;
}
const readdRows = new Map();   // account + contract -> persistent row toggle across 4s book polls
function positionKey(p) {
  return `${state.account}:${p.con_id || `${p.symbol}:${p.sec_type || ""}:${p.expiry || ""}`}`;
}
function fastActionQty(position, fraction) {
  return Math.floor(Math.abs(Number(position) || 0) * Number(fraction) + 0.5);
}
function positionIdentity(p) {
  const out = {
    symbol: p.symbol, sec_type: p.sec_type, expiry: p.expiry || null,
    expected_position: Number(p.position),
  };
  if (p.con_id) out.con_id = Number(p.con_id);
  if (p.currency) out.currency = String(p.currency).toUpperCase();
  return out;
}
function trimReaddPayload(p, fraction = 0.5) {
  return { ...positionIdentity(p), fraction, close_order_type: "MKT", readd: true, readd_tif: "DAY" };
}
function addPositionPayload(p, fraction) {
  return { ...positionIdentity(p), fraction, order_type: "MKT" };
}
function samePositionContract(p, o) {
  if (p.con_id && o.con_id) return Number(p.con_id) === Number(o.con_id);
  return String(p.symbol || "").toUpperCase() === String(o.symbol || "").toUpperCase()
    && (!p.sec_type || !o.sec_type || String(p.sec_type).toUpperCase() === String(o.sec_type).toUpperCase())
    && (!p.currency || !o.currency || String(p.currency).toUpperCase() === String(o.currency).toUpperCase())
    && (!p.expiry || !o.expiry || String(o.expiry).startsWith(String(p.expiry)));
}
function hasVisibleProtectiveExit(p) {
  const ab = acctBook();
  const close = Number(p.position) > 0 ? "SELL" : "BUY";
  return ((ab && ab.orders) || []).some((o) => {
    if (!samePositionContract(p, o)
        || String(o.action || "").toUpperCase() !== close) return false;
    const typ = String(o.order_type || "").toUpperCase();
    return (typ === "STP" && Number(o.aux) > 0)
      || (typ === "MKT" && Boolean(o.good_after));
  });
}
// ANY working closing-direction order (incl. plain LMT targets) — attach is only
// offered on positions with nothing working against them (the agent rejects the rest).
function hasAnyClosingOrder(p) {
  const ab = acctBook();
  const close = Number(p.position) > 0 ? "SELL" : "BUY";
  return ((ab && ab.orders) || []).some((o) =>
    samePositionContract(p, o) && String(o.action || "").toUpperCase() === close);
}
function renderPositions() {
  const ab = acctBook();
  const head = `<div style="font:700 14px inherit;margin:0 0 6px">Positions <span class="cap" style="display:inline;font-weight:400">${ab && ab.label ? "· " + esc(ab.label) : ""}</span></div>`;
  if (!ab) return head + panelNote("No book yet — the agent publishes positions when it's online and TWS is up.");
  if (ab.error) return head + panelNote(`${esc(ab.label)}: ${esc(ab.error)}.`);
  const pos = ab.positions || [];
  if (!pos.length) return head + panelNote("Flat — no open positions.");
  const rows = pos.map((p) => {
    const long = (p.position || 0) > 0;
    const pct = pnlPct(p);
    let sym = esc(p.symbol);
    if (p.sec_type === "OPT" && p.strike != null) {
      // e.g. XOM 08/14 105C — from the book's expiry_full/strike/right fields
      const ef = String(p.expiry_full || p.expiry || "");
      const d = ef.length >= 8 ? `${ef.slice(4, 6)}/${ef.slice(6, 8)}` : esc(ef);
      sym = `${esc(p.symbol)} <span class="cap" style="display:inline">${d} ${fmt.num(p.strike, p.strike % 1 ? 1 : 0)}${esc(p.right || "")}</span>`;
    } else if (p.sec_type === "FUT" && p.expiry) {
      sym = `${esc(p.symbol)} <span class="cap" style="display:inline">${esc(p.expiry)}</span>`;
    } else if (p.sec_type === "CASH") {
      sym = `${esc(p.symbol)}/${esc(p.currency || "USD")} <span class="cap" style="display:inline">FX</span>`;
    }
    // OPT rows: no Flatten/Trim — a symbol-scoped MKT close would tear one leg
    // out of a spread. Close via a closing combo ticket (later phase) or TWS.
    const hasProtection = hasVisibleProtectiveExit(p);
    const readdOn = readdRows.get(positionKey(p)) === true;
    const noProtection = ' disabled data-static-disabled="true" title="Requires a visible price stop or scheduled time stop"';
    const bare = p.sec_type !== "OPT" && !hasAnyClosingOrder(p);
    const protectBtn = bare
      ? `<button class="btn xs ghost" style="color:#ffc14d" onclick='execProtectTicket(${posJson(p)})' title="No working exits — prefill the attach-exits ticket (stop / target / time stop)">Protect&hellip;</button>`
      : "";
    const actions = p.sec_type === "OPT"
      ? '<span class="cap">combo — close via TWS</span>'
      : p.sec_type === "STK"
        ? `<button class="btn xs" data-mutation onclick='execFlatten(${posJson(p)},1)'>Flatten</button>
          <button class="btn xs ghost" data-mutation${readdOn && !hasProtection ? noProtection : ""} onclick='execTrim(${posJson(p)},0.25)'>Trim&frac14;</button>
          <button class="btn xs ghost" data-mutation${readdOn && !hasProtection ? noProtection : ""} onclick='execTrim(${posJson(p)},0.5)'>Trim&frac12;</button>
          <button class="btn xs ${readdOn ? "" : "ghost"}"${hasProtection ? "" : noProtection} onclick='execToggleReadd(${posJson(p)})'>Re-add ${readdOn ? "on" : "off"}</button>
          <button class="btn xs ghost" data-mutation${hasProtection ? "" : noProtection} onclick='execAddToPosition(${posJson(p)},0.5)'>Add&frac12;</button>
          <button class="btn xs ghost" data-mutation${hasProtection ? "" : noProtection} onclick='execAddToPosition(${posJson(p)},1)'>Add 1x</button>
          ${protectBtn}<button class="btn xs ghost" onclick='execSellTicket(${posJson(p)})' title="Prefill the close ticket: shares / LMT / outside RTH">Close&hellip;</button>`
        : `<button class="btn xs" data-mutation onclick='execFlatten(${posJson(p)},1)'>Flatten</button>
          <button class="btn xs ghost" data-mutation onclick='execFlatten(${posJson(p)},0.25)'>Trim&frac14;</button>
          <button class="btn xs ghost" data-mutation onclick='execFlatten(${posJson(p)},0.5)'>Trim&frac12;</button>
          ${protectBtn}<button class="btn xs ghost" onclick='execSellTicket(${posJson(p)})' title="Prefill the close ticket: shares / LMT / outside RTH">Close&hellip;</button>`;
    const priceDigits = p.sec_type === "CASH" ? 5 : 2;
    return `<tr>
      <td class="l" style="font-weight:600">${sym}</td>
      <td class="${long ? "pos" : "neg"}" style="font-weight:600">${fmt.num(p.position, 0)}</td>
      <td>${fmt.num(p.avg_cost, priceDigits)}</td>
      <td>${p.market_price != null ? fmt.num(p.market_price, priceDigits) : "&mdash;"}</td>
      <td>${p.market_value != null ? fmt.money(p.market_value) : "&mdash;"}</td>
      <td class="${clsSign(p.unrealized_pnl)}" style="font-weight:600">${p.unrealized_pnl != null ? fmt.money(p.unrealized_pnl) : "&mdash;"}</td>
      <td class="${clsSign(pct)}">${pct != null ? fmt.pct(pct, 1) : "&mdash;"}</td>
      <td class="l" style="white-space:nowrap">${actions}</td></tr>`;
  }).join("");
  return head + `<div class="tblwrap"><table class="tbl"><thead><tr>
    <th class="l">Symbol</th><th>Pos</th><th>Avg</th><th>Last</th><th>Mkt Val</th><th>uP&amp;L $</th><th>uP&amp;L %</th><th class="l">Actions</th>
    </tr></thead><tbody>${rows}</tbody></table></div>`;
}

/* ---------- open orders ---------- */
function orderPx(o) {
  const t = String(o.order_type || "").toUpperCase();
  if (t.startsWith("STP")) return o.aux != null ? o.aux : o.lmt;   // stop trigger, not the 0.00 lmt
  if (o.lmt) return o.lmt;
  if (o.aux) return o.aux;
  return null;                                                      // MKT / MOC / MOO — no price
}
function fmtOrderTime(s) {
  if (!s) return "";
  const m = String(s).match(/(\d{4})(\d{2})(\d{2})[ -](\d{2}):(\d{2})/);
  return m ? `${m[2]}/${m[3]} ${m[4]}:${m[5]}` : esc(String(s));    // MM/DD HH:MM
}
function orderKey(o) { return `${o.perm_id || 0}:${o.order_id || 0}`; }
function contractDisplay(x) {
  const sym = String((x && x.symbol) || "").toUpperCase();
  return x && x.sec_type === "CASH" ? `${sym}/${String(x.currency || "USD").toUpperCase()}` : sym;
}
function orderGroupKey(x) { return contractDisplay(x); }
function orderRow(o) {
  const buy = String(o.action).toUpperCase() === "BUY";
  const px = orderPx(o);
  if (orderEdit.key && orderEdit.key === orderKey(o) && (o.perm_id || o.order_id)) return orderEditRow(o);
  const canModify = !!(o.perm_id || o.order_id);   // no id yet: IBKR can't address the order to modify it
  return `<tr>
    <td class="l" style="font-weight:600">${esc(contractDisplay(o))}</td>
    <td class="l ${buy ? "pos" : "neg"}" style="font-weight:600">${esc(o.action)}</td>
    <td>${fmt.num(o.qty, 0)}</td>
    <td class="l">${esc(o.order_type)}</td>
    <td>${px != null ? fmt.num(px, o.sec_type === "CASH" ? 5 : 2) : "&mdash;"}</td>
    <td class="l">${esc(o.tif || "")}</td>
    <td class="l" style="color:#8c95a2">${fmtOrderTime(o.good_after) || "&mdash;"}</td>
    <td class="l" style="color:#8c95a2">${fmtOrderTime(o.good_till) || "&mdash;"}</td>
    <td class="l" style="color:#8c95a2">${esc(o.status || "")}</td>
    <td class="l" style="white-space:nowrap">${canModify ? `<button class="btn xs ghost" data-mutation onclick='execModifyStart("${orderKey(o)}")'>Modify</button> ` : ""}<button class="btn xs ghost" data-mutation onclick='execCancel(${o.perm_id || 0},${o.order_id || 0},"${esc(o.symbol)}")'>Cancel</button></td>
  </tr>`;
}
// Inline edit row: qty always; limit price on *LMT orders; stop trigger on STP*.
// While an edit is open the Open Orders panel is NOT re-rendered by the 4s poll
// (renderPanels skips it), so typed values survive until Save / Esc.
function orderEditRow(o) {
  const typ = String(o.order_type || "").toUpperCase();
  const hasLmt = typ.includes("LMT");
  const hasStp = typ.startsWith("STP");
  const pxCell =
    (hasStp ? `<span class="cap" style="display:inline">stop</span> <input id="me_stp" value="${o.aux != null ? esc(String(o.aux)) : ""}" style="width:70px"> ` : "") +
    (hasLmt ? `<span class="cap" style="display:inline">lmt</span> <input id="me_lmt" value="${o.lmt != null ? esc(String(o.lmt)) : ""}" style="width:70px">` : "") +
    (!hasStp && !hasLmt ? "&mdash;" : "");
  return `<tr style="background:rgba(77,163,255,.08)">
    <td class="l" style="font-weight:600">${esc(contractDisplay(o))}</td>
    <td class="l" style="font-weight:600">${esc(o.action)}</td>
    <td><input id="me_qty" value="${o.qty != null ? esc(String(o.qty)) : ""}" style="width:60px"></td>
    <td class="l">${esc(o.order_type)}</td>
    <td class="l" style="white-space:nowrap">${pxCell}</td>
    <td class="l">${esc(o.tif || "")}</td>
    <td class="l" style="color:#8c95a2">${fmtOrderTime(o.good_after) || "&mdash;"}</td>
    <td class="l" style="color:#8c95a2">${fmtOrderTime(o.good_till) || "&mdash;"}</td>
    <td class="l" style="color:#8c95a2">${esc(o.status || "")}</td>
    <td class="l" style="white-space:nowrap">
      <button class="btn xs" data-mutation onclick='execModifySave(${o.perm_id || 0},${o.order_id || 0},"${esc(o.symbol)}")'>Save</button>
      <button class="btn xs ghost" onclick='execModifyAbort()'>&times;</button></td>
  </tr>`;
}
const expandedTickers = new Set();   // Open Orders: which tickers are expanded (persists across 4s polls)
const orderEdit = { key: null, orig: null };   // inline Modify: row being edited + its pre-edit values
function toggleOrderGroup(sym) {
  const k = String(sym).toUpperCase();
  if (expandedTickers.has(k)) expandedTickers.delete(k); else expandedTickers.add(k);
  set("orders", renderOrders());
}
window.toggleOrderGroup = toggleOrderGroup;
function typeRank(o) {                                    // sort order within a ticker
  const t = String(o.order_type || "").toUpperCase();
  if (t.startsWith("STP")) return 1;                     // stops after limit legs
  if (t === "MKT" || t.startsWith("MO")) return 2;       // MKT / MOC / MOO time-stops last
  return 0;                                              // LMT (entry / target) first
}
function orderLegPreview(o) {
  const px = orderPx(o);
  return px != null ? `${esc(o.order_type || "")} ${fmt.num(px, o.sec_type === "CASH" ? 5 : 2)}` : esc(o.order_type || "");
}
function pnlSpan(label, v) {
  return v == null ? `${label} &mdash;` : `${label} <b class="${clsSign(v)}">${fmt.money(v)}</b>`;
}
// Best case (all profit-target LMTs fill) / worst case (all stops fill) for one bracket's
// exit legs against its own basis. MKT/MOC time-stops carry no price and are ignored.
function exitPnl(exits, entryPrice, sign, mult) {
  let best = null, worst = null;
  for (const o of exits) {
    const t = String(o.order_type || "").toUpperCase();
    const q = o.qty || 0;
    if (t.startsWith("STP")) {
      const px = o.aux != null ? o.aux : o.lmt;
      if (px != null) worst = (worst || 0) + (px - entryPrice) * mult * sign * q;
    } else if (t === "LMT" && o.lmt != null) {
      best = (best || 0) + (o.lmt - entryPrice) * mult * sign * q;
    }
  }
  return { best, worst };
}
// Best/worst per ticker. Entry (position-increasing) orders are NEVER counted as exits:
// each pending bracket is keyed by parent_id and priced off its OWN entry limit/side/qty;
// the on-position group uses only genuine exit legs (children of filled parents, or
// parent-less opposite-side orders) against the position's derived basis. entryPrice is
// derived from live PnL (avoids the futures averageCost-includes-multiplier ambiguity).
// Rows the data model can't classify make the group indeterminate -> {na:true} ("n/a").
function groupPnl(sym, legs, ab) {
  const ref = legs[0] || {};
  // CASH P&L settles in the quote currency; avoid labeling an approximate
  // conversion as USD in the stock/futures best/worst calculation.
  if (ref.sec_type === "CASH") return { best: null, worst: null, na: true };
  const root = String(ref.symbol || "").toUpperCase();
  const pos = (ab.positions || []).find((p) => p.position && samePositionContract(p, ref));
  const ids = new Set(legs.map((o) => o.order_id).filter(Boolean));
  const hasChild = (o) => !!o.order_id && legs.some((c) => (c.parent_id || 0) === o.order_id);
  const acc = { best: null, worst: null, na: false };
  const add = (part) => {
    if (part.best != null) acc.best = (acc.best || 0) + part.best;
    if (part.worst != null) acc.worst = (acc.worst || 0) + part.worst;
  };
  const bracket = (parent) => {                       // a working entry order + its own children
    const ep = orderPx(parent);
    const kids = parent.order_id ? legs.filter((c) => (c.parent_id || 0) === parent.order_id) : [];
    if (ep == null || !kids.length) { acc.na = true; return; }   // unpriced or naked entry: no defined best/worst
    const mult = parent.sec_type === "FUT" ? ((futSpec(root) || {}).multiplier || 1) : 1;
    const sign = String(parent.action).toUpperCase() === "BUY" ? 1 : -1;
    add(exitPnl(kids, ep, sign, mult));
  };
  if (pos) {
    const mult = pos.sec_type === "FUT" ? ((futSpec(root) || {}).multiplier || 1) : 1;
    const sign = pos.position > 0 ? 1 : -1;
    let entryPrice = null;
    if (pos.market_price != null && pos.unrealized_pnl != null) {
      entryPrice = pos.market_price - pos.unrealized_pnl / (pos.position * mult);
    } else if (pos.avg_cost != null) {
      entryPrice = pos.avg_cost / mult;
    }
    const posSide = sign > 0 ? "BUY" : "SELL";
    const posExits = [];
    for (const o of legs) {
      const pid = o.parent_id || 0;
      if (pid && ids.has(pid)) continue;              // child of a working entry: priced with its bracket
      if (pid) { posExits.push(o); continue; }        // child of a filled parent: exit leg of the position
      if (hasChild(o)) { bracket(o); continue; }      // working bracket parent (add-on entry): own basis
      if (String(o.action).toUpperCase() === posSide) { acc.na = true; continue; }  // bare position-increasing entry: no exits to price
      posExits.push(o);                               // parent-less opposite-side order: manual/OCA exit
    }
    if (posExits.length) {
      if (entryPrice == null) acc.na = true;          // exits exist but the basis is unknowable
      else add(exitPnl(posExits, entryPrice, sign, mult));
    }
  } else {
    for (const o of legs) {
      const pid = o.parent_id || 0;
      if (pid && ids.has(pid)) continue;              // child: priced with its parent via bracket()
      if (pid) { acc.na = true; continue; }           // orphan child (parent not in book): can't classify
      bracket(o);
    }
  }
  return acc;
}
function ordersSection(title, list, ab) {
  const h = `<div class="cap" style="font-weight:700;margin:10px 0 4px">${title} <span style="font-weight:400">&middot; ${list.length}</span></div>`;
  if (!list.length) return h + `<div class="cap" style="color:#8c95a2;margin-bottom:6px">(none)</div>`;
  const groups = new Map();
  for (const o of list) {
    const k = orderGroupKey(o);
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(o);
  }
  let body = "";
  for (const sym of [...groups.keys()].sort()) {
    const legs = groups.get(sym).slice()
      .sort((a, b) => typeRank(a) - typeRank(b) || (a.order_id || 0) - (b.order_id || 0));
    const open = expandedTickers.has(sym);
    const caret = open ? "&#9662;" : "&#9656;";          // triangle down / right
    const preview = open ? "" : legs.map(orderLegPreview).join(" &middot; ");
    const bw = groupPnl(sym, legs, ab);
    const bwFrag = bw.na
      ? ` &nbsp;&middot;&nbsp; best/worst n/a`
      : (bw.best != null || bw.worst != null)
        ? ` &nbsp;&middot;&nbsp; ${pnlSpan("best", bw.best)} &middot; ${pnlSpan("worst", bw.worst)}`
        : "";
    body += `<tr style="cursor:pointer;background:rgba(255,255,255,.03)" onclick="toggleOrderGroup('${esc(sym)}')">
      <td class="l" colspan="10" style="font-weight:600">${caret} ${esc(sym)}
        <span class="cap" style="font-weight:400;display:inline">&nbsp;(${legs.length})${preview ? " &nbsp;&middot;&nbsp; " + preview : ""}${bwFrag}</span></td></tr>`;
    if (open) body += legs.map(orderRow).join("");
  }
  return h + `<div class="tblwrap"><table class="tbl"><thead><tr>
    <th class="l">Symbol</th><th class="l">Side</th><th>Qty</th><th class="l">Type</th><th>Price</th><th class="l">TIF</th><th class="l">Start</th><th class="l">End</th><th class="l">Status</th><th class="l"></th>
    </tr></thead><tbody>${body}</tbody></table></div>`;
}
function renderOrders() {
  const ab = acctBook();
  const head = `<div style="font:700 14px inherit;margin:0 0 6px">Open Orders</div>`;
  if (!ab || ab.error) return head + panelNote("&mdash;");
  const ords = ab.orders || [];
  if (!ords.length) return head + panelNote("No working orders.");
  // Split by whether the symbol has an open position: exits-on-positions vs pending entries.
  const posSyms = new Set((ab.positions || []).map(orderGroupKey));
  const onPos = ords.filter((o) => posSyms.has(orderGroupKey(o)));
  const pending = ords.filter((o) => !posSyms.has(orderGroupKey(o)));
  return head
    + ordersSection("On open positions (working exits)", onPos, ab)
    + ordersSection("Pending entries — not filled yet", pending, ab);
}

/* ---------- scheduled closing orders (fire at today's close) ---------- */
// A working order is a scheduled closer when its goodAfterTime lands TODAY (ET)
// in the close window: TIME-exit MKT legs (15:59), OVS Friday EOD-DD stops
// (15:58). MOC orders count too. Earlier goodAfterTimes (e.g. a stop arming at
// 09:30) are protective legs, not scheduled closes. good_after is stamped in
// the TWS timezone (ET), so "today" is computed in America/New_York.
const CLOSE_WINDOW_START = "15:00";
function etToday() {
  return new Intl.DateTimeFormat("en-CA", { timeZone: "America/New_York" }).format(new Date()).replace(/-/g, "");
}
function etNowHM() {
  return new Intl.DateTimeFormat("en-GB", { timeZone: "America/New_York", hour: "2-digit", minute: "2-digit", hour12: false }).format(new Date());
}
function firesAtClose(o) {
  if (String(o.order_type || "").toUpperCase() === "MOC") return "MOC";
  const m = String(o.good_after || "").match(/(\d{8})[ -](\d{2}):(\d{2})/);
  if (!m || m[1] !== etToday()) return null;
  const hm = `${m[2]}:${m[3]}`;
  return hm >= CLOSE_WINDOW_START ? hm : null;
}
function untilFrag(hm) {
  if (hm === "MOC") return "at close";
  const mins = (+hm.slice(0, 2)) * 60 + (+hm.slice(3)) - ((+etNowHM().slice(0, 2)) * 60 + (+etNowHM().slice(3)));
  if (mins <= 0) return "due";
  return mins >= 60 ? `in ${Math.floor(mins / 60)}h ${mins % 60}m` : `in ${mins}m`;
}
function renderClosers() {
  const ab = acctBook();
  const head = `<div style="font:700 14px inherit;margin:0 0 6px">Scheduled closing orders <span class="cap" style="display:inline;font-weight:400">&mdash; fire at today's close</span></div>`;
  if (!ab || ab.error) return head + panelNote("&mdash;");
  const ords = ab.orders || [];
  const hits = ords.map((o) => ({ o, at: firesAtClose(o) })).filter((x) => x.at);
  if (!hits.length) return head + panelNote("None &mdash; nothing is scheduled to close at today's close.");
  const ids = new Set(ords.map((o) => o.order_id).filter(Boolean));
  const rows = hits
    .sort((a, b) => String(a.at).localeCompare(String(b.at)) || String(a.o.symbol).localeCompare(String(b.o.symbol)))
    .map(({ o, at }) => {
      const sym = String(o.symbol).toUpperCase();
      const buy = String(o.action).toUpperCase() === "BUY";
      const t = String(o.order_type || "").toUpperCase();
      const mult = o.sec_type === "FUT" ? ((futSpec(sym) || {}).multiplier || 1) : 1;
      const pos = (ab.positions || []).find((p) => p.position && samePositionContract(p, o));
      // MKT/MOC legs close at the market: value off the position's last price; STP off its trigger.
      const px = t.startsWith("STP") ? (o.aux != null ? o.aux : o.lmt) : (pos && pos.market_price != null ? pos.market_price : null);
      const val = px == null || !o.qty ? null
        : o.sec_type === "CASH"
          ? (String(o.currency || "USD").toUpperCase() === "USD" ? o.qty * px
            : sym === "USD" ? o.qty : null)
          : o.qty * px * mult;
      const conditional = (o.parent_id || 0) && ids.has(o.parent_id);   // parent entry still working
      return `<tr>
        <td class="l" style="font-weight:600">${esc(contractDisplay(o))}</td>
        <td class="l ${buy ? "pos" : "neg"}" style="font-weight:600">${esc(o.action)}</td>
        <td>${fmt.num(o.qty, 0)}</td>
        <td class="l">${esc(t)}${t.startsWith("STP") ? ` @ ${fmt.num(px, o.sec_type === "CASH" ? 5 : 2)}` : ""}</td>
        <td class="l">${at === "MOC" ? "MOC" : at + " ET"}</td>
        <td class="l" style="color:#8c95a2">${untilFrag(at)}</td>
        <td>${val != null ? fmt.money(val) : "&mdash;"}</td>
        <td class="l" style="color:${conditional ? "#ffc14d" : "#8c95a2"}">${conditional ? "only if entry fills first" : esc(o.status || "")}</td>
      </tr>`;
    }).join("");
  return head + `<div class="tblwrap"><table class="tbl"><thead><tr>
    <th class="l">Symbol</th><th class="l">Side</th><th>Qty</th><th class="l">Type</th><th class="l">Fires</th><th class="l"></th><th>Est. value</th><th class="l">Note</th>
    </tr></thead><tbody>${rows}</tbody></table></div>`;
}

function panelNote(html) { return `<div class="card" style="padding:12px 14px"><span class="cap">${html}</span></div>`; }
function posJson(p) {
  return JSON.stringify({
    symbol: p.symbol, sec_type: p.sec_type, expiry: p.expiry,
    con_id: p.con_id || 0, position: p.position, avg_cost: p.avg_cost,
    currency: p.currency || null,
  }).replace(/'/g, "&#39;");
}

/* ---------- row actions (live / dry-run / unknown commands) ---------- */
function isLive() { return execMode() !== "dry-run"; }   // unknown fails DANGEROUS: treated as live
function actionLead(verb) {
  const m = execMode();
  return m === "dry-run" ? `Dry-run: ${verb}`
    : m === "live" ? `⚠️ LIVE — ${verb}`
    : `⚠️ MODE UNKNOWN (may be LIVE) — ${verb}`;
}

function execFlatten(pos, fraction) {
  if (rejectUnknownMutation()) return;
  const pct = `${Math.round(Number(fraction) * 100)}%`;
  if (!confirm(`${actionLead("flatten")} ${pct} of ${pos.symbol} (${state.account})? Cancels its working orders first.`)) return;
  sendCommand("flatten", { ...positionIdentity(pos), fraction, order_type: "MKT" });
}
function execToggleReadd(pos) {
  if (!hasVisibleProtectiveExit(pos)) return;
  const key = positionKey(pos);
  readdRows.set(key, readdRows.get(key) !== true);
  set("positions", renderPositions());
  syncMutationControls();
}
function execTrim(pos, fraction = 0.5) {
  if (readdRows.get(positionKey(pos)) !== true) {
    execFlatten(pos, fraction);
    return;
  }
  if (rejectUnknownMutation()) return;
  if (!hasVisibleProtectiveExit(pos)) {
    alert("Re-add requires a visible price stop or scheduled time stop. Refresh the book or manage the position in TWS.");
    return;
  }
  const held = Math.abs(Number(pos.position));
  const qty = fastActionQty(pos.position, fraction);
  if (!(qty > 0 && qty < held)) { alert("This position is too small for a partial trim/re-add."); return; }
  const close = Number(pos.position) > 0 ? "SELL" : "BUY";
  const add = Number(pos.position) > 0 ? "BUY" : "SELL";
  const post = Number(pos.position) > 0 ? Number(pos.position) - qty : Number(pos.position) + qty;
  const avg = Number(pos.avg_cost);
  const summary = `${actionLead("trim + re-add")} ${close} ${qty} ${pos.symbol} MKT on ${state.account}; `
    + `expected post-trim position ${post}. Then stage ${add} ${qty} LMT at Avg ${fmt.num(avg, 2)} (DAY) `
    + "with the same stop, target, time-stop, and proportional OCA bracket?";
  if (!confirm(summary)) return;
  sendCommand("trim_readd", trimReaddPayload(pos, fraction));
}
function execAddToPosition(pos, fraction) {
  if (rejectUnknownMutation()) return;
  if (!hasVisibleProtectiveExit(pos)) {
    alert("Add requires a visible price stop or scheduled time stop. Refresh the book or manage the position in TWS.");
    return;
  }
  const qty = fastActionQty(pos.position, fraction);
  if (!(qty > 0)) { alert("Add quantity resolves to zero."); return; }
  const side = Number(pos.position) > 0 ? "BUY" : "SELL";
  const post = Number(pos.position) > 0 ? Number(pos.position) + qty : Number(pos.position) - qty;
  const summary = `${actionLead("add")} ${side} ${qty} ${pos.symbol} MKT on ${state.account}; `
    + `expected post-add position ${post}. Resize the same stop, target, time-stop, and proportional `
    + `OCA bracket to ${Math.abs(post)}?`;
  if (!confirm(summary)) return;
  sendCommand("add_to_position", addPositionPayload(pos, fraction));
}
function execCancel(permId, orderId, symbol) {
  if (rejectUnknownMutation()) return;
  if (permId || orderId) {
    // A real order/perm id in hand: cancel EXACTLY this order. The agent matches on
    // perm_id and falls through to order_id, so a nonzero order_id alone is enough — a
    // missing perm_id must NEVER widen to symbol scope (that would take out a position's
    // protective stop/target).
    if (!confirm(`${actionLead("cancel")} order ${orderId || permId} (${symbol}, ${state.account})?`)) return;
    sendCommand("cancel", { scope: "order", perm_id: permId || null, order_id: orderId || null });
    return;
  }
  // No id at all: the only available command is SYMBOL-scoped, which cancels EVERY working
  // order on the ticker, INCLUDING protective stops/targets of any open position. Require an
  // explicit symbol type-in and say so plainly — never send it silently.
  const typed = prompt(
    `${actionLead("cancel")} — this ${symbol} order has no id yet, so this will cancel ALL working ` +
    `orders for ${symbol} on ${state.account}, INCLUDING protective stops/targets of any open ` +
    `position.\n\nType the symbol (${symbol}) to proceed, or Cancel to abort:`);
  if (typed == null || typed.trim().toUpperCase() !== String(symbol).toUpperCase().trim()) return;
  sendCommand("cancel", { scope: "symbol", symbol });
}
window.execFlatten = execFlatten;
window.execToggleReadd = execToggleReadd;
window.execTrim = execTrim;
window.execAddToPosition = execAddToPosition;
window.execCancel = execCancel;

/* "Close…" on a position row: prefill the close-only ticket (shares / percent /
   LMT / outside RTH live there) instead of sending anything. */
function execSellTicket(pos) {
  const t = document.getElementById("cmdType");
  if (!t) return;
  t.value = "close_only";
  syncFields();                      // rebuilds fields (snapshots the old ticket first)
  const s = document.getElementById("f_symbol");
  if (s) s.value = pos.symbol;
  ticketDraft.f_symbol = pos.symbol; // survive later cmdType toggles
  ticketDraft.fl_position = { account: state.account, ...positionIdentity(pos) };
  updateReadout();
  const q = document.getElementById("fl_qty");
  if (q) q.focus();
  t.scrollIntoView({ behavior: "smooth", block: "center" });
}
window.execSellTicket = execSellTicket;

/* "Protect…" on a bare position row: prefill the attach-exits ticket. Sends
   nothing — stop/target/time stop are typed and confirmed like any ticket. */
function execProtectTicket(pos) {
  const t = document.getElementById("cmdType");
  if (!t) return;
  t.value = "exit_attach";
  syncFields();
  const s = document.getElementById("f_symbol");
  if (s) s.value = pos.symbol;
  ticketDraft.f_symbol = pos.symbol;
  ticketDraft.ea_position = { account: state.account, ...positionIdentity(pos) };
  updateReadout();
  const st = document.getElementById("f_stop");
  if (st) st.focus();
  t.scrollIntoView({ behavior: "smooth", block: "center" });
}
window.execProtectTicket = execProtectTicket;

/* ---------- inline order modify ---------- */
function findBookOrder(key) {
  const ab = acctBook();
  return ((ab && ab.orders) || []).find((o) => orderKey(o) === key) || null;
}
function execModifyStart(key) {
  if (rejectUnknownMutation()) return;
  const o = findBookOrder(key);
  if (!o) return;
  orderEdit.key = key;
  orderEdit.orig = { qty: o.qty, lmt: o.lmt, aux: o.aux };
  set("orders", renderOrders());
  const q = document.getElementById("me_qty");
  if (q) q.focus();
}
function execModifyAbort() {
  orderEdit.key = null; orderEdit.orig = null;
  set("orders", renderOrders());
}
// Only CHANGED fields are sent: the agent + executor treat absent fields as
// "leave alone", so an untouched stop trigger is never re-quoted from a possibly
// stale book value.
function execModifySave(permId, orderId, symbol) {
  if (rejectUnknownMutation()) return;
  const orig = orderEdit.orig || {};
  const read = (id) => {
    const e = document.getElementById(id);
    if (!e) return undefined;                       // field not rendered for this order type
    const v = String(e.value).trim();
    return v === "" ? null : Number(v);
  };
  const qty = read("me_qty"), lmt = read("me_lmt"), stp = read("me_stp");
  const bad = [qty, lmt, stp].some((v) => v !== undefined && v !== null && (!isFinite(v) || v <= 0));
  if (bad) { alert("qty / prices must be positive numbers"); return; }
  const payload = { symbol };
  if (permId) payload.perm_id = permId;
  if (orderId) payload.order_id = orderId;
  const changes = [];
  if (qty != null && qty !== Number(orig.qty)) { payload.new_qty = qty; changes.push(`qty ${orig.qty} -> ${qty}`); }
  if (lmt !== undefined && lmt != null && lmt !== Number(orig.lmt)) { payload.new_limit = lmt; changes.push(`lmt ${orig.lmt} -> ${lmt}`); }
  if (stp !== undefined && stp != null && stp !== Number(orig.aux)) { payload.new_stop = stp; changes.push(`stop ${orig.aux} -> ${stp}`); }
  if (!changes.length) { execModifyAbort(); return; }   // nothing changed: just close the editor
  if (!confirm(`${actionLead("modify")} order ${orderId || permId} (${symbol}, ${state.account})?\n${changes.join("\n")}`)) return;
  sendCommand("modify", payload);
  execModifyAbort();
}
window.execModifyStart = execModifyStart;
window.execModifyAbort = execModifyAbort;
window.execModifySave = execModifySave;

/* ---------- new-order ticket ---------- */
// Ticket fields ship EMPTY: the examples are placeholder hints, never submitted values,
// so a reflexive Send can't queue a stale live order (numOrNull reads "" as null, and
// bracketWarnings blocks empty qty/entry/stop). ticketDraft carries the last user entry
// across cmdType toggles so switching Type and back doesn't wipe a typed ticket.
const ticketDraft = {};
const TICKET_FIELDS = ["f_note", "f_symbol", "f_qty", "f_entry_type", "f_entry", "f_entry_cap", "f_stop", "f_target", "f_expiry", "f_timestop", "f_strategy", "f_so_frac", "f_so_target",
                       "f_currency", "f_futexch", "fl_qty", "fl_pct", "fl_limit", "so_symbol", "so_right",
                       "so_delta", "so_budget", "so_date", "so_time", "so_expiry_mode", "so_min_dte", "so_expiry"];
function snapshotTicket() {
  TICKET_FIELDS.forEach((id) => { const e = document.getElementById(id); if (e) ticketDraft[id] = e.value; });
}
function inp(id, ph, w) {
  const v = ticketDraft[id] != null ? esc(String(ticketDraft[id])) : "";
  return `<input id="${id}" value="${v}" placeholder="${esc(ph)}" style="width:${w || 90}px">`;
}
function syncFields() {
  snapshotTicket();   // preserve what's typed before the fields are rebuilt
  const t = document.getElementById("cmdType").value;
  const f = document.getElementById("cmdFields");
  if (t === "echo") {
    f.innerHTML = `<label class="cap">Note</label>${inp("f_note", "ping from site", 200)}`;
  } else if (t === "scheduled_option") {
    const right = String(ticketDraft.so_right || "P").toUpperCase();
    const mode = ticketDraft.so_expiry_mode || "min_dte";
    f.innerHTML = `<label class="cap">Underlying</label>${inp("so_symbol", "SPY", 80)}
      <label class="cap">Right</label><select id="so_right"><option value="P"${right === "P" ? " selected" : ""}>Put</option><option value="C"${right === "C" ? " selected" : ""}>Call</option></select>
      <label class="cap">Abs delta</label><input id="so_delta" value="${esc(ticketDraft.so_delta != null ? ticketDraft.so_delta : "0.15")}" style="width:68px">
      <label class="cap">Premium $</label><input id="so_budget" value="${esc(ticketDraft.so_budget != null ? ticketDraft.so_budget : "1000")}" style="width:82px">
      <label class="cap">Execute (ET)</label><input type="date" id="so_date" value="${esc(ticketDraft.so_date || "")}" style="width:140px">
      <input type="time" id="so_time" value="${esc(ticketDraft.so_time || "15:45")}" style="width:108px">
      <label class="cap">Expiry rule</label><select id="so_expiry_mode"><option value="min_dte"${mode === "min_dte" ? " selected" : ""}>minimum DTE</option><option value="specific"${mode === "specific" ? " selected" : ""}>specific expiry</option></select>
      <span id="so_min_dte_wrap"><label class="cap">Days</label><input id="so_min_dte" value="${esc(ticketDraft.so_min_dte != null ? ticketDraft.so_min_dte : "30")}" style="width:58px"></span>
      <span id="so_expiry_wrap"><label class="cap">Expiry</label><input type="date" id="so_expiry" value="${esc(ticketDraft.so_expiry || "")}" style="width:140px"></span>`;
    const expiryMode = document.getElementById("so_expiry_mode");
    if (expiryMode) expiryMode.addEventListener("change", () => {
      ticketDraft.so_expiry_mode = expiryMode.value;
      syncScheduledExpiryFields();
      updateReadout();
    });
    syncScheduledExpiryFields();
  } else if (t === "flatten" || t === "close_only") {
    // Close ticket: shares (takes precedence) or any percentage; MKT (RTH,
    // fill-gated) or LMT (resting close — required for outside-RTH). close_only
    // deliberately leaves every working order untouched.
    f.innerHTML = `<label class="cap">Symbol</label>${inp("f_symbol", "USO", 90)}
      <label class="cap">Shares</label>${inp("fl_qty", "blank = percent", 110)}
      <label class="cap">or Percent</label>${inp("fl_pct", "100", 65)}
      <label class="cap">Type</label><select id="fl_type"><option value="MKT">MKT</option><option value="LMT">LMT</option></select>
      <label class="cap">Limit</label>${inp("fl_limit", "", 80)}
      <label class="cap"><input type="checkbox" id="fl_rth" style="vertical-align:-2px"> Outside RTH</label>
      <label class="cap">TIF</label><select id="fl_tif"><option value="DAY">DAY</option><option value="GTC">GTC</option></select>`;
    const pct = document.getElementById("fl_pct");
    if (pct && ticketDraft.fl_pct == null) pct.value = "100";
    const rth = document.getElementById("fl_rth");
    if (rth) rth.addEventListener("change", () => {
      // outside-RTH is LMT-only at IBKR — flip the type so the ticket can't lie
      if (rth.checked) document.getElementById("fl_type").value = "LMT";
      updateReadout();
    });
    const typ = document.getElementById("fl_type");
    if (typ) typ.addEventListener("change", () => {
      if (typ.value !== "LMT") { const r = document.getElementById("fl_rth"); if (r) r.checked = false; }
      updateReadout();
    });
  } else if (t === "exit_attach") {
    // Attach exits: full-held-size OCA group (any subset of stop / target /
    // time stop) on a position with nothing working. Prefilled by Protect….
    f.innerHTML = `<label class="cap">Symbol</label>${inp("f_symbol", "USO", 90)}
      <label class="cap">Stop</label>${inp("f_stop", "", 80)}
      <label class="cap">Target</label>${inp("f_target", "", 80)}
      <label class="cap">Time stop</label><input type="date" id="f_timestop" value="${ticketDraft.f_timestop ? esc(ticketDraft.f_timestop) : ""}" style="width:140px">
      <label class="cap"><input type="checkbox" id="ea_rth" style="vertical-align:-2px"> Outside RTH</label>`;
    const rth = document.getElementById("ea_rth");
    if (rth) rth.addEventListener("change", updateReadout);
  } else {
    const entryType = String(ticketDraft.f_entry_type || "LMT").toUpperCase();
    f.innerHTML = `<label class="cap">Instr</label><select id="f_sectype"><option value="STK">Stock</option><option value="FUT">Future</option><option value="CASH">FX (USD pair)</option></select>
      <label class="cap">Symbol</label>${inp("f_symbol", "USO", 80)}
      <label class="cap">Side</label><select id="f_action"><option>BUY</option><option>SELL</option></select>
      <label class="cap">Qty</label>${inp("f_qty", "692", 70)}
      <label class="cap">Entry type</label><select id="f_entry_type">
        <option value="LMT"${entryType === "LMT" ? " selected" : ""}>Limit (LMT)</option>
        <option value="STP_LMT"${entryType === "STP_LMT" ? " selected" : ""}>Stop-limit (STP LMT)</option>
        <option value="MKT"${entryType === "MKT" ? " selected" : ""}>Market (MKT)</option>
        <option value="MOO"${entryType === "MOO" ? " selected" : ""}>Market-on-open (MOO)</option>
        <option value="MOC"${entryType === "MOC" ? " selected" : ""}>Market-on-close (MOC)</option>
      </select>
      <label class="cap" id="f_entry_label">Entry</label>${inp("f_entry", "104.80", 80)}
      <span id="f_entry_cap_wrap"><label class="cap">Limit cap</label>${inp("f_entry_cap", "", 80)}</span>
      <label class="cap">Stop</label>${inp("f_stop", "103.29", 80)}
      <label class="cap">Target</label>${inp("f_target", "123.21", 80)}
      <span id="f_futrow"></span>
      <span id="f_expiry_wrap"><label class="cap">Entry exp</label><input type="date" id="f_expiry" value="${ticketDraft.f_expiry ? esc(ticketDraft.f_expiry) : ""}" style="width:140px"></span>
      <label class="cap">Time stop</label><input type="date" id="f_timestop" value="${ticketDraft.f_timestop ? esc(ticketDraft.f_timestop) : ""}" style="width:140px">
      <label class="cap">Strategy</label>${inp("f_strategy", "blank = Discretionary", 150)}
      <label class="cap">Scale-out</label>${inp("f_so_frac", "frac e.g. .3333", 110)}
      ${inp("f_so_target", "near target", 90)}`;
    const st = document.getElementById("f_sectype");
    if (st) st.addEventListener("change", () => {
      if (val("f_sectype") === "FUT" && !futSpec(val("f_symbol"))) {
        const s = document.getElementById("f_symbol"); if (s) s.value = "ES";   // sensible FUT default
      } else if (val("f_sectype") === "CASH" && !/^[A-Za-z]{3}$/.test(val("f_symbol") || "")) {
        const s = document.getElementById("f_symbol"); if (s) s.value = "NZD";
      }
      frontState.manual = false;
      renderFutRow(); updateReadout(); scheduleFrontResolve();
    });
    const sym = document.getElementById("f_symbol");
    if (sym) sym.addEventListener("input", () => {
      frontState.manual = false;
      clearFutExp();                 // a previous symbol's month must NEVER survive a symbol change
      syncFutExchange();
      scheduleFrontResolve();
      updateReadout();
    });
    const entrySel = document.getElementById("f_entry_type");
    if (entrySel) entrySel.addEventListener("change", () => {
      ticketDraft.f_entry_type = entrySel.value;
      syncEntryTypeFields();
      updateReadout();
    });
    renderFutRow();
    syncEntryTypeFields();
  }
  updateReadout();
}

function syncScheduledExpiryFields() {
  const mode = val("so_expiry_mode") || ticketDraft.so_expiry_mode || "min_dte";
  const dte = document.getElementById("so_min_dte_wrap");
  const exact = document.getElementById("so_expiry_wrap");
  if (dte) dte.style.display = mode === "min_dte" ? "contents" : "none";
  if (exact) exact.style.display = mode === "specific" ? "contents" : "none";
}

function entryType() {
  return String(val("f_entry_type") || ticketDraft.f_entry_type || "LMT").toUpperCase();
}

function syncEntryTypeFields() {
  const typ = entryType();
  // STP LMT splits the entry into two prices: the trigger (f_entry) and the
  // limit cap (f_entry_cap), which is the WORST fill the order can take and so
  // the number every risk figure is computed against.
  const label = document.getElementById("f_entry_label");
  if (label) label.textContent = typ === "LMT" ? "Limit" : typ === "STP_LMT" ? "Trigger" : "Ref price";
  const cap = document.getElementById("f_entry_cap_wrap");
  if (cap) cap.style.display = typ === "STP_LMT" ? "contents" : "none";
  const wrap = document.getElementById("f_expiry_wrap");
  if (wrap) wrap.style.display = (typ === "LMT" || typ === "STP_LMT") ? "contents" : "none";
}

function futSpec(sym) { return FUT_SPECS[String(sym || "").toUpperCase().trim()] || null; }
function selectedFutExchange() {
  const spec = futSpec(val("f_symbol"));
  return String(val("f_futexch") || (spec && spec.exchange) || "").toUpperCase();
}
function syncFutExchange() {
  const el = document.getElementById("f_futexch");
  const spec = futSpec(val("f_symbol"));
  if (el && spec && spec.exchange) el.value = spec.exchange;
}
function renderFutRow() {
  const row = document.getElementById("f_futrow");
  if (!row) return;
  if (val("f_sectype") === "CASH") {
    const selected = String(ticketDraft.f_currency || "USD").toUpperCase();
    const currencies = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"];
    row.innerHTML = `<label class="cap">Quote</label><select id="f_currency">${
      currencies.map((c) => `<option value="${c}"${c === selected ? " selected" : ""}>${c}</option>`).join("")
    }</select><span class="cap" style="display:inline">IDEALPRO · qty is base-currency units · one leg must be USD · no hard notional cap · 5% NLV stop-risk guard (stopped entries)</span>`;
    const quote = document.getElementById("f_currency");
    if (quote) quote.addEventListener("change", updateReadout);
    return;
  }
  if (val("f_sectype") !== "FUT") { row.innerHTML = ""; return; }
  const spec = futSpec(val("f_symbol"));
  const selected = String(ticketDraft.f_futexch || (spec && spec.exchange) || "").toUpperCase();
  const exchanges = ["", "CME", "CBOT", "NYMEX", "COMEX"];
  row.innerHTML = `<label class="cap">Venue</label><select id="f_futexch">${exchanges.map((x) =>
      `<option value="${x}"${x === selected ? " selected" : ""}>${x || "choose"}</option>`).join("")}</select>
    <label class="cap">Contract</label><input id="f_futexp" placeholder="auto" style="width:78px">
    <span id="f_futnote" class="cap" style="display:inline;color:#ff6b6b"></span>
    <span id="f_futhint" class="cap" style="display:inline"></span>`;
  const venue = document.getElementById("f_futexch");
  if (venue) venue.addEventListener("change", () => {
    ticketDraft.f_futexch = venue.value;
    frontState.manual = false;
    clearFutExp(); scheduleFrontResolve(); updateReadout();
  });
  const exp = document.getElementById("f_futexp");
  if (exp) exp.addEventListener("input", () => { frontState.manual = true; setFutNote(""); });   // stop auto-fill once typed
}
function setFutNote(txt) { const n = document.getElementById("f_futnote"); if (n) n.textContent = txt || ""; }
// Blank the auto-filled month BEFORE a new resolve: if the resolve fails the field stays
// empty and the "enter the contract month" gate blocks submission (no stale month).
function clearFutExp() {
  const exp = document.getElementById("f_futexp");
  if (exp) { exp.value = ""; exp.placeholder = "resolving…"; }
  setFutNote("");
}

// Hard gate shared by the readout and sendTicket. Plain-text messages; [] = sendable.
// Empty inputs are null (never 0): a bracket REQUIRES qty/entry/stop; target may be
// null (= NO TARGET, entry + stop only) but 0/negative is rejected.
function fxPairWarnings(base, quote) {
  const b = String(base || "").toUpperCase().trim();
  const q = String(quote || "").toUpperCase().trim();
  const warns = [];
  if (!/^[A-Z]{3}$/.test(b)) warns.push("FX symbol must be a 3-letter base currency");
  if (!/^[A-Z]{3}$/.test(q) || b === q) warns.push("FX quote must be a different 3-letter currency");
  else if (b !== "USD" && q !== "USD") warns.push("FX entry supports USD pairs only initially");
  return warns;
}

function fxUsdMetrics(base, quote, qty, entry, stop) {
  const b = String(base || "").toUpperCase();
  const q = String(quote || "").toUpperCase();
  const dist = Math.abs(Number(entry) - Number(stop));
  if (q === "USD") return { notional: Number(qty) * Number(entry), risk: Number(qty) * dist };
  if (b === "USD") return { notional: Number(qty), risk: Number(qty) * dist / Number(entry) };
  return null;
}

function bracketWarnings() {
  const isFut = (val("f_sectype") === "FUT");
  const isFx = (val("f_sectype") === "CASH");
  const orderType = entryType();
  const sym = String(val("f_symbol") || "").toUpperCase().trim();
  const spec = isFut ? futSpec(sym) : null;
  const qty = numOrNull("f_qty"), entry = numOrNull("f_entry"), stop = numOrNull("f_stop"), target = numOrNull("f_target");
  const cap = orderType === "STP_LMT" ? numOrNull("f_entry_cap") : null;
  // A stop-limit fills anywhere between trigger and cap, so `worst` (the cap) is
  // what the ordering rules and the readout's risk numbers must use.
  const worst = cap != null ? cap : entry;
  const action = val("f_action");
  const warns = [];
  if (!["LMT", "STP_LMT", "MKT", "MOO", "MOC"].includes(orderType)) warns.push("entry type must be LMT, STP_LMT, MKT, MOO, or MOC");
  if (orderType === "MOC" && (isFut || isFx)) warns.push("MOC entry supports stocks only");
  if (orderType === "MOO" && isFx) warns.push("MOO entry does not support FX");
  if (orderType === "STP_LMT" && (isFut || isFx)) warns.push("STP_LMT entry supports stocks only");
  if (!sym) warns.push("symbol required");
  if (!(qty > 0) || qty !== Math.round(qty)) warns.push("qty must be a whole number > 0");
  if (!(entry > 0)) warns.push(orderType === "LMT" ? "limit price required"
    : orderType === "STP_LMT" ? "stop trigger required" : "reference price required");
  if (orderType === "STP_LMT" && !(cap > 0)) warns.push("limit cap required (the worst fill you will take)");
  if (cap > 0 && entry > 0) {
    if (action === "BUY" && !(entry < cap)) warns.push("BUY STP_LMT needs trigger < cap");
    if (action === "SELL" && !(cap < entry)) warns.push("SELL STP_LMT needs cap < trigger");
  }
  // Stop is OPTIONAL (2026-07-27): blank = UNPROTECTED entry, surfaced in amber by
  // the readout + confirm, and risk-gated agent-side (2×ATR vs 50 bps NLV → secondary
  // approval). An explicit 0/negative stop is still rejected.
  if (stop != null && !(stop > 0)) warns.push("stop must be > 0 (leave blank for NO STOP)");
  if (target != null && !(target > 0)) warns.push("target must be > 0 (leave blank for NO TARGET)");
  if (entry > 0 && stop > 0) {
    if (action === "BUY" && !(stop < entry && (target == null || worst < target)))
      warns.push(target == null ? "BUY needs stop < entry" : "BUY needs stop < entry and worst fill < target");
    if (action === "SELL" && !(entry < stop && (target == null || target < worst)))
      warns.push(target == null ? "SELL needs entry < stop" : "SELL needs entry < stop and target < worst fill");
  } else if (entry > 0 && stop == null && target > 0) {
    if (action === "BUY" && !(worst < target)) warns.push("BUY needs worst fill < target");
    if (action === "SELL" && !(target < worst)) warns.push("SELL needs target < worst fill");
  }
  // orderRef tag: a pipe or space would corrupt the downstream field split
  // that the execution report and Trade Log parse strategy out of.
  // Spaces are legal — the scan pipeline's own tags read "Oversold Low Volume".
  // Only the pipe breaks the field split the report and Trade Log parse.
  const strat = (val("f_strategy") || "").trim();
  if (strat && !/^[A-Za-z0-9 _.-]{1,32}$/.test(strat))
    warns.push("strategy tag: letters, digits, spaces, _ . and - only, max 32 chars");
  const soFrac = numOrNull("f_so_frac"), soTgt = numOrNull("f_so_target");
  if (soFrac != null || soTgt != null) {
    if (!(soFrac > 0 && soFrac < 1)) warns.push("scale-out fraction must be between 0 and 1");
    if (!(soTgt > 0)) warns.push("scale-out needs a near target");
    else if (action === "BUY" && !(worst < soTgt)) warns.push("BUY scale-out needs worst fill < near target");
    else if (action === "SELL" && !(soTgt < worst)) warns.push("SELL scale-out needs near target < worst fill");
    if (isFut || isFx) warns.push("scale-out is stock-only");
  }
  if (isFut && !selectedFutExchange()) warns.push("choose CME, CBOT, NYMEX, or COMEX");
  if (isFut && sym && !spec) warns.push("futures contract is not resolved by IBKR yet");
  if (isFut && !val("f_futexp")) warns.push("enter the contract month (e.g. 202609)");
  if (isFx) warns.push(...fxPairWarnings(sym, val("f_currency") || "USD"));
  // Date guards: an ISO YYYY-MM-DD string compares chronologically as text. A past
  // time-stop closes the position at market the instant the entry fills; a past entry
  // expiry ships an already-dead GTD order; expiry after the time-stop is contradictory.
  const todayISO = new Date().toLocaleDateString("en-CA");
  const ts = val("f_timestop"), ex = (orderType === "LMT" || orderType === "STP_LMT") ? val("f_expiry") : null;
  if (ts && ts < todayISO) warns.push("time stop date is in the past");
  if (ex && ex < todayISO) warns.push("entry expiry date is in the past");
  if (ts && ex && ex > ts) warns.push("entry expiry is after the time stop");
  return warns;
}
// Hard gate for the close/flatten ticket (mirrors the agent's checks; [] = sendable).
function flattenWarnings() {
  const sym = String(val("f_symbol") || "").toUpperCase().trim();
  const warns = [];
  if (!sym) warns.push("symbol required");
  const ab = acctBook();
  const identity = ticketDraft.fl_position;
  const pos = ((ab && ab.positions) || []).find((p) => p.position
    && String(p.symbol).toUpperCase() === sym
    && (!identity || identity.account !== state.account || !identity.con_id
      || Number(p.con_id) === Number(identity.con_id)));
  const held = pos ? Math.abs(pos.position) : null;
  const qn = numOrNull("fl_qty");
  if (qn != null) {
    if (!(qn > 0) || qn !== Math.round(qn)) warns.push("shares must be a whole number > 0");
    else if (held != null && qn > held) warns.push(`shares ${qn} exceeds held ${held}`);
  } else {
    const pct = numOrNull("fl_pct");
    if (!(pct > 0 && pct <= 100)) warns.push("percent must be above 0 and no more than 100");
  }
  const typ = val("fl_type") || "MKT";
  const rth = document.getElementById("fl_rth") && document.getElementById("fl_rth").checked;
  if (typ === "LMT") {
    const lim = numOrNull("fl_limit");
    if (!(lim > 0)) warns.push("LMT close needs a limit price");
  }
  if (rth && typ !== "LMT") warns.push("outside-RTH close must be LMT");
  return warns;
}
// The exit_attach ticket's position: symbol match, narrowed by the Protect…
// button's stashed con_id when it belongs to this account.
function attachPosition() {
  const sym = String(val("f_symbol") || "").toUpperCase().trim();
  if (!sym) return null;
  const ab = acctBook();
  const identity = ticketDraft.ea_position;
  return ((ab && ab.positions) || []).find((p) => p.position
    && String(p.symbol).toUpperCase() === sym
    && (!identity || identity.account !== state.account || !identity.con_id
      || Number(p.con_id) === Number(identity.con_id))) || null;
}
// Hard gate for the attach-exits ticket (mirrors the agent's checks; [] = sendable).
function attachWarnings() {
  const warns = [];
  const sym = String(val("f_symbol") || "").toUpperCase().trim();
  if (!sym) warns.push("symbol required");
  const stop = numOrNull("f_stop"), target = numOrNull("f_target");
  const ts = val("f_timestop");
  if (stop == null && target == null && !ts) warns.push("need at least one of stop / target / time stop");
  if (stop != null && !(stop > 0)) warns.push("stop must be > 0");
  if (target != null && !(target > 0)) warns.push("target must be > 0");
  const todayISO = new Date().toLocaleDateString("en-CA");
  if (ts && ts < todayISO) warns.push("time stop date is in the past");
  const pos = attachPosition();
  if (sym && !pos) warns.push(`no open ${sym} position in ${state.account}`);
  if (pos) {
    if (pos.sec_type === "OPT") warns.push("option positions not supported");
    const long = Number(pos.position) > 0;
    if (stop > 0 && target > 0) {
      if (long && !(stop < target)) warns.push("long needs stop < target");
      if (!long && !(target < stop)) warns.push("short needs target < stop");
    }
    const mark = Number(pos.market_price) > 0 ? Number(pos.market_price)
      : Number(pos.avg_cost) > 0 ? Number(pos.avg_cost) : 0;
    if (mark > 0) {
      if (stop > 0 && (long ? stop >= mark : stop <= mark))
        warns.push(`stop ${stop} is on the wrong side of the market (~${mark})`);
      if (target > 0 && (long ? target <= mark : target >= mark))
        warns.push(`target ${target} is on the wrong side of the market (~${mark})`);
    }
    const close = long ? "SELL" : "BUY";
    const mine = (((acctBook() || {}).orders) || []).filter((o) => samePositionContract(pos, o));
    if (mine.some((o) => String(o.action || "").toUpperCase() === close))
      warns.push("closing exit(s) already working — cancel or modify them instead");
    else if (mine.length)
      warns.push("same-direction entry/add order already working");
  }
  return warns;
}

function scheduledOptionWarnings() {
  const warns = [];
  const sym = String(val("so_symbol") || "").toUpperCase().trim();
  const right = String(val("so_right") || "").toUpperCase();
  const delta = numOrNull("so_delta"), budget = numOrNull("so_budget");
  const date = val("so_date"), time = val("so_time");
  const mode = val("so_expiry_mode") || "min_dte";
  if (state.account !== "primary") warns.push("scheduled options are enabled for Primary only");
  if (!/^[A-Z][A-Z0-9.]{0,9}$/.test(sym)) warns.push("enter a valid stock or ETF underlying");
  if (!['P', 'C'].includes(right)) warns.push("choose put or call");
  if (!(delta >= 0.01 && delta <= 0.50)) warns.push("absolute delta must be between 0.01 and 0.50");
  if (!(budget > 0)) warns.push("premium budget must be > 0");
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date || "")) warns.push("execution date required");
  if (!/^\d{2}:\d{2}$/.test(time || "")) warns.push("execution time required");
  if (date && time) {
    const when = new Date(`${date}T${time}:00`);
    if (!Number.isFinite(when.getTime()) || when.getTime() <= Date.now()) warns.push("execution time must be in the future");
  }
  if (mode === "min_dte") {
    const dte = numOrNull("so_min_dte");
    if (!(dte >= 0 && dte <= 730) || dte !== Math.round(dte)) warns.push("minimum DTE must be a whole number from 0 to 730");
  } else if (mode === "specific") {
    const expiry = val("so_expiry");
    if (!/^\d{4}-\d{2}-\d{2}$/.test(expiry || "")) warns.push("specific expiry required");
    else if (date && expiry < date) warns.push("specific expiry cannot precede the execution date");
  } else warns.push("choose an expiry rule");
  return warns;
}

function updateReadout() {
  const t = document.getElementById("cmdType").value;
  const el = document.getElementById("ticketReadout");
  if (!el) return;
  if (t === "scheduled_option") {
    const warns = scheduledOptionWarnings();
    if (warns.length) { el.innerHTML = `<span style="color:#ff6b6b">${warns.map(esc).join(" &middot; ")}</span>`; return; }
    const right = val("so_right") === "C" ? "call" : "put";
    const expiry = val("so_expiry_mode") === "specific"
      ? `expiry <b>${esc(val("so_expiry"))}</b>`
      : `first listed expiry with at least <b>${esc(val("so_min_dte"))} DTE</b>`;
    el.innerHTML = `<span style="color:#9aa3b2">At <b>${esc(val("so_date"))} ${esc(val("so_time"))} ET</b>, resolve ${esc(String(val("so_symbol")).toUpperCase())} ${right} nearest <b>${esc(val("so_delta"))} absolute delta</b>, ${expiry}, size from the live ask toward approximately <b>${fmt.money(numOrNull("so_budget"))}</b>, then send <b>MKT DAY</b>. <b style="color:#ffc14d">The fill can exceed the premium target.</b> Contract resolution expires after 5 minutes.</span>`;
  } else if (t === "entry_bracket") {
    const isFut = (val("f_sectype") === "FUT");
    const isFx = (val("f_sectype") === "CASH");
    const sym = String(val("f_symbol") || "").toUpperCase().trim();
    const currency = String(val("f_currency") || "USD").toUpperCase().trim();
    const spec = isFut ? futSpec(sym) : null;
    const mult = spec ? spec.multiplier : 1;
    const hint = document.getElementById("f_futhint");
    if (hint) hint.innerHTML = spec
      ? `${esc(spec.exchange)} · mult ${spec.multiplier} · tick ${spec.min_tick}`
      : (sym ? `<span style="color:#ffc14d">waiting for live IBKR contract details</span>` : "");
    const warns = bracketWarnings();
    if (warns.length) { el.innerHTML = `<span style="color:#ff6b6b">${warns.map(esc).join(" &middot; ")}</span>`; return; }
    const qty = numOrNull("f_qty"), entry = numOrNull("f_entry"), stop = numOrNull("f_stop"), target = numOrNull("f_target");
    const orderType = entryType();
    // Risk, R:R and notional read the WORST acceptable fill (the cap on a
    // stop-limit) so the readout shows the same numbers the agent gates on.
    const cap = orderType === "STP_LMT" ? numOrNull("f_entry_cap") : null;
    const worst = cap != null ? cap : entry;
    const dist = Math.abs(worst - stop);
    const fxMetrics = isFx ? fxUsdMetrics(sym, currency, qty, worst, stop) : null;
    const parts = [];
    if (orderType === "LMT") parts.push(`Entry <b>LMT @ ${entry}</b>`);
    else if (orderType === "STP_LMT") parts.push(`Entry <b>STP LMT trigger ${entry}</b> <span class="cap" style="display:inline">(fills up to ${cap}; risk shown at that worst fill)</span>`);
    else if (orderType === "MKT") parts.push(`Entry <b>MKT</b> <span class="cap" style="display:inline">(risk ref ${entry}; no price protection)</span>`);
    else if (orderType === "MOO") parts.push(`Entry <b>MOO</b> <span class="cap" style="display:inline">(opening auction; risk ref ${entry}; no price protection)</span>`);
    else parts.push(`Entry <b>MOC</b> <span class="cap" style="display:inline">(close auction; risk ref ${entry})</span>`);
    if (isFut && qty) parts.push(`<b>${qty} contract${qty === 1 ? "" : "s"}</b>`);
    if (isFx && qty) parts.push(`<b>${fmt.num(qty, 0)} ${esc(sym)} units</b> in ${esc(sym)}/${esc(currency)}`);
    if (stop == null) parts.push(`<b style="color:#ffc14d">NO STOP — UNPROTECTED</b> <span class="cap" style="display:inline">(risk gate at execution: 2&times;ATR% &times; notional vs 50 bps NLV)</span>`);
    else if (qty && dist) parts.push(`Risk <b>${fmt.money(fxMetrics ? fxMetrics.risk : qty * dist * mult)}</b>`);
    if (target == null) parts.push(`<b style="color:#ffc14d">NO TARGET</b>`);
    else if (stop != null && dist) parts.push(`R:R <b>${(Math.abs(target - worst) / dist).toFixed(2)}:1</b>`);
    if (qty && worst) parts.push(`Notional <b>${fmt.money(fxMetrics ? fxMetrics.notional : qty * worst * mult)}</b>`);
    const soF = numOrNull("f_so_frac"), soT = numOrNull("f_so_target");
    if (soF > 0 && soT > 0 && qty > 0) {
      const near = Math.round(qty * soF), far = qty - near;
      parts.push(near >= 1 && far >= 1
        ? `Scale-out <b>${near} @ ${soT}</b> + <b>${far}</b> runner <span class="cap" style="display:inline">(two brackets)</span>`
        : `<b style="color:#ffc14d">Scale-out ignored</b> <span class="cap" style="display:inline">(a tranche rounds below 1 share)</span>`);
    }
    const ts = val("f_timestop");
    if (ts) parts.push(`Time-exit <b>${ts}</b>`);
    const ex = (orderType === "LMT" || orderType === "STP_LMT") ? val("f_expiry") : null;
    parts.push(`TIF <b>${orderType === "MOO" ? "OPG" : ex ? "GTD " + ex : "DAY"}</b>`);
    el.innerHTML = `<span style="color:#9aa3b2">${parts.join(" &nbsp;·&nbsp; ")}</span>`;
  } else if (t === "exit_attach") {
    const warns = attachWarnings();
    if (warns.length) { el.innerHTML = `<span style="color:#ff6b6b">${warns.map(esc).join(" &middot; ")}</span>`; return; }
    const pos = attachPosition();
    if (!pos) { el.innerHTML = ""; return; }
    const held = Math.abs(pos.position);
    const long = Number(pos.position) > 0;
    const close = long ? "SELL" : "BUY";
    const stop = numOrNull("f_stop"), target = numOrNull("f_target");
    const ts = val("f_timestop");
    const parts = [`Position <b>${fmt.num(pos.position, 0)}</b> ${esc(String(pos.symbol).toUpperCase())}`];
    if (stop != null) parts.push(`Stop <b>${close} ${fmt.num(held, 0)} STP @ ${stop}</b>`);
    if (target != null) parts.push(`Target <b>${close} ${fmt.num(held, 0)} LMT @ ${target}</b>`);
    if (ts) parts.push(`Time <b>MKT ${ts} 15:59 ET</b>`);
    parts.push(`OCA group &middot; GTC`);
    el.innerHTML = `<span style="color:#9aa3b2">${parts.join(" &nbsp;&middot;&nbsp; ")}</span>`;
  } else if (t === "flatten" || t === "close_only") {
    const warns = flattenWarnings();
    if (warns.length) { el.innerHTML = `<span style="color:#ff6b6b">${warns.map(esc).join(" &middot; ")}</span>`; return; }
    const sym = String(val("f_symbol") || "").toUpperCase();
    const ab = acctBook();
    const identity = ticketDraft.fl_position;
    const pos = ((ab && ab.positions) || []).find((p) => p.position
      && String(p.symbol).toUpperCase() === sym
      && (!identity || identity.account !== state.account || !identity.con_id
        || Number(p.con_id) === Number(identity.con_id)));
    if (!pos) { el.innerHTML = `<span style="color:#ffc14d">No ${esc(sym || "?")} position in ${state.account}</span>`; return; }
    const held = Math.abs(pos.position);
    const qn = numOrNull("fl_qty");
    const pct = numOrNull("fl_pct");
    const n = qn != null ? qn : Math.round(held * Number(pct || 100) / 100);
    const rem = held - n;
    const close = pos.position > 0 ? "SELL" : "BUY";
    const typ = val("fl_type") || "MKT";
    const displaySymbol = pos.sec_type === "CASH" ? `${sym}/${pos.currency || "USD"}` : sym;
    const parts = [`Position <b>${fmt.num(pos.position, 0)}</b> ${esc(displaySymbol)}`,
                   `Close <b>${close} ${n}</b> ${typ === "LMT" ? `LMT @ <b>${numOrNull("fl_limit")}</b>` : "MKT"}`];
    if (document.getElementById("fl_rth") && document.getElementById("fl_rth").checked) parts.push(`<b style="color:#ffc14d">outside RTH</b>`);
    parts.push(`TIF <b>${val("fl_tif") || "DAY"}</b>`);
    if (t === "close_only") {
      parts.push(`<b style="color:#ffc14d">all working orders remain unchanged</b>`);
    } else if (rem > 0) parts.push(`working stop/target resize to <b>${rem}</b>`);
    else if (typ === "LMT") parts.push(`<b style="color:#ffc14d">all exits cancelled — unprotected while the close rests</b>`);
    el.innerHTML = `<span style="color:#9aa3b2">${parts.join(" &nbsp;·&nbsp; ")}</span>`;
  } else {
    el.innerHTML = "";
  }
}
function val(id) { const e = document.getElementById(id); return e ? e.value : undefined; }
// Empty/whitespace inputs are null, NEVER 0 (Number("") === 0 turned a cleared stop into a $0.00 stop).
function numOrNull(id) {
  const v = val(id);
  return v == null || String(v).trim() === "" ? null : Number(v);
}
function ticketPayload(t) {
  if (t === "echo") return { note: val("f_note") };
  if (t === "scheduled_option") {
    const mode = val("so_expiry_mode") || "min_dte";
    return {
      symbol: String(val("so_symbol") || "").toUpperCase().trim(),
      right: String(val("so_right") || "P").toUpperCase(),
      target_delta: numOrNull("so_delta"), delta_tolerance: 0.03,
      premium_budget: numOrNull("so_budget"), order_type: "MKT", tif: "DAY",
      execute_date: val("so_date"), execute_time: val("so_time"), timezone: "America/New_York",
      grace_minutes: 5, expiry_mode: mode,
      min_dte: mode === "min_dte" ? numOrNull("so_min_dte") : null,
      expiry: mode === "specific" ? val("so_expiry") : null,
    };
  }
  if (t === "flatten" || t === "close_only") {
    const qn = numOrNull("fl_qty");
    const typ = val("fl_type") || "MKT";
    const p = { symbol: val("f_symbol"), order_type: typ,
                tif: val("fl_tif") || "DAY",
                outside_rth: !!(document.getElementById("fl_rth") && document.getElementById("fl_rth").checked) };
    const identity = ticketDraft.fl_position;
    if (identity && identity.account === state.account
        && String(identity.symbol).toUpperCase() === String(p.symbol).toUpperCase()) {
      Object.assign(p, identity);
      delete p.account;
    }
    if (t === "close_only") {
      const ab = acctBook();
      const pos = ((ab && ab.positions) || []).find((x) => x.position
        && String(x.symbol).toUpperCase() === String(p.symbol).toUpperCase()
        && (!p.con_id || !x.con_id || Number(x.con_id) === Number(p.con_id)));
      if (pos) p.action = Number(pos.position) > 0 ? "SELL" : "BUY";
    }
    if (qn != null) p.qty = qn; else p.fraction = Number(numOrNull("fl_pct")) / 100;
    if (typ === "LMT") p.limit = numOrNull("fl_limit");
    return p;
  }
  if (t === "exit_attach") {
    const p = { symbol: val("f_symbol"), stop: numOrNull("f_stop"), target: numOrNull("f_target"),
                time_stop: val("f_timestop") || null,
                outside_rth: !!(document.getElementById("ea_rth") && document.getElementById("ea_rth").checked) };
    const pos = attachPosition();
    if (pos) {
      const identity = positionIdentity(pos);
      delete identity.expected_position;   // attach sizes to the LIVE held qty agent-side
      Object.assign(p, identity);
    }
    return p;
  }
  const sec_type = val("f_sectype") || "STK";
  const entry_type = entryType();
  const fut_expiry = sec_type === "FUT" ? String(val("f_futexp") || "").replace(/\D/g, "") : null;
  const currency = sec_type === "CASH" ? String(val("f_currency") || "USD").toUpperCase() : "USD";
  const spec = sec_type === "FUT" ? futSpec(val("f_symbol")) : null;
  return { symbol: val("f_symbol"), sec_type, currency, fut_expiry,
    exchange: sec_type === "FUT" ? selectedFutExchange() : null,
    fut_ib_symbol: spec ? (spec.ib_symbol || spec.symbol || val("f_symbol")) : null,
    fut_trading_class: spec ? (spec.trading_class || val("f_symbol")) : null,
    fut_multiplier: spec ? spec.multiplier : null,
    fut_min_tick: spec ? spec.min_tick : null,
    action: val("f_action"), quantity: numOrNull("f_qty"), entry_type,
    entry: numOrNull("f_entry"), stop: numOrNull("f_stop"), target: numOrNull("f_target"),
    entry_cap: entry_type === "STP_LMT" ? numOrNull("f_entry_cap") : null,
    strategy: (val("f_strategy") || "").trim() || null,
    // Two independent brackets when set: near = frac of qty targeting the near
    // price, far = the remainder taking `target` (which may be null).
    scaleout: (numOrNull("f_so_frac") != null || numOrNull("f_so_target") != null)
      ? { frac: numOrNull("f_so_frac"), target: numOrNull("f_so_target") } : null,
    time_stop: val("f_timestop") || null,
    expiry: (entry_type === "LMT" || entry_type === "STP_LMT") ? (val("f_expiry") || null) : null };
}
function sendTicket() {
  const t = document.getElementById("cmdType").value;
  const p = ticketPayload(t);
  const msg = document.getElementById("cmdMsg");
  if (mutationBlocked(t) && rejectUnknownMutation("cmdMsg")) return;
  if (t === "entry_bracket") {
    const warns = bracketWarnings();   // hard block: never submit while any warning is up
    if (warns.length) { if (msg) msg.textContent = "BLOCKED: " + warns.join("; "); return; }
  }
  if (t === "scheduled_option") {
    const warns = scheduledOptionWarnings();
    if (warns.length) { if (msg) msg.textContent = "BLOCKED: " + warns.join("; "); return; }
  }
  if (t === "exit_attach") {
    const warns = attachWarnings();
    if (warns.length) { if (msg) msg.textContent = "BLOCKED: " + warns.join("; "); return; }
  }
  if (t === "flatten" || t === "close_only") {
    const warns = flattenWarnings();
    if (warns.length) { if (msg) msg.textContent = "BLOCKED: " + warns.join("; "); return; }
  }
  if (t === "scheduled_option") {
    const expiry = p.expiry_mode === "specific" ? `expiry ${p.expiry}` : `minimum ${p.min_dte} DTE`;
    const right = p.right === "C" ? "call" : "put";
    if (!confirm(`${actionLead("schedule")} at ${p.execute_date} ${p.execute_time} ET, BUY approximately ${fmt.money(p.premium_budget)} of the ${p.symbol} ${right} nearest ${p.target_delta} absolute delta (${expiry}) via SMART MKT DAY on ${state.account}?\n\nThe quantity will be sized from the live ask at execution, but a market fill can exceed the premium target. The instruction expires after five minutes if it cannot run.`)) return;
    const ab = acctBook();
    const nlv = Number(ab && ab.nlv);
    if (!(nlv > 0)) {
      if (!confirm(`SECONDARY RISK APPROVAL\n\nCurrent NLV is unavailable. The scheduled option premium target is ${fmt.money(p.premium_budget)} and the eventual market fill can be higher. Really schedule it?`)) return;
      p.risk_ack = true;
    } else if (p.premium_budget > nlv * 0.05) {
      if (!confirm(`SECONDARY RISK APPROVAL\n\nThe scheduled option premium target is ${fmt.money(p.premium_budget)}, or ${(p.premium_budget / nlv * 100).toFixed(1)}% of NLV, and the eventual market fill can be higher. Really schedule it?`)) return;
      p.risk_ack = true;
    }
  } else if (t !== "echo") {
    const inst = p.sec_type === "FUT" ? `${p.symbol} FUT ${p.fut_expiry || p.expiry || ""}`.trim()
      : p.sec_type === "CASH" ? `${p.symbol}/${p.currency || "USD"} FX` : p.symbol;
    const closeUnit = p.sec_type === "CASH" ? ` ${p.symbol} units` : " sh";
    const stopTxt = p.stop == null ? "NO STOP — UNPROTECTED" : "stop " + p.stop;
    const entryDesc = p.entry_type === "LMT" ? `LMT @ ${p.entry}`
      : p.entry_type === "STP_LMT" ? `STP LMT trigger ${p.entry}, worst fill ${p.entry_cap}`
      : `${p.entry_type} (risk ref ${p.entry}; no price protection)`;
    const summary = t === "entry_bracket"
      ? `${p.action} ${p.quantity} ${inst} ${entryDesc} [${p.entry_type === "MOO" ? "OPG" : p.expiry ? "GTD " + p.expiry : "DAY"}] (${stopTxt}, ${p.target == null ? "NO TARGET" : "target " + p.target}${p.time_stop ? ", time " + p.time_stop : ""})`
      : t === "exit_attach"
        ? `attach exits to ${p.symbol} (${[p.stop != null ? "stop " + p.stop : "", p.target != null ? "target " + p.target : "", p.time_stop ? "time " + p.time_stop : ""].filter(Boolean).join(", ")}) — full held size, OCA GTC`
        : `close ${p.qty != null ? p.qty + closeUnit : Math.round((p.fraction || 1) * 100) + "%"} of ${p.symbol}${p.sec_type === "CASH" ? "/" + (p.currency || "USD") : ""} via ${p.order_type}` +
          `${p.order_type === "LMT" ? " @ " + p.limit : ""}${p.outside_rth ? " OUTSIDE RTH" : ""} (${p.tif})` +
          `${t === "close_only" ? " — ALL WORKING ORDERS STAY UNCHANGED" : p.qty != null || p.fraction < 1 ? " — remaining exits auto-resize" : ""}`;
    const verb = t === "entry_bracket" ? "place" : t === "exit_attach" ? "attach" : t === "close_only" ? "close only" : "flatten";
    if (!confirm(`${actionLead(verb)} ${summary} on ${state.account}?`)) return;
  }
  if (t === "entry_bracket" && p.sec_type === "FUT" && state.account === "primary" && p.stop != null) {
    const ab = acctBook();
    const nlv = Number(ab && ab.nlv);
    const risk = Number(p.quantity) * Math.abs(Number(p.entry) - Number(p.stop)) * Number(p.fut_multiplier || 0);
    if (!(nlv > 0)) {
      if (!confirm(`SECONDARY RISK APPROVAL\n\nThis Primary futures order is uncapped and current NLV is unavailable. Defined stop risk is about ${fmt.money(risk)}. Really continue?`)) return;
      p.risk_ack = true;
    } else if (risk > nlv * 0.05) {
      if (!confirm(`SECONDARY RISK APPROVAL\n\nThis Primary futures order has defined stop risk of about ${fmt.money(risk)}, or ${(risk / nlv * 100).toFixed(1)}% of NLV. There is no hard size cap. Really continue?`)) return;
      p.risk_ack = true;
    }
  }
  sendCommand(t, p, "cmdMsg").then((id) => {
    // Unprotected entry: remember the intent so a RISK_ACK_REQUIRED bounce can
    // re-prompt for secondary approval and resend with risk_ack.
    if (id && t === "entry_bracket" && (p.stop == null ||
        (p.sec_type === "FUT" && state.account === "primary" && !p.risk_ack))) {
      riskAckPending.set(id, { type: t, payload: p });
    }
  });
}

// Idempotency: the command id is minted HERE, once per confirmed intent, and reused on a
// retry of the SAME {type, account, payload} after a failed send (network error / non-2xx),
// so the server can dedup a double-submit. A payload change or a confirmed 2xx success
// mints a fresh id for the next intent.
const idemState = { id: null, key: null };
function commandId(type, account, payload) {
  const key = JSON.stringify({ type, account, payload });
  if (idemState.key !== key || !idemState.id) {
    idemState.id = crypto.randomUUID();
    idemState.key = key;
  }
  return idemState.id;
}
async function sendCommand(type, payload, msgId) {
  if (mutationBlocked(type) && rejectUnknownMutation(msgId)) return null;
  const msg = msgId ? document.getElementById(msgId) : null;
  if (msg) msg.textContent = "sending...";
  const id = commandId(type, state.account, payload);
  let sentId = null;
  try {
    const r = await fetch("/exec-command", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id, type, account: state.account, payload }),
    });
    const d = await r.json();
    const ok = r.ok && d && d.ok;
    if (ok) { idemState.id = null; idemState.key = null; sentId = d.id || id; }   // confirmed success: next intent gets a fresh id
    if (msg) msg.textContent = ok ? `queued ${(d.id || id).slice(0, 8)}` : `error: ${(d && d.error) || ("HTTP " + r.status)}`;
  } catch (e) {
    if (msg) msg.textContent = "error: " + e;                // id kept: an unchanged resend reuses it
  }
  setTimeout(poll, 600);
  return sentId;
}

/* ---------- unprotected-entry secondary approval (RISK_ACK) ----------
   A stop-less entry whose 2×ATR risk estimate exceeds 50 bps of NLV is BOUNCED
   by the executor with fill.needs_risk_ack (an approval gate, not a cap). When
   that rejection lands in the commands feed, re-prompt with the machine's own
   numbers and resend the identical payload + risk_ack:true on approval. */
const riskAckPending = new Map();   // command id -> {type, payload}
function checkRiskAck() {
  for (const c of state.commands || []) {
    if (!c || !c.id || !riskAckPending.has(c.id)) continue;
    const st = String(c.state || "");
    if (st === "rejected") {
      const intent = riskAckPending.get(c.id);
      riskAckPending.delete(c.id);
      const f = (c.result && c.result.fill) || {};
      if (!f.needs_risk_ack) continue;             // rejected for some other reason
      const p = intent.payload;
      const basis = p.stop == null ? "2xATR basis" : "defined stop basis";
      const detail = f.est_bps != null
        ? `The agent estimates risk ${fmt.money(f.est_risk)} = ${f.est_bps} bps of NLV (${basis}).`
        : `The agent could not compare risk with NLV (${basis}; NLV unavailable).`;
      const approve = confirm(
        `⚠️ SECONDARY RISK APPROVAL\n\n${detail}\n\n` +
        `Approve and resend ${p.action} ${p.quantity} ${p.symbol} @ ${p.entry}${p.stop == null ? " with NO STOP" : ` with stop ${p.stop}`} on ${state.account}?`);
      if (approve) sendCommand(intent.type, { ...p, risk_ack: true }, "cmdMsg");
      else { const m = document.getElementById("cmdMsg"); if (m) m.textContent = "secondary risk approval declined — nothing sent"; }
    } else if (st && st !== "pushed" && st !== "queued" && st !== "pending") {
      riskAckPending.delete(c.id);                 // resolved without needing an ack
    }
  }
}

/* ---------- futures sizing (read-only: risk -> contracts + notional) ---------- */
const sizeState = { id: null, timer: null };
async function sizeFutures() {
  const msg = document.getElementById("fs_msg");
  const symbol = String(val("fs_symbol") || "").toUpperCase().trim();
  if (!symbol) { msg.textContent = "symbol required"; return; }
  const entry = Number(val("fs_entry")), stop = Number(val("fs_stop"));
  if (!entry || !stop) { msg.textContent = "entry and stop required"; return; }
  const target = val("fs_target") ? Number(val("fs_target")) : null;
  const risk = val("fs_risk") ? Number(val("fs_risk")) : null;
  const risk_pct = val("fs_riskpct") ? Number(val("fs_riskpct")) : null;
  if (risk == null && risk_pct == null) { msg.textContent = "enter risk $ or %"; return; }
  msg.textContent = "sizing…";
  clearTimeout(sizeState.timer);
  try {
    const r = await fetch("/exec-futures-size", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbol, entry, stop, target, risk, risk_pct, account_key: state.account }),
    });
    const d = await r.json();
    if (!d.ok) { msg.textContent = "error: " + (d.error || ("HTTP " + r.status)); return; }
    sizeState.id = d.id;
    pollSize(0);
  } catch (e) { msg.textContent = "error: " + e; }
}
async function pollSize(n) {
  if (n > 30) { document.getElementById("fs_msg").textContent = "timed out — is the agent online?"; return; }
  const d = (await fetchJSONOrNull("/exec-futures-size")) || {};
  const q = d.query;
  if (q && q.id === sizeState.id && q.result) {
    document.getElementById("fs_msg").textContent = "";
    renderSize(q.result);
    return;
  }
  sizeState.timer = setTimeout(() => pollSize(n + 1), 1500);
}
function renderSize(data) {
  const el = document.getElementById("fs_result");
  if (!data || data.error) {
    el.innerHTML = `<div class="card" style="padding:10px 14px"><span class="neg">${esc((data && data.error) || "no result")}</span></div>`;
    return;
  }
  const notPct = data.notional_pct != null ? ` · ${fmt.num(data.notional_pct, 1)}% of acct (${fmt.num(data.leverage, 2)}x)` : "";
  const rr = data.rr != null ? ` · R:R ${fmt.num(data.rr, 2)}:1` : "";
  const note = data.note ? `<div class="cap" style="color:#ffc14d;margin-top:8px">${esc(data.note)}</div>` : "";
  el.innerHTML = `<div class="card" style="padding:12px 14px">
    <span style="font:700 16px inherit">${esc(data.symbol)} ${esc(data.action)} &mdash;
      <span style="color:#4da3ff">${data.contracts} contract${data.contracts === 1 ? "" : "s"}</span></span>
    <div class="kv" style="margin-top:10px">
      <div class="k">Risk / contract</div><div class="v">${fmt.money(data.risk_per_contract)} (${fmt.num(data.stop_ticks, 0)} ticks)</div>
      <div class="k">Total risk</div><div class="v">${fmt.money(data.total_risk)} <span class="cap" style="display:inline">/ budget ${fmt.money(data.risk_budget)}</span></div>
      <div class="k">Total notional</div><div class="v">${fmt.money(data.total_notional)}${notPct}</div>
      <div class="k">Multiplier</div><div class="v">${fmt.num(data.multiplier, 2)} <span class="cap" style="display:inline">· tick ${data.min_tick}${rr}</span></div>
    </div>${note}</div>`;
}

/* ---------- futures contract-month auto-resolve (read-only reqContractDetails) ---------- */
function scheduleFrontResolve() {
  if (val("f_sectype") !== "FUT") return;
  clearTimeout(frontState.timer);
  frontState.timer = setTimeout(resolveFront, 500);   // debounce while the symbol is typed
}
async function resolveFront() {
  const symbol = String(val("f_symbol") || "").toUpperCase().trim();
  const exchange = selectedFutExchange();
  const exp = document.getElementById("f_futexp");
  if (!symbol || !exchange) {
    if (exp) exp.placeholder = exchange ? "auto" : "choose venue";
    return;
  }
  try {
    const r = await fetch("/exec-futures-front", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbol, exchange }),
    });
    const d = await r.json();
    if (!d.ok) { frontResolveFailed(); return; }
    frontState.id = d.id;
    if (exp && !frontState.manual && !exp.value) exp.placeholder = "resolving…";
    pollFront(0);
  } catch (e) { frontResolveFailed(); }               // field stays BLANK; the submit gate blocks
}
function frontResolveFailed() {
  const exp = document.getElementById("f_futexp");
  if (exp) exp.placeholder = "202609";
  setFutNote("front-month resolve failed — enter the contract month");
}
async function pollFront(n) {
  const exp = document.getElementById("f_futexp");
  if (n > 20) { frontResolveFailed(); return; }
  const d = (await fetchJSONOrNull("/exec-futures-front")) || {};
  const q = d.query;
  if (q && q.id === frontState.id && q.result) {
    const res = q.result;
    const cur = String(val("f_symbol") || "").toUpperCase().trim();
    const curExchange = selectedFutExchange();
    if (exp && res.expiry && res.multiplier > 0 && res.min_tick > 0 && !res.error
        && !frontState.manual && cur === res.symbol && curExchange === res.exchange) {
      FUT_SPECS[cur] = { ...res, symbol: cur, ib_symbol: res.ib_symbol || cur };
      exp.value = res.expiry; exp.placeholder = "auto"; setFutNote(""); updateReadout();
    } else if (!frontState.manual) { frontResolveFailed(); }
    return;
  }
  setTimeout(() => pollFront(n + 1), 1200);
}

/* ---------- activity ---------- */
function stateBadge(state) {
  const map = { dry_run: ["#3ddb8f", "DRY-RUN OK"], rejected: ["#ff6b6b", "REJECTED"],
                executed: ["#ffc14d", "EXECUTED"], duplicate: ["#9aa3b2", "duplicate"],
                scheduled: ["#4da3ff", "SCHEDULED"], executing: ["#ffc14d", "EXECUTING"],
                cancelled: ["#9aa3b2", "CANCELLED"], expired: ["#ff6b6b", "EXPIRED"],
                unknown: ["#ff6b6b", "VERIFY IN TWS"],
                pushed: ["#9aa3b2", "pushed"], error: ["#ffc14d", "ERROR"] };
  const [c, t] = map[state] || ["#9aa3b2", state || ""];
  return `<span style="color:${c};font-weight:600">${esc(t)}</span>`;
}
function resultCell(c) {
  const res = c.result || {};
  const tone = res.ok === true ? "pos" : res.ok === false ? "neg" : "neu";
  let html = `<span class="${tone}">${esc(res.detail || c.state || "pending")}</span>`;
  const pv = res.preview || {};
  if (pv.legs && pv.legs.length) {
    html += `<div class="exec-legs">${pv.legs.map((l) => esc(l)).join("<br>")}` +
      (pv.summary ? `<br><span style="color:#c7ccd6;font-weight:600">${esc(pv.summary)}</span>` : "") + `</div>`;
  }
  if (res.fill) {
    const f = res.fill;
    html += `<div class="exec-legs" style="color:#ffc14d">filled ${esc(String(f.filled ?? "?"))} @ ${esc(String(f.avg_fill ?? "—"))} · #${esc(String(f.order_id ?? ""))} (${esc(String(f.status ?? ""))})</div>`;
  }
  if (c.type === "scheduled_option" && c.state === "scheduled") {
    html += `<div style="margin-top:6px"><button class="btn ghost" data-mutation onclick="cancelScheduledOption('${esc(c.id)}')">Cancel schedule</button></div>`;
  }
  return html;
}

function cancelScheduledOption(scheduleId) {
  if (!scheduleId || rejectUnknownMutation("cmdMsg")) return;
  if (!confirm(`${actionLead("cancel")} scheduled option instruction ${scheduleId.slice(0, 8)} on ${state.account}?`)) return;
  sendCommand("scheduled_option_cancel", { schedule_id: scheduleId }, "cmdMsg");
}
window.cancelScheduledOption = cancelScheduledOption;
function clockTime(ms) {
  if (!ms) return "";
  try { return new Date(ms).toLocaleTimeString("en-US", { hour12: false }); } catch (e) { return ""; }
}
function renderActivity() {
  const cmds = state.commands || [];
  const m = execMode();
  const trailLabel = m === "live" ? "LIVE" : m === "dry-run" ? "dry-run, places nothing" : "mode unknown — may be LIVE";
  if (!cmds.length) return "";
  const rows = cmds.map((c) => `<tr style="vertical-align:top">
      <td class="l" style="color:#8c95a2">${esc(clockTime(c.created_at))}</td>
      <td class="l" style="font-weight:600">${esc(c.type || "")}</td>
      <td class="l">${esc(c.account || "")}</td>
      <td class="l">${stateBadge(c.state)}</td>
      <td class="l">${resultCell(c)}</td></tr>`).join("");
  return `<div style="font:700 14px inherit;margin-bottom:6px">Activity / audit
      <span class="cap" style="display:inline;font-weight:400">· ${trailLabel} · last ${cmds.length}</span></div>
    <div class="tblwrap"><table class="tbl"><thead><tr>
    <th class="l">time</th><th class="l">type</th><th class="l">acct</th><th class="l">state</th><th class="l">result / order preview</th>
    </tr></thead><tbody>${rows}</tbody></table></div>`;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
