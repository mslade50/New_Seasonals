/* Read-only status projection. Producer state never proves a broker fill. */
export const INPUTS = {
  runtime: "ops/sleeve_runtime_status.json",
  event: "event_sleeve_last_actions.json",
  trend: "trend_sleeve_state.json",
  dial: "dial_sleeve_paper.json",
};

const HOUR = 3600000;
const object = value => value && typeof value === "object" && !Array.isArray(value);
const date = value => {
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return null;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) && new Date(parsed).toISOString().slice(0, 10) === value ? value : null;
};
// Defense in depth: the publisher scrubs broker account ids, but this runtime
// file is the only input the browser sees verbatim, so scrub again here.
const ACCOUNT_ID = /(?<![A-Za-z0-9])(?:DU|DF|U|F)\d{5,}(?!\d)/g;
export const text = value => typeof value === "string" ? value.replace(ACCOUNT_ID, "[account]").slice(0, 300) : "";
const count = value => object(value) ? Object.keys(value).length : null;

export async function readInputs(bucket) {
  const entries = await Promise.all(Object.entries(INPUTS).map(async ([name, key]) => {
    try {
      const file = await bucket.get(key);
      if (!file) return [name, { error: "Not yet reported" }];
      if (file.size > 262144) return [name, { error: "Report exceeds the size limit" }];
      const data = await file.json();
      if (!object(data)) return [name, { error: "Invalid report" }];
      return [name, { data, uploaded: file.uploaded.toISOString() }];
    } catch {
      return [name, { error: "Report unavailable" }];
    }
  }));
  return Object.fromEntries(entries);
}

function source(input, days, now) {
  const at = input && Date.parse(input.uploaded);
  const valid = Number.isFinite(at) && at <= now + 5 * 60000;
  return {
    updated_at: valid ? input.uploaded : null,
    health: !input || input.error || !valid ? "unavailable" : now - at > days * 24 * HOUR ? "aging" : "reported",
  };
}

function scheduled(task) { return task && ["Ready", "Running"].includes(task.state); }
function failed(task) {
  return task?.state === "Ready" && Number.isFinite(Date.parse(task.last_run_at))
    && Number.isInteger(task.last_result) && task.last_result !== 0;
}

function checkReportDate(row, days, now) {
  const at = Date.parse(row.report_date);
  // An old report re-uploaded today is still old. Allow the current ET date.
  if (!Number.isFinite(at) || at > now) row.health = "unavailable";
  else if (row.health === "reported" && now - at > days * 24 * HOUR) row.health = "aging";
}

export function buildStatus(inputs, now = Date.now()) {
  const raw = inputs.runtime?.data;
  const checked = Date.parse(raw?.checked_at);
  const runtimeValid = raw?.schema === "sleeve-runtime.v2" && object(raw.tasks)
    && ["event", "trend", "chain", "legend", "legend_verify"].every(key =>
      object(raw.tasks[key]) && ["Ready", "Running", "Disabled", "Missing", "Unknown"].includes(raw.tasks[key].state))
    && ["event_enabled", "trend_moo_enabled", "legend_enabled"].every(key => typeof raw[key] === "boolean")
    && Number.isFinite(checked) && checked <= now + 5 * 60000;
  // This report is refreshed by a separate read-only Windows collector.
  const runtimeFresh = runtimeValid && now - checked <= 2 * HOUR;
  const runtime = runtimeValid ? raw : null;
  const tasks = runtime?.tasks || {};
  const runtimeNote = runtimeFresh ? "Machine check current" : runtimeValid ? "Machine check needs refresh" : "Machine check unavailable";
  const base = (id, name, kind, next, href) => ({ id, name, kind, next, href,
    deployment: "Not verified", checked_at: runtimeValid ? raw.checked_at : null,
    runtime_health: runtimeFresh ? "reported" : "unavailable", runtime_note: runtimeNote });

  const event = { ...base("event", "Calendar Event", "Algorithmic", "Review the next event's order and fill receipts.", "events.html"),
    ...source(inputs.event, 4, now), summary: "No valid producer report", report_date: null };
  if (runtimeFresh) {
    event.deployment = scheduled(tasks.event) && raw.event_enabled === true ? "Scheduled" : "Not activated";
    if (failed(tasks.event)) event.deployment = "Last runner failed";
  }
  const es = inputs.event?.data;
  if (date(es?.asof) && Array.isArray(es.rows) && count(es.positions) !== null) {
    event.report_date = es.asof;
    event.summary = `${es.rows.length} actions staged · ${count(es.positions)} intended positions`;
  } else event.health = "unavailable";
  checkReportDate(event, 4, now);

  const trend = { ...base("trend", "Monthly Trend", "Algorithmic", "Complete the pre-open auction runner rollout; reconcile intended versus tagged inventory.", "execution.html"),
    ...source(inputs.trend, 40, now), summary: "No valid producer report", report_date: null };
  if (runtimeFresh) {
    trend.deployment = raw.trend_moo_enabled === true
      ? scheduled(tasks.trend) ? "Opening-auction runner" : "Runner missing or disabled"
      : scheduled(tasks.chain) ? "Legacy 9:31 execution" : "No active route verified";
    if (raw.trend_moo_enabled === true && failed(tasks.trend)) trend.deployment = "Last runner failed";
    if (raw.trend_moo_enabled === false && failed(tasks.chain)) trend.deployment = "Last legacy chain failed";
  }
  const ts = inputs.trend?.data;
  if (date(ts?.asof) && count(ts.positions) !== null) {
    trend.report_date = ts.asof;
    const gate = ts.fragility_gate;
    trend.summary = gate?.state === "CASH" ? `Cash per rule · ${text(gate.reason)}` : `${count(ts.positions)} intended positions at last rebalance`;
    if (trend.deployment === "Opening-auction runner") trend.next = "Verify the next rebalance against strategy-tagged broker fills.";
  } else trend.health = "unavailable";
  checkReportDate(trend, 40, now);

  const legend = buildLegend(base, runtimeFresh ? raw : null, tasks, now);
  const breakout = buildBreakout(base, runtimeFresh ? raw : null, now);

  const dial = { ...base("dial", "Dial-gated SPY", "Paper only", "Continue the registered paper track. This is a long-SPY allocation study.", "risk.html"),
    ...source(inputs.dial, 4, now), deployment: "Paper only", summary: "No valid paper report", report_date: null,
    checked_at: null, runtime_health: "not_applicable", runtime_note: "No order-staging path" };
  const ds = inputs.dial?.data;
  if (date(ds?.last_evaluated) && ["FLAT", "LONG"].includes(ds.position) && Array.isArray(ds.transitions)) {
    dial.report_date = ds.last_evaluated;
    dial.summary = `${ds.position === "FLAT" ? "Flat" : "Long"} in paper track · ${ds.transitions.length} recorded transitions`;
  } else dial.health = "unavailable";
  checkReportDate(dial, 4, now);

  const hedge = { ...base("hedge", "Dial-based hedge", "Research pending", "Finish the D10 hedge specification and futures execution validation. No automatic hedge is active.", "execution.html#hedge"),
    deployment: "Manual analysis only", summary: "Hedge research replays exist. The current panel sizes scenarios from holdings beta; the dial-driven execution protocol is unfinished.",
    health: "reference", report_date: "2026-09-06", updated_at: null, checked_at: null,
    runtime_health: "not_applicable", runtime_note: "Scope review · 6 Sep 2026 · D10 remains open" };

  return { schema: "sleeve-status.v1", fetched_at: new Date(now).toISOString(),
    runtime_checked_at: runtimeValid ? raw.checked_at : null,
    sleeves: [event, trend, legend, breakout, dial, hedge] };
}

const et = value => {
  const at = Date.parse(value);
  return Number.isFinite(at) ? new Date(at).toLocaleString("en-US", { timeZone: "America/New_York", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }) + " ET" : "not reported";
};
export function nyDate(now) {
  const parts = Object.fromEntries(new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hourCycle: "h23" })
    .formatToParts(new Date(now)).map(p => [p.type, p.value]));
  return { date: `${parts.year}-${parts.month}-${parts.day}`, minutes: Number(parts.hour) * 60 + Number(parts.minute) };
}
const taskLine = (label, task) => !object(task) ? `${label}: not reported`
  : task.state === "Missing" ? `${label}: task missing`
  : `${label}: last ran ${et(task.last_run_at)} · result ${Number.isInteger(task.last_result) ? task.last_result : "n/a"} · next ${et(task.next_run_at)}`;

function buildLegend(base, raw, tasks, now) {
  const row = { ...base("legend", "Legend EMA", "Algorithmic", "Check the next 09:29 ET session and its 10:40 verify.", "execution.html"),
    health: "unavailable", updated_at: null, report_date: null, summary: "No session evidence connected to this view", details: [] };
  if (!raw) return row;
  const task = tasks.legend;
  row.updated_at = raw.checked_at;
  row.deployment = task?.state === "Missing" ? "Runner task missing" : task?.state === "Disabled" ? "Runner task disabled"
    : failed(task) ? `Last run failed (code ${task.last_result})`
    : raw.legend_enabled === true ? "Armed for live orders" : "Dry run (activation flag absent)";
  if (failed(tasks.legend_verify) && !failed(task)) row.deployment = `Verify failed (code ${tasks.legend_verify.last_result})`;
  row.attention = !scheduled(task) || failed(task) || failed(tasks.legend_verify);
  row.details = [taskLine("Runner", task), taskLine("Verify", tasks.legend_verify)];
  const s = raw.legend;
  if (!object(s)) row.summary = "unavailable: not reported";
  else if (s.available !== true) row.summary = `unavailable: ${text(s.reason) || "unknown"}`;
  else if (!date(s.session_date) || !Array.isArray(s.trades) || !Array.isArray(s.skips)) row.summary = "unavailable: invalid session report";
  else {
    row.report_date = s.session_date;
    row.health = "reported";
    const trades = s.trades.filter(object).map(t => `${text(t.symbol)} ${t.side === "SELL" ? "short" : "long"} ${Number.isInteger(t.qty) ? t.qty : "?"} sh, ${text(t.outcome) || "outcome unknown"}`);
    const skips = s.skips.filter(object).map(k => `${text(k.symbol)} ${String(text(k.status)).toLowerCase().replace(/_/g, " ")}${k.note ? ` (${text(k.note)})` : ""}`);
    row.summary = `last: ${s.session_date} ${trades.length ? trades.join("; ") : "no entry"}${skips.length ? ` · skipped ${skips.join("; ")}` : ""}`;
    if (s.result_error) row.details.push(`Run flagged ${text(s.result_error)}`);
    if (Array.isArray(s.notes) && s.notes.length) row.details.push(`Partial evidence: ${s.notes.map(text).join(", ")}`);
    checkReportDate(row, 4, now);
  }
  return row;
}

const TERMINAL = new Set(["SESSION_COMPLETE", "STOPPED", "STOPPED_BY_REQUEST", "FAILED", "HALTED"]);
export function breakoutSession(s, checkedAt, now) {
  if (s === null || s === undefined) return { available: false, reason: "no session found" };
  if (!object(s)) return { available: false, reason: "invalid report" };
  if (s.available !== true) return { available: false, reason: text(s.reason) || "unknown", run_dir: text(s.run_dir) };
  if (!date(s.session)) return { available: false, reason: "invalid session date", run_dir: text(s.run_dir) };
  const today = nyDate(now);
  const phase = text(s.phase) || "UNKNOWN";
  const terminal = TERMINAL.has(phase) || Boolean(s.finished_at);
  // Launched before 09:00 ET (often the prior evening), exits at 16:01 ET on the session day.
  const running = !terminal && (s.session > today.date || (s.session === today.date && today.minutes <= 16 * 60 + 5));
  const beat = Date.parse(s.heartbeat_at);
  // The collector samples every few minutes, so age is measured at the machine check, not at view time.
  const age = Number.isFinite(beat) ? Math.max(0, Math.round((Date.parse(checkedAt) - beat) / 1000)) : null;
  const skipped = object(s.prior_range) ? Object.entries(s.prior_range).filter(([, r]) => object(r) && r.skip === true).map(([m]) => text(m)) : [];
  return { available: true, mode: s.mode === "live" ? "live" : "shadow", session: s.session, today: s.session === today.date,
    phase, running, heartbeat_age_s: age, stale: running && (age === null || age > 120), skipped,
    events: Number.isInteger(s.events) ? s.events : null, last_error: text(s.last_error) || null, run_dir: text(s.run_dir) };
}

function describe(label, v) {
  if (!v.available) return `${label}: unavailable: ${v.reason}`;
  const beat = !v.running ? "not running" : v.heartbeat_age_s === null ? "no heartbeat" : `heartbeat ${v.heartbeat_age_s}s old at check`;
  return `${label} ${v.session} · ${v.phase} · ${v.stale ? `STALE ${beat}` : beat}`
    + (v.skipped.length ? ` · prior-range skip: ${v.skipped.join(", ")}` : "")
    + (v.last_error && !v.running ? ` · ${v.last_error}` : "");
}

function buildBreakout(base, raw, now) {
  const row = { ...base("open_breakout", "Open Breakout", "Futures pilot", "Stage the next session's shadow and live runs per the runbook.", "execution.html"),
    health: "unavailable", updated_at: null, report_date: null, summary: "No session evidence connected to this view", details: [], sessions: null };
  if (!raw) return row;
  row.updated_at = raw.checked_at;
  const ob = raw.open_breakout;
  if (!object(ob) || ob.available !== true) {
    row.deployment = "Not verified";
    row.summary = `unavailable: ${object(ob) ? text(ob.reason) || "unknown" : "not reported"}`;
    return row;
  }
  const shadow = breakoutSession(ob.shadow, raw.checked_at, now);
  const live = breakoutSession(ob.live, raw.checked_at, now);
  row.sessions = { shadow, live };
  row.health = "reported";
  row.details = [describe("Live", live), describe("Shadow", shadow)];
  const todays = [live, shadow].filter(v => v.available && v.today);
  const dates = [live, shadow].filter(v => v.available).map(v => v.session).sort();
  row.report_date = dates.length ? dates[dates.length - 1] : null;
  const today = nyDate(now).date;
  const unreadableToday = [live, shadow].some(v => !v.available && v.run_dir?.startsWith(today));
  if ([live, shadow].some(v => v.available && v.stale)) row.deployment = "Heartbeat stale";
  else if (!todays.length && unreadableToday) row.deployment = "Session today · state unavailable";
  else if (!todays.length) row.deployment = "No session today";
  else row.deployment = todays.map(v => `${v.mode === "live" ? "Live" : "Shadow"} ${v.phase}`).join(" · ");
  row.summary = todays.length ? `Session ${todays[0].session}` : `No session today · last ${row.report_date || "none recorded"}`;
  row.attention = unreadableToday || [live, shadow].some(v => v.available && (v.stale || (v.today && !v.running && v.phase !== "SESSION_COMPLETE")));
  if ([live, shadow].some(v => v.available && v.stale)) row.health = "aging";
  return row;
}
