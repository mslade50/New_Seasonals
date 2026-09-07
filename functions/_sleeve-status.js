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
const text = value => typeof value === "string" ? value.slice(0, 300) : "";
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
  const runtimeValid = raw?.schema === "sleeve-runtime.v1" && object(raw.tasks)
    && ["event", "trend", "chain", "legend_signals", "legend_session", "legend_watchdog"].every(key =>
      object(raw.tasks[key]) && ["Ready", "Running", "Disabled", "Missing", "Unknown"].includes(raw.tasks[key].state))
    && ["event_enabled", "trend_moo_enabled", "legend_live_today"].every(key => typeof raw[key] === "boolean")
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

  const legend = { ...base("legend", "Legend EMA", "Algorithmic", "Complete runtime setup, data preparation, and broker verification before activation.", "execution.html"),
    health: "unavailable", updated_at: null, report_date: null, summary: "No session evidence connected to this view" };
  if (runtimeFresh) {
    const active = [tasks.legend_signals, tasks.legend_session, tasks.legend_watchdog].filter(scheduled).length;
    legend.deployment = active === 0 ? "Not activated" : active < 3 ? "Setup incomplete"
      : raw.legend_live_today === true ? "Configured for live today" : "Shadow / unarmed";
    legend.summary = `${active}/3 scheduled components enabled · session fills not verified here`;
    legend.health = "reported";
    const failures = ["legend_signals", "legend_session", "legend_watchdog"].filter(key => failed(tasks[key]));
    if (failures.length) {
      legend.deployment = "Runner failure reported";
      legend.summary += ` · ${failures.map(key => key.replace("legend_", "")).join(", ")}`;
    }
  }

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
    sleeves: [event, trend, legend, dial, hedge] };
}
