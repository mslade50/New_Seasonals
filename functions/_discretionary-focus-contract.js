const SCHEMA = "discretionary-focus.v1";
const PHASES = new Set(["PROVISIONAL", "FINAL"]);
const STATUSES = new Set(["READY", "NO_QUALIFIED_SETUP"]);
const TOP_KEYS = new Set([
  "schema_version", "research_only", "quick_review_created", "live_actions_enabled",
  "order_staging_enabled", "status", "phase", "as_of", "valid_for", "generated_at",
  "expires_at", "focus", "screen_summary", "provenance", "no_setup_reason",
]);
const CARD_KEYS = new Set([
  "rank", "ticker", "company_name", "why_now", "setup", "trigger", "invalidation",
  "catalyst", "priced_in", "next_proof", "event_date", "earnings_td", "technical",
  "sources",
]);
const FORBIDDEN = new Set([
  "action", "action_id", "allocation", "approval_status", "approved_for_capital",
  "broker", "decision", "dry_run_required", "limit_order", "notional", "order",
  "order_id", "order_type", "position_size", "position_size_pct", "proposed_weight_pct",
  "quantity", "quick_review", "risk_amt", "risk_bps", "shares", "side", "tif",
]);
const PRICE_KEYS = [
  "price", "pivot", "level", "trigger_price", "stop_price", "invalidation_price",
];

function record(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function text(value) {
  return typeof value === "string" && value.trim().length > 0;
}

function isoDate(value) {
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const parsed = new Date(`${value}T00:00:00Z`);
  return !Number.isNaN(parsed.getTime()) && parsed.toISOString().slice(0, 10) === value;
}

function zonedTime(value) {
  if (typeof value !== "string" || !/(?:Z|[+-]\d{2}:\d{2})$/.test(value)) return null;
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function nyParts(value) {
  const parts = Object.fromEntries(new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23",
  }).formatToParts(value).filter((part) => part.type !== "literal")
    .map((part) => [part.type, part.value]));
  return {
    date: `${parts.year}-${parts.month}-${parts.day}`,
    hour: Number(parts.hour), minute: Number(parts.minute), second: Number(parts.second),
  };
}

function safeURL(value) {
  try {
    if (!/^https?:\/\//i.test(String(value || "").trim())) return false;
    return ["http:", "https:"].includes(new URL(value).protocol);
  } catch (_) { return false; }
}

function forbidden(value) {
  if (Array.isArray(value)) return value.some(forbidden);
  if (!record(value)) return false;
  return Object.entries(value).some(([key, nested]) =>
    FORBIDDEN.has(String(key).trim().toLowerCase()) || forbidden(nested));
}

function onlyKeys(value, allowed) {
  return record(value) && Object.keys(value).every((key) => allowed.has(key));
}

function priceExpression(value) {
  if (text(value)) return true;
  if (!record(value) || !text(value.condition)) return false;
  let numeric = false;
  for (const key of PRICE_KEYS) {
    if (value[key] == null) continue;
    if (typeof value[key] !== "number" || !Number.isFinite(value[key])) return false;
    numeric = true;
  }
  return !numeric || value.price_basis === "RAW_AS_TRADED";
}

function validSource(source, generatedAt) {
  if (!record(source) || !text(source.source_id) || !text(source.label) ||
      !safeURL(source.url) || typeof source.primary !== "boolean") return false;
  const generatedDate = nyParts(generatedAt).date;
  let sourceTime;
  if (isoDate(source.as_of)) {
    if (source.as_of > generatedDate) return false;
    sourceTime = new Date(`${source.as_of}T12:00:00Z`);
  } else {
    sourceTime = zonedTime(source.as_of);
    if (!sourceTime || sourceTime > new Date(generatedAt.getTime() + 300000)) return false;
  }
  return generatedAt - sourceTime <= 550 * 86400000;
}

function validSummary(summary, selected) {
  if (!record(summary)) return false;
  const fields = ["input_count", "technical_pass_count", "research_pass_count", "selected_count"];
  if (!fields.every((key) => Number.isInteger(summary[key]) && summary[key] >= 0)) return false;
  if (summary.selected_count !== selected || summary.technical_pass_count > summary.input_count ||
      summary.research_pass_count > summary.technical_pass_count ||
      summary.selected_count > summary.research_pass_count || !record(summary.rejected_counts)) return false;
  const rejected = Object.values(summary.rejected_counts);
  return rejected.every((value) => Number.isInteger(value) && value >= 0) &&
    rejected.reduce((total, value) => total + value, 0) + selected === summary.input_count;
}

function validCard(card, index, payload, generatedAt) {
  if (!onlyKeys(card, CARD_KEYS) || card.rank !== index + 1 ||
      !/^[A-Z0-9][A-Z0-9.^\/-]{0,19}$/.test(card.ticker || "")) return false;
  for (const key of ["company_name", "why_now", "setup", "catalyst", "priced_in", "next_proof"])
    if (!text(card[key])) return false;
  if (!priceExpression(card.trigger) || !record(card.invalidation) ||
      !priceExpression(card.invalidation.technical) || !text(card.invalidation.thesis_kill) ||
      !isoDate(card.event_date) || card.event_date < payload.valid_for ||
      !Number.isInteger(card.earnings_td) || card.earnings_td <= 5) return false;
  const technical = card.technical;
  const observed = record(technical) ? zonedTime(technical.observed_at) : null;
  if (!observed || observed > new Date(generatedAt.getTime() + 300000) ||
      generatedAt - observed > 96 * 3600000 || technical.setup_gate !== "PASS" ||
      technical.liquidity_gate !== "PASS" || typeof technical.setup_quality !== "number" ||
      !Number.isFinite(technical.setup_quality) || technical.setup_quality < 0 ||
      technical.setup_quality > 100) return false;
  return Array.isArray(card.sources) && card.sources.length > 0 &&
    card.sources.every((source) => validSource(source, generatedAt)) &&
    card.sources.some((source) => source.primary === true);
}

export function validateFocusEnvelope(payload, now = new Date()) {
  if (!onlyKeys(payload, TOP_KEYS) || forbidden(payload) || payload.schema_version !== SCHEMA ||
      payload.research_only !== true || payload.quick_review_created !== false ||
      payload.live_actions_enabled !== false || payload.order_staging_enabled !== false ||
      !PHASES.has(payload.phase) || !STATUSES.has(payload.status) || !isoDate(payload.as_of) ||
      !isoDate(payload.valid_for) || payload.as_of > payload.valid_for) return false;
  const generatedAt = zonedTime(payload.generated_at);
  const expiresAt = zonedTime(payload.expires_at);
  if (!generatedAt || !expiresAt || expiresAt <= generatedAt ||
      generatedAt > new Date(now.getTime() + 300000)) return false;
  const generatedLocal = nyParts(generatedAt);
  const expiresLocal = nyParts(expiresAt);
  if ((payload.phase === "FINAL" && generatedLocal.date !== payload.valid_for) ||
      expiresLocal.date !== payload.valid_for ||
      expiresLocal.second !== 0 || expiresLocal.minute !== 15 ||
      ![13, 16].includes(expiresLocal.hour)) return false;
  if (!Array.isArray(payload.focus) || payload.focus.length > 2 ||
      (payload.status === "READY" && payload.focus.length === 0) ||
      (payload.status === "NO_QUALIFIED_SETUP" && payload.focus.length !== 0) ||
      (payload.status === "NO_QUALIFIED_SETUP" && !text(payload.no_setup_reason)) ||
      (payload.status === "READY" && payload.no_setup_reason)) return false;
  if (!validSummary(payload.screen_summary, payload.focus.length)) return false;
  const provenance = payload.provenance;
  const screenAt = record(provenance) ? zonedTime(provenance.screen_captured_at) : null;
  const researchAt = record(provenance) ? zonedTime(provenance.research_as_of) : null;
  if (!screenAt || !researchAt || !text(provenance.screen_snapshot_id) ||
      !text(provenance.research_snapshot_id) || !text(provenance.policy_version) ||
      screenAt > new Date(generatedAt.getTime() + 300000) ||
      researchAt > new Date(generatedAt.getTime() + 300000) ||
      generatedAt - screenAt > 96 * 3600000 || generatedAt - researchAt > 36 * 3600000) return false;
  for (const key of ["screen_digest", "research_digest"])
    if (Object.hasOwn(provenance, key) && !/^[0-9a-f]{64}$/.test(provenance[key])) return false;
  const tickers = new Set();
  for (let index = 0; index < payload.focus.length; index += 1) {
    if (!validCard(payload.focus[index], index, payload, generatedAt) ||
        tickers.has(payload.focus[index].ticker)) return false;
    tickers.add(payload.focus[index].ticker);
  }
  return true;
}
