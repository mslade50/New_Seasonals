/* Server-side policy for commands crossing the private-site execution bridge.
 *
 * This module is deliberately dependency-free so the same rules can be tested
 * under Node and bundled by Cloudflare Pages. Browser validation is only a UX
 * aid; this policy is the authoritative gate before a command is signed.
 */

export const BOOK_STALE_MS = 90_000;
export const POLICY_VERSION = "2026-08-20.2";

export const COMMAND_TYPES = Object.freeze([
  "echo", "entry_bracket", "close_only", "flatten", "cancel", "modify",
  "trim_readd", "add_to_position", "exit_attach", "scheduled_option",
  "scheduled_option_cancel", "option_spread",
]);

const COMMAND_TYPE_SET = new Set(COMMAND_TYPES);
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const SYMBOL_RE = /^[A-Z0-9][A-Z0-9.^=_\/-]{0,31}$/;
const DATE_RE = /^\d{4}-\d{2}-\d{2}$/;
const EXPIRY_RE = /^(?:\d{6}|\d{8}|\d{4}-\d{2}-\d{2})$/;

const TYPE_FIELDS = Object.freeze({
  echo: ["note"],
  entry_bracket: [
    "symbol", "sec_type", "currency", "fut_expiry", "exchange",
    "fut_ib_symbol", "fut_trading_class", "fut_multiplier", "fut_min_tick",
    "action", "quantity", "entry_type", "entry", "stop", "target",
    "entry_cap", "strategy", "scaleout", "time_stop", "expiry", "risk_ack",
  ],
  close_only: [
    "symbol", "sec_type", "expiry", "con_id", "currency", "expected_position",
    "action", "qty", "fraction", "order_type", "limit", "tif", "outside_rth",
  ],
  flatten: [
    "symbol", "sec_type", "expiry", "con_id", "currency", "expected_position",
    "qty", "fraction", "order_type", "limit", "tif", "outside_rth",
  ],
  cancel: ["scope", "perm_id", "order_id", "symbol"],
  modify: ["symbol", "perm_id", "order_id", "new_qty", "new_limit", "new_stop"],
  trim_readd: [
    "symbol", "sec_type", "expiry", "con_id", "currency", "expected_position",
    "fraction", "close_order_type", "readd", "readd_tif",
  ],
  add_to_position: [
    "symbol", "sec_type", "expiry", "con_id", "currency", "expected_position",
    "fraction", "order_type",
  ],
  exit_attach: [
    "symbol", "sec_type", "expiry", "con_id", "currency", "expected_position",
    "stop", "target", "time_stop", "outside_rth",
  ],
  scheduled_option: [
    "symbol", "right", "target_delta", "delta_tolerance", "premium_budget",
    "order_type", "tif", "execute_date", "execute_time", "timezone",
    "grace_minutes", "expiry_mode", "min_dte", "expiry", "risk_ack",
  ],
  scheduled_option_cancel: ["schedule_id"],
  option_spread: [
    "symbol", "action", "quantity", "limit", "tif", "structure", "debit_risk",
    "risk_per_unit", "credit", "legs", "strategy", "signal_date",
    "entry_condition", "risk_ack",
  ],
});

function fail(status, error) {
  return { ok: false, status, error };
}

function isPlainObject(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

function envFlag(value) {
  return ["1", "true", "yes", "on"].includes(String(value || "").trim().toLowerCase());
}

function envSet(value) {
  return new Set(String(value || "").split(",").map((x) => x.trim()).filter(Boolean));
}

function positiveEnv(value, fallback, max) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return fallback;
  return Math.min(n, max);
}

function finitePositive(value) {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

function positiveInteger(value) {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0;
}

function finiteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function optionalBoolean(value) {
  return value == null || typeof value === "boolean";
}

function safeText(value, max = 200) {
  return value == null || (typeof value === "string" && value.length <= max && !/[\u0000-\u001f]/.test(value));
}

function safeSymbol(value) {
  return typeof value === "string" && SYMBOL_RE.test(value.toUpperCase().trim());
}

function exactKeys(value, allowed, label) {
  if (!isPlainObject(value)) return `${label} must be an object`;
  const allow = new Set(allowed);
  const extra = Object.keys(value).filter((key) => !allow.has(key));
  return extra.length ? `${label} has unsupported field(s): ${extra.join(", ")}` : null;
}

export function epochMs(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return null;
  return n < 1e12 ? n * 1000 : n;
}

export function bookAgeMs(book, now = Date.now()) {
  const at = epochMs(book && book.at);
  return at == null ? null : Math.max(0, now - at);
}

function accountRow(book, account) {
  return ((book && book.accounts) || []).find((row) => row && row.key === account) || null;
}

function sameContract(item, payload) {
  if (payload.con_id != null) {
    return item.con_id != null && Number(payload.con_id) === Number(item.con_id);
  }
  return String(item.symbol || "").toUpperCase() === String(payload.symbol || "").toUpperCase()
    && (!payload.sec_type || !item.sec_type || String(item.sec_type).toUpperCase() === String(payload.sec_type).toUpperCase())
    && (!payload.currency || !item.currency || String(item.currency).toUpperCase() === String(payload.currency).toUpperCase())
    && (!payload.expiry || !item.expiry || String(item.expiry).startsWith(String(payload.expiry)));
}

function findPosition(account, payload) {
  return ((account && account.positions) || []).find((position) => sameContract(position, payload)) || null;
}

function findOrder(account, payload) {
  return ((account && account.orders) || []).find((order) => {
    if (payload.perm_id != null && Number(order.perm_id) === Number(payload.perm_id)) return true;
    return payload.order_id != null && Number(order.order_id) === Number(payload.order_id);
  }) || null;
}

function findProtectiveStop(account, payload, position = findPosition(account, payload)) {
  if (!position) return null;
  const closeAction = Number(position.position) > 0 ? "SELL" : "BUY";
  const reference = Number(position.market_price ?? position.avg_cost);
  return ((account && account.orders) || []).find((order) => {
    if (!sameContract(order, payload)) return false;
    if (String(order.action || "").toUpperCase() !== closeAction) return false;
    if (String(order.order_type || "").toUpperCase() !== "STP") return false;
    const stop = Number(order.aux ?? order.stop ?? order.stop_price);
    if (!(stop > 0)) return false;
    if (reference > 0 && Number(position.position) > 0 && stop >= reference) return false;
    if (reference > 0 && Number(position.position) < 0 && stop <= reference) return false;
    return true;
  }) || null;
}

function validateIdentity(payload) {
  if (!safeSymbol(payload.symbol)) return "symbol is required and must be a supported contract symbol";
  if (payload.con_id != null && !positiveInteger(payload.con_id)) return "con_id must be a positive integer";
  if (payload.expected_position != null && !finiteNumber(payload.expected_position)) return "expected_position must be a number";
  if (!safeText(payload.sec_type, 8) || !safeText(payload.currency, 8) || !safeText(payload.expiry, 16)) return "contract identity is invalid";
  return null;
}

function validateLivePositionIdentity(payload) {
  if (!positiveInteger(payload.con_id)) return "live position commands require con_id";
  if (!finiteNumber(payload.expected_position) || Number(payload.expected_position) === 0) {
    return "live position commands require a non-zero expected_position";
  }
  if (typeof payload.sec_type !== "string" || !/^[A-Z]{3,8}$/.test(payload.sec_type)) {
    return "live position commands require sec_type";
  }
  if (typeof payload.currency !== "string" || !/^[A-Z]{3}$/.test(payload.currency)) {
    return "live position commands require currency";
  }
  if (["FUT", "OPT", "FOP"].includes(payload.sec_type)
      && (typeof payload.expiry !== "string" || !EXPIRY_RE.test(payload.expiry))) {
    return "live derivative position commands require expiry";
  }
  return null;
}

function validateEntry(payload, dryRun) {
  const identityError = validateIdentity(payload);
  if (identityError) return identityError;
  if (!["STK", "FUT", "CASH"].includes(payload.sec_type)) return "sec_type must be STK, FUT, or CASH";
  if (typeof payload.currency !== "string" || !/^[A-Z]{3}$/.test(payload.currency)) return "currency must be a 3-letter uppercase code";
  if (!["BUY", "SELL"].includes(payload.action)) return "action must be BUY or SELL";
  if (!positiveInteger(payload.quantity)) return "quantity must be a positive integer";
  if (!["LMT", "STP_LMT", "MKT", "MOO", "MOC"].includes(payload.entry_type)) return "entry_type is not allowed";
  if (payload.entry_type === "STP_LMT" && payload.sec_type !== "STK") return "STP_LMT is allowed only for stock entries";
  if (payload.entry_type === "MOC" && payload.sec_type !== "STK") return "MOC is allowed only for stock entries";
  if (payload.entry_type === "MOO" && payload.sec_type === "CASH") return "MOO is not allowed for FX entries";
  if (!finitePositive(payload.entry)) return "entry must be > 0";
  if (payload.entry_type === "STP_LMT" && !finitePositive(payload.entry_cap)) return "entry_cap is required for STP_LMT";
  if (payload.entry_type !== "STP_LMT" && payload.entry_cap != null) return "entry_cap is only allowed for STP_LMT";
  if (payload.stop != null && !finitePositive(payload.stop)) return "stop must be > 0 when present";
  if (!dryRun && !finitePositive(payload.stop)) return "live entries require a defined stop";
  if (payload.target != null && !finitePositive(payload.target)) return "target must be > 0 when present";
  const worstEntry = payload.entry_type === "STP_LMT" ? Number(payload.entry_cap) : Number(payload.entry);
  if (finitePositive(payload.stop)) {
    if (payload.action === "BUY" && Number(payload.stop) >= worstEntry) return "BUY stop must be below the entry risk price";
    if (payload.action === "SELL" && Number(payload.stop) <= worstEntry) return "SELL stop must be above the entry risk price";
  }
  if (finitePositive(payload.target)) {
    if (payload.action === "BUY" && Number(payload.target) <= worstEntry) return "BUY target must be above the entry risk price";
    if (payload.action === "SELL" && Number(payload.target) >= worstEntry) return "SELL target must be below the entry risk price";
  }
  if (payload.time_stop != null && (!safeText(payload.time_stop, 16) || !DATE_RE.test(payload.time_stop))) return "time_stop must be YYYY-MM-DD";
  if (payload.expiry != null && (!safeText(payload.expiry, 16) || !DATE_RE.test(payload.expiry))) return "expiry must be YYYY-MM-DD";
  if (payload.strategy != null
      && (typeof payload.strategy !== "string" || !/^[A-Za-z0-9 _.-]{1,32}$/.test(payload.strategy))) {
    return "strategy tag is invalid";
  }
  if (!optionalBoolean(payload.risk_ack)) return "risk_ack is invalid";
  if (payload.scaleout != null) {
    const keysError = exactKeys(payload.scaleout, ["frac", "target"], "scaleout");
    if (keysError) return keysError;
    const frac = payload.scaleout.frac;
    if (!finiteNumber(frac) || !(frac > 0 && frac < 1) || !finitePositive(payload.scaleout.target)) return "scaleout requires 0 < frac < 1 and target > 0";
  }
  if (payload.scaleout != null && payload.sec_type !== "STK") {
    return "scaleout is allowed only for stock entries";
  }
  if (payload.sec_type === "CASH") {
    if (!/^[A-Z]{3}$/.test(payload.symbol) || payload.symbol === payload.currency
        || (payload.symbol !== "USD" && payload.currency !== "USD")) {
      return "FX entries require two different 3-letter currencies with one USD leg";
    }
  }
  if (payload.sec_type === "FUT") {
    if (!finitePositive(payload.fut_multiplier)) return "futures entries require fut_multiplier > 0";
    if (!finitePositive(payload.fut_min_tick)) return "futures entries require fut_min_tick > 0";
    if (!safeText(payload.fut_expiry, 10) || !/^\d{6,8}$/.test(String(payload.fut_expiry || ""))) return "futures entries require a numeric contract month";
    if (!safeText(payload.exchange, 12) || !["CME", "CBOT", "NYMEX", "COMEX"].includes(payload.exchange)) return "futures exchange is not allowed";
    if (!safeSymbol(payload.fut_ib_symbol) || !safeText(payload.fut_trading_class, 32)
        || !String(payload.fut_trading_class || "").trim()) return "futures contract identity is incomplete";
  } else if ([
    payload.fut_expiry, payload.exchange, payload.fut_ib_symbol,
    payload.fut_trading_class, payload.fut_multiplier, payload.fut_min_tick,
  ].some((value) => value != null)) {
    return "futures contract fields are allowed only for FUT entries";
  }
  return null;
}

function validateClose(payload, account, dryRun) {
  const identityError = validateIdentity(payload);
  if (identityError) return identityError;
  if (!dryRun) {
    const liveIdentityError = validateLivePositionIdentity(payload);
    if (liveIdentityError) return liveIdentityError;
  }
  const position = findPosition(account, payload);
  if (!position || !Number(position.position)) return "matching live position was not found";
  if (payload.expected_position != null && Number(payload.expected_position) !== Number(position.position)) return "position changed since the command was composed";
  const hasQty = payload.qty != null;
  const hasFraction = payload.fraction != null;
  if (hasQty === hasFraction) return "provide exactly one of qty or fraction";
  const held = Math.abs(Number(position.position));
  if (hasQty && (!positiveInteger(payload.qty) || Number(payload.qty) > held)) return "close qty exceeds the live position";
  if (hasFraction && (!finiteNumber(payload.fraction) || !(payload.fraction > 0 && payload.fraction <= 1))) return "close fraction must be in (0, 1]";
  if (!["MKT", "LMT"].includes(payload.order_type)) return "close order_type must be MKT or LMT";
  if (payload.order_type === "LMT" && !finitePositive(payload.limit)) return "limit is required for a limit close";
  if (!optionalBoolean(payload.outside_rth)) return "outside_rth must be boolean";
  if (payload.outside_rth === true && payload.order_type !== "LMT") return "outside-RTH closes must be limit orders";
  if (payload.tif != null && !["DAY", "GTC"].includes(payload.tif)) return "close tif must be DAY or GTC";
  if (payload.action != null) {
    const expected = Number(position.position) > 0 ? "SELL" : "BUY";
    if (payload.action !== expected) return "close action does not reduce the live position";
  }
  return null;
}

function validateCancel(payload, dryRun, env) {
  if (payload.scope === "order") {
    if (!positiveInteger(payload.perm_id) && !positiveInteger(payload.order_id)) return "order cancel requires perm_id or order_id";
    return null;
  }
  if (payload.scope === "symbol") {
    if (!safeSymbol(payload.symbol)) return "symbol cancel requires a valid symbol";
    if (!dryRun && !envFlag(env.EXEC_ALLOW_SYMBOL_CANCEL)) return "live symbol-wide cancel is disabled by server policy";
    return null;
  }
  return "cancel scope must be order or symbol";
}

function validateModify(payload, account, dryRun, env) {
  if (!safeSymbol(payload.symbol)) return "modify requires a valid symbol";
  if (!positiveInteger(payload.perm_id) && !positiveInteger(payload.order_id)) return "modify requires perm_id or order_id";
  const changes = [payload.new_qty, payload.new_limit, payload.new_stop].filter((x) => x != null);
  if (!changes.length) return "modify requires positive qty, limit, or stop changes";
  if (payload.new_qty != null && !positiveInteger(payload.new_qty)) return "new_qty must be a positive integer";
  if (payload.new_limit != null && !finitePositive(payload.new_limit)) return "new_limit must be > 0";
  if (payload.new_stop != null && !finitePositive(payload.new_stop)) return "new_stop must be > 0";
  if (!dryRun) {
    const order = findOrder(account, payload);
    if (!order) return "matching live order was not found";
    if (!sameContract(order, payload)) return "live order identity does not match the requested symbol";
    const currentQty = Number(order.qty ?? order.quantity);
    if (payload.new_qty != null && Number.isFinite(currentQty) && Number(payload.new_qty) > currentQty) return "live modify cannot increase order quantity";
    if ((payload.new_limit != null || payload.new_stop != null)
        && !envFlag(env.EXEC_ALLOW_PRICE_MODIFY)) {
      return "live price/stop modification is disabled by server policy";
    }
  }
  return null;
}

function validatePositionMutation(payload, account, env, dryRun) {
  const identityError = validateIdentity(payload);
  if (identityError) return identityError;
  if (!dryRun) {
    const liveIdentityError = validateLivePositionIdentity(payload);
    if (liveIdentityError) return liveIdentityError;
  }
  const position = findPosition(account, payload);
  if (!position || !Number(position.position)) return "matching live position was not found";
  if (Number(payload.expected_position) !== Number(position.position)) return "position changed since the command was composed";
  const maxFraction = positiveEnv(env.EXEC_MAX_ADD_FRACTION, 0.5, 1.0);
  if (!finiteNumber(payload.fraction) || !(payload.fraction > 0 && payload.fraction <= maxFraction)) return `fraction exceeds server cap ${maxFraction}`;
  return null;
}

function validateExitAttach(payload, account, dryRun) {
  const identityError = validateIdentity(payload);
  if (identityError) return identityError;
  if (!dryRun) {
    const liveIdentityError = validateLivePositionIdentity(payload);
    if (liveIdentityError) return liveIdentityError;
  }
  const position = findPosition(account, payload);
  if (!position) return "matching live position was not found";
  if (payload.expected_position != null
      && Number(payload.expected_position) !== Number(position.position)) {
    return "position changed since the command was composed";
  }
  if (payload.stop == null && payload.target == null && payload.time_stop == null) return "attach requires a stop, target, or time stop";
  if (payload.stop != null && !finitePositive(payload.stop)) return "stop must be > 0";
  if (payload.target != null && !finitePositive(payload.target)) return "target must be > 0";
  if (payload.time_stop != null && (!safeText(payload.time_stop, 16) || !DATE_RE.test(payload.time_stop))) return "time_stop must be YYYY-MM-DD";
  if (!optionalBoolean(payload.outside_rth)) return "outside_rth must be boolean";
  const reference = Number(position.market_price ?? position.avg_cost);
  if (reference > 0 && Number(position.position) > 0) {
    if (payload.stop != null && payload.stop >= reference) return "long stop must be below the fresh reference price";
    if (payload.target != null && payload.target <= reference) return "long target must be above the fresh reference price";
  } else if (reference > 0 && Number(position.position) < 0) {
    if (payload.stop != null && payload.stop <= reference) return "short stop must be above the fresh reference price";
    if (payload.target != null && payload.target >= reference) return "short target must be below the fresh reference price";
  }
  return null;
}

function validateScheduledOption(payload) {
  if (!safeSymbol(payload.symbol)) return "scheduled option requires a valid symbol";
  if (!["P", "C"].includes(payload.right)) return "right must be P or C";
  if (!finiteNumber(payload.target_delta) || !(payload.target_delta > 0 && payload.target_delta < 1)) return "target_delta must be in (0, 1)";
  if (!finitePositive(payload.premium_budget)) return "premium_budget must be > 0";
  if (payload.order_type !== "MKT" || payload.tif !== "DAY") return "scheduled options must be MKT DAY";
  if (!DATE_RE.test(String(payload.execute_date || "")) || !/^\d{2}:\d{2}$/.test(String(payload.execute_time || ""))) return "scheduled option execution time is invalid";
  if (payload.timezone !== "America/New_York") return "scheduled option timezone must be America/New_York";
  if (!finiteNumber(payload.delta_tolerance) || !(payload.delta_tolerance > 0 && payload.delta_tolerance <= 0.25)) return "delta_tolerance is invalid";
  if (!positiveInteger(payload.grace_minutes) || Number(payload.grace_minutes) > 15) return "grace_minutes must be 1-15";
  if (!["min_dte", "specific"].includes(payload.expiry_mode)) return "expiry_mode is invalid";
  if (payload.expiry_mode === "min_dte" && !positiveInteger(payload.min_dte)) return "min_dte is required";
  if (payload.expiry_mode === "specific" && !EXPIRY_RE.test(String(payload.expiry || ""))) return "specific expiry is invalid";
  if (!optionalBoolean(payload.risk_ack)) return "risk_ack must be boolean";
  return null;
}

function validateOptionSpread(payload, dryRun) {
  if (!safeSymbol(payload.symbol)) return "option spread requires a valid symbol";
  if (!["BUY", "SELL"].includes(payload.action)) return "option action must be BUY or SELL";
  if (!positiveInteger(payload.quantity) || !finitePositive(payload.limit)) return "option quantity and limit must be positive";
  if (!["DAY", "GTC"].includes(payload.tif)) return "option tif must be DAY or GTC";
  if (typeof payload.credit !== "boolean" || !finitePositive(payload.risk_per_unit) || !finitePositive(payload.debit_risk)) return "defined option risk is required";
  if (Math.abs(Number(payload.risk_per_unit) - Number(payload.debit_risk)) > 1e-6) return "option risk fields disagree";
  if (!payload.credit && Number(payload.risk_per_unit) + 1e-6 < Number(payload.limit)) return "debit risk understates the order debit";
  if (!Array.isArray(payload.legs) || payload.legs.length < 1 || payload.legs.length > 4) return "option order requires 1-4 legs";
  for (const leg of payload.legs) {
    const keysError = exactKeys(leg, ["side", "right", "expiry", "strike", "ratio", "con_id"], "option leg");
    if (keysError) return keysError;
    if (!["BUY", "SELL"].includes(leg.side) || !["P", "C"].includes(leg.right)) return "option leg side/right is invalid";
    if (!EXPIRY_RE.test(String(leg.expiry || "")) || !finitePositive(leg.strike) || !positiveInteger(leg.ratio)) return "option leg contract is invalid";
    if (!dryRun && !positiveInteger(leg.con_id)) return "live option legs require con_id";
    if (leg.con_id != null && !positiveInteger(leg.con_id)) return "option con_id must be a positive integer";
  }
  if (!safeText(payload.structure, 80) || !safeText(payload.strategy, 120)
      || !safeText(payload.signal_date, 16) || !safeText(payload.entry_condition, 500)
      || !optionalBoolean(payload.risk_ack)) return "option metadata is invalid";
  return null;
}

function riskForEntry(payload) {
  if (!finitePositive(payload.stop)) return null;
  const entry = payload.entry_type === "STP_LMT" ? Number(payload.entry_cap) : Number(payload.entry);
  const quantity = Number(payload.quantity);
  const distance = Math.abs(entry - Number(payload.stop));
  if (payload.sec_type === "FUT") return quantity * distance * Number(payload.fut_multiplier);
  if (payload.sec_type === "CASH" && String(payload.symbol).toUpperCase() === "USD") return quantity * distance / entry;
  return quantity * distance;
}

function notionalForEntry(payload) {
  const entry = payload.entry_type === "STP_LMT" ? Number(payload.entry_cap) : Number(payload.entry);
  const quantity = Number(payload.quantity);
  if (payload.sec_type === "FUT") return null;
  if (payload.sec_type === "CASH" && String(payload.symbol).toUpperCase() === "USD") return quantity;
  return quantity * entry;
}

function enforceRiskCaps(type, payload, account, env) {
  const nlv = Number(account && account.nlv);
  if (!["entry_bracket", "scheduled_option", "option_spread", "add_to_position"].includes(type)) return null;
  if (!(nlv > 0)) return "live risk-increasing commands require a positive, fresh NLV";
  const maxRiskBps = positiveEnv(env.EXEC_MAX_NEW_RISK_BPS, 500, 10_000);
  let risk;
  if (type === "entry_bracket") risk = riskForEntry(payload);
  else if (type === "scheduled_option") risk = Number(payload.premium_budget);
  else if (type === "option_spread") risk = Number(payload.risk_per_unit) * 100 * Number(payload.quantity);
  else {
    const position = findPosition(account, payload);
    const stopOrder = findProtectiveStop(account, payload, position);
    const price = Number(position && (position.market_price ?? position.avg_cost));
    const stop = Number(stopOrder && (stopOrder.aux ?? stopOrder.stop ?? stopOrder.stop_price));
    const addQty = Math.ceil(Math.abs(Number(position && position.position)) * Number(payload.fraction));
    if (!(price > 0) || !(stop > 0) || !(addQty > 0)) {
      return "live add risk cannot be verified from the fresh book";
    }
    risk = addQty * Math.abs(price - stop);
    const maxAddNotionalPct = positiveEnv(env.EXEC_MAX_ADD_NOTIONAL_PCT, 50, 1_000);
    if (addQty * price > nlv * maxAddNotionalPct / 100) {
      return `add notional exceeds server cap ${maxAddNotionalPct}% of NLV`;
    }
  }
  if (!(risk >= 0) || risk > nlv * maxRiskBps / 10_000) return `defined risk exceeds server cap ${maxRiskBps} bps of NLV`;

  if (type === "entry_bracket") {
    const notional = notionalForEntry(payload);
    const maxNotionalPct = positiveEnv(env.EXEC_MAX_NEW_NOTIONAL_PCT, 200, 10_000);
    if (notional != null && notional > nlv * maxNotionalPct / 100) return `entry notional exceeds server cap ${maxNotionalPct}% of NLV`;
  }
  return null;
}

function validatePayload(type, payload, account, dryRun, env) {
  const keysError = exactKeys(payload, TYPE_FIELDS[type], "payload");
  if (keysError) return keysError;
  switch (type) {
    case "echo": return safeText(payload.note, 200) ? null : "echo note is invalid";
    case "entry_bracket": return validateEntry(payload, dryRun);
    case "close_only":
    case "flatten": return validateClose(payload, account, dryRun);
    case "cancel": return validateCancel(payload, dryRun, env);
    case "modify": return validateModify(payload, account, dryRun, env);
    case "trim_readd": {
      const error = validatePositionMutation(payload, account, env, dryRun);
      if (error) return error;
      if (!dryRun && !findProtectiveStop(account, payload)) {
        return "live trim/re-add requires a matching protective price stop";
      }
      if (payload.close_order_type !== "MKT" || payload.readd !== true || payload.readd_tif !== "DAY") {
        return "trim/re-add execution shape is not allowed";
      }
      return null;
    }
    case "add_to_position": {
      const error = validatePositionMutation(payload, account, env, dryRun);
      if (error) return error;
      if (!dryRun && !findProtectiveStop(account, payload)) {
        return "live add requires a matching protective price stop";
      }
      return payload.order_type === "MKT" ? null : "add order_type must be MKT";
    }
    case "exit_attach": return validateExitAttach(payload, account, dryRun);
    case "scheduled_option": return validateScheduledOption(payload);
    case "scheduled_option_cancel": return UUID_RE.test(String(payload.schedule_id || "")) ? null : "schedule_id must be a UUID";
    case "option_spread": return validateOptionSpread(payload, dryRun);
    default: return "command type is not allowed";
  }
}

export function validateCommandRequest(body, { env = {}, status = {}, book = null, now = Date.now() } = {}) {
  const topKeysError = exactKeys(body, ["id", "type", "account", "dry_run", "payload"], "request");
  if (topKeysError) return fail(400, topKeysError);
  if (!UUID_RE.test(String(body.id || ""))) return fail(400, "id must be a client-generated UUID");
  if (!COMMAND_TYPE_SET.has(body.type)) return fail(400, "command type is not allowed");
  if (!["primary", "pa"].includes(body.account)) return fail(400, "account must be primary or pa");
  if (body.dry_run != null && typeof body.dry_run !== "boolean") return fail(400, "dry_run must be boolean");
  if (!isPlainObject(body.payload)) return fail(400, "payload must be an object");

  // Omitted dry_run is deliberately safe: it means preview, never live.
  const dryRun = body.dry_run !== false || body.type === "echo";
  if (status.online !== true) return fail(503, "agent is offline or heartbeat is stale");
  const age = bookAgeMs(book, now);
  const maxBookAge = positiveEnv(env.EXEC_BOOK_MAX_AGE_MS, BOOK_STALE_MS, 300_000);
  if (age == null || age > maxBookAge) return fail(409, "execution book is missing or stale");
  if (!book || !["live", "dry-run"].includes(book.mode)) return fail(409, "execution mode is not confirmed by the fresh book");
  if (!status.session_id || book._broker_session_id !== status.session_id) {
    return fail(409, "execution status and book do not belong to the same agent session");
  }
  if (!dryRun && book.mode !== "live") return fail(409, "fresh book does not confirm live mode");

  const account = accountRow(book, body.account);
  if (!account) return fail(409, "account is absent from the fresh execution book");
  if (["scheduled_option", "option_spread"].includes(body.type) && body.account !== "primary") {
    return fail(403, "options execution is disabled for this account");
  }
  if (!dryRun) {
    if (!envFlag(env.EXEC_LIVE_ENABLED)) return fail(403, "server live-order kill switch is off");
    if (!envSet(env.EXEC_LIVE_TYPES).has(body.type)) return fail(403, "command type is not armed server-side");
    if (!envSet(env.EXEC_LIVE_ACCOUNTS).has(body.account)) return fail(403, "account is not armed server-side");
    if (body.type === "entry_bracket") {
      const armedInstruments = envSet(env.EXEC_LIVE_INSTRUMENTS || "STK");
      const instrument = String(body.payload.sec_type || "STK").toUpperCase();
      if (!armedInstruments.has(instrument)) {
        return fail(403, `instrument ${instrument} is not armed server-side`);
      }
      const armedEntryTypes = envSet(env.EXEC_LIVE_ENTRY_TYPES || "LMT,STP_LMT");
      if (!armedEntryTypes.has(body.payload.entry_type)) {
        return fail(403, `entry type ${body.payload.entry_type} is not armed server-side`);
      }
    }
    if (["add_to_position", "trim_readd"].includes(body.type)) {
      const instruments = envSet(env.EXEC_LIVE_POSITION_INSTRUMENTS || "STK");
      const instrument = String(body.payload.sec_type || "STK").toUpperCase();
      if (!instruments.has(instrument)) {
        return fail(403, `position-mutation instrument ${instrument} is not armed server-side`);
      }
    }
  }

  const payloadError = validatePayload(body.type, body.payload, account, dryRun, env);
  if (payloadError) return fail(400, payloadError);
  if (!dryRun) {
    const riskError = enforceRiskCaps(body.type, body.payload, account, env);
    if (riskError) return fail(403, riskError);
    if ([
      "entry_bracket", "scheduled_option", "option_spread",
      "add_to_position", "trim_readd",
    ].includes(body.type)) {
      return fail(
        403,
        "risk-increasing live commands are disabled until the broker provides "
          + "an atomic aggregate risk reservation",
      );
    }
  }

  return {
    ok: true,
    command: {
      id: body.id, type: body.type, account: body.account,
      dry_run: dryRun, payload: body.payload,
    },
    policy_version: POLICY_VERSION,
  };
}
