/* Pure fill-reconciliation helpers for the execution broker.
 *
 * The local agent republishes the current day's IBKR executions on every book
 * snapshot.  Command results can arrive before a resting order fills (or before
 * IBKR supplies its average price), so the Durable Object uses these helpers to
 * enrich its command audit rows from later execution snapshots.
 */

const FILLABLE_COMMAND_TYPES = new Set([
  "entry_bracket",
  "close_only",
  "flatten",
  "trim_readd",
  "add_to_position",
  "exit_attach",
  "option_spread",
  "scheduled_option",
]);

// Must stay aligned with execute_order.py PORTS. A mismatch fails closed: the
// audit backfill stops rather than borrowing a same-numbered order from another
// IBKR API client on the same account.
const EXECUTION_CLIENT_IDS = { primary: 123, pa: 147 };

function finiteNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function positiveNumber(value) {
  const n = finiteNumber(value);
  return n != null && n > 0 ? n : null;
}

function executionPrice(value) {
  const n = finiteNumber(value);
  return n != null && n !== 0 ? n : null;
}

function normalizedOrderId(value) {
  if (value == null || value === "") return null;
  const n = Number(value);
  return Number.isFinite(n) && n > 0 ? String(n) : null;
}

function normalizedSymbol(value) {
  return String(value || "").trim().toUpperCase();
}

function normalizedSide(value) {
  const side = String(value || "").trim().toUpperCase();
  if (side === "BUY" || side === "BOT") return "BUY";
  if (side === "SELL" || side === "SLD") return "SELL";
  return "";
}

function sameNumber(a, b) {
  const aa = finiteNumber(a), bb = finiteNumber(b);
  return aa != null && bb != null && Math.abs(aa - bb) < 1e-10;
}

function executionIdParts(value) {
  const id = String(value || "");
  const match = /^(.*)\.(\d+)$/.exec(id);
  return match
    ? { family: match[1], revision: Number(match[2]) }
    : { family: id, revision: null };
}

// IBKR publishes a correction as the same execution id with only the digits
// after the final period incremented (for example .01 -> .02). Store one row
// per execution family so a correction replaces, rather than duplicates, it.
export function executionFamilyId(value) {
  return executionIdParts(value).family;
}

/* Do not let a later sparse snapshot erase a field IBKR supplied earlier.
   In particular, price/avg_price can briefly be null or zero while execution
   details settle, then become authoritative on a later snapshot. */
export function mergeExecutionFill(previous, incoming, now = Date.now()) {
  const prior = previous || {};
  const priorId = executionIdParts(prior.exec_id);
  const incomingId = executionIdParts(incoming && incoming.exec_id);
  const sameFamily = priorId.family && priorId.family === incomingId.family;
  if (sameFamily && priorId.revision != null && incomingId.revision != null
      && incomingId.revision < priorId.revision) return prior;
  const corrected = sameFamily && priorId.revision != null && incomingId.revision != null
    && incomingId.revision > priorId.revision;
  // A higher revision supersedes every value on the earlier execution. Do not
  // carry its price/quantity into a temporarily sparse correction.
  const next = corrected ? {} : { ...prior };
  for (const [key, value] of Object.entries(incoming || {})) {
    if (value == null || value === "") {
      if (!(key in next)) next[key] = value;
      continue;
    }
    if ((key === "price" || key === "avg_price") && executionPrice(value) == null
        && executionPrice(next[key]) != null) {
      continue;
    }
    next[key] = value;
  }
  next.ingested_at = corrected ? now : (prior.ingested_at || next.ingested_at || now);
  return next;
}

/* Persist only the non-sensitive fields needed to distinguish an entry parent
   from the opposite-side stop/target children returned in the same order_ids
   array. */
export function commandFillMatch(command) {
  if (!FILLABLE_COMMAND_TYPES.has(String((command || {}).type || ""))) return null;
  const payload = (command && command.payload) || {};
  const match = {};
  const symbol = normalizedSymbol(payload.fut_ib_symbol || payload.symbol);
  let side = normalizedSide(payload.action);
  let secType = normalizedSymbol(payload.sec_type);
  // SMART combo executions can be accompanied by executions for each OPT leg.
  // Only the BAG row carries the authoritative net spread price. Single-leg
  // option tickets, including scheduled delta-selected buys, execute as OPT.
  if (command.type === "option_spread") {
    const legCount = Array.isArray(payload.legs) ? payload.legs.length : 0;
    secType = legCount > 1 ? "BAG" : (legCount === 1 ? "OPT" : "");
  } else if (command.type === "scheduled_option") {
    side = "BUY";
    secType = "OPT";
  }
  const expectedQty = positiveNumber(payload.quantity ?? payload.qty);
  const clientId = positiveNumber(EXECUTION_CLIENT_IDS[command.account]);
  if (symbol) match.symbol = symbol;
  if (side) match.side = side;
  if (secType) match.sec_type = secType;
  if (expectedQty != null) match.expected_qty = expectedQty;
  if (clientId != null) match.client_id = clientId;
  return Object.keys(match).length ? match : null;
}

function trackedOrderIds(command) {
  const resultFill = command && command.result && command.result.fill;
  if (!resultFill) return [];
  const ids = Array.isArray(resultFill.order_ids)
    ? resultFill.order_ids.map(normalizedOrderId).filter(Boolean)
    : [];

  // entry_bracket returns every parent and child id. New records carry side +
  // symbol match metadata, which safely admits every same-side parent (including
  // scale-out parents). For pre-deploy records, the first id is the first parent;
  // restricting the fallback to it avoids attributing a later stop/target exit.
  if (command.type === "entry_bracket") {
    if (ids.length) {
      return command.fill_match && command.fill_match.side && command.fill_match.symbol
        ? ids
        : ids.slice(0, 1);
    }
  }
  const nestedExecution = resultFill.trim || resultFill.add || {};
  const direct = normalizedOrderId(resultFill.order_id || nestedExecution.order_id);
  if (direct) return [direct];
  return ids;
}

function trackedPermIds(command) {
  const resultFill = command && command.result && command.result.fill;
  if (!resultFill) return [];
  const nestedExecution = resultFill.trim || resultFill.add || {};
  const direct = normalizedOrderId(resultFill.perm_id || nestedExecution.perm_id);
  return direct ? [direct] : [];
}

function fillMatchesCommand(command, fill, ids) {
  const orderId = normalizedOrderId(fill && fill.order_id);
  if (!orderId || !ids.includes(orderId)) return false;
  if (command.account && fill.account_key && command.account !== fill.account_key) return false;
  const commandAt = finiteNumber(command.created_at);
  const fillAt = Date.parse(fill.time);
  // Client-scoped IB order ids can overlap. Ignore executions that predate the
  // command (with a small clock-skew allowance) even when account/id match.
  if (commandAt != null && Number.isFinite(fillAt) && fillAt < commandAt - 120_000) return false;
  const match = command.fill_match || {};
  // Option command records created before the discriminator was introduced do
  // not contain enough information to distinguish a BAG aggregate from its
  // component leg executions. Fail closed instead of publishing a leg premium.
  if ((command.type === "option_spread" || command.type === "scheduled_option")
      && !match.sec_type) return false;
  const expectedClientId = positiveNumber(match.client_id || EXECUTION_CLIENT_IDS[command.account]);
  if (expectedClientId && positiveNumber(fill.client_id) !== expectedClientId) return false;
  const permIds = trackedPermIds(command);
  if (permIds.length && !permIds.includes(normalizedOrderId(fill.perm_id))) return false;
  if (match.symbol) {
    const symbol = normalizedSymbol(fill.symbol || fill.local_symbol);
    if (!symbol || symbol !== match.symbol) return false;
  }
  if (match.sec_type && normalizedSymbol(fill.sec_type) !== match.sec_type) return false;
  if (match.side) {
    const side = normalizedSide(fill.side);
    if (!side || side !== match.side) return false;
  }
  return true;
}

/* Aggregate execution rows one order at a time. Normally ex.price is present on
   every partial. If it is delayed, the newest cumulative ex.avgPrice is the
   correct order-level fallback; weighting cumulative averages as if they were
   individual prints would be wrong. */
function aggregateMatchedFills(fills) {
  const byOrder = new Map();
  for (const fill of fills) {
    const orderId = normalizedOrderId(fill.order_id);
    if (!orderId) continue;
    if (!byOrder.has(orderId)) byOrder.set(orderId, []);
    byOrder.get(orderId).push(fill);
  }

  let totalQty = 0;
  let totalNotional = 0;
  const matchedOrderIds = [];
  for (const [orderId, rows] of byOrder) {
    const rowQty = rows.reduce((sum, row) => sum + (positiveNumber(row.qty) || 0), 0);
    if (!(rowQty > 0)) continue;
    const cumulativeQty = rows.reduce(
      (largest, row) => Math.max(largest, positiveNumber(row.cum_qty) || 0), 0,
    );
    // A larger cumulative quantity proves that older partials are absent from
    // retention/the cap. Even direct prices cannot reconstruct the full order
    // from that truncated set, so preserve the previously reconciled result.
    if (cumulativeQty > 0 && Math.abs(cumulativeQty - rowQty) >= 1e-9) return null;
    let orderAvg = null;
    if (rows.every((row) => executionPrice(row.price) != null)) {
      orderAvg = rows.reduce(
        (sum, row) => sum + positiveNumber(row.qty) * executionPrice(row.price), 0,
      ) / rowQty;
    } else {
      const cumulative = rows
        // ex.avgPrice is cumulative only through ex.cumQty. It is safe as an
        // order-level fallback solely when it covers every execution row we are
        // about to report; otherwise wait for the delayed print/average.
        .filter((row) => executionPrice(row.avg_price) != null
          && positiveNumber(row.cum_qty) != null
          && Math.abs(positiveNumber(row.cum_qty) - rowQty) < 1e-9)
        .sort((a, b) => (positiveNumber(b.cum_qty) || 0) - (positiveNumber(a.cum_qty) || 0));
      if (cumulative.length) orderAvg = executionPrice(cumulative[0].avg_price);
    }
    if (orderAvg == null) return null; // do not publish a knowingly partial VWAP
    totalQty += rowQty;
    totalNotional += rowQty * orderAvg;
    matchedOrderIds.push(orderId);
  }
  if (!(totalQty > 0) || !matchedOrderIds.length) return null;
  matchedOrderIds.sort((a, b) => Number(a) - Number(b));
  return { filled: totalQty, avg_fill: totalNotional / totalQty, matched_order_ids: matchedOrderIds };
}

export function reconcileCommandFills(commands, fills, now = Date.now()) {
  let changed = false;
  const nextCommands = (commands || []).map((command) => {
    if (!FILLABLE_COMMAND_TYPES.has(String(command.type || ""))) return command;
    const ids = trackedOrderIds(command);
    if (!ids.length) return command;
    const matched = (fills || []).filter((fill) => fillMatchesCommand(command, fill, ids));
    const aggregate = aggregateMatchedFills(matched);
    if (!aggregate) return command;

    const oldFill = (command.result && command.result.fill) || {};
    const sameIds = JSON.stringify(oldFill.matched_order_ids || []) ===
      JSON.stringify(aggregate.matched_order_ids);
    if (sameNumber(oldFill.filled, aggregate.filled)
        && sameNumber(oldFill.avg_fill, aggregate.avg_fill) && sameIds) return command;

    const updatedFill = {
      ...oldFill,
      ...aggregate,
      reconciled_at: now,
    };
    if (!normalizedOrderId(updatedFill.order_id)
        && !Array.isArray(updatedFill.order_ids) && aggregate.matched_order_ids.length === 1) {
      updatedFill.order_id = Number(aggregate.matched_order_ids[0]);
    }
    const expected = positiveNumber(command.fill_match && command.fill_match.expected_qty);
    if (expected != null && aggregate.filled >= expected) {
      updatedFill.status = "Filled";
    } else if (["", "Submitted", "PreSubmitted", "PendingSubmit"].includes(String(updatedFill.status || ""))) {
      updatedFill.status = "Execution recorded";
    }
    changed = true;
    const oldResult = command.result || {};
    const suffix = "later execution reconciled from IBKR";
    const detail = oldResult.detail && !String(oldResult.detail).includes(suffix)
      ? `${oldResult.detail} · ${suffix}`
      : oldResult.detail;
    return { ...command, result: { ...oldResult, detail, fill: updatedFill, reconciled_at: now } };
  });
  return { commands: nextCommands, changed };
}

/* A duplicate/reconnect result from the agent can be sparser than a command row
   already reconciled from executions. Preserve the authoritative backfill. */
export function mergeCommandResult(previous, incoming) {
  const next = { ...(incoming || {}) };
  const priorFill = previous && previous.fill;
  const incomingFill = (incoming && incoming.fill) || {};
  if (!priorFill || !priorFill.reconciled_at) return next;
  const merged = { ...priorFill, ...incomingFill };
  // Once execution snapshots have reconciled the row, they are the source of
  // truth even if a delayed/replayed command result carries a non-zero partial.
  if (executionPrice(priorFill.avg_fill) != null) merged.avg_fill = priorFill.avg_fill;
  if (positiveNumber(priorFill.filled) != null) merged.filled = priorFill.filled;
  merged.reconciled_at = priorFill.reconciled_at;
  if (priorFill.matched_order_ids) merged.matched_order_ids = priorFill.matched_order_ids;
  if (["Filled", "Execution recorded"].includes(priorFill.status)
      && ["", "Submitted", "PreSubmitted", "PendingSubmit"].includes(String(incomingFill.status || ""))) {
    merged.status = priorFill.status;
  }
  next.fill = merged;
  next.reconciled_at = previous.reconciled_at || priorFill.reconciled_at;
  const suffix = "later execution reconciled from IBKR";
  const detail = next.detail || previous.detail;
  if (detail && !String(detail).includes(suffix)) next.detail = `${detail} · ${suffix}`;
  return next;
}
