/* Defense-in-depth command gate inside the standalone Durable Object.
 * Pages performs the detailed schema/risk checks; this gate independently
 * refuses stale state, unknown envelopes, and unarmed live delivery.
 */

export const HEARTBEAT_STALE_MS = 30_000;
export const BOOK_STALE_MS = 90_000;
export const COMMAND_POLICY_VERSION = "2026-08-20.1";

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const COMMAND_TYPES = new Set([
  "echo", "entry_bracket", "close_only", "flatten", "cancel", "modify",
  "trim_readd", "add_to_position", "exit_attach", "scheduled_option",
  "scheduled_option_cancel", "option_spread",
]);

function fail(status, error) {
  return { ok: false, status, error };
}

function epochMs(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return null;
  return n < 1e12 ? n * 1000 : n;
}

function envFlag(value) {
  return ["1", "true", "yes", "on"].includes(String(value || "").trim().toLowerCase());
}

function envSet(value) {
  return new Set(String(value || "").split(",").map((x) => x.trim()).filter(Boolean));
}

export async function validHmacHex(secret, message, signature) {
  if (!secret || typeof message !== "string" || !/^[0-9a-f]{64}$/i.test(String(signature || ""))) {
    return false;
  }
  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw", enc.encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["verify"],
  );
  const bytes = new Uint8Array(String(signature).match(/.{2}/g).map((pair) => parseInt(pair, 16)));
  return crypto.subtle.verify("HMAC", key, bytes, enc.encode(message));
}

export function validateBrokerCommand(cmd, {
  env = {}, now = Date.now(), lastSeen = 0, book = null, socketCount = 0,
  socketSession = null,
} = {}) {
  if (!cmd || typeof cmd !== "object" || Array.isArray(cmd)
      || !UUID_RE.test(String(cmd.id || ""))
      || !COMMAND_TYPES.has(cmd.type)
      || cmd.policy_version !== COMMAND_POLICY_VERSION
      || !["primary", "pa"].includes(cmd.account)
      || typeof cmd.dry_run !== "boolean"
      || !cmd.payload || typeof cmd.payload !== "object" || Array.isArray(cmd.payload)) {
    return fail(400, "command envelope rejected by broker policy");
  }

  const createdAt = Number(cmd.created_at);
  const expiresAt = Number(cmd.expires_at);
  if (!Number.isFinite(createdAt) || !Number.isFinite(expiresAt)
      || createdAt > now + 5_000 || expiresAt <= now
      || expiresAt - createdAt > 60_000) {
    return fail(400, "command envelope is expired or invalid");
  }

  const heartbeatAt = Number(lastSeen);
  if (socketCount !== 1 || !heartbeatAt || now - heartbeatAt < 0
      || now - heartbeatAt >= HEARTBEAT_STALE_MS) {
    return fail(503, "agent offline, heartbeat stale, or socket session ambiguous");
  }

  const bookAt = epochMs(book && book.at);
  if (!bookAt || now - bookAt < 0 || now - bookAt > BOOK_STALE_MS
      || !["live", "dry-run"].includes(book && book.mode)) {
    return fail(409, "execution book missing, stale, or mode unknown");
  }
  if (!socketSession || book._broker_session_id !== socketSession) {
    return fail(409, "execution book does not belong to the active agent session");
  }

  if (cmd.dry_run === false) {
    if (!envFlag(env.EXEC_LIVE_ENABLED)) return fail(403, "broker live-order kill switch is off");
    if (!envSet(env.EXEC_LIVE_TYPES).has(cmd.type)
        || !envSet(env.EXEC_LIVE_ACCOUNTS).has(cmd.account)) {
      return fail(403, "command type or account is not armed at broker");
    }
    if (cmd.type === "entry_bracket") {
      const instruments = envSet(env.EXEC_LIVE_INSTRUMENTS || "STK");
      const instrument = String(cmd.payload.sec_type || "STK").toUpperCase();
      if (!instruments.has(instrument)) {
        return fail(403, `instrument ${instrument} is not armed at broker`);
      }
      const entryTypes = envSet(env.EXEC_LIVE_ENTRY_TYPES || "LMT,STP_LMT");
      if (!entryTypes.has(cmd.payload.entry_type)) {
        return fail(403, `entry type ${cmd.payload.entry_type} is not armed at broker`);
      }
    }
    if (["add_to_position", "trim_readd"].includes(cmd.type)) {
      const instruments = envSet(env.EXEC_LIVE_POSITION_INSTRUMENTS || "STK");
      const instrument = String(cmd.payload.sec_type || "STK").toUpperCase();
      if (!instruments.has(instrument)) {
        return fail(403, `position-mutation instrument ${instrument} is not armed at broker`);
      }
    }
    if (cmd.type === "cancel" && cmd.payload.scope === "symbol"
        && !envFlag(env.EXEC_ALLOW_SYMBOL_CANCEL)) {
      return fail(403, "live symbol-wide cancel is disabled at broker");
    }
    if (cmd.type === "modify"
        && (cmd.payload.new_limit != null || cmd.payload.new_stop != null)
        && !envFlag(env.EXEC_ALLOW_PRICE_MODIFY)) {
      return fail(403, "live price/stop modification is disabled at broker");
    }
    if (book.mode !== "live") return fail(409, "fresh broker book does not confirm live mode");
  }
  return { ok: true };
}
