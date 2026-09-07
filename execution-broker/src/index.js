/* execution-broker — Cloudflare Worker + Durable Object.
 *
 * The cloud broker for the site's execution bridge. A single Durable Object
 * instance ("main") holds the local agent's OUTBOUND, hibernatable WebSocket,
 * tracks a heartbeat, and relays signed commands to the agent + collects results.
 *
 * Phase 2b adds the command loop — but the broker is still a DUMB RELAY: it does
 * not build, validate, or transmit orders. The site signs a command, the broker
 * pushes it down the open socket, and the LOCAL AGENT verifies the signature,
 * validates, and (in dry-run) only logs what it WOULD do. No order ever originates
 * here.
 *
 * Endpoints (all DO-routed):
 *   GET  /agent     agent WS upgrade            (Bearer AGENT_TOKEN)
 *   GET  /status    heartbeat / online state    (Bearer STATUS_TOKEN)
 *   POST /command   {signed, sig} -> push to agent  (Bearer STATUS_TOKEN)
 *   GET  /commands  recent commands + results   (Bearer STATUS_TOKEN)
 *   GET  /fills     accumulated executions ring (Bearer STATUS_TOKEN)
 *   GET  /health    plain liveness
 *
 * Deploy standalone (NOT part of the Pages site). See README.md.
 */
import { DurableObject } from "cloudflare:workers";
import {
  commandFillMatch,
  executionFamilyId,
  mergeCommandResult,
  mergeExecutionFill,
  reconcileCommandFills,
} from "./fill-reconcile.mjs";

const BROKER_NAME = "main";          // single book -> single DO instance
const HEARTBEAT_STALE_MS = 30_000;   // online iff a heartbeat landed within this
const CMD_CAP = 50;                  // recent-command ring size (audit trail)
const SCHEDULED_CMD_CAP = 100;       // long-lived option schedules survive recent-ring churn
const FILLS_RETENTION_DAYS = 14;     // Trade Log trailing window
const FILLS_DAY_CAP = 500;           // per-day row cap (keeps each value < DO 128KiB limit)

function fillStorageKey(fill) {
  return `${String((fill && fill.account_key) || "")}\u0000${executionFamilyId(fill && fill.exec_id)}`;
}

function effectiveFillRows(rows, now) {
  const byExecution = new Map();
  for (const fill of rows || []) {
    if (!fill || !fill.exec_id) continue;
    const key = fillStorageKey(fill);
    byExecution.set(key, mergeExecutionFill(byExecution.get(key), fill, now));
  }
  return [...byExecution.values()]
    .sort((a, b) => {
      const byTime = String(b.time || "").localeCompare(String(a.time || ""));
      if (byTime) return byTime;
      const byIngest = Number(b.ingested_at || 0) - Number(a.ingested_at || 0);
      if (byIngest) return byIngest;
      return String(b.exec_id || "").localeCompare(String(a.exec_id || ""));
    });
}

function boundedFillRows(rows, now) {
  return effectiveFillRows(rows, now).slice(0, FILLS_DAY_CAP);
}

function publicCommand(record) {
  const { envelope, intent_identity, ...visible } = record;
  return visible;
}

export class ExecBroker extends DurableObject {
  _authed(request, token) {
    return typeof token === "string" && token.trim().length > 0 &&
      (request.headers.get("Authorization") || "") === `Bearer ${token}`;
  }

  // Newest socket = most recently accepted or heartbeated (attachment stamps).
  // With >1 socket connected (zombie left by a restart/reconnect), commands go
  // to this one only — never fanned out — so a stale socket can't double-execute.
  _newestSocket(sockets) {
    let best = sockets[0], bestAt = -1;
    for (const s of sockets) {
      let att;
      try { att = s.deserializeAttachment() || {}; } catch { att = {}; }
      const at = Math.max(att.lastSeenAt || 0, att.connectedAt || 0);
      if (at > bestAt) { bestAt = at; best = s; }
    }
    return best;
  }

  async fetch(request) {
    const url = new URL(request.url);

    // --- Agent WebSocket (outbound dial from the trading machine) ---
    if (url.pathname === "/agent") {
      if (!this._authed(request, this.env.AGENT_TOKEN)) return new Response("unauthorized", { status: 401 });
      if (request.headers.get("Upgrade") !== "websocket") return new Response("expected websocket upgrade", { status: 426 });
      const [client, server] = Object.values(new WebSocketPair());
      this.ctx.acceptWebSocket(server);                 // hibernatable accept
      const now = Date.now();
      server.serializeAttachment({ connectedAt: now });
      await this.ctx.storage.put("connected_at", now);
      await this.ctx.storage.put("last_seen", now);
      await this.ctx.storage.delete("disconnected_at");
      return new Response(null, { status: 101, webSocket: client });
    }

    // --- Status read (from the site via the Pages proxy) ---
    if (url.pathname === "/status") {
      if (!this._authed(request, this.env.STATUS_TOKEN)) return new Response("unauthorized", { status: 401 });
      const lastSeen = (await this.ctx.storage.get("last_seen")) || 0;
      const connectedAt = (await this.ctx.storage.get("connected_at")) || null;
      const sockets = this.ctx.getWebSockets().length;
      const now = Date.now();
      const age = lastSeen ? now - lastSeen : null;
      const online = sockets > 0 && age != null && age < HEARTBEAT_STALE_MS;
      return Response.json({
        online, sockets, last_seen: lastSeen || null, connected_at: connectedAt,
        heartbeat_age_ms: age, stale_after_ms: HEARTBEAT_STALE_MS, server_now: now,
      });
    }

    // --- Command in (from the site via the Pages /exec-command proxy) ---
    if (url.pathname === "/command" && request.method === "POST") {
      if (!this._authed(request, this.env.STATUS_TOKEN)) return new Response("unauthorized", { status: 401 });
      let body;
      try { body = await request.json(); } catch { return new Response("bad json", { status: 400 }); }
      const { signed, sig } = body || {};
      if (!signed || !sig) return Response.json({ ok: false, error: "missing signed/sig" }, { status: 400 });
      let cmd;
      try { cmd = JSON.parse(signed); } catch { return Response.json({ ok: false, error: "bad signed payload" }, { status: 400 }); }
      if (!cmd || typeof cmd.id !== "string" || !/^[a-zA-Z0-9-]{1,100}$/.test(cmd.id)) {
        return Response.json({ok:false,error:"invalid command identity"},{status:400});
      }
      const identity = JSON.stringify({type:cmd.type,account:cmd.account,dry_run:cmd.dry_run,payload:cmd.payload});
      const key = `command:${cmd.id}`;
      const recent = (await this.ctx.storage.get("recent_commands")) || [];
      const scheduled = (await this.ctx.storage.get("scheduled_commands")) || [];
      let record = await this.ctx.storage.get(key);
      const prior = record || recent.find(r => r.id === cmd.id) || scheduled.find(r => r.id === cmd.id);
      if (prior && prior.intent_identity && prior.intent_identity !== identity) {
        return Response.json({ok:false,error:"command id belongs to a different intent"},{status:409});
      }
      if (prior && !["queued","delivery_unknown"].includes(prior.state)) {
        return Response.json({ok:true,deduped:true,id:cmd.id,state:prior.state,command:publicCommand(prior)});
      }
      if (prior && !record) {
        // Legacy records cannot prove safe replay. Retain their identity.
        return Response.json({ok:true,deduped:true,id:cmd.id,state:prior.state,command:publicCommand(prior)});
      }
      if (!record) {
        if (!Number.isFinite(cmd.expires_at) || cmd.expires_at <= Date.now()) {
          return Response.json({ok:false,error:"command expired before acceptance"},{status:400});
        }
        record = {id:cmd.id,type:cmd.type,account:cmd.account,dry_run:cmd.dry_run,
          state:"queued",created_at:Date.now(),result:null,intent_identity:identity,
          envelope:{signed,sig},expires_at:cmd.expires_at};
        const fillMatch=commandFillMatch(cmd); if(fillMatch)record.fill_match=fillMatch;
        await this.ctx.storage.put(key,record);
      }
      if (record.expires_at <= Date.now()) {
        return Response.json({ok:false,id:record.id,state:record.state,error:"original delivery window expired; reconcile this intent before a new order"},{status:409});
      }
      const sockets=this.ctx.getWebSockets();
      if (!sockets.length) return Response.json({ok:false,id:record.id,state:record.state,error:"agent offline; intent retained"},{status:503});
      // Persist uncertainty BEFORE attempting the socket write. A crash or throw
      // must not masquerade as delivery. Same-ID retries use the original envelope.
      record.state="delivery_unknown";
      record.delivery_attempts=(record.delivery_attempts||0)+1;
      await this.ctx.storage.put(key,record);
      try {
        this._newestSocket(sockets).send(JSON.stringify({type:"command",...record.envelope}));
        record.state="pushed";
        record.delivery_error=null;
      } catch(e) {
        record.delivery_error=String(e && e.message || e);
      }
      await this.ctx.storage.put(key,record);
      const visible=publicCommand(record);
      const next=[visible,...recent.filter(r=>r.id!==record.id)].slice(0,CMD_CAP);
      await this.ctx.storage.put("recent_commands",next);
      if(record.type==="scheduled_option") {
        await this.ctx.storage.put("scheduled_commands",[visible,...scheduled.filter(r=>r.id!==record.id)].slice(0,SCHEDULED_CMD_CAP));
      }
      const delivered=record.state==="pushed";
      return Response.json({ok:delivered,id:record.id,state:record.state,
        ...(delivered?{}:{error:"delivery uncertain; retry this same intent"})},{status:delivered?200:503});
    }

    // --- Recent commands + results (site polls this) ---
    if (url.pathname === "/commands") {
      if (!this._authed(request, this.env.STATUS_TOKEN)) return new Response("unauthorized", { status: 401 });
      const recent = (await this.ctx.storage.get("recent_commands")) || [];
      const scheduled = (await this.ctx.storage.get("scheduled_commands")) || [];
      const ids = new Set(recent.map((r) => r.id));
      const commands = recent.concat(scheduled.filter((r) => !ids.has(r.id)))
        .sort((a, b) => Number(b.created_at || 0) - Number(a.created_at || 0));
      return Response.json({ commands, server_now: Date.now() });
    }

    // --- Live book (positions / orders / NLV) the site polls ---
    if (url.pathname === "/book") {
      if (!this._authed(request, this.env.STATUS_TOKEN)) return new Response("unauthorized", { status: 401 });
      const book = (await this.ctx.storage.get("book")) || null;
      return Response.json({ book, server_now: Date.now() });
    }

    // --- Accumulated executions (Trade Log tab). IBKR only serves the current
    //     day's fills, so the ring built by _mergeFills IS the history. ---
    if (url.pathname === "/fills") {
      if (!this._authed(request, this.env.STATUS_TOKEN)) return new Response("unauthorized", { status: 401 });
      const fills = await this._retainedFills();
      const receipt = (await this.ctx.storage.get("fill_receipt")) || {accounts:{}};
      const now=Date.now(), accounts={};
      for(const [key,value] of Object.entries(receipt.accounts || {})) {
        const sourceTime=Date.parse(value.source_at || "");
        const fresh=now-Number(value.received_at_ms || 0)<=90000 && Number.isFinite(sourceTime) && now-sourceTime<=90000;
        accounts[key]={...value,complete:value.complete===true && fresh};
        if(!fresh)accounts[key].error="fill receipt stale";
      }
      const legacy = await this._listAll("fill_incomplete:");
      const cutoff=new Date(now-FILLS_RETENTION_DAYS*86400000).toISOString().slice(0,10);
      const incompleteDays=[...legacy.keys()].map(k=>k.slice("fill_incomplete:".length)).filter(d=>d>=cutoff);
      const mergeError=await this.ctx.storage.get("fill_merge_error");
      const complete=Object.keys(accounts).length>0 && Object.values(accounts).every(a=>a.complete) && !incompleteDays.length && !mergeError;
      const reasons=[...(!Object.keys(accounts).length?["no verified fill receipt"]:[]),
        ...Object.entries(accounts).filter(([,a])=>!a.complete).map(([k,a])=>`${k}: ${a.error || "unverified source"}`),
        ...(incompleteDays.length?["legacy fill history may be truncated"]:[]),...(mergeError?[String(mergeError)]:[])];
      return Response.json({fills,retention_days:FILLS_RETENTION_DAYS,server_now:now,
        completeness:{complete,complete_through:complete?receipt.complete_through:null,
          reasons,merge_error:mergeError || null,truncated:incompleteDays.length>0,incomplete_days:incompleteDays,accounts}});

    }

    // --- Option spread query: POST kicks off a read-only chain fetch on the agent ---
    if (url.pathname === "/option" && request.method === "POST") {
      if (!this._authed(request, this.env.STATUS_TOKEN)) return new Response("unauthorized", { status: 401 });
      let body;
      try { body = await request.json(); } catch { return Response.json({ ok: false, error: "bad json" }, { status: 400 }); }
      const ticker = String((body && body.ticker) || "").toUpperCase().trim();
      if (!ticker) return Response.json({ ok: false, error: "ticker required" }, { status: 400 });
      const sockets = this.ctx.getWebSockets();
      if (!sockets.length) return Response.json({ ok: false, error: "agent offline" }, { status: 503 });
      const id = crypto.randomUUID();
      const expiry = (body && body.expiry) || null;
      await this.ctx.storage.put("option_query", { id, ticker, expiry, at: Date.now(), result: null });
      for (const s of sockets) s.send(JSON.stringify({ type: "option_query", id, ticker, expiry }));
      return Response.json({ ok: true, id, ticker });
    }
    if (url.pathname === "/option") {
      if (!this._authed(request, this.env.STATUS_TOKEN)) return new Response("unauthorized", { status: 401 });
      return Response.json({ query: (await this.ctx.storage.get("option_query")) || null, server_now: Date.now() });
    }

    // --- Workbench query: term structure + chain band for the options workbench.
    //     Small ring (not the /option single slot): expiry-change re-queries overlap
    //     the prior poll, so each query keeps its own entry addressed by id. ---
    if (url.pathname === "/workbench" && request.method === "POST") {
      if (!this._authed(request, this.env.STATUS_TOKEN)) return new Response("unauthorized", { status: 401 });
      let body;
      try { body = await request.json(); } catch { return Response.json({ ok: false, error: "bad json" }, { status: 400 }); }
      const ticker = String((body && body.ticker) || "").toUpperCase().trim();
      if (!ticker) return Response.json({ ok: false, error: "ticker required" }, { status: 400 });
      const sockets = this.ctx.getWebSockets();
      if (!sockets.length) return Response.json({ ok: false, error: "agent offline" }, { status: 503 });
      const id = crypto.randomUUID();
      const q = { id, ticker, mode: body.mode || "full", expiry: body.expiry || null,
        max_expiries: body.max_expiries || null, context: body.context || null,
        at: Date.now(), result: null };
      const ring = (await this.ctx.storage.get("workbench_queries")) || [];
      ring.unshift(q);
      await this.ctx.storage.put("workbench_queries", ring.slice(0, 8));
      this._newestSocket(sockets).send(JSON.stringify({ type: "workbench_query", ...q }));
      return Response.json({ ok: true, id, ticker });
    }
    if (url.pathname === "/workbench") {
      if (!this._authed(request, this.env.STATUS_TOKEN)) return new Response("unauthorized", { status: 401 });
      const ring = (await this.ctx.storage.get("workbench_queries")) || [];
      const id = url.searchParams.get("id");
      const query = id ? ring.find((r) => r.id === id) || null : ring[0] || null;
      return Response.json({ query, server_now: Date.now() });
    }

    // --- Futures sizing query: POST kicks off a pure read-only sizing calc on the agent ---
    if (url.pathname === "/futures_size" && request.method === "POST") {
      if (!this._authed(request, this.env.STATUS_TOKEN)) return new Response("unauthorized", { status: 401 });
      let body;
      try { body = await request.json(); } catch { return Response.json({ ok: false, error: "bad json" }, { status: 400 }); }
      const symbol = String((body && body.symbol) || "").toUpperCase().trim();
      if (!symbol) return Response.json({ ok: false, error: "symbol required" }, { status: 400 });
      const sockets = this.ctx.getWebSockets();
      if (!sockets.length) return Response.json({ ok: false, error: "agent offline" }, { status: 503 });
      const id = crypto.randomUUID();
      const q = { id, symbol, entry: body.entry ?? null, stop: body.stop ?? null,
        target: body.target ?? null, risk: body.risk ?? null, risk_pct: body.risk_pct ?? null,
        account_key: body.account_key || "primary", account_value: body.account_value ?? null,
        at: Date.now(), result: null };
      await this.ctx.storage.put("futures_size", q);
      for (const s of sockets) s.send(JSON.stringify({ type: "futures_size", ...q }));
      return Response.json({ ok: true, id, symbol });
    }
    if (url.pathname === "/futures_size") {
      if (!this._authed(request, this.env.STATUS_TOKEN)) return new Response("unauthorized", { status: 401 });
      return Response.json({ query: (await this.ctx.storage.get("futures_size")) || null, server_now: Date.now() });
    }

    // --- Futures front-month resolve: POST kicks off a read-only reqContractDetails on the agent ---
    if (url.pathname === "/futures_front" && request.method === "POST") {
      if (!this._authed(request, this.env.STATUS_TOKEN)) return new Response("unauthorized", { status: 401 });
      let body;
      try { body = await request.json(); } catch { return Response.json({ ok: false, error: "bad json" }, { status: 400 }); }
      const symbol = String((body && body.symbol) || "").toUpperCase().trim();
      if (!symbol) return Response.json({ ok: false, error: "symbol required" }, { status: 400 });
      const exchange = String((body && body.exchange) || "").toUpperCase().trim();
      if (exchange && !["CME", "CBOT", "NYMEX", "COMEX"].includes(exchange)) {
        return Response.json({ ok: false, error: "exchange must be CME, CBOT, NYMEX, or COMEX" }, { status: 400 });
      }
      const sockets = this.ctx.getWebSockets();
      if (!sockets.length) return Response.json({ ok: false, error: "agent offline" }, { status: 503 });
      const id = crypto.randomUUID();
      await this.ctx.storage.put("futures_front", { id, symbol, exchange: exchange || null, at: Date.now(), result: null });
      for (const s of sockets) s.send(JSON.stringify({ type: "futures_front", id, symbol, exchange: exchange || null }));
      return Response.json({ ok: true, id, symbol, exchange: exchange || null });
    }
    if (url.pathname === "/futures_front") {
      if (!this._authed(request, this.env.STATUS_TOKEN)) return new Response("unauthorized", { status: 401 });
      return Response.json({ query: (await this.ctx.storage.get("futures_front")) || null, server_now: Date.now() });
    }

    return new Response("not found", { status: 404 });
  }

  async webSocketMessage(ws, message) {
    let msg;
    try { msg = JSON.parse(typeof message === "string" ? message : ""); }
    catch { msg = { type: "raw" }; }

    if (msg.type === "hello" || msg.type === "heartbeat") {
      await this.ctx.storage.put("last_seen", Date.now());
      // stamp the socket so _newestSocket can prefer the live one over a zombie
      try { ws.serializeAttachment({ ...(ws.deserializeAttachment() || {}), lastSeenAt: Date.now() }); }
      catch (_) { /* best effort — connectedAt still breaks the tie */ }
      ws.send(JSON.stringify({ type: "ack", of: msg.type, server_now: Date.now() }));
      return;
    }

    // Live read-only book snapshot from the agent (positions / orders / NLV
    // + today's fills). Fills are folded into their own per-day ring and
    // stripped from the stored book (keeps the "book" value small).
    if (msg.type === "book") {
      const book = { ...(msg.book || {}), at: msg.at || Date.now() };
      const accounts = (book.accounts || []).map(({ fills, ...rest }) => rest);
      await this.ctx.storage.put("book", { ...book, accounts });
      await this.ctx.storage.put("last_seen", Date.now());
      try { await this._mergeFills(book); }
      catch (e) {
        const error=`mergeFills: ${String((e && e.message) || e)}`;
        await this.ctx.storage.put("last_error",error);
        await this.ctx.storage.put("fill_merge_error",error);
      }
      return;
    }

    // Option-spread result from the agent -> attach to the pending query.
    if (msg.type === "option_result" && msg.id) {
      const q = await this.ctx.storage.get("option_query");
      if (q && q.id === msg.id) {
        q.result = msg.data; q.result_at = Date.now();
        await this.ctx.storage.put("option_query", q);
      }
      return;
    }

    // Workbench result from the agent -> attach to its ring entry by id.
    if (msg.type === "workbench_result" && msg.id) {
      const ring = (await this.ctx.storage.get("workbench_queries")) || [];
      const i = ring.findIndex((r) => r.id === msg.id);
      if (i >= 0) {
        ring[i].result = msg.data; ring[i].result_at = Date.now();
        await this.ctx.storage.put("workbench_queries", ring);
      }
      return;
    }

    // Futures-sizing result from the agent -> attach to the pending query.
    if (msg.type === "futures_result" && msg.id) {
      const q = await this.ctx.storage.get("futures_size");
      if (q && q.id === msg.id) {
        q.result = msg.data; q.result_at = Date.now();
        await this.ctx.storage.put("futures_size", q);
      }
      return;
    }

    // Futures front-month result from the agent -> attach to the pending query.
    if (msg.type === "futures_front_result" && msg.id) {
      const q = await this.ctx.storage.get("futures_front");
      if (q && q.id === msg.id) {
        q.result = msg.data; q.result_at = Date.now();
        await this.ctx.storage.put("futures_front", q);
      }
      return;
    }

    // Command result from the agent -> attach to the recent-commands ring.
    if (msg.type === "result" && msg.id) {
      const durable=await this.ctx.storage.get(`command:${msg.id}`);
      if(durable) {
        durable.state=msg.state || "done";
        durable.result=mergeCommandResult(durable.result,{ok:msg.ok,detail:msg.detail,validation:msg.validation,preview:msg.preview,fill:msg.fill,at:msg.at});
        await this.ctx.storage.put(`command:${msg.id}`,durable);
      }

      const recent = (await this.ctx.storage.get("recent_commands")) || [];
      const i = recent.findIndex((r) => r.id === msg.id);
      if (i >= 0) {
        recent[i].state = msg.state || "done";
        recent[i].result = mergeCommandResult(recent[i].result, {
          ok: msg.ok, detail: msg.detail, validation: msg.validation,
          preview: msg.preview, fill: msg.fill, at: msg.at,
        });
        await this.ctx.storage.put("recent_commands", recent);
      }
      const scheduled = (await this.ctx.storage.get("scheduled_commands")) || [];
      const si = scheduled.findIndex((r) => r.id === msg.id);
      if (si >= 0) {
        scheduled[si].state = msg.state || "done";
        scheduled[si].result = mergeCommandResult(scheduled[si].result, {
          ok: msg.ok, detail: msg.detail, validation: msg.validation,
          preview: msg.preview, fill: msg.fill, at: msg.at,
        });
        await this.ctx.storage.put("scheduled_commands", scheduled);
      }
    }
  }

  // Fold a book push's per-account fills into per-day storage keys
  // ("fills:YYYY-MM-DD", UTC day of the fill time). Upsert by IBKR execution
  // family so a corrected .02 execution supersedes its original .01 row. The
  // agent re-pushes the same day's fills every cycle, and commission reports
  // lag the execution by a beat, so later pushes fill in commission/PnL.
  // Day keys older than the retention window are pruned on every merge.
  async _listAll(prefix) {
    const rows=new Map(); let startAfter;
    for(;;) {
      const page=await this.ctx.storage.list({prefix,limit:1000,...(startAfter?{startAfter}:{})});
      for(const [key,value] of page)rows.set(key,value);
      if(page.size<1000)break;
      const last=[...page.keys()].at(-1);
      if(last===startAfter)throw Error("storage cursor did not advance");
      startAfter=last;
    }
    return rows;
  }

  async _retainedFills() {
    const now=Date.now(), cutoff=new Date(now-FILLS_RETENTION_DAYS*86400000).toISOString().slice(0,10);
    const legacy=await this._listAll("fills:");
    const archive=await this._listAll("fill_row:");
    const rows=[];
    for(const [key,value] of legacy)if(key.slice(6)>=cutoff && Array.isArray(value))rows.push(...value);
    for(const [key,value] of archive)if(key.slice(9,19)>=cutoff)rows.push(value);
    return effectiveFillRows(rows,now);
  }

  async _mergeFills(book) {
    const incoming = [];
    for (const acc of (book && book.accounts) || []) {
      for (const f of acc.fills || []) {
        if (f && f.exec_id) incoming.push({ ...f, account_key: acc.key, account_label: acc.label });
      }
    }
    const now = Date.now();
    const byDay = new Map();
    for (const f of incoming) {
      const t = Date.parse(f.time);
      const day = new Date(Number.isFinite(t) ? t : now).toISOString().slice(0, 10);
      if (!byDay.has(day)) byDay.set(day, []);
      byDay.get(day).push(f);
    }
    for (const [day, dayFills] of byDay) {
      const key = `fills:${day}`;
      const ring = (await this.ctx.storage.get(key)) || [];
      // Keep the newest executions when an unusually busy day exceeds the
      // storage cap. Map insertion order would otherwise discard every later
      // fill once the first 500 rows had been retained.
      const archivedDay=await this.ctx.storage.get(`fill_archive_day:${day}`);
      if(!archivedDay && ring.length>=FILLS_DAY_CAP) {
        await this.ctx.storage.put(`fill_incomplete:${day}`,{reason:"legacy capped history"});
      }
      // Each effective execution has its own small value; the old ring is only
      // a compatibility cache and never the complete history after migration.
      for(const f of effectiveFillRows([...ring,...dayFills],now)) {
        const rowKey=`fill_row:${day}:${encodeURIComponent(fillStorageKey(f))}`;
        const prior=await this.ctx.storage.get(rowKey);
        const next=mergeExecutionFill(prior,f,now);
        if(JSON.stringify(prior)!==JSON.stringify(next)) await this.ctx.storage.put(rowKey,next);
      }
      await this.ctx.storage.put(`fill_archive_day:${day}`,true);
      await this.ctx.storage.put(key, boundedFillRows([...ring, ...dayFills], now));
    }
    const cutoffDay = new Date(now - FILLS_RETENTION_DAYS * 86_400_000).toISOString().slice(0, 10);
    // The per-execution archive follows the same existing 14-day retention
    // contract as the legacy ring; long history belongs in the durable harvester.
    for (const prefix of ["fill_row:", "fill_archive_day:", "fill_incomplete:"]) {
      for (const [key] of await this._listAll(prefix)) {
        if (key.slice(prefix.length, prefix.length + 10) < cutoffDay) await this.ctx.storage.delete(key);
      }
    }
    const days = await this.ctx.storage.list({ prefix: "fills:" });
    const retained = [];
    for (const [key, fills] of days) {
      if (key.slice("fills:".length) < cutoffDay) await this.ctx.storage.delete(key);
      else if (Array.isArray(fills)) {
        const day=key.slice("fills:".length);
        if(fills.length>=FILLS_DAY_CAP && !await this.ctx.storage.get(`fill_archive_day:${day}`)) {
          await this.ctx.storage.put(`fill_incomplete:${day}`,{reason:"legacy capped history"});
        }
        // Migrate any pre-deploy .01/.02 duplicates even when that historical
        // day is no longer present in the agent's current-day snapshot.
        const identities = fills.map(fillStorageKey);
        if (new Set(identities).size !== identities.length) {
          const collapsed = boundedFillRows(fills, now);
          await this.ctx.storage.put(key, collapsed);
          retained.push(...collapsed);
        } else {
          retained.push(...fills);
        }
      }
    }
    await this._reconcileCommandFills(await this._retainedFills(), now);
    const accounts={};
    for(const acc of book.accounts || []) {
      const inputTime=acc.fills_source_at || book.at || 0;
      const raw=Number.isFinite(Number(inputTime))?Number(inputTime):Date.parse(inputTime);
      const sourceMs=raw>0 && raw<1e12?raw*1000:raw;
      const sourceFresh=Number.isFinite(sourceMs)&&sourceMs>0&&now-sourceMs<=90000&&sourceMs<=now+5000;
      const complete=acc.fills_complete===true && !acc.error && !acc.fills_error && sourceFresh;
      accounts[acc.key]={complete,received_at:new Date(now).toISOString(),received_at_ms:now,
        source_at:sourceFresh?new Date(sourceMs).toISOString():null,
        complete_through:complete?new Date(sourceMs).toISOString():null,
        error:acc.fills_error || acc.error || (!sourceFresh?"source timestamp unavailable/stale":!complete?"source completeness unverified":null)};
    }
    const verifiedTimes=Object.values(accounts).filter(a=>a.complete).map(a=>Date.parse(a.source_at));
    await this.ctx.storage.put("fill_receipt",{accounts,complete_through:verifiedTimes.length?new Date(Math.min(...verifiedTimes)).toISOString():null});
    await this.ctx.storage.put("fill_merge_error",null);
  }

  // Resting orders often return to the Activity table as Submitted, before
  // IBKR has an execution price. Reconcile both audit rings from later book
  // snapshots so the same command row fills itself in after the fact.
  async _reconcileCommandFills(incoming, now) {
    for (const key of ["recent_commands", "scheduled_commands"]) {
      const ring = (await this.ctx.storage.get(key)) || [];
      const reconciled = reconcileCommandFills(ring, incoming, now);
      if (reconciled.changed) {
        await this.ctx.storage.put(key, reconciled.commands);
        for (const command of reconciled.commands) {
          const durableKey=`command:${command.id}`;
          const durable=await this.ctx.storage.get(durableKey);
          if(durable && (durable.state!==command.state || JSON.stringify(durable.result)!==JSON.stringify(command.result))) {
            await this.ctx.storage.put(durableKey,{...durable,state:command.state,result:command.result});
          }
        }
      }
    }
  }

  async webSocketClose(ws, code, reason, wasClean) {
    await this.ctx.storage.put("disconnected_at", Date.now());
    try { ws.close(code, reason); } catch (_) { /* already closing */ }
  }

  async webSocketError(ws, err) {
    await this.ctx.storage.put("last_error", String((err && err.message) || err));
  }
}

const DO_PATHS = new Set(["/agent", "/status", "/command", "/commands", "/book", "/fills", "/option", "/workbench", "/futures_size", "/futures_front"]);

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/" || url.pathname === "/health") {
      return new Response("execution-broker ok\n", { status: 200 });
    }
    if (DO_PATHS.has(url.pathname)) {
      const id = env.EXEC_BROKER.idFromName(BROKER_NAME);
      return env.EXEC_BROKER.get(id).fetch(request);
    }
    return new Response("not found", { status: 404 });
  },
};
