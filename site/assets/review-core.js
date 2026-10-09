/* Pure decision ledger; the endpoint atomically queues authorized staging. */
const root = globalThis;
  'use strict';
  const canonical = value => {
    if (Array.isArray(value)) return '[' + value.map(canonical).join(',') + ']';
    if (value && typeof value === 'object') return '{' + Object.keys(value).sort().map(k => JSON.stringify(k) + ':' + canonical(value[k])).join(',') + '}';
    if (value === undefined || (typeof value === 'number' && !Number.isFinite(value))) throw Error('Non-JSON value');
    return JSON.stringify(value);
  };
  const clone = x => JSON.parse(canonical(x));
  const freeze = x => { if (x && typeof x === 'object') { Object.values(x).forEach(freeze); Object.freeze(x); } return x; };
  const instant = x => typeof x === 'string' && /^\d{4}-\d{2}-\d{2}T.*(?:Z|[+-]\d{2}:\d{2})$/.test(x) && Number.isFinite(Date.parse(x));
  async function digest(x) {
    const bytes = new TextEncoder().encode(canonical(x));
    return [...new Uint8Array(await root.crypto.subtle.digest('SHA-256', bytes))].map(b => b.toString(16).padStart(2, '0')).join('');
  }
  function validate(p) {
    if (!p || p.schema !== 1 || !['pitch', 'seasonal'].includes(p.product)) throw Error('Unknown proposal schema/product');
    if (!/^\d{4}-\d{2}-\d{2}$/.test(p.source_date) || !p.source_idea_id || !p.title || !p.thesis || !p.account) throw Error('Missing proposal identity/details/account');
    if (!instant(p.published_at) || !instant(p.review_deadline) || (!p.manual_only && !instant(p.execution_deadline))) throw Error('Explicit timezone deadlines required');
    if (p.execution_deadline && Date.parse(p.review_deadline) > Date.parse(p.execution_deadline)) throw Error('Invalid deadline ordering');
    if (!Array.isArray(p.orders) || !p.orders.length) throw Error('No legs: publish stand-down separately');
    const legs = new Set();
    for (const o of p.orders) {
      if (o.Idea_Id !== p.source_idea_id || !o.Leg || legs.has(o.Leg)) throw Error('Invalid/duplicate leg identity');
      legs.add(o.Leg);
      if (!o.Ticker || !o.Sec_Type || !['BUY', 'SELL_SHORT'].includes(o.Action) || !o.Entry_Type || !o.Order_Type || !o.TIF || !o.Execute_On || !o.Time_Exit_Date || !o.Time_Exit_Order || !(Number(o.Quantity) > 0)) throw Error('Incomplete trade details');
    }
  }
  async function seal(raw) {
    validate(raw);
    const payload = clone(raw), hash = await digest(payload);
    return freeze({ id: `${payload.product}:${payload.source_idea_id}:${hash.slice(0, 16)}`, hash, canonical: canonical(payload), payload });
  }
  async function verify(envelope) {
    validate(envelope.payload);
    if (typeof envelope.canonical !== 'string' || canonical(JSON.parse(envelope.canonical)) !== canonical(envelope.payload)) throw Error('Proposal content changed');
    const bytes = new TextEncoder().encode(envelope.canonical);
    const hash = [...new Uint8Array(await root.crypto.subtle.digest('SHA-256', bytes))].map(b => b.toString(16).padStart(2, '0')).join('');
    const p = envelope.payload;
    if (envelope.hash !== hash || envelope.id !== `${p.product}:${p.source_idea_id}:${hash.slice(0,16)}`) throw Error('Proposal content changed');
    return freeze(clone(envelope));
  }
  function view(envelope, events, now) {
    if (!instant(now)) throw Error('Invalid trusted clock');
    const rows = events.filter(e => e.proposal_id === envelope.id);
    const last = rows.at(-1);
    const expired = Date.parse(now) >= Date.parse(envelope.payload.review_deadline);
    return {
      revision: rows.length,
      status: last ? last.decision === 'approve_review' ? 'approved_review' : 'rejected' : expired ? 'expired' : 'pending',
      execution_window_closed: !envelope.payload.execution_deadline || Date.parse(now) >= Date.parse(envelope.payload.execution_deadline),
      review_window_closed: expired,
      event: last || null,
    };
  }
  async function decide(envelope, events, command, context) {
    await verify(envelope);
    if (!context.actor || !instant(context.now)) throw Error('Trusted actor/clock required');
    if (!['approve_review', 'reject'].includes(command.decision) || command.confirmed !== true) throw Error('Explicit confirmed decision required');
    if (!/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(command.id || '')) throw Error('Idempotency UUID required');
    if (command.proposal_id !== envelope.id || command.proposal_hash !== envelope.hash) throw Error('Stale proposal: reload');
    if (typeof command.reason !== 'string' || command.reason.length > 1000 || (command.decision === 'reject' && !command.reason.trim())) throw Error('Rejection needs a reason; maximum 1000 characters');
    const request_hash = await digest({ ...command, actor: context.actor });
    const replay = events.find(e => e.id === command.id);
    if (replay) {
      if (replay.request_hash !== request_hash) throw Error('Idempotency conflict');
      return { events, event: replay, replay: true };
    }
    const state = view(envelope, events, context.now);
    if (command.expected_revision !== state.revision) throw Error('Review changed: reload');
    if (state.status !== 'pending') throw Error(`Proposal is ${state.status}; no new decision allowed`);
    const stage = command.decision === 'approve_review' && command.stage === true;
    const event = freeze({ id: command.id, proposal_id: envelope.id, proposal_hash: envelope.hash,
      revision: state.revision + 1, decision: command.decision, reason: command.reason.trim(),
      actor: context.actor, at: context.now, request_hash, scope: stage ? 'review_and_stage' : 'human_review_only', execution: stage ? 'queued' : 'not_submitted',
      ...(stage ? {accounts: clone(envelope.payload.execution_accounts)} : {}) });
    return { events: [...events, event], event, replay: false };
  }
export { canonical, seal, verify, view, decide };
