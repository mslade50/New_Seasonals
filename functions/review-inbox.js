/* Human review only. This route has no broker/Sheets/runner dependency. */
import { requireAccessIdentity } from './_access.js';
import { verify, view, decide } from '../site/assets/review-core.js';

const PREFIX = 'review_inbox/v1';
const headers = {'Content-Type':'application/json','Cache-Control':'no-store','X-Content-Type-Options':'nosniff'};
const reply = (status, value) => new Response(JSON.stringify(value), {status,headers});
export function etDate(now) {
  return new Intl.DateTimeFormat('en-CA',{timeZone:'America/New_York',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date(now));
}
function validDate(date) {
  return /^\d{4}-\d{2}-\d{2}$/.test(date) && !Number.isNaN(Date.parse(date)) && new Date(date).toISOString().slice(0,10) === date;
}
export async function readRecord(bucket, product, date) {
  const object = await bucket.get(`${PREFIX}/${product}/${date}.json`);
  if (!object) return null;
  if (object.size > 2_000_000) throw Error('Review history exceeds record limit');
  const record = await object.json();
  if (record.schema !== 'review-inbox.v1' || record.product !== product || record.date !== date || !Array.isArray(record.current_ids) || !Array.isArray(record.events) || !record.proposals || !record.delivery?.delivery_id) throw Error('Invalid review ledger');
  if (Object.keys(record.proposals).length > 128 || record.events.length > 128) throw Error('Review history exceeds version limit');
  for (const [id, e] of Object.entries(record.proposals)) {
    await verify(e);
    if (e.id !== id || e.payload.product !== product || e.payload.source_date !== date) throw Error('Proposal identity mismatch');
  }
  if (record.current_ids.some(id=>!record.proposals[id])) throw Error('Current version missing');
  for (const event of record.events) {
    if (!record.proposals[event.proposal_id] || event.proposal_hash !== record.proposals[event.proposal_id].hash || event.execution !== 'not_submitted' || event.scope !== 'human_review_only' || !['approve_review','reject'].includes(event.decision)) throw Error('Invalid review history');
  }
  const receiptObject=await bucket.get(`${product==='pitch'?'pitch_delivery_receipts':'seasonal_agent_delivery_receipts'}/${date}.json`);
  const receipt=receiptObject?await receiptObject.json():null;
  const deliveryCurrent=receipt?.status==='sent' && receipt.date===date && receipt.delivery_id===record.delivery.delivery_id;
  return {record,etag:object.etag,deliveryCurrent};
}
export async function handleReview(request, env, identity, now = () => new Date().toISOString()) {
  if (!env.CHARTS) return reply(503,{error:'Existing CHARTS store is unavailable'});
  const url = new URL(request.url), today=etDate(now()), date=url.searchParams.get('date') || today;
  if (!validDate(date) || date > today) return reply(400,{error:'Valid current or historical review date required'});
  if (request.method === 'GET') {
    try {
      const results=await Promise.all(['pitch','seasonal'].map(async product=>{
        const loaded=await readRecord(env.CHARTS,product,date);
        if(!loaded) return {product,date,status:'missing',proposals:[],events:[],error:'No delivery-gated review feed published for this date'};
        const {record,deliveryCurrent}=loaded;
        const proposals=Object.values(record.proposals).map(envelope=>({envelope,current:deliveryCurrent&&record.current_ids.includes(envelope.id),state:view(envelope,record.events,now())}));
        return {product,date,status:!deliveryCurrent?'stale':date===today?'fresh':'historical',stand_down:record.stand_down,stand_down_reason:record.stand_down_reason,delivery:record.delivery,proposals,events:record.events};
      }));
      return reply(200,{schema:1,date,today,server_time:now(),actor:identity.subject,read_only:date!==today,products:results});
    } catch { return reply(503,{error:'Review store cannot be verified. Decisions are unavailable.'}); }
  }
  if(request.method!=='POST') return reply(405,{error:'GET or POST required'});
  if(request.headers.get('Origin')!==url.origin || request.headers.get('Content-Type')?.split(';')[0].trim()!=='application/json') return reply(403,{error:'Same-origin JSON decision required'});
  if(date!==today) return reply(409,{error:'History is read-only; return to today'});
  let command;
  try {
    const text=await request.text();
    if(text.length>8000) return reply(413,{error:'Decision too large'});
    command=JSON.parse(text);
    if(!command || !['pitch','seasonal'].includes(command.product)) return reply(400,{error:'Explicit product required'});
  } catch {return reply(400,{error:'Invalid decision JSON'});}
  try {
    for(let attempt=0;attempt<3;attempt++) {
      const loaded=await readRecord(env.CHARTS,command.product,date);
      if(!loaded) return reply(404,{error:'No confirmed proposal published'});
      const {record,etag,deliveryCurrent}=loaded, envelope=record.proposals[command.proposal_id];
      if(!envelope) return reply(409,{error:'Proposal changed or is unavailable. Reload.'});
      const retry=record.events.some(e=>e.id===command.id);
      if(!retry&&!deliveryCurrent) return reply(409,{error:'Current delivery has not been reconciled. Reload after the confirmed feed publishes.'});
      if(!retry && !record.current_ids.includes(envelope.id)) return reply(409,{error:'Proposal superseded. Reload and review the new version.'});
      const result=await decide(envelope,record.events,command,{actor:identity.subject,now:now()});
      if(result.replay) return reply(200,{event:result.event,replay:true,execution:'not_submitted'});
      if(etDate(now())!==date) return reply(409,{error:'Review date changed; reload'});
      if(Date.parse(now())>=Date.parse(envelope.payload.review_deadline)) return reply(409,{error:'Proposal expired before the write; reload'});
      const stored=await env.CHARTS.put(`${PREFIX}/${command.product}/${date}.json`,JSON.stringify({...record,events:result.events}),{onlyIf:{etagMatches:etag},httpMetadata:{contentType:'application/json',cacheControl:'no-store'}});
      if(stored) return reply(200,{event:result.event,replay:false,execution:'not_submitted'});
      // CAS lost: re-read and resolve identical retry, stale revision or supersession.
    }
    return reply(409,{error:'Review changed concurrently. Reload; no success was recorded.'});
  } catch(e) {
    if(/required|invalid|missing|no legs|changed|stale|conflict|pending|expired|approved_review|rejected|reason|confirmed|superseded/i.test(e.message)) return reply(409,{error:e.message});
    return reply(503,{error:'Review write could not be confirmed. Reload before retrying.'});
  }
}
export async function onRequest({request,env}) {
  const identity=await requireAccessIdentity(request,env);
  if(identity instanceof Response) return identity;
  return handleReview(request,env,identity);
}
