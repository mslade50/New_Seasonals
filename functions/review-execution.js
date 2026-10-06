/* Disabled-by-default whole-idea bridge. Review approval never calls this route.
 * Reuses Access, CHARTS CAS, STATUS_TOKEN HMAC and the existing broker/agent.
 * Every wire intent is durably reserved before delivery. No automatic POST retry.
 */
import { requireAccessIdentity } from './_access.js';
import { readRecord, etDate } from './review-inbox.js';
import { canonical, view } from '../site/assets/review-core.js';

const UUID=/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const headers={'Content-Type':'application/json','Cache-Control':'no-store','X-Content-Type-Options':'nosniff'};
const reply=(status,value)=>new Response(JSON.stringify(value),{status,headers});
export function settings(env){return {preview_enabled:env.REVIEW_EXECUTION_PREVIEW_ENABLED==='1',live_enabled:env.REVIEW_EXECUTION_LIVE_ENABLED==='1',accounts:{pitch:env.REVIEW_EXECUTION_PITCH_ACCOUNT||'primary',seasonal:env.REVIEW_EXECUTION_SEASONAL_ACCOUNT||null}};}
async function hash(text){return [...new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(text)))].map(x=>x.toString(16).padStart(2,'0')).join('');}
export async function verifyPlan(plan){if(!plan||typeof plan.canonical!=='string'||canonical(JSON.parse(plan.canonical))!==canonical(plan.payload)||await hash(plan.canonical)!==plan.hash||plan.payload?.schema!=='review-execution-plan.v1')throw Error('Execution preview cannot be verified');return plan.payload;}
async function broker(env,path,command){
 if(!env.EXEC_BROKER_URL||!env.STATUS_TOKEN)throw Error('Existing broker configuration unavailable');
 const url=env.EXEC_BROKER_URL.replace(/\/$/,'')+path;
 const options={headers:{Authorization:`Bearer ${env.STATUS_TOKEN}`},redirect:'error'};
 if(command){const signed=JSON.stringify(command),key=await crypto.subtle.importKey('raw',new TextEncoder().encode(env.STATUS_TOKEN),{name:'HMAC',hash:'SHA-256'},false,['sign']);const sig=[...new Uint8Array(await crypto.subtle.sign('HMAC',key,new TextEncoder().encode(signed)))].map(x=>x.toString(16).padStart(2,'0')).join('');options.method='POST';options.headers['Content-Type']='application/json';options.body=JSON.stringify({signed,sig});}
 const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),12000);options.signal=controller.signal;
 try{const response=await fetch(url,options);const data=await response.json();if(!response.ok&&!command)throw Error('Broker status unavailable');return data;}finally{clearTimeout(timer);}
}
async function persist(bucket,key,loaded,record){return bucket.put(key,JSON.stringify(record),{onlyIf:{etagMatches:loaded.etag},httpMetadata:{contentType:'application/json',cacheControl:'no-store'}});}

export async function handleExecution(request,env,identity,now=()=>new Date().toISOString(),transport=broker){
 const cfg=settings(env),url=new URL(request.url),today=etDate(now());
 if(request.method==='GET'&&!url.searchParams.has('id'))return reply(200,{...cfg,mode:cfg.live_enabled?'Site execution gate enabled; local gates still decide':cfg.preview_enabled?'Preview only':'Disabled until user handoff',scope:'whole_idea',broker_activation_verified:false});
 if(!env.CHARTS)return reply(503,{error:'Existing review store unavailable'});
 if(!cfg.preview_enabled)return reply(503,{error:'Execution adapter disabled; user runtime/configuration handoff required',configuration:cfg});
 try{
  if(request.method==='GET'){
   const product=url.searchParams.get('product'),date=url.searchParams.get('date'),id=url.searchParams.get('id');
   if(!['pitch','seasonal'].includes(product)||!/^\d{4}-\d{2}-\d{2}$/.test(date||'')||!UUID.test(id||''))return reply(400,{error:'Explicit product/date/request UUID required'});
   const key=`review_inbox/v1/${product}/${date}.json`;
   for(let attempt=0;attempt<3;attempt++){
    const loaded=await readRecord(env.CHARTS,product,date),saved=loaded?.record.execution_requests?.[id];
    if(!saved)return reply(404,{error:'Reserved execution request not found'});
    if(saved.actor!==identity.subject)return reply(403,{error:'Execution request actor mismatch'});
    const data=await transport(env,'/commands'),found=(data.commands||[]).find(c=>c.id===id);
    if(!found)return reply(200,{id,state:saved.result?.state||'unknown',result:saved.result||null,stale:true,detail:'Broker result outside available history; read-only reconciliation is required'});
    if(found.type!=='review_execution'||found.account!==saved.command.account)return reply(503,{error:'Broker command identity mismatch'});
    const result={state:found.state,...found.result};
    if(result.preview){const plan=await verifyPlan(result.preview);if(plan.actor!==identity.subject||plan.proposal_hash!==saved.command.payload.proposal.hash||plan.account!==saved.command.account)return reply(503,{error:'Broker preview identity mismatch'});}
    const record={...loaded.record,execution_requests:{...loaded.record.execution_requests,[id]:{...saved,result,checked_at:now()}}};
    if(await persist(env.CHARTS,key,loaded,record))return reply(200,{id,...result,stale:false});
   }
   return reply(409,{error:'Execution history changed concurrently; check status again'});
  }
  if(request.method!=='POST')return reply(405,{error:'GET or POST required'});
  if(request.headers.get('Origin')!==url.origin||request.headers.get('Content-Type')?.split(';')[0].trim()!=='application/json')return reply(403,{error:'Same-origin JSON required'});
  const text=await request.text();if(text.length>12000)return reply(413,{error:'Request too large'});
  const body=JSON.parse(text),product=body.product,date=body.date,operation=body.operation;
  if(!['pitch','seasonal'].includes(product)||!['preview','execute','reconcile'].includes(operation)||!UUID.test(body.id||'')||!/^\d{4}-\d{2}-\d{2}$/.test(date||''))return reply(400,{error:'Explicit operation/product/date/id required'});
  const account=cfg.accounts[product];
  if(!['primary','pa'].includes(account)||body.account!==account)return reply(409,{error:'Execution account unassigned or does not match the configured product account'});
  if(operation==='execute'&&!cfg.live_enabled)return reply(503,{error:'Live execution adapter disabled; no broker request sent'});
  const key=`review_inbox/v1/${product}/${date}.json`;
  let saved;
  for(let attempt=0;attempt<3;attempt++){
   const loaded=await readRecord(env.CHARTS,product,date);if(!loaded)return reply(404,{error:'Review record unavailable'});
   const record=loaded.record;
   const requestHash=await hash(canonical({...body,actor:identity.subject}));
   const previous=record.execution_requests?.[body.id];
   if(previous){if(previous.request_hash!==requestHash||previous.actor!==identity.subject)return reply(409,{error:'Idempotency conflict'});saved=previous;break;}
   if(Object.keys(record.execution_requests||{}).length>=32)return reply(409,{error:'Execution audit limit reached; history cannot be truncated'});
   let payload,claim;
   if(operation==='reconcile'){
    const old=Object.values(record.execution_requests||{}).find(r=>r.command.payload.operation==='execute'&&r.command.payload.plan_hash===body.plan_hash&&r.actor===identity.subject);
    if(!old)return reply(409,{error:'Reserved execution intent unavailable; no run identity inferred'});
    const runKey=await hash(canonical([product,old.command.payload.proposal.payload.source_idea_id,account]));
    payload={operation,product,actor:identity.subject,run_key:runKey};
   }else{
    const envelope=record.proposals[body.proposal_id],state=envelope&&view(envelope,record.events,now());
    if(date!==today||!loaded.deliveryCurrent||!record.current_ids.includes(body.proposal_id)||envelope?.hash!==body.proposal_hash||state?.status!=='approved_review'||state.review_window_closed)return reply(409,{error:'Current matching unexpired approved proposal required'});
    payload={operation,product,actor:identity.subject,proposal:envelope,review:state.event,delivery_id:record.delivery.delivery_id,current_at_authorization:true};
    if(operation==='execute'){
     const prior=record.execution_requests?.[body.preview_id];
     if(!prior||prior.actor!==identity.subject||prior.command.payload.operation!=='preview'||prior.command.payload.proposal.hash!==envelope.hash||prior.result?.state!=='preview'||!prior.result.preview)return reply(409,{error:'Fresh verified broker preview required'});
     const plan=await verifyPlan(prior.result.preview);
     if(prior.result.preview.hash!==body.plan_hash||plan.account!==account||plan.proposal_hash!==envelope.hash||plan.actor!==identity.subject||Date.parse(plan.expires_at)<=Date.parse(now())||plan.review_event_id!==state.event.id||plan.delivery_id!==record.delivery.delivery_id)return reply(409,{error:'Preview changed/expired/account mismatch; preview and confirm again'});
     if(body.confirmed!==true||plan.non_atomic&&body.non_atomic_ack!==true||plan.risk_ack_required&&body.risk_ack!==true)return reply(409,{error:'Explicit whole-idea, non-atomic and risk confirmations required'});
     payload={...payload,confirmed:true,plan_hash:body.plan_hash,non_atomic_ack:body.non_atomic_ack===true,risk_ack:body.risk_ack===true};
     claim=canonical([product,envelope.payload.source_idea_id,account]);
     if(record.execution_claims?.[claim])return reply(409,{error:'Permanent idea/account execution claim exists; check status and reconcile, never create a second order'});
    }
   }
   const stamp=Date.parse(now());
   const command={id:body.id,type:'review_execution',account,dry_run:operation!=='execute',payload,created_at:stamp,expires_at:stamp+60000};
   saved={actor:identity.subject,request_hash:requestHash,command,state:'delivery_unknown',created_at:now()};
   const next={...record,execution_requests:{...record.execution_requests,[body.id]:saved},execution_claims:{...record.execution_claims,...(claim?{[claim]:body.id}:{})}};
   if(await persist(env.CHARTS,key,loaded,next))break;
   saved=null;
  }
  if(!saved)return reply(409,{error:'Review/execution changed concurrently; reload; nothing sent'});
  if(saved.command.expires_at<=Date.parse(now()))return reply(409,{id:body.id,state:'expired',error:'Original command delivery window expired; reconcile before another intent'});
  // The complete persisted envelope and UUID are identical on an explicit retry.
  // A failed HTTP response NEVER clears the permanent claim or mints another ID.
  const delivered=await transport(env,'/command',saved.command);
  if(delivered.id && delivered.id!==body.id)throw Error('Broker delivery identity mismatch');
  return reply(202,{id:body.id,state:delivered.state||'delivery_unknown',execution:'broker_acknowledgment_pending',detail:'Delivery is not order acknowledgment or a fill. Check this same request ID.'});
 }catch(error){return reply(503,{error:String(error.message||error),state:'unknown',detail:'Request may already be reserved/delivered. Check status; no automatic retry.'});}
}
export async function onRequest({request,env}){const identity=await requireAccessIdentity(request,env);if(identity instanceof Response)return identity;return handleExecution(request,env,identity);}
