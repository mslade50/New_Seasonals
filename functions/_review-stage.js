/* One review authorizes all published accounts. Persist before broker delivery. */
import {canonical} from '../site/assets/review-core.js';

const digest=async text=>[...new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(text)))].map(x=>x.toString(16).padStart(2,'0')).join('');
export function stagingEnabled(env){return env.REVIEW_EXECUTION_LIVE_ENABLED==='1'&&env.REVIEW_EXECUTION_PREVIEW_ENABLED==='1';}
export async function stagingBlock(envelope,env){
 if(!stagingEnabled(env))return 'Automatic staging is not activated.';
 const p=envelope.payload,accounts=p.execution_accounts;
 if(!Array.isArray(accounts)||!accounts.length||new Set(accounts).size!==accounts.length||accounts.some(a=>!['primary','pa'].includes(a)))return 'Published execution accounts are unavailable.';
 const override=env.REVIEW_EXECUTION_PA_RISK_MULTIPLIER;
 if(accounts.includes('pa')&&override!==undefined&&override!==null&&override!==''&&(!['string','number'].includes(typeof override)||Number(override)!==1))return 'PA sizing configuration conflicts with the approved agent policy.';
 if(!p.source_sizing||typeof p.source_sizing_canonical!=='string'||canonical(JSON.parse(p.source_sizing_canonical))!==canonical(p.source_sizing))return 'The agent sizing instruction is unavailable.';
 const sizingHash=await digest(p.source_sizing_canonical);
 if(accounts.some(a=>p.account_proposals?.[a]?.account!==a||p.account_proposals[a].status!=='requires_fresh_account_preview'||p.account_proposals[a].sizing_hash!==sizingHash))return 'The agent sizing instruction is unavailable for a published account.';
 if(!Array.isArray(p.orders)||!p.orders.length||p.orders.some(o=>o.Manual_Only===true||['true','1','yes','y'].includes(String(o.Manual_Only).toLowerCase())||o.Sec_Type!=='STK'||o.Proxy_Ticker||Number(o.Trail_ATR||0)||Number(o.Trail_Arm_ATR||0)))return 'This idea contains an instruction that requires manual execution.';
 return null;
}
export function reserveStaging(record,envelope,event,now){
 const requests={...record.execution_requests},claims={...record.execution_claims},sources={...record.execution_sources};
 for(const account of envelope.payload.execution_accounts){
  const source=canonical([record.product,envelope.payload.source_idea_id,account]);
  const claim=canonical([record.product,envelope.payload.source_idea_id,envelope.hash,account]);
  if(sources[source]||claims[source]||claims[claim])throw Error('This idea already has an execution request. Check its status.');
  const id=crypto.randomUUID(),stamp=Date.parse(now);
  requests[id]={actor:event.actor,created_at:now,state:'queued',review_event_id:event.id,
   command:{id,type:'review_execution',account,dry_run:false,created_at:stamp,expires_at:Date.parse(envelope.payload.review_deadline),
    payload:{operation:'stage',product:record.product,actor:event.actor,proposal:envelope,review:event,
     delivery_id:record.delivery.delivery_id,current_at_authorization:true}}};
  sources[source]=id;claims[claim]=id;
 }
 return {...record,execution_requests:requests,execution_claims:claims,execution_sources:sources};
}
export async function broker(env,path,command){
 if(!env.EXEC_BROKER_URL||!env.STATUS_TOKEN)throw Error('Broker connection is not configured.');
 const options={headers:{Authorization:`Bearer ${env.STATUS_TOKEN}`},redirect:'error'};
 if(command){const signed=JSON.stringify(command),key=await crypto.subtle.importKey('raw',new TextEncoder().encode(env.STATUS_TOKEN),{name:'HMAC',hash:'SHA-256'},false,['sign']);
  const sig=[...new Uint8Array(await crypto.subtle.sign('HMAC',key,new TextEncoder().encode(signed)))].map(x=>x.toString(16).padStart(2,'0')).join('');
  options.method='POST';options.headers['Content-Type']='application/json';options.body=JSON.stringify({signed,sig});}
 const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),10000);
 try{const r=await fetch(env.EXEC_BROKER_URL.replace(/\/$/,'')+path,{...options,signal:controller.signal});if(!r.ok)throw Error('Broker request was not acknowledged.');return await r.json();}finally{clearTimeout(timer);}
}
export async function deliverStaging(record,event,env,transport=broker){
 // The same durable UUID is used after a connection interruption. The broker's
 // UUID deduplication and local source/account claim prevent a second placement.
 return Promise.all(Object.values(record.execution_requests||{}).filter(r=>r.review_event_id===event.id).map(async r=>{
  try{const result=await transport(env,'/command',r.command);if(result.id!==r.command.id)throw Error('Broker request identity mismatch.');return {id:r.command.id,account:r.command.account,state:result.state||'queued'};}
  catch{console.warn(JSON.stringify({event:'review_staging_delivery_uncertain',request_id:r.command.id,account:r.command.account}));return {id:r.command.id,account:r.command.account,state:'unknown'};}
 }));
}
export async function stagingStatus(record,env,transport=broker){
 const jobs=Object.values(record.execution_requests||{}).filter(r=>r.command?.payload?.operation==='stage');
 if(!jobs.length)return [];
 let commands=[];try{commands=(await transport(env,'/commands')).commands||[];}catch{}
 return jobs.map(job=>{
  const command=commands.find(c=>c.id===job.command.id&&c.account===job.command.account&&c.type==='review_execution');
  const result=command?.result||{},fill=result.fill;
  const healthy=['working','partially_filled','filled','closed','partially_closed','cancelled'];
  let state=result.state||command?.state||'unknown';
  if(healthy.includes(state)&&(!fill||fill.state!==state||fill.command_id!==job.command.id||fill.plan?.payload?.account!==job.command.account||fill.plan?.payload?.proposal_hash!==job.command.payload.proposal.hash))state='unknown';
  return {id:job.command.id,proposal_id:job.command.payload.proposal.id,review_event_id:job.review_event_id,account:job.command.account,state,
   detail:command?result.detail||command.delivery_error||'': 'Broker status unavailable; check this request before retrying.',
   quantities:fill?.plan?.payload?.legs?.map(l=>({ticker:l.payload.symbol,action:l.payload.source_action,quantity:l.payload.quantity}))||[]};
 });
}
