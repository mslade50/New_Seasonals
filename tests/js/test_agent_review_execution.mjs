import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import crypto from 'node:crypto';
if (!globalThis.crypto) Object.defineProperty(globalThis,'crypto',{value:crypto.webcrypto});
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const source=rel=>fs.readFileSync(path.join(root,rel),'utf8');
const inherited=rel=>fs.readFileSync(path.join(process.env.REVIEW_EXECUTION_TEST_REPO||root,rel),'utf8');
const uri=text=>'data:text/javascript;base64,'+Buffer.from(text).toString('base64');
const coreUri=uri(inherited('site/assets/review-core.js')),authUri=uri(inherited('functions/_access.js'));
const inboxUri=uri(inherited('functions/review-inbox.js').replace("'./_access.js'",JSON.stringify(authUri)).replace("'../site/assets/review-core.js'",JSON.stringify(coreUri)));
const api=await import(uri(source('functions/review-execution.js').replace("'./_access.js'",JSON.stringify(authUri)).replace("'./review-inbox.js'",JSON.stringify(inboxUri)).replace("'../site/assets/review-core.js'",JSON.stringify(coreUri))));
const C=await import(coreUri),clock=()=> '2026-10-06T14:00:00Z',actor={subject:'signed-human'};
const uuid=()=>crypto.randomUUID(), sha=text=>crypto.createHash('sha256').update(text).digest('hex');
const row={Idea_Id:'2026-10-06-1',Leg:1,Ticker:'XLE',Sec_Type:'STK',Action:'BUY',Entry_Type:'LIMIT',Order_Type:'LMT',TIF:'DAY',Quantity:10,Execute_On:'2026-10-06',Time_Exit_Date:'2026-10-13',Time_Exit_Order:'MOO'};
async function fixture(product='pitch',settings={}){
 const payload={schema:1,product,source_idea_id:row.Idea_Id,source_date:'2026-10-06',title:'Mock idea',thesis:'Mock only',account:product==='pitch'?'primary':'Unassigned',manual_only:product==='seasonal',published_at:'2026-10-06T09:00:00Z',review_deadline:'2026-10-06T20:00:00Z',execution_deadline:product==='pitch'?'2026-10-06T20:00:00Z':null,orders:[row]};
 const envelope=await C.seal(payload),key=`review_inbox/v1/${product}/2026-10-06.json`,event={id:uuid(),proposal_id:envelope.id,proposal_hash:envelope.hash,revision:1,decision:'approve_review',scope:'human_review_only',execution:'not_submitted',actor:actor.subject,at:'2026-10-06T13:00:00Z',reason:''};
 let record={schema:'review-inbox.v1',product,date:'2026-10-06',proposals:{[envelope.id]:envelope},current_ids:[envelope.id],events:[event],delivery:{delivery_id:'mock-delivery'}},etag=0;
 const bucket={receipt:'mock-delivery',fail:false,hook:null,get:async k=>{if(bucket.fail)throw Error('store unavailable');if(k.endsWith('_delivery_receipts/2026-10-06.json'))return {json:async()=>({status:'sent',date:'2026-10-06',delivery_id:bucket.receipt})};if(k!==key)return null;return {size:1000,etag:String(etag),json:async()=>structuredClone(record)};},put:async(k,text,options)=>{assert.equal(k,key);if(bucket.hook){const f=bucket.hook;bucket.hook=null;f();}if(options.onlyIf.etagMatches!==String(etag))return null;record=JSON.parse(text);etag++;return {etag:String(etag)};}};
 const commands=[],results=[];let timeout=false;
 const transport=async(env,route,command)=>{if(route==='/commands')return {commands:results};commands.push(command);assert.ok(record.execution_requests[command.id],'reserve must precede delivery');if(timeout)throw Error('mock delivery uncertain');return {id:command.id,state:'pushed'};};
 const env={CHARTS:bucket,REVIEW_EXECUTION_PREVIEW_ENABLED:'1',REVIEW_EXECUTION_LIVE_ENABLED:'1',REVIEW_EXECUTION_SEASONAL_ACCOUNT:'primary',...settings};
 const body={id:uuid(),operation:'preview',product,date:payload.source_date,proposal_id:envelope.id,proposal_hash:envelope.hash,account:'primary'};
 const request=(b=body,headers={})=>new Request('https://mock.invalid/review-execution',{method:'POST',headers:{Origin:'https://mock.invalid','Content-Type':'application/json',...headers},body:JSON.stringify(b)});
 const post=(b=body,who=actor,now=clock,headers={})=>api.handleExecution(request(b,headers),env,who,now,transport);
 const get=(id=body.id,who=actor)=>api.handleExecution(new Request(`https://mock.invalid/review-execution?product=${product}&date=2026-10-06&id=${id}`),env,who,clock,transport);
 const planPayload={schema:'review-execution-plan.v1',product,source_idea_id:row.Idea_Id,proposal_id:envelope.id,proposal_hash:envelope.hash,review_event_id:event.id,delivery_id:'mock-delivery',account:'primary',broker_account:'MOCK_ACCOUNT',actor:actor.subject,expires_at:'2026-10-06T14:05:00Z',non_atomic:false,risk_ack_required:false,legs:[]};
 const plan=()=>{const text=C.canonical(planPayload);return {payload:structuredClone(planPayload),canonical:text,hash:sha(text)};};
 async function preview(){assert.equal((await post()).status,202);results.push({id:body.id,type:'review_execution',account:'primary',state:'preview',result:{preview:plan()}});assert.equal((await get()).status,200);return {id:uuid(),operation:'execute',product,date:body.date,account:'primary',proposal_id:envelope.id,proposal_hash:envelope.hash,preview_id:body.id,plan_hash:plan().hash,confirmed:true};}
 return {env,bucket,body,commands,results,post,get,planPayload,preview,record:()=>record,mutate:fn=>{fn(record);etag++;},timeout:()=>{timeout=true;}};
}
let count=0;const test=async(name,fn)=>{await fn();count++;console.log('PASS '+name);};
await test('missing auth and default flags fail closed',async()=>{assert.equal((await api.onRequest({request:new Request('https://mock.invalid/review-execution'),env:{}})).status,503);assert.deepEqual(api.settings({}),{preview_enabled:false,live_enabled:false,accounts:{pitch:'primary',seasonal:null}});const f=await fixture('pitch',{REVIEW_EXECUTION_PREVIEW_ENABLED:'0'});assert.equal((await f.post()).status,503);assert.equal(f.commands.length,0);});
for(const product of ['pitch','seasonal'])await test(product+' exact review -> preview -> explicit execution -> readonly reconcile',async()=>{const f=await fixture(product),execute=await f.preview();assert.equal(f.commands[0].dry_run,true);assert.equal(f.commands[0].payload.proposal.hash,f.body.proposal_hash);assert.equal((await f.post(execute)).status,202);assert.equal(f.commands[1].dry_run,false);assert.equal((await f.post({id:uuid(),operation:'reconcile',product,date:f.body.date,account:'primary',plan_hash:execute.plan_hash})).status,202);const cmd=f.commands.at(-1);assert.equal(cmd.dry_run,true);assert.equal(cmd.payload.run_key,sha(C.canonical([product,row.Idea_Id,'primary'])));assert.equal(f.record().events.length,1);});
await test('preview-only cannot forward execution',async()=>{const f=await fixture('pitch',{REVIEW_EXECUTION_LIVE_ENABLED:'0'}),e=await f.preview();assert.equal((await f.post(e)).status,503);assert.equal(f.commands.length,1);});
await test('account assignment and same-origin JSON required',async()=>{const f=await fixture('seasonal',{REVIEW_EXECUTION_SEASONAL_ACCOUNT:''});assert.equal((await f.post()).status,409);for(const headers of [{Origin:'https://evil.invalid'},{'Content-Type':'text/plain'}])assert.equal((await f.post(f.body,actor,clock,headers)).status,403);assert.equal(f.commands.length,0);});
await test('pending/rejected/version/current delivery/expiry all block before wire',async()=>{for(const mutate of [r=>r.events=[],r=>r.events[0].decision='reject',r=>r.current_ids=[]]){const f=await fixture();f.mutate(mutate);assert.equal((await f.post()).status,409);assert.equal(f.commands.length,0);}const f=await fixture();assert.equal((await f.post({...f.body,proposal_hash:'changed'})).status,409);f.bucket.receipt='new';assert.equal((await f.post()).status,409);f.bucket.receipt='mock-delivery';assert.equal((await f.post(f.body,actor,()=> '2026-10-06T20:00:00Z')).status,409);});
await test('whole/non-atomic/risk confirmations independently required',async()=>{for(const field of ['confirmed','non_atomic_ack','risk_ack']){const f=await fixture();f.planPayload.non_atomic=field==='non_atomic_ack';f.planPayload.risk_ack_required=field==='risk_ack';const e=await f.preview();assert.equal((await f.post({...e,[field]:false})).status,409);assert.equal(f.commands.length,1);}});
await test('plan hash/actor/account/expiry cannot be altered',async()=>{const f=await fixture(),e=await f.preview();assert.equal((await f.post({...e,plan_hash:'bad'})).status,409);assert.equal((await f.post(e,{subject:'other'})).status,409);assert.equal((await f.post({...e,account:'pa'})).status,409);assert.equal((await f.post(e,actor,()=> '2026-10-06T14:06:00Z')).status,409);assert.equal(f.commands.length,1);});
await test('concurrent execution intents reserve one permanent idea/account claim',async()=>{const f=await fixture(),e=await f.preview();const responses=await Promise.all([f.post(e),f.post({...e,id:uuid()})]);assert.deepEqual(responses.map(r=>r.status).sort(),[202,409]);assert.equal(f.commands.filter(c=>!c.dry_run).length,1);assert.equal(Object.keys(f.record().execution_claims).length,1);});
await test('explicit UUID retry forwards identical command; switching intent rejected',async()=>{const f=await fixture(),e=await f.preview();assert.equal((await f.post(e)).status,202);assert.equal((await f.post(e)).status,202);assert.deepEqual(f.commands[1],f.commands[2]);assert.equal((await f.post({...e,risk_ack:true})).status,409);});
await test('publication race prevents stale delivery without erasing audit',async()=>{const f=await fixture();f.bucket.hook=()=>f.mutate(r=>r.current_ids=[]);assert.equal((await f.post()).status,409);assert.equal(f.commands.length,0);});
await test('network uncertainty preserves durable claim and never retries automatically',async()=>{const f=await fixture(),e=await f.preview();f.timeout();assert.equal((await f.post(e)).status,503);assert.ok(f.record().execution_requests[e.id]);assert.equal((await f.post({...e,id:uuid()})).status,409);assert.equal(f.commands.filter(c=>!c.dry_run).length,1);});
await test('ring eviction is unknown/stale, never zero inventory; actor isolated',async()=>{const f=await fixture();assert.equal((await f.post()).status,202);const result=await (await f.get()).json();assert.equal(result.state,'unknown');assert.equal(result.stale,true);assert.equal((await f.get(f.body.id,{subject:'other'})).status,403);});
await test('broker contract identity and malformed plans fail closed',async()=>{const f=await fixture();await f.preview();f.results[0].account='pa';assert.equal((await f.get()).status,503);await assert.rejects(api.verifyPlan({canonical:'{}',payload:{schema:'review-execution-plan.v1'},hash:'bad'}));});
await test('store outage cannot report reservation or forward',async()=>{const f=await fixture();f.bucket.fail=true;assert.equal((await f.post()).status,503);assert.equal(f.commands.length,0);});
await test('generic route cannot bypass verified adapter; unrelated legacy default preserved',async()=>{
 const generic=await import(uri(source('functions/exec-command.js').replace('./_access.js',uri('export async function requireAccess(){return null;}'))));
 const previous=globalThis.fetch,calls=[];globalThis.fetch=async(url,options)=>{calls.push(JSON.parse(options.body));return new Response(JSON.stringify({state:'pushed'}),{status:202,headers:{'Content-Type':'application/json'}});};
 try{
  const base={EXEC_BROKER_URL:'https://mock-broker.invalid',STATUS_TOKEN:'NONSECRET_TEST_TOKEN'},req=body=>new Request('https://mock.invalid/exec-command',{method:'POST',body:JSON.stringify({account:'primary',dry_run:false,...body})});
  assert.equal((await generic.onRequestPost({request:req({type:'review_execution'}),env:base})).status,409);
  for(const strategy of ['Pitch-fixture','Seasonal_Agent-fixture'])assert.equal((await generic.onRequestPost({request:req({type:'entry_bracket',payload:{strategy}}),env:{...base,REVIEW_EXECUTION_LIVE_ENABLED:'1'}})).status,409);
  assert.equal(calls.length,0);
  assert.equal((await generic.onRequestPost({request:req({type:'entry_bracket',payload:{strategy:'Pitch-fixture'}}),env:base})).status,202);
  assert.equal(calls.length,1);
 }finally{globalThis.fetch=previous;}
});
console.log(`${count} offline review-execution endpoint checks passed`);
