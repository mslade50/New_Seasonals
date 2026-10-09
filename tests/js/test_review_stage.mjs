/* Real single-click endpoint + UI modules, with isolated storage/transport/DOM. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
if(!globalThis.crypto)Object.defineProperty(globalThis,'crypto',{value:crypto.webcrypto});
const source=rel=>fs.readFileSync(new URL('../../'+rel,import.meta.url),'utf8');
const uri=text=>'data:text/javascript;base64,'+Buffer.from(text).toString('base64');
const coreUri=uri(source('site/assets/review-core.js')),authUri=uri(source('functions/_access.js'));
const stageUri=uri(source('functions/_review-stage.js').replace("'../site/assets/review-core.js'",JSON.stringify(coreUri)));
const api=await import(uri(source('functions/review-inbox.js').replace("'./_access.js'",JSON.stringify(authUri)).replace("'./_review-stage.js'",JSON.stringify(stageUri)).replace("'../site/assets/review-core.js'",JSON.stringify(coreUri))));
const C=await import(coreUri),clock=()=> '2026-10-09T10:00:00Z',actor={subject:'fixture-human'},uuid=()=>crypto.randomUUID();
async function fixture(product='pitch',enabled=true){
 const sizing={schema:'agent-sizing.v1',risk_bps:20},text=C.canonical(sizing),sizingHash=crypto.createHash('sha256').update(text).digest('hex');
 const row={Idea_Id:'2026-10-09-1',Leg:1,Ticker:'SPY',Sec_Type:'STK',Action:'BUY',Entry_Type:'MOC',Order_Type:'MKT',TIF:'MOC',Quantity:25,Execute_On:'2026-10-09',Time_Exit_Date:'2026-11-09',Time_Exit_Order:'MOC'};
 const payload={schema:1,product,source_date:'2026-10-09',source_idea_id:row.Idea_Id,title:'Fixture idea',thesis:'Fixture thesis',account:'Primary + PA',published_at:'2026-10-09T09:00:00Z',review_deadline:'2026-10-09T19:30:00Z',execution_deadline:'2026-10-09T19:30:00Z',orders:[row,{...row,Leg:2,Ticker:'EEM',Action:'SELL_SHORT',Quantity:302}],execution_accounts:['primary','pa'],source_sizing:sizing,source_sizing_canonical:text,account_proposals:Object.fromEntries(['primary','pa'].map(account=>[account,{account,status:'requires_fresh_account_preview',sizing_hash:sizingHash}]))};
 let envelope=await C.seal(payload),record={schema:'review-inbox.v1',product,date:payload.source_date,current_ids:[envelope.id],proposals:{[envelope.id]:envelope},events:[],delivery:{delivery_id:'fixture-delivery'}},etag=0;
 const key=`review_inbox/v1/${product}/${payload.source_date}.json`,commands=[],results=[];
 const bucket={hook:null,fail:false,receipt:'fixture-delivery',get:async k=>{if(bucket.fail)throw Error('offline');if(k===`${product==='pitch'?'pitch':'seasonal_agent'}_delivery_receipts/2026-10-09.json`)return {json:async()=>({date:payload.source_date,status:'sent',delivery_id:bucket.receipt})};return k===key?{size:1000,etag:String(etag),json:async()=>structuredClone(record)}:null;},put:async(k,value,options)=>{if(bucket.fail)throw Error('offline');if(bucket.hook){const f=bucket.hook;bucket.hook=null;f();}if(options.onlyIf.etagMatches!==String(etag))return null;record=JSON.parse(value);etag++;return {etag:String(etag)};}};
 const env={CHARTS:bucket,REVIEW_EXECUTION_PREVIEW_ENABLED:enabled?'1':'0',REVIEW_EXECUTION_LIVE_ENABLED:enabled?'1':'0'};
 const transport=async(env,path,command)=>{if(path==='/commands')return {commands:results};assert.ok(record.execution_requests[command.id]);assert.equal(record.events[0].scope,'review_and_stage');assert.equal(Object.keys(record.execution_requests).length,2);commands.push(command);return {id:command.id,state:'pushed'};};
 const body={id:uuid(),product,proposal_id:envelope.id,proposal_hash:envelope.hash,expected_revision:0,decision:'approve_review',confirmed:true,stage:true,reason:''};
 const req=body=>new Request('https://fixture.invalid/review-inbox?date=2026-10-09',{method:'POST',headers:{Origin:'https://fixture.invalid','Content-Type':'application/json'},body:JSON.stringify(body)});
 const post=(b=body,who=actor,now=clock,t=transport,background=null)=>api.handleReview(req(b),env,who,now,t,background);
 const get=()=>api.handleReview(new Request('https://fixture.invalid/review-inbox?date=2026-10-09'),env,actor,clock,transport);
 return {env,bucket,commands,results,body,post,get,record:()=>record,mutate:fn=>{fn(record);etag++;},amend:async fn=>{fn(payload);envelope=await C.seal(payload);record.proposals[envelope.id]=envelope;record.current_ids=[envelope.id];body.proposal_id=envelope.id;body.proposal_hash=envelope.hash;etag++;}};
}
let count=0;const test=async(name,fn)=>{await fn();count++;console.log('PASS '+name);};
for(const product of ['pitch','seasonal'])await test(product+' Yes reserves both accounts and immediately delivers all-leg commands',async()=>{
 const f=await fixture(product),r=await f.post();assert.equal(r.status,200);const reply=await r.json();
 assert.equal(reply.execution,'queued');assert.equal(reply.event.scope,'review_and_stage');assert.deepEqual(reply.event.accounts,['primary','pa']);
 assert.deepEqual(f.commands.map(c=>c.account),['primary','pa']);
 for(const c of f.commands){assert.equal(c.payload.operation,'stage');assert.equal(c.dry_run,false);assert.equal(c.payload.proposal.hash,f.body.proposal_hash);assert.equal(c.payload.proposal.payload.orders.length,2);assert.equal(c.payload.review.id,f.body.id);assert.equal(c.payload.plan_hash,undefined);}
 assert.equal(Object.keys(f.record().execution_sources).length,2);
});
await test('No records rejection and creates zero staging requests',async()=>{const f=await fixture();assert.equal((await f.post({...f.body,decision:'reject',stage:false,reason:'Declined'})).status,200);assert.equal(f.record().events[0].decision,'reject');assert.equal(f.commands.length,0);assert.equal(f.record().execution_requests,undefined);});
await test('double click and concurrent devices create only one pair of jobs',async()=>{const f=await fixture(),r=await Promise.all([f.post(),f.post({...f.body,id:uuid()})]);assert.deepEqual(r.map(x=>x.status).sort(),[200,409]);assert.equal((await f.post()).status,200);assert.equal(f.record().events.length,1);assert.equal(f.commands.length,2);});
await test('disabled staging cannot silently accept a Yes as research only',async()=>{const f=await fixture('seasonal',false);assert.equal((await f.post()).status,409);assert.equal(f.record().events.length,0);assert.equal(f.commands.length,0);assert.equal((await f.post({...f.body,decision:'reject',stage:false,reason:'Declined'})).status,200);});
await test('late/stale delivery/version approval cannot stage',async()=>{const f=await fixture();assert.equal((await f.post(f.body,actor,()=> '2026-10-09T19:30:00Z')).status,409);f.bucket.receipt='new';assert.equal((await f.post()).status,409);f.bucket.receipt='fixture-delivery';assert.equal((await f.post({...f.body,proposal_hash:'wrong'})).status,409);assert.equal(f.commands.length,0);});
await test('publication CAS race never stages superseded proposals',async()=>{const f=await fixture();f.bucket.hook=()=>f.mutate(r=>r.current_ids=[]);assert.equal((await f.post()).status,409);assert.equal(f.commands.length,0);assert.equal(f.record().events.length,0);});
await test('store failure cannot trigger broker delivery',async()=>{const f=await fixture();f.bucket.fail=true;assert.equal((await f.post()).status,503);assert.equal(f.commands.length,0);});
await test('missing sizing or manual leg blocks before approval',async()=>{for(const change of [p=>delete p.source_sizing,p=>p.orders[1].Manual_Only=true,p=>p.orders[1].Trail_ATR=1]){const f=await fixture();await f.amend(change);assert.equal((await f.post()).status,409);assert.equal(f.record().events.length,0);assert.equal(f.commands.length,0);}});
await test('one account transport failure does not erase either durable intent',async()=>{const f=await fixture();let calls=0;assert.equal((await f.post(f.body,actor,clock,async()=>{calls++;throw Error('mock timeout');})).status,200);assert.equal(calls,2);assert.equal(Object.keys(f.record().execution_requests).length,2);const fresh=await (await f.get()).json();assert.equal(fresh.products[0].staging.length,2);assert.ok(fresh.products[0].staging.every(j=>j.state==='unknown'));});
await test('HTTP response may finish while background broker handoff is pending',async()=>{const f=await fixture();let finish;const promise=new Promise(r=>finish=r),tasks=[];const r=await f.post(f.body,actor,clock,async(e,p,c)=>{await promise;return {id:c.id,state:'pushed'};},p=>tasks.push(p));assert.equal(r.status,200);assert.equal(tasks.length,1);finish();await tasks[0];});
await test('status shows broker rejection and never calls an order route',async()=>{const f=await fixture();await f.post();f.results.push({id:f.commands[0].id,type:'review_execution',account:'primary',state:'rejected',result:{state:'rejected',detail:'Insufficient buying power'}});const d=await (await f.get()).json();assert.equal(d.products[0].staging[0].state,'rejected');assert.equal(d.products[0].staging[0].detail,'Insufficient buying power');assert.equal(f.commands.length,2);});
await test('old research-only approval never queues orders on page refresh',async()=>{const f=await fixture();assert.equal((await f.post({...f.body,stage:false})).status,200);const d=await (await f.get()).json();assert.equal(d.products[0].staging.length,0);assert.equal(f.commands.length,0);});

// Exercise the real UI: one button click sends one decision, no dialog/preview.
let serial=0;
async function uiFixture({decision='approve_review',fail=false,enabled=true}={}){
 const f=await fixture('seasonal',enabled),nodes=new Map(),requests=[],storage=new Map();
 function node(id){if(!nodes.has(id))nodes.set(id,{id,value:id==='filter'?'all':id==='product'?'all':'',textContent:'',innerHTML:'',listeners:{},hidden:false,addEventListener(n,fn){this.listeners[n]=fn;},querySelectorAll(){return this.id==='queue'?[{dataset:{id:f.body.proposal_id,decision},addEventListener(n,fn){node('button').listeners[n]=fn;}}]:[];}});return nodes.get(id);}
 globalThis.document={getElementById:node,querySelector:()=>null,addEventListener(){},activeElement:null};
 globalThis.location={hash:'',search:''};globalThis.renderNav=()=>{};globalThis.setInterval=()=>0;
 globalThis.sessionStorage={getItem:k=>storage.get(k)||null,setItem:(k,v)=>storage.set(k,v),removeItem:k=>storage.delete(k)};
 const originalNow=Date.now;Date.now=()=>Date.parse(clock());
 globalThis.fetch=async(url,options)=>{if(options.method==='POST'){requests.push(JSON.parse(options.body));if(fail)throw Error('mock uncertain timeout');return f.post(JSON.parse(options.body));}return f.get();};
 await import(uri(source('site/assets/review.js').replace("'./review-core.js'",JSON.stringify(coreUri))+`\n//fixture ${++serial}`));
 return {f,node,requests,cleanup:()=>{Date.now=originalNow;}};
}
await test('UI Yes is one click and stages without opening another screen',async()=>{const u=await uiFixture();try{assert.ok(u.node('queue').innerHTML.includes('Yes — stage orders'));await u.node('button').listeners.click();assert.equal(u.requests.length,1);assert.equal(u.requests[0].stage,true);assert.equal(u.f.commands.length,2);assert.ok(u.node('notice').textContent.includes('Orders queued'));assert.equal(u.node('confirm').open,undefined);}finally{u.cleanup();}});
await test('UI No is one click with no mandatory note dialog',async()=>{const u=await uiFixture({decision:'reject'});try{await u.node('button').listeners.click();assert.equal(u.requests.length,1);assert.equal(u.requests[0].decision,'reject');assert.equal(u.requests[0].stage,false);assert.equal(u.f.commands.length,0);}finally{u.cleanup();}});
await test('UI double click while request is pending sends one Yes',async()=>{const u=await uiFixture();try{await Promise.all([u.node('button').listeners.click(),u.node('button').listeners.click()]);assert.equal(u.requests.length,1);assert.equal(u.f.commands.length,2);}finally{u.cleanup();}});
await test('UI failure prompts status check and never automatically resubmits',async()=>{const u=await uiFixture({fail:true});try{await u.node('button').listeners.click();assert.equal(u.requests.length,1);assert.ok(u.node('notice').textContent.includes('Check status'));assert.equal(u.f.commands.length,0);}finally{u.cleanup();}});
await test('UI disabled setup shows the reason and prevents Yes',async()=>{const u=await uiFixture({enabled:false});try{assert.ok(u.node('queue').innerHTML.includes('Automatic staging is not activated'));await u.node('button').listeners.click();assert.equal(u.requests.length,0);}finally{u.cleanup();}});
console.log(`${count} single-review staging checks passed`);
