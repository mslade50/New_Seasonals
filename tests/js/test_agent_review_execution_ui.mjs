/* Real UI module, isolated DOM/fetch fixtures. No live site or broker traffic. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import crypto from 'node:crypto';
if(!globalThis.crypto)Object.defineProperty(globalThis,'crypto',{value:crypto.webcrypto});
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const coreSource=fs.readFileSync(path.join(process.env.REVIEW_EXECUTION_TEST_REPO||root,'site/assets/review-core.js'),'utf8');
const uri=text=>'data:text/javascript;base64,'+Buffer.from(text).toString('base64');
const coreUri=uri(coreSource),C=await import(coreUri),at=Date.parse('2026-10-06T14:00:00Z');
const originalNow=Date.now;Date.now=()=>at;
globalThis.renderNav=()=>{};globalThis.setInterval=()=>0;
let serial=0;
async function fixture(product='pitch',{live=false,enabled=true,account='primary',pending=null,nonAtomic=false,risk=false,fail=false}={}){
 const payload={schema:1,product,source_date:'2026-10-06',source_idea_id:'2026-10-06-1',title:'Mock whole idea',thesis:'Fixture',account:'primary',published_at:'2026-10-06T09:00:00Z',review_deadline:'2026-10-06T20:00:00Z',execution_deadline:'2026-10-06T20:00:00Z',orders:[{Idea_Id:'2026-10-06-1',Leg:1,Ticker:'XLE',Sec_Type:'STK',Action:'BUY',Entry_Type:'LIMIT',Order_Type:'LMT',TIF:'DAY',Quantity:10,Execute_On:'2026-10-06',Time_Exit_Date:'2026-10-13',Time_Exit_Order:'MOO'}]};
 const envelope=await C.seal(payload),nodes=new Map(),storage=new Map(),requests=[],posted=new Map();
 const storageKey=`review-execution:${product}:2026-10-06:${envelope.id}`;
 if(pending)storage.set(storageKey,JSON.stringify(pending));
 function node(id){if(!nodes.has(id))nodes.set(id,{id,disabled:['preview','check','reconcile','submit-execution'].includes(id),checked:false,hidden:false,textContent:'',innerHTML:'',listeners:{},addEventListener(name,fn){this.listeners[name]=fn;},showModal(){this.open=true;},close(){this.open=false;}});return nodes.get(id);}
 globalThis.document={getElementById:node};
 globalThis.location={search:`?product=${product}&date=2026-10-06&proposal=${encodeURIComponent(envelope.id)}&hash=${envelope.hash}`};
 globalThis.sessionStorage={getItem:k=>storage.get(k)||null,setItem:(k,v)=>storage.set(k,v)};
 let expired=false;
 globalThis.fetch=async(url,options)=>{
  const body=options.method==='POST'?JSON.parse(options.body):null;requests.push({url,body});
  if(fail&&body)throw Error('mock timeout after uncertain delivery');
  let value;
  if(body){posted.set(body.id,body);value={id:body.id,state:'pushed'};}
  else if(url==='/review-execution')value={preview_enabled:enabled,live_enabled:live,accounts:{pitch:account,seasonal:account}};
  else if(url.startsWith('/review-inbox'))value={read_only:false,products:[{product,proposals:[{envelope,current:true,state:{status:'approved_review',review_window_closed:false}}]}]};
  else {
   const id=new URL('https://mock.invalid'+url).searchParams.get('id'),saved=posted.get(id);
   if(saved?.operation==='preview'){
    const planPayload={schema:'review-execution-plan.v1',product,proposal_id:envelope.id,proposal_hash:envelope.hash,account:'primary',broker_account:'MOCK_ACCOUNT',expires_at:expired?'2026-10-06T13:00:00Z':'2026-10-06T14:05:00Z',non_atomic:nonAtomic,risk_ack_required:risk,exit_convention:'Fixture time exit',risk_usd:20,notional_usd:1000,nlv:100000,legs:[{leg:1,con_id:11,payload:{symbol:'XLE',action:'BUY',quantity:10,entry_type:'LMT',entry:100,stop:98,target:104,time_stop:'2026-10-13',time_stop_at:'open',stop_arm:'next_session',expiry:null}}]};
    const canonical=C.canonical(planPayload);value={state:'preview',preview:{payload:planPayload,canonical,hash:crypto.createHash('sha256').update(canonical).digest('hex')}};
   }else value={state:'working',stale:false};
  }
  return new Response(JSON.stringify(value),{headers:{'Content-Type':'application/json'}});
 };
 const script=fs.readFileSync(path.join(root,'site/assets/execute-review.js'),'utf8').replace("'./review-core.js'",JSON.stringify(coreUri));
 await import(uri(script+`\n// fixture ${serial++}`));
 const click=async(id,name='click')=>{const n=node(id);if(n.disabled)return;await n.listeners[name]?.({preventDefault(){}});};
 return {node,click,requests,storage,storageKey,expire:()=>{expired=true;}};
}
let checks=0;const test=async(name,fn)=>{await fn();checks++;console.log('PASS '+name);};
await test('default off and unassigned account disable preview without POST',async()=>{for(const options of [{enabled:false},{account:null}]){const f=await fixture('seasonal',options);assert.equal(f.node('preview').disabled,true);assert.equal(f.requests.filter(r=>r.body).length,0);}});
for(const product of ['pitch','seasonal'])await test(product+' preview shows entire qualified plan but cannot submit when live off',async()=>{const f=await fixture(product);await f.click('preview');assert.ok(f.node('plan').innerHTML.includes('conId 11'));assert.equal(f.node('whole-ack').checked,false);f.node('whole-ack').checked=true;await f.click('whole-ack','change');assert.equal(f.node('submit-execution').disabled,true);assert.equal(f.requests.filter(r=>r.body).length,1);assert.equal(f.requests.find(r=>r.body).body.operation,'preview');});
await test('all required unchecked confirmations gate execution',async()=>{const f=await fixture('seasonal',{live:true,nonAtomic:true,risk:true});await f.click('preview');assert.equal(f.node('submit-execution').disabled,true);for(const id of ['whole-ack','non-atomic-ack']){f.node(id).checked=true;await f.click(id,'change');assert.equal(f.node('submit-execution').disabled,true);}f.node('risk-ack').checked=true;await f.click('risk-ack','change');assert.equal(f.node('submit-execution').disabled,false);await f.click('execution-form','submit');assert.equal(f.requests.filter(r=>r.body?.operation==='execute').length,1);assert.equal(f.node('reconcile').disabled,false);await f.click('reconcile');const saved=JSON.parse(f.storage.get(f.storageKey));assert.equal(saved.current.operation,'reconcile');assert.equal(saved.execution.operation,'execute');});
await test('expired preview cannot enable execution',async()=>{const f=await fixture('pitch',{live:true});f.expire();await f.click('preview');f.node('whole-ack').checked=true;await f.click('whole-ack','change');assert.equal(f.node('submit-execution').disabled,true);});
await test('delivery timeout never automatically retries a POST',async()=>{const f=await fixture('pitch',{live:true,fail:true});await f.click('preview');assert.equal(f.requests.filter(r=>r.body).length,1);assert.ok(f.node('status').textContent.includes('No automatic POST retry'));assert.equal(f.node('check').disabled,false);});
await test('restoring reconcile intent retains original execution ID and makes no POST',async()=>{const execution={id:crypto.randomUUID(),operation:'execute',account:'primary',plan_hash:'fixture-hash'},current={id:crypto.randomUUID(),operation:'reconcile',account:'primary',plan_hash:'fixture-hash'};const f=await fixture('pitch',{pending:{current,execution}});assert.equal(f.node('reconcile').disabled,false);assert.equal(f.requests.filter(r=>r.body).length,0);assert.equal(JSON.parse(f.storage.get(f.storageKey)).execution.id,execution.id);});
Date.now=originalNow;
console.log(`${checks} isolated whole-idea UI checks passed`);
