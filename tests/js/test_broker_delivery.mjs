import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import * as reconciliation from '../../execution-broker/src/fill-reconcile.mjs';
const source=fs.readFileSync(new URL('../../execution-broker/src/index.js',import.meta.url),'utf8')
  .replace(/^import[\s\S]*?;\r?$/gm,'').replace('export class ExecBroker','class ExecBroker').replace(/export default[\s\S]*$/,'')+'\nthis.Broker=ExecBroker;';
class FakeDO {constructor(ctx,env){this.ctx=ctx;this.env=env;}}
function make(env={STATUS_TOKEN:'fixture',AGENT_TOKEN:'agent-fixture'}) {
  const memory=new Map();let calls=0, fail=true, connected=true;
  const socket={deserializeAttachment:()=>({}),send:()=>{calls++;if(fail)throw Error('socket closed');}};
  const ctx={getWebSockets:()=>connected?[socket]:[],storage:{
    get:async key=>structuredClone(memory.get(key)),
    put:async(key,value)=>{memory.set(key,structuredClone(value));},
    delete:async key=>memory.delete(key),
    list:async(opts={})=>new Map([...memory].filter(([k])=>k.startsWith(opts.prefix||'')&&(!opts.startAfter||k>opts.startAfter)).sort(([a],[b])=>a.localeCompare(b)).slice(0,opts.limit||1000)),
  }};
  const c={DurableObject:FakeDO,URL,Request,Response,Headers,console,TextEncoder,...reconciliation};
  vm.createContext(c);vm.runInContext(source,c);
  return {broker:new c.Broker(ctx,env),memory,calls:()=>calls,succeed:()=>{fail=false;},offline:()=>{connected=false;}};
}
const failures=[];
async function test(name, fn){try{await fn();console.log('PASS',name);}catch(e){failures.push(name+': '+e.message);}}
const command=(payload={symbol:'AAA'},id='fixture-command')=>new Request('https://fixture.invalid/command',{
  method:'POST',headers:{Authorization:'Bearer fixture'},body:JSON.stringify({
    signed:JSON.stringify({id,type:'echo',account:'primary',dry_run:true,payload,created_at:Date.now(),expires_at:Date.now()+60000}),sig:'test-signature'})});
await test('missing and empty credentials never authenticate',async()=>{
  for(const token of [undefined,'','   ']){
    const {broker}=make({STATUS_TOKEN:token});
    const r=await broker.fetch(new Request('https://fixture.invalid/status',{headers:{Authorization:`Bearer ${token}`}}));assert.equal(r.status,401);
  }
});
await test('failed delivery retries original durable intent once',async()=>{
  const x=make();
  const first=await x.broker.fetch(command());assert.equal(first.status,503);
  assert.equal((await first.json()).state,'delivery_unknown');
  x.succeed();const retry=await(await x.broker.fetch(command())).json();
  assert.equal(retry.ok,true);assert.equal(x.calls(),2);
  await x.broker.fetch(command());assert.equal(x.calls(),2);
  x.memory.set('recent_commands',[]);
  await x.broker.fetch(command());assert.equal(x.calls(),2,'dedup survives ring churn');
  assert.equal((await x.broker.fetch(command({symbol:'BBB'}))).status,409);
});
await test('fill retention does not discard 501st or 1001st execution',async()=>{
  const x=make();const now=Date.now();
  const fills=Array.from({length:1101},(_,i)=>({account:'fixture-primary',exec_id:`execution-${i}.01`,time:new Date(now).toISOString(),shares:1,price:100}));
  await x.broker._mergeFills({at:now,accounts:[{key:'primary',broker_account:'fixture-primary',fills_complete:true,fills}]});
  const response=await(await x.broker.fetch(new Request('https://fixture.invalid/fills',{headers:{Authorization:'Bearer fixture'}}))).json();
  assert.equal(response.fills.length,1101);assert.equal(response.completeness.truncated,false);
  assert.equal(response.completeness.accounts.primary.complete,true);
  await x.broker._mergeFills({at:now,accounts:[{key:'primary',broker_account:'fixture-primary',fills_complete:true,fills:[{...fills[0],exec_id:'execution-0.02',shares:2}]}]});
  const corrected=await(await x.broker.fetch(new Request('https://fixture.invalid/fills',{headers:{Authorization:'Bearer fixture'}}))).json();
  assert.equal(corrected.fills.length,1101);assert.equal(corrected.fills.find(f=>f.exec_id==='execution-0.02').shares,2);
});
await test('failed and legacy-capped sources never claim complete',async()=>{
  const x=make();
  await x.broker._mergeFills({at:Date.now(),accounts:[{key:'primary',error:'fixture offline',fills:[]}]});
  const result=await(await x.broker.fetch(new Request('https://fixture.invalid/fills',{headers:{Authorization:'Bearer fixture'}}))).json();
  assert.equal(result.completeness.complete,false);assert.equal(result.completeness.accounts.primary.complete,false);
  const day=new Date().toISOString().slice(0,10);
  x.memory.set(`fills:${day}`,Array.from({length:500},(_,i)=>({exec_id:`legacy-${i}.01`,account_key:'primary',time:new Date().toISOString()})));
  await x.broker._mergeFills({at:Date.now(),accounts:[{key:'primary',broker_account:'fixture-primary',fills_complete:true,fills:[{account:'fixture-primary',exec_id:'new.01',time:new Date().toISOString()}]}]});
  const legacy=await(await x.broker.fetch(new Request('https://fixture.invalid/fills',{headers:{Authorization:'Bearer fixture'}}))).json();
  assert.equal(legacy.completeness.truncated,true);
  assert.equal(legacy.completeness.complete,false);
});
await test('offline command stays visible as queued without transmission',async()=>{
  const x=make();x.offline();
  assert.equal((await x.broker.fetch(command())).status,503);
  const activity=await(await x.broker.fetch(new Request('https://fixture.invalid/commands',{headers:{Authorization:'Bearer fixture'}}))).json();
  assert.ok(activity.commands.some(c=>c.id==='fixture-command'&&c.state==='queued'));
  assert.equal(x.calls(),0);
});
await test('malformed or unattributed executions cannot attest completeness',async()=>{
  for(const account of [
    {key:'primary',fills_complete:true,fills:[]},
    {key:'primary',broker_account:'fixture-primary',fills_complete:true,fills:[{exec_id:'bad.01',time:'bad',account:'fixture-primary'}]},
    {key:'primary',broker_account:'fixture-primary',fills_complete:true,fills:[{exec_id:'bad.01',time:new Date().toISOString(),account:'another-account'}]},
  ]) {
    const x=make();await x.broker._mergeFills({at:Date.now(),accounts:[account]});
    const result=await(await x.broker.fetch(new Request('https://fixture.invalid/fills',{headers:{Authorization:'Bearer fixture'}}))).json();
    assert.equal(result.completeness.accounts.primary.complete,false);
    assert.equal(result.fills.length,0);
  }
});
if(failures.length)throw Error(failures.join('\n'));
