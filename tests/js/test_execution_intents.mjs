// Financial intent boundaries, entirely offline.
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import {webcrypto} from 'node:crypto';

const asset = file => fs.readFileSync(new URL(`../../${file}`, import.meta.url), 'utf8');
function endpoint() {
  const sent = [];
  const c = {Request, Response, TextEncoder, crypto:webcrypto, requireAccess:async()=>null,
    fetch:async(_, opts)=>{sent.push(JSON.parse(JSON.parse(opts.body).signed)); return Response.json({ok:true});}};
  vm.createContext(c);
  vm.runInContext(asset('functions/exec-command.js').replace(/^import .*;\r?$/gm,'').replace('export async function','async function'), c);
  return {c, sent};
}
function browser(file, storage = new Map()) {
  const sent=[], confirmations=[];
  const c={console, location:{search:''}, URLSearchParams, window:{},
    document:{addEventListener(){},getElementById(){return null;}},
    crypto:webcrypto, setTimeout(){},clearTimeout(){},setInterval(){},clearInterval(){},
    sessionStorage:{getItem:k=>storage.get(k)??null,setItem:(k,v)=>storage.set(k,v)},
    confirm:m=>{confirmations.push(m);return true;},fmt:{money:String},
    fetch:async(_,o)=>{sent.push(JSON.parse(o.body));throw Error('response lost');}};
  vm.createContext(c);
  const helper=new URL('../../site/assets/command-intents.js', import.meta.url);
  if(fs.existsSync(helper))vm.runInContext(fs.readFileSync(helper,'utf8'),c);
  vm.runInContext(asset(file),c);
  return {c,sent,confirmations,storage};
}
const checks=[];
async function check(name, fn) {try {await fn();console.log('PASS',name);}catch(e){checks.push(name+': '+e.message);}}
await check('invalid account never signs',async()=>{
  const {c,sent}=endpoint();
  for(const account of [undefined,'PRIMARY','invalid']) {
    const response=await c.onRequestPost({request:new Request('https://offline.invalid/command',{method:'POST',body:JSON.stringify({account,type:'echo',payload:{},dry_run:true})}),env:{STATUS_TOKEN:'test-only',EXEC_BROKER_URL:'https://offline.invalid'}});
    assert.equal(response.status,400);
  }
  assert.equal(sent.length,0);
});
for(const file of ['site/assets/execution.js','site/assets/options.js']) {
  await check(file+' preserves A after uncertain B and reload',async()=>{
    const {c,sent,storage}=browser(file);
    for(const symbol of ['AAA','BBB','AAA'])await vm.runInContext(`sendCommand('entry_bracket',{symbol:'${symbol}',quantity:1},null)`,c);
    assert.equal(sent[0].id,sent[2].id);
    const reloaded=browser(file,storage);
    await vm.runInContext("sendCommand('entry_bracket',{symbol:'AAA',quantity:1},null)",reloaded.c);
    assert.equal(reloaded.sent[0].id,sent[0].id);
  });
  await check(file+' transmits confirmed dry-run',async()=>{
    const {c,sent}=browser(file);
    vm.runInContext("state.book={mode:'dry-run',at:Date.now()};state.status={online:true}",c);
    await vm.runInContext("sendCommand('echo',{},null)",c);
    assert.equal(sent[0].dry_run,true);
    vm.runInContext("state.book.mode='live'",c);
    await vm.runInContext("sendCommand('echo',{},null)",c);
    assert.equal(sent[1].dry_run,false);
    assert.notEqual(sent[1].id,sent[0].id);
  });
}
await check('secondary acknowledgment retains Primary and dry-run',async()=>{
  const {c,sent,confirmations}=browser('site/assets/execution.js');
  vm.runInContext(`state.account='pa';riskAckPending.set('prior',{type:'entry_bracket',account:'primary',dryRun:true,payload:{action:'BUY',symbol:'AAA',quantity:1,entry:100}});state.commands=[{id:'prior',state:'rejected',result:{fill:{needs_risk_ack:true}}}];checkRiskAck()`,c);
  await Promise.resolve();
  assert.equal(sent[0].account,'primary');assert.equal(sent[0].dry_run,true);
  assert.match(confirmations[0],/on primary/);
});
if(checks.length)throw new Error(checks.join('\n'));
