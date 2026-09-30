"use strict";
const assert=require('assert'),fs=require('fs'),vm=require('vm');
const source=fs.readFileSync(require('path').join(__dirname,'../../site/assets/execution.js'),'utf8');
function fixture(){
  const fields={cmdType:{value:'entry_bracket'},f_entry_type:{value:'MKT'},f_symbol:{value:'MNQ'},
    f_sectype:{value:'FUT'},f_futexch:{value:'CME'},f_futexp:{value:'202612'},f_entry:{value:'23000'},f_refnote:{textContent:''}};
  const timers=[],posts=[];
  const ctx={console,document:{addEventListener(){},getElementById:id=>fields[id]||null},window:{},location:{search:''},URLSearchParams,
    setTimeout:fn=>{timers.push(fn);return timers.length;},clearTimeout(){},setInterval(){},clearInterval(){},Date,structuredClone};
  vm.createContext(ctx);vm.runInContext(source,ctx);vm.runInContext("state.account='primary';updateReadout=()=>{};",ctx);
  const wanted={symbol:'MNQ',sec_type:'FUT',currency:'USD',exchange:'CME',expiry:'202612'};
  ctx.fetch=async(url,opts)=>{posts.push(JSON.parse(opts.body));return {ok:true,json:async()=>({ok:true,id:'quote-1'})};};
  ctx.fetchJSONOrNull=async()=>({query:{id:'quote-1',result:{instrument:wanted,last:30844.5,asof:Date.now()/1000,market_data_type:1,con_id:42}}});
  return {ctx,fields,timers,posts,wanted,run:code=>vm.runInContext(code,ctx)};
}
(async()=>{
  {
    const f=fixture();f.ctx.syncReferenceQuote();assert.equal(f.fields.f_entry.value,'');
    await f.timers.shift()();assert.equal(f.fields.f_entry.value,'30844.5');assert.deepEqual(f.posts[0],f.wanted);
    assert.match(f.fields.f_refnote.textContent,/Last price/);
    f.fields.f_futexp.value='202703';f.ctx.syncReferenceQuote();assert.equal(f.fields.f_entry.value,'');
  }
  for(const bad of [{market_data_type:3},{asof:Date.now()/1000-60},{last:NaN},{instrument:{symbol:'NQ'}},{con_id:0}]){
    const f=fixture();f.ctx.fetchJSONOrNull=async()=>({query:{id:'quote-1',result:{instrument:f.wanted,last:30844.5,asof:Date.now()/1000,market_data_type:1,con_id:42,...bad}}});
    f.ctx.syncReferenceQuote();await f.timers.shift()();assert.equal(f.fields.f_entry.value,'');
    assert.match(f.fields.f_refnote.textContent,/unavailable/);
  }
  {
    const f=fixture();let release,entered;
    const reading=new Promise(r=>{entered=r;});
    f.ctx.fetchJSONOrNull=()=>new Promise(r=>{release=r;entered();});
    f.ctx.syncReferenceQuote();const pending=f.timers.shift()();await reading;
    f.fields.f_symbol.value='MES';f.ctx.syncReferenceQuote();
    release({query:{id:'quote-1',result:{instrument:f.wanted,last:30844.5,asof:Date.now()/1000,market_data_type:1,con_id:42}}});
    await pending;assert.equal(f.fields.f_entry.value,'','old symbol response cannot repopulate field');
  }
  {
    const f=fixture();f.ctx.syncReferenceQuote();await f.timers.shift()();
    f.fields.f_entry.value='30800';f.ctx.referenceEdited();f.ctx.syncReferenceQuote(true);
    assert.equal(f.fields.f_entry.value,'30800','manual override survives poll');
    assert.equal(f.fields.f_refnote.textContent,'Manual reference');
    f.fields.f_entry_type.value='LMT';f.ctx.syncReferenceQuote();f.fields.f_entry.value='30000';
    f.ctx.syncEntryTypeFields();f.fields.f_entry_type.value='MKT';f.ctx.syncEntryTypeFields();f.ctx.syncReferenceQuote();
    await f.timers.at(-1)();f.fields.f_entry_type.value='LMT';f.ctx.syncEntryTypeFields();f.ctx.syncReferenceQuote();
    assert.equal(f.fields.f_entry.value,'30000','switching back restores typed limit');
  }
  {
    const f=fixture();f.fields.f_entry_type.value='STP_LMT';f.fields.f_entry.value='30010';f.ctx.syncReferenceQuote();
    assert.equal(f.timers.length,0);assert.equal(f.fields.f_entry.value,'30010');
  }
  {
    const f=fixture();f.fields.f_sectype.value='STK';f.fields.f_symbol.value='SPY';
    assert.equal(f.ctx.referenceInstrument().symbol,'SPY');
    f.fields.f_sectype.value='CASH';f.fields.f_symbol.value='EUR';f.fields.f_currency={value:'USD'};
    assert.equal(f.ctx.referenceInstrument().currency,'USD');
    f.fields.f_sectype.value='FUT';f.fields.f_futexp.value='';assert.equal(f.ctx.referenceInstrument(),null);
  }
  console.log('PASS exact last-price autofill, stale/delayed rejection, races, manual edits, limit preservation');
})().catch(e=>{console.error(e);process.exitCode=1;});
