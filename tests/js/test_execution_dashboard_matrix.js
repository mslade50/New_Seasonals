"use strict";
const assert = require("assert"), fs = require("fs"), path = require("path"), vm = require("vm");
const source = fs.readFileSync(path.join(__dirname, "../../site/assets/execution.js"), "utf8");
const failures = [];
let checks = 0;
async function check(name, test) { try { await test(); checks++; } catch (error) { failures.push(name + ": " + error.stack); } }
function fixture() {
  const nodes = new Map(), commands = [], confirms = [];
  const attribute = (html, name) => new RegExp("\\b" + name + '="([^"]*)"').exec(html)?.[1];
  function remove(id) {
    for (const child of nodes.get(id)?.descendants || []) remove(child);
    nodes.delete(id);
  }
  function node(id, value = "") {
    const el = {id,value,checked:false,style:{},textContent:"",tagName:"INPUT",focus(){},scrollIntoView(){},
      addEventListener(event,callback){this[event]=callback;},_html:"",descendants:[],
      get innerHTML(){return this._html;},
      set innerHTML(html){
        for(const child of this.descendants) remove(child);
        this._html=html;this.descendants=[];
        for(const match of html.matchAll(/<([a-z]+)\b([^>]*\bid="([^"]+)"[^>]*)>/gi)){
          const child=node(match[3],attribute(match[2],"value")||"");
          child.tagName=match[1].toUpperCase();child.checked=/\bchecked\b/.test(match[2]);
          if(child.tagName==="SELECT"){
            const end=html.indexOf("</select>",match.index);
            const options=[...html.slice(match.index,end).matchAll(/<option\b([^>]*)>([^<]*)/g)];
            const selected=options.find(o=>/\bselected\b/.test(o[1]))||options[0];
            child.value=selected?(attribute(selected[1],"value")||selected[2]):"";
          }
          this.descendants.push(child.id);
        }
      }};
    nodes.set(id,el);return el;
  }
  const c={console,window:{},location:{search:""},URLSearchParams,Intl,structuredClone,
    document:{addEventListener(){},querySelectorAll:()=>[],getElementById:id=>nodes.get(id)||null},
    setTimeout:()=>0,clearTimeout(){},setInterval:()=>0,clearInterval(){},
    fmt:{money:String,num:String,pct:String},clsSign:()=>"",confirm:text=>{confirms.push(text);return true;},alert:text=>{c.lastAlert=text;},
    fetch:async()=>{throw Error("Unexpected network request");},fetchJSONOrNull:async()=>{throw Error("Unexpected network read");},
    Date:class extends Date{constructor(...args){super(...(args.length?args:["2026-09-14T20:00:00Z"]));}static now(){return new Date("2026-09-14T20:00:00Z").valueOf();}}
  };
  vm.createContext(c);vm.runInContext(source,c);
  const run=code=>vm.runInContext(code,c);
  c.capture=(type,payload,msg,context={})=>{commands.push({type,payload:JSON.parse(JSON.stringify(payload)),context});return Promise.resolve("fixture");};
  run("sendCommand=capture;state.status={online:true};state.book={at:Date.now(),mode:'dry-run',accounts:[{key:'primary',nlv:1000000,positions:[],orders:[]},{key:'pa',nlv:100000,positions:[],orders:[]}]};FUT_SPECS={MES:{exchange:'CME',multiplier:5,min_tick:.25}}");
  function fields(values){nodes.clear();for(const [id,value]of Object.entries(values))node(id,String(value));node("cmdMsg");node("ticketReadout");}
  return {c,run,json:code=>JSON.parse(run("JSON.stringify("+code+")")),fields,node,nodes,commands,confirms};
}
const position={symbol:"TEST",sec_type:"STK",con_id:42,position:100,avg_cost:100,market_price:100};
function book(f,positions=[position],orders=[]){f.run("for(const a of state.book.accounts){a.positions="+JSON.stringify(positions)+";a.orders="+JSON.stringify(orders)+"}");}
const entry={cmdType:"entry_bracket",f_symbol:"TEST",f_sectype:"STK",f_action:"BUY",f_qty:"10",f_entry:"100",f_stop:"90",f_target:"120",f_entry_cap:"105",f_entry_type:"LMT",f_futexp:"202612",f_futexch:"CME",f_currency:"USD",f_expiry:"",f_timestop:""};
const close={cmdType:"close_resize",f_symbol:"TEST",fl_qty:"25",fl_pct:"",fl_type:"MKT",fl_tif:"DAY",fl_limit:"101"};
const schedule={cmdType:"scheduled_option",so_symbol:"SPY",so_right:"P",so_delta:"0.15",so_budget:"1000",so_date:"2026-09-14",so_time:"16:01",so_expiry_mode:"min_dte",so_min_dte:"30",so_expiry:""};
(async()=>{
await check("all entry types, sides and account payloads",async()=>{
 for(const account of ["primary","pa"])for(const sectype of ["STK","FUT","CASH"])for(const type of ["LMT","STP_LMT","MKT","MOO","MOC"])for(const action of ["BUY","SELL"]){
  const f=fixture();f.run("state.account='"+account+"'");
  f.fields({...entry,f_sectype:sectype,f_symbol:sectype==="FUT"?"MES":sectype==="CASH"?"EUR":"TEST",f_entry_type:type,f_action:action,f_stop:action==="BUY"?"90":"110",f_target:action==="BUY"?"120":"80",f_entry_cap:action==="BUY"?"105":"95"});
  const allowed=sectype==="STK"||(sectype==="FUT"&&["LMT","MKT","MOO"].includes(type))||(sectype==="CASH"&&["LMT","MKT"].includes(type));
  assert.equal(f.json("bracketWarnings()").length===0,allowed,[account,sectype,type,action].join(" "));
  f.run("sendTicket()");await Promise.resolve();assert.equal(f.commands.length,Number(allowed));
  if(allowed){const x=f.commands[0];assert.equal(x.type,"entry_bracket");assert.equal(x.payload.action,action);assert.equal(x.payload.sec_type,sectype);assert.equal(x.context.account,account);assert.equal(x.context.dryRun,true);assert.equal(x.payload.entry_type,type);}
 }
});
await check("ticket rebuild preserves side/instrument and close settings",()=>{
 const f=fixture();f.fields({cmdType:"entry_bracket"});f.node("cmdFields");f.run("syncFields()");
 f.nodes.get("f_action").value="SELL";f.nodes.get("f_sectype").value="FUT";f.nodes.get("f_sectype").change();
 f.nodes.get("f_symbol").value="MES";f.nodes.get("f_symbol").input();f.nodes.get("f_futexp").value="202612";f.nodes.get("f_futexp").input();
 f.nodes.get("cmdType").value="echo";f.run("syncFields()");
 assert.equal(f.nodes.has("f_action"),false);assert.equal(f.nodes.has("f_futexp"),false,"removed nested controls must not remain in the fixture DOM");
 f.nodes.get("cmdType").value="entry_bracket";f.run("syncFields()");
 assert.equal(f.nodes.get("f_action").value,"SELL");assert.equal(f.nodes.get("f_sectype").value,"FUT");assert.equal(f.nodes.get("f_symbol").value,"MES");assert.equal(f.nodes.get("f_futexp").value,"202612");
 f.fields(close);f.node("cmdFields");f.nodes.get("fl_type").value="LMT";f.nodes.get("fl_tif").value="GTC";f.node("fl_rth").checked=true;
 f.run("syncFields()");f.nodes.get("cmdType").value="echo";f.run("syncFields()");f.nodes.get("cmdType").value="close_resize";f.run("syncFields()");
 assert.equal(f.nodes.get("fl_type").value,"LMT");assert.equal(f.nodes.get("fl_tif").value,"GTC");assert.equal(f.nodes.get("fl_rth").checked,true);
});
await check("ambiguous position never selects first contract",()=>{
 const f=fixture();book(f,[{...position,sec_type:"FUT",con_id:1,expiry:"202609"},{...position,sec_type:"FUT",con_id:2,expiry:"202612"}]);
 f.fields({...close,cmdType:"exit_attach",f_stop:"90"});assert.equal(f.run("attachPosition()"),null);assert.ok(f.json("attachWarnings()").length);
 f.fields(close);assert.ok(f.json("flattenWarnings()").length);f.run("sendTicket()");assert.equal(f.commands.length,0);
 f.run("ticketDraft.fl_position={account:'primary',symbol:'TEST',con_id:2}");assert.equal(f.json("ticketPayload('close_resize')").con_id,2);
});
await check("schedule cancel retains originating account",()=>{
 const f=fixture();f.run("state.account='pa';state.commands=[{id:'schedule-1',type:'scheduled_option',account:'primary',state:'scheduled'}];cancelScheduledOption('schedule-1','primary')");
 assert.equal(f.commands[0].context.account,"primary");assert.match(f.confirms[0],/on primary/);
});
await check("ET schedule validation independent of browser timezone",()=>{
 const saved=process.env.TZ;
 try{for(const tz of ["Pacific/Honolulu","Asia/Tokyo","America/New_York"]){process.env.TZ=tz;const f=fixture();f.fields(schedule);assert.deepEqual(f.json("scheduledOptionWarnings()"),[],tz);f.nodes.get("so_time").value="15:59";assert.ok(f.json("scheduledOptionWarnings()").some(x=>x.includes("future")),tz);}}
 finally{if(saved===undefined)delete process.env.TZ;else process.env.TZ=saved;}
});
await check("nonfinite numeric values block before commands",()=>{
 for(const key of ["f_qty","f_entry","f_stop","f_target","f_entry_cap"]){const f=fixture();f.fields({...entry,f_entry_type:"STP_LMT",[key]:"Infinity"});assert.ok(f.json("bracketWarnings()").length,key);f.run("sendTicket()");assert.equal(f.commands.length,0,key);}
 const f=fixture();f.fields({...schedule,so_budget:"Infinity"});assert.ok(f.json("scheduledOptionWarnings()").length);f.run("sendTicket()");assert.equal(f.commands.length,0);
});
await check("close/add retain account and exact position identity",async()=>{
 for(const account of ["primary","pa"])for(const sign of [1,-1])for(const type of ["close_only","close_resize","flatten","add_to_position"]){const f=fixture();book(f,[{...position,position:100*sign}]);f.run("state.account='"+account+"'");f.fields({...close,cmdType:type});f.run("sendTicket()");await Promise.resolve();assert.equal(f.commands.length,1);assert.equal(f.commands[0].payload.con_id,42);assert.equal(f.commands[0].payload.expected_position,100*sign);assert.equal(f.commands[0].payload.qty,25);assert.equal(f.commands[0].context.account,account);if(["close_only","close_resize"].includes(type))assert.equal(f.commands[0].payload.action,sign>0?"SELL":"BUY");}
});
await check("fractional edit and account switch never submit",()=>{
 const f=fixture();f.fields({me_qty:"1.5"});f.run("orderEdit.orig={account:'primary',con_id:42,client_id:99,qty:100};execModifySave(7,8,'TEST')");assert.equal(f.commands.length,0);
 f.run("orderEdit.key='7:8';renderPanels=()=>{};setAccount('pa')");assert.equal(f.run("orderEdit.key"),null);
});
await check("working order controls match type",()=>{
 const f=fixture();for(const type of ["LMT","STP","STP LMT","MKT","MOC","MOO"]){const html=f.c.orderEditRow({...position,action:"SELL",qty:100,order_type:type,perm_id:7,order_id:8,client_id:99,lmt:120,aux:90});assert.match(html,/id="me_qty"/);assert.equal(html.includes('id="me_lmt"'),type.includes("LMT"));assert.equal(html.includes('id="me_stp"'),type.startsWith("STP"));assert.doesNotMatch(html,/me_(kind|risk|direction)/);}
});
await check("futures sizing validates finite positive inputs before lookup",async()=>{
 for(const patch of [{fs_entry:"Infinity"},{fs_stop:"-1"},{fs_stop:"100"},{fs_risk:"bad"},{fs_risk:"-1"},{fs_target:"-1"},{fs_riskpct:"1"}]){
  const f=fixture();f.fields({fs_symbol:"MES",fs_entry:"100",fs_stop:"90",fs_target:"120",fs_risk:"1000",fs_riskpct:"",...patch});f.node("fs_msg");let reads=0;f.c.fetch=async()=>{reads++;return {ok:true,json:async()=>({ok:false,error:"fixture"})};};
  await f.c.sizeFutures();assert.equal(reads,0,JSON.stringify(patch));
 }
});
await check("sizing results cannot cross account changes",async()=>{
 const f=fixture();f.fields({fs_symbol:"MES",fs_entry:"100",fs_stop:"90",fs_target:"120",fs_risk:"",fs_riskpct:"1"});f.node("fs_msg");f.node("fs_result");let release;
 f.c.fetch=async(url,options)=>{assert.equal(url,"/exec-futures-size");assert.equal(JSON.parse(options.body).account_key,"primary");return new Promise(resolve=>{release=resolve;});};
 f.c.fetchJSONOrNull=async()=>({query:{id:"sizing-1",result:{symbol:"MES",contracts:99}}});
 const pending=f.c.sizeFutures();f.run("renderPanels=()=>{};setAccount('pa')");release({ok:true,json:async()=>({ok:true,id:"sizing-1"})});await pending;await Promise.resolve();await Promise.resolve();
 assert.doesNotMatch(f.nodes.get("fs_result").innerHTML,/99 contract/);assert.equal(f.run("sizeState.id"),null);
});
await check("new front resolve cannot be overwritten by older response",async()=>{
 const f=fixture();f.fields({...entry,f_sectype:"FUT",f_symbol:"MES"});f.node("f_futnote");const releases=[];
 f.c.fetch=async()=>new Promise(resolve=>releases.push(resolve));f.c.fetchJSONOrNull=async()=>({});
 const old=f.c.resolveFront();const latest=f.c.resolveFront();
 releases[1]({ok:true,json:async()=>({ok:true,id:"new"})});await latest;
 releases[0]({ok:true,json:async()=>({ok:true,id:"old"})});await old;
 assert.equal(f.run("frontState.id"),"new");
});
await check("futures expiries remain separate order groups",()=>{
 const f=fixture();const positions=[{...position,symbol:"MES",sec_type:"FUT",con_id:1,expiry:"202609"},{...position,symbol:"MES",sec_type:"FUT",con_id:2,expiry:"202612"}];
 const orders=positions.map((p,index)=>({...p,action:"SELL",qty:100,order_type:"LMT",lmt:120,order_id:10+index,perm_id:20+index}));book(f,positions,orders);
 assert.notEqual(f.c.orderGroupKey(positions[0]),f.c.orderGroupKey(positions[1]));const html=f.run("renderOrders()");assert.match(html,/exec-orders-MES%20202609/);assert.match(html,/exec-orders-MES%20202612/);
});
await check("attach combinations, sides and accounts validate and bind",async()=>{
 for(const account of ["primary","pa"])for(const sign of [1,-1])for(const mask of [1,2,3,4,5,6,7]){
  const f=fixture();book(f,[{...position,position:sign*100}]);f.run("state.account='"+account+"'");f.fields({cmdType:"exit_attach",f_symbol:"TEST",f_stop:mask&1?(sign>0?"90":"110"):"",f_target:mask&2?(sign>0?"120":"80"):"",f_timestop:mask&4?"2099-12-15":""});
  assert.deepEqual(f.json("attachWarnings()"),[]);f.run("sendTicket()");await Promise.resolve();assert.equal(f.commands.length,1);assert.equal(f.commands[0].payload.con_id,42);assert.equal(f.commands[0].context.account,account);assert.equal(f.commands[0].payload.expected_position,undefined);
 }
});
await check("close quantity, percentage, outside-hours and ownership gates",()=>{
 for(const patch of [{fl_qty:"0"},{fl_qty:"101"},{fl_qty:"1.5"},{fl_qty:"",fl_pct:"0"},{fl_qty:"",fl_pct:"101"},{fl_type:"LMT",fl_limit:""}]){const f=fixture();book(f);f.fields({...close,...patch});assert.ok(f.json("flattenWarnings()").length,JSON.stringify(patch));f.run("sendTicket()");assert.equal(f.commands.length,0);}
 const f=fixture();book(f);f.fields(close);f.node("fl_rth").checked=true;assert.ok(f.json("flattenWarnings()").length);f.nodes.get("fl_type").value="LMT";assert.deepEqual(f.json("flattenWarnings()"),[]);
});
await check("hedge controls change display preferences without commands",()=>{
 const f=fixture();const mount=f.node("hedge");const beta=f.node("hedge_beta","beta63"),contract=f.node("hedge_contract","ES"),target=f.node("hedge_target","175"),strategy=f.node("scope");strategy.dataset={hedgeStrategy:encodeURIComponent("Oversold Low Volume")};strategy.checked=false;
 mount.querySelector=selector=>f.nodes.get(selector.slice(1));mount.querySelectorAll=()=>[strategy];f.run("refreshHedge=()=>{};updateHedgeTargetViews=()=>{};bindHedgeControls()");beta.change();contract.change();target.input();strategy.change();
 assert.equal(f.run("hedgePrefs.betaKey"),"beta63");assert.equal(f.run("hedgePrefs.contract"),"ES");assert.equal(f.run("hedgePrefs.targetPct"),150);assert.equal(f.run("hedgeScopeForAccount().size"),0);assert.equal(f.commands.length,0);
});
await check("activity escapes messages and distinguishes terminal outcomes",()=>{
 const f=fixture();const states=["dry_run","rejected","executed","scheduled","executing","cancelled","expired","unknown","error"];
 for(const state of states){const html=f.c.resultCell({id:"abc",type:"echo",account:"primary",state,result:{detail:"<script>unsafe</script>"}});assert.match(html,/&lt;script&gt;/);assert.doesNotMatch(html,/<script>/);assert.ok(f.c.stateBadge(state));}
 assert.match(f.c.stateBadge("unknown"),/VERIFY IN TWS/);f.run("state.commands=[]");assert.equal(f.run("renderActivity()"),"");assert.equal(f.commands.length,0);
});
await check("IB contract cost displays in quoted price units",()=>{
 const f=fixture();assert.equal(f.c.quotedAverageCost({...position,symbol:"MES",sec_type:"FUT",avg_cost:38204}),7640.8);
 assert.equal(f.c.quotedAverageCost({...position,sec_type:"OPT",multiplier:100,avg_cost:305}),3.05);
 assert.equal(f.c.quotedAverageCost({...position,sec_type:"FUT",symbol:"UNKNOWN"}),null);
 const model=f.c.attributeBook({key:"primary",positions:[{...position,symbol:"MES",sec_type:"FUT",position:2,avg_cost:38204,market_price:null}],orders:[]},null,{MES:{multiplier:5}},{today:"20260914"});
 assert.equal(model.futures[0].price,7640.8);assert.equal(model.futures[0].spyEquiv,76408);
});
await check("echo sends note only and needs no trade confirmation",async()=>{
 const f=fixture();f.fields({cmdType:"echo",f_note:"offline ping"});f.run("sendTicket()");await Promise.resolve();assert.equal(f.confirms.length,0);assert.equal(f.commands[0].type,"echo");assert.deepEqual(f.commands[0].payload,{note:"offline ping"});
});
await check("credit combo quantity edits preserve the existing signed limit",()=>{
 const f=fixture();const order={symbol:"TEST",sec_type:"BAG",con_id:42,action:"BUY",order_type:"LMT",qty:2,lmt:-1.25,order_id:8,perm_id:7,client_id:99};
 book(f,[],[order]);f.node("orders");f.run("expandedTickers.add('TEST');execModifyStart('7:8')");
 assert.equal(f.nodes.get("me_lmt").value,"-1.25");f.nodes.get("me_qty").value="1";f.run("execModifySave(7,8,'TEST')");
 assert.equal(f.commands.length,1);assert.equal(f.commands[0].payload.new_qty,1);assert.equal(f.commands[0].payload.new_limit,undefined);assert.equal(f.commands[0].payload.con_id,42);assert.equal(f.confirms.length,0);
});
await check("signed combo limits stay distinct from single-option prices",()=>{
 for(const [sectype,original,changed,allowed]of [["BAG",-1.25,-1.5,true],["OPT",1.25,1.5,true],["OPT",1.25,-1.5,false],["BAG",-1.25,Infinity,false]]){
  const f=fixture();f.fields({me_qty:"2",me_lmt:String(changed)});f.run("orderEdit.orig="+JSON.stringify({account:"primary",sec_type:sectype,con_id:42,client_id:99,qty:2,lmt:original})+";execModifySave(7,8,'TEST')");
  assert.equal(f.commands.length,Number(allowed),sectype+" "+changed);if(allowed){assert.equal(f.commands[0].payload.new_limit,changed);assert.equal(f.commands[0].payload.new_qty,undefined);assert.equal(f.confirms.length,0);}
 }
});
if(failures.length){console.error(failures.join("\n\n"));process.exitCode=1;}else console.log("PASS "+checks+" execution dashboard control matrices, offline only");
})();
