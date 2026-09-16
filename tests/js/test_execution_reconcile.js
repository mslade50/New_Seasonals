"use strict";
const assert = require("assert"), fs = require("fs"), path = require("path"), vm = require("vm");
const source = fs.readFileSync(path.join(__dirname, "../../site/assets/execution.js"), "utf8");
const position = {symbol:"TEST", con_id:42, sec_type:"STK", currency:"USD", position:100, account:"PRIMARY"};
function order(i, qty, group="a", extra={}) {
  return {symbol:"TEST", con_id:42, account:"PRIMARY", sec_type:"STK", action:"SELL",
    client_id:99, order_id:i, perm_id:1000+i, qty, filled:0, remaining:qty,
    oca_group:group, oca_type:1, order_type:i%2?"LMT":"MKT", lmt:120,
    good_after:i%2?null:"20261001 15:59:00 US/Eastern", status:"Submitted", ...extra};
}
for (const account of ["primary","pa"]) {
  const context={console,window:{},document:{addEventListener(){},getElementById:()=>null},
    location:{search:""},URLSearchParams,setTimeout,clearTimeout,setInterval,clearInterval,
    fmt:{money:String,num:String,pct:String},clsSign:()=>"",
    confirm(){throw Error("Reconcile should not require a second confirmation");},account};
  vm.createContext(context);vm.runInContext(source,context);
  const accountId=account.toUpperCase();
  function check(pos, orders, mismatch) {
    context.pos={...pos,account:accountId};
    context.orders=orders.map(o=>({...o,account:o.account==="PRIMARY"?accountId:o.account}));
    vm.runInContext(`state.account=account;state.book={accounts:[{key:account,positions:[pos],orders}]};`,context);
    assert.equal(context.exitCoverage(context.pos).mismatch,mismatch);
    assert.equal(context.renderPositions().includes(">Reconcile</button>"),mismatch);
  }
  check(position,[],false);
  check(position,[order(1,100),order(2,100)],false);
  check(position,[order(1,60),order(2,60),order(3,40,"b"),order(4,40,"b")],false);
  check(position,[order(1,120),order(2,120)],true);
  check(position,[order(1,80),order(2,80)],true);
  check(position,[order(1,80),order(2,100)],true);
  check({...position,position:75},[order(1,100,"a",{filled:25}),order(2,75)],false);
  check(position,[order(1,150,"a",{con_id:43})],false);
  check(position,[order(1,150,"a",{account:"UNRELATED"})],false);
  check(position,[order(1,150,"a",{status:"Cancelled"})],false);
  check(position,[order(10,100,"",{action:"BUY"}),order(11,100,"a",{parent_id:10})],false);
  check({...position,position:-100},[order(1,120,"a",{action:"BUY"}),order(2,120,"a",{action:"BUY"})],true);
  check({...position,position:1.5},[order(1,.7),order(2,.7),order(3,.8,"b"),order(4,.8,"b")],false);
  check(position,[order(1,150),order(2,150)],true);
  vm.runInContext(`sendCommand=(type,payload)=>globalThis.sent={type,payload,account:state.account};execReconcileExits(pos);`,context);
  assert.equal(context.sent.type,"reconcile_exits");assert.equal(context.sent.account,account);
  assert.equal(context.sent.payload.con_id,42);assert.equal(context.sent.payload.qty,undefined);
  check(position,[order(1,100),order(2,100)],false);
  vm.runInContext(`sent=null;execReconcileExits(pos);`,context);
  assert.equal(context.sent,null,"A repaired row must not submit another reconcile");
}
console.log("PASS conditional Reconcile button: both accounts, OCA totals, unequal siblings, partial fills, pending entries, exact identity, long/short and fractional quantities");
