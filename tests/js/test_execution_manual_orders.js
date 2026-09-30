"use strict";
const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const source = fs.readFileSync(path.join(__dirname, "../../site/assets/execution.js"), "utf8");

for (const account of ["primary", "pa"]) {
  const inputs = {me_qty: {value: "100000"}, me_lmt: {value: "-2.5"}};
  const sent = [], alerts = [];
  const context = {console, document: {addEventListener() {}, getElementById: id => inputs[id] || null},
    window: {}, location: {search: ""}, URLSearchParams, setTimeout, clearTimeout, setInterval, clearInterval,
    confirm() { throw new Error("Manual order action unexpectedly requested confirmation"); },
    alert: text => alerts.push(text), sent, account};
  vm.createContext(context);
  vm.runInContext(source, context);
  vm.runInContext(`state.account = account;
    sendCommand = (type, payload) => sent.push({type, account: state.account, payload});
    set = () => {}; renderOrders = () => '';
    orderEdit.orig = {qty: 100, lmt: 95, account, con_id: 42, client_id: 99};`, context);
  context.execCancel(701, 7, "SPY", 42, 99);
  context.execModifySave(701, 7, "SPY");
  assert.equal(sent.length, 2);
  assert.equal(sent[0].type, "cancel");
  assert.equal(sent[1].type, "modify");
  assert.equal(sent[1].payload.new_qty, 100000);
  assert.equal(sent[1].payload.new_limit, -2.5);
  assert.equal(sent[1].payload.con_id, 42);
  assert.equal(sent[1].payload.client_id, 99);
  assert.equal(sent[1].payload.mutation_kind, undefined);
  assert.equal(sent[1].payload.risk_usd, undefined);
  assert.equal(alerts.length, 0);
  vm.runInContext(`orderEdit.orig = {qty: 100, account: account === 'pa' ? 'primary' : 'pa'};`, context);
  context.execModifySave(701, 7, "SPY");
  assert.equal(sent.length, 2, "An account switch must not redirect an edit");
}
assert(!source.includes('id="me_risk"'));
assert(!source.includes('id="me_kind"'));
{
  const ctx = {console, document: {addEventListener() {}}, window: {}, location: {search: ""},
    URLSearchParams, setTimeout, clearTimeout, setInterval, clearInterval};
  vm.createContext(ctx); vm.runInContext(source, ctx);
  vm.runInContext(`state.account='primary';
    state.book={accounts:[{key:'primary',orders:[{symbol:'MES',sec_type:'FUT',con_id:42,
      order_ref:'MES|BUY|OpenBreakout|2026-09-30|ES-1-TIME'}]}]};`,ctx);
  assert.match(vm.runInContext("dayTradeControlNote({symbol:'MES',sec_type:'FUT',con_id:42})",ctx),/manual control/);
  assert.equal(vm.runInContext("dayTradeControlNote({symbol:'MES',sec_type:'FUT',con_id:43})",ctx),"");
  assert.equal(vm.runInContext("isDayTradeOrder({sec_type:'STK',order_ref:'MES|BUY|OpenBreakout|2026-09-30|ES-1-TIME'})",ctx),false);
}
console.log("PASS manual cancel/modify for Primary and PA: direct actions, no risk/purpose prompts, exact account routing");
