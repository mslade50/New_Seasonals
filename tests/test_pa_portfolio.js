'use strict';
const assert = require('node:assert/strict');
const {quantity,replay,metrics,validate} = require('../site/assets/pa-portfolio.js');
const fs = require('node:fs'), vm = require('node:vm'), path = require('node:path');
const pages = vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../site/assets/common.js'),'utf8') + '\nJSON.stringify(PAGES)');
const nav = JSON.parse(pages);
assert.deepEqual(nav.slice(0,3).map(p=>p.label),['Execution','Portfolio','Seasonal']);
assert.ok(nav.findIndex(p=>p.href==='pa-portfolio.html')>=3);
assert.equal(nav.filter(p=>p.href==='pa-portfolio.html').length,1);
const config = {primary_anchor:750000, risk_multiplier:1.3};
// Synthetic capital chosen near a whole-share boundary; no account balance
// belongs in repository source or the site's payload.
assert.deepEqual([945,1360,33].map(q=>quantity(q,168700,config)),[276,397,9]);
assert.equal(quantity(1,10000,config),0);
const dates = ['2024-01-02','2024-01-03','2024-01-04','2024-01-05','2024-01-08'];
const trade = overrides => ({stage:0,entry:0,exit:2,primary_qty:100,risk_per_share:2,
  unit_pnl:[10,-5,5],...overrides});
const data = {version:1,config:{primary_anchor:1000,risk_multiplier:1.3},dates,
  trades:[trade(),trade({stage:1,entry:1,exit:3,primary_qty:100,unit_pnl:[0,0,10]})]};
const run = replay(data,1000);
assert.deepEqual(run.orders.map(t=>t.qty),[130,299]); // 2300 prior EOD equity
assert.deepEqual(run.pnl,[1300,-650,650,2990,0]);
assert.deepEqual(run.equity,[2300,1650,2300,5290,5290]);
assert.deepEqual(replay(data,1000,{compound:false}).orders.map(t=>t.qty),[130,130]);
assert.equal(run.orders[0].qty,130); // no resizing of its held quantity
assert.equal(run.returns.at(-1),0); // zero trading day retained
assert.deepEqual(replay(data,1000,{start:'2024-01-03'}).orders.map(t=>t.qty),[130]);
const sameDay = {...data,trades:[trade({primary_qty:50}),trade({primary_qty:50})]};
assert.deepEqual(replay(sameDay,1000).orders.map(t=>t.qty),[65,65]);
assert.equal(replay({...data,trades:[trade({primary_qty:1})]},1).orders.length,0);
const m = metrics(run);
assert.ok(Math.abs(m.maxDD - (1650/2300-1)) < 1e-12);
assert.equal(m.peak,'2024-01-02'); assert.equal(m.trough,'2024-01-03');
assert.equal(m.recovery,'2024-01-04'); assert.equal(m.underwaterSessions,2);
assert.equal(m.annualized,m.daily*Math.sqrt(252));
assert.throws(()=>validate({...data,trades:[trade({unit_pnl:[10]})]}),/MTM/);
assert.throws(()=>replay(data,0),/starting equity/);
const bankrupt = replay({...data,trades:[trade({unit_pnl:[-20,0,0]})]},1000);
assert.equal(bankrupt.equity[0],-1600); assert.equal(metrics(bankrupt).insolvent,true);
console.log('PA browser replay: whole-share sizing, flooring, overlaps, staging equity, fixed/scaled sizing, zero days, restart, drawdown/recovery, invalid data, insolvency passed.');
