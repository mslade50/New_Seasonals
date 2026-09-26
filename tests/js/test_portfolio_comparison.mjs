import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const c={console,document:{addEventListener(){}}};
vm.createContext(c);
vm.runInContext(fs.readFileSync(new URL('../../site/assets/comparison.js',import.meta.url),'utf8'),c);
const trades={n:4,columns:{Strategy:['Algo','Algo','Algo','Algo'],Exit_Date:['2026-09-01','2026-09-02','2026-09-08','2026-09-03'],Open:[false,true,false,false],PnL_flat:[10,99,88,NaN]}};
const fills=[
 {account_key:'primary',time:'2026-09-01T15:00:00Z',order_ref:'SPY|SELL|Algo|2026-08-28',realized_pnl:8,commission:1},
 {account_key:'primary',time:'2026-09-02T15:00:00Z',order_ref:'SPY|BUY|Algo|2026-09-01',realized_pnl:null,commission:null},
 {account_key:'pa',time:'2026-09-01T15:00:00Z',order_ref:'SPY|SELL|Algo|2026-08-28',realized_pnl:700},
 {account_key:'primary',time:'2026-09-01T15:00:00Z',order_ref:'manual',realized_pnl:900},
 {account_key:'primary',time:'2026-09-01T00:00:00Z',order_ref:'SPY|SELL|Algo|2026-08-28',realized_pnl:300},
];
const result=JSON.parse(JSON.stringify(c.comparisonRows(trades,fills,['Algo'],'2026-09-01','2026-09-06')));
assert.deepEqual(result,{rows:[{strategy:'Algo',theo:10,theo_closed:1,actual:8,actual_fills:2,actual_missing:1,commissions:1}],actualOnly:[],excluded:1,missingPnl:1});
assert.equal(c.comparisonRows({n:0},fills.slice(1,2),['Algo'],'2026-09-01','2026-09-06').rows[0].actual,null);

// Catalog tags without a ledger replay leave "excluded" for the Actual-only
// section; unknown tags stay excluded; a roster tag stays in the roster.
const catalog={strategies:[
 {id:'open_breakout',name:'NQ/ES Opening Breakout',order_ref_tags:['OpenBreakout']},
 {id:'legend_ema',name:'Legend EMA',order_ref_tags:['Legend_EMA']},
 {id:'algo',name:'Algo',order_ref_tags:['Algo']},
]};
const tagMap=c.catalogTagMap(catalog);
assert.equal(tagMap.get('OpenBreakout'),'NQ/ES Opening Breakout');
assert.equal(c.catalogTagMap(null).size,0);
const liveFills=[
 {account_key:'primary',time:'2026-09-02T15:00:00Z',order_ref:'MES|BUY|OpenBreakout|2026-09-02|ES-A1-ENTRY',symbol:'MES',sec_type:'FUT',side:'BOT',qty:1,price:7780,multiplier:'5',realized_pnl:0,commission:0.61},
 {account_key:'primary',time:'2026-09-02T15:30:00Z',order_ref:'MES|SELL|OpenBreakout|2026-09-02|ES-A-TIME',symbol:'MES',sec_type:'FUT',side:'SLD',qty:1,price:7776,realized_pnl:-22.47,commission:0.61},
 {account_key:'primary',time:'2026-09-02T14:30:00Z',order_ref:'SPY|BUY|Legend_EMA|2026-09-01',symbol:'SPY',sec_type:'STK',side:'BOT',qty:2,price:700,realized_pnl:null},
 {account_key:'primary',time:'2026-09-02T14:30:00Z',order_ref:'QQQ|BUY|MysteryTag|2026-09-01',symbol:'QQQ',sec_type:'STK',side:'BOT',qty:1,price:500,realized_pnl:5},
 {account_key:'primary',time:'2026-09-02T14:30:00Z',order_ref:'SPY|SELL|Algo|2026-08-28',symbol:'SPY',sec_type:'STK',side:'SLD',qty:1,price:700,realized_pnl:4},
];
const withCatalog=JSON.parse(JSON.stringify(c.comparisonRows({n:0},liveFills,['Algo'],'2026-09-01','2026-09-06',tagMap)));
assert.equal(withCatalog.excluded,1,'only the unknown MysteryTag is excluded');
assert.deepEqual(withCatalog.rows.map(r=>[r.strategy,r.actual_fills,r.actual]),[['Algo',1,4]]);
assert.deepEqual(withCatalog.actualOnly,[
 {strategy:'Legend_EMA',name:'Legend EMA',fills:1,symbols:'SPY',buys:1,sells:0,notional:1400,notional_unknown:0,actual:null,actual_missing:1,commissions:null},
 {strategy:'OpenBreakout',name:'NQ/ES Opening Breakout',fills:2,symbols:'MES',buys:1,sells:1,notional:38900,notional_unknown:1,actual:-22.47,actual_missing:0,commissions:1.22},
]);
assert.equal(withCatalog.missingPnl,1);
// A wildcard catalog tag (Pitch-*) matches by prefix and groups under the wildcard.
const pitchMap=c.catalogTagMap({strategies:[{id:'daily_pitch',name:'Daily Pitch',order_ref_tags:['Pitch-*']}]});
const pitch=c.comparisonRows({n:0},[
 {account_key:'primary',time:'2026-09-02T14:30:00Z',order_ref:'XLE|BUY|Pitch-2026-09-02-a|2026-09-02',symbol:'XLE',side:'BOT',qty:10,price:90,realized_pnl:0},
 {account_key:'primary',time:'2026-09-02T14:30:00Z',order_ref:'XLU|BUY|Pitch-2026-09-02-b|2026-09-02',symbol:'XLU',side:'BOT',qty:5,price:80,realized_pnl:0},
 {account_key:'primary',time:'2026-09-02T14:30:00Z',order_ref:'XLU|BUY|Pitchfork|2026-09-02',symbol:'XLU',side:'BOT',qty:5,price:80,realized_pnl:0},
],[],'2026-09-01','2026-09-06',pitchMap);
assert.equal(pitch.actualOnly.length,1);
assert.equal(pitch.actualOnly[0].strategy,'Pitch-*');
assert.equal(pitch.actualOnly[0].fills,2);
assert.equal(pitch.actualOnly[0].symbols,'XLE, XLU');
assert.equal(pitch.excluded,1,'Pitchfork does not match the Pitch- prefix');
// Without a catalog, the old behavior holds: both new tags count as excluded.
assert.equal(c.comparisonRows({n:0},liveFills,['Algo'],'2026-09-01','2026-09-06').excluded,4);
console.log('PASS comparison excludes PA, untagged, open/future trades, preserves unknown PnL, and routes catalog tags to Actual-only');
