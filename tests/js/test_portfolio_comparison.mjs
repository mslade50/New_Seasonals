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
assert.deepEqual(result,{rows:[{strategy:'Algo',theo:10,theo_closed:1,actual:8,actual_fills:2,actual_missing:1,commissions:1}],excluded:1,missingPnl:1});
assert.equal(c.comparisonRows({n:0},fills.slice(1,2),['Algo'],'2026-09-01','2026-09-06').rows[0].actual,null);
console.log('PASS comparison excludes PA, untagged, open/future trades and preserves unknown PnL');
