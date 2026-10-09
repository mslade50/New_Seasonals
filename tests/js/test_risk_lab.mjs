import assert from 'node:assert/strict';
import fs from 'node:fs';
const source=fs.readFileSync(new URL('../../site/assets/risk_lab_core.js',import.meta.url),'utf8');
const c=await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const dates=Array.from({length:30},(_,i)=>`2026-01-${String(i+1).padStart(2,'0')}`);
const all=dates.slice(0,20), reduced=[dates[0],dates[11]];
const d={asof:dates.at(-1),dates,spy_series:{close:dates.map((_,i)=>100+i)},
  sizing_state:{asof:dates.at(-1)},forward_returns:{'63d':{current_score:50,band_low:45,band_high:55,episode_dates:reduced}},
  return_samples:{version:1,asof:dates.at(-1),score_asof:dates.at(-1),current_score:50,band_low:45,band_high:55,
    all:{episode_dates:all,outcomes:{21:[{date:dates[0],status:'complete',value:.123,endDate:dates[21]}]},returns:{21:{mean:.123,n:1}}},
    reduced:{episode_dates:reduced}}};
assert.equal(c.validFullSample(d),true);
assert.equal(c.anchorsOf(d,'all').length,20);
assert.equal(c.anchorsOf(d).length,2);
assert.equal(c.resolveState(d,new URLSearchParams('sample=all&episode='+dates[1])).selected,dates[1]);
assert.equal(c.resolveState(d,new URLSearchParams('sample=reduced&episode='+dates[1])).selected,null);
assert.equal(c.episodeOutcome(d,dates[0],21,'SPY','all').value,.123);
assert.equal(c.returnSample(d,21,'all').all[0].value,.123);
assert.equal(c.returnStats(d,21,'all').mean,.123);
assert.equal(c.validFullSample({...d,return_samples:{...d.return_samples,score_asof:dates[0]}}),false);
assert.equal(c.resolveState({...d,return_samples:null},new URLSearchParams('sample=all')).sample,'reduced');
const missing={...d,return_samples:null,spy_series:{close:[100,null,...Array(28).fill(110)]}};
assert.equal(c.episodeOutcome(missing,dates[0],1).status,'unavailable');
assert.equal(c.pathsFor(missing,1)[0].complete,false);
assert.deepEqual(c.WINDOWS,[5,10,21]);
c.assertSharedRisk({shared_redacted:true,sizing_state:{score:50}});
for(const key of ['threshold','throttle_on','exposure','sleeve','basis'])
  assert.throws(()=>c.assertSharedRisk({shared_redacted:true,sizing_state:{[key]:null}}));
assert.throws(()=>c.assertSharedRisk({sizing_state:{score:50}}));
console.log('Risk Lab core checks passed: exact ledgers, cohort switching, stale data, missing prices and shared privacy.');

assert.equal(c.resolveState(d).sample,'all');
assert.equal(c.resolveState(d,new URLSearchParams('sample=reduced')).sample,'reduced');
const row=(date,status,hit)=>({date,anchor_date:date,status,
  max_drawdown_atr:hit?2.1:0,breaches:{1:hit,2:hit,3:false,5:false},iv_change_points:null});
const downs={...d,atr_downside:{baseline:{'5d':{1:10,2:10,3:10,5:10}},dial:{table:{'5d':{2:99}}}},
  downside_samples:{version:1,asof:d.asof,score_asof:d.asof,current_score:50,band_low:45,band_high:55,
    all:{episode_dates:all,windows:{5:{outcomes:all.map((date,i)=>row(date,i<3?'complete':'incomplete',i===0))}}},
    reduced:{episode_dates:reduced,windows:{5:{outcomes:[row(reduced[0],'complete',true),row(reduced[1],'incomplete',false)]}}}}};
const fullDD=c.drawdownSample(downs,5,2,'all');
assert.equal(fullDD.n,3);assert.equal(fullDD.hits,1);assert.equal(fullDD.selected,20);
assert.equal(fullDD.pending,17);assert.equal(fullDD.rate,1/3);assert.equal(fullDD.ivN,0);
assert.equal(c.drawdownSample(downs,5,2,'reduced').rate,1);
assert.ok(Math.abs(c.downsideCells(downs,'all').find(r=>r.window===5&&r.threshold===2).value-100/3)<1e-10);
const stale={...downs,downside_samples:{...downs.downside_samples,current_score:49}};
assert.equal(c.drawdownSample(stale,5,2,'all').n,0);
assert.equal(c.downsideCells(stale,'all')[0].value,null);
assert.equal(c.drawdownSample(d,5,2,'all').unavailable,20);
console.log('Unified downside checks passed: full default, non-breaches, reduced selection and stale-data rejection.');


const longDates=Array.from({length:400},(_,i)=>new Date(Date.UTC(2025,0,1+i)).toISOString().slice(0,10));
assert.deepEqual(c.episodeChartRange(longDates,longDates[200]),{start:137,end:263,range:[longDates[137],longDates[263]],matched:true});
assert.deepEqual(c.episodeChartRange(longDates,longDates[399]).range,[longDates[273],longDates[399]]);
assert.deepEqual(c.episodeChartRange(longDates,longDates[0]).range,[longDates[0],longDates[126]]);
assert.deepEqual(c.episodeChartRange(longDates,'absent'),{start:273,end:399,range:[longDates[273],longDates[399]],matched:false});
assert.deepEqual(c.episodeChartRange(dates,dates[15]).range,[dates[0],dates[29]]);
assert.equal(c.episodeChartRange([]),null);
const nonoverlap=[dates[0],dates[22]];
const extended={...d,return_samples:{...d.return_samples,all:{episode_dates:dates},nonoverlap_gap:21,nonoverlap:{episode_dates:nonoverlap}}};
assert.equal(c.validNonoverlapSample(extended),true);
assert.deepEqual(c.anchorsOf(extended,'nonoverlap'),nonoverlap);
assert.equal(c.resolveState(extended,new URLSearchParams('sample=nonoverlap&episode='+dates[22])).selected,dates[22]);
assert.equal(c.resolveState(extended,new URLSearchParams('sample=nonoverlap&episode='+dates[1])).selected,null);
assert.equal(c.resolveState(extended,new URLSearchParams('sample=reduced')).sample,'nonoverlap');
assert.equal(c.resolveState(d,new URLSearchParams('sample=nonoverlap')).sample,'all');
assert.equal(c.validNonoverlapSample({...extended,return_samples:{...extended.return_samples,nonoverlap:{episode_dates:reduced}}}),false);
assert.equal(c.validNonoverlapSample({...extended,return_samples:{...extended.return_samples,nonoverlap_gap:10}}),false);
const extendedDD={...extended,downside_samples:{...downs.downside_samples,nonoverlap:{episode_dates:nonoverlap,windows:{5:{outcomes:[row(nonoverlap[0],'complete',false),row(nonoverlap[1],'complete',true)]}}}}};
assert.equal(c.drawdownSample(extendedDD,5,2,'nonoverlap').rate,.5);
assert.equal(c.drawdownSample(extendedDD,5,2,'nonoverlap').selected,2);
console.log('Six-month chart bounds and genuine non-overlapping cohort checks passed.');
