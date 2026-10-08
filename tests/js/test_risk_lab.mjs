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
