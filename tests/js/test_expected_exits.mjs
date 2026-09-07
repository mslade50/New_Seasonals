import assert from "node:assert/strict";
import fs from "node:fs";
const source=fs.readFileSync(new URL('../../functions/_expected-exit-status.js',import.meta.url),'utf8');
const {projectExpectedExits}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
const now=Date.parse('2026-09-08T20:10:00Z');
const report={schema_version:1,account_key:'primary',generated_at:'2026-09-08T20:09:30Z',counts:{missed:999},
  private_account:'must not be exposed',notifications:[{message:'private data'}],
  obligations:[{id:'a',symbol:'AAA',strategy:'Synthetic Algo',status:'missed',remaining_tagged_qty:12,account:'private'}]};
const projected=projectExpectedExits(report,now);
assert.equal(projected.counts.missed,1);
assert.equal(projected.stale,false);
assert.ok(!JSON.stringify(projected).includes('private'));
assert.equal(projectExpectedExits({...report,generated_at:'2026-09-08T20:00:00Z'},now).stale,true);
assert.throws(()=>projectExpectedExits({...report,account_key:'pa'},now));
assert.throws(()=>projectExpectedExits({...report,generated_at:'2026-09-09T20:00:00Z'},now));
assert.throws(()=>projectExpectedExits({...report,obligations:[{status:'filled'}]},now));
const unknown=projectExpectedExits({...report,source_error:'Seed unavailable',obligations:[]},now);
assert.equal(unknown.counts.unable_to_verify,1);
assert.equal(unknown.obligations[0].remaining_tagged_qty,null);
console.log('Expected exits project exact Primary status and retain true source age');
