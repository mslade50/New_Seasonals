import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const source=fs.readFileSync(new URL('../../functions/exec-last-price.js',import.meta.url),'utf8')
  .replace(/^import .*;\r?$/gm,'').replace(/export async function/g,'async function')+'\nthis.post=onRequestPost;this.get=onRequestGet;';
const calls=[];
const ctx={URL,Response,JSON,requireAccess:async()=>null,
  fetch:async(url,opts)=>{calls.push({url,opts});return Response.json({ok:true,id:'id/1'});}};
vm.createContext(ctx);vm.runInContext(source,ctx);
const env={EXEC_BROKER_URL:'https://fixture.invalid',STATUS_TOKEN:'fixture-token'};
const body={symbol:'MNQ',sec_type:'FUT',currency:'USD',exchange:'CME',expiry:'202612',quantity:100,mode:'full'};
await ctx.post({request:{json:async()=>body},env});
assert.deepEqual(JSON.parse(calls[0].opts.body),{ticker:'MNQ',mode:'last_price',context:{sec_type:'FUT',currency:'USD',exchange:'CME',expiry:'202612'}});
assert.equal(calls[0].url,'https://fixture.invalid/workbench');
await ctx.get({request:{url:'https://site.invalid/exec-last-price?id=id%2F1'},env});
assert.equal(calls[1].url,'https://fixture.invalid/workbench?id=id%2F1');
assert.equal((await ctx.get({request:{url:'https://site.invalid/exec-last-price'},env})).status,400);
ctx.requireAccess=async()=>new Response('denied',{status:401});
assert.equal((await ctx.post({request:{json:async()=>body},env})).status,401);
assert.equal(calls.length,2,'denied or missing-id requests do not reach broker');
console.log('PASS protected last-price proxy uses read-only query mode and exact query id');
